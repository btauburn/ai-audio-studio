#!/usr/bin/env python3
"""Measure a real Qwen3-TTS runtime through the production audiobook path.

The harness is a *driver*, not a second renderer: every sample is rendered by
``scripts/make_audiobook.py render --renderer backend`` in a child process, and
every number it reports is either read back from the project state that command
wrote or sampled from ``nvidia-smi`` while it ran.  Nothing here re-implements
synthesis, so a report can never disagree with what the studio actually did.

Two metrics need a wall clock outside the production process:

* wall time -- the child process lifetime of one production render;
* peak VRAM -- a sampled maximum of ``nvidia-smi`` memory.used, which is only
  observable from another process.

Cold-start timing is optional and uses the service's own documented warm-up
endpoint (``POST /api/v1/tts/load-model``) rather than an invented one.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BACKEND = "qwen3tts_cv"
RUNTIME_ENVIRONMENT = "qwen3tts"
#: Interpreter that runs ``scripts/make_audiobook.py``.  The production path is
#: the project's own environment -- the TTS runtime below only serves audio over
#: HTTP and never imports the studio, so it is not the process that renders.
CLI_PYTHON = Path(".venv") / "Scripts" / "python.exe"
#: Interpreter that owns torch/CUDA and the Qwen3-TTS packages, used to report
#: the runtime identity of whatever is actually synthesizing.
RUNTIME_PYTHON = Path("environments") / RUNTIME_ENVIRONMENT / ".venv" / "Scripts" / "python.exe"

#: Consecutive probe failures tolerated before the sampler gives up; a monitor
#: that cannot read the GPU must not silently report a peak of zero.
MAX_SAMPLE_ERRORS = 5


class BenchmarkError(RuntimeError):
    """Raised when the harness cannot produce a trustworthy measurement."""


# --------------------------------------------------------------------------- #
# GPU sampling
# --------------------------------------------------------------------------- #


@dataclass
class GpuSample:
    memory_used_mib: int
    utilization_percent: Optional[int]


def query_gpu() -> GpuSample:
    """Read one instantaneous GPU sample, or fail loudly if it is unreadable."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise BenchmarkError(f"nvidia-smi failed: {completed.stderr.strip()}")
    row = next(csv.reader([completed.stdout.strip().splitlines()[0]]))
    used = int(row[0].strip())
    util_text = row[1].strip()
    return GpuSample(
        memory_used_mib=used,
        utilization_percent=int(util_text) if util_text.isdigit() else None,
    )


def query_gpu_identity() -> Dict[str, str]:
    """Read the stable GPU identity fields for the report header."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise BenchmarkError(f"nvidia-smi failed: {completed.stderr.strip()}")
    name, driver, total = next(csv.reader([completed.stdout.strip().splitlines()[0]]))
    return {
        "name": name.strip(),
        "driver_version": driver.strip(),
        "memory_total_mib": total.strip().replace(" MiB", "").strip(),
    }


class VramMonitor:
    """Sample ``memory.used`` in a background thread and keep the maximum."""

    def __init__(
        self,
        interval_seconds: float = 0.25,
        probe: Callable[[], GpuSample] = query_gpu,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.probe = probe
        self.baseline: Optional[GpuSample] = None
        self.peak: Optional[GpuSample] = None
        self.last: Optional[GpuSample] = None
        self.samples: int = 0
        self.history: List[GpuSample] = []
        self.errors: List[str] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _record(self, sample: GpuSample) -> None:
        self.last = sample
        self.samples += 1
        self.history.append(sample)
        if self.peak is None or sample.memory_used_mib > self.peak.memory_used_mib:
            self.peak = sample

    def mark(self) -> int:
        """Return a cursor so a window of samples can be attributed to one render."""

        return len(self.history)

    def peak_since(self, mark: int) -> Optional[int]:
        """Highest ``memory.used`` recorded at or after ``mark``."""

        window = self.history[mark:]
        if not window:
            return self.last.memory_used_mib if self.last else None
        return max(sample.memory_used_mib for sample in window)

    def _sample_once(self) -> bool:
        """Record one probe result; ``False`` means sampling should stop."""

        try:
            self._record(self.probe())
            return True
        except Exception as exc:  # noqa: BLE001 - sampling must never kill a run
            self.errors.append(str(exc))
            return len(self.errors) <= MAX_SAMPLE_ERRORS

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            if not self._sample_once():
                return

    def start(self, *, background: bool = True) -> "VramMonitor":
        self.baseline = self.probe()
        self._record(self.baseline)
        if background:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> Dict[str, Any]:
        baseline = self.baseline.memory_used_mib if self.baseline else None
        peak = self.peak.memory_used_mib if self.peak else None
        return {
            "baseline_mib": baseline,
            "peak_mib": peak,
            "delta_mib": (peak - baseline) if (peak is not None and baseline is not None) else None,
            "samples": self.samples,
            "history_samples": len(self.history),
            "sample_errors": list(self.errors),
        }


# --------------------------------------------------------------------------- #
# Production-path invocation
# --------------------------------------------------------------------------- #


@dataclass
class RenderSample:
    sample_id: str
    segment_id: str
    slot: str
    chars: int
    wall_seconds: float
    exit_code: int
    audio_seconds: Optional[float]
    chars_per_second: Optional[float]
    rtf: Optional[float]
    audio_sha256: Optional[str]
    take_id: Optional[str]
    renderer_id: Optional[str]
    engine_id: Optional[str]
    model_id: Optional[str]
    qa_passed: Optional[bool]
    chunked: Optional[bool]
    peak_vram_mib: Optional[int]
    baseline_vram_mib: Optional[int]
    run_peak_vram_mib: Optional[int] = None
    accepted: bool = False
    selected: Optional[bool] = None
    synthesis_seconds: Optional[float] = None
    stage_timings: Dict[str, Any] = field(default_factory=dict)
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    stdout_tail: str = ""
    stderr_tail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        row = dict(self.__dict__)
        row["stage_timings"] = dict(self.stage_timings or {})
        row["generation_metadata"] = dict(self.generation_metadata or {})
        synthesis_rtf = (
            self.synthesis_seconds / self.audio_seconds
            if self.synthesis_seconds and self.audio_seconds
            else None
        )
        row["rtf_wall_over_audio"] = (
            round(synthesis_rtf, 6) if synthesis_rtf is not None else None
        )
        row["end_to_end_rtf_wall_over_audio"] = self.rtf
        row["audio_seconds_per_wall_second"] = (
            round(1.0 / synthesis_rtf, 6)
            if synthesis_rtf and synthesis_rtf > 0
            else None
        )
        return row


def _tail(text: str, limit: int = 4000) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[-limit:]


def _parse_stdout_json(stdout: str) -> Optional[Dict[str, Any]]:
    """Return the CLI's JSON payload, ignoring any incidental leading lines."""

    stripped = (stdout or "").strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for start in range(len(stripped)):
        if stripped[start] != "{":
            continue
        try:
            return json.loads(stripped[start:])
        except json.JSONDecodeError:
            continue
    return None


def _production_command(
    python: Path,
    *,
    project_id: str,
    segment_id: str,
    backend: str,
    slot: str,
    run_qa: bool,
) -> List[str]:
    command = [
        str(python),
        str(PROJECT_ROOT / "scripts" / "make_audiobook.py"),
        "render",
        "--project-id",
        project_id,
        "--utterance",
        segment_id,
        "--renderer",
        "backend",
        "--backend",
        backend,
        "--slot",
        slot,
    ]
    if not run_qa:
        command.append("--no-qa")
    return command


def _take_metadata(project_id: str, segment_id: str, take_id: str) -> Dict[str, Any]:
    """Read back the persisted Take so reported metadata is production state.

    The Take is looked up by id across every slot, so the harness cannot report
    a slot's current pointer in place of the Take the render actually created.
    """

    from dubbing_service.app.storage.segments import load_segments, project_segments_lock

    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        for segment in segments:
            if segment.id != segment_id:
                continue
            for take in getattr(segment, "takes", []) or []:
                if take.id == take_id:
                    return {
                        "audio_sha256": take.audio_sha256,
                        "renderer_id": take.renderer_id,
                        "engine_id": take.engine_id,
                        "model_id": take.model_id,
                        "audio_file": take.audio_file,
                        "input_fingerprint": take.input_fingerprint,
                        "generation_metadata": dict(take.generation_metadata or {}),
                    }
    raise BenchmarkError(f"take {take_id} for utterance {segment_id} was not persisted")


def _segment_text(project_id: str, segment_id: str) -> str:
    from dubbing_service.app.storage.segments import load_segments, project_segments_lock

    with project_segments_lock(project_id):
        for segment in load_segments(project_id):
            if segment.id == segment_id:
                return segment.text or ""
    raise BenchmarkError(f"utterance {segment_id} is not in project {project_id}")


def _service_synthesis_seconds(metadata: Dict[str, Any]) -> Optional[float]:
    generation = metadata.get("generation_metadata")
    if not isinstance(generation, dict):
        return None
    calls = generation.get("provider_calls")
    if not isinstance(calls, list):
        return None
    total = 0.0
    found = False
    for call in calls:
        if not isinstance(call, dict):
            continue
        timings = call.get("stage_timings")
        inclusive = timings.get("inclusive") if isinstance(timings, dict) else None
        value = inclusive.get("model_generate_seconds") if isinstance(inclusive, dict) else None
        if isinstance(value, (int, float)):
            total += float(value)
            found = True
    return round(total, 6) if found else None


def render_one(
    python: Path,
    *,
    project_id: str,
    segment_id: str,
    backend: str,
    slot: str,
    run_qa: bool,
    timeout_seconds: float,
    monitor: Optional[VramMonitor] = None,
    env: Optional[Dict[str, str]] = None,
) -> RenderSample:
    """Run one production render and assemble its measured sample."""

    command = _production_command(
        python,
        project_id=project_id,
        segment_id=segment_id,
        backend=backend,
        slot=slot,
        run_qa=run_qa,
    )
    before = query_gpu() if monitor is None else None
    mark = monitor.mark() if monitor is not None else None
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        stdout, stderr, code = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        wall = time.perf_counter() - started
        return RenderSample(
            sample_id=segment_id,
            segment_id=segment_id,
            slot=slot,
            chars=0,
            wall_seconds=wall,
            exit_code=124,
            audio_seconds=None,
            chars_per_second=None,
            rtf=None,
            audio_sha256=None,
            take_id=None,
            renderer_id=None,
            engine_id=None,
            model_id=None,
            qa_passed=None,
            chunked=None,
            peak_vram_mib=None,
            baseline_vram_mib=None,
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
        )
    wall = time.perf_counter() - started

    payload = _parse_stdout_json(stdout) or {}
    takes = payload.get("takes") or []
    take = takes[0] if takes else None
    text = _segment_text(project_id, segment_id)
    chars = len(text)

    metadata: Dict[str, Any] = {}
    take_id = take.get("take_id") if isinstance(take, dict) else None
    if take_id:
        try:
            metadata = _take_metadata(project_id, segment_id, take_id)
        except BenchmarkError:
            metadata = {}

    duration_ms = take.get("duration_ms") if isinstance(take, dict) else None
    audio_seconds = (duration_ms / 1000.0) if isinstance(duration_ms, (int, float)) else None
    qa_passed = take.get("qa_passed") if isinstance(take, dict) else None
    selected = take.get("selected") if isinstance(take, dict) else None
    stage_timings = take.get("stage_timings") if isinstance(take, dict) else {}
    if not isinstance(stage_timings, dict):
        stage_timings = {}
    generation_metadata = metadata.get("generation_metadata")
    if not isinstance(generation_metadata, dict):
        generation_metadata = {}
    execution_profile = None
    calls = generation_metadata.get("provider_calls")
    if isinstance(calls, list) and calls and isinstance(calls[0], dict):
        execution_profile = calls[0].get("execution_profile")
    actual_model_id = metadata.get("model_id")
    if not actual_model_id and isinstance(execution_profile, dict):
        actual_model_id = execution_profile.get("model_id")
    peak_mib = (
        monitor.peak_since(mark)
        if (monitor is not None and mark is not None)
        else (before.memory_used_mib if before else None)
    )

    return RenderSample(
        sample_id=segment_id,
        segment_id=segment_id,
        slot=slot,
        chars=chars,
        wall_seconds=round(wall, 4),
        exit_code=code,
        audio_seconds=round(audio_seconds, 4) if audio_seconds else None,
        chars_per_second=round(chars / wall, 3) if (chars and wall > 0) else None,
        rtf=round(wall / audio_seconds, 4) if audio_seconds else None,
        audio_sha256=metadata.get("audio_sha256"),
        take_id=take_id,
        renderer_id=metadata.get("renderer_id"),
        engine_id=metadata.get("engine_id"),
        model_id=actual_model_id,
        qa_passed=qa_passed,
        chunked=take.get("chunked") if isinstance(take, dict) else None,
        peak_vram_mib=peak_mib,
        baseline_vram_mib=(monitor.baseline.memory_used_mib if monitor and monitor.baseline else (before.memory_used_mib if before else None)),
        run_peak_vram_mib=(monitor.peak.memory_used_mib if (monitor and monitor.peak) else None),
        accepted=bool(code == 0 and qa_passed is True and selected is True),
        selected=selected if isinstance(selected, bool) else None,
        synthesis_seconds=_service_synthesis_seconds(metadata),
        stage_timings=stage_timings,
        generation_metadata=generation_metadata,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
    )


# --------------------------------------------------------------------------- #
# Service probes
# --------------------------------------------------------------------------- #


def http_json(url: str, *, method: str = "GET", timeout: float = 30.0) -> Dict[str, Any]:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", "replace")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{url} did not return JSON: {body[:200]}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkError(f"{url} returned {type(payload).__name__}, expected an object")
    return payload


def measure_cold_load(url: str, timeout: float = 600.0) -> Dict[str, Any]:
    """Time the service's own model warm-up endpoint.

    Every failure mode is surfaced as ``ok: False`` with the reason, so a report
    can never present an unmeasured cold start as a fast one.
    """

    started = time.perf_counter()
    try:
        payload = http_json(url, method="POST", timeout=timeout)
        elapsed = time.perf_counter() - started
        return {
            "ok": True,
            "url": url,
            "seconds": round(elapsed, 4),
            "response": payload,
            "gpu_after": query_gpu().memory_used_mib,
        }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "url": url, "error": f"HTTP {exc.code}: {exc.read()[:400]!r}"}
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        return {"ok": False, "url": url, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Runtime identity
# --------------------------------------------------------------------------- #


def runtime_identity(python: Path) -> Dict[str, Any]:
    """Report the interpreter/GPU facts, or the reason they are unavailable."""

    script = (
        "import json, platform, torch\n"
        "info = {\n"
        "    'python': platform.python_version(),\n"
        "    'torch': torch.__version__,\n"
        "    'cuda_runtime': torch.version.cuda,\n"
        "    'cuda_available': torch.cuda.is_available(),\n"
        "}\n"
        "if info['cuda_available']:\n"
        "    props = torch.cuda.get_device_properties(0)\n"
        "    info['device'] = torch.cuda.get_device_name(0)\n"
        "    info['capability'] = f'{props.major}.{props.minor}'\n"
        "    info['total_vram_mib'] = round(props.total_memory / 1024**2, 1)\n"
        "    info['bf16_supported'] = torch.cuda.is_bf16_supported()\n"
        "print(json.dumps(info))\n"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    if completed.returncode != 0:
        return {"ok": False, "error": completed.stderr.strip()[-2000:]}
    payload = _parse_stdout_json(completed.stdout)
    if not payload:
        return {"ok": False, "error": f"unparseable runtime identity: {completed.stdout[:400]}"}
    payload["ok"] = True
    return payload


def library_identity(python: Path) -> Dict[str, Any]:
    """Report the pinned generation library versions actually installed."""

    script = (
        "import json\n"
        "from importlib import metadata\n"
        "names = ['qwen-tts', 'qwen-tts-hf', 'faster-qwen3-tts', 'modelscope', 'transformers', 'safetensors', 'fastapi', 'uvicorn', 'soundfile']\n"
        "out = {}\n"
        "for name in names:\n"
        "    try:\n"
        "        out[name] = metadata.version(name)\n"
        "    except Exception:\n"
        "        out[name] = None\n"
        "print(json.dumps(out))\n"
    )
    completed = subprocess.run(
        [str(python), "-c", script], capture_output=True, text=True, timeout=300
    )
    if completed.returncode != 0:
        return {"ok": False, "error": completed.stderr.strip()[-2000:]}
    payload = _parse_stdout_json(completed.stdout) or {}
    payload["ok"] = True
    return payload


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def summarize(samples: Sequence[RenderSample]) -> Dict[str, Any]:
    """Aggregate with totals; per-segment medians remain diagnostics only."""

    if not samples:
        raise BenchmarkError("no samples: a summary over an empty population is not a result")

    succeeded = [s for s in samples if s.exit_code == 0]
    completed_audio = [s for s in samples if s.audio_seconds is not None]
    accepted = [
        s
        for s in samples
        if s.accepted
        or (
            s.exit_code == 0
            and s.qa_passed is True
            and s.selected is not False
        )
    ]
    rtfs = [s.rtf for s in succeeded if s.rtf is not None]
    throughput = [s.chars_per_second for s in succeeded if s.chars_per_second is not None]
    wall = [s.wall_seconds for s in succeeded]
    total_wall = sum(s.wall_seconds for s in samples)
    accepted_wall = sum(s.wall_seconds for s in accepted)
    total_completed_audio = sum(s.audio_seconds or 0.0 for s in completed_audio)
    total_accepted_audio = sum(s.audio_seconds or 0.0 for s in accepted)
    accepted_chars = sum(s.chars for s in accepted)
    synthesis_seconds = sum(s.synthesis_seconds or 0.0 for s in completed_audio)
    accepted_synthesis_seconds = sum(s.synthesis_seconds or 0.0 for s in accepted)
    end_to_end_rtf = (
        total_wall / total_accepted_audio if total_accepted_audio > 0 else None
    )
    synthesis_rtf = (
        accepted_synthesis_seconds / total_accepted_audio
        if accepted_synthesis_seconds > 0 and total_accepted_audio > 0
        else None
    )
    accepted_audio_rate = (
        total_accepted_audio / total_wall if total_wall > 0 else None
    )
    return {
        "rendered": len(samples),
        "succeeded": len(succeeded),
        "failed": len(samples) - len(succeeded),
        "completed_audio": len(completed_audio),
        "accepted": len(accepted),
        "first_pass_acceptance": round(len(accepted) / len(samples), 6),
        "total_end_to_end_wall_seconds": round(total_wall, 4),
        "accepted_end_to_end_wall_seconds": round(accepted_wall, 4),
        "total_completed_audio_seconds": round(total_completed_audio, 3),
        "total_accepted_audio_seconds": round(total_accepted_audio, 3),
        "rtf_wall_over_audio": round(synthesis_rtf, 6) if synthesis_rtf else None,
        "audio_seconds_per_wall_second": (
            round(1.0 / synthesis_rtf, 6) if synthesis_rtf else None
        ),
        "end_to_end_rtf_wall_over_accepted_audio": (
            round(end_to_end_rtf, 6) if end_to_end_rtf else None
        ),
        "end_to_end_accepted_audio_seconds_per_wall_second": (
            round(accepted_audio_rate, 6) if accepted_audio_rate else None
        ),
        "accepted_source_characters_per_wall_second": (
            round(accepted_chars / total_wall, 6) if total_wall > 0 else None
        ),
        "accepted_audio_minutes_per_wall_hour": (
            round(accepted_audio_rate * 60.0, 6) if accepted_audio_rate else None
        ),
        "synthesis_wall_seconds": round(synthesis_seconds, 6),
        "completed_audio_synthesis_rtf_wall_over_audio": (
            round(synthesis_seconds / total_completed_audio, 6)
            if synthesis_seconds and total_completed_audio else None
        ),
        # Diagnostics only: decisions use the aggregate totals above.
        "median_rtf": round(statistics.median(rtfs), 4) if rtfs else None,
        "median_chars_per_second": round(statistics.median(throughput), 3) if throughput else None,
        "median_wall_seconds": round(statistics.median(wall), 4) if wall else None,
        "total_audio_seconds": round(
            sum(s.audio_seconds or 0.0 for s in succeeded), 3
        ),
    }


def write_report(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-id", required=True, help="existing project to render into")
    parser.add_argument("--segment-id", action="append", default=[], help="utterance to render; repeatable")
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--slot", default="a")
    parser.add_argument(
        "--python",
        default=str(PROJECT_ROOT / CLI_PYTHON),
        help="interpreter that runs the production render CLI",
    )
    parser.add_argument(
        "--runtime-python",
        default=str(PROJECT_ROOT / RUNTIME_PYTHON),
        help="interpreter whose torch/CUDA/Qwen packages are reported as the runtime identity",
    )
    parser.add_argument("--label", default="", help="free-form label recorded in the report")
    parser.add_argument("--output", default=None, help="write the JSON report here")
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=900.0, help="per-render timeout in seconds")
    parser.add_argument("--no-qa", action="store_true", help="skip QA gate for this measurement")
    parser.add_argument(
        "--health-url",
        default=None,
        help="service /health URL recorded in the report",
    )
    parser.add_argument(
        "--cold-load-url",
        default=None,
        help="service POST endpoint that loads the model, timed as cold start",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cli_python = Path(args.python)
    runtime_python = Path(args.runtime_python)

    if not args.segment_id:
        raise BenchmarkError("at least one --segment-id is required")

    report: Dict[str, Any] = {
        "label": args.label,
        "backend": args.backend,
        "slot": args.slot,
        "project_id": args.project_id,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": query_gpu_identity(),
        "cli_python": str(cli_python),
        "runtime_python": str(runtime_python),
        "runtime": runtime_identity(runtime_python) if runtime_python.exists() else {"ok": False, "error": "runtime python missing"},
        "libraries": library_identity(runtime_python) if runtime_python.exists() else {"ok": False},
        "service_before": {},
        "service_after": {},
        "cold_load": None,
        "samples": [],
    }

    if args.health_url:
        try:
            report["service_before"] = http_json(args.health_url, timeout=60.0)
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
            report["service_before"] = {"ok": False, "error": str(exc)}

    if args.cold_load_url:
        report["cold_load"] = measure_cold_load(args.cold_load_url)

    monitor = VramMonitor(interval_seconds=args.poll_interval).start()
    sample_objects: List[RenderSample] = []
    try:
        for segment_id in args.segment_id:
            sample = render_one(
                cli_python,
                project_id=args.project_id,
                segment_id=segment_id,
                backend=args.backend,
                slot=args.slot,
                run_qa=not args.no_qa,
                timeout_seconds=args.timeout,
                monitor=monitor,
            )
            sample_objects.append(sample)
            report["samples"].append(sample.as_dict())
    finally:
        monitor.stop()

    report["vram"] = monitor.summary()
    if args.health_url:
        try:
            report["service_after"] = http_json(args.health_url, timeout=60.0)
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
            report["service_after"] = {"ok": False, "error": str(exc)}
    report["summary"] = summarize(sample_objects)
    if args.output:
        report["report_path"] = str(write_report(Path(args.output), report))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
