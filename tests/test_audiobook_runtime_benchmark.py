"""Tests for the RTX runtime benchmark harness.

The harness exists to produce *measured* claims about a real GPU run, so the
tests here cover the parts that decide what a report is allowed to say: the
production-path command shape, the metric arithmetic, and the refusal to
summarise an empty population.  Nothing here needs a GPU, a model, or a service.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import scripts.benchmark_qwen_runtime as harness


def test_production_command_always_uses_the_backend_renderer(tmp_path: Path) -> None:
    command = harness._production_command(
        Path("python.exe"),
        project_id="proj-1",
        segment_id="seg-1",
        backend="qwen3tts_cv",
        slot="a",
        run_qa=True,
    )

    assert command[1].endswith("make_audiobook.py")
    assert command[2] == "render"
    assert command[command.index("--renderer") + 1] == "backend"
    assert command[command.index("--backend") + 1] == "qwen3tts_cv"
    assert command[command.index("--utterance") + 1] == "seg-1"
    assert "--no-qa" not in command


def test_production_command_skips_qa_only_when_asked(tmp_path: Path) -> None:
    command = harness._production_command(
        Path("python.exe"),
        project_id="proj-1",
        segment_id="seg-1",
        backend="qwen3tts_cv",
        slot="b",
        run_qa=False,
    )

    assert command[command.index("--slot") + 1] == "b"
    assert command[-1] == "--no-qa"


def test_parse_stdout_json_ignores_incidental_leading_lines() -> None:
    payload = harness._parse_stdout_json('warning: deprecated\n{"ok": true, "takes": []}\n')

    assert payload == {"ok": True, "takes": []}


def test_parse_stdout_json_returns_none_for_non_json() -> None:
    assert harness._parse_stdout_json("not json at all") is None
    assert harness._parse_stdout_json("") is None


def test_summarize_refuses_an_empty_population() -> None:
    with pytest.raises(harness.BenchmarkError):
        harness.summarize([])


def test_summarize_reports_medians_and_failure_counts() -> None:
    samples = [
        harness.RenderSample(
            sample_id="a",
            segment_id="a",
            slot="a",
            chars=100,
            wall_seconds=2.0,
            exit_code=0,
            audio_seconds=4.0,
            chars_per_second=50.0,
            rtf=0.5,
            audio_sha256="x",
            take_id="t1",
            renderer_id="backend-renderer",
            engine_id="qwen3tts_customvoice",
            model_id="qwen3_tts_customvoice",
            qa_passed=True,
            chunked=False,
            peak_vram_mib=6000,
            baseline_vram_mib=2500,
        ),
        harness.RenderSample(
            sample_id="b",
            segment_id="b",
            slot="a",
            chars=300,
            wall_seconds=6.0,
            exit_code=0,
            audio_seconds=4.0,
            chars_per_second=50.0,
            rtf=1.5,
            audio_sha256="y",
            take_id="t2",
            renderer_id="backend-renderer",
            engine_id="qwen3tts_customvoice",
            model_id="qwen3_tts_customvoice",
            qa_passed=True,
            chunked=True,
            peak_vram_mib=6000,
            baseline_vram_mib=2500,
        ),
        harness.RenderSample(
            sample_id="c",
            segment_id="c",
            slot="a",
            chars=0,
            wall_seconds=0.4,
            exit_code=1,
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
        ),
    ]

    summary = harness.summarize(samples)

    assert summary["rendered"] == 3
    assert summary["succeeded"] == 2
    assert summary["failed"] == 1
    assert summary["median_rtf"] == 1.0
    assert summary["median_chars_per_second"] == 50.0
    assert summary["median_wall_seconds"] == 4.0
    assert summary["total_audio_seconds"] == 8.0
    assert summary["first_pass_acceptance"] == pytest.approx(2 / 3, abs=1e-6)
    assert summary["end_to_end_rtf_wall_over_accepted_audio"] == 1.05
    assert summary["accepted_source_characters_per_wall_second"] == pytest.approx(
        400 / 8.4, abs=1e-6
    )


def test_summarize_uses_total_synthesis_wall_over_total_audio() -> None:
    samples = [
        harness.RenderSample(
            sample_id="short",
            segment_id="short",
            slot="a",
            chars=10,
            wall_seconds=10.0,
            exit_code=0,
            audio_seconds=2.0,
            chars_per_second=1.0,
            rtf=5.0,
            audio_sha256="a",
            take_id="t1",
            renderer_id="backend-renderer",
            engine_id="qwen3tts_customvoice",
            model_id="qwen3_tts_customvoice",
            qa_passed=True,
            chunked=False,
            peak_vram_mib=6000,
            baseline_vram_mib=2500,
            selected=True,
            accepted=True,
            synthesis_seconds=1.0,
        ),
        harness.RenderSample(
            sample_id="long",
            segment_id="long",
            slot="a",
            chars=100,
            wall_seconds=20.0,
            exit_code=0,
            audio_seconds=10.0,
            chars_per_second=5.0,
            rtf=2.0,
            audio_sha256="b",
            take_id="t2",
            renderer_id="backend-renderer",
            engine_id="qwen3tts_customvoice",
            model_id="qwen3_tts_customvoice",
            qa_passed=True,
            chunked=False,
            peak_vram_mib=6000,
            baseline_vram_mib=2500,
            selected=True,
            accepted=True,
            synthesis_seconds=4.0,
        ),
    ]

    summary = harness.summarize(samples)

    assert summary["rtf_wall_over_audio"] == pytest.approx(5 / 12, abs=1e-6)
    assert summary["audio_seconds_per_wall_second"] == pytest.approx(12 / 5, abs=1e-6)
    assert summary["end_to_end_rtf_wall_over_accepted_audio"] == 2.5


def test_query_gpu_parses_a_single_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(command, 0, stdout="2511, 16\n", stderr="")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    sample = harness.query_gpu()

    assert sample.memory_used_mib == 2511
    assert sample.utilization_percent == 16


def test_query_gpu_rejects_a_failed_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="no devices")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    with pytest.raises(harness.BenchmarkError):
        harness.query_gpu()


def test_query_gpu_identity_reads_name_and_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003
        return subprocess.CompletedProcess(
            command, 0, stdout="NVIDIA GeForce RTX 4070, 616.56, 12282 MiB\n", stderr=""
        )

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    identity = harness.query_gpu_identity()

    assert identity["name"] == "NVIDIA GeForce RTX 4070"
    assert identity["driver_version"] == "616.56"
    assert identity["memory_total_mib"] == "12282"


def test_vram_monitor_keeps_the_peak_and_the_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    readings = iter(
        [
            harness.GpuSample(memory_used_mib=2500, utilization_percent=1),
            harness.GpuSample(memory_used_mib=6200, utilization_percent=80),
            harness.GpuSample(memory_used_mib=4100, utilization_percent=10),
        ]
    )

    monitor = harness.VramMonitor(interval_seconds=0.01, probe=lambda: next(readings))
    monitor.start(background=False)

    assert monitor.baseline.memory_used_mib == 2500
    assert monitor.summary()["peak_mib"] == 2500

    assert monitor._sample_once() is True  # noqa: SLF001 - sampler is under test
    assert monitor.summary()["peak_mib"] == 6200
    assert monitor.summary()["delta_mib"] == 3700

    assert monitor._sample_once() is True  # noqa: SLF001
    assert monitor.summary()["peak_mib"] == 6200


def test_vram_monitor_attributes_a_peak_to_the_window_since_a_mark() -> None:
    readings = iter(
        [
            harness.GpuSample(memory_used_mib=2500, utilization_percent=1),
            harness.GpuSample(memory_used_mib=5100, utilization_percent=70),
            harness.GpuSample(memory_used_mib=4300, utilization_percent=40),
            harness.GpuSample(memory_used_mib=7000, utilization_percent=95),
        ]
    )
    monitor = harness.VramMonitor(interval_seconds=0.01, probe=lambda: next(readings))
    monitor.start(background=False)

    first = monitor.mark()
    assert monitor._sample_once() is True  # noqa: SLF001 - sampler is under test
    assert monitor._sample_once() is True  # noqa: SLF001
    window_one = monitor.peak_since(first)

    second = monitor.mark()
    assert monitor._sample_once() is True  # noqa: SLF001
    window_two = monitor.peak_since(second)

    assert window_one == 5100
    assert window_two == 7000
    assert monitor.summary()["peak_mib"] == 7000
    assert monitor.summary()["history_samples"] == 4


def test_vram_monitor_falls_back_to_the_last_reading_for_an_empty_window() -> None:
    readings = iter(
        [
            harness.GpuSample(memory_used_mib=2500, utilization_percent=1),
            harness.GpuSample(memory_used_mib=3800, utilization_percent=55),
        ]
    )
    monitor = harness.VramMonitor(interval_seconds=0.01, probe=lambda: next(readings))
    monitor.start(background=False)
    assert monitor._sample_once() is True  # noqa: SLF001

    assert monitor.peak_since(monitor.mark()) == 3800
    assert monitor.peak_since(monitor.mark()) == 3800


def test_vram_monitor_stops_after_repeated_probe_failures() -> None:
    def boom() -> harness.GpuSample:
        raise harness.BenchmarkError("nvidia-smi unavailable")

    monitor = harness.VramMonitor(interval_seconds=0.01, probe=boom)

    outcomes = [monitor._sample_once() for _ in range(harness.MAX_SAMPLE_ERRORS + 1)]  # noqa: SLF001

    assert outcomes[: harness.MAX_SAMPLE_ERRORS] == [True] * harness.MAX_SAMPLE_ERRORS
    assert outcomes[-1] is False
    assert len(monitor.errors) == harness.MAX_SAMPLE_ERRORS + 1
    assert monitor.samples == 0


def test_vram_monitor_reports_a_baseline_but_no_peak_when_it_never_sampled() -> None:
    monitor = harness.VramMonitor(interval_seconds=0.01, probe=lambda: harness.GpuSample(0, 0))

    assert monitor.summary()["peak_mib"] is None
    assert monitor.summary()["samples"] == 0


def test_write_report_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "report.json"

    written = harness.write_report(target, {"ok": True})

    assert written == target
    assert target.read_text(encoding="utf-8").strip() == '{\n  "ok": true\n}'


def test_runtime_python_default_is_the_isolated_environment() -> None:
    default = harness.build_parser().parse_args(["--project-id", "p"]).runtime_python

    assert "environments" in default
    assert "qwen3tts" in default
    assert default.endswith("python.exe") or default.endswith("python")


def test_cli_python_default_is_the_project_environment_not_the_tts_runtime() -> None:
    """The render CLI runs in the studio's own environment.

    The TTS runtime is a separate process reached over HTTP; it has no
    ``faster_whisper``, so the production QA gate cannot run there.
    """

    parser = harness.build_parser()
    cli_default = parser.parse_args(["--project-id", "p"]).python

    assert ".venv" in cli_default
    assert "environments" not in cli_default
    assert cli_default != parser.parse_args(["--project-id", "p"]).runtime_python


def test_harness_module_imports_the_repo_root(tmp_path: Path) -> None:
    assert str(harness.PROJECT_ROOT) in sys.path
    assert (harness.PROJECT_ROOT / "scripts" / "make_audiobook.py").is_file()
