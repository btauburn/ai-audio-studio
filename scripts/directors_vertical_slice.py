"""Isolated real-audio demo. Copies approved source state; never edits that state.

Prepare: python scripts/directors_vertical_slice.py --source-root ROOT --source-project ID --output-dir OUT
Generate: same command with --generate. One GPU worker at a time; no downloads.
The four-arm benchmark is review-only, and its extra audition role has no human
approval. Its line text consists of exact existing audition sentence slices.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode()


def publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing immutable demo artifact conflicts: {path}")


def sentences(text):
    # Exact character slices: punctuation and words are never rewritten.
    return [(m.group().strip(), m.start() + len(m.group()) - len(m.group().lstrip()))
            for m in re.finditer(r"[^.!?]+[.!?]+", text)]


def prepare(args):
    from dubbing_service.app.storage.projects import create_project, load_project, save_project
    from dubbing_service.app.storage.segments import load_segments, save_segments
    from dubbing_service.app.models.project import Chapter, Role, Segment
    from dubbing_service.app.audiobook import casting_director as director
    from dubbing_service.app.core.config import clear_config_caches
    destination = args.output_dir / "projects" / args.source_project
    source = args.source_root / args.source_project
    if not destination.exists():
        shutil.copytree(source, destination)
    os.environ["DUBBING_PROJECTS_ROOT"] = str(args.output_dir / "projects")
    clear_config_caches()
    state_path = args.output_dir / "demo-state.json"
    if state_path.exists():
        state = json.loads(state_path.read_bytes())
        if state["source_project_sha256"] != sha((source / "project.json").read_bytes()):
            raise RuntimeError("Source changed since preparation; preserve this run and choose a new output directory")
        return state
    p = load_project(args.source_project)
    rows = load_segments(p.id)
    names = ["narrator", "harry", "ron", "dumbledore"]
    by_role = {rid: next(s for s in rows if s.role_id == rid and s.text.strip()) for rid in names}
    picked = {rid: sentences(by_role[rid].text)[1:4] for rid in names}
    if any(len(v) < 3 for v in picked.values()):
        raise RuntimeError("This source does not supply three exact audition sentences per required role")
    # Narration plus a dialogue exchange. Reordering audition excerpts is explicit
    # evaluation-set construction, never an edit/export of the book itself.
    order = [("narrator", 0), ("harry", 0), ("ron", 0), ("harry", 1),
             ("narrator", 1), ("ron", 1), ("harry", 2), ("dumbledore", 0),
             ("ron", 2), ("narrator", 2), ("dumbledore", 1), ("dumbledore", 2)]
    utterances = []
    for i, (rid, line) in enumerate(order):
        text, start = picked[rid][line]
        utterances.append({"utterance_id": f"review-{i+1:02d}",
                           "role_id": "audition-guest" if rid == "dumbledore" else rid,
                           "role_name": "Unassigned review role" if rid == "dumbledore" else rid.title(),
                           "role_type": "narration" if rid == "narrator" else "dialogue",
                           "text": text, "source_role_id": rid,
                           "source_segment_id": by_role[rid].id, "start_offset": start,
                           "end_offset": start + len(text)})
    for u in utterances:
        original = next(s for s in rows if s.id == u["source_segment_id"]).text
        if original[u["start_offset"]:u["end_offset"]] != u["text"]:
            raise RuntimeError("Source sentence slice mismatch")
    audition = create_project("Casting Director — unassigned review role")
    audition.voice_lab.scope = "isolated_listening_test"
    audition.chapters = [Chapter(id="review", title="Original audition excerpts", index=0)]
    audition.roles = [Role(id="audition-guest", name="Unassigned review role", tts_backend="qwen3tts")]
    audition.metadata["requested_runtime"] = "faster-base-032"
    audition.metadata["director_demo"] = {"review_only": True, "source_project": p.id,
                                           "original_cast_preserved": True,
                                           "source_kind": "existing_original_audition_excerpts"}
    save_project(audition)
    save_segments(audition.id, [Segment(id=u["utterance_id"], index=i, chapter_id="review",
                                       role_id="audition-guest", text=u["text"])
                               for i, u in enumerate(utterances) if u["role_id"] == "audition-guest"])
    report = director.analyze(audition.id)
    receipt = director.prepare_auditions(audition.id, request_id="directors-demo-round-v1",
                                         role_id="audition-guest", report_sha256=report["report_sha256"])
    state = {"source_project": p.id, "source_project_sha256": sha((source / "project.json").read_bytes()),
             "source_sha256": sha(encoded(utterances)), "audition_project": audition.id,
             "audition_receipt": receipt, "utterances": utterances,
             "source_kind": "existing_original_audition_excerpts", "fixture": False,
             "human_review": "pending", "production_ready": False}
    publish(state_path, encoded(state))
    publish(args.output_dir / "analysis.json", encoded(report))
    return state


class Worker:
    def __init__(self, output, directed=False):
        self.output, self.directed, self.process = output, directed, None

    def __enter__(self):
        import httpx
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", 8143)) == 0:
                raise RuntimeError("Demo port 8143 already has an owner")
        memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip()
        if int(memory) < 8500:
            raise RuntimeError("GPU is busy; demo refuses concurrent inference")
        script = "faster_qwen3tts_base_direction_server.py" if self.directed else "faster_qwen3tts_base_server.py"
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        args = [str(REPO / "environments/faster_qwen_base032/.venv/Scripts/python.exe"),
                str(REPO / "service" / script), "--host", "127.0.0.1", "--port", "8143",
                "--device", "cuda:0", "--dtype", "bfloat16", "--outputs-keep-days", "0", "--outputs-max-files", "0"]
        self.log = (self.output / ("directed-worker.log" if self.directed else "baseline-worker.log")).open("ab")
        self.process = subprocess.Popen(args, cwd=REPO, env=env, stdout=self.log, stderr=self.log,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        publish(self.output / f"worker-dispatch-{self.process.pid}.json", encoded({"pid": self.process.pid, "command": args}))
        self.client = httpx.Client(trust_env=False, timeout=300)
        for i in range(90):
            if self.process.poll() is not None:
                raise RuntimeError("Worker exited; inspect its log")
            try:
                r = self.client.get("http://127.0.0.1:8143/health", timeout=2)
                if r.status_code == 200:
                    self.identity = self.client.get("http://127.0.0.1:8143/api/v1/tts/runtime-identity").json()
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(1)
        self.__exit__(None, None, None)
        raise RuntimeError("Worker readiness timeout")

    def __exit__(self, *args):
        if self.process and self.process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"], capture_output=True)
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"], capture_output=True)
                else:
                    self.process.kill()
                self.process.wait()
        self.client.close()
        self.log.close()


def render(worker, project_id, identity, line, path, native_instruction=None):
    receipt_path = path.with_suffix(".receipt.json")
    from dubbing_service.app.audiobook.performance_provider import component_sha256
    payload = {k: identity[k] for k in ("prompt_artifact", "prompt_sha256", "reference_audio", "reference_audio_sha256")}
    payload.update(project_id=project_id, identity_id=identity["id"], text=line["text"], language="English", performance={})
    if native_instruction:
        payload["performance"] = {"control_policy": "native-instruct-icl-x1", "native_instruction": native_instruction,
                                  "seed_salt": 0, "direction_sha256": component_sha256(native_instruction, 0)}
    request_sha = sha(encoded(payload))
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_bytes())
        if receipt["request_sha256"] != request_sha or sha(path.read_bytes()) != receipt["audio_sha256"]:
            raise RuntimeError("Saved render receipt mismatch; existing sample preserved")
        return receipt
    legacy_pending = path.with_suffix(".pending.json")
    if legacy_pending.exists() and not receipt_path.exists():
        failure_path = path.with_suffix(".failure.json")
        old = json.loads(legacy_pending.read_bytes())
        failure = json.loads(failure_path.read_bytes()) if failure_path.exists() else {}
        if failure.get("status") not in {400, 409} or old.get("request_sha256") == request_sha:
            raise RuntimeError("Unknown or unchanged previous outcome; inspect saved dispatch evidence")
        # A recorded HTTP validation refusal returned no audio. The corrected
        # request gets a new dispatch identity; every earlier record is retained.
    pending = path.with_suffix(f".pending-{request_sha[:16]}.json")
    if pending.exists():
        raise RuntimeError("Unknown previous synthesis outcome; inspect pending record before another GPU call")
    publish(pending, encoded({"request_sha256": request_sha, "request": payload, "state": "dispatch_reserved"}))
    memory_samples = []
    done = threading.Event()
    def memory_probe():
        while not done.is_set():
            try:
                memory_samples.append(int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            done.wait(0.25)
    monitor = threading.Thread(target=memory_probe, daemon=True)
    monitor.start()
    start = time.perf_counter()
    try:
        response = worker.client.post("http://127.0.0.1:8143/api/v1/tts/character/synthesize", json=payload)
        elapsed = time.perf_counter() - start
        body = response.json()
        if response.status_code != 200:
            publish(path.with_suffix(".failure.json"), encoded({"status": response.status_code, "response": body}))
            raise RuntimeError(f"Synthesis held ({response.status_code}): {str(body)[:600]}")
    finally:
        done.set()
        monitor.join(timeout=3)
    audio = Path(body["audio_file"]).read_bytes()
    publish(path, audio)
    receipt = {"utterance_id": line["utterance_id"], "audio_sha256": sha(audio),
               "text_sha256": sha(line["text"].encode()), "request_sha256": request_sha,
               "runtime_sha256": body["worker_runtime_identity"]["sha256"],
               "latency_seconds": elapsed, "peak_vram_mb": max(memory_samples) if memory_samples else None,
               "vram_measurement": "sampled total-device MiB via nvidia-smi, includes desktop/background allocations",
               "worker_receipt": body, "native_instruction": native_instruction,
               "human_review": "pending", "production_ready": False}
    publish(receipt_path, encoded(receipt))
    print(path.name, "seconds", round(elapsed, 2), "audio", round(body["duration_seconds"], 2), flush=True)
    return receipt


def assemble(output, arm, samples, lines, *, file_prefix=None):
    import numpy as np
    import soundfile as sf
    from dubbing_service.app.audiobook import performance_boundaries as boundaries
    chunks, timing, previous = [], [], None
    source_cursor = 0
    rate = None
    for sample, line in zip(samples, lines):
        line = {**line, "source_revision_id": "review-script-v1", "start_offset": source_cursor,
                "end_offset": source_cursor + len(line["text"])}
        source_cursor = line["end_offset"] + 2
        path = Path(sample["audio_path"])
        audio, this_rate = sf.read(path, dtype="float32", always_2d=True)
        if rate is not None and rate != this_rate:
            raise RuntimeError("Sample domains differ; no implicit resampling")
        rate = this_rate
        measure = boundaries.measure_take(path, sample["audio_sha256"], text=line["text"])
        if previous:
            prev_line, prev_measure = previous
            classification = boundaries.classify_boundary(prev_line, line,
                role_types={u["role_id"]: "narrator" if u["role_type"] == "narration" else "character" for u in lines}, aliases={})
            klass = classification.get("boundary_class") or classification.get("class") or classification.get("classification")
            target = boundaries.default_policy()["targets_ms"].get(klass, 350)
            existing = (prev_measure["trailing"]["silent_frames"] + measure["leading"]["silent_frames"]) * 1000 / rate
            directed = arm in {"performance", "combined"}
            padding_ms = max(0, target - existing) if directed else 350
            chunks.append(np.zeros((round(rate * padding_ms / 1000), audio.shape[1]), dtype="float32"))
            timing.append({"left": prev_line["utterance_id"], "right": line["utterance_id"],
                           "classification": classification, "target_ms": target, "existing_silence_ms": existing,
                           "padding_ms": padding_ms, "shortening": False, "quality_improvement": "unmeasured"})
        chunks.append(audio)
        previous = line, measure
    data = io.BytesIO()
    sf.write(data, np.concatenate(chunks), rate, format="WAV", subtype="PCM_16")
    publish(output / f"{file_prefix or arm}-scene.wav", data.getvalue())
    publish(output / f"{file_prefix or arm}-timing.json", encoded(timing))


def generate(args, state):
    from dubbing_service.app.audiobook import casting_review as casting, voice_lab as lab
    from dubbing_service.app.storage.projects import load_project
    from dubbing_service.app.core import tts_backends
    from dubbing_service.app.audiobook.performance_scene_analysis import analyze_scenes
    from dubbing_service.app.models.project import Segment
    from dubbing_service.app.audiobook.director_evaluation import evaluate_comparison
    # Process-local harness endpoint. The persisted app default and registry files
    # are unchanged. Ordinary audition providers and Takes still own generation.
    original_spec = tts_backends.BACKEND_SPECS["qwen3tts"]
    tts_backends.BACKEND_SPECS["qwen3tts"] = replace(original_spec, port=8143)
    os.environ["DUBBING_QWEN3_TTS_BASE_URL"] = "http://127.0.0.1:8143"
    from dubbing_service.app.core.config import clear_config_caches
    clear_config_caches()
    existing_round = load_project(state["audition_project"]).voice_lab.casting["roles"]["audition-guest"]["round"]
    if not all(c["status"] == "ready" for c in existing_round["candidates"]):
        with Worker(args.output_dir) as worker:
            casting._round_worker(state["audition_project"], existing_round["intent_id"])
    audition = load_project(state["audition_project"])
    candidates = audition.voice_lab.casting["roles"]["audition-guest"]["round"]["candidates"]
    publish(args.output_dir / "audition-outcomes.json", encoded(candidates))
    ready = [c for c in candidates if c["status"] == "ready"]
    if not ready:
        raise RuntimeError("Production-path auditions are held; inspect audition-outcomes.json and jobs")
    # Recommendations remain unassigned: selection here is a review comparator,
    # not a human casting decision or a reusable approval.
    choice = lab._record(audition, ready[0]["key"])
    choice_project = load_project(choice["details"]["audition_project_id"])
    choice_identity = choice_project.roles[0].voice_identity.model_dump(mode="json")
    source = load_project(state["source_project"])
    identities = {r.id: r.voice_identity.model_dump(mode="json") for r in source.roles if r.voice_identity}
    lines = state["utterances"]
    analysis_lines = [Segment(id=u["utterance_id"], index=i, role_id=u["source_role_id"], text=u["text"])
                      for i, u in enumerate(lines)]
    scene_path = args.output_dir / "performance-analysis.json"
    if scene_path.exists():
        scene = json.loads(scene_path.read_bytes())
        bound = {d["utterance_id"]: d["text_sha256"] for d in scene["directions"]}
        if bound != {u["utterance_id"]: sha(u["text"].encode()) for u in lines}:
            raise RuntimeError("Saved scene interpretation belongs to changed prose")
    else:
        scene = analyze_scenes(source, analysis_lines, cache_root=args.output_dir,
                               runtime_profile="faster-base-032-direction-x1")
        publish(scene_path, encoded(scene))
    directions = {d["utterance_id"]: d for d in scene["directions"]}
    hypotheses = {"harry": "Speak with quiet urgency and concern.",
                  "ron": "Speak with dry amusement and restrained sarcasm.",
                  "dumbledore": "Speak gently and reassuringly, with an unhurried cadence."}
    publish(args.output_dir / "direction-hypotheses.json", encoded({
        "origin": "engineering_comparator_hypothesis", "approved": False,
        "reason": "The conservative analyzer supplied no specific emotion on these neutral audition excerpts; these independent directions test native instruction conditioning, not book facts.",
        "directions": hypotheses, "narrator_forwarding": False, "source_text_unchanged": True}))
    arms = {}
    with Worker(args.output_dir, directed=True) as worker:
        for arm in ("baseline", "casting", "performance", "combined"):
            samples = []
            for line in lines:
                use_cast = arm in {"casting", "combined"} and line["role_id"] == "audition-guest"
                identity = choice_identity if use_cast else identities[line["source_role_id"]]
                project_id = choice_project.id if use_cast else source.id
                directed = arm in {"performance", "combined"} and line["role_type"] != "narration"
                candidate = directions[line["utterance_id"]].get("native_instruction_candidate") or {}
                instruction = (candidate.get("text") or hypotheses[line["source_role_id"]]) if directed else None
                # No invented extra direction if evidence supplies none.
                prefix = arm + "-native" if arm in {"performance", "combined"} else arm
                path = args.output_dir / "samples" / f"{prefix}-{line['utterance_id']}.wav"
                receipt = render(worker, project_id, identity, line, path, instruction)
                runtime = receipt["runtime_sha256"]
                samples.append({"utterance_id": line["utterance_id"], "role_id": line["role_id"], "text": line["text"],
                                "audio_path": str(path), "audio_sha256": receipt["audio_sha256"],
                                "provenance": {"source_sha256": state["source_sha256"]},
                                "voice": {"voice_id": identity["id"], "version": identity["prompt_sha256"],
                                          "identity_sha256": sha(encoded(identity))},
                                "direction": {"direction_sha256": sha(instruction.encode())} if instruction else None,
                                "receipt": {"path": str(path.with_suffix('.receipt.json'))}})
            arms[arm] = {"engine": {"name": "faster-base-032-direction-x1 (same pinned Base, neutral baseline through parent)",
                                    "runtime_sha256": runtime}, "samples": samples}
            assemble(args.output_dir, arm, samples, lines, file_prefix=arm + "-semantic-v2")
    manifest = {"schema_version": "director-comparison-manifest-v1", "comparison_id": "directors-real-review-v1",
                "base_dir": str(args.output_dir),
                "source": {"source_sha256": state["source_sha256"], "kind": state["source_kind"]},
                "utterances": lines, "arms": arms}
    publish(args.output_dir / "comparison-manifest-directed.json", encoded(manifest))
    evaluation = evaluate_comparison(manifest, output_dir=args.output_dir / "evaluation")
    print("Comparison completed; human review pending", flush=True)
    tts_backends.BACKEND_SPECS["qwen3tts"] = original_spec
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-project", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    args.source_root, args.output_dir = args.source_root.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir.is_relative_to(args.source_root):
        raise RuntimeError("Demo output must be outside the source project store")
    state = prepare(args)
    print(json.dumps({"audition_project": state["audition_project"], "lines": len(state["utterances"]),
                      "output": str(args.output_dir), "human_review": "pending"}), flush=True)
    if args.generate:
        generate(args, state)


if __name__ == "__main__":
    main()
