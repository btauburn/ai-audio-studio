"""CPU tests for the resumable coached full-chapter renderer.

Everything here is a labeled synthetic fixture: invented prose, tone-signal "Takes", fake clone files
and a fake worker. No real book, voice, model, GPU, network or live project is read or generated, and
nothing here is evidence about production readiness or audio quality.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubbing_service.app.audiobook import coached_chapter_batch as batch
from dubbing_service.app.audiobook import qa, scene_coaching
from dubbing_service.app.audiobook import voice_lab_chapter_export as exporter
from dubbing_service.app.audiobook.fingerprint import text_sha256, voice_revision_for_role
from dubbing_service.app.audiobook.scene_dialogue_plan import _sha
from dubbing_service.app.models.audiobook import Attribution, QaCheckResult, SourceAnchor, SourceRevision
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.models.project import Role, Segment, SegmentTake
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import build_project, use_projects_root
from tests.test_epub_chapter_selection import NS, epub, xhtml

RATE = 24000
pilot = batch.pilot()
RENDER = pilot.default_render_tools()[1]       # the real render() owner; the real Worker is never constructed


def sha(data):
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- synthetic world

# (kind, role, text). "n" narrator, "e" Erick, "j" Jane, "b" scene break.
CHAPTER_4 = [("n", "004"), ("n", "The gate was shut and the road behind them was dark."),
             ("e", "“Open it,”"), ("n", "said Erick."), ("j", "“I will not,”"), ("n", "said Jane."),
             ("b", "***"), ("n", "Morning came slowly over the wall."), ("e", "“Now?”"), ("n", "asked Erick.")]
CHAPTER_5 = [("n", "005"), ("n", "A new road began beyond the gate."), ("j", "“We walk,”"), ("n", "said Jane."),
             ("e", "“We walk together,”"), ("n", "said Erick.")]
CHAPTERS = {4: CHAPTER_4, 5: CHAPTER_5}


def write_tone(path, seconds, frequency):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(RATE * seconds)
    samples = [int(0.4 * 32767 * math.sin(2 * math.pi * frequency * i / RATE)) for i in range(frames)]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(struct.pack(f"<{frames}h", *samples))
    return path


def make_identity(folder, name, seed):
    voice_id = f"voice_{name}"
    vfolder = folder / "casting" / "voices" / voice_id
    vfolder.mkdir(parents=True)
    reference = bytes((seed * 7 + i * 13) % 251 for i in range(300))
    prompt = bytes((seed * 11 + i * 17) % 241 for i in range(400))
    (vfolder / "reference.wav").write_bytes(reference)
    (vfolder / "conditioning.npz").write_bytes(prompt)
    return CharacterVoiceIdentity(
        id=voice_id, role_id=name, status="LOCKED", mechanism="qwen_reference", backend="qwen3tts",
        dossier_revision="fixture", casting_profile_revision="fixture", design_instruction="fixture only",
        model="mock-no-model", reference_audio=f"casting/voices/{voice_id}/reference.wav",
        reference_audio_sha256=sha(reference), prompt_artifact=f"casting/voices/{voice_id}/conditioning.npz",
        prompt_sha256=sha(prompt),
        provenance={"origin": "eligible_reference_clone", "actor_audio_used": True})


def epub_for(chapters, *, mutate=None):
    docs = [("Cover.xhtml", "cover", f'<html {NS}><body><img src="cover.jpg" alt=""/></body></html>', None)]
    for number, rows in chapters.items():
        body = "".join(f"<p>{text}</p>" for _, text in rows[1:])
        docs.append((f"{number:04d}_{number:03d}.xhtml", f"c{number}", xhtml(f"{number:03d}", body), f"{number:03d}"))
    return epub(docs)


def build_chapter(root, number, rows, epub_bytes, *, seed):
    project = build_project(name=f"coached batch synthetic fixture {number}", chapter_titles=[f"{number:03d}"])
    chapter = project.chapters[0]
    base = root / project.id
    (base / "source.snapshot").write_bytes(epub_bytes)
    source = SourceRevision(id="source_fixture", uri="fixture:epub", sha256=sha(epub_bytes), snapshot_file="source.snapshot")
    project.source_revisions = [source]
    project.metadata["coached_batch_fixture"] = True
    project.roles = []
    names = {"n": ("narrator", "narrator"), "e": ("erick", "Erick"), "j": ("jane", "Jane")}
    for index, (key, (rid, label)) in enumerate(names.items()):
        project.roles.append(Role(id=rid, name=label, role_type="narrator" if key == "n" else "character",
                                  tts_backend="qwen3tts", voice_identity=make_identity(base, rid, seed + index)))
    roles = {r.id: r for r in project.roles}
    segments, offset = [], 0
    for index, (key, text) in enumerate(rows):
        rid = names.get(key, names["n"])[0]
        anchor = SourceAnchor(source_revision_id=source.id, chapter_id=chapter.id, start_offset=offset,
                              end_offset=offset + len(text), occurrence_index=index)
        offset += len(text) + 2
        segment = Segment(id=f"{project.id}-{index:02}", chapter_id=chapter.id, index=index, role_id=rid, text=text,
                          source_anchor=anchor, attribution=Attribution(speaker_id=rid, source="user"))
        if key != "b":
            audio = write_tone(base / f"audio/{segment.id}.wav", max(0.5, 0.05 * len(text)), 200 + 30 * index)
            identity = roles[rid].voice_identity
            checksum = exporter.file_sha256(audio)
            take = SegmentTake(
                id=f"take_{segment.id}", role_id=rid, audio_file=audio.relative_to(base).as_posix(), audio_sha256=checksum,
                source_text_hash=text_sha256(text), input_fingerprint="fixture-current",
                fingerprint_components={"voice_revision": voice_revision_for_role(roles[rid])},
                generation_metadata={"character_voice_identity": identity.model_dump(mode="json"),
                                     "runtime_profile": "faster-base-032",
                                     "provider_calls": [{"identity_id": identity.id, "prompt_sha256": identity.prompt_sha256}]})
            take.qa_report = qa.build_report(take.id, checksum, [QaCheckResult(check_id="decodable", check_version="fixture", status="passed")])
            segment.takes = [take]
            segment.selected_take_id = take.id
        segments.append(segment)
    breaks = [s for s, (k, _) in zip(segments, rows) if k == "b"]
    project.metadata["chapter_rerun"] = {
        "authorized": True, "source_project_id": "original-untouched", "chapter_id": chapter.id,
        "source_utterance_ids": [s.id for s in segments],
        "speech_utterance_ids": [s.id for s in segments if s not in breaks],
        "source_revisions": [{"id": source.id, "sha256": source.sha256, "snapshot_file": source.snapshot_file}],
        "source_utterances": [{"utterance_id": s.id, "index": s.index, "chapter_id": s.chapter_id,
                               "source_text_sha256": text_sha256(s.text),
                               "source_anchor": s.source_anchor.model_dump(mode="json"),
                               "spoken_text_revision_id": s.spoken_text_revision_id} for s in segments],
        "scene_breaks": [{"utterance_id": s.id, "index": s.index, "source_text_sha256": text_sha256(s.text),
                          "duration_seconds": 1.25} for s in breaks]}
    from dubbing_service.app.audiobook.voice_lab_chapter import _hash, _source_proof
    project.metadata["chapter_rerun"]["expected_source_hash"] = _hash(
        _source_proof(project, project.metadata["chapter_rerun"]["source_utterances"]))
    save_project(project)
    save_segments(project.id, segments)
    return project.id


@pytest.fixture
def world(monkeypatch, tmp_path, request):
    """Optional indirect param: the chapter rows the EPUB (and so the source revision) is built from."""
    root = use_projects_root(monkeypatch, tmp_path)
    epub_bytes = epub_for(getattr(request, "param", None) or CHAPTERS)
    epub_path = tmp_path / "synthetic.epub"
    epub_path.write_bytes(epub_bytes)
    ids = [build_chapter(root, number, rows, epub_bytes, seed=10 * number) for number, rows in CHAPTERS.items()]
    monkeypatch.setattr(exporter.pipeline, "_requirement_for_take", lambda *_: SimpleNamespace(fingerprint=lambda: "fixture-current"))
    monkeypatch.setattr(batch, "_mp3", lambda master, target: target.write_bytes(master.read_bytes()))  # signal copy, never an MP3
    monkeypatch.setattr(batch, "_decoded_duration", lambda path: wave_seconds(path))
    monkeypatch.setattr(batch, "_loudness", lambda _: -20.0)
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path / "ambient-unrelated"))
    return SimpleNamespace(root=root, epub=epub_path, ids=ids, chapters=list(CHAPTERS), tmp=tmp_path)


def wave_seconds(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def args_for(world, **overrides):
    values = dict(epub=world.epub, source_root=world.root, projects=world.ids, chapters=world.chapters,
                  output_dir=world.tmp / "out", coaching_dir=world.tmp / "coaching", generate=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def tree_hashes(root):
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in sorted(Path(root).rglob("*")) if p.is_file()}


def make_bundle(request, *, mutate=None):
    """A synthetic stand-in for a captured model response; every field is derived from the request."""
    scenes = []
    for scene in request["scenes"]:
        context = {row["utterance_id"]: row for row in scene["context"]}
        directions = []
        for position, uid in enumerate(scene["selected_dialogue_ids"]):
            text = context[uid]["text"]
            directions.append({
                "utterance_id": uid, "objective": f"Fixture objective {position}", "emotion": "fixture tension",
                "intensity": "strong" if position % 2 else "moderate", "cadence": "steady then firm",
                "emphasis": [text[:3]], "ending": "settle", "confidence": "moderate",
                "native_instruction": f"Deliver this line with controlled urgency number {position}; keep every word clear.",
                "evidence": [{"utterance_id": uid, "start": 0, "end": len(text), "text": text}]})
        scenes.append({"scene_id": scene["scene_id"], "directions": directions})
    response = {"schema_version": scene_coaching.RESPONSE_SCHEMA, "scenes": scenes}
    bundle = {"schema_version": scene_coaching.BUNDLE_SCHEMA, "request": request, "response": response,
              "request_sha256": _sha(request), "response_sha256": _sha(response),
              "provenance": {"origin": "model", "model_requested": "synthetic-test-fixture-not-a-real-model",
                             "receipt_sha256": "a1" * 32, "fixture": True}}
    if mutate:
        mutate(bundle)
        bundle["request_sha256"], bundle["response_sha256"] = _sha(bundle["request"]), _sha(bundle["response"])
    return bundle


def write_bundles(world, state, out, *, only=None, mutate=None):
    folder = world.tmp / "coaching"
    folder.mkdir(exist_ok=True)
    for pid, info in state["requests"].items():
        if only is not None and pid not in only:
            continue
        request = json.loads((out / info["file"]).read_bytes())
        bundle = make_bundle(request, mutate=(lambda b: mutate(pid, b)) if mutate else None)
        (folder / f"coaching-{pid}.json").write_bytes(pilot.encoded(bundle))
    return folder


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def json(self):
        return self.body


class FakeClient:
    def __init__(self, workdir, runtime, *, forward=True):
        self.workdir, self.runtime, self.forward, self.calls = workdir, runtime, forward, []

    def post(self, url, json):  # noqa: A002 - mirrors httpx
        self.calls.append(json)
        seed = int(hashlib.sha256((json["identity_id"] + json["text"]).encode()).hexdigest()[:8], 16)
        path = write_tone(self.workdir / f"{len(self.calls):04}.wav", max(0.5, 0.05 * len(json["text"])), 180 + seed % 90)
        performance = json["performance"]
        direction = None
        if performance.get("native_instruction") is not None:
            text = performance["native_instruction"]
            direction = {"policy": pilot.DIRECTION_POLICY, "forwarded": self.forward,
                         "instruct_sha256": sha(text.encode()), "instruct_chars": len(text), "conditioning_mode": "icl",
                         "seed_salt": 0, "seed_effective": seed % 2**32, "direction_sha256": performance["direction_sha256"]}
        return FakeResponse({"success": True, "audio_file": str(path), "duration_seconds": wave_seconds(path),
                             "worker_runtime_identity": {"sha256": self.runtime}, "fallback": None,
                             "identity_id": json["identity_id"], "prompt_sha256": json["prompt_sha256"],
                             "execution_profile": {"runtime_profile": pilot.DIRECTED_PROFILE}, "performance": {},
                             "generation": {"direction": direction},
                             "generation_settings": {"seed": seed, "temperature": 0.9}})


class FakeWorker:
    instances = []
    runtime = "7" * 64
    profile = pilot.DIRECTED_PROFILE
    forward = True

    def __init__(self, output, directed=False):
        assert directed is True
        scratch = Path(output) / "worker-scratch"
        scratch.mkdir(exist_ok=True)
        self.client = FakeClient(scratch, self.runtime, forward=self.forward)
        self.identity = {"sha256": self.runtime, "contract": {"runtime_profile": self.profile,
                                                              "checkpoint_manifest_sha256": "c" * 64, "model_id": "m"}}
        self.entered = self.exited = False
        FakeWorker.instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *args):
        self.exited = True


@pytest.fixture(autouse=True)
def reset_worker():
    FakeWorker.instances, FakeWorker.runtime, FakeWorker.profile, FakeWorker.forward = [], "7" * 64, pilot.DIRECTED_PROFILE, True


@pytest.fixture
def prepared(world):
    args = args_for(world)
    state = batch.prepare_batch(args)
    return world, args, state


@pytest.fixture
def ready(prepared):
    world, args, state = prepared
    write_bundles(world, state, args.output_dir)
    state = batch.apply_bundles(args, state)
    assert state["status"] == "ready", state["holds"]
    return world, args, state


def render_function():
    return RENDER


def all_calls():
    return [call for worker in FakeWorker.instances for call in worker.client.calls]


# --------------------------------------------------------------------------- prepare


def test_prepare_freezes_source_cast_and_requests_without_coaching_or_gpu(world):
    before = tree_hashes(world.root)
    args = args_for(world)
    state = batch.prepare_batch(args)
    assert state["status"] == "awaiting_coaching" and state["holds"] == []
    assert state["review_only"] and state["production_ready"] is False and state["human_review"] == "pending"
    assert state["fixture"] is True and not state["approvals_created"] and not state["takes_selected"]
    assert [s["chapter_number"] for s in state["sources"]] == [4, 5]
    assert tree_hashes(world.root) == before                      # canonical store read only
    out = args.output_dir
    assert not (out / "plans").glob("coached-*") or not list((out / "plans").glob("coached-*"))
    assert not (out / "samples").exists() and not list(out.glob("worker-*"))   # nothing synthesized
    speech = [l for l in state["lines"]]
    assert len(speech) == 9 + 6                                   # the scene break is not speech
    assert sum(l["narrator"] for l in speech) == 10 and sum(not l["narrator"] for l in speech) == 5
    for pid, info in state["requests"].items():                  # one whole-scene request per chapter project
        request = json.loads((out / info["file"]).read_bytes())
        assert set(request) == {"instruction", "constraints", "response_contract", "scenes"}
        selected = [uid for scene in request["scenes"] for uid in scene["selected_dialogue_ids"]]
        assert selected == info["dialogue_ids"] and len(selected) == len(set(selected))
        narrators = {r["utterance_id"] for scene in request["scenes"] for r in scene["context"] if r["narrator"]}
        assert narrators and not narrators & set(selected)       # narration is context, never a requested direction
        assert all(set(r) == {"utterance_id", "text", "role_id", "narrator", "role_name"}
                   for scene in request["scenes"] for r in scene["context"])
        assert all(scene["project_id"] == pid for scene in request["scenes"])
        scene_coaching.validate_bundle(make_bundle(request))      # request is compatible with the coaching validator


def test_neutral_narration_bytes_are_copied_unchanged_and_dialogue_is_not(prepared):
    world, args, state = prepared
    for line in state["lines"]:
        original = world.root / line["project_id"] / line["neutral_take"]["source_audio_file"]
        assert sha(original.read_bytes()) == line["neutral_take"]["audio_sha256"]
        if line["narrator"]:
            copy_path = args.output_dir / line["neutral_copy"]["file"]
            assert copy_path.read_bytes() == original.read_bytes() and line["mode"] == "neutral_reuse"
        else:
            assert line["neutral_copy"] is None and line["mode"] == "directed_dialogue"
            assert line["neutral_take"]["take_id"]                # the existing neutral Take is retained as a reference
    assert all(line["voice"]["provenance_origin"] == "eligible_reference_clone" for line in state["lines"])


def test_prepare_retry_returns_the_frozen_state_and_changes_nothing(prepared):
    world, args, state = prepared
    before = tree_hashes(args.output_dir)
    again = batch.prepare_batch(args)
    assert again == state and tree_hashes(args.output_dir) == before
    with pytest.raises(batch.BatchError, match="different EPUB, chapter or project"):
        batch.prepare_batch(args_for(world, projects=list(reversed(world.ids)), chapters=[4, 5]))


def test_prepare_resumes_after_an_interrupted_attempt(world):
    args = args_for(world)
    real = batch._build_request
    calls = []

    def crash_second(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return real(*a, **k)
    batch._build_request = crash_second
    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            batch.prepare_batch(args)
    finally:
        batch._build_request = real
    assert not (args.output_dir / "state.json").exists()
    state = batch.prepare_batch(args)                              # published inputs are identical, not conflicts
    assert state["status"] == "awaiting_coaching"


def test_missing_wrong_or_unlisted_epub_is_refused_before_any_output(world):
    other = world.tmp / "other.epub"
    other.write_bytes(epub_for({4: [("n", "004"), ("n", "Different prose.")], 5: CHAPTER_5}))
    with pytest.raises(batch.BatchError, match="exact EPUB"):
        batch.prepare_batch(args_for(world, epub=other))
    with pytest.raises(batch.BatchError, match="No chapter titled"):
        batch.prepare_batch(args_for(world, chapters=[4, 6]))


ALTERED = {4: [(k, t.replace("Open", "Opem")) for k, t in CHAPTER_4], 5: CHAPTER_5}
EXTRA = {4: CHAPTER_4 + [("n", "An extra closing paragraph the source lost.")], 5: CHAPTER_5}
SPACED = {4: [(k, t.replace(" ", " \t ")) if k == "n" else (k, t) for k, t in CHAPTER_4], 5: CHAPTER_5}


@pytest.mark.parametrize("world", [ALTERED, EXTRA], indirect=True, ids=["changed-character", "lost-paragraph"])
def test_every_nonblank_epub_character_must_be_covered(world):
    with pytest.raises(batch.BatchError, match="every EPUB prose character; first difference"):
        batch.prepare_batch(args_for(world))
    assert not (world.tmp / "out" / "state.json").exists()


@pytest.mark.parametrize("world", [SPACED], indirect=True, ids=["whitespace-only"])
def test_whitespace_is_the_only_difference_ignored(world):
    assert batch.prepare_batch(args_for(world))["status"] == "awaiting_coaching"


@pytest.mark.parametrize("change", ["origin", "unlocked", "missing_identity", "backend"])
def test_non_clone_or_unlocked_voices_are_refused(world, change):
    project = load_project(world.ids[0])
    role = next(r for r in project.roles if r.id == "erick")
    if change == "origin":
        role.voice_identity.provenance = {"origin": "generated_original_voice", "actor_audio_used": False}
    elif change == "unlocked":
        role.voice_identity.status = "REVIEW"
    elif change == "missing_identity":
        role.voice_identity = None
    elif change == "backend":
        role.tts_backend = "indextts"
    save_project(project)
    with pytest.raises(batch.BatchError):
        batch.prepare_batch(args_for(world))
    assert not (world.tmp / "out" / "state.json").exists()


def test_clone_file_or_neutral_route_changes_are_refused(world):
    voice = world.root / world.ids[0] / "casting" / "voices" / "voice_erick" / "conditioning.npz"
    voice.write_bytes(b"changed prompt")
    with pytest.raises(batch.BatchError, match="hash-mismatched|artifact"):
        batch.prepare_batch(args_for(world))


def test_neutral_take_must_prove_the_approved_faster_base_route(world):
    segments = load_segments(world.ids[0])
    segments[1].takes[0].generation_metadata["runtime_profile"] = "faster-base-032-direction-x1"
    save_segments(world.ids[0], segments)
    with pytest.raises(batch.BatchError):
        batch.prepare_batch(args_for(world))


def test_output_inside_the_source_store_and_bad_arguments_are_refused(world):
    with pytest.raises(batch.BatchError, match="outside the canonical"):
        batch.prepare_batch(args_for(world, output_dir=world.root / "out"))
    with pytest.raises(batch.BatchError, match="same length"):
        batch.prepare_batch(args_for(world, chapters=[4]))
    with pytest.raises(batch.BatchError, match="ascending"):
        batch.prepare_batch(args_for(world, chapters=[5, 4]))
    with pytest.raises(batch.BatchError, match="distinct valid project ids"):
        batch.prepare_batch(args_for(world, projects=[world.ids[0], world.ids[0]]))
    with pytest.raises(batch.BatchError, match="distinct valid project ids"):
        batch.prepare_batch(args_for(world, projects=["../x", world.ids[1]]))


# --------------------------------------------------------------------------- coaching bundles


@pytest.fixture(autouse=True)
def no_real_worker(monkeypatch):
    def forbidden():
        raise AssertionError("A test reached the real GPU worker")
    monkeypatch.setattr(pilot, "default_render_tools", forbidden)


def test_no_bundles_leaves_the_batch_awaiting_and_publishes_nothing(prepared):
    world, args, state = prepared
    before = tree_hashes(args.output_dir)
    state = batch.apply_bundles(args, state)
    assert state["status"] == "awaiting_coaching"
    assert {v["status"] for v in state["coaching"].values()} == {"awaiting_bundle"}
    assert not (args.output_dir / "applied").exists() and not list((args.output_dir / "plans").glob("coached-*"))
    assert {k: v for k, v in tree_hashes(args.output_dir).items() if k != "state.json"} == \
        {k: v for k, v in before.items() if k != "state.json"}
    with pytest.raises(batch.BatchError, match="no coaching has been applied"):
        batch.generate_batch(args, state, FakeWorker, render_function())
    assert FakeWorker.instances == []


def test_one_missing_bundle_keeps_the_batch_awaiting(prepared):
    world, args, state = prepared
    first = next(iter(state["requests"]))
    write_bundles(world, state, args.output_dir, only={first})
    state = batch.apply_bundles(args, state)
    assert state["status"] == "awaiting_coaching"
    assert state["coaching"][first]["status"] == "applied"
    assert [v["status"] for k, v in state["coaching"].items() if k != first] == ["awaiting_bundle"]


def test_valid_bundles_publish_immutable_coached_plans_covering_all_dialogue(ready):
    world, args, state = ready
    out = args.output_dir
    for pid in state["requests"]:
        coached = json.loads((out / "plans" / f"coached-{pid}.json").read_bytes())
        assert (out / "plans" / f"base-{pid}.json").is_file() and (out / "applied" / f"{pid}.json").is_file()
        assert coached["recommendation_only"] is True and coached["approved"] is False and coached["human_review"] == "pending"
        directions = {d["utterance_id"]: d for d in coached["directions"]}
        for line in (l for l in state["lines"] if l["project_id"] == pid):
            row = directions[line["utterance_id"]]
            if line["narrator"]:
                assert row["native_instruction"] is None
            else:
                assert row["basis"] == "model" and 0 < len(row["native_instruction"]) <= 220 and row["coaching"]
                assert row["text_sha256"] == line["text_sha256"]      # prose is never rewritten
        bundle = (world.tmp / "coaching" / f"coaching-{pid}.json").read_bytes()
        assert (out / "coaching" / f"bundle-{pid}.json").read_bytes() == bundle
        receipt = json.loads((out / "applied" / f"{pid}.json").read_bytes())
        assert receipt["model_provenance"]["origin"] == "model" and receipt["approved"] is False
    assert state["production_ready"] is False and state["approvals_created"] is False


def test_identical_retry_is_a_no_op_and_a_changed_bundle_never_overwrites(ready):
    world, args, state = ready
    before = tree_hashes(args.output_dir)
    again = batch.apply_bundles(args, state)
    assert again["status"] == "ready" and tree_hashes(args.output_dir) == before
    pid = next(iter(state["requests"]))
    changed = make_bundle(json.loads((args.output_dir / state["requests"][pid]["file"]).read_bytes()),
                          mutate=lambda b: b["response"]["scenes"][0]["directions"][0].update(
                              native_instruction="Deliver this with a different, quieter shape; keep every word clear."))
    (world.tmp / "coaching" / f"coaching-{pid}.json").write_bytes(pilot.encoded(changed))
    held = batch.apply_bundles(args, again)
    assert held["status"] == "held" and held["coaching"][pid]["status"] == "refused"
    assert "already applied" in held["holds"][0]["detail"]
    assert {k: v for k, v in tree_hashes(args.output_dir).items() if k != "state.json"} == \
        {k: v for k, v in before.items() if k != "state.json"}


def _unsafe(b):
    b["response"]["scenes"][0]["directions"][0]["native_instruction"] = "Use a Scottish accent and a deep baritone."


def _too_long(b):
    b["response"]["scenes"][0]["directions"][0]["native_instruction"] = "Deliver " + "x" * 215


def _narrator_target(b):
    scene = b["request"]["scenes"][0]
    narrator = next(r["utterance_id"] for r in scene["context"] if r["narrator"])
    b["response"]["scenes"][0]["directions"][0]["utterance_id"] = narrator


def _missing(b):
    b["response"]["scenes"][0]["directions"].pop()


def _duplicate(b):
    rows = b["response"]["scenes"][0]["directions"]
    rows.append(copy.deepcopy(rows[0]))


def _bad_evidence(b):
    b["response"]["scenes"][0]["directions"][0]["evidence"][0]["start"] = 1


def _human(b):
    b["provenance"]["origin"] = "human"


def _other_request(b):
    b["request"]["instruction"] = "A different instruction than the one frozen at preparation."


def _extra_scene(b):
    b["response"]["scenes"].append({"scene_id": "invented", "directions": []})


@pytest.mark.parametrize("mutate", [_unsafe, _too_long, _narrator_target, _missing, _duplicate, _bad_evidence,
                                    _human, _other_request, _extra_scene])
def test_unsafe_incomplete_or_foreign_bundles_hold_and_publish_nothing(prepared, mutate):
    world, args, state = prepared
    first = next(iter(state["requests"]))
    write_bundles(world, state, args.output_dir, only={first}, mutate=lambda pid, b: mutate(b))
    state = batch.apply_bundles(args, state)
    assert state["status"] == "held" and state["coaching"][first]["status"] == "refused"
    out = args.output_dir
    assert not (out / "applied").exists() and not (out / "coaching").exists()
    assert not list((out / "plans").glob("coached-*"))
    with pytest.raises(batch.BatchError):
        batch.generate_batch(args, state, FakeWorker, render_function())
    assert FakeWorker.instances == []


def test_bundle_hash_tamper_is_refused(prepared):
    world, args, state = prepared
    first = next(iter(state["requests"]))
    folder = write_bundles(world, state, args.output_dir, only={first})
    path = folder / f"coaching-{first}.json"
    bundle = json.loads(path.read_bytes())
    bundle["response"]["scenes"][0]["directions"][0]["emotion"] = "tampered after capture"
    path.write_bytes(pilot.encoded(bundle))
    held = batch.apply_bundles(args, state)
    assert held["status"] == "held" and "hash" in held["holds"][0]["detail"]


def test_other_projects_bundle_cannot_satisfy_this_request(prepared):
    world, args, state = prepared
    first, second = list(state["requests"])
    folder = write_bundles(world, state, args.output_dir)
    (folder / f"coaching-{second}.json").write_bytes((folder / f"coaching-{first}.json").read_bytes())
    held = batch.apply_bundles(args, state)
    assert held["status"] == "held" and held["coaching"][first]["status"] == "applied"
    assert held["coaching"][second]["status"] == "refused"


def test_source_or_cast_drift_blocks_coaching_but_notification_counters_do_not(prepared):
    world, args, state = prepared
    write_bundles(world, state, args.output_dir)
    path = world.root / world.ids[0] / "project.json"
    raw = json.loads(path.read_bytes())
    raw["voice_lab"]["revision"] += 7                                  # delivery bookkeeping only
    raw["voice_lab"].setdefault("casting", {})["notifications"] = {"complete": {"next_retry_at": "later"}}
    path.write_bytes(json.dumps(raw).encode())
    assert batch.apply_bundles(args, state)["status"] == "ready"
    raw["roles"][1]["name"] = "Renamed Erick"                          # a real cast/source change
    path.write_bytes(json.dumps(raw).encode())
    with pytest.raises(batch.BatchError, match="Canonical source changed"):
        batch.apply_bundles(args, state)


def test_current_clone_file_change_blocks_coaching(prepared):
    world, args, state = prepared
    write_bundles(world, state, args.output_dir)
    (world.root / world.ids[1] / "casting" / "voices" / "voice_jane" / "reference.wav").write_bytes(b"changed")
    held = batch.apply_bundles(args, state)
    assert held["status"] == "held" and held["coaching"][world.ids[1]]["status"] == "refused"
    assert "hash-mismatched" in held["holds"][0]["detail"] and held["coaching"][world.ids[0]]["status"] == "applied"
    assert not (args.output_dir / "applied" / f"{world.ids[1]}.json").exists()
    assert not (args.output_dir / "plans" / f"coached-{world.ids[1]}.json").exists()
    with pytest.raises(batch.BatchError, match="no coaching has been applied"):
        batch.generate_batch(args, held, FakeWorker, render_function())


# --------------------------------------------------------------------------- generation


def test_generation_synthesizes_only_coached_dialogue_and_assembles_review_chapters(ready):
    world, args, state = ready
    source_before = tree_hashes(world.root)
    result = batch.generate_batch(args, state, FakeWorker, render_function())
    out = args.output_dir
    dialogue = [l for l in state["lines"] if not l["narrator"]]
    narrators = [l for l in state["lines"] if l["narrator"]]
    assert result["status"] == "generated" and len(FakeWorker.instances) == 1
    worker = FakeWorker.instances[0]
    assert worker.entered and worker.exited
    sent = all_calls()
    assert len(sent) == len(dialogue) == 5 and all(c["performance"]["native_instruction"] for c in sent)
    spoken = {c["text"] for c in sent}
    assert spoken == {l["text"] for l in dialogue} and not spoken & {l["text"] for l in narrators}
    assert result["audit"]["directed_forwarded"] == 5 and result["audit"]["narrator_instructions"] == 0
    assert not any((out / "samples" / n["project_id"] / f"{n['utterance_id']}.wav").exists() for n in narrators)
    assert tree_hashes(world.root) == source_before                 # no Take, project or segment changed
    for chapter in state["chapters"]:
        number = chapter["chapter_number"]
        receipt = json.loads((out / "chapters" / chapter["project_id"] / f"chapter-{number:03}.json").read_bytes())
        assert receipt["status"] == "ready_for_listening" and receipt["review_only"] is True
        assert receipt["human_review"] == "pending" and receipt["production_ready"] is False
        assert receipt["takes_selected"] is False and receipt["approvals_created"] is False
        assert receipt["fixture"]["is_fixture"] is True and receipt["technical_qa"] == "passed"
        for kind in ("master", "audio"):
            assert sha((out / receipt[kind + "_file"]).read_bytes()) == receipt[kind + "_sha256"]
        mine = [l for l in state["lines"] if l["project_id"] == chapter["project_id"]]
        speech = [r for r in receipt["source_to_audio"] if r["kind"] == "speech"]
        assert [r["utterance_id"] for r in speech] == [l["utterance_id"] for l in mine]
        assert receipt["counts"]["scene_breaks"] == sum(r["kind"] == "scene_break" for r in chapter["rows"])
        assert {r["kind"] for r in receipt["source_to_audio"]} <= {"speech", "scene_break", "excluded"}
        for row, line in zip(speech, mine):
            assert row["source_anchor"] == line["source"] and row["role_id"] == line["role_id"]
            assert row["voice_id"] == line["voice"]["id"] and row["neutral_take_reference"]["take_id"]
            if line["narrator"]:
                assert row["provenance"] == "existing_neutral_take_reused_unchanged" and row["native_instruction"] is None
                assert row["audio_sha256"] == line["neutral_take"]["audio_sha256"]
            else:
                assert row["provenance"] == "new_experimental_synthesis" and row["receipt"]["raw"]["sha256"]
                assert row["coaching"]["scene_id"] and row["coaching"]["reading"]["objective"]
                assert row["audio_sha256"] != line["neutral_take"]["audio_sha256"]   # new audio, distinct from the neutral Take
        assert receipt["clone_binding"] and receipt["coaching"]["bundle_sha256"] and receipt["runtime"]["runtime_sha256"] == "7" * 64
        # No trimming: each source file's frames appear intact inside the master at its mapped position.
        with wave.open(str(out / receipt["master_file"]), "rb") as master:
            frames = master.readframes(master.getnframes())
        for row in speech:
            with wave.open(str(out / row["audio_file"]), "rb") as clip:
                data = clip.readframes(clip.getnframes())
            start = round(row["start_seconds"] * RATE) * 2
            assert frames[start:start + len(data)] == data
    summary = json.loads((out / "batch-result.json").read_bytes())
    assert summary["review_only"] is True and summary["production_ready"] is False and summary["fixture"]["is_fixture"]
    assert json.loads((out / "state.json").read_bytes())["status"] == "generated"


def test_scene_break_receives_its_canonical_separator_target_without_trimming(ready):
    world, args, state = ready
    batch.generate_batch(args, state, FakeWorker, render_function())
    chapter = state["chapters"][0]
    receipt = json.loads((args.output_dir / "chapters" / chapter["project_id"] / "chapter-004.json").read_bytes())
    boundaries = [r["boundary"] for r in receipt["source_to_audio"] if r["kind"] == "speech" and r["boundary"]]
    crossing = [b for b in boundaries if b["source_separators"]]
    assert len(crossing) == 1 and crossing[0]["target_ms"] >= batch.SEPARATOR_TARGET_MS
    assert all(b["trimming"] is False for b in boundaries)
    assert any(r["kind"] == "scene_break" and r["pause_target_ms"] == 1250 for r in receipt["source_to_audio"])


def test_completed_retry_verifies_bytes_and_starts_no_extra_worker(ready):
    world, args, state = ready
    batch.generate_batch(args, state, FakeWorker, render_function())
    before = tree_hashes(args.output_dir)
    FakeWorker.instances = []
    again = batch.generate_batch(args, batch.load_state(args.output_dir), FakeWorker, render_function())
    assert again["status"] == "generated" and FakeWorker.instances == [] and tree_hashes(args.output_dir) == before


def test_partial_run_resumes_only_the_missing_dialogue_with_one_worker(ready):
    world, args, state = ready
    render = render_function()
    calls = {"n": 0}

    def flaky(worker, *a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("worker dropped")
        return render(worker, *a, **k)
    with pytest.raises(RuntimeError, match="worker dropped"):
        batch.generate_batch(args, state, FakeWorker, flaky)
    assert FakeWorker.instances[0].exited                          # always released, never leaked
    assert not (args.output_dir / "chapters").exists()
    FakeWorker.instances = []
    result = batch.generate_batch(args, batch.load_state(args.output_dir), FakeWorker, render)
    assert len(all_calls()) == 3
    assert len(FakeWorker.instances) == 1 and result["audit"]["new_requests_this_run"] == 3


def test_ambiguous_pending_dispatch_holds_before_any_worker_exists(ready):
    world, args, state = ready
    line = next(l for l in state["lines"] if not l["narrator"])
    sample = args.output_dir / "samples" / line["project_id"] / f"{line['utterance_id']}.wav"
    request_sha = pilot.expected_request_sha(line["project_id"], line["voice"], line["text"], None)
    pilot.publish(sample.with_suffix(f".pending-{request_sha[:16]}.json"), b"{}")
    with pytest.raises(batch.BatchError, match="ambiguous pending dispatch"):
        batch.generate_batch(args, state, FakeWorker, render_function())
    assert FakeWorker.instances == [] and all_calls() == []         # no resend, no GPU lane touched
    assert not (args.output_dir / "chapters").exists()


def test_unconfirmed_forwarding_is_refused(ready):
    world, args, state = ready
    FakeWorker.forward = False
    with pytest.raises(batch.BatchError, match="did not confirm"):
        batch.generate_batch(args, state, FakeWorker, render_function())
    assert not (args.output_dir / "chapters").exists()


def test_wrong_runtime_profile_is_refused_before_dispatch(ready):
    world, args, state = ready
    FakeWorker.profile = "faster-base-032"
    with pytest.raises(batch.BatchError, match="pinned"):
        batch.generate_batch(args, state, FakeWorker, render_function())
    assert all_calls() == []


def test_runtime_pin_blocks_a_different_worker_in_a_resumed_run(ready):
    world, args, state = ready
    pilot.publish(args.output_dir / "runtime-pin.json", pilot.encoded(
        {"runtime_sha256": "9" * 64, "runtime_profile": pilot.DIRECTED_PROFILE,
         "checkpoint_manifest_sha256": "c" * 64, "model_id": "m"}))
    with pytest.raises(batch.BatchError, match="differs from the pinned run"):
        batch.generate_batch(args, state, FakeWorker, render_function())
    assert all_calls() == []


def test_worker_fallback_is_a_refusal_not_a_substituted_preset(ready):
    world, args, state = ready
    render = render_function()

    def with_fallback(worker, *a, **k):
        receipt = render(worker, *a, **k)
        receipt["worker_receipt"]["fallback"] = "generic-preset"
        return receipt
    with pytest.raises(batch.BatchError, match="fallback"):
        batch.generate_batch(args, state, FakeWorker, with_fallback)
    assert not (args.output_dir / "chapters").exists()


def test_different_voice_in_worker_receipt_is_refused(ready):
    world, args, state = ready
    render = render_function()

    def wrong_voice(worker, *a, **k):
        receipt = render(worker, *a, **k)
        receipt["worker_receipt"]["identity_id"] = "another_voice"
        return receipt
    with pytest.raises(batch.BatchError, match="different voice identity"):
        batch.generate_batch(args, state, FakeWorker, wrong_voice)


@pytest.mark.parametrize("drift", ["role_name", "segments", "neutral_take", "neutral_copy", "worker_clone_copy",
                                   "coached_plan", "bundle"])
def test_drift_after_preparation_is_refused_before_a_worker_is_started(ready, drift):
    world, args, state = ready
    out, pid = args.output_dir, world.ids[0]
    narrator = next(l for l in state["lines"] if l["narrator"] and l["project_id"] == pid)
    if drift == "role_name":
        path = world.root / pid / "project.json"
        raw = json.loads(path.read_bytes())
        raw["roles"][1]["name"] = "Renamed"
        path.write_bytes(json.dumps(raw).encode())
    elif drift == "segments":
        path = world.root / pid / "segments.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif drift == "neutral_take":
        (world.root / pid / narrator["neutral_take"]["source_audio_file"]).write_bytes(b"replaced take")
    elif drift == "neutral_copy":
        (out / narrator["neutral_copy"]["file"]).write_bytes(b"replaced copy")
    elif drift == "worker_clone_copy":
        (out / "projects" / pid / "casting" / "voices" / "voice_erick" / "reference.wav").write_bytes(b"changed")
    elif drift == "coached_plan":
        path = out / "plans" / f"coached-{pid}.json"
        plan = json.loads(path.read_bytes())
        next(d for d in plan["directions"] if not d["narrator"])["native_instruction"] = "Whisper conspiratorially."
        path.write_bytes(pilot.encoded(plan))
    elif drift == "bundle":
        path = out / "coaching" / f"bundle-{pid}.json"
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(batch.BatchError):
        batch.generate_batch(args, state, FakeWorker, render_function())
    assert FakeWorker.instances == [] and not (out / "samples").exists()


def test_tampered_or_substituted_state_is_refused(ready):
    world, args, state = ready
    forged = copy.deepcopy(state)
    next(l for l in forged["lines"] if not l["narrator"])["text"] += " tampered"
    with pytest.raises(batch.BatchError, match="differs from the frozen"):
        batch.generate_batch(args, forged, FakeWorker, render_function())
    path = args.output_dir / "state.json"
    disk = json.loads(path.read_bytes())
    disk["lines"][0]["text"] += " tampered"
    path.write_bytes(pilot.encoded(disk))
    with pytest.raises(batch.BatchError, match="changed since preparation"):
        batch.load_state(args.output_dir)
    assert FakeWorker.instances == []


def test_prepare_holds_block_everything(world):
    real = batch._build_plan

    def with_safety_hold(*a, **k):
        plan = real(*a, **k)
        plan["holds"].append({"code": "voice_unbound", "utterance_id": "x", "reason": "synthetic hold"})
        plan["plan_sha256"] = _sha({k: v for k, v in plan.items() if k != "plan_sha256"})
        return plan
    args = args_for(world)
    state = batch.prepare_batch(args, plan_builder=with_safety_hold)
    assert state["status"] == "held" and state["requests"] == {} and state["prepare_holds"]
    with pytest.raises(batch.BatchError, match="Preparation held"):
        batch.apply_bundles(args, state)
    with pytest.raises(batch.BatchError, match="Preparation held"):
        batch.generate_batch(args, state, FakeWorker, render_function())


# --------------------------------------------------------------------------- CLI


def argv_for(world, *extra):
    return ["--epub", str(world.epub), "--source-root", str(world.root), "--projects", *world.ids,
            "--chapters", "4", "5", "--output-dir", str(world.tmp / "cli-out"), *extra]


def test_cli_argument_parsing_is_ordered_and_robust(world):
    args = batch.parse_args(argv_for(world))
    assert args.projects == world.ids and args.chapters == [4, 5] and args.coaching_dir is None and not args.generate
    comma = ["--epub", str(world.epub), "--source-root", str(world.root), "--projects", ",".join(world.ids),
             "--chapters", "4,5", "--output-dir", str(world.tmp / "o"), "--coaching-dir", str(world.tmp / "c"), "--generate"]
    args = batch.parse_args(comma)
    assert args.projects == world.ids and args.chapters == [4, 5] and args.generate and args.coaching_dir == world.tmp / "c"
    for bad in (["--chapters", "5", "4"], ["--chapters", "4"], ["--chapters", "four", "5"]):
        argv = argv_for(world)
        index = argv.index("--chapters")
        argv[index:index + 3] = bad
        with pytest.raises(SystemExit) as caught:
            batch.parse_args(argv)
        assert caught.value.code == 2
    with pytest.raises(SystemExit):
        batch.parse_args(["--epub", str(world.tmp / "missing.epub")] + argv_for(world)[2:])
    with pytest.raises(SystemExit):
        batch.parse_args(argv_for(world)[:-2])                    # no output directory


def test_cli_prepare_apply_generate_flow_and_exit_codes(world, capsys, monkeypatch):
    assert batch.main(argv_for(world)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "awaiting_coaching" and summary["production_ready"] is False
    assert summary["dialogue_lines"] == 5
    assert batch.main(argv_for(world, "--generate")) == 2        # nothing coached: generation is refused
    assert json.loads(capsys.readouterr().out)["status"] == "awaiting_coaching" and FakeWorker.instances == []
    state = batch.load_state(world.tmp / "cli-out")
    folder = write_bundles(world, state, world.tmp / "cli-out")
    monkeypatch.setattr(pilot, "default_render_tools", lambda: (FakeWorker, RENDER))
    assert batch.main(argv_for(world, "--coaching-dir", str(folder), "--generate")) == 0
    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])   # render() prints progress lines first
    assert final["status"] == "generated" and final["generation"]["audit"]["directed_forwarded"] == 5
    assert len(FakeWorker.instances) == 1
    assert batch.main(argv_for(world, "--coaching-dir", str(folder), "--generate")) == 0   # idempotent retry
    assert len(FakeWorker.instances) == 1


def test_cli_script_is_a_thin_entry_point():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_coached_chapters.py"
    spec = importlib.util.spec_from_file_location("render_coached_chapters_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main is batch.main


def test_fixture_labels_reach_every_published_receipt(ready):
    world, args, state = ready
    batch.generate_batch(args, state, FakeWorker, render_function())
    out = args.output_dir
    labeled = [out / "forwarding-audit.json", out / "batch-result.json", *out.glob("applied/*.json"),
               *out.glob("receipts/*/*.json"), *out.glob("chapters/*/chapter-*.json")]
    assert len(labeled) >= 2 + 2 + 5 + 2
    for path in labeled:
        assert json.loads(path.read_bytes())["fixture"]["label"] == batch.FIXTURE_LABEL, path


def test_governed_exclusions_stay_in_the_source_map_without_audio(world, monkeypatch):
    """A source-backed omission keeps its prose in the coverage check but receives no audio or direction."""
    heading = f"{world.ids[1]}-00"
    real = exporter._source_snapshot

    def with_exclusion(project_id, chapter_id):
        project, segments, source = real(project_id, chapter_id)
        if project_id == world.ids[1]:
            source = {**source, "speech_utterance_ids": [s for s in source["speech_utterance_ids"] if s != heading],
                      "exclusions": [{"utterance_id": heading, "decision": {"reason": "fixture"},
                                      "source_text_sha256": text_sha256("005")}]}
        return project, segments, source
    monkeypatch.setattr(exporter, "_source_snapshot", with_exclusion)
    args = args_for(world)
    state = batch.prepare_batch(args)
    assert heading not in {l["utterance_id"] for l in state["lines"]}
    chapter = next(c for c in state["chapters"] if c["project_id"] == world.ids[1])
    assert {r["utterance_id"]: r["kind"] for r in chapter["rows"]}[heading] == "excluded"
    write_bundles(world, state, args.output_dir)
    state = batch.apply_bundles(args, state)
    batch.generate_batch(args, state, FakeWorker, render_function())
    receipt = json.loads((args.output_dir / "chapters" / world.ids[1] / "chapter-005.json").read_bytes())
    row = next(r for r in receipt["source_to_audio"] if r["utterance_id"] == heading)
    assert row["kind"] == "excluded" and row["audio_file"] is None
    assert receipt["counts"]["excluded"] == 1 and receipt["counts"]["speech"] == 5


def test_generation_tolerates_only_delivery_notification_counters(ready):
    world, args, state = ready
    path = world.root / world.ids[0] / "project.json"
    raw = json.loads(path.read_bytes())
    raw["voice_lab"]["revision"] += 3
    raw["voice_lab"].setdefault("casting", {})["notifications"] = {"complete": {"next_retry_at": "later"}}
    path.write_bytes(json.dumps(raw).encode())
    assert batch.generate_batch(args, state, FakeWorker, render_function())["status"] == "generated"


def test_retry_refuses_changed_sample_or_chapter_bytes(ready):
    world, args, state = ready
    batch.generate_batch(args, state, FakeWorker, render_function())
    out = args.output_dir
    sample = next((out / "samples").rglob("*.wav"))
    original = sample.read_bytes()
    sample.write_bytes(original + b"\x00\x00")
    with pytest.raises(RuntimeError, match="receipt mismatch"):
        batch.generate_batch(args, batch.load_state(out), FakeWorker, render_function())
    sample.write_bytes(original)
    master = next((out / "chapters").rglob("chapter-*.wav"))
    good = master.read_bytes()
    master.write_bytes(good + b"\x00\x00")
    with pytest.raises(batch.BatchError, match="changed"):
        batch.generate_batch(args, batch.load_state(out), FakeWorker, render_function())
    master.write_bytes(good)
    assert batch.generate_batch(args, batch.load_state(out), FakeWorker, render_function())["status"] == "generated"


# --------------------------------------------------------------------------- optional performance memory
# Labeled synthetic fixtures only: the "memory" is a hand-written stand-in for a model response (never real
# model output) and nothing below is evidence about audio quality, calibration or approval.

from dubbing_service.app.audiobook import character_performance_memory as cpm       # noqa: E402
from tests.test_character_performance_memory import FIXTURE_PROVENANCE as MEMORY_PROVENANCE      # noqa: E402
from tests.test_character_performance_memory import chapter_request as memory_chapter_request      # noqa: E402
from tests.test_character_performance_memory import make_response as memory_response      # noqa: E402

GOLDEN_FROZEN_KEYS = ("epub", "sources", "chapters", "lines", "requests", "base_plans", "isolated",
                      "boundary_policy_sha256", "prepare_holds")


def plain_requests(state, out):
    return [json.loads((Path(out) / info["file"]).read_bytes()) for info in state["requests"].values()]


def write_memory(world, state, out, name="memory.json", *, mutate=None, book=None, provenance=None, requests=None):
    request = cpm.build_memory_request(requests or plain_requests(state, out),
                                       book_sha256=book or state["epub"]["sha256"])
    bundle = cpm.capture_bundle(request, memory_response(request, mutate=mutate), provenance or MEMORY_PROVENANCE)
    path = world.tmp / name
    path.write_bytes(pilot.encoded(bundle))
    return path, bundle


def memory_args(world, path, tag="mem", **overrides):
    return args_for(world, output_dir=world.tmp / f"out-{tag}", coaching_dir=world.tmp / f"coaching-{tag}",
                    performance_memory=path, **overrides)


def write_memory_bundles(args, state, *, only=None):
    args.coaching_dir.mkdir(exist_ok=True)
    for pid, info in state["requests"].items():
        if only is None or pid in only:
            request = json.loads((args.output_dir / info["file"]).read_bytes())
            (args.coaching_dir / f"coaching-{pid}.json").write_bytes(pilot.encoded(make_bundle(request)))


@pytest.fixture
def bound(prepared):
    """A plain batch, a memory captured from its requests, and a second batch prepared bound to that memory."""
    world, plain_args, plain_state = prepared
    path, bundle = write_memory(world, plain_state, plain_args.output_dir)
    args = memory_args(world, path)
    state = batch.prepare_batch(args)
    return SimpleNamespace(world=world, plain_args=plain_args, plain=plain_state, path=path, bundle=bundle,
                           args=args, state=state)


def replace_memory(args, path):
    return SimpleNamespace(**{**vars(args), "performance_memory": path})


def test_existing_v2_behaviour_is_byte_identical_without_the_option(world):
    assert batch.FROZEN_KEYS == GOLDEN_FROZEN_KEYS                      # no new frozen key invalidates old states
    absent = batch.prepare_batch(args_for(world, output_dir=world.tmp / "no-attr"))
    explicit = batch.prepare_batch(args_for(world, output_dir=world.tmp / "none", performance_memory=None))
    assert "performance_memory" not in absent and "performance_memory" not in explicit
    assert absent["frozen_sha256"] == explicit["frozen_sha256"] and absent["requests"] == explicit["requests"]
    for pid, info in absent["requests"].items():
        assert "performance_memory_sha256" not in info
        request = json.loads((world.tmp / "no-attr" / info["file"]).read_bytes())
        assert set(request) == {"instruction", "constraints", "response_contract", "scenes"}
        assert all(cpm.SCENE_KEY not in scene for scene in request["scenes"])
        assert request["constraints"] == batch.REQUEST_CONSTRAINTS and cpm.enriched_memory_sha256(request) is None
    assert not (world.tmp / "no-attr" / "memory").exists()


def test_a_memory_less_batch_is_not_changed_by_the_new_code_on_resume_apply_and_generate(ready):
    world, args, state = ready
    assert batch.prepare_batch(args)["status"] == "ready"
    assert batch.apply_bundles(args, batch.load_state(args.output_dir))["status"] == "ready"
    result = batch.generate_batch(args, batch.load_state(args.output_dir), FakeWorker, render_function())
    assert result["status"] == "generated"
    receipts = sorted((args.output_dir / "chapters").rglob("chapter-*.json"))
    assert receipts and all("performance_memory" not in json.loads(p.read_bytes()) for p in receipts)
    assert all("performance_memory_sha256" not in json.loads(p.read_bytes())
               for p in (args.output_dir / "applied").glob("*.json"))
    assert not (args.output_dir / "memory").exists()


def test_prepare_with_memory_keeps_the_v2_request_and_adds_only_bound_notes(bound):
    out, plain_out = bound.args.output_dir, bound.plain_args.output_dir
    binding = bound.state["performance_memory"]
    assert binding["memory_sha256"] == bound.bundle["memory_sha256"] and binding["approved"] is False
    assert binding["interpretation"] == "unapproved_model_interpretation" and binding["fixture"] is True
    assert binding["profile_roles"] == ["erick", "jane"] and binding["narrator_roles"] == ["narrator"]
    assert (out / binding["file"]).read_bytes() == bound.path.read_bytes()        # exact private capture
    assert bound.state["status"] == "awaiting_coaching" and bound.state["production_ready"] is False
    assert bound.state["approvals_created"] is False and bound.state["takes_selected"] is False
    assert batch.load_state(out) == bound.state
    assert bound.state["lines"] == bound.plain["lines"] and bound.state["base_plans"] == bound.plain["base_plans"]
    for pid, info in bound.state["requests"].items():
        assert info["performance_memory_sha256"] == bound.bundle["memory_sha256"]
        enriched = json.loads((out / info["file"]).read_bytes())
        plain = json.loads((plain_out / bound.plain["requests"][pid]["file"]).read_bytes())
        assert cpm.enriched_memory_sha256(enriched) == bound.bundle["memory_sha256"]
        stripped = copy.deepcopy(enriched)
        stripped["constraints"].pop()
        for scene in stripped["scenes"]:
            scene.pop(cpm.SCENE_KEY)
        assert stripped == plain                                       # every existing request field is preserved
        assert info["request_sha256"] != bound.plain["requests"][pid]["request_sha256"]
        for scene in enriched["scenes"]:
            roles = {r["role_id"] for r in scene["context"] if not r["narrator"]}
            assert {n["role_id"] for n in scene[cpm.SCENE_KEY]["notes"]} == roles and "narrator" not in roles
        scene_coaching.validate_bundle(make_bundle(enriched))          # still a valid existing coaching request


def test_a_bound_batch_applies_and_generates_and_records_the_memory_everywhere(bound):
    args, state, out = bound.args, bound.state, bound.args.output_dir
    write_memory_bundles(args, state)
    state = batch.apply_bundles(args, state)
    assert state["status"] == "ready", state["holds"]
    for pid in state["requests"]:
        applied = json.loads((out / "applied" / f"{pid}.json").read_bytes())
        assert applied["performance_memory_sha256"] == bound.bundle["memory_sha256"] and applied["approved"] is False
    result = batch.generate_batch(args, state, FakeWorker, render_function())
    assert result["status"] == "generated" and len(FakeWorker.instances) == 1
    assert len(all_calls()) == 5 and all(c["performance"]["native_instruction"] for c in all_calls())
    for path in sorted((out / "chapters").rglob("chapter-*.json")):
        receipt = json.loads(path.read_bytes())
        assert receipt["performance_memory"] == {"memory_sha256": bound.bundle["memory_sha256"], "approved": False,
                                                 "interpretation": "unapproved_model_interpretation",
                                                 "model_requested": MEMORY_PROVENANCE["model_requested"]}
        assert receipt["quality_improvement"] == "not_established" and receipt["fixture"]["is_fixture"] is True
    assert batch.generate_batch(args, batch.load_state(out), FakeWorker, render_function())["status"] == "generated"
    assert len(FakeWorker.instances) == 1                                # the completed retry starts no new worker


def test_role_coverage_name_and_narrator_conflicts_hold_before_any_chapter_output(prepared):
    world, plain_args, plain_state = prepared
    only_erick = [memory_chapter_request("x", [("00", "narrator", "Narrator", True, "Fixture narration."),
                                               ("01", "erick", "Erick", False, "“Open it,”")])]
    renamed = [memory_chapter_request("x", [("00", "narrator", "Narrator", True, "Fixture narration."),
                                            ("01", "erick", "Eric", False, "“Open it,”"),
                                            ("02", "jane", "Jane", False, "“I will not,”")])]
    for tag, requests, message in (("missing", only_erick, "no profile for role jane"),
                                   ("renamed", renamed, "is named 'Erick'")):
        path, _ = write_memory(world, plain_state, plain_args.output_dir, f"{tag}.json", requests=requests)
        args = memory_args(world, path, tag)
        with pytest.raises(batch.BatchError, match=message):
            batch.prepare_batch(args)
        out = args.output_dir
        assert not (out / "state.json").exists() and not (out / "requests").exists()
        assert not (out / "neutral").exists() and not (out / "plans").exists() and not (out / "projects").exists()


def test_wrong_book_unreadable_or_tampered_memory_is_refused_before_the_output_exists(prepared):
    world, plain_args, plain_state = prepared
    wrong_book, _ = write_memory(world, plain_state, plain_args.output_dir, "book.json", book="d" * 64)
    tampered, bundle = write_memory(world, plain_state, plain_args.output_dir, "tampered.json")
    bundle["response"]["profiles"][0]["confidence"] = "high"
    tampered.write_bytes(pilot.encoded(bundle))
    garbage = world.tmp / "garbage.json"
    garbage.write_bytes(b"not json")
    for tag, path, message in (("book", wrong_book, "different book"), ("tamper", tampered, "hash mismatch"),
                               ("garbage", garbage, "not valid JSON")):
        args = memory_args(world, path, tag)
        with pytest.raises(batch.BatchError, match=message):
            batch.prepare_batch(args)
        assert not args.output_dir.exists()
    with pytest.raises(SystemExit):
        batch.parse_args(argv_for(world, "--performance-memory", str(world.tmp / "absent.json")))


def test_a_fixture_memory_cannot_condition_non_fixture_chapters(prepared):
    world, plain_args, plain_state = prepared
    path, _ = write_memory(world, plain_state, plain_args.output_dir)
    with batch._projects_root(world.root):
        for pid in world.ids:
            project = load_project(pid)
            project.metadata["coached_batch_fixture"] = False
            save_project(project)
    args = memory_args(world, path, "nonfixture")
    with pytest.raises(batch.BatchError, match="fixture performance memory"):
        batch.prepare_batch(args)
    assert not (args.output_dir / "state.json").exists() and not (args.output_dir / "requests").exists()


def test_identical_memory_resumes_without_changing_a_byte(bound):
    before = tree_hashes(bound.args.output_dir)
    copy_of_memory = bound.world.tmp / "same-bytes-elsewhere.json"
    copy_of_memory.write_bytes(bound.path.read_bytes())
    again = batch.prepare_batch(memory_args(bound.world, copy_of_memory))
    assert again == bound.state and tree_hashes(bound.args.output_dir) == before


def test_resume_holds_when_the_bound_memory_is_missing_or_changed(bound):
    world = bound.world
    before = tree_hashes(bound.args.output_dir)
    with pytest.raises(batch.BatchError, match="bound to performance memory"):
        batch.prepare_batch(args_for(world, output_dir=bound.args.output_dir, coaching_dir=bound.args.coaching_dir))
    with pytest.raises(batch.BatchError, match="not found"):
        batch.prepare_batch(memory_args(world, world.tmp / "deleted.json"))
    other, _ = write_memory(world, bound.plain, bound.plain_args.output_dir, "other.json",
                            mutate=lambda r: r["profiles"][0].update(confidence="high"))
    with pytest.raises(batch.BatchError, match="differs from the memory bound"):
        batch.prepare_batch(memory_args(world, other))
    changed_model, _ = write_memory(world, bound.plain, bound.plain_args.output_dir, "model.json",
                                    provenance={**MEMORY_PROVENANCE, "model_requested": "another-fixture"})
    with pytest.raises(batch.BatchError, match="differs from the memory bound"):
        batch.prepare_batch(memory_args(world, changed_model))
    assert tree_hashes(bound.args.output_dir) == before                  # nothing was reused or rewritten


def test_apply_and_generate_hold_without_or_with_a_different_memory(bound):
    world, args, state = bound.world, bound.args, bound.state
    write_memory_bundles(args, state)
    no_memory = args_for(world, output_dir=args.output_dir, coaching_dir=args.coaching_dir)
    other, _ = write_memory(world, bound.plain, bound.plain_args.output_dir, "other.json",
                            mutate=lambda r: r["profiles"][1].update(confidence="low"))
    different = memory_args(world, other)
    for held_args in (no_memory, different):
        with pytest.raises(batch.BatchError, match="performance memory"):
            batch.apply_bundles(held_args, state)
    assert not (args.output_dir / "applied").exists()
    ready_state = batch.apply_bundles(args, state)
    assert ready_state["status"] == "ready"
    for held_args in (no_memory, different):
        with pytest.raises(batch.BatchError, match="performance memory"):
            batch.generate_batch(held_args, ready_state, FakeWorker, render_function())
    assert FakeWorker.instances == []


def test_a_memory_less_directory_never_silently_gains_a_memory(prepared):
    world, args, state = prepared
    path, _ = write_memory(world, state, args.output_dir)
    before = tree_hashes(args.output_dir)
    for call in (lambda: batch.prepare_batch(replace_memory(args, path)),
                 lambda: batch.apply_bundles(replace_memory(args, path), state)):
        with pytest.raises(batch.BatchError, match="prepared without performance memory"):
            call()
    assert tree_hashes(args.output_dir) == before


def test_changed_memory_invalidates_coaching_captured_for_the_earlier_memory(bound):
    world, args, state = bound.world, bound.args, bound.state
    write_memory_bundles(args, state)                                    # captured for the first memory's requests
    other, other_bundle = write_memory(world, bound.plain, bound.plain_args.output_dir, "other.json",
                                       mutate=lambda r: r["profiles"][0].update(confidence="high"))
    second = memory_args(world, other, "second")
    second_state = batch.prepare_batch(second)
    assert second_state["performance_memory"]["memory_sha256"] == other_bundle["memory_sha256"] != bound.bundle["memory_sha256"]
    for pid, info in second_state["requests"].items():
        assert info["request_sha256"] != state["requests"][pid]["request_sha256"]
    second.coaching_dir.mkdir()
    for path in args.coaching_dir.glob("coaching-*.json"):               # reuse the stale bundles under the new memory
        (second.coaching_dir / path.name).write_bytes(path.read_bytes())
    held = batch.apply_bundles(second, second_state)
    assert held["status"] == "held" and not (second.output_dir / "applied").exists()
    assert all("not captured for the frozen request" in h["detail"] for h in held["holds"]) and len(held["holds"]) == 2
    assert not list((second.output_dir / "plans").glob("coached-*"))
    write_memory_bundles(second, second_state)                           # captured for the new requests: accepted
    assert batch.apply_bundles(second, second_state)["status"] == "ready"


def test_a_plain_request_bundle_cannot_satisfy_a_memory_bound_batch(bound):
    args, state = bound.args, bound.state
    args.coaching_dir.mkdir()
    for pid, info in bound.plain["requests"].items():
        request = json.loads((bound.plain_args.output_dir / info["file"]).read_bytes())
        (args.coaching_dir / f"coaching-{pid}.json").write_bytes(pilot.encoded(make_bundle(request)))
    held = batch.apply_bundles(args, state)
    assert held["status"] == "held" and len(held["holds"]) == 2
    assert not (args.output_dir / "applied").exists()


def test_private_memory_copy_or_binding_tampering_is_detected_on_every_load(bound):
    out = bound.args.output_dir
    binding = bound.state["performance_memory"]
    memory_file = out / binding["file"]
    original = memory_file.read_bytes()
    memory_file.write_bytes(original + b"\n")
    with pytest.raises(batch.BatchError, match="missing or changed"):
        batch.load_state(out)
    memory_file.write_bytes(original)
    batch.load_state(out)
    state_path = out / "state.json"
    raw = state_path.read_bytes()
    unbound = json.loads(raw)
    unbound.pop("performance_memory")
    state_path.write_bytes(pilot.encoded(unbound))
    with pytest.raises(batch.BatchError, match="no memory binding"):
        batch.load_state(out)
    state_path.write_bytes(raw)
    request_path = out / next(iter(bound.state["requests"].values()))["file"]
    request_raw = request_path.read_bytes()
    stripped = json.loads(request_raw)
    stripped["constraints"].pop()
    request_path.write_bytes(pilot.encoded(stripped))
    with pytest.raises(batch.BatchError):                                # the published request changed
        batch.load_state(out)
    request_path.write_bytes(request_raw)
    memory_file.unlink()
    with pytest.raises(batch.BatchError, match="missing or changed"):
        batch.load_state(out)


def test_memory_never_reaches_narration_or_changes_synthesis_inputs(bound):
    args, state = bound.args, bound.state
    write_memory_bundles(args, state)
    state = batch.apply_bundles(args, state)
    batch.generate_batch(args, state, FakeWorker, render_function())
    narration = {l["text"] for l in state["lines"] if l["narrator"]}
    memory_words = json.dumps(bound.bundle["response"]["profiles"])
    for call in all_calls():
        assert call["text"] not in narration
        assert memory_words not in json.dumps(call)                     # notes inform coaching only, never a worker call
    for line in (l for l in state["lines"] if l["narrator"]):
        assert not (args.output_dir / "samples" / line["project_id"] / f"{line['utterance_id']}.wav").exists()
    assert json.loads((args.output_dir / "forwarding-audit.json").read_bytes())["narrator_instructions"] == 0


def test_cli_accepts_the_option_reports_the_hash_and_holds_on_a_changed_memory(bound, capsys):
    world = bound.world
    argv = argv_for(world, "--performance-memory", str(bound.path))
    assert batch.parse_args(argv).performance_memory == bound.path
    assert batch.parse_args(argv_for(world)).performance_memory is None
    assert batch.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["performance_memory_sha256"] == bound.bundle["memory_sha256"] and summary["status"] == "awaiting_coaching"
    other, _ = write_memory(world, bound.plain, bound.plain_args.output_dir, "cli-other.json",
                            mutate=lambda r: r["profiles"][0].update(confidence="high"))
    assert batch.main(argv_for(world, "--performance-memory", str(other))) == 2
    assert "differs from the memory bound" in json.loads(capsys.readouterr().out)["error"]
    assert batch.main(argv_for(world)) == 2
    assert "bound to performance memory" in json.loads(capsys.readouterr().out)["error"]
