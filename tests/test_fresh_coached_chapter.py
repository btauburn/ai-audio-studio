"""CPU tests for the fresh (no selected Takes) scene-coached review chapter pipeline.

Everything here is a labeled synthetic fixture: invented prose, tone-signal audio, fake clone files, fake approval
and acquisition records, a synthetic coaching bundle standing in for a model response, a fake performance-memory
owner and a fake worker. No real book, voice, model, GPU, network or live project is read or generated, and nothing
here is evidence about production readiness or audio quality.
"""
from __future__ import annotations

import copy
import hashlib
import shutil
import json
import math
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubbing_service.app.audiobook import fresh_coached_chapter as fresh
from dubbing_service.app.audiobook import qa, scene_coaching
from dubbing_service.app.audiobook.fingerprint import text_sha256, voice_revision_for_role
from dubbing_service.app.audiobook.scene_dialogue_plan import _sha
from dubbing_service.app.models.audiobook import Attribution, SourceAnchor, SourceRevision
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.models.project import Role, Segment
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import build_project, use_projects_root
from tests.test_epub_chapter_selection import NS, epub, xhtml

RATE = 24000
NUMBER = 11
pilot = fresh.pilot()
RENDER = pilot.default_render_tools()[1]       # the real render() owner; the real Worker is never constructed


def sha(data):
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- synthetic world

# (kind, text). "n" narrator, "e" Erick, "j" Jane, "b" scene break.
CHAPTER_11 = [("n", "011"), ("n", "The gate was shut and the road behind them was dark."),
              ("e", "“Open it,”"), ("n", "said Erick."), ("j", "“I will not,”"), ("n", "said Jane."),
              ("b", "***"), ("n", "Morning came slowly over the wall."), ("e", "“Now?”"), ("n", "asked Erick.")]
SPEECH = 9
NARRATION_LINES = 6
DIALOGUE_LINES = 3


def write_tone(path, seconds, frequency, *, lead=0.0, tail=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(RATE * seconds)
    samples = [int(0.4 * 32767 * math.sin(2 * math.pi * frequency * i / RATE)) for i in range(frames)]
    samples = [0] * int(RATE * lead) + samples + [0] * int(RATE * tail)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


def wave_seconds(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def binding_sha(entries):
    return pilot.value_sha({rid: {"provider_project_id": e["project_id"], "voice_id": e["voice_id"],
                                  "reference_audio_sha256": e["reference_audio_sha256"],
                                  "prompt_sha256": e["prompt_sha256"]} for rid, e in entries.items()})


def authorization(book_sha, entries):
    return {"schema": "fresh-chapter-owner-review-v1", "authorized": True, "scope": "private_chapter_review",
            "book_sha256": book_sha, "chapter_number": NUMBER, "role_voice_bindings_sha256": binding_sha(entries),
            "author": "user_request", "instruction": "Synthetic fixture text standing in for the owner's exact request.",
            "standing_constraints": "Existing cloned cast only; private listening review; no default switch",
            "quality_approval_created": False}


def eligible_receipt(name, acquisition, reference_sha, *, restrictions=None):
    """A fixture acquisition receipt that passes the real recorded-source eligibility rules (invented source)."""
    source = {"source_id": f"fixture:{name}", "attribution": "Fixture attribution", "license": "CC BY 3.0",
              "license_url": "https://creativecommons.org/licenses/by/3.0",
              "media_url": f"https://upload.wikimedia.org/wikipedia/commons/f/fx/{name}.flac",
              "source_page": f"https://commons.wikimedia.org/wiki/File:{name}.flac",
              "terms_url": "https://creativecommons.org/licenses/by/3.0/", "language": "en", "duration_seconds": 30.0}
    if restrictions:
        source["restrictions"] = restrictions
    evidence = {"status": "approved", "use": "private_synthetic_voice", "authority": "voice_license",
                "evidence_ref": f"fixture-evidence-{name}", "source_id": source["source_id"]}
    return json.dumps({"schema": "voice-lab-acquisition-v1", "acquisition_id": acquisition, "clone_ready": True,
                       "technical_qa": {"status": "pass"}, "reference_audio_sha256": reference_sha, "source": source,
                       "eligibility": {"intended_use_evidence": evidence}}, sort_keys=True).encode()


def make_identity(base, name, seed, *, restrictions=None):
    voice_id = f"voice_{name}"
    folder = base / "casting" / "voices" / voice_id
    folder.mkdir(parents=True)
    reference = bytes((seed * 7 + i * 13) % 251 for i in range(300))
    prompt = bytes((seed * 11 + i * 17) % 241 for i in range(400))
    (folder / "reference.wav").write_bytes(reference)
    (folder / "conditioning.npz").write_bytes(prompt)
    acquisition = (name.encode().hex() * 4)[:32]
    receipt = eligible_receipt(name, acquisition, sha(reference), restrictions=restrictions)
    path = base / "voice_lab" / "discovery" / "acquisitions" / acquisition / "acquisition.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(receipt)
    return CharacterVoiceIdentity(
        id=voice_id, role_id=name, status="LOCKED", mechanism="qwen_reference", backend="qwen3tts",
        dossier_revision="fixture", casting_profile_revision="fixture", design_instruction="fixture only",
        model="mock-no-model", reference_audio=f"casting/voices/{voice_id}/reference.wav",
        reference_audio_sha256=sha(reference), prompt_artifact=f"casting/voices/{voice_id}/conditioning.npz",
        prompt_sha256=sha(prompt),
        provenance={"origin": "eligible_reference_clone", "actor_audio_used": True,
                    "acquisition_id": acquisition, "acquisition_sha256": sha(receipt)})


def approve(project, base, role):
    """Fixture stand-in for a human casting approval: assignment + positive human event + heard sample record."""
    identity = role.voice_identity
    heard = write_tone(base / "voice_lab" / "heard" / f"{identity.id}.wav", 0.5, 310)
    version = voice_revision_for_role(role)
    key, event_id = f"v_{role.id}", f"event_{role.id}"
    project.voice_lab.voices[key] = {
        "key": key, "voice_id": identity.id, "version": version, "identity": identity.model_dump(mode="json"),
        "audio_file": heard.relative_to(base).as_posix(), "audio_sha256": sha(heard.read_bytes())}
    project.voice_lab.casting["roles"][role.id] = {"assignment": {
        "key": key, "approved": True, "voice_revision": version, "event_id": event_id}}
    project.voice_lab.casting["events"][event_id] = {
        "origin": "human", "actor": "brian", "action": "use", "role_id": role.id, "key": key, "voice_id": identity.id,
        "voice_version": version, "audio_sha256": sha(heard.read_bytes()), "selection_confirmed": True}


def epub_for(rows):
    docs = [("Cover.xhtml", "cover", f'<html {NS}><body><img src="cover.jpg" alt=""/></body></html>', None)]
    body = "".join(f"<p>{text}</p>" for _, text in rows[1:])
    docs.append((f"{NUMBER:04d}_{NUMBER:03d}.xhtml", f"c{NUMBER}", xhtml(f"{NUMBER:03d}", body), f"{NUMBER:03d}"))
    return epub(docs)


ROLE_SPECS = (("narrator", "narrator", "narrator"), ("erick", "Erick", "character"), ("jane", "Jane", "character"))


def build_provider(root, origin_root, epub_bytes, *, roles=("narrator", "erick", "jane"), restrictions_for=None):
    """A fixture 'original book-role context': recorded clones, a draft clone with its own positive human selection
    (erick), eligible acquisition receipts, no per-role human 'use' event for the non-draft roles. An exact copy is
    the origin store."""
    project = build_project(name="fresh provider synthetic fixture", chapter_titles=["001"])
    base = root / project.id
    project.source_revisions = [SourceRevision(id="source_provider", uri="fixture:epub", sha256=sha(epub_bytes))]
    project.metadata["fresh_chapter_fixture"] = True
    project.roles, identities = [], {}
    for index, (rid, label, role_type) in enumerate(ROLE_SPECS):
        if rid in roles:
            identities[rid] = make_identity(base, rid, 10 + index, restrictions=["fixture restriction"] if rid == restrictions_for else None)
            project.roles.append(Role(id=rid, name=label, role_type=role_type, tts_backend="qwen3tts",
                                      voice_identity=identities[rid]))
    if "erick" in roles:
        approve(project, base, next(r for r in project.roles if r.id == "erick"))
        (base / "casting" / "voices" / "voice_erick" / "reference.json").write_bytes(json.dumps({"provenance": {
            "audition_draft_policy": {"fixture": True},
            "acquisition_sha256": identities["erick"].provenance["acquisition_sha256"]}}).encode())
    save_project(project)
    save_segments(project.id, [])
    shutil.copytree(base, origin_root / project.id)
    return SimpleNamespace(id=project.id, base=base, identities=identities, origin_root=origin_root,
                           sha=sha((base / "project.json").read_bytes()), origin=origin_root / project.id)


def build_chapter(root, rows, epub_bytes, *, assign=None):
    """``assign`` maps role id -> provider (existing-cast mode, no target approval); other roles get the target's
    own approval as before."""
    assign = assign or {}
    project = build_project(name="fresh chapter synthetic fixture", chapter_titles=[f"{NUMBER:03d}"])
    chapter = project.chapters[0]
    base = root / project.id
    (base / "source.snapshot").write_bytes(epub_bytes)
    source = SourceRevision(id="source_fixture", uri="fixture:epub", sha256=sha(epub_bytes), snapshot_file="source.snapshot")
    project.source_revisions = [source]
    project.metadata["fresh_chapter_fixture"] = True
    project.roles = []
    names = {"n": ("narrator", "narrator"), "e": ("erick", "Erick"), "j": ("jane", "Jane")}
    for index, (key, (rid, label)) in enumerate(names.items()):
        if rid in assign:
            identity = assign[rid].identities[rid]
            shutil.copytree(assign[rid].base / "casting" / "voices" / identity.id, base / "casting" / "voices" / identity.id)
            acquisition = identity.provenance["acquisition_id"]
            shutil.copytree(assign[rid].base / "voice_lab" / "discovery" / "acquisitions" / acquisition,
                            base / "voice_lab" / "discovery" / "acquisitions" / acquisition)
        else:
            identity = make_identity(base, rid, 10 + index)
        project.roles.append(Role(id=rid, name=label, role_type="narrator" if key == "n" else "character",
                                  tts_backend="qwen3tts", voice_identity=identity))
    for role in project.roles:
        if role.id not in assign:
            approve(project, base, role)
    if assign:
        entries = {rid: {"project_id": provider.id, "role_id": rid, "book_sha256": sha(epub_bytes),
                         "voice_id": provider.identities[rid].id,
                         "reference_audio_sha256": provider.identities[rid].reference_audio_sha256,
                         "prompt_sha256": provider.identities[rid].prompt_sha256, "project_sha256": provider.sha,
                         "origin_project_sha256": provider.sha, "origin_source_root": str(provider.origin_root),
                         "target_approval_created": False} for rid, provider in assign.items()}
        project.metadata["existing_book_cast_sources"] = entries
        project.metadata["owner_review_authorization"] = authorization(sha(epub_bytes), entries)
    segments, offset = [], 0
    for index, (key, text) in enumerate(rows):
        rid = names.get(key, names["n"])[0]
        anchor = SourceAnchor(source_revision_id=source.id, chapter_id=chapter.id, start_offset=offset,
                              end_offset=offset + len(text), occurrence_index=index)
        offset += len(text) + 2
        segments.append(Segment(id=f"{project.id}-{index:02}", chapter_id=chapter.id, index=index, role_id=rid, text=text,
                                source_anchor=anchor, attribution=Attribution(speaker_id=rid, source="user")))
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
    epub_bytes = epub_for(getattr(request, "param", None) or CHAPTER_11)
    epub_path = tmp_path / "synthetic.epub"
    epub_path.write_bytes(epub_bytes)
    project_id = build_chapter(root, CHAPTER_11, epub_bytes)
    monkeypatch.setattr(fresh, "_mp3", lambda master, target: target.write_bytes(master.read_bytes()))  # signal copy
    monkeypatch.setattr(fresh, "_decoded_duration", lambda path: wave_seconds(path))
    monkeypatch.setattr(fresh, "_loudness", lambda _: -20.0)
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path / "ambient-unrelated"))
    return SimpleNamespace(root=root, epub=epub_path, id=project_id, tmp=tmp_path)


def args_for(world, **overrides):
    values = dict(epub=world.epub, source_root=world.root, project=world.id, chapter=NUMBER,
                  output_dir=world.tmp / "out", coaching_bundle=None, performance_memory=None, generate=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def tree_hashes(root):
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in sorted(Path(root).rglob("*")) if p.is_file()}


def edit_project(world, mutate):
    path = world.root / world.id / "project.json"
    raw = json.loads(path.read_bytes())
    mutate(raw)
    path.write_bytes(json.dumps(raw).encode())


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


def write_bundle(world, state, out, *, mutate=None, name="bundle.json"):
    request = json.loads((out / state["request"]["file"]).read_bytes())
    path = world.tmp / name
    path.write_bytes(pilot.encoded(make_bundle(request, mutate=mutate)))
    return path


MEMORY_PROVENANCE = {"origin": "model", "model_requested": "synthetic-test-fixture-not-a-real-model",
                     "receipt_sha256": "a1" * 32, "fixture": True}


def memory_response(request):
    """A hand-written stand-in for a captured model response; each field is derived from the memory request."""
    habits = {"erick": ["Short, firm phrases with clean word endings.", "Leans on the final word of a demand."],
              "jane": ["Steady, measured phrasing that keeps endings open.", "Warmer and slower to commit than the other speaker."]}
    contrast = {"erick": "Blunt and brief next to the other speaker's steadier replies.",
                "jane": "Calmer and fuller than the other speaker's clipped demands."}
    profiles = []
    for role in request["roles"]:
        rid = role["role_id"]
        project_id, row = next((source["project_id"], r) for source in request["sources"] for r in source["context"]
                               if r["role_id"] == rid and not r["narrator"])
        profiles.append({"role_id": rid, "role_name": role["role_name"], "delivery_habits": list(habits[rid]),
                         "contrast_notes": contrast[rid], "confidence": "moderate",
                         "evidence": [{"project_id": project_id, "utterance_id": row["utterance_id"], "start": 0,
                                       "end": len(row["text"]), "text": row["text"]}]})
    return {"schema_version": "character-performance-memory-proposals-v1", "profiles": profiles}


def write_memory(world, state, out, *, name="memory.json", book=None, provenance=None, only_request=None):
    """A valid synthetic memory bundle built from the frozen plain coaching request of a prepared chapter."""
    from dubbing_service.app.audiobook import character_performance_memory as cpm
    request = only_request or json.loads((out / state["request"]["file"]).read_bytes())
    memory_request = cpm.build_memory_request([request], book_sha256=book or state["epub"]["sha256"])
    bundle = cpm.capture_bundle(memory_request, memory_response(memory_request), provenance or MEMORY_PROVENANCE)
    path = world.tmp / name
    path.write_bytes(pilot.encoded(bundle))
    return path


@pytest.fixture
def memory(world):
    """(path, plain frozen state): the memory is built from a scratch preparation of the same chapter."""
    scratch = args_for(world, output_dir=world.tmp / "scratch")
    state = fresh.prepare_chapter(scratch)
    return write_memory(world, state, scratch.output_dir), state


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def json(self):
        return self.body


class FakeClient:
    def __init__(self, workdir, runtime, *, forward=True, lead=0.0, tail=0.0):
        self.workdir, self.runtime, self.forward, self.calls = workdir, runtime, forward, []
        self.lead, self.tail = lead, tail

    def post(self, url, json):  # noqa: A002 - mirrors httpx
        self.calls.append(json)
        seed = int(hashlib.sha256((json["identity_id"] + json["text"]).encode()).hexdigest()[:8], 16)
        path = write_tone(self.workdir / f"{len(self.calls):04}.wav", max(0.5, 0.05 * len(json["text"])), 180 + seed % 90,
                          lead=self.lead, tail=self.tail)
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
    lead = tail = 0.0

    def __init__(self, output, directed=False):
        assert directed is True
        scratch = Path(output) / "worker-scratch"
        scratch.mkdir(exist_ok=True)
        self.client = FakeClient(scratch, self.runtime, forward=self.forward, lead=self.lead, tail=self.tail)
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
    FakeWorker.instances, FakeWorker.runtime, FakeWorker.profile = [], "7" * 64, pilot.DIRECTED_PROFILE
    FakeWorker.forward, FakeWorker.lead, FakeWorker.tail = True, 0.0, 0.0


@pytest.fixture(autouse=True)
def no_real_worker(monkeypatch):
    def forbidden():
        raise AssertionError("A test reached the real GPU worker")
    monkeypatch.setattr(pilot, "default_render_tools", forbidden)


@pytest.fixture
def prepared(world):
    args = args_for(world)
    return world, args, fresh.prepare_chapter(args)


@pytest.fixture
def ready(prepared):
    world, args, state = prepared
    args.coaching_bundle = write_bundle(world, state, args.output_dir)
    state = fresh.apply_bundle(args, state)
    assert state["status"] == "ready", state["holds"]
    return world, args, state


def all_calls():
    return [call for worker in FakeWorker.instances for call in worker.client.calls]


def chapter_receipt(args):
    return json.loads((args.output_dir / "chapter" / f"chapter-{NUMBER:03}.json").read_bytes())


# --------------------------------------------------------------------------- prepare


def test_prepare_freezes_source_approved_clones_and_request_without_takes_coaching_or_gpu(world):
    before = tree_hashes(world.root)
    args = args_for(world)
    state = fresh.prepare_chapter(args)
    out = args.output_dir
    assert state["status"] == "awaiting_coaching" and state["holds"] == [] and state["fixture"] is True
    for flag, value in fresh.REVIEW_FLAGS.items():
        assert state[flag] == value, flag
    assert state["review_only"] is True and state["takes_created"] is False and state["default_changed"] is False
    assert tree_hashes(world.root) == before                                   # canonical store read only
    assert all(not s.takes and not s.selected_take_id for s in load_segments(world.id))   # still no Takes at all
    assert not (out / "samples").exists() and not list(out.glob("worker-*")) and not (out / "applied.json").exists()
    lines = state["lines"]
    assert len(lines) == SPEECH and sum(l["narrator"] for l in lines) == NARRATION_LINES
    assert {l["mode"] for l in lines if l["narrator"]} == {"fresh_neutral_narration"}
    assert {l["mode"] for l in lines if not l["narrator"]} == {"fresh_directed_dialogue"}
    assert [l["utterance_id"] for l in lines] == [r["utterance_id"] for r in state["chapter"]["rows"] if r["kind"] == "speech"]
    assert state["narration"]["reused_approved_take"] is False and state["narration"]["native_instruction"] is None
    assert state["chapter"]["nonblank_characters"] > 0 and state["epub"]["sha256"] == sha(world.epub.read_bytes())
    assert state["memory"] is None
    contract = pilot.source_contract(json.loads((world.root / world.id / "project.json").read_bytes()))
    assert state["sources"][0]["source_contract_sha256"] == pilot.value_sha(contract)     # semantic contract is frozen
    request = json.loads((out / state["request"]["file"]).read_bytes())
    assert set(request) == {"instruction", "constraints", "response_contract", "scenes"}
    selected = [uid for scene in request["scenes"] for uid in scene["selected_dialogue_ids"]]
    assert selected == state["request"]["dialogue_ids"] and len(selected) == DIALOGUE_LINES
    narrators = {r["utterance_id"] for scene in request["scenes"] for r in scene["context"] if r["narrator"]}
    assert narrators and not narrators & set(selected)                      # narration is context, never a requested direction
    scene_coaching.validate_bundle(make_bundle(request))
    assert (out / "projects" / world.id / "project.json").is_file()


def test_every_speech_role_is_bound_to_its_approved_clone_source_and_casting_evidence(prepared):
    world, args, state = prepared
    assert set(state["roles"]) == {"narrator", "erick", "jane"}
    for role_id, role in state["roles"].items():
        voice = role["voice"]
        assert voice["provenance_origin"] == "eligible_reference_clone" and voice["status"] == "LOCKED"
        assert role["source_asset"]["acquisition_id"] and role["source_asset"]["receipt_file"].endswith("acquisition.json")
        assert role["casting"]["event"]["origin"] == "human" and role["casting"]["heard_sample_sha256"]
        assert role["casting"]["voice_revision"] == role["voice_revision"] == role["casting"]["version"]
        assert (world.root / world.id / role["source_asset"]["receipt_file"]).is_file()
        assert role["reference_record"] is None                              # fixture has no reference.json
    assert all(l["voice"] == state["roles"][l["role_id"]]["voice"] for l in state["lines"])


def test_prepare_retry_returns_the_frozen_state_and_changes_nothing(prepared):
    world, args, state = prepared
    before = tree_hashes(args.output_dir)
    again = fresh.prepare_chapter(args)
    assert again == state and tree_hashes(args.output_dir) == before
    with pytest.raises(fresh.FreshChapterError, match="different EPUB, chapter or project"):
        fresh.prepare_chapter(args_for(world, chapter=12))


def test_prepare_resumes_after_an_interrupted_attempt(world, monkeypatch):
    args = args_for(world)
    real = fresh._build_request

    def crash(*a, **k):
        raise RuntimeError("interrupted")
    monkeypatch.setattr(fresh, "_build_request", crash)
    with pytest.raises(RuntimeError, match="interrupted"):
        fresh.prepare_chapter(args)
    assert not (args.output_dir / "state.json").exists()
    monkeypatch.setattr(fresh, "_build_request", real)
    assert fresh.prepare_chapter(args)["status"] == "awaiting_coaching"      # published inputs are identical, not conflicts


def test_wrong_epub_or_unlisted_chapter_is_refused_before_any_state(world):
    other = world.tmp / "other.epub"
    other.write_bytes(epub_for([("n", "011"), ("n", "Different prose.")]))
    with pytest.raises(fresh.FreshChapterError, match="exact EPUB"):
        fresh.prepare_chapter(args_for(world, epub=other))
    with pytest.raises(fresh.FreshChapterError, match="No chapter titled"):
        fresh.prepare_chapter(args_for(world, chapter=12))
    assert not (world.tmp / "out" / "state.json").exists()


ALTERED = [(k, t.replace("Open", "Opem")) for k, t in CHAPTER_11]
EXTRA = CHAPTER_11 + [("n", "An extra closing paragraph the source lost.")]
SPACED = [(k, t.replace(" ", " \t ")) if k == "n" else (k, t) for k, t in CHAPTER_11]


@pytest.mark.parametrize("world", [ALTERED, EXTRA], indirect=True, ids=["changed-character", "lost-paragraph"])
def test_every_nonblank_epub_character_must_be_covered(world):
    with pytest.raises(fresh.FreshChapterError, match="every EPUB prose character; first difference"):
        fresh.prepare_chapter(args_for(world))
    assert not (world.tmp / "out" / "state.json").exists()


@pytest.mark.parametrize("world", [SPACED], indirect=True, ids=["whitespace-only"])
def test_whitespace_is_the_only_difference_ignored(world):
    assert fresh.prepare_chapter(args_for(world))["status"] == "awaiting_coaching"


@pytest.mark.parametrize("change", ["origin", "unlocked", "missing_identity", "backend", "policy_malformed"])
def test_non_clone_unlocked_or_wrong_route_voices_are_refused(world, change):
    def mutate(raw):
        role = next(r for r in raw["roles"] if r["id"] == "erick")
        if change == "origin":
            role["voice_identity"]["provenance"] = {"origin": "generated_original_voice", "actor_audio_used": False}
        elif change == "unlocked":
            role["voice_identity"]["status"] = "REVIEW"
        elif change == "missing_identity":
            role["voice_identity"] = None
        elif change == "backend":
            role["tts_backend"] = "indextts"
        else:
            raw["metadata"]["director_voice_policy"] = {"mode": "anything"}
    edit_project(world, mutate)
    with pytest.raises(fresh.FreshChapterError):
        fresh.prepare_chapter(args_for(world))
    assert not (world.tmp / "out" / "state.json").exists()


def test_provider_preset_or_synthetic_reference_is_refused(world):
    def mutate(raw):
        identity = next(r for r in raw["roles"] if r["id"] == "jane")["voice_identity"]
        identity.update(mechanism="builtin", builtin_speaker="preset-voice")
    edit_project(world, mutate)
    with pytest.raises(fresh.FreshChapterError, match="eligible cloned identity"):
        fresh.prepare_chapter(args_for(world))


@pytest.mark.parametrize("break_it", [
    "no_assignment", "rejected", "unapproved", "event_not_human", "event_not_positive", "event_other_role",
    "event_no_actor", "version_mismatch", "heard_sample_changed", "heard_sample_missing", "record_other_conditioning"])
def test_status_alone_is_not_approval_evidence(world, break_it):
    def mutate(raw):
        lab = raw["voice_lab"]
        assignment = lab["casting"]["roles"]["erick"]["assignment"]
        event = lab["casting"]["events"][assignment["event_id"]]
        record = lab["voices"][assignment["key"]]
        if break_it == "no_assignment":
            del lab["casting"]["roles"]["erick"]
        elif break_it == "rejected":
            assignment["rejected"] = True
        elif break_it == "unapproved":
            assignment["approved"] = False
        elif break_it == "event_not_human":
            event["origin"] = "automation"
        elif break_it == "event_not_positive":
            event["selection_confirmed"] = False
        elif break_it == "event_other_role":
            event["role_id"] = "jane"
        elif break_it == "event_no_actor":
            event["actor"] = " "
        elif break_it == "version_mismatch":
            assignment["voice_revision"] = "0" * 32
        elif break_it == "heard_sample_changed":
            (world.root / world.id / record["audio_file"]).write_bytes(b"replaced heard sample")
        elif break_it == "heard_sample_missing":
            (world.root / world.id / record["audio_file"]).unlink()
        else:
            record["identity"]["prompt_sha256"] = "f" * 64
    edit_project(world, mutate)
    with pytest.raises(fresh.FreshChapterError, match="approval evidence"):
        fresh.prepare_chapter(args_for(world))
    assert not (world.tmp / "out" / "state.json").exists()


def test_clone_source_asset_must_remain_verifiable(world):
    receipt = next((world.root / world.id / "voice_lab" / "discovery" / "acquisitions").glob("*/acquisition.json"))
    receipt.write_bytes(b"changed acquisition receipt")
    with pytest.raises(fresh.FreshChapterError, match="acquisition receipt"):
        fresh.prepare_chapter(args_for(world))


def test_clone_without_a_recorded_acquisition_is_unverifiable(world):
    def mutate(raw):
        role = next(r for r in raw["roles"] if r["id"] == "narrator")
        del role["voice_identity"]["provenance"]["acquisition_id"]
        lab = raw["voice_lab"]
        lab["voices"]["v_narrator"]["identity"] = role["voice_identity"]
    edit_project(world, mutate)
    with pytest.raises(fresh.FreshChapterError, match="names no recorded acquisition"):
        fresh.prepare_chapter(args_for(world))


def test_recorded_draft_clones_still_require_their_positive_human_selection(world):
    voice = world.root / world.id / "casting" / "voices" / "voice_erick"
    role = next(r for r in load_project(world.id).roles if r.id == "erick")
    provenance = role.voice_identity.provenance
    (voice / "reference.json").write_bytes(json.dumps({"provenance": {
        "audition_draft_policy": {"fixture": True}, "acquisition_sha256": provenance["acquisition_sha256"]}}).encode())
    # The recorded-draft owner reads the positive selection from the identity's own acquisition digest.
    edit_project(world, lambda raw: raw["voice_lab"]["voices"]["v_erick"]["identity"]["provenance"].update(
        acquisition_sha256=provenance["acquisition_sha256"]))
    state = fresh.prepare_chapter(args_for(world))
    assert state["roles"]["erick"]["reference_record"]["audition_draft_policy"] is True
    other = world.tmp / "other-out"
    edit_project(world, lambda raw: raw["voice_lab"]["casting"]["events"]["event_erick"].update(selection_confirmed=False))
    with pytest.raises(fresh.FreshChapterError):
        fresh.prepare_chapter(args_for(world, output_dir=other))


def test_absent_or_unresolved_attribution_is_refused(world):
    segments = load_segments(world.id)
    segments[3].attribution = None
    save_segments(world.id, segments)
    with pytest.raises(fresh.FreshChapterError, match="source-backed speaker attribution"):
        fresh.prepare_chapter(args_for(world))
    segments[3].attribution = Attribution(speaker_id="narrator", source="unknown")
    save_segments(world.id, segments)
    with pytest.raises(fresh.FreshChapterError, match="source-backed speaker attribution"):
        fresh.prepare_chapter(args_for(world))
    assert not (world.tmp / "out" / "state.json").exists()


def test_clone_file_change_is_refused(world):
    (world.root / world.id / "casting" / "voices" / "voice_erick" / "conditioning.npz").write_bytes(b"changed prompt")
    with pytest.raises(fresh.FreshChapterError, match="hash-mismatched|artifact"):
        fresh.prepare_chapter(args_for(world))


def test_output_inside_the_source_store_and_bad_arguments_are_refused(world):
    with pytest.raises(fresh.FreshChapterError, match="outside the canonical"):
        fresh.prepare_chapter(args_for(world, output_dir=world.root / "out"))
    with pytest.raises(fresh.FreshChapterError, match="positive integer"):
        fresh.prepare_chapter(args_for(world, chapter=0))
    with pytest.raises(fresh.FreshChapterError, match="valid project id"):
        fresh.prepare_chapter(args_for(world, project="../x"))


def test_prepare_holds_block_coaching_and_generation(world):
    real = fresh._build_plan

    def with_safety_hold(*a, **k):
        plan = real(*a, **k)
        plan["holds"].append({"code": "voice_unbound", "utterance_id": "x", "reason": "synthetic hold"})
        plan["plan_sha256"] = _sha({k: v for k, v in plan.items() if k != "plan_sha256"})
        return plan
    args = args_for(world)
    state = fresh.prepare_chapter(args, plan_builder=with_safety_hold)
    assert state["status"] == "held" and state["request"] is None and state["prepare_holds"]
    with pytest.raises(fresh.FreshChapterError, match="Preparation held"):
        fresh.apply_bundle(args, state)
    with pytest.raises(fresh.FreshChapterError, match="Preparation held"):
        fresh.generate_chapter(args, state, FakeWorker, RENDER)
    assert FakeWorker.instances == []


# --------------------------------------------------------------------------- performance memory


def test_performance_memory_enriches_the_frozen_request_and_is_bound_by_hash(world, memory):
    path, plain = memory
    args = args_for(world, performance_memory=path)
    state = fresh.prepare_chapter(args)
    out = args.output_dir
    binding = state["memory"]
    assert binding["file_sha256"] == sha(path.read_bytes()) == sha((out / binding["file"]).read_bytes())
    assert binding["interpretation"] == "unapproved_model_interpretation" and binding["approved"] is False
    assert binding["fixture"] is True and binding["profile_roles"] == ["erick", "jane"]
    assert binding["narrator_roles"] == ["narrator"]
    assert binding["base_request_sha256"] == plain["request"]["request_sha256"]
    request = json.loads((out / state["request"]["file"]).read_bytes())
    assert state["request"]["performance_memory_sha256"] == binding["memory_sha256"]
    assert all(scene["character_performance_memory"]["memory_sha256"] == binding["memory_sha256"]
               for scene in request["scenes"])
    assert request["instruction"] == fresh.REQUEST_INSTRUCTION
    assert len(request["constraints"]) == len(fresh.REQUEST_CONSTRAINTS) + 1
    # Narration gets no notes and the unchanged source fields of every scene remain.
    plain_request = json.loads((world.tmp / "scratch" / plain["request"]["file"]).read_bytes())
    for new, old in zip(request["scenes"], plain_request["scenes"]):
        assert {k: new[k] for k in old} == old
        assert all(note["role_id"] != "narrator" for note in new["character_performance_memory"]["notes"])
    args.coaching_bundle = write_bundle(world, state, out)
    ready = fresh.apply_bundle(args, state)
    assert ready["status"] == "ready"
    applied = json.loads((out / "applied.json").read_bytes())
    assert applied["performance_memory"]["memory_sha256"] == binding["memory_sha256"]
    assert fresh.prepare_chapter(args)["status"] == "ready"                  # idempotent with the same memory
    fresh.generate_chapter(args, ready, FakeWorker, RENDER)
    assert chapter_receipt(args)["coaching"]["performance_memory"]["memory_sha256"] == binding["memory_sha256"]
    args.performance_memory = None
    with pytest.raises(fresh.FreshChapterError, match="supply the same --performance-memory"):
        fresh.prepare_chapter(args)                                          # never silently reused


def test_a_bundle_for_the_unenriched_request_is_refused(world, memory):
    path, plain = memory
    args = args_for(world, performance_memory=path)
    state = fresh.prepare_chapter(args)
    plain_request = json.loads((world.tmp / "scratch" / plain["request"]["file"]).read_bytes())
    bundle_path = world.tmp / "plain.json"
    bundle_path.write_bytes(pilot.encoded(make_bundle(plain_request)))
    args.coaching_bundle = bundle_path
    held = fresh.apply_bundle(args, state)
    assert held["status"] == "held" and "frozen request" in held["holds"][0]["detail"]
    assert not (args.output_dir / "applied.json").exists()


def test_memory_cannot_be_added_swapped_or_tampered_after_preparation(world, memory):
    path, _ = memory
    fresh.prepare_chapter(args_for(world))
    with pytest.raises(fresh.FreshChapterError, match="prepared without performance memory"):
        fresh.prepare_chapter(args_for(world, performance_memory=path))
    other = args_for(world, output_dir=world.tmp / "out2", performance_memory=path)
    state = fresh.prepare_chapter(other)
    scratch = fresh.load_state(world.tmp / "scratch")
    different = write_memory(world, scratch, world.tmp / "scratch", name="different.json",
                             provenance={**MEMORY_PROVENANCE, "receipt_sha256": "b2" * 32})
    with pytest.raises(fresh.FreshChapterError, match="differs from the memory bound"):
        fresh.prepare_chapter(args_for(world, output_dir=world.tmp / "out2", performance_memory=different))
    private = other.output_dir / state["memory"]["file"]
    private.write_bytes(private.read_bytes() + b" ")
    with pytest.raises(fresh.FreshChapterError, match="performance memory bound to this chapter is missing or changed"):
        fresh.load_state(other.output_dir)


def test_memory_for_another_book_cast_or_a_non_fixture_chapter_is_refused(world, memory):
    path, plain = memory
    scratch = world.tmp / "scratch"
    wrong_book = write_memory(world, plain, scratch, name="book.json", book="d" * 64)
    with pytest.raises(fresh.FreshChapterError, match="different book"):
        fresh.prepare_chapter(args_for(world, output_dir=world.tmp / "o1", performance_memory=wrong_book))
    erick_only = json.loads((scratch / plain["request"]["file"]).read_bytes())
    for scene in erick_only["scenes"]:
        roles = {r["utterance_id"]: r["role_id"] for r in scene["context"]}
        scene["selected_dialogue_ids"] = [u for u in scene["selected_dialogue_ids"] if roles[u] == "erick"]
        scene["context"] = [r for r in scene["context"] if r["role_id"] != "jane"]
    erick_only["scenes"] = [s for s in erick_only["scenes"] if s["selected_dialogue_ids"]]
    partial = write_memory(world, plain, scratch, name="partial.json", only_request=erick_only)
    with pytest.raises(fresh.FreshChapterError, match="does not fit this cast"):
        fresh.prepare_chapter(args_for(world, output_dir=world.tmp / "o2", performance_memory=partial))
    edit_project(world, lambda raw: raw["metadata"].update(fresh_chapter_fixture=False))
    with pytest.raises(fresh.FreshChapterError, match="fixture performance memory"):
        fresh.prepare_chapter(args_for(world, output_dir=world.tmp / "o3", performance_memory=path))
    for name in ("o1", "o2", "o3"):
        assert not (world.tmp / name / "state.json").exists() and not (world.tmp / name / "requests").exists()


def test_memory_owner_unavailable_or_overreaching_is_refused(world, memory, monkeypatch):
    path, _ = memory
    from dubbing_service.app.audiobook import character_performance_memory as real

    def unavailable():
        raise fresh.FreshChapterError("Character performance memory is unavailable (character_performance_memory)")
    monkeypatch.setattr(fresh, "_memory_module", unavailable)
    with pytest.raises(fresh.FreshChapterError, match="unavailable"):
        fresh.prepare_chapter(args_for(world, output_dir=world.tmp / "o1", performance_memory=path))

    def overreaching(mutate):
        def enrich(request, bundle, *, book_sha256):
            result = real.enrich_coaching_request(request, bundle, book_sha256=book_sha256)
            mutate(result)
            return result
        return SimpleNamespace(validate_bundle=real.validate_bundle, require_role_coverage=real.require_role_coverage,
                               enriched_memory_sha256=real.enriched_memory_sha256, STATUS=real.STATUS,
                               enrich_coaching_request=enrich)
    changes = {
        "drops_a_scene": lambda r: r.update(scenes=r["scenes"][:-1] if len(r["scenes"]) > 1 else []),
        "rewrites_context": lambda r: r["scenes"][0]["context"][0].update(text="Rewritten prose."),
        "drops_a_constraint": lambda r: r.update(constraints=r["constraints"][1:]),
        "changes_contract": lambda r: r["response_contract"].update(extra="field"),
    }
    for index, (label, mutate) in enumerate(changes.items()):
        monkeypatch.setattr(fresh, "_memory_module", lambda mutate=mutate: overreaching(mutate))
        with pytest.raises(fresh.FreshChapterError, match="may only add context|does not carry"):
            fresh.prepare_chapter(args_for(world, output_dir=world.tmp / f"x{index}", performance_memory=path))
        assert not (world.tmp / f"x{index}" / "state.json").exists(), label


# --------------------------------------------------------------------------- coaching bundle


def test_no_bundle_leaves_the_chapter_awaiting_and_publishes_nothing(prepared):
    world, args, state = prepared
    before = tree_hashes(args.output_dir)
    state = fresh.apply_bundle(args, state)
    assert state["status"] == "awaiting_coaching"
    assert {k: v for k, v in tree_hashes(args.output_dir).items() if k != "state.json"} == \
        {k: v for k, v in before.items() if k != "state.json"}
    with pytest.raises(fresh.FreshChapterError, match="no coaching has been applied"):
        fresh.generate_chapter(args, state, FakeWorker, RENDER)
    assert FakeWorker.instances == []


def test_valid_bundle_publishes_an_immutable_coached_plan_covering_all_dialogue(ready):
    world, args, state = ready
    out = args.output_dir
    coached = json.loads((out / "plans" / "coached.json").read_bytes())
    assert coached["recommendation_only"] is True and coached["approved"] is False and coached["human_review"] == "pending"
    directions = {d["utterance_id"]: d for d in coached["directions"]}
    for line in state["lines"]:
        row = directions[line["utterance_id"]]
        assert row["text_sha256"] == line["text_sha256"]                       # prose is never rewritten
        if line["narrator"]:
            assert row["native_instruction"] is None
        else:
            assert row["basis"] == "model" and 0 < len(row["native_instruction"]) <= 220 and row["coaching"]
    assert (out / "coaching" / "bundle.json").read_bytes() == args.coaching_bundle.read_bytes()
    receipt = json.loads((out / "applied.json").read_bytes())
    assert receipt["model_provenance"]["origin"] == "model" and receipt["approved"] is False
    assert receipt["narration"]["reused_approved_take"] is False and receipt["narration_lines"] == NARRATION_LINES
    assert receipt["fixture"]["label"] == fresh.FIXTURE_LABEL


def test_identical_retry_is_a_no_op_and_a_changed_bundle_never_overwrites(ready):
    world, args, state = ready
    before = tree_hashes(args.output_dir)
    again = fresh.apply_bundle(args, state)
    assert again["status"] == "ready" and tree_hashes(args.output_dir) == before
    args.coaching_bundle = write_bundle(world, state, args.output_dir, name="changed.json", mutate=lambda b: b["response"][
        "scenes"][0]["directions"][0].update(native_instruction="Deliver this with a different, quieter shape; keep every word clear."))
    held = fresh.apply_bundle(args, again)
    assert held["status"] == "held" and "already applied" in held["holds"][0]["detail"]
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
    args.coaching_bundle = write_bundle(world, state, args.output_dir, mutate=mutate)
    state = fresh.apply_bundle(args, state)
    out = args.output_dir
    assert state["status"] == "held" and state["coaching"]["status"] == "refused"
    assert not (out / "applied.json").exists() and not (out / "coaching").exists()
    assert not (out / "plans" / "coached.json").exists()
    with pytest.raises(fresh.FreshChapterError):
        fresh.generate_chapter(args, state, FakeWorker, RENDER)
    assert FakeWorker.instances == []


def test_bundle_hash_tamper_is_refused_and_a_corrected_bundle_can_still_apply(prepared):
    world, args, state = prepared
    path = write_bundle(world, state, args.output_dir)
    bundle = json.loads(path.read_bytes())
    bundle["response"]["scenes"][0]["directions"][0]["emotion"] = "tampered after capture"
    tampered = world.tmp / "tampered.json"
    tampered.write_bytes(pilot.encoded(bundle))
    args.coaching_bundle = tampered
    held = fresh.apply_bundle(args, state)
    assert held["status"] == "held" and "hash" in held["holds"][0]["detail"]
    args.coaching_bundle = path
    assert fresh.apply_bundle(args, held)["status"] == "ready"


def test_source_or_cast_drift_blocks_coaching_but_notification_counters_do_not(prepared):
    world, args, state = prepared
    args.coaching_bundle = write_bundle(world, state, args.output_dir)

    def counters(raw):
        raw["voice_lab"]["revision"] += 7                                  # delivery bookkeeping only
        raw["voice_lab"]["casting"]["notifications"] = {"complete": {"next_retry_at": "later"}}
    edit_project(world, counters)
    assert fresh.apply_bundle(args, state)["status"] == "ready"
    edit_project(world, lambda raw: raw["roles"][1].update(name="Renamed Erick"))
    with pytest.raises(fresh.FreshChapterError, match="source contract changed|drifted since preparation"):
        fresh.apply_bundle(args, state)


# --------------------------------------------------------------------------- generation


def test_generation_synthesizes_neutral_narration_and_directed_dialogue_through_one_worker(ready):
    world, args, state = ready
    source_before = tree_hashes(world.root)
    result = fresh.generate_chapter(args, state, FakeWorker, RENDER)
    out = args.output_dir
    dialogue = [l for l in state["lines"] if not l["narrator"]]
    narration = [l for l in state["lines"] if l["narrator"]]
    assert result["status"] == "generated" and len(FakeWorker.instances) == 1
    worker = FakeWorker.instances[0]
    assert worker.entered and worker.exited
    sent = all_calls()
    assert len(sent) == SPEECH == len(dialogue) + len(narration)
    by_text = {c["text"]: c for c in sent}
    for line in narration:                                                   # narration is really dispatched, neutrally
        call = by_text[line["text"]]
        assert call["performance"] == {} and call["identity_id"] == line["voice"]["id"]
        assert call["prompt_sha256"] == line["voice"]["prompt_sha256"]
    for line in dialogue:                                                    # dialogue forwards the captured instruction
        call = by_text[line["text"]]
        instruction = call["performance"]["native_instruction"]
        assert 0 < len(instruction) <= 220 and call["performance"]["control_policy"] == pilot.DIRECTION_POLICY
        assert call["identity_id"] == line["voice"]["id"]
    assert [c["text"] for c in sent] == [l["text"] for l in state["lines"]]  # strictly serial, in source order
    audit = result["audit"]
    assert audit["dialogue_instruction_forwarded"] == DIALOGUE_LINES and audit["narration_instructions"] == 0
    assert audit["narration_fresh_neutral_synthesis"] == NARRATION_LINES and audit["narration_reused_takes"] == 0
    assert audit["narration_dispatched_this_run"] == NARRATION_LINES and audit["workers_started_this_run"] == 1
    assert tree_hashes(world.root) == source_before                         # no Take, project or segment changed
    assert all(not s.takes and not s.selected_take_id for s in load_segments(world.id))
    assert json.loads((out / "state.json").read_bytes())["status"] == "generated"
    assert sorted(p.name for p in (out / "receipts").iterdir()) == sorted(f"{l['utterance_id']}.json" for l in state["lines"])


def test_every_receipt_and_map_row_says_narration_is_new_audio_not_a_reused_take(ready):
    world, args, state = ready
    fresh.generate_chapter(args, state, FakeWorker, RENDER)
    out = args.output_dir
    for line in state["lines"]:
        record = json.loads((out / "receipts" / f"{line['utterance_id']}.json").read_bytes())
        assert record["schema_version"] == fresh.SAMPLE_SCHEMA and record["arm"] == "directed"
        assert record["review_only"] is True and record["takes_created"] is False and record["production_ready"] is False
        assert record["raw_render_receipt"]["sha256"] and record["technical_qa"]["passed"] is True
        assert record["clone_binding"]["voice_id"] == line["voice"]["id"]
        assert record["fixture"]["label"] == fresh.FIXTURE_LABEL
        if line["narrator"]:
            assert record["audio_provenance"] == "fresh_neutral_review_synthesis" and record["instruction"] is None
            assert record["instruction_forwarded"] is False and record["narration"]["reused_approved_take"] is False
            assert "not a reused, selected or approved Take" in record["narration"]["statement"]
        else:
            assert record["audio_provenance"] == "fresh_directed_review_synthesis" and record["narration"] is None
            assert record["instruction_forwarded"] is True and record["coaching"]["reading"]["objective"]
    receipt = chapter_receipt(args)
    assert receipt["schema_version"] == fresh.CHAPTER_SCHEMA and receipt["status"] == "ready_for_listening"
    for flag, value in fresh.REVIEW_FLAGS.items():
        assert receipt[flag] == value, flag
    assert receipt["narration"]["reused_approved_take"] is False and receipt["fixture"]["is_fixture"] is True
    assert receipt["quality_improvement"] == "not_established" and receipt["technical_qa"] == "passed"
    speech = [r for r in receipt["source_to_audio"] if r["kind"] == "speech"]
    assert [r["utterance_id"] for r in speech] == [l["utterance_id"] for l in state["lines"]]
    for row, line in zip(speech, state["lines"]):
        assert row["source_anchor"] == line["source"] and row["text"] == line["text"] and row["role_id"] == line["role_id"]
        assert row["cloned_reference"]["reference_audio_sha256"] == line["voice"]["reference_audio_sha256"]
        assert row["cloned_reference"]["source_asset"]["acquisition_id"] and row["receipt"]["raw"]["sha256"]
        assert sha((out / row["audio_file"]).read_bytes()) == row["audio_sha256"]
        if line["narrator"]:
            assert row["provenance"] == "fresh_neutral_review_synthesis" and row["native_instruction"] is None
            assert row["narration"]["reused_approved_take"] is False and row["coaching"] is None
        else:
            assert row["provenance"] == "fresh_directed_review_synthesis" and row["native_instruction"]
            assert row["coaching"]["scene_id"] and row["narration"] is None
    assert {b["role_id"] for b in receipt["clone_binding"]} == {"narrator", "erick", "jane"}
    assert all(b["casting_approval"]["event_id"] and b["source_asset"]["acquisition_sha256"] for b in receipt["clone_binding"])
    audit = json.loads((out / "forwarding-audit.json").read_bytes())
    assert audit["narration"]["reused_approved_take"] is False and audit["narration_reused_takes"] == 0
    summary = json.loads((out / "result.json").read_bytes())
    assert summary["schema_version"] == fresh.RESULT_SCHEMA and summary["production_ready"] is False
    assert summary["narration"]["selected_take_used"] is False


def test_assembly_keeps_source_order_separators_and_exact_accounting_without_trimming(ready):
    world, args, state = ready
    FakeWorker.lead, FakeWorker.tail = 0.1, 0.2                              # measured edge silence is credited, not doubled
    fresh.generate_chapter(args, state, FakeWorker, RENDER)
    out = args.output_dir
    receipt = chapter_receipt(args)
    speech = [r for r in receipt["source_to_audio"] if r["kind"] == "speech"]
    boundaries = [r["boundary"] for r in speech if r["boundary"]]
    assert len(boundaries) == SPEECH - 1 and all(b["trimming"] is False for b in boundaries)
    crossing = [b for b in boundaries if b["source_separators"]]
    assert len(crossing) == 1 and crossing[0]["target_ms"] >= fresh.SEPARATOR_TARGET_MS
    assert any(r["kind"] == "scene_break" and r["pause_target_ms"] == 1250 for r in receipt["source_to_audio"])
    assert any(b["existing_silence_ms"] > 0 for b in boundaries)             # measured by the boundary owner, then deducted
    for b in boundaries:
        assert b["added_frames"] == max(0, round((b["target_ms"] - b["existing_silence_ms"]) * RATE / 1000))
    accounting = receipt["accounting"]
    clips = sum(wave_seconds(out / r["audio_file"]) for r in speech)
    padding = sum(b["added_frames"] for b in boundaries) / RATE
    assert accounting["speech_seconds"] == pytest.approx(clips, abs=1e-6)
    assert accounting["added_padding_seconds"] == pytest.approx(padding, abs=1e-6) == pytest.approx(receipt["added_padding_seconds"])
    assert receipt["master_duration_seconds"] == pytest.approx(clips + padding, abs=1e-3)
    assert accounting["existing_silence_credited_seconds"] > 0 and accounting["trimming"] is False
    # No trimming: each clip's frames appear intact inside the master at its mapped position, in source order.
    with wave.open(str(out / receipt["master_file"]), "rb") as master:
        frames = master.readframes(master.getnframes())
    last = -1.0
    for row in speech:
        with wave.open(str(out / row["audio_file"]), "rb") as clip:
            data = clip.readframes(clip.getnframes())
        start = round(row["start_seconds"] * RATE) * 2
        assert frames[start:start + len(data)] == data and row["start_seconds"] > last
        last = row["start_seconds"]
    for kind in ("master", "audio"):
        assert sha((out / receipt[kind + "_file"]).read_bytes()) == receipt[kind + "_sha256"]


def test_completed_retry_verifies_everything_and_starts_no_worker_and_reuses_nothing(ready):
    world, args, state = ready
    fresh.generate_chapter(args, state, FakeWorker, RENDER)
    before = tree_hashes(args.output_dir)
    FakeWorker.instances = []
    again = fresh.generate_chapter(args, fresh.load_state(args.output_dir), FakeWorker, RENDER)
    assert again["status"] == "generated" and FakeWorker.instances == [] and all_calls() == []
    assert again["audit"]["workers_started_this_run"] == 0 and again["audit"]["new_requests_this_run"] == 0
    assert again["audit"]["narration_reused_takes"] == 0
    assert {k: v for k, v in tree_hashes(args.output_dir).items() if not k.startswith("source-checks/")} == \
        {k: v for k, v in before.items() if not k.startswith("source-checks/")}


def test_partial_run_resumes_only_the_missing_samples_with_one_worker(ready):
    world, args, state = ready
    calls = {"n": 0}

    def flaky(worker, *a, **k):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("worker dropped")
        return RENDER(worker, *a, **k)
    with pytest.raises(RuntimeError, match="worker dropped"):
        fresh.generate_chapter(args, state, FakeWorker, flaky)
    assert FakeWorker.instances[0].exited                                    # always released, never leaked
    assert not (args.output_dir / "chapter").exists()
    first = all_calls()
    assert len(first) == 3
    FakeWorker.instances = []
    result = fresh.generate_chapter(args, fresh.load_state(args.output_dir), FakeWorker, RENDER)
    assert len(all_calls()) == SPEECH - 3 and len(FakeWorker.instances) == 1
    assert result["audit"]["new_requests_this_run"] == SPEECH - 3 and result["audit"]["workers_started_this_run"] == 1
    assert {c["text"] for c in all_calls()}.isdisjoint({c["text"] for c in first})   # nothing resent


def test_ambiguous_pending_dispatch_holds_before_any_worker_exists(ready):
    world, args, state = ready
    line = next(l for l in state["lines"] if l["narrator"])
    sample = args.output_dir / "samples" / f"{line['utterance_id']}.wav"
    request_sha = pilot.expected_request_sha(line["project_id"], line["voice"], line["text"], None)
    pilot.publish(sample.with_suffix(f".pending-{request_sha[:16]}.json"), b"{}")
    with pytest.raises(fresh.FreshChapterError, match="ambiguous pending dispatch"):
        fresh.generate_chapter(args, state, FakeWorker, RENDER)
    assert FakeWorker.instances == [] and all_calls() == []                  # no resend, no GPU lane touched
    assert not (args.output_dir / "chapter").exists()


def test_failed_technical_qa_on_any_clip_holds_without_assembly_or_resend(ready, monkeypatch):
    world, args, state = ready
    real = qa.qa_take_audio
    seen = []

    def failing(*a, take_id="", **k):
        if take_id.startswith("sample_"):
            seen.append(take_id)
            if len(seen) == 2:
                return SimpleNamespace(passed=False, blocking_failures=lambda: ["fixture"], executed_check_ids=lambda: ["x"])
        return real(*a, take_id=take_id, **k)
    monkeypatch.setattr(qa, "qa_take_audio", failing)
    with pytest.raises(fresh.FreshChapterError, match="fails blocking technical QA"):
        fresh.generate_chapter(args, state, FakeWorker, RENDER)
    assert not (args.output_dir / "chapter").exists() and len(all_calls()) == 2
    assert FakeWorker.instances[0].exited
    monkeypatch.setattr(qa, "qa_take_audio", real)                           # the saved clip is verified, never resent
    FakeWorker.instances = []
    fresh.generate_chapter(args, fresh.load_state(args.output_dir), FakeWorker, RENDER)
    assert len(all_calls()) == SPEECH - 2


def test_unconfirmed_forwarding_wrong_profile_pin_fallback_and_voice_are_refused(ready):
    world, args, state = ready
    FakeWorker.forward = False
    with pytest.raises(fresh.FreshChapterError, match="did not confirm"):
        fresh.generate_chapter(args, state, FakeWorker, RENDER)
    assert not (args.output_dir / "chapter").exists()
    FakeWorker.forward = True
    FakeWorker.instances = []
    FakeWorker.profile = "faster-base-032"
    with pytest.raises(fresh.FreshChapterError, match="pinned"):
        fresh.generate_chapter(args, state, FakeWorker, RENDER)
    FakeWorker.profile = pilot.DIRECTED_PROFILE

    def with_fallback(worker, *a, **k):
        receipt = RENDER(worker, *a, **k)
        receipt["worker_receipt"]["fallback"] = "generic-preset"
        return receipt
    with pytest.raises(fresh.FreshChapterError, match="fallback"):
        fresh.generate_chapter(args, state, FakeWorker, with_fallback)

    def wrong_voice(worker, *a, **k):
        receipt = RENDER(worker, *a, **k)
        receipt["worker_receipt"]["identity_id"] = "another_voice"
        return receipt
    with pytest.raises(fresh.FreshChapterError, match="different voice identity"):
        fresh.generate_chapter(args, state, FakeWorker, wrong_voice)


def test_runtime_pin_blocks_a_different_worker_in_a_resumed_run(ready):
    world, args, state = ready
    pilot.publish(args.output_dir / "runtime-pin.json", pilot.encoded(
        {"runtime_sha256": "9" * 64, "runtime_profile": pilot.DIRECTED_PROFILE,
         "checkpoint_manifest_sha256": "c" * 64, "model_id": "m"}))
    with pytest.raises(fresh.FreshChapterError, match="differs from the pinned run"):
        fresh.generate_chapter(args, state, FakeWorker, RENDER)
    assert all_calls() == []


def test_narration_may_never_carry_an_instruction(ready):
    world, args, state = ready
    sent = []

    def spy(worker, project_id, voice, line, path, instruction=None):
        sent.append((line["narrator"], instruction))
        return RENDER(worker, project_id, voice, line, path, instruction)
    fresh.generate_chapter(args, state, FakeWorker, spy)
    assert [i for narrator, i in sent if narrator] == [None] * NARRATION_LINES
    assert all(i for narrator, i in sent if not narrator)


@pytest.mark.parametrize("drift", ["role_name", "segments", "clone_file", "worker_clone_copy", "coached_plan", "bundle",
                                   "approval_revoked", "approval_event", "heard_sample", "acquisition_receipt",
                                   "voice_changed", "request_file", "project_name"])
def test_drift_after_preparation_is_refused_before_a_worker_is_started(ready, drift):
    world, args, state = ready
    out, base = args.output_dir, world.root / world.id
    if drift == "role_name":
        edit_project(world, lambda raw: raw["roles"][1].update(name="Renamed"))
    elif drift == "segments":
        path = base / "segments.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif drift == "clone_file":
        (base / "casting" / "voices" / "voice_erick" / "reference.wav").write_bytes(b"changed")
    elif drift == "worker_clone_copy":
        (out / "projects" / world.id / "casting" / "voices" / "voice_narrator" / "reference.wav").write_bytes(b"changed")
    elif drift == "coached_plan":
        path = out / "plans" / "coached.json"
        plan = json.loads(path.read_bytes())
        next(d for d in plan["directions"] if not d["narrator"])["native_instruction"] = "Whisper conspiratorially."
        path.write_bytes(pilot.encoded(plan))
    elif drift == "bundle":
        path = out / "coaching" / "bundle.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif drift == "approval_revoked":
        edit_project(world, lambda raw: raw["voice_lab"]["casting"]["roles"]["jane"]["assignment"].update(approved=False))
    elif drift == "approval_event":
        edit_project(world, lambda raw: raw["voice_lab"]["casting"]["events"]["event_narrator"].update(origin="automation"))
    elif drift == "heard_sample":
        record = json.loads((base / "project.json").read_bytes())["voice_lab"]["voices"]["v_erick"]
        (base / record["audio_file"]).write_bytes(b"replaced heard sample")
    elif drift == "acquisition_receipt":
        next((base / "voice_lab" / "discovery" / "acquisitions").glob("*/acquisition.json")).write_bytes(b"changed")
    elif drift == "voice_changed":
        edit_project(world, lambda raw: raw["roles"][2]["voice_identity"].update(model="another-model"))
    elif drift == "project_name":
        edit_project(world, lambda raw: raw.update(name="A renamed project"))
    elif drift == "request_file":
        path = out / state["request"]["file"]
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(fresh.FreshChapterError):
        fresh.generate_chapter(args, state, FakeWorker, RENDER)
    assert FakeWorker.instances == [] and not (out / "samples").exists()


def test_tampered_or_substituted_state_is_refused(ready):
    world, args, state = ready
    forged = copy.deepcopy(state)
    next(l for l in forged["lines"] if not l["narrator"])["text"] += " tampered"
    with pytest.raises(fresh.FreshChapterError, match="differs from the frozen"):
        fresh.generate_chapter(args, forged, FakeWorker, RENDER)
    forged = copy.deepcopy(state)
    forged["roles"]["narrator"]["casting"]["event_id"] = "event_forged"
    with pytest.raises(fresh.FreshChapterError, match="differs from the frozen"):
        fresh.generate_chapter(args, forged, FakeWorker, RENDER)
    path = args.output_dir / "state.json"
    disk = json.loads(path.read_bytes())
    disk["lines"][0]["text"] += " tampered"
    path.write_bytes(pilot.encoded(disk))
    with pytest.raises(fresh.FreshChapterError, match="changed since preparation"):
        fresh.load_state(args.output_dir)
    assert FakeWorker.instances == []


def test_generation_tolerates_only_delivery_notification_counters(ready):
    world, args, state = ready

    def counters(raw):
        raw["voice_lab"]["revision"] += 3
        raw["voice_lab"]["casting"]["notifications"] = {"complete": {"next_retry_at": "later"}}
    edit_project(world, counters)
    assert fresh.generate_chapter(args, state, FakeWorker, RENDER)["status"] == "generated"


def test_published_outputs_are_immutable_and_verified_on_retry(ready):
    world, args, state = ready
    fresh.generate_chapter(args, state, FakeWorker, RENDER)
    out = args.output_dir
    sample = next((out / "samples").glob("*.wav"))
    original = sample.read_bytes()
    sample.write_bytes(original + b"\x00\x00")
    with pytest.raises(RuntimeError, match="receipt mismatch"):
        fresh.generate_chapter(args, fresh.load_state(out), FakeWorker, RENDER)
    sample.write_bytes(original)
    master = out / chapter_receipt(args)["master_file"]
    good = master.read_bytes()
    master.write_bytes(good + b"\x00\x00")
    with pytest.raises(fresh.FreshChapterError, match="changed"):
        fresh.generate_chapter(args, fresh.load_state(out), FakeWorker, RENDER)
    master.write_bytes(good)
    receipt_path = out / "chapter" / f"chapter-{NUMBER:03}.json"
    saved = receipt_path.read_bytes()
    forged = json.loads(saved)
    forged["clone_binding"][0]["prompt_sha256"] = "e" * 64
    receipt_path.write_bytes(pilot.encoded(forged))
    with pytest.raises(fresh.FreshChapterError, match="clone bindings"):
        fresh.generate_chapter(args, fresh.load_state(out), FakeWorker, RENDER)
    forged = json.loads(saved)
    forged["master_duration_seconds"] += 5
    receipt_path.write_bytes(pilot.encoded(forged))
    with pytest.raises(fresh.FreshChapterError, match="durations"):
        fresh.generate_chapter(args, fresh.load_state(out), FakeWorker, RENDER)
    receipt_path.write_bytes(saved)
    assert fresh.generate_chapter(args, fresh.load_state(out), FakeWorker, RENDER)["status"] == "generated"
    with pytest.raises(pilot.PilotError, match="conflicts"):
        pilot.publish(out / "plans" / "coached.json", b"different")


def test_governed_exclusions_stay_in_the_source_map_without_audio(world, monkeypatch):
    """A source-backed omission keeps its prose in the coverage check but receives no audio or direction."""
    from dubbing_service.app.audiobook import voice_lab_chapter_export as exporter
    heading = f"{world.id}-00"
    real = exporter._source_snapshot

    def with_exclusion(project_id, chapter_id):
        project, segments, source = real(project_id, chapter_id)
        source = {**source, "speech_utterance_ids": [s for s in source["speech_utterance_ids"] if s != heading],
                  "exclusions": [{"utterance_id": heading, "decision": {"reason": "fixture"},
                                  "source_text_sha256": text_sha256("011")}]}
        return project, segments, source
    monkeypatch.setattr(exporter, "_source_snapshot", with_exclusion)
    args = args_for(world)
    state = fresh.prepare_chapter(args)
    assert heading not in {l["utterance_id"] for l in state["lines"]}
    args.coaching_bundle = write_bundle(world, state, args.output_dir)
    state = fresh.apply_bundle(args, state)
    fresh.generate_chapter(args, state, FakeWorker, RENDER)
    receipt = chapter_receipt(args)
    row = next(r for r in receipt["source_to_audio"] if r["utterance_id"] == heading)
    assert row["kind"] == "excluded" and row["audio_file"] is None
    assert receipt["counts"]["excluded"] == 1 and receipt["counts"]["speech"] == SPEECH - 1
    assert heading not in {c["text"] for c in all_calls()}


# --------------------------------------------------------------------------- CLI


def argv_for(world, *extra):
    return ["--epub", str(world.epub), "--source-root", str(world.root), "--project", world.id,
            "--chapter", str(NUMBER), "--output-dir", str(world.tmp / "cli-out"), *extra]


def test_cli_argument_parsing_is_strict(world):
    args = fresh.parse_args(argv_for(world))
    assert args.project == world.id and args.chapter == NUMBER and args.coaching_bundle is None
    assert args.performance_memory is None and not args.generate
    args = fresh.parse_args(argv_for(world, "--coaching-bundle", str(world.epub), "--performance-memory", str(world.epub),
                                     "--generate"))
    assert args.generate and args.coaching_bundle == world.epub and args.performance_memory == world.epub
    for bad in (["--chapter", "four"], ["--chapter", "0"], ["--project", "../x"]):
        argv = argv_for(world)
        index = argv.index(bad[0])
        argv[index:index + 2] = bad
        with pytest.raises(SystemExit) as caught:
            fresh.parse_args(argv)
        assert caught.value.code == 2
    with pytest.raises(SystemExit):
        fresh.parse_args(["--epub", str(world.tmp / "missing.epub")] + argv_for(world)[2:])
    with pytest.raises(SystemExit):
        fresh.parse_args(argv_for(world, "--coaching-bundle", str(world.tmp / "missing.json")))
    with pytest.raises(SystemExit):
        fresh.parse_args(argv_for(world)[:-2])                                # no output directory


def test_cli_prepare_apply_generate_flow_and_exit_codes(world, capsys, monkeypatch):
    assert fresh.main(argv_for(world)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "awaiting_coaching" and summary["production_ready"] is False
    assert summary["dialogue_lines"] == DIALOGUE_LINES and summary["narration_lines"] == NARRATION_LINES
    assert fresh.main(argv_for(world, "--generate")) == 2                    # nothing coached: generation is refused
    assert json.loads(capsys.readouterr().out)["status"] == "awaiting_coaching" and FakeWorker.instances == []
    state = fresh.load_state(world.tmp / "cli-out")
    bundle = write_bundle(world, state, world.tmp / "cli-out")
    monkeypatch.setattr(pilot, "default_render_tools", lambda: (FakeWorker, RENDER))
    assert fresh.main(argv_for(world, "--coaching-bundle", str(bundle), "--generate")) == 0
    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])     # render() prints progress lines first
    assert final["status"] == "generated" and final["generation"]["audit"]["dialogue_instruction_forwarded"] == DIALOGUE_LINES
    assert len(FakeWorker.instances) == 1
    assert fresh.main(argv_for(world, "--coaching-bundle", str(bundle), "--generate")) == 0   # idempotent retry
    assert len(FakeWorker.instances) == 1


def test_cli_reports_a_refusal_as_held_with_exit_code_two(world, capsys):
    edit_project(world, lambda raw: next(r for r in raw["roles"] if r["id"] == "erick")["voice_identity"].update(status="REVIEW"))
    assert fresh.main(argv_for(world)) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "held"


def test_cli_script_is_a_thin_entry_point():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_fresh_coached_chapter.py"
    spec = importlib.util.spec_from_file_location("render_fresh_coached_chapter_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main is fresh.main


# --------------------------------------------------------------------------- existing book cast (private review)
# The retained recorded clones are reused under an explicit owner request. The target chapter holds NO per-role human
# approval of its own; the immutable original book-role project authorizes each backend call.

ALL_ROLES = ("narrator", "erick", "jane")


@pytest.fixture
def cast_world(monkeypatch, tmp_path, request):
    """Optional indirect param: {"declared": role ids reusing the retained cast, "split": jane lives in a 2nd provider,
    "restrictions_for": role whose source receipt carries a restriction}."""
    options = getattr(request, "param", None) or {}
    declared = options.get("declared", ALL_ROLES)
    root = use_projects_root(monkeypatch, tmp_path)
    origin_root = tmp_path / "origin-projects"
    origin_root.mkdir()
    epub_bytes = epub_for(CHAPTER_11)
    epub_path = tmp_path / "synthetic.epub"
    epub_path.write_bytes(epub_bytes)
    if options.get("split"):
        first = build_provider(root, origin_root, epub_bytes, roles=("narrator", "erick"))
        second = build_provider(root, origin_root, epub_bytes, roles=("jane",))
        providers = {"narrator": first, "erick": first, "jane": second}
    else:
        only = build_provider(root, origin_root, epub_bytes, restrictions_for=options.get("restrictions_for"))
        providers = {rid: only for rid in ALL_ROLES}
    assign = {rid: providers[rid] for rid in declared}
    project_id = build_chapter(root, CHAPTER_11, epub_bytes, assign=assign)
    monkeypatch.setattr(fresh, "_mp3", lambda master, target: target.write_bytes(master.read_bytes()))
    monkeypatch.setattr(fresh, "_decoded_duration", lambda path: wave_seconds(path))
    monkeypatch.setattr(fresh, "_loudness", lambda _: -20.0)
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(tmp_path / "ambient-unrelated"))
    return SimpleNamespace(root=root, epub=epub_path, id=project_id, tmp=tmp_path, origin_root=origin_root,
                           providers=providers, provider=providers["narrator"], declared=declared)


@pytest.fixture
def cast_prepared(cast_world):
    args = args_for(cast_world)
    return cast_world, args, fresh.prepare_chapter(args)


@pytest.fixture
def cast_ready(cast_prepared):
    world, args, state = cast_prepared
    args.coaching_bundle = write_bundle(world, state, args.output_dir)
    state = fresh.apply_bundle(args, state)
    assert state["status"] == "ready", state["holds"]
    return world, args, state


def edit_json(path, mutate):
    raw = json.loads(Path(path).read_bytes())
    mutate(raw)
    Path(path).write_bytes(json.dumps(raw).encode())


def restage_provider(world, mutate, provider=None):
    """Root re-stages a provider with an edit: staged and origin stay identical and the target declarations follow."""
    provider = provider or world.provider
    for folder in (provider.base, provider.origin):
        edit_json(folder / "project.json", mutate)
    digest = sha((provider.base / "project.json").read_bytes())

    def declarations(raw):
        for entry in raw["metadata"]["existing_book_cast_sources"].values():
            if entry["project_id"] == provider.id:
                entry["project_sha256"] = entry["origin_project_sha256"] = digest
    edit_project(world, declarations)
    provider.sha = digest


def rewrite_cast(world, mutate_entry=None, mutate_auth=None, *, rebind=True):
    def mutate(raw):
        entries = raw["metadata"]["existing_book_cast_sources"]
        if mutate_entry:
            mutate_entry(entries)
        auth = raw["metadata"]["owner_review_authorization"]
        if rebind:
            auth["role_voice_bindings_sha256"] = binding_sha(entries)
        if mutate_auth:
            mutate_auth(auth)
    edit_project(world, mutate)


def test_retained_cast_prepares_with_separate_source_and_provider_without_any_target_approval(cast_world):
    world = cast_world
    target_before, origin_before = tree_hashes(world.root / world.id), tree_hashes(world.provider.origin)
    provider_before = tree_hashes(world.provider.base)
    args = args_for(world)
    state = fresh.prepare_chapter(args)
    out = args.output_dir
    assert state["status"] == "awaiting_coaching" and state["holds"] == []
    for flag, value in fresh.REVIEW_FLAGS.items():
        assert state[flag] == value, flag
    assert {r["mode"] for r in state["roles"].values()} == {fresh.MODE_EXISTING}
    assert state["cast_reuse"]["target_approvals_created"] == 0
    assert state["cast_reuse"]["role_fit_approval"] == "not_established"
    assert state["cast_reuse"]["human_listening"] == "pending"
    assert state["owner_review"]["schema"] == "fresh-chapter-owner-review-v1" and state["owner_review"]["authorized"] is True
    # Source project and provider are separate everywhere; the prose stays the target chapter.
    assert all(l["project_id"] == world.id and l["provider_project_id"] == world.provider.id for l in state["lines"])
    assert set(state["providers"]) == {world.provider.id}
    provider = state["providers"][world.provider.id]
    assert set(provider["roles"]) == set(ALL_ROLES)
    assert provider["origin_bytes_identical"] is True and provider["project_sha256"] == world.provider.sha
    # No approval was manufactured, copied or consulted on the target; no human event exists there.
    target = json.loads((world.root / world.id / "project.json").read_bytes())
    assert target["voice_lab"]["casting"]["roles"] == {} and target["voice_lab"]["casting"]["events"] == {}
    for rid, role in state["roles"].items():
        assert role["casting"] is None and role["lineage"]["target_approval_created"] is False
        assert role["lineage"]["role_fit_approval"] == "not_established"
        assert role["provider_project_id"] == world.provider.id
        use = role["lineage"]["evidence"]["source_use"]
        assert use["eligible_recorded_source"] is True and use["clone_ready"] is True and use["technical_qa"] == "pass"
        assert "not voice-quality or role-fit approval" in use["meaning"]
    # The draft clone keeps its ORIGINAL context positive human selection; non-draft roles have none and need none.
    assert state["roles"]["erick"]["lineage"]["evidence"]["original_selection"]["event"]["origin"] == "human"
    assert state["roles"]["jane"]["lineage"]["evidence"]["original_selection"] is None
    assert state["roles"]["narrator"]["lineage"]["evidence"]["original_selection"] is None
    original = json.loads((world.provider.base / "project.json").read_bytes())
    assert set(original["voice_lab"]["casting"]["roles"]) == {"erick"}      # only the draft had a human selection
    # Exact provider context copied byte for byte beside the retained target project.
    isolated = out / "projects" / world.provider.id
    assert (out / "projects" / world.id / "project.json").is_file()
    assert (isolated / "project.json").read_bytes() == (world.provider.base / "project.json").read_bytes()
    assert provider["tree"]
    assert all(sha((isolated / rel).read_bytes()) == digest for rel, digest in provider["tree"].items())
    assert "voice_lab/discovery/acquisitions" in " ".join(provider["tree"])
    assert any("heard" in rel for rel in provider["tree"])
    # Nothing outside the output changed.
    assert tree_hashes(world.root / world.id) == target_before and tree_hashes(world.provider.origin) == origin_before
    assert tree_hashes(world.provider.base) == provider_before
    assert all(not s.takes and not s.selected_take_id for s in load_segments(world.id))


def test_generation_requests_in_the_provider_context_with_the_real_request_sha_and_labels_both_projects(cast_ready):
    world, args, state = cast_ready
    before = {"target": tree_hashes(world.root / world.id), "provider": tree_hashes(world.provider.base),
              "origin": tree_hashes(world.provider.origin)}
    result = fresh.generate_chapter(args, state, FakeWorker, RENDER)
    out = args.output_dir
    assert result["status"] == "generated" and len(FakeWorker.instances) == 1
    sent = all_calls()
    assert len(sent) == SPEECH and {c["project_id"] for c in sent} == {world.provider.id}
    by_text = {c["text"]: c for c in sent}
    for line in state["lines"]:
        record = json.loads((out / "receipts" / f"{line['utterance_id']}.json").read_bytes())
        instruction = record["instruction"]
        real = pilot.expected_request_sha(world.provider.id, line["voice"], line["text"], instruction)
        assert record["request_sha256"] == real
        assert real != pilot.expected_request_sha(world.id, line["voice"], line["text"], instruction)
        raw = json.loads((out / "samples" / f"{line['utterance_id']}.receipt.json").read_bytes())
        assert raw["request_sha256"] == real and by_text[line["text"]]["identity_id"] == line["voice"]["id"]
        assert record["project_id"] == record["source_project_id"] == world.id          # source line stays the target
        assert record["provider_project_id"] == record["request_project_id"] == world.provider.id
        assert record["cast_mode"] == fresh.MODE_EXISTING and record["cast_reuse"]["target_approvals_created"] == 0
        assert record["clone_binding"]["existing_cast"]["provider_project_id"] == world.provider.id
        assert record["clone_binding"]["casting_approval"] is None
        assert line["text_sha256"] == record["text_sha256"] and line["source"]["source_revision_id"]
    receipt = chapter_receipt(args)
    assert receipt["cast_reuse"]["mode"] == fresh.MODE_EXISTING and receipt["approvals_created"] is False
    assert receipt["cast_reuse"]["existing_book_cast_roles"] == sorted(ALL_ROLES)
    assert receipt["cast_reuse"]["owner_review_authorization"]["scope"] == "private_chapter_review"
    for row in (r for r in receipt["source_to_audio"] if r["kind"] == "speech"):
        assert row["source_project_id"] == world.id and row["provider_project_id"] == world.provider.id
        cast = row["cloned_reference"]["existing_cast"]
        assert cast["origin_project_sha256"] == world.provider.sha and cast["target_approval_created"] is False
        assert cast["source_use"]["eligible_recorded_source"] is True
    for flag, value in fresh.REVIEW_FLAGS.items():
        assert receipt[flag] == value, flag
    assert json.loads((out / "forwarding-audit.json").read_bytes())["cast_reuse"]["mode"] == fresh.MODE_EXISTING
    assert json.loads((out / "result.json").read_bytes())["cast_reuse"]["target_approvals_created"] == 0
    assert tree_hashes(world.root / world.id) == before["target"]          # no target approval/event/Take written
    assert tree_hashes(world.provider.base) == before["provider"]
    assert tree_hashes(world.provider.origin) == before["origin"]
    assert all(not s.takes and not s.selected_take_id for s in load_segments(world.id))


@pytest.mark.parametrize("cast_world", [{"split": True}], indirect=True, ids=["two-providers"])
def test_each_role_is_requested_in_its_own_provider_context(cast_world):
    world = cast_world
    args = args_for(world)
    state = fresh.prepare_chapter(args)
    assert set(state["providers"]) == {world.providers["narrator"].id, world.providers["jane"].id}
    args.coaching_bundle = write_bundle(world, state, args.output_dir)
    fresh.generate_chapter(args, fresh.apply_bundle(args, state), FakeWorker, RENDER)
    texts = {l["text"]: l for l in state["lines"]}
    for call in all_calls():
        assert call["project_id"] == world.providers[texts[call["text"]]["role_id"]].id
    assert {c["project_id"] for c in all_calls()} == {p.id for p in world.providers.values()}


@pytest.mark.parametrize("cast_world", [{"declared": ("narrator", "jane")}], indirect=True, ids=["mixed-modes"])
def test_target_approval_mode_and_retained_cast_can_coexist_per_role(cast_world):
    world = cast_world
    args = args_for(world)
    state = fresh.prepare_chapter(args)
    assert {r: v["mode"] for r, v in state["roles"].items()} == {
        "narrator": fresh.MODE_EXISTING, "jane": fresh.MODE_EXISTING, "erick": fresh.MODE_TARGET}
    assert state["roles"]["erick"]["casting"]["event"]["origin"] == "human" and state["roles"]["erick"]["lineage"] is None
    args.coaching_bundle = write_bundle(world, state, args.output_dir)
    fresh.generate_chapter(args, fresh.apply_bundle(args, state), FakeWorker, RENDER)
    projects = {c["identity_id"]: c["project_id"] for c in all_calls()}
    assert projects["voice_erick"] == world.id
    assert projects["voice_jane"] == projects["voice_narrator"] == world.provider.id


# --- owner authorization and declarations are exact


@pytest.mark.parametrize("break_it", [
    "missing", "schema", "unauthorized", "scope", "book", "chapter", "author", "no_instruction", "binding",
    "quality_claimed"])
def test_owner_review_authorization_must_exist_and_match_exactly(cast_world, break_it):
    def mutate(raw):
        auth = raw["metadata"]["owner_review_authorization"]
        if break_it == "missing":
            del raw["metadata"]["owner_review_authorization"]
        elif break_it == "schema":
            auth["schema"] = "fresh-chapter-owner-review-v0"
        elif break_it == "unauthorized":
            auth["authorized"] = False
        elif break_it == "scope":
            auth["scope"] = "production"
        elif break_it == "book":
            auth["book_sha256"] = "d" * 64
        elif break_it == "chapter":
            auth["chapter_number"] = 12
        elif break_it == "author":
            auth["author"] = "automation"
        elif break_it == "no_instruction":
            auth["instruction"] = " "
        elif break_it == "binding":
            auth["role_voice_bindings_sha256"] = "e" * 64
        else:
            auth["quality_approval_created"] = True
    edit_project(cast_world, mutate)
    with pytest.raises(fresh.FreshChapterError, match="Owner review authorization|owner_review_authorization"):
        fresh.prepare_chapter(args_for(cast_world))
    assert not (cast_world.tmp / "out" / "state.json").exists()


@pytest.mark.parametrize("break_it", [
    "other_book", "other_role", "other_voice", "other_conditioning", "target_approval", "provider_is_target",
    "unrecorded_hash", "hash_mismatch", "no_origin", "unknown_field"])
def test_declaration_drift_across_book_role_identity_or_hash_is_refused(cast_world, break_it):
    def mutate(entries):
        entry = entries["jane"]
        if break_it == "other_book":
            entry["book_sha256"] = "d" * 64
        elif break_it == "other_role":
            entry["role_id"] = "erick"
        elif break_it == "other_voice":
            entry["voice_id"] = "voice_erick"
        elif break_it == "other_conditioning":
            entry["prompt_sha256"] = "c" * 64
        elif break_it == "target_approval":
            entry["target_approval_created"] = True
        elif break_it == "provider_is_target":
            entry["project_id"] = cast_world.id
        elif break_it == "unrecorded_hash":
            entry["project_sha256"] = "b" * 63
        elif break_it == "hash_mismatch":
            entry["origin_project_sha256"] = "b" * 64
        elif break_it == "no_origin":
            entry["origin_source_root"] = " "
        else:
            entry["invented"] = True
    rewrite_cast(cast_world, mutate)
    with pytest.raises(fresh.FreshChapterError, match="existing-cast declaration refused"):
        fresh.prepare_chapter(args_for(cast_world))
    assert not (cast_world.tmp / "out" / "state.json").exists()


def test_a_provider_switch_without_re_authorization_is_refused(cast_world):
    other = build_provider(cast_world.root, cast_world.origin_root, epub_for(CHAPTER_11))
    rewrite_cast(cast_world, lambda e: e["jane"].update(project_id=other.id, project_sha256=other.sha,
                                                         origin_project_sha256=other.sha), rebind=False)
    with pytest.raises(fresh.FreshChapterError, match="Owner review authorization.*bindings"):
        fresh.prepare_chapter(args_for(cast_world))


# --- the original provider context is verified, not trusted


def test_provider_bound_to_another_book_is_refused(cast_world):
    restage_provider(cast_world, lambda raw: raw["source_revisions"][0].update(sha256="d" * 64))
    with pytest.raises(fresh.FreshChapterError, match="not bound to this book"):
        fresh.prepare_chapter(args_for(cast_world))


@pytest.mark.parametrize("change", ["role_name", "role_missing", "clone_model", "clone_unlocked", "preset", "generated"])
def test_provider_role_must_be_the_same_canonical_role_and_exact_clone(cast_world, change):
    def mutate(raw):
        role = next(r for r in raw["roles"] if r["id"] == "jane")
        if change == "role_name":
            role["name"] = "Jane Flatt"
        elif change == "role_missing":
            role["id"] = "someone_else"
        elif change == "clone_model":
            role["voice_identity"]["model"] = "another-model"
        elif change == "clone_unlocked":
            role["voice_identity"]["status"] = "REVIEW"
        elif change == "preset":
            role["voice_identity"].update(mechanism="builtin", builtin_speaker="preset")
        else:
            role["voice_identity"]["provenance"]["origin"] = "generated_original_voice"
    restage_provider(cast_world, mutate)
    with pytest.raises(fresh.FreshChapterError):
        fresh.prepare_chapter(args_for(cast_world))
    assert not (cast_world.tmp / "out" / "state.json").exists()


def test_staged_provider_bytes_must_match_the_recorded_hash(cast_world):
    edit_json(cast_world.provider.base / "project.json", lambda raw: raw.update(description="staged copy drifted"))
    with pytest.raises(fresh.FreshChapterError, match="staged original context differs from its recorded hash"):
        fresh.prepare_chapter(args_for(cast_world))


def test_origin_must_still_be_the_staged_project_but_notification_counters_may_move(cast_world):
    world = cast_world

    def counters(raw):
        raw["voice_lab"]["revision"] += 9
        raw["voice_lab"]["casting"]["notifications"] = {"complete": {"next_retry_at": "later"}}
    edit_json(world.provider.origin / "project.json", counters)        # live Studio bookkeeping at the origin
    state = fresh.prepare_chapter(args_for(world))
    assert state["providers"][world.provider.id]["origin_bytes_identical"] is False
    edit_json(world.provider.origin / "project.json", lambda raw: raw["roles"][1].update(name="Renamed"))
    with pytest.raises(fresh.FreshChapterError, match="origin book-role context changed"):
        fresh.prepare_chapter(args_for(world, output_dir=world.tmp / "out2"))
    shutil.rmtree(world.provider.origin)
    with pytest.raises(fresh.FreshChapterError, match="origin book-role context is unavailable"):
        fresh.prepare_chapter(args_for(world, output_dir=world.tmp / "out3"))


def test_a_withdrawn_draft_selection_in_the_original_context_holds(cast_world):
    restage_provider(cast_world, lambda raw: raw["voice_lab"]["casting"]["events"]["event_erick"].update(
        selection_confirmed=False))
    with pytest.raises(fresh.FreshChapterError, match="positive human selection"):
        fresh.prepare_chapter(args_for(cast_world))
    assert not (cast_world.tmp / "out" / "state.json").exists()


def test_a_withdrawn_draft_selection_after_preparation_blocks_generation(cast_ready):
    world, args, state = cast_ready
    edit_json(world.provider.origin / "project.json", lambda raw: raw["voice_lab"]["casting"]["events"][
        "event_erick"].update(selection_confirmed=False))
    with pytest.raises(fresh.FreshChapterError, match="origin book-role context changed|positive human selection"):
        fresh.generate_chapter(args, state, FakeWorker, RENDER)
    assert FakeWorker.instances == []


def test_non_draft_roles_need_no_human_event_but_still_need_eligible_source_use(cast_world):
    world = cast_world
    original = json.loads((world.provider.base / "project.json").read_bytes())
    casting = original["voice_lab"]["casting"]
    assert "jane" not in casting["roles"] and "narrator" not in casting["roles"]
    assert not casting["events"].get("event_jane") and not casting["events"].get("event_narrator")
    assert fresh.prepare_chapter(args_for(world))["roles"]["jane"]["mode"] == fresh.MODE_EXISTING


@pytest.mark.parametrize("cast_world", [{"restrictions_for": "jane"}], indirect=True, ids=["restricted-source"])
def test_ineligible_source_use_is_refused_even_with_owner_authorization(cast_world):
    with pytest.raises(fresh.FreshChapterError, match="source-use is not proven"):
        fresh.prepare_chapter(args_for(cast_world))
    assert not (cast_world.tmp / "out" / "state.json").exists()


def test_changed_source_receipt_or_clone_assets_are_refused(cast_world):
    receipt = next((cast_world.provider.base / "voice_lab" / "discovery" / "acquisitions").glob("*/acquisition.json"))
    original = receipt.read_bytes()
    receipt.write_bytes(original + b" ")
    with pytest.raises(fresh.FreshChapterError):
        fresh.prepare_chapter(args_for(cast_world))
    receipt.write_bytes(original)
    clone = cast_world.root / cast_world.id / "casting" / "voices" / "voice_jane" / "conditioning.npz"
    clone.write_bytes(b"changed")
    with pytest.raises(fresh.FreshChapterError, match="hash-mismatched"):
        fresh.prepare_chapter(args_for(cast_world, output_dir=cast_world.tmp / "out2"))


# --- drift after preparation, checked before apply/generate and on retry


@pytest.mark.parametrize("drift", [
    "staged_provider_project", "staged_provider_asset", "staged_provider_receipt", "isolated_provider_asset",
    "isolated_provider_project", "origin_semantic", "origin_missing", "origin_asset", "target_declaration",
    "target_authorization", "target_clone_asset", "target_receipt"])
def test_provider_drift_is_refused_before_apply_and_before_a_worker_starts(cast_prepared, drift):
    world, args, state = cast_prepared
    out, provider = args.output_dir, world.provider
    args.coaching_bundle = write_bundle(world, state, out)
    target = world.root / world.id
    if drift == "staged_provider_project":
        edit_json(provider.base / "project.json", lambda raw: raw.update(description="drift"))
    elif drift == "staged_provider_asset":
        (provider.base / "casting" / "voices" / "voice_jane" / "reference.wav").write_bytes(b"changed")
    elif drift == "staged_provider_receipt":
        next((provider.base / "voice_lab" / "discovery" / "acquisitions").glob("*/acquisition.json")).write_bytes(b"{}")
    elif drift == "isolated_provider_asset":
        (out / "projects" / provider.id / "casting" / "voices" / "voice_narrator" / "reference.wav").write_bytes(b"changed")
    elif drift == "isolated_provider_project":
        edit_json(out / "projects" / provider.id / "project.json", lambda raw: raw["voice_lab"]["casting"]["events"][
            "event_erick"].update(selection_confirmed=False))
    elif drift == "origin_semantic":
        edit_json(provider.origin / "project.json", lambda raw: raw["roles"][1].update(name="Renamed"))
    elif drift == "origin_missing":
        shutil.rmtree(provider.origin)
    elif drift == "origin_asset":
        (provider.origin / "casting" / "voices" / "voice_jane" / "conditioning.npz").write_bytes(b"changed")
    elif drift == "target_declaration":
        edit_project(world, lambda raw: raw["metadata"]["existing_book_cast_sources"]["jane"].update(project_id="another"))
    elif drift == "target_authorization":
        edit_project(world, lambda raw: raw["metadata"]["owner_review_authorization"].update(authorized=False))
    elif drift == "target_clone_asset":
        (target / "casting" / "voices" / "voice_narrator" / "reference.wav").write_bytes(b"changed")
    else:
        next((target / "voice_lab" / "discovery" / "acquisitions").glob("*/acquisition.json")).write_bytes(b"{}")
    with pytest.raises(fresh.FreshChapterError):
        fresh.apply_bundle(args, state)
    assert not (out / "applied.json").exists() and FakeWorker.instances == []


@pytest.mark.parametrize("drift", ["staged_provider_project", "origin_semantic", "isolated_provider_asset",
                                   "target_declaration", "staged_provider_receipt"])
def test_provider_drift_after_apply_is_refused_before_a_worker(cast_ready, drift):
    world, args, state = cast_ready
    out, provider = args.output_dir, world.provider
    if drift == "staged_provider_project":
        edit_json(provider.base / "project.json", lambda raw: raw.update(description="drift"))
    elif drift == "origin_semantic":
        edit_json(provider.origin / "project.json", lambda raw: raw["roles"][2].update(name="Renamed"))
    elif drift == "isolated_provider_asset":
        (out / "projects" / provider.id / "casting" / "voices" / "voice_erick" / "conditioning.npz").write_bytes(b"x")
    elif drift == "target_declaration":
        edit_project(world, lambda raw: raw["metadata"]["existing_book_cast_sources"]["erick"].update(
            origin_source_root="elsewhere"))
    else:
        next((provider.base / "voice_lab" / "discovery" / "acquisitions").glob("*/acquisition.json")).write_bytes(b"{}")
    with pytest.raises(fresh.FreshChapterError):
        fresh.generate_chapter(args, state, FakeWorker, RENDER)
    assert FakeWorker.instances == [] and not (out / "samples").exists()


def test_origin_notification_counters_moving_after_preparation_do_not_block_generation(cast_ready):
    world, args, state = cast_ready

    def counters(raw):
        raw["voice_lab"]["revision"] += 5
        raw["voice_lab"]["casting"]["notifications"] = {"complete": {"next_retry_at": "later"}}
    edit_json(world.provider.origin / "project.json", counters)
    assert fresh.generate_chapter(args, state, FakeWorker, RENDER)["status"] == "generated"


def test_completed_retained_cast_run_retries_with_zero_workers_and_unchanged_bytes(cast_ready):
    world, args, state = cast_ready
    fresh.generate_chapter(args, state, FakeWorker, RENDER)
    before = tree_hashes(args.output_dir)
    FakeWorker.instances = []
    again = fresh.generate_chapter(args, fresh.load_state(args.output_dir), FakeWorker, RENDER)
    assert again["status"] == "generated" and FakeWorker.instances == [] and all_calls() == []
    assert again["audit"]["workers_started_this_run"] == 0 and again["audit"]["new_requests_this_run"] == 0
    assert {k: v for k, v in tree_hashes(args.output_dir).items() if not k.startswith("source-checks/")} == \
        {k: v for k, v in before.items() if not k.startswith("source-checks/")}
    assert chapter_receipt(args)["cast_reuse"]["mode"] == fresh.MODE_EXISTING
    edit_json(world.provider.base / "project.json", lambda raw: raw.update(description="drift after completion"))
    with pytest.raises(fresh.FreshChapterError):                      # a retry re-checks the provider, not only bytes
        fresh.generate_chapter(args, fresh.load_state(args.output_dir), FakeWorker, RENDER)


def test_partial_retained_cast_run_resumes_only_missing_samples_in_the_provider_context(cast_ready):
    world, args, state = cast_ready
    calls = {"n": 0}

    def flaky(worker, *a, **k):
        calls["n"] += 1
        if calls["n"] == 5:
            raise RuntimeError("worker dropped")
        return RENDER(worker, *a, **k)
    with pytest.raises(RuntimeError, match="worker dropped"):
        fresh.generate_chapter(args, state, FakeWorker, flaky)
    first = all_calls()
    FakeWorker.instances = []
    fresh.generate_chapter(args, fresh.load_state(args.output_dir), FakeWorker, RENDER)
    assert len(first) == 4 and len(all_calls()) == SPEECH - 4 and len(FakeWorker.instances) == 1
    assert {c["project_id"] for c in first + all_calls()} == {world.provider.id}


def test_a_render_in_another_provider_context_cannot_pass_verification(cast_ready, monkeypatch):
    world, args, state = cast_ready
    use_projects_root(monkeypatch, world.tmp)          # the pipeline restored the ambient root; build beside the others
    other = build_provider(world.root, world.origin_root, epub_for(CHAPTER_11))

    def wrong_context(worker, project_id, voice, line, path, instruction=None):
        return RENDER(worker, other.id, voice, {**line, "project_id": other.id}, path, instruction)
    with pytest.raises(fresh.FreshChapterError, match="request differs from the frozen identity"):
        fresh.generate_chapter(args, state, FakeWorker, wrong_context)
    assert not (args.output_dir / "chapter").exists()
