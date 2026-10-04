"""Exact source-selected recorded-draft staging in disposable stores only.

The audio is a generated tone fixture. Only the pinned-recording policy seam is
replaced with equally exact fixture bindings; acquisition replay, human casting
validation, library storage and the worker authorization gate remain real. A
trap stops the worker before any model is loaded or inference is performed.
"""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook import recorded_audition_policy as draft
from dubbing_service.app.audiobook import reference_portability as portability
from dubbing_service.app.audiobook import voice_lab as lab, voice_lab_library as shelf
from dubbing_service.app.audiobook.fingerprint import voice_revision_for_role
from dubbing_service.app.audiobook.reference_delivery_fit import evaluate
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.storage import voices as library
from dubbing_service.app.storage.projects import load_project, save_project
from service import qwen_character_voice as voice
from tests.test_reference_portability import external, source, acquisition_fixture
from tests.test_voice_lab_reference_clone import Prompt, Tensor


def _human_selection(project_id, record, role_id, event_id, *, enabled=True):
    """Submit the normal explicit-selection operation against a ready test round."""
    project = load_project(project_id)
    role = next(item for item in project.roles if item.id == role_id)
    candidate_id = "candidate_" + event_id
    project.voice_lab.casting = {
        "enabled": enabled, "revision": 0, "events": {}, "roles": {role_id: {
            "round_revision": 1, "assignment_revision": 0, "assignment": None,
            "history": [], "rejections": [], "round": {
                "id": "round_" + event_id, "revision": 1,
                "status": "awaiting_review", "direction": {},
                "baseline_voice_revision": voice_revision_for_role(role),
                "candidates": [{"candidate_id": candidate_id, "status": "ready",
                    "key": record["key"], "audio_sha256": record["audio_sha256"],
                    "identity_id": record["voice_id"], "rating": None}]
            }
        }}
    }
    save_project(project)
    result = casting.feedback(project_id, {
        "event_id": event_id, "role_id": role_id, "candidate_id": candidate_id,
        "round_revision": 1, "assignment_revision": 0, "key": record["key"],
        "audio_sha256": record["audio_sha256"], "action": "use",
        "selection_confirmed": True, "raw_direction": ""
    })
    assert result["saved"] and not result["conflict"]
    return load_project(project_id).voice_lab.casting["events"][event_id]


@pytest.fixture
def reviewed_draft(external, monkeypatch):
    source_store, fixture, acquired, original_record, _ = external
    root, _, project, target, _ = source_store
    source_root = root / project.id
    receipt_path = source_root / "voice_lab/discovery" / (
        f"acquisitions/{acquired['acquisition_id']}/acquisition.json")
    receipt = json.loads(receipt_path.read_text("utf-8"))
    wave, rate = sf.read(str(source_root / "voice_lab/discovery" /
                             receipt["reference_path"]), dtype="float32")
    policy = {"schema": draft.SCHEMA, "draft_id": "fixture-exact-recorded-draft",
        "evidence_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
        "authorization_event": "fixture-only-recorded-draft",
        "source_delivery_fit": "unknown", "human_verified": False,
        "production_approved": False, "selection_requires_human_casting": True}
    verifications = []

    def verify_fixture(actual_policy, actual_wave, actual_rate, **binding):
        assert actual_policy == policy
        assert actual_rate == rate and np.array_equal(actual_wave, wave)
        assert binding == {"source": receipt["source"],
            "original_sha256": receipt["original_sha256"],
            "transcript": receipt["reference_text"],
            "start_seconds": receipt["start_seconds"],
            "end_seconds": receipt["end_seconds"]}
        verifications.append(binding)
        return copy.deepcopy(policy)

    monkeypatch.setattr(draft, "verify_policy", verify_fixture)
    receipt.update(audition_draft_policy=policy, production_approved=False,
        audition_draft_ready=True, source_fit_review_required=True)
    receipt["narration_delivery_fit"] = evaluate(wave, rate,
        source=receipt["source"], start_seconds=receipt["start_seconds"],
        end_seconds=receipt["end_seconds"], technical_status="pass",
        transcript=receipt["reference_text"], audio_assessment=None)
    assert receipt["narration_delivery_fit"]["delivery_decision"] == "review"
    assert receipt["narration_delivery_fit"]["selected_reference"] is False
    receipt_path.write_text(json.dumps(receipt), "utf-8")
    acquired = {**receipt, "acquisition_sha256": voice.sha256(receipt_path)}
    monkeypatch.setattr(voice, "safe_project_root", lambda pid: root / pid)
    record = copy.deepcopy(original_record)
    identity = record["identity"]
    server_receipt = fixture.runtime.import_reference(project.id, identity["id"],
        acquired["acquisition_id"], acquired["acquisition_sha256"])
    identity["provenance"] = copy.deepcopy(server_receipt["provenance"])
    prompt_path = source_root / identity["prompt_artifact"]
    prompt_path.unlink()
    prompt = Prompt(Tensor(np.ones((12, 16), dtype=np.int64)),
        Tensor(np.arange(128, dtype=np.float32)), False, True, identity["reference_text"])
    voice.save_prompt(prompt_path, prompt, {"schema": "qwen-original-prompt-v1",
        "identity_id": identity["id"], "origin": "eligible_reference_clone",
        "reference_audio_sha256": identity["reference_audio_sha256"],
        "acquisition_id": acquired["acquisition_id"],
        "acquisition_sha256": acquired["acquisition_sha256"],
        "audition_draft_policy": policy, "model": "fixture-base",
        "model_revision": "revision-1", "runtime_version": "0.1.1"})
    identity["prompt_sha256"] = voice.sha256(prompt_path)
    loaded = load_project(project.id)
    # Production-shaped owner metadata is confined to this disposable fixture;
    # casting.feedback itself creates the human event under test.
    loaded.voice_lab.scope = "production"
    assigned_identity = lab.identity_for_assignment(CharacterVoiceIdentity.model_validate(identity))
    record["version"] = voice_revision_for_role(loaded.roles[0].model_copy(
        update={"voice_identity": assigned_identity}))
    record["key"] = "v_" + lab.digest([record["voice_id"], record["version"]])[:28]
    record.update(kind="reference_clone", human_review="pending")
    loaded.voice_lab.voices = {record["key"]: record}
    save_project(loaded)
    event = _human_selection(project.id, record, "narrator", "fixture-source-use")
    assert event["origin"] == "human" and event["actor"] == "brian"
    assert draft.human_casting_context(source_root, identity["id"], acquired["acquisition_sha256"])
    published = shelf.publish_version(project.id, record["key"])
    return SimpleNamespace(root=root, source=project, target=target, fixture=fixture,
        record=record, acquired=acquired, policy=policy, published=published,
        event=event, verifications=verifications)


def _entry(fixture):
    row = next(item for item in library.load_voices()
               if item.id == fixture.published["library_voice_id"])
    return copy.deepcopy(row.voice_lab_versions[fixture.published["version"]])


def _prepare(fixture):
    entry = _entry(fixture)
    prompt = library.voices_audio_dir() / entry["assets"]["prompt_artifact"]["path"]
    return portability.prepare(entry, fixture.published["library_voice_id"],
        prompt.read_bytes(), library.voices_audio_dir(), fixture.root / fixture.source.id)


def _import(fixture):
    return shelf.import_version(fixture.target.id,
        fixture.published["library_voice_id"], fixture.published["version"])


def test_exact_source_selected_draft_imports_pending_without_inference(reviewed_draft):
    f = reviewed_draft
    source_before = (f.root / f.source.id / "project.json").read_bytes()
    registry_before = library._voices_file_path().read_bytes()
    target_roles = load_project(f.target.id).roles
    prompt_before = (f.root / f.source.id / f.record["identity"]["prompt_artifact"]).read_bytes()
    result = _import(f)
    assert result["imported"] and result["reference_portability"]["cached_prompt_reused"]
    target = load_project(f.target.id)
    imported = target.voice_lab.voices[f.record["key"]]
    assert imported["role_id"] is None and imported["human_review"] == "pending"
    assert target.roles == target_roles
    assert not target.voice_lab.casting.get("events") and not target.voice_lab.casting.get("roles")
    assert not target.voice_lab.preferences and not target.voice_lab.preference_history
    assert "rating" not in imported and "favorite" not in imported
    assert imported["identity"]["provenance"] == f.record["identity"]["provenance"]
    assert imported["identity"]["provenance"]["production_approved"] is False
    assert imported["identity"]["provenance"]["audition_draft_policy"] == f.policy
    assert (f.root / f.target.id / imported["identity"]["prompt_artifact"]).read_bytes() == prompt_before
    prefix = f"casting/voices/{f.record['voice_id']}/"
    for name in ("reference.json", "original.audio", "reference.wav"):
        assert (f.root / f.target.id / prefix / name).read_bytes() == (
            f.root / f.source.id / prefix / name).read_bytes()
    manifest_path = f.root / f.target.id / result["reference_portability"]["manifest_file"]
    marker = json.loads(manifest_path.read_text("utf-8"))["recorded_draft_review"]
    assert marker["schema"] == "recorded-draft-review-staging-v1"
    assert marker["target_selection_required"] is True
    assert marker["source_delivery_approval_inferred"] is False
    assert marker["portability_status"] == "guarded_review_material"
    assert marker["source_grant_sha256"] == lab.digest(marker["source_grant"])
    assert marker["source_grant"]["event_id"] == f.event["event_id"]
    assert marker["source_grant"]["key"] == f.record["key"]
    assert marker["source_grant"]["version"] == f.record["version"]
    assert (f.root / f.source.id / "project.json").read_bytes() == source_before
    assert library._voices_file_path().read_bytes() == registry_before
    with pytest.raises(ValueError, match="narration delivery evidence"):
        voice._eligible_acquisition(f.root / f.target.id,
            f.acquired["acquisition_id"], f.acquired["acquisition_sha256"])
    checked = voice._eligible_acquisition(f.root / f.target.id,
        f.acquired["acquisition_id"], f.acquired["acquisition_sha256"], allow_draft=True)[0]
    assert checked["production_approved"] is False
    assert checked["narration_delivery_fit"]["selected_reference"] is False
    assert len(f.verifications) >= 4 and not f.fixture.calls
    target_before = (f.root / f.target.id / "project.json").read_bytes()
    assert not _import(f)["imported"]
    assert (f.root / f.target.id / "project.json").read_bytes() == target_before


@pytest.mark.parametrize("mutation", ["absent", "revoked", "wrong_version", "wrong_sample",
    "wrong_actor", "favorite_only", "missing_source", "same_identity_other_version",
    "stale_role_conditioning"])
def test_draft_requires_current_exact_source_human_selection(reviewed_draft, mutation):
    f = reviewed_draft
    source_root = f.root / f.source.id
    project = load_project(f.source.id)
    item = project.voice_lab.casting["roles"]["narrator"]
    event = project.voice_lab.casting["events"][f.event["event_id"]]
    if mutation == "absent":
        project.voice_lab.casting["events"] = {}
    elif mutation == "revoked":
        item["assignment"].update(approved=False, rejected=True)
    elif mutation == "wrong_version":
        event["voice_version"] = "0" * 32
    elif mutation == "wrong_sample":
        event["audio_sha256"] = "0" * 64
    elif mutation == "wrong_actor":
        event["actor"] = "automatic"
    elif mutation == "favorite_only":
        project.voice_lab.casting = {}
        project.voice_lab.preferences[f.record["key"]] = {"favorite": True,
            "audio_sha256": f.record["audio_sha256"]}
    elif mutation == "same_identity_other_version":
        other = copy.deepcopy(f.record)
        other["version"] = "1" * 32
        other["key"] = "v_" + lab.digest([other["voice_id"], other["version"]])[:28]
        project.voice_lab.voices = {other["key"]: other}
        item["assignment"].update(key=other["key"], voice_revision=other["version"])
        event.update(key=other["key"], voice_version=other["version"])
    elif mutation == "stale_role_conditioning":
        project.roles[0].voice_identity.model_revision = "changed-source-conditioning"
    save_project(project)
    if mutation in {"same_identity_other_version", "stale_role_conditioning"}:
        assert draft.human_casting_context(source_root,
            f.record["voice_id"], f.acquired["acquisition_sha256"])
    if mutation == "missing_source":
        source_root.rename(f.root / (f.source.id + "-unavailable"))
    before = (f.root / f.target.id / "project.json").read_bytes()
    with pytest.raises((ValueError, OSError)):
        _import(f)
    assert (f.root / f.target.id / "project.json").read_bytes() == before
    assert not load_project(f.target.id).voice_lab.voices and not f.fixture.calls


@pytest.mark.parametrize("asset", ["original", "reference", "receipt", "server_receipt"])
def test_draft_actual_source_tamper_fails_before_target_metadata(reviewed_draft, asset):
    f = reviewed_draft
    relative = ({"original": "voice_lab/discovery/" + f.acquired["original_path"],
        "reference": "voice_lab/discovery/" + f.acquired["reference_path"],
        "receipt": f"voice_lab/discovery/acquisitions/{f.acquired['acquisition_id']}/acquisition.json",
        "server_receipt": f"casting/voices/{f.record['voice_id']}/reference.json"})[asset]
    (f.root / f.source.id / relative).write_bytes(b"tampered exact fixture evidence")
    before = (f.root / f.target.id / "project.json").read_bytes()
    with pytest.raises((ValueError, OSError)):
        _import(f)
    assert (f.root / f.target.id / "project.json").read_bytes() == before
    assert not load_project(f.target.id).voice_lab.voices and not f.fixture.calls


def test_current_source_grant_is_rechecked_between_prepare_and_install(reviewed_draft):
    f = reviewed_draft
    bundle = _prepare(f)
    project = load_project(f.source.id)
    # Still positive and exact, but a new current decision is not the prepared grant.
    new_event_id = "fixture-new-current-use"
    replacement = copy.deepcopy(project.voice_lab.casting["events"][f.event["event_id"]])
    replacement["event_id"] = new_event_id
    project.voice_lab.casting["events"][new_event_id] = replacement
    project.voice_lab.casting["roles"]["narrator"]["assignment"]["event_id"] = new_event_id
    save_project(project)
    assert draft.human_casting_context(f.root / f.source.id,
        f.record["voice_id"], f.acquired["acquisition_sha256"])
    before = {str(path.relative_to(f.root / f.target.id)): voice.sha256(path)
        for path in (f.root / f.target.id).rglob("*") if path.is_file()}
    with pytest.raises((ValueError, OSError)):
        portability.install(bundle, f.root / f.target.id,
            f.published["library_voice_id"], f.published["version"])
    after = {str(path.relative_to(f.root / f.target.id)): voice.sha256(path)
        for path in (f.root / f.target.id).rglob("*") if path.is_file()}
    assert after == before and not f.fixture.calls


@pytest.mark.parametrize("mismatch", ["library_voice_id", "version"])
def test_install_requires_the_exact_prepared_library_version(reviewed_draft, mismatch):
    f = reviewed_draft
    bundle = _prepare(f)
    arguments = {"library_voice_id": f.published["library_voice_id"],
        "version": f.published["version"]}
    arguments[mismatch] = "vl_unrelated_fixture" if mismatch == "library_voice_id" else "0" * 32
    before = {str(path.relative_to(f.root / f.target.id)): voice.sha256(path)
        for path in (f.root / f.target.id).rglob("*") if path.is_file()}
    with pytest.raises((ValueError, OSError)):
        portability.install(bundle, f.root / f.target.id, **arguments)
    after = {str(path.relative_to(f.root / f.target.id)): voice.sha256(path)
        for path in (f.root / f.target.id).rglob("*") if path.is_file()}
    assert after == before and not f.fixture.calls


def test_target_worker_requires_its_own_real_selection_before_model_loading(reviewed_draft, monkeypatch):
    f = reviewed_draft
    _import(f)
    imported = load_project(f.target.id).voice_lab.voices[f.record["key"]]
    identity = imported["identity"]
    reached = []
    class BeforeModel(RuntimeError):
        pass
    def trap(kind):
        reached.append(kind)
        raise BeforeModel("Validated target selection; stopped before model loading")
    runtime = voice.CharacterVoiceRuntime(f.root, trap)
    monkeypatch.setattr(runtime, "_model_metadata", lambda kind: {
        "model": "fixture-base", "model_revision": "revision-1", "runtime_version": "0.1.1"})
    def synthesize(current):
        runtime.synthesize(f.target.id, current["id"], text="New target chapter text.",
            language="English", **{name: current[name] for name in (
                "prompt_artifact", "prompt_sha256", "reference_audio", "reference_audio_sha256")})
    with pytest.raises(ValueError, match="human selection required"):
        synthesize(identity)
    assert reached == [] and not draft.human_casting_context(f.root / f.target.id,
        f.record["voice_id"], f.acquired["acquisition_sha256"])
    event = _human_selection(f.target.id, imported, "reader", "fixture-target-use", enabled=False)
    assert event["origin"] == "human" and event["selection_confirmed"] is True
    assert load_project(f.target.id).voice_lab.casting["enabled"] is False
    assert draft.human_casting_context(f.root / f.target.id,
        f.record["voice_id"], f.acquired["acquisition_sha256"])
    current = load_project(f.target.id).roles[0].voice_identity.model_dump(mode="json")
    with pytest.raises(BeforeModel):
        synthesize(current)
    assert reached == ["base"] and not f.fixture.calls
