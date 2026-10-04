"""Existing exact-selection owner over isolated chapter forks; no model/network."""
import pytest

from dubbing_service.app.audiobook import casting_review as casting, voice_lab as lab
from dubbing_service.app.audiobook import voice_lab_chapter as chapter, voice_lab_library as shelf
from dubbing_service.app.audiobook.fingerprint import file_sha256, voice_revision_for_role
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.storage import voices as library
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from dubbing_service.app.core.tts_operations import try_register_running_batch, unregister_running_batch
from tests.audiobook_helpers import make_tone_wav
from tests.test_voice_lab_chapter import source as chapter_source


@pytest.fixture
def imported(chapter_source, monkeypatch, tmp_path):
    original, root = chapter_source
    voices_root = tmp_path / "library" / "voices"
    voices_root.mkdir(parents=True)
    monkeypatch.setattr(library, "voices_audio_dir", lambda: voices_root)
    monkeypatch.setattr(library, "_voices_file_path", lambda: voices_root.parent / "voices.json")
    reference = make_tone_wav(root / original.id / "casting/voices/review_fixture/reference.wav", seconds=.5)
    prompt = reference.with_suffix(".npz")
    prompt.write_bytes(b"ISOLATED TEST CONDITIONING; NEVER MODEL INPUT")
    sample = make_tone_wav(root / original.id / "voices/sample.wav", seconds=.5)
    identity = CharacterVoiceIdentity(id="review_fixture", role_id="alice", mechanism="qwen_reference",
        dossier_revision="fixture", casting_profile_revision="fixture", design_instruction="Isolated fixture.",
        reference_audio=reference.relative_to(root / original.id).as_posix(), reference_audio_sha256=file_sha256(reference),
        prompt_artifact=prompt.relative_to(root / original.id).as_posix(), prompt_sha256=file_sha256(prompt),
        reference_text="Isolated source sample.", model="fixture/base", provenance={"origin": "generated_original_voice"})
    version = voice_revision_for_role(original.roles[1].model_copy(update={"voice_identity": identity}))
    key = "v_" + lab.digest([identity.id, version])[:28]
    record = {"key": key, "voice_id": identity.id, "version": version, "identity": identity.model_dump(mode="json"),
        "name": "Isolated review fixture", "kind": "original_inspired", "backend": "qwen3tts",
        "audio_file": sample.relative_to(root / original.id).as_posix(), "audio_sha256": file_sha256(sample),
        "technical_qa": "passed", "duration_seconds": .5}
    original.voice_lab.voices[key] = record
    save_project(original)
    published = shelf.publish_version(original.id, key)
    target_id = chapter.fork_chapter(original.id, chapter_id="chap1", name="Isolated target", request_id="fork-review")["project_id"]
    shelf.import_version(target_id, published["library_voice_id"], version)
    target = load_project(target_id)
    role = next(r for r in target.roles if r.id == "alice")
    arguments = {"role_id": "alice", "key": key, "library_voice_id": published["library_voice_id"], "version": version,
        "audio_sha256": record["audio_sha256"], "expected_identity_id": voice_revision_for_role(role), "request_id": "review-one"}
    return root, original.id, target_id, arguments


def prepare(imported, **changes):
    return casting.prepare_imported_review(imported[2], **{**imported[3], **changes})


def test_preparation_is_metadata_only_pending_and_retry_stable(imported):
    root, original, target_id, args = imported
    source_before = (root / original / "project.json").read_bytes()
    library_before = library._voices_file_path().read_bytes()
    segments_before = (root / target_id / "segments.json").read_bytes()
    result = prepare(imported)
    saved_before_retry = (root / target_id / "project.json").read_bytes()
    assert prepare(imported) == result
    assert (root / target_id / "project.json").read_bytes() == saved_before_retry
    target = load_project(target_id)
    state = target.voice_lab.casting
    assert not state.get("enabled") and not state["events"] and not state["intents"] and not state["productions"]
    assert not state["roles"]["alice"]["assignment"] and not casting.approved(target, "alice")
    assert all(r.voice_identity is None for r in target.roles)
    assert not target.voice_lab.preferences and not result["approved"] and not result["assigned"]
    assert not result["new_audio_generated"]
    assert library._voices_file_path().read_bytes() == library_before
    assert (root / original / "project.json").read_bytes() == source_before
    assert (root / target_id / "segments.json").read_bytes() == segments_before
    with pytest.raises(chapter.ChapterRenditionError, match="locked, ready"):
        chapter._bound_state(target_id, "chap1")


def test_existing_feedback_commits_exact_selection_without_stars_or_generation(imported):
    receipt = prepare(imported)
    payload = {k: v for k, v in receipt.items() if k in {
        "role_id", "candidate_id", "key", "audio_sha256", "round_revision", "assignment_revision"}}
    payload.update(event_id="explicit-use", action="use", selection_confirmed=True, raw_direction="Explicit authorized reuse; no new listening claim.")
    result = casting.feedback(imported[2], payload)
    assert casting.feedback(imported[2], payload) == result
    target = load_project(imported[2])
    event = target.voice_lab.casting["events"]["explicit-use"]
    assert casting.approved(target, "alice") and event["origin"] == "human"
    assert event.get("stars") is None and event["selection_confirmed"] is True
    assert event["audio_sha256"] == imported[3]["audio_sha256"]
    assert not target.voice_lab.casting.get("enabled") and not target.voice_lab.casting["intents"]
    assert not target.voice_lab.preferences and not target.voice_lab.preference_history
    assert next(r for r in target.roles if r.id == "alice").voice_identity.id == "review_fixture"


@pytest.mark.parametrize("change", ["strict", "source", "audio", "reference", "prompt", "cas", "version", "outside_role"])
def test_stale_or_unverified_review_fails_without_a_round(imported, change):
    root, _, target_id, args = imported
    extra = {}
    if change == "strict":
        target = load_project(target_id); target.voice_lab.casting["enabled"] = True; save_project(target)
    elif change == "source":
        rows = load_segments(target_id); rows[0].text = "Changed source"; save_segments(target_id, rows)
    elif change == "audio":
        record = load_project(target_id).voice_lab.voices[args["key"]]
        (root / target_id / record["audio_file"]).write_bytes(b"corrupt")
    elif change in {"reference", "prompt"}:
        record = load_project(target_id).voice_lab.voices[args["key"]]
        field = "reference_audio" if change == "reference" else "prompt_artifact"
        (root / target_id / record["identity"][field]).write_bytes(b"corrupt")
    elif change == "cas": extra["expected_identity_id"] = "stale-identity"
    elif change == "version": extra["version"] = "unavailable-version"
    elif change == "outside_role": extra["role_id"] = "absent-role"
    before = (root / target_id / "project.json").read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        prepare(imported, **extra)
    assert (root / target_id / "project.json").read_bytes() == before
    assert not load_project(target_id).voice_lab.casting["roles"]


def test_request_id_cannot_be_rebound_to_another_decision(imported):
    prepare(imported)
    before = load_project(imported[2]).model_dump(mode="json")
    with pytest.raises(casting.CastingError, match="different imported review"):
        prepare(imported, expected_identity_id="changed-command")
    assert load_project(imported[2]).model_dump(mode="json") == before


def test_preparation_waits_for_render_lease(imported):
    lease = try_register_running_batch(imported[2])
    assert lease
    try:
        with pytest.raises(casting.CastingError, match="Rendering owns"):
            prepare(imported)
    finally:
        unregister_running_batch(lease)
    assert not load_project(imported[2]).voice_lab.casting["roles"]


def test_initial_review_does_not_replace_an_assigned_role(imported):
    args = imported[3]
    lab.assign(imported[2], args["key"], "alice", args["expected_identity_id"])
    before = load_project(imported[2]).model_dump(mode="json")
    with pytest.raises(casting.CastingError, match="unassigned chapter role"):
        prepare(imported, expected_identity_id="review_fixture")
    assert load_project(imported[2]).model_dump(mode="json") == before


def test_intervening_source_edit_cannot_create_a_stale_ready_round(imported, monkeypatch):
    original = chapter.source_scope
    def interleaved(*args, **kwargs):
        state = original(*args, **kwargs)
        rows = load_segments(imported[2])
        rows[0].text = "Changed between validation and locked review commit."
        save_segments(imported[2], rows)
        return state
    monkeypatch.setattr(chapter, "source_scope", interleaved)
    before = (imported[0] / imported[2] / "project.json").read_bytes()
    with pytest.raises(casting.CastingError, match="changed during review"):
        prepare(imported)
    assert (imported[0] / imported[2] / "project.json").read_bytes() == before
