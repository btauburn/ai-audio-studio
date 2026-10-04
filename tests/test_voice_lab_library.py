"""Real canonical storage, isolated roots, no GPU and no production projects."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import voice_lab as lab
from dubbing_service.app.audiobook import voice_lab_library as shelf
from dubbing_service.app.audiobook.fingerprint import file_sha256, voice_revision_for_role
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.models.project import Role, Segment, SegmentTake
from dubbing_service.app.models.audiobook import ChunkRecord
from dubbing_service.app.storage import voices as library
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import save_segments
from tests.audiobook_helpers import make_tone_wav, use_projects_root


@pytest.fixture
def source(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    voices_root = tmp_path / "library" / "voices"
    voices_root.mkdir(parents=True)
    monkeypatch.setattr(library, "voices_audio_dir", lambda: voices_root)
    monkeypatch.setattr(library, "_voices_file_path", lambda: voices_root.parent / "voices.json")
    p = create_project("Isolated library fixture")
    p.voice_lab.scope = "regression_fixture"
    p.roles = [Role(id="narrator", name="Narrator", tts_backend="qwen3tts_cv", qwen3_speaker="ryan")]
    save_project(p)
    audio = make_tone_wav(root / p.id / "audio/segments/sample.wav", seconds=.25)
    take = SegmentTake(id="sample-take", role_id="narrator", audio_file="audio/segments/sample.wav",
        audio_sha256=file_sha256(audio), engine_id="qwen3tts_customvoice",
        fingerprint_components={"voice_revision": voice_revision_for_role(p.roles[0]), "builtin_speaker": "ryan"},
        chunk_map=[ChunkRecord(index=0, text_start=0, text_end=5, audio_start_ms=0, audio_end_ms=250)])
    segment = Segment(id="utterance", index=0, role_id="narrator", text="Fixture.", takes=[take], selected_take_id=take.id)
    save_segments(p.id, [segment])
    target = create_project("Isolated target")
    target.roles = [Role(id="reader", name="Reader", tts_backend="qwen3tts_cv", qwen3_speaker="aiden")]
    save_project(target)
    return root, voices_root, p, target, audio


def first_record(p):
    return next(iter(lab.voice_records(load_project(p.id)).values()))


def test_builtin_publish_import_is_idempotent_exact_and_does_not_transfer_preferences_or_cast(source):
    root, voices_root, p, target, audio = source
    record = first_record(p)
    lab.save_preference(p.id, {"key": record["key"], "audio_sha256": record["audio_sha256"], "favorite": True, "rating": 5})
    original_source = (root / p.id / "project.json").read_bytes()
    original_segments = (root / p.id / "segments.json").read_bytes()
    original_roles = load_project(target.id).roles
    published = shelf.publish_version(p.id, record["key"])
    files = {str(path): file_sha256(path) for path in voices_root.rglob("*") if path.is_file()}
    again = shelf.publish_version(p.id, record["key"])
    assert published["published"] and not again["published"]
    assert files == {str(path): file_sha256(path) for path in voices_root.rglob("*") if path.is_file()}
    imported = shelf.import_version(target.id, published["library_voice_id"], published["version"])
    assert imported["imported"]
    stored = load_project(target.id)
    sample = stored.voice_lab.voices[record["key"]]
    assert file_sha256(root / target.id / sample["audio_file"]) == file_sha256(audio)
    assert sample["builtin_speaker"] == "ryan" and sample["role_id"] is None
    assert not stored.voice_lab.preferences and not stored.voice_lab.preference_history
    assert stored.roles == original_roles
    assert "favorite" not in sample and "rating" not in sample
    assert sample["origin_project_id"] == p.id and sample["origin_take_id"] == "sample-take"
    target_bytes = (root / target.id / "project.json").read_bytes()
    assert not shelf.import_version(target.id, published["library_voice_id"], published["version"])["imported"]
    assert (root / target.id / "project.json").read_bytes() == target_bytes
    assert (root / p.id / "project.json").read_bytes() == original_source
    assert (root / p.id / "segments.json").read_bytes() == original_segments
    listed = shelf.list_versions()[0]
    assert listed["versions"][0]["audio_sha256"] == record["audio_sha256"]
    assert "audio_file" not in str(listed) and "identity" not in listed
    assert library.load_voices()[0].audio_path is None


def add_reference_identity(source):
    root, _, p, _, _ = source
    reference = make_tone_wav(root / p.id / "casting/voices/voice_fixture/reference.wav", seconds=.1)
    prompt = reference.parent / "prompt.npz"
    prompt.write_bytes(b"isolated fixture prompt, never sent to a model")
    identity = CharacterVoiceIdentity(id="voice_fixture", role_id="narrator", mechanism="qwen_reference",
        dossier_revision="d", casting_profile_revision="p", design_instruction="An original warm voice",
        reference_audio=reference.relative_to(root / p.id).as_posix(), reference_audio_sha256=file_sha256(reference),
        prompt_artifact=prompt.relative_to(root / p.id).as_posix(), prompt_sha256=file_sha256(prompt),
        reference_text="Generated local conditioning fixture.", model="fixture/base", provenance={"origin": "generated_original_voice", "license": "fixture", "intended_use_eligible": True})
    record = first_record(p)
    record.update(voice_id=identity.id, version=voice_revision_for_role(p.roles[0].model_copy(update={"voice_identity": identity})))
    record["key"] = "v_" + lab.digest([record["voice_id"], record["version"]])[:28]
    record.update(identity=identity.model_dump(mode="json"), name="Original Cedar", kind="original_inspired", builtin_speaker=None,
                  inspiration={"performer": "Fixture reference", "performance": "Fixture", "source_urls": ["https://example.test/reference"]})
    loaded = load_project(p.id); loaded.voice_lab.voices[record["key"]] = record; save_project(loaded)
    return record


def test_qwen_conditioning_and_evidence_are_copied_without_changing_voice_identity(source):
    root, _, p, target, _ = source
    record = add_reference_identity(source)
    published = shelf.publish_version(p.id, record["key"])
    shelf.import_version(target.id, published["library_voice_id"], published["version"])
    imported = load_project(target.id).voice_lab.voices[record["key"]]
    assert imported["voice_id"] == record["voice_id"] and imported["version"] == record["version"]
    for field, hash_field in (("reference_audio", "reference_audio_sha256"), ("prompt_artifact", "prompt_sha256")):
        assert file_sha256(root / target.id / imported["identity"][field]) == record["identity"][hash_field]
        assert imported["identity"][field].startswith("casting/voices/voice_fixture/")
    assert imported["identity"]["provenance"] == record["identity"]["provenance"]
    assert imported["inspiration"] == record["inspiration"]
    assert not load_project(target.id).voice_lab.preferences


def test_tampered_source_and_library_audio_are_refused(source):
    _, voices_root, p, target, audio = source
    record = first_record(p)
    published = shelf.publish_version(p.id, record["key"])
    entry = library.load_voices()[0].voice_lab_versions[published["version"]]
    path = voices_root / entry["assets"]["sample"]["path"]
    path.write_bytes(b"tampered test fixture")
    with pytest.raises(shelf.VoiceLibraryError, match="bytes"):
        shelf.import_version(target.id, published["library_voice_id"], published["version"])
    assert not load_project(target.id).voice_lab.voices
    audio.write_bytes(b"tampered source fixture")
    with pytest.raises(shelf.VoiceLibraryError, match="bytes"):
        shelf.publish_version(p.id, record["key"])


@pytest.mark.parametrize("relative", ["../outside.wav", "C:/outside.wav", "/outside.wav", "audio/../outside.wav", "sample.wav:alternate"])
def test_source_traversal_is_refused(source, relative):
    _, _, p, _, _ = source
    record = first_record(p); record["audio_file"] = relative
    project = load_project(p.id); project.voice_lab.voices[record["key"]] = record; save_project(project)
    with pytest.raises(shelf.VoiceLibraryError):
        shelf.publish_version(p.id, record["key"])
    assert library.load_voices() == []


def test_metadata_tampering_is_refused(source):
    _, _, p, target, _ = source
    published = shelf.publish_version(p.id, first_record(p)["key"])
    voices = library.load_voices(); voices[0].voice_lab_versions[published["version"]]["record"]["name"] = "Changed without a new version"
    library.save_voices(voices)
    with pytest.raises(shelf.VoiceLibraryError, match="metadata"):
        shelf.import_version(target.id, published["library_voice_id"], published["version"])


def test_new_versions_append_and_legacy_updates_preserve_snapshots(source):
    _, _, p, _, _ = source
    record = first_record(p); first = shelf.publish_version(p.id, record["key"])
    changed = copy.deepcopy(record); changed["version"] = "new-profile-version"
    changed["key"] = "v_" + lab.digest([changed["voice_id"], changed["version"]])[:28]
    changed["name"] = "New original profile"
    project = load_project(p.id); project.voice_lab.voices[changed["key"]] = changed; save_project(project)
    second = shelf.publish_version(p.id, changed["key"])
    assert second["library_voice_id"] == first["library_voice_id"]
    voices = library.load_voices(); original = copy.deepcopy(voices[0].voice_lab_versions)
    voices[0].name = "Legacy display name edit"; library.save_voices(voices)
    assert library.load_voices()[0].voice_lab_versions == original
    assert len(original) == 2


def test_import_waits_for_existing_project_render(source):
    _, _, p, target, _ = source
    published = shelf.publish_version(p.id, first_record(p)["key"])
    from dubbing_service.app.core.tts_operations import try_register_running_batch, unregister_running_batch
    lease = try_register_running_batch(target.id)
    assert lease
    try:
        with pytest.raises(shelf.VoiceLibraryError, match="rendering"):
            shelf.import_version(target.id, published["library_voice_id"], published["version"])
    finally:
        unregister_running_batch(lease)
    assert not load_project(target.id).voice_lab.voices


def test_profile_edit_cannot_overwrite_existing_version(source):
    _, _, p, _, _ = source
    record = add_reference_identity(source)
    published = shelf.publish_version(p.id, record["key"])
    before = library._voices_file_path().read_bytes()
    project = load_project(p.id)
    project.voice_lab.voices[record["key"]]["identity"]["design_instruction"] = "Changed traits without versioning"
    save_project(project)
    with pytest.raises(shelf.VoiceLibraryError, match="new voice version"):
        shelf.publish_version(p.id, record["key"])
    assert library._voices_file_path().read_bytes() == before
    assert shelf.list_versions()[0]["versions"][0]["version"] == published["version"]


def test_concurrent_publish_creates_one_immutable_version(source):
    from concurrent.futures import ThreadPoolExecutor
    _, voices_root, p, _, _ = source
    record = first_record(p)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: shelf.publish_version(p.id, record["key"]), range(2)))
    assert sorted(row["published"] for row in results) == [False, True]
    assert len(library.load_voices()) == 1
    assert len(library.load_voices()[0].voice_lab_versions) == 1
    assert len([path for path in voices_root.rglob("*") if path.is_file()]) == 1


def test_import_will_not_overwrite_tampered_existing_destination(source):
    root, _, p, target, _ = source
    published = shelf.publish_version(p.id, first_record(p)["key"])
    imported = shelf.import_version(target.id, published["library_voice_id"], published["version"])
    record = load_project(target.id).voice_lab.voices[imported["key"]]
    destination = root / target.id / record["audio_file"]
    destination.write_bytes(b"changed fixture bytes")
    before = (root / target.id / "project.json").read_bytes()
    with pytest.raises(shelf.VoiceLibraryError, match="bytes"):
        shelf.import_version(target.id, published["library_voice_id"], published["version"])
    assert destination.read_bytes() == b"changed fixture bytes"
    assert (root / target.id / "project.json").read_bytes() == before
