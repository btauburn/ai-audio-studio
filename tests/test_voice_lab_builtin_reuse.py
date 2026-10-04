"""Built-in review must preserve the heard configuration, never today's role."""
import copy

import pytest

from dubbing_service.app.audiobook import voice_lab as lab
from dubbing_service.app.audiobook.fingerprint import file_sha256, voice_revision_for_role
from dubbing_service.app.models.project import Role, Segment, SegmentTake
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import make_tone_wav, use_projects_root


@pytest.fixture
def saved_voice(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = create_project("Built-in configuration regression fixture")
    source = Role(id="source", name="Source", tts_backend="qwen3tts_cv", qwen3_speaker="ryan",
                  qwen3_language="English", style_instruction="Thoughtful, measured delivery.",
                  default_emotion_text="calm", default_emotion_vector=[0.1] * 8, emotion_alpha=0.4)
    target = Role(id="target", name="Target", tts_backend="qwen3tts_cv", qwen3_speaker="vivian",
                  qwen3_language="Chinese", style_instruction="Energetic delivery.", emotion_alpha=0.9)
    project.roles = [source, target]
    save_project(project)
    audio = make_tone_wav(root / project.id / "audio/segments/fixture.wav", seconds=0.1)
    take = SegmentTake(id="historical", role_id=source.id, audio_file="audio/segments/fixture.wav",
        audio_sha256=file_sha256(audio), engine_id="qwen3tts_customvoice", tts_backend="qwen3tts_cv",
        fingerprint_components={"voice_revision": voice_revision_for_role(source), "builtin_speaker": "ryan",
                                "language": "English", "style_instruction": source.style_instruction},
        generation_metadata={"builtin_voice_snapshot": source.model_dump(mode="json")})
    save_segments(project.id, [Segment(id="u1", index=0, role_id=source.id, text="Fixture only.", takes=[take], selected_take_id=take.id)])
    return root, project, source, target


def only_record(project):
    return next(iter(lab.voice_records(load_project(project.id)).values()))


def test_full_immutable_settings_reuse_survives_source_edits_without_relabeling(saved_voice):
    root, project, source, target = saved_voice
    original = only_record(project)
    changed = load_project(project.id)
    changed.roles[0].qwen3_speaker = "aiden"
    changed.roles[0].qwen3_language = "German"
    changed.roles[0].style_instruction = "Changed after recording."
    save_project(changed)
    record = only_record(project)
    assert record["key"] == original["key"]
    assert record["builtin_snapshot"]["settings"]["style_instruction"] == source.style_instruction
    before = (root / project.id / "segments.json").read_bytes()
    result = lab.assign(project.id, record["key"], target.id, voice_revision_for_role(target))
    assigned = load_project(project.id).roles[1]
    for field, value in record["builtin_snapshot"]["settings"].items():
        assert assigned.model_dump(mode="json")[field] == value
    assert result["source_voice_revision"] == voice_revision_for_role(source)
    assert result["assigned_voice_revision"] == voice_revision_for_role(assigned)
    assert result["same_voice_revision"] is False  # Legacy hash includes target role ID.
    assert (root / project.id / "segments.json").read_bytes() == before
    assert not load_project(project.id).voice_lab.preferences


@pytest.mark.parametrize("missing", ["language", "style_instruction"])
def test_incomplete_effective_take_stamp_rejects_without_mutation(saved_voice, missing):
    root, project, source, target = saved_voice
    segments = load_segments(project.id)
    del segments[0].takes[0].fingerprint_components[missing]
    save_segments(project.id, segments)
    record = only_record(project)
    assert record["builtin_snapshot"] is None and record["assignment_hold"]
    before = (root / project.id / "project.json").read_bytes()
    with pytest.raises(lab.VoiceLabError, match="settings are incomplete"):
        lab.assign(project.id, record["key"], target.id, voice_revision_for_role(target))
    assert (root / project.id / "project.json").read_bytes() == before


def test_legacy_scalar_recovery_requires_full_matching_source_revision(saved_voice):
    _, project, source, target = saved_voice
    segments = load_segments(project.id)
    segments[0].takes[0].generation_metadata = {}
    save_segments(project.id, segments)
    record = only_record(project)
    assert record["builtin_snapshot"]["proof"] == "configuration_matches_immutable_take_revision"
    changed = load_project(project.id)
    changed.roles[0].style_instruction = "Unrelated new settings"
    save_project(changed)
    held = only_record(project)
    assert held["builtin_snapshot"] is None
    assert held["key"] == record["key"]
    with pytest.raises(lab.VoiceLabError, match="settings are incomplete"):
        lab.assign(project.id, held["key"], target.id, voice_revision_for_role(target))


def test_saved_favorite_configuration_survives_legacy_source_edit(saved_voice):
    _, project, source, target = saved_voice
    segments = load_segments(project.id)
    segments[0].takes[0].generation_metadata = {}
    save_segments(project.id, segments)
    record = only_record(project)
    lab.save_preference(project.id, {"key": record["key"], "audio_sha256": record["audio_sha256"], "favorite": True})
    changed = load_project(project.id)
    changed.roles[0].style_instruction = "New source style"
    save_project(changed)
    lab.assign(project.id, record["key"], target.id, voice_revision_for_role(target))
    assert load_project(project.id).roles[1].style_instruction == source.style_instruction
    assert load_project(project.id).voice_lab.preferences[record["key"]]["version"] == record["version"]


@pytest.mark.parametrize("field,value", [("language", "German"), ("builtin_speaker", "aiden"),
                                        ("style_instruction", "Different historical instruction")])
def test_effective_take_overrides_cannot_be_mislabeled_as_source_defaults(saved_voice, field, value):
    _, project, source, target = saved_voice
    segments = load_segments(project.id)
    segments[0].takes[0].fingerprint_components[field] = value
    save_segments(project.id, segments)
    assert only_record(project)["builtin_snapshot"] is None


def test_corrupt_saved_snapshot_cannot_recast_target(saved_voice):
    _, project, source, target = saved_voice
    record = only_record(project)
    changed = load_project(project.id)
    corrupt = copy.deepcopy(record)
    corrupt["builtin_snapshot"]["settings"]["style_instruction"] = "Tampered"
    changed.voice_lab.voices[record["key"]] = corrupt
    save_project(changed)
    with pytest.raises(lab.VoiceLabError, match="do not match"):
        lab.assign(project.id, record["key"], target.id, voice_revision_for_role(target))
    assert load_project(project.id).roles[1].qwen3_speaker == "vivian"
