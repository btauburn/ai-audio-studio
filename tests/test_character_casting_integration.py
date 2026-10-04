"""Public ingest -> research -> audition -> persisted identity -> render seams."""

from __future__ import annotations

import json
import subprocess
import sys

from dubbing_service.app.audiobook.adapter import (
    AudiobookerAdapter,
    BookAnalysis,
    analyze_text_offline,
)
from dubbing_service.app.audiobook.character_casting import prepare_cast, refresh_cast
from dubbing_service.app.audiobook.fingerprint import (
    file_sha256,
    voice_revision_for_role,
)
from dubbing_service.app.audiobook.pipeline import render_utterances
from dubbing_service.app.audiobook.renderer import BackendRenderer
from dubbing_service.app.models.project import Project
from dubbing_service.app.providers.tts import TTSResult
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import make_tone_wav, use_projects_root

TEXT = """Chapter 1
Morgan was a woman. Morgan was thirty years old. Morgan spoke in a clear, high voice.
Morgan: The door is open. We can leave when you are ready.
Morgan: Please remember to bring the map with you.
Morgan: I know where we should go next.
Morgan: This is the place I was telling you about.
Morgan: We have more than enough time to get there.
Morgan: Listen carefully, and tell me what you hear.
"""


class OriginalVoiceTestProvider:
    """Audible deterministic signal double; never labelled model-generated audio."""

    def __init__(self, root):
        self.root = root
        self.design_calls = 0
        self.rendered_identities = []

    def design_character_reference(
        self, project_id, identity_id, instruction, text, seed
    ):
        self.design_calls += 1
        target = make_tone_wav(
            self.root / project_id / "audio/cast" / identity_id / "reference.wav",
            seconds=2.5,
            frequency=240,
        )
        return {
            "mechanism": "fixture",
            "model": "signal-double",
            "model_revision": "v1",
            "runtime_version": "test",
            "reference_audio": target.relative_to(self.root / project_id).as_posix(),
            "reference_audio_sha256": file_sha256(target),
            "reference_text": text,
            "provenance": {"audio_kind": "synthetic signal, not TTS"},
        }

    def prepare_character_voice(self, project_id, identity_id, metadata):
        path = self.root / project_id / "audio/cast" / identity_id / "prompt.json"
        path.write_text(json.dumps({"fixture_identity": identity_id}), encoding="utf-8")
        return {
            "prompt_artifact": path.relative_to(self.root / project_id).as_posix(),
            "prompt_sha256": file_sha256(path),
        }

    def synthesize_segment(self, project_id, role, segment):
        assert role.voice_identity is not None
        self.rendered_identities.append((role.voice_identity.id, segment.emotion_text))
        path = make_tone_wav(
            self.root
            / project_id
            / "audio/cast"
            / role.voice_identity.id
            / (segment.id + ".wav"),
            seconds=3,
            frequency=240,
        )
        return TTSResult(
            success=True, audio_path=str(path), raw={"performance_support": "fixture"}
        )


def ingested(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    result = AudiobookerAdapter().reconcile_into_project(analyze_text_offline(TEXT))
    project = load_project(result.project_id)
    role = next(r for r in project.roles if r.name == "Morgan")
    project.character_cast.policy.max_candidates = 2
    save_project(project)
    return root, project, role


def test_old_project_migrates_additively():
    project = Project.model_validate(
        {"id": "old", "name": "old", "roles": [{"id": "r", "name": "R"}]}
    )
    assert project.character_cast.schema_version == 1
    assert not project.character_cast.dossiers
    assert project.roles[0].voice_identity is None
    assert Project.model_validate_json(project.model_dump_json()) == project


def test_reingest_does_not_use_alias_beyond_spoiler_boundary(monkeypatch, tmp_path):
    from dubbing_service.app.models.character import CharacterFact, StoryScope

    _root, project, role = ingested(monkeypatch, tmp_path)
    dossier = project.character_cast.dossiers[role.id]
    dossier.aliases.append("The Mask")
    dossier.facts.append(
        CharacterFact(
            id="future-alias",
            field="aliases",
            value="The Mask",
            source="book",
            level="EXPLICIT",
            confidence=1,
            scope=StoryScope(chapter_start=80),
            available_after_chapter=80,
        )
    )
    project.character_cast.policy.max_chapter = 5
    save_project(project)
    analysis = analyze_text_offline("Chapter 1\nThe Mask: You cannot know who I am.")
    AudiobookerAdapter().reconcile_into_project(analysis, project_id=project.id)
    updated = load_project(project.id)
    masked = next(r for r in updated.roles if r.name == "The Mask")
    assert masked.id != role.id
    assert masked.id not in updated.character_cast.role_aliases


def test_baked_scene_volume_changes_requirement_but_legacy_playback_does_not():
    from dubbing_service.app.audiobook.fingerprint import (
        changed_components,
        requirement_for_segment,
    )
    from dubbing_service.app.models.character import CharacterVoiceIdentity
    from dubbing_service.app.models.project import Role, Segment

    role = Role(id="r", name="R")
    segment = Segment(id="s", text="A short sentence.", role_id="r")

    def requirement():
        return requirement_for_segment(
            segment, role=role, project_id="p", engine_id="qwen3tts"
        )

    legacy = requirement()
    segment.volume_gain = 0.3
    assert requirement().fingerprint() == legacy.fingerprint()
    role.voice_identity = CharacterVoiceIdentity(
        id="v",
        role_id="r",
        mechanism="qwen_reference",
        dossier_revision="d",
        casting_profile_revision="c",
        design_instruction="Original voice",
        model="test",
    )
    quiet = requirement()
    segment.volume_gain = 1
    loud = requirement()
    assert quiet.fingerprint() != loud.fingerprint()
    assert changed_components(quiet.components(), loud.components()) == ["volume"]


def test_audiobooker_sentinel_labels_are_not_characters(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    analysis = BookAnalysis.from_dict(
        {
            "speakers": ["narrator", "unknown", "Ada"],
            "chapters": [
                {
                    "index": 0,
                    "title": "One",
                    "utterances": [
                        {
                            "speaker": "narrator",
                            "text": "Ada was a girl.",
                            "line_index": 0,
                        },
                        {
                            "speaker": "unknown",
                            "text": "Who is there?",
                            "line_index": 1,
                        },
                        {"speaker": "Ada", "text": "I am here.", "line_index": 2},
                    ],
                }
            ],
        }
    )
    result = AudiobookerAdapter().reconcile_into_project(analysis)
    project = load_project(result.project_id)
    assert {
        r.name for r in project.roles if r.id in project.character_cast.profiles
    } == {"Ada"}
    assert any(
        "Unresolved speaker attribution" in r
        for r in project.character_cast.review_reasons
    )


def test_ingest_automatically_researches_and_preserves_source(monkeypatch, tmp_path):
    root, project, role = ingested(monkeypatch, tmp_path)
    assert role.id in project.character_cast.dossiers
    assert role.id in project.character_cast.profiles
    assert project.source_revisions
    revision = project.source_revisions[0]
    assert file_sha256(root / project.id / revision.snapshot_file) == revision.sha256
    assert all(s.source_anchor for s in load_segments(project.id))
    assert role.voice_identity is None  # ingest does no unrequested GPU work


def test_locked_identity_survives_process_reload_and_emotion(monkeypatch, tmp_path):
    root, project, role = ingested(monkeypatch, tmp_path)
    provider = OriginalVoiceTestProvider(root)
    prepare_cast(project.id, provider=provider, role_ids=[role.id])
    reloaded = load_project(project.id)
    locked = next(r for r in reloaded.roles if r.id == role.id)
    assert locked.voice_identity is not None
    before = voice_revision_for_role(locked)
    code = "import sys; from dubbing_service.app.storage.projects import load_project; p=load_project(sys.argv[1]); print(next(r.voice_identity.id for r in p.roles if r.id==sys.argv[2]))"
    result = subprocess.run(
        [sys.executable, "-c", code, project.id, role.id],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == locked.voice_identity.id
    segments = load_segments(project.id)
    dialogue = next(s for s in segments if s.role_id == role.id)
    dialogue.emotion_text = "angry"
    save_segments(project.id, segments)
    renderer = BackendRenderer(
        backend="qwen3tts",
        engine_id="qwen3tts",
        provider=provider,
        project_root=root / project.id,
    )
    records = render_utterances(
        project.id, [dialogue.id], renderer=renderer, run_qa=False
    )
    assert records[0].success
    assert provider.rendered_identities[-1] == (locked.voice_identity.id, "angry")
    updated = load_project(project.id)
    assert (
        voice_revision_for_role(next(r for r in updated.roles if r.id == role.id))
        == before
    )
    take = next(s for s in load_segments(project.id) if s.id == dialogue.id).takes[-1]
    assert (
        take.generation_metadata["character_voice_identity"]["id"]
        == locked.voice_identity.id
    )
    assert take.generation_metadata["scene_performance"]["emotion"] == "angry"


def test_unchanged_material_does_not_redesign_and_new_research_proposes(
    monkeypatch, tmp_path
):
    root, project, role = ingested(monkeypatch, tmp_path)
    provider = OriginalVoiceTestProvider(root)
    prepare_cast(project.id, provider=provider, role_ids=[role.id])
    first = next(
        r for r in load_project(project.id).roles if r.id == role.id
    ).voice_identity
    assert first is not None
    calls = provider.design_calls
    AudiobookerAdapter().reconcile_into_project(
        analyze_text_offline(TEXT), project_id=project.id
    )
    prepare_cast(project.id, provider=provider, role_ids=[role.id])
    assert provider.design_calls == calls
    p = load_project(project.id)
    assert next(r for r in p.roles if r.id == role.id).voice_identity == first
    p.character_cast.policy.inspiration_palette = "harry-potter-film-cast"
    save_project(p)
    refresh_cast(p.id)
    p = load_project(p.id)
    assert next(r for r in p.roles if r.id == role.id).voice_identity == first
    assert role.id in p.character_cast.proposed_revisions


def test_failure_is_durable_and_retry_is_bounded(monkeypatch, tmp_path):
    root, project, role = ingested(monkeypatch, tmp_path)
    provider = OriginalVoiceTestProvider(root)

    def fail(*args, **kwargs):
        raise RuntimeError("offline model unavailable")

    provider.design_character_reference = fail
    report = prepare_cast(project.id, provider=provider, role_ids=[role.id])
    assert report["locked"] == 0
    rows = load_project(project.id).character_cast.auditions[role.id]
    assert 1 <= len(rows) <= 2
    assert all(c.error and "unavailable" in c.error for c in rows)
    assert report["review_reasons"]


def test_first_ingest_alias_uses_one_persistent_voice(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    text = "Chapter 1\nSamuel, also known as Sam, was a man.\nSamuel: I will carry the map.\nSam: We can go together."
    result = AudiobookerAdapter().reconcile_into_project(analyze_text_offline(text))
    project = load_project(result.project_id)
    canonical = next(r for r in project.roles if r.name == "Samuel")
    alias = next(r for r in project.roles if r.name == "Sam")
    assert project.character_cast.role_aliases[alias.id] == canonical.id
    assert alias.id not in project.character_cast.profiles
    project.character_cast.policy.max_candidates = 1
    save_project(project)
    provider = OriginalVoiceTestProvider(root)
    prepare_cast(project.id, provider=provider, role_ids=[alias.id])
    project = load_project(project.id)
    identities = [
        r.voice_identity for r in project.roles if r.id in {alias.id, canonical.id}
    ]
    assert all(identities)
    assert identities[0].id == identities[1].id
    assert provider.design_calls == 1


def test_scene_pace_and_volume_change_audio_without_changing_pitch(tmp_path):
    import numpy as np
    import soundfile as sf

    from dubbing_service.app.audiobook.auditions import measure_audio
    from dubbing_service.app.audiobook.performance import apply_scene_audio

    sample = make_tone_wav(
        tmp_path / "scene.wav", seconds=2, frequency=220, amplitude=0.4
    )
    before = measure_audio(sample)
    assert apply_scene_audio(sample, pace=1.5, volume=0.5)
    after = measure_audio(sample)
    assert 1.2 < after["duration_seconds"] < 1.4
    assert abs(after["pitch_hz"] - before["pitch_hz"]) < 5
    assert 0.45 < after["rms"] / before["rms"] < 0.55
    assert np.isfinite(sf.read(sample)[0]).all()
