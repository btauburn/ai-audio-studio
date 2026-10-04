"""Real renderer/pipeline/QA/Take-merge regression with a local signal provider.

Only the TTS provider is replaced. Temporary WAVs are explicit test tones, not
model output, character auditions or the requested listening test. The excerpt
test stops at its real duration gate, after verifying actual Take promotion.
"""
from dataclasses import replace

import pytest

from tests.audiobook_helpers import build_project, make_tone_wav, use_projects_root
from tests.test_voice_lab_chapter import source as chapter_source, ready  # noqa: F401
from tests.test_listening_excerpt import source as excerpt_source  # noqa: F401
from dubbing_service.app.audiobook import pipeline, renderer, voice_lab_chapter as chapter, voice_lab_render as excerpt
from dubbing_service.app.audiobook.listening_excerpt import build_excerpt_manifest, text_hash
from dubbing_service.app.models.audiobook import Attribution, SourceAnchor, SourceRevision
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.models.project import Chapter, Role, Segment
from dubbing_service.app.providers.tts import TTSResult
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments


class LocalSignalProvider:
    """No worker, model, network or GPU. A fresh WAV for each provider request."""

    def __init__(self, root):
        self.root = root
        self.calls = []

    def synthesize_segment(self, project_id, role, segment):
        self.calls.append((project_id, segment.id))
        path = make_tone_wav(self.root / f"signal-{len(self.calls)}.wav",
                             seconds=max(.25, len(segment.text) * .06), frequency=247)
        return TTSResult(success=True, audio_path=str(path), raw={
            "identity_id": role.voice_identity.id,
            "prompt_sha256": role.voice_identity.prompt_sha256,
            "execution_profile": {"model_id": "LOCAL-TEST-SIGNAL-NOT-TTS", "fixture_only": True},
        })


def provider(monkeypatch, tmp_path):
    fake = LocalSignalProvider(tmp_path / "provider-fixture")
    def create(backend):
        assert backend == "qwen3tts"
        return fake
    monkeypatch.setattr(renderer.TTSProviderFactory, "create", create)
    return fake


def assert_selected_current(project_id, segment_ids):
    project = load_project(project_id)
    rows = {s.id: s for s in load_segments(project_id)}
    for segment_id in segment_ids:
        segment = rows[segment_id]
        assert segment.selected_take_id is not None
        take = next(t for t in segment.takes if t.id == segment.selected_take_id)
        assert take.renderer_id == take.engine_id == "qwen3tts"
        assert take.fingerprint_components["renderer_id"] == "qwen3tts"
        current = pipeline._requirement_for_take(project, segment, take)
        assert current.fingerprint() == take.input_fingerprint
        assert not take.stale and not take.stale_components
        assert take.qa_report.passed and take.qa_report.matches(take.id, take.audio_sha256)
        assert take.generation_metadata["provider_calls"][0]["execution_profile"]["fixture_only"]
    return {sid: rows[sid].selected_take_id for sid in segment_ids}


def test_chapter_default_renderer_promotes_actual_pipeline_take_and_resumes_without_provider(
        chapter_source, monkeypatch, tmp_path):
    project = ready(chapter_source)
    fake = provider(monkeypatch, tmp_path)
    result = chapter.render_chapter(project.id, chapter_id="chap1", request_id="real-pipeline-default-renderer")
    assert result["job"]["status"] == "succeeded", result
    assert len(fake.calls) == 2
    selected = assert_selected_current(project.id, ["seg0", "seg2"])
    result = chapter.render_chapter(project.id, chapter_id="chap1", request_id="real-pipeline-default-renderer")
    assert result["job"]["status"] == "succeeded", result
    assert all(record["reused"] and not record["dispatched"] for record in result["records"])
    assert len(fake.calls) == 2
    assert assert_selected_current(project.id, ["seg0", "seg2"]) == selected


def test_regression_control_unqualified_renderer_is_rejected_as_stale(chapter_source, monkeypatch, tmp_path):
    project = ready(chapter_source)
    fake = provider(monkeypatch, tmp_path)
    # Reproduce the faulty constructor without changing the real pipeline or
    # weakening its merge rule. This establishes what the positive test catches.
    legacy = renderer.BackendRenderer(backend="qwen3tts", project_root=chapter_source[1] / project.id)
    result = chapter.render_chapter(project.id, chapter_id="chap1", request_id="legacy-constructor-control", renderer=legacy)
    assert result["job"]["status"] == "interrupted"
    assert len(fake.calls) == 1
    record = result["records"][0]
    assert record["success"] and record["qa_passed"]
    assert not record["selected"] and record["stale_components"] == ["renderer_id"]
    assert next(s for s in load_segments(project.id) if s.id == "seg0").selected_take_id is None


def test_excerpt_default_renderer_reaches_duration_gate_after_real_pipeline_promotes_takes(
        excerpt_source, monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = build_project(name="Renderer identity regression: test signals only")
    project.voice_lab.scope = "isolated_listening_test"
    project.character_cast.policy.enabled = False
    project.character_cast.policy.audition_mode = "off"
    source = dict(excerpt_source)
    identity = replace(source["chapter"], source_sha256=text_hash(source["source_text"]))
    source["chapter"] = identity
    project.chapters = [Chapter(id=identity.chapter_id, title=identity.chapter_title)]
    project.roles = []
    for role_id in ["narrator", "alice", "bea", "charlie", "dani"]:
        reference = make_tone_wav(root / project.id / f"{role_id}.wav", seconds=.15)
        prompt = root / project.id / f"{role_id}.npz"
        prompt.write_bytes(b"NOT-A-MODEL-PROMPT-LOCAL-SIGNAL-TEST")
        voice = CharacterVoiceIdentity(id="signal_" + role_id, role_id=role_id, mechanism="qwen_reference",
            backend="qwen3tts", model="LOCAL-TEST-SIGNAL-NOT-TTS", dossier_revision="fixture",
            casting_profile_revision="fixture", design_instruction="No real voice; isolated signal test only",
            reference_audio=reference.name, reference_audio_sha256=renderer.sha256_file(reference),
            prompt_artifact=prompt.name, prompt_sha256=renderer.sha256_file(prompt))
        project.roles.append(Role(id=role_id, name=role_id, role_type="narrator" if role_id == "narrator" else "character",
                                  tts_backend="qwen3tts", voice_identity=voice))
    snapshot = root / project.id / "fixture.snapshot"
    snapshot.write_bytes(source["source_text"].encode("utf-8"))
    project.source_revisions = [SourceRevision(id=identity.source_revision_id, uri="fixture:never-a-book",
        sha256=renderer.sha256_file(snapshot), snapshot_file=snapshot.name, byte_length=snapshot.stat().st_size)]
    save_project(project)
    rows = [Segment(id=row.utterance_id, chapter_id=identity.chapter_id, role_id=row.role_id, index=i,
        text=source["source_text"][row.start_offset:row.end_offset], tts_backend="qwen3tts",
        source_anchor=SourceAnchor(source_revision_id=identity.source_revision_id, chapter_id=identity.chapter_id,
                                   start_offset=row.start_offset, end_offset=row.end_offset),
        attribution=Attribution(speaker_id=row.role_id, source="tag")) for i, row in enumerate(source["utterances"])]
    save_segments(project.id, rows)
    manifest = build_excerpt_manifest(**source)
    excerpt.persist_excerpt_manifest(project.id, manifest, source_text=source["source_text"])
    fake = provider(monkeypatch, tmp_path)
    # Six short QA-valid signals intentionally do not pretend to be a 5-minute
    # reading. Reaching this later gate proves render/QA/promotion all completed.
    with pytest.raises(excerpt._TimingAdjustmentNeeded) as first:
        excerpt._render_listening_excerpt_once(project.id, expected_manifest_id=manifest["manifest_id"])
    assert len(first.value.records) == 6 and all(r.selected and r.qa_passed for r in first.value.records)
    assert len(fake.calls) == 6
    selected = assert_selected_current(project.id, manifest["utterance_ids"])
    with pytest.raises(excerpt._TimingAdjustmentNeeded) as resumed:
        excerpt._render_listening_excerpt_once(project.id, expected_manifest_id=manifest["manifest_id"])
    assert all(r.reused and not r.dispatched for r in resumed.value.records)
    assert len(fake.calls) == 6
    assert assert_selected_current(project.id, manifest["utterance_ids"]) == selected
