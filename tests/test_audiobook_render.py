"""Phases 8-12: chunking, one-Take-per-utterance, QA binding and promotion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import qa, release, renderer, takes
from dubbing_service.app.audiobook import pipeline
from dubbing_service.app.providers.tts import TTSResult
from tests.audiobook_helpers import (
    build_project,
    build_segments,
    make_silence_wav,
    make_tone_wav,
    use_projects_root,
)

LONG_TEXT = " ".join(
    f"Sentence number {index} keeps the narration going for a while." for index in range(20)
)


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, ["A short utterance.", LONG_TEXT])
    return root, project, segments


@dataclass
class FailingRenderer:
    """A renderer that always fails, for failure-path assertions."""

    renderer_id: str = "fixture-failing"
    engine_id: str = "fixture"

    def render(self, request):  # noqa: ANN001 - test double
        return renderer.RenderOutcome(
            success=False,
            chunks=[],
            provider_calls=1,
            chunked=False,
            error="synthetic engine failure",
        )


@dataclass
class TelemetryProvider:
    root: Path

    def synthesize_segment(self, project_id, role, segment):  # noqa: ANN001
        output = make_tone_wav(
            self.root / f"{segment.id}.wav", seconds=max(0.2, len(segment.text) * 0.03)
        )
        return TTSResult(
            success=True,
            audio_path=str(output),
            raw={
                "request_id": "req-telemetry",
                "utterance_id": segment.id,
                "execution_profile": {
                    "execution_profile_id": "profile-sdpa",
                    "model_id": "qwen3_tts_customvoice",
                    "attention_resolved": "sdpa",
                },
                "stage_timings": {
                    "inclusive": {"model_generate_seconds": 0.125}
                },
                "generation": {
                    "termination_reason": "eos",
                    "generated_codec_steps": 12,
                },
            },
        )


def fixture_renderer(root, project_id):
    return renderer.FixtureRenderer(
        project_root=pipeline.project_root(project_id), renderer_id="fixture"
    )


def test_short_utterance_creates_exactly_one_take(studio):
    root, project, segments = studio
    rendering = fixture_renderer(root, project.id)

    record = pipeline.render_segment(
        project, segments[0], renderer=rendering, root=pipeline.project_root(project.id),
        promote=True,
    )

    assert record.success is True
    assert record.chunked is False
    assert record.qa_passed is True
    assert record.selected is True

    segment = segments[0]
    assert len(segment.takes) == 1
    take = segment.takes[0]
    assert take.id == record.take_id
    assert take.audio_sha256 == record.audio_sha256
    assert len(take.chunk_map) == 1
    assert segment.selected_take_id == take.id
    assert (pipeline.project_root(project.id) / take.audio_file).is_file()


def test_backend_execution_profile_and_stage_timings_persist_on_take(studio, tmp_path):
    root, project, segments = studio
    rendering = renderer.BackendRenderer(
        backend="qwen3tts_cv",
        engine_id="qwen3tts_customvoice",
        provider=TelemetryProvider(tmp_path / "provider"),
        project_root=pipeline.project_root(project.id),
    )

    record = pipeline.render_segment(
        project,
        segments[0],
        renderer=rendering,
        root=pipeline.project_root(project.id),
        promote=True,
    )

    assert record.success is True
    take = takes.find_take(segments[0], record.take_id)
    calls = take.generation_metadata["provider_calls"]
    assert calls[0]["request_id"] == "req-telemetry"
    assert calls[0]["utterance_id"] == segments[0].id
    assert calls[0]["execution_profile"]["execution_profile_id"] == "profile-sdpa"
    assert calls[0]["stage_timings"]["inclusive"]["model_generate_seconds"] == 0.125
    assert "transport_and_renderer" in calls[0]
    assert "pipeline_precommit_stage_timings" in take.generation_metadata
    assert record.stage_timings["exclusive"]["qa_seconds"] >= 0
    assert take.qa_report is not None
    assert take.qa_report.matches(take.id, take.audio_sha256) is True


def test_long_utterance_is_chunked_into_a_single_take(studio):
    root, project, segments = studio
    rendering = fixture_renderer(root, project.id)
    segment = segments[1]
    assert len(renderer.plan_chunks(segment.text)) > 1

    record = pipeline.render_segment(
        project, segment, renderer=rendering, root=pipeline.project_root(project.id),
        promote=True,
    )

    assert record.success is True
    assert record.chunked is True
    take = takes.find_take(segment, record.take_id)
    assert len(take.chunk_map) > 1
    assert all(chunk.chunk_sha256 for chunk in take.chunk_map)
    joined = "".join(
        segment.text[chunk.text_start : chunk.text_end] for chunk in take.chunk_map
    )
    assert joined == segment.text
    assert len(segment.takes) == 1

    audio = pipeline.project_root(project.id) / take.audio_file
    assert renderer.sha256_file(audio) == take.audio_sha256


def test_chunk_planner_does_not_create_sentence_tail_micro_chunks():
    text = (
        "The argument ran the length of the corridor and back again, past the pantry, "
        "past the room where the maps were kept, past the door that nobody repaired, "
        "and it did not resolve, because neither of them had come to resolve anything; "
        "they had come to be right in front of a witness, and the witness, who had been "
        "folding linen the entire time, kept folding linen, which is its own kind of "
        "verdict. Later that evening the corridor was swept, the maps were rolled and "
        "tied, and the door that nobody repaired was still not repaired, though someone "
        "had chalked a small mark on the frame to show that it had been considered."
    )

    chunks = renderer.plan_chunks(text)

    assert "".join(chunks) == text
    assert len(chunks) == 3
    assert max(map(len, chunks)) <= renderer.CHUNK_MAX_CHARS
    assert min(map(len, chunks)) >= renderer.CHUNK_MIN_TRAILING_CHARS
    assert "resolve " not in chunks
    assert "the " not in chunks


def test_chunk_planner_rebalances_a_short_final_remainder():
    text = ("alpha beta gamma delta " * 19) + "omega"

    chunks = renderer.plan_chunks(text, trigger_chars=100, max_chars=80)

    assert "".join(chunks) == text
    assert max(map(len, chunks)) <= 80
    assert min(map(len, chunks)) >= renderer.CHUNK_MIN_TRAILING_CHARS


def test_chunk_map_offsets_are_contiguous(studio):
    root, project, segments = studio
    rendering = fixture_renderer(root, project.id)
    record = pipeline.render_segment(
        project, segments[1], renderer=rendering, root=pipeline.project_root(project.id)
    )
    take = takes.find_take(segments[1], record.take_id)

    offsets = [(chunk.audio_start_ms, chunk.audio_end_ms) for chunk in take.chunk_map]
    assert offsets == sorted(offsets)
    for index, (start, end) in enumerate(offsets):
        assert start < end
        if index:
            assert start == offsets[index - 1][1]
    assert offsets[-1][1] == record.duration_ms


def test_failed_synthesis_keeps_the_previous_take(studio):
    root, project, segments = studio
    segment = segments[0]
    good = pipeline.render_segment(
        project,
        segment,
        renderer=fixture_renderer(root, project.id),
        root=pipeline.project_root(project.id),
        promote=True,
    )
    assert segment.selected_take_id == good.take_id
    before = [(take.id, take.audio_sha256) for take in segment.takes]

    failed = pipeline.render_segment(
        project,
        segment,
        renderer=FailingRenderer(),
        root=pipeline.project_root(project.id),
        promote=True,
    )

    assert failed.success is False
    assert failed.error and "synthetic engine failure" in failed.error
    assert [(take.id, take.audio_sha256) for take in segment.takes] == before
    assert segment.selected_take_id == good.take_id


def test_missing_role_is_refused_unless_allow_missing_role(studio):
    root, project, segments = studio
    project.roles.clear()
    with pytest.raises(pipeline.PipelineError):
        pipeline.render_segment(
            project,
            segments[0],
            renderer=fixture_renderer(root, project.id),
            root=pipeline.project_root(project.id),
        )

    record = pipeline.render_segment(
        project,
        segments[0],
        renderer=fixture_renderer(root, project.id),
        root=pipeline.project_root(project.id),
        allow_missing_role=True,
    )
    assert record.success is True


def test_qa_report_is_bound_to_the_exact_bytes(studio):
    root, project, segments = studio
    segment = segments[0]
    record = pipeline.render_segment(
        project, segment, renderer=fixture_renderer(root, project.id),
        root=pipeline.project_root(project.id), promote=True,
    )
    take = takes.find_take(segment, record.take_id)
    report = take.qa_report
    assert report is not None
    assert report.matches(take.id, take.audio_sha256) is True
    assert report.matches("another_take", take.audio_sha256) is False
    assert report.matches(take.id, "different-hash") is False

    audio = pipeline.project_root(project.id) / take.audio_file
    audio.write_bytes(b"unrelated bytes")
    coverage = release.check_coverage(
        segments, project_root=pipeline.project_root(project.id)
    )
    assert any(
        issue.code == release.RESOLUTION_HASH_MISMATCH for issue in coverage.issues
    )


def test_qa_never_claims_a_pass_when_nothing_ran():
    skipped = [qa._check("asr_completeness", "not_run", detail="no model", tier=2)]
    report = qa.build_report("take_x", "hash", skipped, tier=2)
    assert report.passed is False
    assert report.executed_check_ids() == []

    passed = qa.build_report("take_x", "hash", [qa._check("file_exists", "pass")])
    assert passed.passed is True
    assert passed.executed_check_ids() == ["file_exists"]


def test_silence_and_clipping_are_rejected(studio, tmp_path):
    # 0.2 s of silence for a five letter word keeps the duration plausibility
    # check happy, so the only blocking failure is the silence itself.
    silent = make_silence_wav(tmp_path / "silent.wav", seconds=0.2)
    checks = qa.run_tier1_checks(audio_path=silent, text="hello")
    assert any(
        item.check_id == "not_silent" and item.status == "failed" for item in checks
    )
    assert next(
        item for item in checks if item.check_id == "coverage_plausible"
    ).status == "passed"
    report = qa.build_report("take_silent", qa.sha256_file(silent), checks)
    assert report.passed is False
    assert [item.check_id for item in report.blocking_failures()] == ["not_silent"]

    # a full-scale square wave is technically not silent but risks clipping
    clipped = tmp_path / "clipped.wav"
    import struct
    import wave

    frames = 2400
    with wave.open(str(clipped), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(struct.pack(f"<{frames}h", *([32767, -32768] * (frames // 2))))
    clipped_checks = qa.run_tier1_checks(audio_path=clipped, text="hello")
    assert any(
        item.check_id == "peak_ok" and item.status == "failed" for item in clipped_checks
    )


def test_low_volume_is_a_warning_not_a_blocker(tmp_path):
    quiet = make_tone_wav(tmp_path / "quiet.wav", amplitude=0.00001, seconds=0.5)
    checks = qa.run_tier1_checks(audio_path=quiet, text="hello")
    warning = next(item for item in checks if item.check_id == "rms_ok")
    assert warning.status == "passed"
    report = qa.build_report("take_quiet", qa.sha256_file(quiet), [warning])
    assert report.passed is True


def test_text_edit_makes_the_selected_take_stale_and_blocks_release(studio):
    root, project, segments = studio
    rendering = fixture_renderer(root, project.id)

    # Render the whole project first, otherwise "not ready" would be blamed on
    # an utterance that never had a Take at all.
    pipeline.render_utterances(
        project.id, [segment.id for segment in segments], renderer=rendering
    )
    project, segments = _loaded(project.id)
    assert release.check_coverage(
        segments, project_root=pipeline.project_root(project.id)
    ).ready is True
    original_take_id = segments[0].selected_take_id

    segments[0].text = "A short utterance, edited."
    _save(project.id, segments)
    stale = pipeline.refresh_staleness(project, segments)
    assert segments[0].id in stale
    assert original_take_id in stale[segments[0].id]
    assert "spoken_text_hash" in takes.find_take(
        segments[0], original_take_id
    ).stale_components
    _save(project.id, segments)

    coverage = release.check_coverage(
        segments, project_root=pipeline.project_root(project.id)
    )
    assert coverage.ready is False
    assert any(issue.code == release.RESOLUTION_STALE for issue in coverage.issues)

    # The pipeline reloads from disk, so the edit above had to be persisted
    # first: disk state, not the caller's in-memory copy, is the authority.
    reports = pipeline.render_utterances(
        project.id,
        [segments[0].id],
        renderer=fixture_renderer(root, project.id),
    )
    assert reports[0].selected is True
    assert reports[0].take_id != original_take_id

    project, segments = _loaded(project.id)
    assert segments[0].text == "A short utterance, edited."
    assert len(segments[0].takes) == 2
    assert segments[0].selected_take_id == reports[0].take_id
    # A stale Take is kept as history; it is never deleted or silently re-selected.
    assert takes.find_take(segments[0], original_take_id).stale is True
    assert release.check_coverage(
        segments, project_root=pipeline.project_root(project.id)
    ).ready is True


def _save(project_id, segments):
    from dubbing_service.app.storage.segments import save_segments

    save_segments(project_id, segments)


def _loaded(project_id):
    from dubbing_service.app.storage.projects import load_project
    from dubbing_service.app.storage.segments import load_segments

    return load_project(project_id), load_segments(project_id)


def test_status_report_summarises_the_project(studio):
    root, project, segments = studio
    pipeline.render_segment(
        project, segments[0], renderer=fixture_renderer(root, project.id),
        root=pipeline.project_root(project.id), promote=True,
    )
    from dubbing_service.app.storage.segments import save_segments

    save_segments(project.id, segments)

    status = pipeline.status_report(project.id)
    assert status["project_id"] == project.id
    assert status["live_utterances"] == 2
    assert status["takes"] == 1
    assert status["release_ready"] is False
    assert status["coverage"]["missing"] == 1
