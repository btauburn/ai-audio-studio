"""Review-only pilot assembly contracts (M3, section P10).

Every mutable artifact is written inside the disposable per-run pytest root that
``tests/conftest.py`` installs.  Source Takes are synthetic CPU WAVs; no TTS, no
model and nothing is listened to.  The shared builders in this file are imported
by the ledger and blind-review suites.
"""

from __future__ import annotations

import hashlib
import json
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from dubbing_service.app.audiobook import performance_boundaries as boundaries
from dubbing_service.app.audiobook import performance_pilot as pilot
from dubbing_service.app.audiobook import takes as takes_module
from dubbing_service.app.audiobook.fingerprint import file_sha256, text_sha256
from dubbing_service.app.audiobook.voice_lab_render import _assemble_pcm
from dubbing_service.app.models.audiobook import QaCheckResult, TakeQaReport
from dubbing_service.app.models.project import Segment, SegmentTake, SourceAnchor
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import load_segments, save_segments

RATE = 24000
HOP = 240  # one 10 ms window at 24 kHz
SCOPE_SCHEMA = pilot.SCOPE_SCHEMA


# --------------------------------------------------------------------------- builders


def write_pcm(path: Path, samples: np.ndarray, *, width: int = 2, channels: int = 1,
              rate: int = RATE) -> Path:
    """Write exact PCM of a chosen width and channel count."""

    data = np.asarray(samples, dtype=np.float64)
    if channels > 1:
        data = np.repeat(data.reshape(-1, 1), channels, axis=1)
    flat = np.clip(data, -1.0, 1.0).reshape(-1)
    if width == 1:
        raw = np.round(flat * 127.0 + 128.0).astype(np.uint8).tobytes()
    elif width == 2:
        raw = np.round(flat * 32767.0).astype("<i2").tobytes()
    elif width == 3:
        whole = np.round(flat * 8388607.0).astype("<i4")
        raw = whole.view(np.uint8).reshape(-1, 4)[:, :3].copy().tobytes()
    else:
        raw = np.round(flat * 2147483647.0).astype("<i4").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(raw)
    return path


def tone_samples(frames: int, *, level: float = 0.4) -> np.ndarray:
    """Whole cycles per 10 ms window, so a window is exactly content."""

    amplitude = np.sqrt(2.0) * level
    return amplitude * np.sin(2.0 * np.pi * np.arange(frames) / HOP)


def take_wave(*, silence_windows: int = 30, content_windows: int = 60,
              tail_windows: int = 30, extra_frames: int = 0,
              width: int = 2, channels: int = 1) -> np.ndarray:
    """Silence, content, silence, with a deliberate partial final window."""

    parts = [np.zeros(silence_windows * HOP), tone_samples(content_windows * HOP),
             np.zeros(tail_windows * HOP)]
    if extra_frames:
        parts.append(tone_samples(extra_frames))
    return np.concatenate(parts)


def make_source_project(name: str = "source-book",
                        *, texts=("First line.", "Second line.")) -> tuple:
    """A canonical source project with eligible, owner-bound Takes and QA reports.

    Every Take is built the way the pipeline builds one: the live synthesis
    requirement comes from the real fingerprint owner, the segment carries a
    resolved attribution and source anchor, the role owns a LOCKED voice
    identity, and the Take records the matching source text and spoken-text
    revision.  A fixture that assigned arbitrary digests would be refused by the
    assembler, which is the point.
    """

    from dubbing_service.app.audiobook import engines, pipeline
    from dubbing_service.app.models.audiobook import Attribution
    from dubbing_service.app.models.character import CharacterVoiceIdentity
    from dubbing_service.app.models.project import Role

    project = create_project(name)
    from dubbing_service.app.models.audiobook import SourceRevision
    root = pilot.safe_project_root(project.id)
    snapshot = root / "source-fixture.txt"
    snapshot.write_text("\n".join(texts), encoding="utf-8")
    project.source_revisions = [SourceRevision(id="rev-1", uri="fixture:synthetic-text", kind="inline",
                                              sha256=file_sha256(snapshot), byte_length=snapshot.stat().st_size,
                                              snapshot_file=snapshot.name)]
    role = Role(id="narrator", name="Narrator", role_type="narrator",
                voice_identity=CharacterVoiceIdentity(
                    id="voice-narrator", role_id="narrator", status="LOCKED",
                    mechanism="qwen_reference", dossier_revision="dossier-1",
                    casting_profile_revision="casting-1",
                    design_instruction="measured and unhurried",
                    reference_audio_sha256="a" * 64,
                    model="qwen3-tts-base", model_revision="model-rev-1",
                    runtime_version="runtime-rev-1"))
    project.roles = [role]
    project.chapters = []
    save_project(project)

    segments = [
        Segment(id=f"utt{index}", chapter_id=None, role_id=role.id, index=index, text=text,
                spoken_text_revision_id="spoken-1",
                source_anchor=SourceAnchor(source_revision_id="rev-1"),
                attribution=Attribution(speaker_id=role.id, source="user"))
        for index, text in enumerate(texts)
    ]
    save_segments(project.id, segments)
    project = load_project(project.id)
    segments = load_segments(project.id)
    root = pilot.safe_project_root(project.id)
    for index, segment in enumerate(segments):
        audio = root / "audio" / "segments" / f"{segment.id}.wav"
        write_pcm(audio, take_wave(content_windows=30 + index * 10))
        digest = file_sha256(audio)
        requirement = pipeline.requirement_for(project, segment, role=role, renderer_id="fixture")
        take = SegmentTake(
            id=f"take-{segment.id}", audio_file=str(audio.relative_to(root)),
            audio_sha256=digest, text=segment.text, role_id=role.id, renderer_id="fixture",
            spoken_text_revision_id=segment.spoken_text_revision_id,
            source_text_hash=text_sha256(segment.text),
            input_fingerprint=requirement.fingerprint(),
            fingerprint_components=requirement.components(),
            generation_metadata={"synthetic_fixture": True,
                                 "character_voice_identity": {"id": role.voice_identity.id,
                                     "mechanism": "qwen_reference", "prompt_sha256": "f" * 64,
                                     "reference_audio_sha256": role.voice_identity.reference_audio_sha256},
                                 "provider_calls": [{"identity_id": role.voice_identity.id,
                                                     "prompt_sha256": "f" * 64}]},
            playback_speed=1.0, baked_speed=1.0,
            qa_report=qa_report(f"take-{segment.id}", digest))
        segment.takes = [take]
        segment.selected_take_id = take.id
    save_segments(project.id, segments)
    return load_project(project.id), load_segments(project.id)


def qa_report(take_id: str, digest: str, *, blocking: bool = False) -> TakeQaReport:
    return TakeQaReport(
        take_id=take_id, output_audio_hash=digest, qa_check_version="tier1-v1", tier=1,
        passed=not blocking,
        results=[QaCheckResult(check_id="duration", check_version="tier1-v1", status="passed",
                               blocking=True),
                 QaCheckResult(check_id="clipping", check_version="tier1-v1",
                               status="failed" if blocking else "passed", blocking=True)])


def measurement_for(path: Path) -> dict:
    """A real M2 measurement of a synthetic Take."""

    return boundaries.measure_take(path, file_sha256(path), text=None)


def boundary_row(index: int, klass: str) -> dict:
    """One M2 classification.  ``baseline_spacer_frames`` is a ROW field, not a
    classifier field: M2 reads it from the row, so putting it here would silently
    do nothing."""

    return {"schema_version": boundaries.BOUNDARY_CLASSIFICATION_SCHEMA,
            "classifier_version": boundaries.CLASSIFIER_VERSION, "class": klass,
            "features": {"index": index}, "reason": None, "a_role_id": "narrator",
            "b_role_id": "narrator", "machine_generated": True, "human_verified": False}


def timeline_row(project_id: str, utterance_id: str, take_id: str, path: Path,
                 boundary: dict | None, *, baseline_spacer_frames: int | None = None) -> dict:
    measured = measurement_for(path)
    return {
        "project_id": project_id, "utterance_id": utterance_id, "take_id": take_id,
        "audio_sha256": measured["audio_sha256"], "measurement_sha256": measured["content_sha256"],
        "sample_rate": measured["sample_rate"], "source_start_frame": 0,
        "source_end_frame": measured["frames"], "measurement": measured, "boundary": boundary,
        "baseline_spacer_frames": baseline_spacer_frames if boundary is not None else None,
    }


def scope_manifest(*, source_projects, utterance_ids, arms=("baseline", "variant"),
                   operation_id: str = "op-0001", assemblies: int = 4,
                   destination: Path | None = None, kind: str = "synthetic_test",
                   name: str = "pilot") -> dict:
    root = destination if destination is not None else pilot._projects_root()
    return {
        "schema_version": SCOPE_SCHEMA,
        "kind": "review_only",
        "operation_id": operation_id,
        "name": name,
        "scope": {"source_projects": list(source_projects),
                  "utterance_ids": list(utterance_ids), "arms": list(arms)},
        "ceilings": {"tts_calls": 0, "model_calls": 0, "assemblies": assemblies, "retries": 0,
                     "deadline": "2026-12-31T23:59:59+00:00"},
        "expected_revision_sha256": hashlib.sha256(b"revision").hexdigest(),
        "authorized_destination_root": str(root),
        "authorization": {"kind": kind, "event_id": "evt-1", "by": "synthetic",
                          "at": "2026-10-07T00:00:00+00:00"},
    }


@pytest.fixture()
def studio(tmp_path, monkeypatch):
    """A disposable projects root with one source project and one pilot."""

    from tests.audiobook_helpers import use_projects_root

    root = use_projects_root(monkeypatch, tmp_path)
    project, segments = make_source_project()
    manifest = scope_manifest(source_projects=[project.id],
                              utterance_ids=[segment.id for segment in segments],
                              destination=root)
    created = pilot.create_pilot("pilot", manifest)
    return {"root": root, "project": load_project(project.id), "segments": segments,
            "pilot_id": created["pilot_id"], "manifest": manifest}


def source_path(project, segment) -> Path:
    return pilot.safe_project_root(project.id) / takes_module.selected_take(segment).audio_file


def build_inputs(studio, *, klass="D_D_REPLY", allow_lead_offset=False,
                 baseline_spacer_frames=0, hints=None) -> dict:
    rows = []
    for index, segment in enumerate(studio["segments"]):
        rows.append(timeline_row(studio["project"].id, segment.id,
                                 takes_module.selected_take(segment).id,
                                 source_path(studio["project"], segment),
                                 None if index == 0 else boundary_row(index, klass),
                                 baseline_spacer_frames=baseline_spacer_frames))
    return {"policy": boundaries.default_policy(), "hints": hints or {},
            "allow_lead_offset": allow_lead_offset, "rows": rows}


def prepare(studio, **kwargs) -> tuple[dict, dict]:
    inputs = build_inputs(studio, **kwargs)
    pilot.store_timeline_inputs(studio["pilot_id"], inputs)
    # Assembly rechecks the operation reservation before publishing, so the
    # workflow opens one exactly as a real caller would.
    pilot.reserve_operation(studio["pilot_id"], studio["manifest"]["operation_id"],
                            assemblies=1)
    timeline = boundaries.compile_timeline(inputs["rows"], policy=inputs["policy"],
                                           hints=inputs["hints"],
                                           allow_lead_offset=inputs["allow_lead_offset"])
    return inputs, timeline


# --------------------------------------------------------------------------- unit C


def test_a_zero_adjustment_baseline_is_byte_identical_to_the_render_owner(studio, tmp_path) -> None:
    _inputs, timeline = prepare(studio, klass="D_D_REPLY")
    # Every boundary is left alone, so assembly must be a plain concatenation.
    assert all(item["spacer_frames"] == 0 for item in timeline["boundaries"])
    assert all(item["offset_frames"] == 0 for item in timeline["boundaries"])
    pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    directory = pilot.safe_project_root(studio["pilot_id"]) / "performance" / "pilots" \
        / studio["pilot_id"] / "baseline" / timeline["timeline_sha256"]
    assembled = (directory / "review-master.wav").read_bytes()

    sources = [source_path(studio["project"], segment) for segment in studio["segments"]]
    reference = tmp_path / "reference.wav"
    _assemble_pcm(sources, reference)
    assert assembled == reference.read_bytes()


@pytest.mark.parametrize(("width", "channels"), [(1, 1), (2, 1), (3, 1), (4, 1), (2, 2)])
def test_every_supported_width_and_channel_assembles_frame_exactly(studio, tmp_path,
                                                                    width: int,
                                                                    channels: int) -> None:
    root = pilot.safe_project_root(studio["project"].id)
    sources = []
    expected_frames = 0
    segments = load_segments(studio["project"].id)
    for index, segment in enumerate(segments):
        path = root / "audio" / "segments" / f"{segment.id}.wav"
        write_pcm(path, take_wave(content_windows=10 + index * 5), width=width, channels=channels)
        take = takes_module.selected_take(segment)
        take.audio_sha256 = file_sha256(path)
        take.qa_report = qa_report(take.id, take.audio_sha256)
        with wave.open(str(path), "rb") as handle:
            expected_frames += handle.getnframes()
        sources.append(path)
    save_segments(studio["project"].id, segments)

    _inputs, timeline = prepare(studio, klass="D_D_REPLY")
    pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    directory = pilot.safe_project_root(studio["pilot_id"]) / "performance" / "pilots" \
        / studio["pilot_id"] / "baseline" / timeline["timeline_sha256"]
    with wave.open(str(directory / "review-master.wav"), "rb") as handle:
        assert handle.getnframes() == expected_frames
        assert handle.getnchannels() == channels
        assert handle.getsampwidth() == width
        assert handle.getframerate() == RATE
    reference = tmp_path / f"reference-{width}-{channels}.wav"
    _assemble_pcm(sources, reference)
    assert (directory / "review-master.wav").read_bytes() == reference.read_bytes()

    # No analysis was performed on the assembly target beyond this decode.
    assert (directory / "source-to-audio-map.json").is_file()


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_a_partial_final_frame_survives_assembly(studio, tmp_path, width: int) -> None:
    root = pilot.safe_project_root(studio["project"].id)
    path = root / "audio" / "segments" / f"partial-{width}.wav"
    write_pcm(path, np.concatenate([np.zeros(3 * HOP), tone_samples(2 * HOP) + 1e-4,
                                    np.zeros(HOP), tone_samples(1)]), width=width)
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        params = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
    assert frames == 6 * HOP + 1
    out = tmp_path / f"copy-{width}.wav"
    _assemble_pcm([path], out)
    with wave.open(str(out), "rb") as handle:
        assert handle.getnframes() == frames
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == params
    assert out.read_bytes() == path.read_bytes()


def test_the_map_accounts_for_every_frame_and_matches_the_decoded_master(studio) -> None:
    # D_D_AFTER_TRAILOFF targets 650 ms against 600 ms of existing edges, so this
    # plan carries a real spacer and exercises spacer accounting.
    _inputs, timeline = prepare(studio, klass="D_D_AFTER_TRAILOFF")
    assert any(item["spacer_frames"] > 0 for item in timeline["boundaries"])
    receipt = pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    directory = pilot.safe_project_root(studio["pilot_id"]) / "performance" / "pilots" \
        / studio["pilot_id"] / "baseline" / timeline["timeline_sha256"]
    mapping = json.loads((directory / "source-to-audio-map.json").read_text(encoding="utf-8"))
    assert mapping["total_frames"] == sum(row["frames"] for row in mapping["rows"])
    with wave.open(str(directory / "review-master.wav"), "rb") as handle:
        assert handle.getnframes() == mapping["total_frames"]
    assert receipt["total_frames"] == mapping["total_frames"]
    # Every selected source frame appears once, in order.
    for row in mapping["rows"]:
        if row["kind"] == "speech":
            assert row["effective_source_start_frame"] == \
                row["source_start_frame"] + row["offset_frames"]
            assert row["source_end_frame"] - row["effective_source_start_frame"] == row["frames"]


def test_a_forged_timeline_output_is_refused(studio) -> None:
    _inputs, timeline = prepare(studio, klass="D_D_REPLY")
    forged = json.loads(json.dumps(timeline))
    forged["placements"][0]["derived_frames"] += 1
    # The hash field is untouched, so only recompilation can catch it.
    assert forged["timeline_sha256"] == timeline["timeline_sha256"]
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.assemble_review(studio["pilot_id"], forged, arm="baseline")
    assert refused.value.code == "plan_stale"


def _mutate_and_reload(studio, mutate) -> list:
    """Apply a mutation through the canonical segment store and reload."""

    project_id = studio["project"].id
    segments = load_segments(project_id)
    mutate(segments[0])
    save_segments(project_id, segments)
    return load_segments(project_id)


def _expect_assembly_refusal(studio, timeline) -> str:
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    return refused.value.code


def test_a_changed_qa_report_before_publish_is_refused(studio) -> None:
    _inputs, timeline = prepare(studio, klass="D_D_REPLY")

    def mutate(segment):
        take = takes_module.selected_take(segment)
        take.qa_report = qa_report(take.id, take.audio_sha256, blocking=True)

    reloaded = _mutate_and_reload(studio, mutate)
    # The storage owner withdraws a selection its shared rule rejects, so the
    # refusal is durable rather than a caller-side note.
    assert reloaded[0].selected_take_id is None
    assert _expect_assembly_refusal(studio, timeline) == "input_ineligible"


def test_a_changed_selected_take_before_publish_is_refused(studio) -> None:
    _inputs, timeline = prepare(studio, klass="D_D_REPLY")

    def mutate(segment):
        segment.selected_take_id = None

    assert _mutate_and_reload(studio, mutate) is not None
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    assert refused.value.code == "input_ineligible"
    assert "no selected Take" in str(refused.value)


def test_a_derived_take_is_refused(studio) -> None:
    _inputs, timeline = prepare(studio, klass="D_D_REPLY")

    def mutate(segment):
        takes_module.selected_take(segment).source_take_id = "take-origin"

    _mutate_and_reload(studio, mutate)
    assert _expect_assembly_refusal(studio, timeline) == "input_ineligible"


def test_a_stale_take_is_refused(studio) -> None:
    _inputs, timeline = prepare(studio, klass="D_D_REPLY")

    def mutate(segment):
        takes_module.selected_take(segment).stale = True

    reloaded = _mutate_and_reload(studio, mutate)
    assert reloaded[0].selected_take_id is None
    assert _expect_assembly_refusal(studio, timeline) == "input_ineligible"


def test_the_eligibility_rule_refuses_each_ineligible_take_shape_directly(studio) -> None:
    """The specific branches, proved on in-memory state the storage owner would withdraw."""

    project_id = studio["project"].id
    project = load_project(project_id)
    base = load_segments(project_id)
    digest = takes_module.selected_take(base[0]).audio_sha256

    def check(mutate, expected_code):
        probe = load_segments(project_id)
        mutate(probe[0])
        with pytest.raises(pilot.PerformanceDirectorError) as refused:
            pilot._eligible_source(project, probe, probe[0].id, expected_audio=digest)
        assert refused.value.code == expected_code, str(refused.value)

    check(lambda segment: setattr(takes_module.selected_take(segment), "stale", True),
          "plan_stale")
    check(lambda segment: setattr(takes_module.selected_take(segment), "source_take_id", "t"),
          "input_ineligible")
    check(lambda segment: setattr(segment, "source_anchor", None), "attribution_unresolved")
    check(lambda segment: setattr(segment, "attribution", None), "attribution_unresolved")
    check(lambda segment: setattr(segment, "text", "   "), "input_ineligible")
    check(lambda segment: setattr(segment, "playback_speed", 1.25), "input_ineligible")

    def blocking(segment):
        take = takes_module.selected_take(segment)
        take.qa_report = qa_report(take.id, take.audio_sha256, blocking=True)

    check(blocking, "input_ineligible")

    def stale_qa(segment):
        take = takes_module.selected_take(segment)
        take.qa_report = qa_report(take.id, "1" * 64)

    # A QA report bound to other bytes is a measurement hold, not a pass.
    check(stale_qa, "measurement_uncertain")

    def missing_qa(segment):
        takes_module.selected_take(segment).qa_report = None

    check(missing_qa, "measurement_uncertain")

    def mismatched_bytes(segment):
        take = takes_module.selected_take(segment)
        take.audio_sha256 = "0" * 64
        take.qa_report = qa_report(take.id, "0" * 64)

    check(mismatched_bytes, "plan_stale")

    def wrong_role(segment):
        takes_module.selected_take(segment).role_id = "someone-else"

    check(wrong_role, "plan_stale")

    def wrong_source_text(segment):
        takes_module.selected_take(segment).source_text_hash = "0" * 64

    check(wrong_source_text, "plan_stale")

    def wrong_spoken_revision(segment):
        takes_module.selected_take(segment).spoken_text_revision_id = "spoken-2"

    check(wrong_spoken_revision, "plan_stale")

    def stale_requirement(segment):
        takes_module.selected_take(segment).input_fingerprint = "0" * 64

    check(stale_requirement, "plan_stale")

    def unresolved_voice(segment):
        takes_module.selected_take(segment).fingerprint_components = {}

    check(unresolved_voice, "plan_stale")

    def unlocked_voice(segment):
        segment.role_id = "unlocked"
        probe_role = project.roles[0].model_copy(deep=True)
        probe_role.id = "unlocked"
        probe_role.voice_identity.status = "REVIEW"
        project.roles.append(probe_role)
        takes_module.selected_take(segment).role_id = "unlocked"
        takes_module.selected_take(segment).source_text_hash = text_sha256(segment.text)
        segment.attribution.speaker_id = "unlocked"

    check(unlocked_voice, "plan_stale")


def test_mixed_pcm_formats_are_refused(studio) -> None:
    project_id = studio["project"].id
    segments = load_segments(project_id)
    take = takes_module.selected_take(segments[1])
    path = pilot.safe_project_root(project_id) / take.audio_file
    write_pcm(path, tone_samples(4 * HOP), rate=48000)
    take.audio_sha256 = file_sha256(path)
    take.qa_report = qa_report(take.id, take.audio_sha256)
    save_segments(project_id, segments)
    # The accepted M2 compiler refuses a timeline whose rows sit in two sample
    # domains, so this never reaches assembly.
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        prepare(studio, klass="D_D_REPLY")
    assert refused.value.code == "input_ineligible"


def test_the_assembler_refuses_mixed_pcm_formats_directly(studio, tmp_path) -> None:
    """Defence in depth: the low-level writer refuses mixed params itself."""

    wide = write_pcm(tmp_path / "a.wav", tone_samples(4 * HOP), width=2)
    odd = write_pcm(tmp_path / "b.wav", tone_samples(4 * HOP), width=1)
    timeline = {"placements": [
        {"kind": "speech", "utterance_id": "u1", "source_start_frame": 0,
         "source_end_frame": 4 * HOP, "offset_frames": 0},
        {"kind": "speech", "utterance_id": "u2", "source_start_frame": 0,
         "source_end_frame": 4 * HOP, "offset_frames": 0},
    ]}
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot._assemble_timeline_pcm(timeline, {"u1": wide, "u2": odd}, tmp_path / "out.wav")
    assert refused.value.code == "input_ineligible"


def test_an_offset_never_exceeds_its_bounds(studio) -> None:
    inputs, timeline = prepare(studio, klass="D_D_REPLY", allow_lead_offset=True)
    for item in timeline["boundaries"]:
        leading = inputs["rows"][1]["measurement"]["leading"]["silent_frames"]
        assert item["offset_frames"] <= leading
        assert item["offset_frames"] >= 0


def test_an_arm_outside_the_approved_scope_is_refused(studio) -> None:
    _inputs, timeline = prepare(studio, klass="D_D_REPLY")
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.assemble_review(studio["pilot_id"], timeline, arm="gamma")
    assert refused.value.code == "input_ineligible"

    # A pilot whose approved scope lists only the baseline arm refuses "variant"
    # even though both arms exist in the taxonomy.
    other = dict(studio["manifest"])
    other["operation_id"] = "op-baseline-only"
    other["scope"] = {**studio["manifest"]["scope"], "arms": ["baseline"]}
    created = pilot.create_pilot("baseline-only", other)
    pilot.store_timeline_inputs(created["pilot_id"],
                               build_inputs(studio, klass="D_D_REPLY"))
    pilot.reserve_operation(created["pilot_id"], other["operation_id"], assemblies=1)
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.assemble_review(created["pilot_id"], timeline, arm="variant")
    assert refused.value.code == "input_ineligible"


def test_reassembly_reuses_verified_output_and_refuses_damage(studio) -> None:
    _inputs, timeline = prepare(studio, klass="D_D_REPLY")
    first = pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    second = pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    assert second["receipt_sha256"] == first["receipt_sha256"]

    directory = pilot.safe_project_root(studio["pilot_id"]) / "performance" / "pilots" \
        / studio["pilot_id"] / "baseline" / timeline["timeline_sha256"]
    master = directory / "review-master.wav"
    master.write_bytes(master.read_bytes() + b"\x00\x00")
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    assert refused.value.code == "input_ineligible"


def test_assembly_leaves_the_source_project_untouched(studio) -> None:
    project_id = studio["project"].id
    before = (pilot.safe_project_root(project_id) / "project.json").read_bytes()
    digests = {}
    for segment in load_segments(project_id):
        path = pilot.safe_project_root(project_id) / takes_module.selected_take(segment).audio_file
        digests[segment.id] = file_sha256(path)
    _inputs, timeline = prepare(studio, klass="D_D_REPLY")
    pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    assert (pilot.safe_project_root(project_id) / "project.json").read_bytes() == before
    for segment in load_segments(project_id):
        path = pilot.safe_project_root(project_id) / takes_module.selected_take(segment).audio_file
        assert file_sha256(path) == digests[segment.id]


def test_the_receipt_is_review_only_and_promises_nothing(studio) -> None:
    _inputs, timeline = prepare(studio, klass="D_D_REPLY")
    receipt = pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    assert receipt["kind"] == "review_only"
    assert receipt["production_enablement"] is False
    assert receipt["human_review"] == "pending"
    assert receipt["inputs"]
    for proof in receipt["inputs"]:
        assert len(proof["audio_sha256"]) == 64
    assert set(receipt["artifacts"]) == {"review-master.wav", "listening.mp3",
                                         "source-to-audio-map.json"}

@pytest.fixture()
def three_studio(tmp_path, monkeypatch):
    """A disposable root with three approved utterances, so a separator fits."""

    from tests.audiobook_helpers import use_projects_root

    root = use_projects_root(monkeypatch, tmp_path)
    project, segments = make_source_project("three-book", texts=("One.", "Two.", "Three."))
    manifest = scope_manifest(source_projects=[project.id],
                              utterance_ids=[segment.id for segment in segments],
                              destination=root)
    created = pilot.create_pilot("pilot", manifest)
    return {"root": root, "project": load_project(project.id), "segments": segments,
            "pilot_id": created["pilot_id"], "manifest": manifest}


# --------------------------------------------------- repair regressions (M3 review)

def test_an_utterance_outside_the_approved_scope_is_refused_at_assembly(studio) -> None:
    """The manifest's exact ordered utterances are enforced where audio is published."""

    inputs, timeline = prepare(studio, klass="D_D_REPLY")
    # One extra row, appended to the stored inputs: the compiler still accepts
    # it, so only the assembler can notice that the pilot never approved it.
    extra = dict(inputs["rows"][-1])
    extra["utterance_id"] = "utt-not-approved"
    inputs["rows"] = list(inputs["rows"]) + [extra]
    pilot.store_timeline_inputs(studio["pilot_id"], inputs)
    forged = boundaries.compile_timeline(inputs["rows"], policy=inputs["policy"],
                                         hints=inputs["hints"], allow_lead_offset=False)
    assert _expect_assembly_refusal(studio, forged) == "input_ineligible"

    # A missing row and a reordered pair are refused the same way.
    reordered = [dict(inputs["rows"][1]), dict(inputs["rows"][0])]
    reordered[0]["boundary"] = None
    reordered[0]["baseline_spacer_frames"] = None
    reordered[1]["boundary"] = boundary_row(1, "D_D_REPLY")
    reordered[1]["baseline_spacer_frames"] = None
    for rows in ([{**inputs["rows"][0], "boundary": None, "baseline_spacer_frames": None}],
                 reordered):
        trimmed = {**inputs, "rows": rows}
        pilot.store_timeline_inputs(studio["pilot_id"], trimmed)
        forged = boundaries.compile_timeline(rows, policy=trimmed["policy"],
                                             hints=trimmed["hints"], allow_lead_offset=False)
        assert _expect_assembly_refusal(studio, forged) == "input_ineligible"


def test_the_baseline_arm_keeps_exactly_the_declared_baseline_spacing(studio) -> None:
    """``baseline`` applies no director timing decision; ``variant`` applies the plan."""

    inputs = build_inputs(studio, klass="D_D_AFTER_TRAILOFF")
    pilot.store_timeline_inputs(studio["pilot_id"], inputs)
    pilot.reserve_operation(studio["pilot_id"], studio["manifest"]["operation_id"], assemblies=2)
    plan = boundaries.compile_timeline(inputs["rows"], policy=inputs["policy"],
                                       hints=inputs["hints"], allow_lead_offset=False)
    boundary = plan["boundaries"][0]
    assert boundary["spacer_frames"] > 0, "this fixture must give the director something to do"
    assert boundary["baseline_spacer_frames"] == 0

    sources = [row["source_end_frame"] - row["source_start_frame"] for row in inputs["rows"]]
    baseline = pilot.assemble_review(studio["pilot_id"], plan, arm="baseline")
    variant = pilot.assemble_review(studio["pilot_id"], plan, arm="variant")

    assert baseline["total_frames"] == sum(sources)
    assert variant["total_frames"] == plan["total_frames"] == sum(sources) + boundary["spacer_frames"]
    assert baseline["artifacts"]["review-master.wav"] != variant["artifacts"]["review-master.wav"]
    editions = pilot.pilot_editions(studio["pilot_id"])
    arms = {info["arm"] for info in editions.values()}
    assert arms == {"baseline", "variant"}


def test_a_stereo_spacer_and_separator_decode_to_their_declared_frames(three_studio) -> None:
    """Silence is counted per channel, so a multi-channel spacer is never short."""

    project_id = three_studio["project"].id
    segments = load_segments(project_id)
    for segment in segments:
        path = source_path(three_studio["project"], segment)
        write_pcm(path, take_wave(), channels=2)
        take = takes_module.selected_take(segment)
        take.audio_sha256 = file_sha256(path)
        take.qa_report = qa_report(take.id, take.audio_sha256)
    save_segments(project_id, segments)
    studio = {**three_studio, "segments": load_segments(project_id),
              "project": load_project(project_id)}

    classes = {1: boundaries.SCENE_SEPARATOR, 2: "UNKNOWN"}
    rows = []
    for index, segment in enumerate(studio["segments"]):
        rows.append(timeline_row(project_id, segment.id, takes_module.selected_take(segment).id,
                                 source_path(studio["project"], segment),
                                 None if index == 0 else boundary_row(index, classes[index]),
                                 baseline_spacer_frames=240))
    inputs = {"policy": boundaries.default_policy(), "hints": {}, "allow_lead_offset": False,
              "rows": rows}
    pilot.store_timeline_inputs(studio["pilot_id"], inputs)
    pilot.reserve_operation(studio["pilot_id"], studio["manifest"]["operation_id"], assemblies=1)
    timeline = boundaries.compile_timeline(rows, policy=inputs["policy"], hints={},
                                           allow_lead_offset=False)
    separators = [item for item in timeline["placements"] if item["kind"] == "scene_separator"]
    spacers = [item for item in timeline["placements"] if item["kind"] == "spacer"]
    assert separators and spacers and spacers[0]["frames"] == 240

    receipt = pilot.assemble_review(studio["pilot_id"], timeline, arm="variant")
    assert receipt["total_frames"] == timeline["total_frames"]
    directory = (pilot.safe_project_root(studio["pilot_id"]) / "performance" / "pilots"
                 / studio["pilot_id"] / "variant" / timeline["timeline_sha256"])
    with wave.open(str(directory / "review-master.wav"), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getnframes() == timeline["total_frames"]
    mapping = json.loads((directory / "source-to-audio-map.json").read_text(encoding="utf-8"))
    assert mapping["channels"] == 2
    assert mapping["total_frames"] == timeline["total_frames"]
    assert sum(row["frames"] for row in mapping["rows"]) == timeline["total_frames"]
    for row in mapping["rows"]:
        if row["kind"] == "spacer":
            assert row["frames"] == 240
        elif row["kind"] == "scene_separator":
            assert row["frames"] == separators[0]["frames"]


def test_cancellation_during_encoding_never_publishes_a_receipt(studio, monkeypatch) -> None:
    """The durable operation is re-read after encoding, before the first published byte."""

    _inputs, timeline = prepare(studio, klass="D_D_REPLY")
    original = pilot._encode_mp3
    seen = {}

    def cancel_then_encode(master, staging):
        data = original(master, staging)
        pilot.complete_operation(studio["pilot_id"], studio["manifest"]["operation_id"],
                                 outcome="cancelled")
        seen["staging"] = staging
        return data

    monkeypatch.setattr(pilot, "_encode_mp3", cancel_then_encode)
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    assert refused.value.code == "plan_stale"

    directory = (pilot.safe_project_root(studio["pilot_id"]) / "performance" / "pilots"
                 / studio["pilot_id"] / "baseline" / timeline["timeline_sha256"])
    assert not (directory / "receipt.json").exists()
    assert not (directory / "review-master.wav").exists()
    # Interrupted staging is retained as evidence, never cleaned up.
    assert seen["staging"].exists()
    assert (seen["staging"] / "review-master.wav").exists()


def test_a_changed_source_between_staging_and_publication_is_refused(studio, monkeypatch) -> None:
    """A Take that moves while the derivative encodes cannot be promoted."""

    project_id = studio["project"].id
    _inputs, timeline = prepare(studio, klass="D_D_REPLY")
    original = pilot._encode_mp3

    def damage_then_encode(master, staging):
        data = original(master, staging)
        segments = load_segments(project_id)
        take = takes_module.selected_take(segments[0])
        path = source_path(load_project(project_id), segments[0])
        write_pcm(path, take_wave(content_windows=44))
        take.audio_sha256 = file_sha256(path)
        take.qa_report = qa_report(take.id, take.audio_sha256)
        save_segments(project_id, segments)
        return data

    monkeypatch.setattr(pilot, "_encode_mp3", damage_then_encode)
    assert _expect_assembly_refusal(studio, timeline) == "plan_stale"
    directory = (pilot.safe_project_root(studio["pilot_id"]) / "performance" / "pilots"
                 / studio["pilot_id"] / "baseline" / timeline["timeline_sha256"])
    assert not (directory / "receipt.json").exists()


def test_an_unattributed_project_cannot_be_assembled(studio) -> None:
    """Why the pre-review factory is refused: no canonical cast, no eligibility."""

    project_id = studio["project"].id
    project = load_project(project_id)
    project.roles = []
    save_project(project)
    _inputs, timeline = prepare(studio, klass="D_D_REPLY")
    assert load_project(project_id).roles == []
    assert _expect_assembly_refusal(studio, timeline) == "attribution_unresolved"


def test_an_unlocked_or_replaced_voice_identity_is_refused(studio) -> None:
    project_id = studio["project"].id
    _inputs, timeline = prepare(studio, klass="D_D_REPLY")

    project = load_project(project_id)
    project.roles[0].voice_identity.status = "REVIEW"
    save_project(project)
    assert _expect_assembly_refusal(studio, timeline) == "plan_stale"

    project = load_project(project_id)
    project.roles[0].voice_identity.status = "LOCKED"
    project.roles[0].voice_identity.reference_audio_sha256 = "b" * 64
    project.roles[0].voice_identity.model_revision = "model-rev-2"
    save_project(project)
    # The Take's recorded requirement identity no longer matches its cast voice.
    assert _expect_assembly_refusal(studio, timeline) == "plan_stale"


def test_a_changed_scope_cannot_reuse_an_allocation(studio) -> None:
    """The allocation binds the complete scope, not just its name."""

    changed = dict(studio["manifest"])
    changed["scope"] = {**studio["manifest"]["scope"], "arms": ["baseline"]}
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.create_pilot("pilot", changed)
    assert refused.value.code == "direction_conflict"


def test_an_unpublished_digest_leaves_the_pair_unresolved(studio) -> None:
    """Arbitrary digests are not editions: a pair is only resolved by published artifacts."""

    assert pilot.pilot_editions(studio["pilot_id"]) == {}
    public, key = pilot.blind_pairs(studio["pilot_id"],
                                    [{"pair_id": "pair-x", "baseline_audio_sha256": "b" * 64,
                                      "variant_audio_sha256": "c" * 64}], seed=1)
    assert key["key"][0]["resolved"] is False
    assert key["key"][0]["x_edition"] is None and key["key"][0]["y_edition"] is None
    assert pilot.improvement_gate(studio["pilot_id"], pair_id="pair-x")["supports_improvement"] is False
    # The public index stays opaque and carries no arm name.
    assert set(public["pairs"][0]) == {"pair_id", "label", "x_audio_sha256", "y_audio_sha256"}


# Independent-review regressions, retained in the repository's M3 suite.
import copy

def test_strict_cast_without_current_frozen_approval_refused(studio):
    _, timeline = prepare(studio)
    source = load_project(studio["project"].id)
    source.voice_lab.casting["enabled"] = True
    source.voice_lab.casting["productions"] = {}
    save_project(source)
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")

def test_qwen_conditioning_receipt_missing_refused(studio):
    _, timeline = prepare(studio)
    segments = load_segments(studio["project"].id)
    for segment in segments:
        take = pilot.takes.selected_take(segment)
        take.generation_metadata = {}
    save_segments(studio["project"].id, segments)
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")

def test_reuse_rechecks_complete_source_proof(studio):
    _, timeline = prepare(studio)
    pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    segments = load_segments(studio["project"].id)
    segments[0].source_anchor.source_revision_id = "different-source-revision"
    save_segments(studio["project"].id, segments)
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")

def test_changed_inputs_during_encode_refused(studio, monkeypatch):
    inputs, timeline = prepare(studio)
    original = pilot._encode_mp3
    def encode_then_change(master, staging):
        data = original(master, staging)
        changed = copy.deepcopy(inputs)
        changed["policy"]["separator_ms"] += 1
        pilot.store_timeline_inputs(studio["pilot_id"], changed)
        return data
    monkeypatch.setattr(pilot, "_encode_mp3", encode_then_change)
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")

def test_operation_deadline_is_checked_at_publication(studio, monkeypatch):
    inputs, timeline = prepare(studio)
    original = pilot._encode_mp3
    def encode_then_expire(master, staging):
        data = original(master, staging)
        project = load_project(studio["pilot_id"])
        project.metadata[pilot.PILOT_LEDGER_KEY]["operations"][studio["manifest"]["operation_id"]]["payload"]["deadline"] = "2000-01-01T00:00:00+00:00"
        save_project(project)
        return data
    monkeypatch.setattr(pilot, "_encode_mp3", encode_then_expire)
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")

def test_missing_canonical_source_revision_refused(studio):
    _, timeline = prepare(studio)
    source = load_project(studio["project"].id)
    source.source_revisions = []
    save_project(source)
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")

def test_self_hashed_forged_measurement_is_refused(studio):
    from dubbing_service.app.audiobook import performance_boundaries as boundaries
    inputs, _ = prepare(studio)
    measurement = inputs["rows"][0]["measurement"]
    measurement["detector"]["dc_offset"] += 0.123
    measurement["content_sha256"] = pilot.content_sha256(measurement)
    inputs["rows"][0]["measurement_sha256"] = measurement["content_sha256"]
    pilot.store_timeline_inputs(studio["pilot_id"], inputs)
    timeline = boundaries.compile_timeline(inputs["rows"], policy=inputs["policy"], hints=inputs["hints"], allow_lead_offset=False)
    with pytest.raises(pilot.PerformanceDirectorError) as refused:
        pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    assert refused.value.code == "measurement_uncertain"

def test_reuse_rechecks_valid_qa_identity(studio):
    _, timeline = prepare(studio)
    pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    segments = load_segments(studio["project"].id)
    pilot.takes.selected_take(segments[0]).qa_report.qa_check_version = "new-qa-version"
    save_segments(studio["project"].id, segments)
    with pytest.raises(pilot.PerformanceDirectorError):
        pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")

def test_source_transient_change_does_not_change_captured_audio(studio, monkeypatch):
    import wave
    from tests.test_performance_pilot_assembly import source_path, write_pcm, take_wave
    inputs, timeline = prepare(studio)
    path = source_path(studio["project"], studio["segments"][0])
    before = path.read_bytes()
    original = pilot._assemble_timeline_pcm
    def transient_change(plan, sources, output):
        write_pcm(path, take_wave(content_windows=7))
        try:
            return original(plan, sources, output)
        finally:
            path.write_bytes(before)  # isolated fixture simulates a transient write race
    monkeypatch.setattr(pilot, "_assemble_timeline_pcm", transient_change)
    receipt = pilot.assemble_review(studio["pilot_id"], timeline, arm="baseline")
    directory = pilot.safe_project_root(studio["pilot_id"]) / "performance" / "pilots" / studio["pilot_id"] / "baseline" / timeline["timeline_sha256"]
    with wave.open(str(path), "rb") as source, wave.open(str(directory / "review-master.wav"), "rb") as master:
        frames = source.getnframes()
        assert master.readframes(frames) == source.readframes(frames)
    assert receipt["total_frames"] == timeline["total_frames"]
