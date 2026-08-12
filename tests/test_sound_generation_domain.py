from __future__ import annotations

from pathlib import Path

import pytest

from dubbing_service.app.core.config import clear_config_caches
from dubbing_service.app.models.production import (
    AudioAsset,
    AudioAssetOrigin,
    SoundDesign,
    SoundTrack,
    TimelineClip,
)
from dubbing_service.app.models.sound_generation import (
    GenerationProvenance,
    SemanticAnchor,
    SemanticAnchorEdge,
    SemanticPlacement,
    SoundCandidate,
    SoundCandidateStatus,
    SoundCue,
    SoundCueReviewStatus,
    SoundCueTrack,
    SoundPlan,
    SoundPlanDensity,
)
from dubbing_service.app.services.sound_generation import (
    SoundAlignmentError,
    SoundCandidateAdoptionError,
    adopt_sound_candidate,
    attach_sound_candidate,
    bind_sound_plan,
    refresh_sound_plan_staleness,
    resolve_semantic_anchor,
    resolve_semantic_placement,
    resolve_sound_design,
)
from dubbing_service.app.storage.sound_generation import (
    SoundGenerationStorageError,
    candidate_audio_root,
    delete_sound_plan,
    list_sound_plans,
    load_sound_plan,
    save_sound_plan,
)


def _provenance() -> GenerationProvenance:
    return GenerationProvenance(
        provider="local",
        model_id="org/model",
        model_revision="weights-v1",
        code_revision="worker-v2",
        model_source="https://example.test/model",
        model_license="Community License",
        commercial_use=True,
        license_confirmation_id="approval-42",
        license_snapshot={"accepted": True, "scope": "commercial"},
        prompt="a wooden door closes",
        negative_prompt="speech, music",
        seed=123,
        steps=80,
        guidance_scale=7.0,
        requested_duration_seconds=2.0,
        sample_rate=44100,
        parameters={"sampler": "dpmpp-3m-sde"},
    )


def _plan() -> SoundPlan:
    return SoundPlan(
        id="soundplan-1",
        project_id="project-1",
        density=SoundPlanDensity.CINEMATIC,
        cues=[SoundCue(
            id="cue-1",
            segment_id="seg-1",
            track=SoundCueTrack.SFX,
            description_cn="木门沉重地关上",
            prompt="a wooden door closes",
            negative_prompt="speech, music",
            placement=SemanticPlacement(start=SemanticAnchor(segment_id="seg-1")),
            requested_duration_seconds=2.0,
            review_status=SoundCueReviewStatus.APPROVED,
        )],
    )


def test_generated_asset_models_keep_legacy_json_compatible() -> None:
    legacy_asset = AudioAsset(
        id="old", track="sfx", name="old", audio_file="production/assets/sfx/old.wav",
        audio_sha256="a" * 64, duration_seconds=1.0,
    )
    legacy_clip = TimelineClip(id="old", track="sfx", asset_id="old")

    assert legacy_asset.origin == AudioAssetOrigin.UPLOADED
    assert legacy_asset.generation is None
    assert legacy_clip.semantic_placement is None


def test_candidate_adoption_copies_full_provenance_and_stale_prompt_is_blocked() -> None:
    segments = [{
        "id": "seg-1", "index": 0, "chapter_id": "chapter-1", "role_id": "narrator",
        "text": "他推开门，又把门重重关上。", "emotion_text": "紧张",
    }]
    plan = bind_sound_plan(_plan(), segments)
    candidate = SoundCandidate(
        id="candidate-1",
        status=SoundCandidateStatus.READY,
        audio_file="production/sound_generation/candidates/candidate-1.wav",
        audio_sha256="b" * 64,
        duration_seconds=1.8,
        provenance=_provenance(),
    )
    plan = attach_sound_candidate(plan, "cue-1", candidate)
    adopted = adopt_sound_candidate(
        plan, "cue-1", "candidate-1", asset_id="asset-1", clip_id="clip-1",
    )

    assert adopted.asset.origin == AudioAssetOrigin.GENERATED
    assert adopted.asset.generation == candidate.provenance
    assert adopted.asset.generation.license_snapshot["scope"] == "commercial"
    assert adopted.asset.license.commercial_use is True
    assert adopted.clip.semantic_placement == plan.cues[0].placement
    assert adopted.plan.cues[0].selected_candidate_id == "candidate-1"
    assert adopted.plan.cues[0].candidates[0].status == SoundCandidateStatus.ADOPTED

    changed_prompt = plan.model_copy(deep=True)
    changed_prompt.cues[0].prompt = "a quiet modern door closes"
    changed_prompt = refresh_sound_plan_staleness(changed_prompt, segments)
    assert changed_prompt.cues[0].candidates[0].stale is True
    with pytest.raises(SoundCandidateAdoptionError, match="变化"):
        adopt_sound_candidate(
            changed_prompt, "cue-1", "candidate-1", asset_id="asset-2", clip_id="clip-2",
        )

    changed_source = refresh_sound_plan_staleness(
        plan, [{**segments[0], "text": "这句话已经改写。"}],
    )
    assert changed_source.stale is True
    assert changed_source.cues[0].stale is True
    assert changed_source.cues[0].candidates[0].stale is True


def test_semantic_anchors_resolve_timeline_dict_without_mutating_design() -> None:
    timeline = {"items": [
        {"segment_id": "seg-1", "start": 1.0, "end": 3.0},
        {"segment_id": "seg-2", "start": 3.0, "end": 7.0},
    ]}
    placement = SemanticPlacement(
        start=SemanticAnchor(segment_id="seg-1", edge=SemanticAnchorEdge.END, offset_seconds=-0.25),
        end=SemanticAnchor(segment_id="seg-2", edge=SemanticAnchorEdge.END, offset_seconds=-0.5),
    )
    assert resolve_semantic_anchor(placement.start, timeline) == pytest.approx(2.75)
    assert resolve_semantic_anchor(placement.start, timeline["items"]) == pytest.approx(2.75)
    resolved = resolve_semantic_placement(placement, timeline)
    assert resolved.start == pytest.approx(2.75)
    assert resolved.duration == pytest.approx(3.75)

    semantic = TimelineClip(
        id="semantic", track=SoundTrack.AMBIENCE, asset_id="ambience", start=99,
        duration=1, loop=True, semantic_placement=placement,
    )
    legacy = TimelineClip(id="legacy", track=SoundTrack.SFX, asset_id="sfx", start=4, duration=1)
    design = SoundDesign(clips=[semantic, legacy])
    rendered = resolve_sound_design(design, timeline)

    assert rendered.clips[0].start == pytest.approx(2.75)
    assert rendered.clips[0].duration == pytest.approx(3.75)
    assert rendered.clips[1] == legacy
    assert design.clips[0].start == 99
    assert design.clips[0].duration == 1
    with pytest.raises(SoundAlignmentError, match="缺少分段"):
        resolve_semantic_anchor(SemanticAnchor(segment_id="missing"), timeline)


def test_sound_plan_storage_is_project_local_and_rejects_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    project = projects / "project-1"
    project.mkdir(parents=True)
    (project / "project.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    clear_config_caches()
    try:
        plan = _plan()
        save_sound_plan("project-1", plan)
        assert load_sound_plan("project-1", plan.id) == plan
        assert [item.id for item in list_sound_plans("project-1")] == [plan.id]
        assert candidate_audio_root("project-1", ensure=True) == (
            project / "production" / "sound_generation" / "candidates"
        )
        with pytest.raises(SoundGenerationStorageError, match="非法"):
            load_sound_plan("project-1", "../outside")
        assert delete_sound_plan("project-1", plan.id) is True
        assert delete_sound_plan("project-1", plan.id) is False
    finally:
        clear_config_caches()
