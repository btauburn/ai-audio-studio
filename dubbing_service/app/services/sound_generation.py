"""Pure domain operations for planning, aligning and adopting generated sound."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from dubbing_service.app.models.production import (
    AssetLicense,
    AudioAsset,
    AudioAssetOrigin,
    SoundDesign,
    SoundTrack,
    TimelineClip,
)
from dubbing_service.app.models.sound_generation import (
    SemanticAnchor,
    SemanticAnchorEdge,
    SemanticPlacement,
    SoundCandidate,
    SoundCandidateStatus,
    SoundCue,
    SoundCueReviewStatus,
    SoundPlan,
)


class SoundPlanValidationError(ValueError):
    pass


class SoundAlignmentError(ValueError):
    pass


class SoundCandidateAdoptionError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedPlacement:
    start: float
    duration: float | None


class CandidateAdoption(BaseModel):
    plan: SoundPlan
    asset: AudioAsset
    clip: TimelineClip


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    data = getattr(value, "__dict__", None)
    if isinstance(data, Mapping):
        return data
    raise SoundPlanValidationError("源分段必须是对象或映射")


def _collection(value: object, *container_keys: str) -> list[object]:
    if isinstance(value, Mapping):
        for key in container_keys:
            if key in value:
                value = value[key]
                break
    elif isinstance(value, BaseModel):
        dumped = value.model_dump(mode="json")
        for key in container_keys:
            if key in dumped:
                value = dumped[key]
                break
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise SoundPlanValidationError("源分段必须是列表")
    return list(value)


def _canonical_segment(segment: object) -> dict[str, object]:
    item = _mapping(segment)
    segment_id = str(item.get("id") or item.get("segment_id") or "").strip()
    if not segment_id:
        raise SoundPlanValidationError("源分段缺少 ID")
    return {
        "id": segment_id,
        "index": item.get("index"),
        "chapter_id": item.get("chapter_id"),
        "role_id": item.get("role_id"),
        "text": str(item.get("text") or ""),
        "emotion_mode": item.get("emotion_mode"),
        "emotion_text": item.get("emotion_text"),
        "emotion_vector": item.get("emotion_vector"),
        "emotion_alpha": item.get("emotion_alpha"),
    }


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_segment_source(segment: object) -> str:
    return _sha256_json(_canonical_segment(segment))


def fingerprint_sound_source(segments: object) -> str:
    items = [_canonical_segment(item) for item in _collection(segments, "segments", "items")]
    ids = [str(item["id"]) for item in items]
    if len(ids) != len(set(ids)):
        raise SoundPlanValidationError("源分段 ID 不能重复")
    items.sort(key=lambda item: (
        item["index"] if isinstance(item["index"], int) else 2**31,
        str(item["id"]),
    ))
    return _sha256_json(items)


def fingerprint_sound_cue(cue: SoundCue) -> str:
    """Fingerprint only the inputs that determine the generated waveform.

    Placement and mix parameters intentionally do not participate: an editor
    may slide or rebalance an adopted clip without making the underlying WAV
    stale. Source-text drift is tracked separately by ``cue.source_fingerprint``
    so a deliberate semantic re-anchor can be approved without pretending the
    WAV bytes changed.
    """

    return _sha256_json({
        "id": cue.id,
        "track": cue.track.value,
        "prompt": cue.prompt,
        "negative_prompt": cue.negative_prompt,
        "requested_duration_seconds": cue.requested_duration_seconds,
    })


def _segments_by_id(segments: object) -> tuple[list[object], dict[str, object]]:
    items = _collection(segments, "segments", "items")
    by_id: dict[str, object] = {}
    for item in items:
        canonical = _canonical_segment(item)
        segment_id = str(canonical["id"])
        if segment_id in by_id:
            raise SoundPlanValidationError("源分段 ID 不能重复")
        by_id[segment_id] = item
    return items, by_id


def bind_sound_plan(plan: SoundPlan, segments: object) -> SoundPlan:
    """Bind a newly planned cue set to the exact source text it describes."""

    items, by_id = _segments_by_id(segments)
    rebound = plan.model_copy(deep=True)
    for cue in rebound.cues:
        segment = by_id.get(cue.segment_id)
        if segment is None:
            raise SoundPlanValidationError(f"音效提示 {cue.id} 引用了不存在的分段 {cue.segment_id}")
        cue.source_fingerprint = fingerprint_segment_source(segment)
        cue.stale = False
        current_cue_fingerprint = fingerprint_sound_cue(cue)
        for candidate in cue.candidates:
            candidate.stale = candidate.cue_fingerprint != current_cue_fingerprint
    rebound.source_fingerprint = fingerprint_sound_source(items)
    rebound.stale = False
    rebound.updated_at = _utc_now()
    return SoundPlan.model_validate(rebound.model_dump(mode="python"))


def refresh_sound_plan_staleness(plan: SoundPlan, segments: object) -> SoundPlan:
    """Return a copy with conservative plan/cue/candidate stale flags refreshed."""

    items, by_id = _segments_by_id(segments)
    refreshed = plan.model_copy(deep=True)
    current_source = fingerprint_sound_source(items)
    refreshed.stale = refreshed.source_fingerprint != current_source
    for cue in refreshed.cues:
        segment = by_id.get(cue.segment_id)
        current_segment = fingerprint_segment_source(segment) if segment is not None else None
        cue.stale = cue.source_fingerprint != current_segment
        current_cue_fingerprint = fingerprint_sound_cue(cue)
        for candidate in cue.candidates:
            candidate.stale = cue.stale or candidate.cue_fingerprint != current_cue_fingerprint
        refreshed.stale = refreshed.stale or cue.stale
    return SoundPlan.model_validate(refreshed.model_dump(mode="python"))


def attach_sound_candidate(plan: SoundPlan, cue_id: str, candidate: SoundCandidate) -> SoundPlan:
    """Attach a candidate and stamp it with the cue version it was made for."""

    updated = plan.model_copy(deep=True)
    cue = next((item for item in updated.cues if item.id == cue_id), None)
    if cue is None:
        raise SoundPlanValidationError("音效提示不存在")
    if cue.stale or not cue.source_fingerprint:
        raise SoundPlanValidationError("音效提示未绑定当前源文本或已经过期")
    if any(item.id == candidate.id for item in cue.candidates):
        raise SoundPlanValidationError("候选 ID 已存在")
    attached = candidate.model_copy(deep=True)
    attached.cue_fingerprint = fingerprint_sound_cue(cue)
    attached.stale = False
    attached.updated_at = _utc_now()
    cue.candidates.append(attached)
    updated.updated_at = _utc_now()
    return SoundPlan.model_validate(updated.model_dump(mode="python"))


def _timeline_map(dialogue_timeline: object) -> dict[str, tuple[float, float]]:
    try:
        raw_items = _collection(dialogue_timeline, "items")
    except SoundPlanValidationError as exc:
        raise SoundAlignmentError(str(exc)) from exc
    result: dict[str, tuple[float, float]] = {}
    for raw in raw_items:
        try:
            item = _mapping(raw)
        except SoundPlanValidationError as exc:
            raise SoundAlignmentError(str(exc)) from exc
        segment_id = str(item.get("segment_id") or item.get("id") or "").strip()
        if not segment_id:
            raise SoundAlignmentError("dialogue_timeline 条目缺少 segment_id")
        if segment_id in result:
            raise SoundAlignmentError(f"dialogue_timeline 中分段 {segment_id} 重复")
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SoundAlignmentError(f"分段 {segment_id} 的时间范围无效") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise SoundAlignmentError(f"分段 {segment_id} 的时间范围无效")
        result[segment_id] = (start, end)
    return result


def resolve_semantic_anchor(anchor: SemanticAnchor | Mapping[str, object], dialogue_timeline: object) -> float:
    parsed = SemanticAnchor.model_validate(anchor)
    timing = _timeline_map(dialogue_timeline).get(parsed.segment_id)
    if timing is None:
        raise SoundAlignmentError(f"dialogue_timeline 缺少分段 {parsed.segment_id}")
    base = timing[0] if parsed.edge == SemanticAnchorEdge.START else timing[1]
    resolved = base + parsed.offset_seconds
    if not math.isfinite(resolved):
        raise SoundAlignmentError("语义锚点解析结果无效")
    return max(0.0, resolved)


def resolve_semantic_placement(
    placement: SemanticPlacement | Mapping[str, object],
    dialogue_timeline: object,
    *,
    fallback_duration: float | None = None,
) -> ResolvedPlacement:
    parsed = SemanticPlacement.model_validate(placement)
    start = resolve_semantic_anchor(parsed.start, dialogue_timeline)
    if parsed.end is None:
        if fallback_duration is not None and fallback_duration <= 0:
            raise SoundAlignmentError("回退时长必须大于 0")
        return ResolvedPlacement(start=start, duration=fallback_duration)
    end = resolve_semantic_anchor(parsed.end, dialogue_timeline)
    if end <= start:
        raise SoundAlignmentError("语义结束锚点必须晚于开始锚点")
    return ResolvedPlacement(start=start, duration=end - start)


def resolve_sound_design(sound_design: SoundDesign, dialogue_timeline: object) -> SoundDesign:
    """Resolve semantic clips on a deep copy; never mutate persisted settings."""

    resolved = sound_design.model_copy(deep=True)
    clips: list[TimelineClip] = []
    for clip in resolved.clips:
        if clip.semantic_placement is None:
            clips.append(clip)
            continue
        placement = resolve_semantic_placement(
            clip.semantic_placement,
            dialogue_timeline,
            fallback_duration=clip.duration,
        )
        clips.append(clip.model_copy(update={"start": placement.start, "duration": placement.duration}))
    return SoundDesign.model_validate(
        resolved.model_copy(update={"clips": clips}).model_dump(mode="python")
    )


def adopt_sound_candidate(
    plan: SoundPlan,
    cue_id: str,
    candidate_id: str,
    *,
    asset_id: str,
    clip_id: str,
    asset_name: str | None = None,
    license: AssetLicense | Mapping[str, object] | None = None,
) -> CandidateAdoption:
    """Create production objects for a ready candidate without filesystem I/O."""

    if plan.stale:
        raise SoundCandidateAdoptionError("音效计划已经过期，请先基于当前文本重新确认")
    updated = plan.model_copy(deep=True)
    cue = next((item for item in updated.cues if item.id == cue_id), None)
    if cue is None:
        raise SoundCandidateAdoptionError("音效提示不存在")
    if not cue.enabled or cue.review_status != SoundCueReviewStatus.APPROVED:
        raise SoundCandidateAdoptionError("音效提示必须启用并通过人工确认后才能采用")
    if cue.stale or not cue.source_fingerprint:
        raise SoundCandidateAdoptionError("音效提示未绑定当前源文本或已经过期")
    candidate = next((item for item in cue.candidates if item.id == candidate_id), None)
    if candidate is None:
        raise SoundCandidateAdoptionError("音效候选不存在")
    if candidate.status != SoundCandidateStatus.READY:
        raise SoundCandidateAdoptionError("只有生成完成的候选才能采用")
    if candidate.stale or candidate.cue_fingerprint != fingerprint_sound_cue(cue):
        raise SoundCandidateAdoptionError("候选对应的提示或源文本已经变化")
    if not candidate.audio_file or not candidate.audio_sha256 or candidate.duration_seconds is None:
        raise SoundCandidateAdoptionError("候选音频记录不完整")

    provenance = candidate.provenance.model_copy(deep=True)
    if license is None:
        revision = f"@{provenance.model_revision}" if provenance.model_revision else ""
        evidence = f"Generated by {provenance.provider}: {provenance.model_id}{revision}"
        if provenance.seed is not None:
            evidence += f"; seed={provenance.seed}"
        asset_license = AssetLicense(
            source=provenance.model_source,
            license_name=provenance.model_license,
            commercial_use=provenance.commercial_use,
            proof_note=evidence,
        )
    else:
        asset_license = AssetLicense.model_validate(license)

    track = SoundTrack(cue.track.value)
    asset = AudioAsset(
        id=asset_id,
        track=track,
        name=asset_name or cue.prompt[:160],
        audio_file=candidate.audio_file,
        audio_sha256=candidate.audio_sha256,
        duration_seconds=candidate.duration_seconds,
        tags=["generated", track.value, provenance.model_id],
        license=asset_license,
        origin=AudioAssetOrigin.GENERATED,
        generation=provenance,
    )
    clip = TimelineClip(
        id=clip_id,
        track=track,
        asset_id=asset_id,
        start=0.0,
        duration=cue.playback_duration_seconds or candidate.duration_seconds,
        gain_db=cue.gain_db,
        fade_in=cue.fade_in,
        fade_out=cue.fade_out,
        loop=cue.loop,
        semantic_placement=cue.placement.model_copy(deep=True),
    )

    for item in cue.candidates:
        if item.status == SoundCandidateStatus.ADOPTED:
            item.status = SoundCandidateStatus.READY
        if item.id == candidate_id:
            item.status = SoundCandidateStatus.ADOPTED
            item.updated_at = _utc_now()
    cue.selected_candidate_id = candidate_id
    cue.adopted_asset_id = asset_id
    cue.adopted_clip_id = clip_id
    updated.updated_at = _utc_now()
    return CandidateAdoption(plan=updated, asset=asset, clip=clip)
