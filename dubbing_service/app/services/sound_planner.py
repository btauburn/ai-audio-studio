"""LLM-assisted, human-reviewed sound-plan drafting.

Only semantic dialogue context is sent to the configured OpenAI-compatible
provider.  Audio, credentials, local paths and existing candidate provenance
never leave the machine through this planner.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from dubbing_service.app.models.project import Project, Segment
from dubbing_service.app.models.sound_generation import (
    SemanticAnchor,
    SemanticAnchorEdge,
    SemanticPlacement,
    SoundCue,
    SoundCueTrack,
    SoundPlan,
    SoundPlanDensity,
)
from dubbing_service.app.services.sound_generation import bind_sound_plan


CHUNK_SIZE = 20
CHUNK_OVERLAP = 2
DENSITY_PER_MINUTE = {
    SoundPlanDensity.CONSERVATIVE: 2,
    SoundPlanDensity.BALANCED: 5,
    SoundPlanDensity.CINEMATIC: 8,
}


class SoundPlannerError(RuntimeError):
    pass


class _DraftCue(BaseModel):
    segment_id: str = Field(..., min_length=1, max_length=160)
    track: SoundCueTrack
    description_cn: str = Field(..., min_length=1, max_length=1000)
    prompt_en: str = Field(..., min_length=3, max_length=1000)
    start_edge: SemanticAnchorEdge = SemanticAnchorEdge.START
    start_offset_seconds: float = Field(default=0.0, ge=-30.0, le=30.0)
    end_segment_id: str | None = Field(default=None, max_length=160)
    end_edge: SemanticAnchorEdge | None = None
    end_offset_seconds: float = Field(default=0.0, ge=-30.0, le=30.0)
    source_duration_seconds: float
    gain_db: float = Field(default=0.0, ge=-30.0, le=12.0)
    fade_in: float = Field(default=0.0, ge=0.0, le=10.0)
    fade_out: float = Field(default=0.0, ge=0.0, le=10.0)

    @field_validator("prompt_en")
    @classmethod
    def prompt_must_contain_english(cls, value: str) -> str:
        clean = " ".join(value.strip().split())
        if not re.search(r"[A-Za-z]", clean):
            raise ValueError("prompt_en 必须包含英文描述")
        return clean

    @model_validator(mode="after")
    def duration_and_end_are_valid(self) -> "_DraftCue":
        if self.track == SoundCueTrack.SFX:
            if not 1.0 <= self.source_duration_seconds <= 15.0:
                raise ValueError("SFX source_duration_seconds 必须为 1–15")
            if self.end_segment_id is not None or self.end_edge is not None:
                raise ValueError("事件 SFX 不应设置结束锚点")
        else:
            if not 20.0 <= self.source_duration_seconds <= 60.0:
                raise ValueError("Ambience source_duration_seconds 必须为 20–60")
            if not self.end_segment_id or self.end_edge is None:
                raise ValueError("Ambience 必须设置结束句段和 end_edge")
        return self


class _DraftResponse(BaseModel):
    cues: list[_DraftCue] = Field(default_factory=list, max_length=80)


def default_sound_density(content_mode: str) -> SoundPlanDensity:
    return (
        SoundPlanDensity.BALANCED
        if str(content_mode or "story").lower() == "story"
        else SoundPlanDensity.CONSERVATIVE
    )


def chunk_segments(
    segments: Sequence[Segment],
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[list[Segment]]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("分块参数无效")
    ordered = sorted(segments, key=lambda item: item.index)
    if not ordered:
        return []
    chunks: list[list[Segment]] = []
    # Chapters/scenes are the natural directing boundary.  Legacy projects
    # without chapter assignments use the documented 20-item sliding window.
    groups: list[list[Segment]] = []
    if any(item.chapter_id for item in ordered):
        current: list[Segment] = []
        current_key: str | None = None
        for segment in ordered:
            key = segment.chapter_id or f"__unassigned_{segment.id}"
            if current and key != current_key:
                groups.append(current)
                current = []
            current.append(segment)
            current_key = key
        if current:
            groups.append(current)
    else:
        groups = [ordered]
    for group in groups:
        start = 0
        while start < len(group):
            chunks.append(group[start : start + size])
            if start + size >= len(group):
                break
            start += size - overlap
    return chunks


def _segment_context(project: Project, segment: Segment) -> dict[str, Any]:
    role = next((item for item in project.roles if item.id == segment.role_id), None)
    return {
        "segment_id": segment.id,
        "text": segment.text,
        "role": role.name if role else None,
        "emotion": segment.emotion_text,
    }


def _messages(
    project: Project,
    chunk: Sequence[Segment],
    density: SoundPlanDensity,
    *,
    repair_error: str | None = None,
) -> list[dict[str, str]]:
    schema_example = {
        "cues": [
            {
                "segment_id": "原样使用输入中的 ID",
                "track": "sfx",
                "description_cn": "中文编导说明",
                "prompt_en": "A precise English sound-only description, no speech, no music",
                "start_edge": "start",
                "start_offset_seconds": 0.0,
                "end_segment_id": None,
                "end_edge": None,
                "end_offset_seconds": 0.0,
                "source_duration_seconds": 4.0,
                "gain_db": 0.0,
                "fade_in": 0.0,
                "fade_out": 0.1,
            }
        ]
    }
    constraints = (
        "你是有声作品声音设计编导。只规划前景音效 sfx 和环境声 ambience，不规划音乐。"
        "宁缺毋滥，所有 cue 都必须由人批准后才会生成。"
        "description_cn 用中文；prompt_en 必须用具体英文描述，只描述声音，不含台词、旁白、嗓音或音乐。"
        "SFX 源时长 1–15 秒，只设开始锚点。Ambience 源时长 20–60 秒，必须设开始和结束锚点，并可循环。"
        "只能引用输入中真实存在的 segment_id。offset 单位秒，范围 -30 到 30。"
        f"密度档位为 {density.value}。返回一个合法 JSON 对象，不要 Markdown。"
    )
    if repair_error:
        constraints += f" 上次输出校验失败：{repair_error[:1000]}。请修复整个 JSON 后重新返回。"
    return [
        {"role": "system", "content": constraints},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "segments": [_segment_context(project, item) for item in chunk],
                    "output_schema_example": schema_example,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _decode_result(result: Any, allowed_ids: set[str]) -> _DraftResponse:
    if not getattr(result, "success", False):
        raw = getattr(result, "raw", {})
        raise SoundPlannerError(f"LLM 请求失败: {str(raw)[:1000]}")
    text = str(getattr(result, "text", "") or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        decoded = json.loads(text)
        response = _DraftResponse.model_validate(decoded)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise SoundPlannerError(f"LLM JSON 校验失败: {exc}") from exc
    unknown = sorted({cue.segment_id for cue in response.cues} - allowed_ids)
    unknown.extend(
        sorted(
            {
                cue.end_segment_id
                for cue in response.cues
                if cue.end_segment_id and cue.end_segment_id not in allowed_ids
            }
        )
    )
    if unknown:
        raise SoundPlannerError(f"LLM 引用了未知句段: {', '.join(unknown[:10])}")
    return response


def _call_chunk(
    provider: Any,
    project: Project,
    chunk: Sequence[Segment],
    density: SoundPlanDensity,
) -> _DraftResponse:
    allowed = {item.id for item in chunk}
    first_error: SoundPlannerError | None = None
    for attempt in range(2):
        result = provider.chat(
            _messages(
                project,
                chunk,
                density,
                repair_error=str(first_error) if first_error else None,
            ),
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=6000,
        )
        try:
            return _decode_result(result, allowed)
        except SoundPlannerError as exc:
            if attempt == 1:
                raise
            first_error = exc
    raise SoundPlannerError("LLM JSON 校验失败")


def _estimated_minutes(segments: Sequence[Segment]) -> float:
    # Chinese narration commonly lands near four non-whitespace characters per
    # second.  This estimate is used only for a conservative planning cap; the
    # exact placement is resolved later from the rendered dialogue timeline.
    characters = sum(len("".join(str(item.text or "").split())) for item in segments)
    return max(characters / 4.0 / 60.0, 1.0 / 60.0)


def _cue_key(cue: _DraftCue) -> tuple[str, str, str, str | None]:
    return (
        cue.segment_id,
        cue.track.value,
        re.sub(r"\W+", " ", cue.prompt_en.lower()).strip(),
        cue.end_segment_id,
    )


def _enforce_ambience_non_overlap(cues: list[_DraftCue], segments: Sequence[Segment]) -> list[_DraftCue]:
    index = {item.id: position for position, item in enumerate(sorted(segments, key=lambda item: item.index))}
    kept: list[_DraftCue] = []
    intervals: list[tuple[int, int]] = []
    for cue in cues:
        if cue.track != SoundCueTrack.AMBIENCE:
            kept.append(cue)
            continue
        start = index[cue.segment_id]
        end = index[cue.end_segment_id or cue.segment_id]
        if end < start:
            raise SoundPlannerError(
                f"Ambience {cue.segment_id} 的结束锚点早于开始锚点 {cue.end_segment_id}"
            )
        if end == start:
            start_rank = 0 if cue.start_edge == SemanticAnchorEdge.START else 1
            end_rank = 0 if cue.end_edge == SemanticAnchorEdge.START else 1
            if end_rank < start_rank or (
                end_rank == start_rank
                and cue.end_offset_seconds <= cue.start_offset_seconds
            ):
                raise SoundPlannerError(f"Ambience {cue.segment_id} 的同段结束锚点必须晚于开始锚点")
        lo, hi = start, end
        if any(not (hi < existing_lo or lo > existing_hi) for existing_lo, existing_hi in intervals):
            continue
        intervals.append((lo, hi))
        kept.append(cue)
    return kept


def draft_sound_plan(
    *,
    project: Project,
    segments: Sequence[Segment],
    density: SoundPlanDensity,
    provider: Any,
    planner_provider: str | None,
    planner_model: str | None,
) -> SoundPlan:
    """Draft and validate an entire plan in memory; caller persists on success."""

    if not segments:
        raise SoundPlannerError("项目没有可规划音效的句段")
    drafts: list[_DraftCue] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    for chunk in chunk_segments(segments):
        response = _call_chunk(provider, project, chunk, density)
        for cue in response.cues:
            key = _cue_key(cue)
            if key in seen:
                continue
            seen.add(key)
            drafts.append(cue)
    drafts = _enforce_ambience_non_overlap(drafts, segments)
    foreground = [cue for cue in drafts if cue.track == SoundCueTrack.SFX]
    cap = max(1, math.ceil(_estimated_minutes(segments) * DENSITY_PER_MINUTE[density]))
    allowed_foreground = {id(item) for item in foreground[:cap]}
    drafts = [
        cue
        for cue in drafts
        if cue.track == SoundCueTrack.AMBIENCE or id(cue) in allowed_foreground
    ]
    cues: list[SoundCue] = []
    for draft in drafts:
        end = None
        if draft.end_segment_id and draft.end_edge:
            end = SemanticAnchor(
                segment_id=draft.end_segment_id,
                edge=draft.end_edge,
                offset_seconds=draft.end_offset_seconds,
            )
        cues.append(
            SoundCue(
                id=f"cue-{uuid.uuid4().hex[:12]}",
                segment_id=draft.segment_id,
                track=draft.track,
                description_cn=draft.description_cn,
                prompt=draft.prompt_en,
                placement=SemanticPlacement(
                    start=SemanticAnchor(
                        segment_id=draft.segment_id,
                        edge=draft.start_edge,
                        offset_seconds=draft.start_offset_seconds,
                    ),
                    end=end,
                ),
                requested_duration_seconds=draft.source_duration_seconds,
                playback_duration_seconds=None,
                gain_db=draft.gain_db,
                fade_in=draft.fade_in,
                fade_out=draft.fade_out,
                loop=draft.track == SoundCueTrack.AMBIENCE,
            )
        )
    plan = SoundPlan(
        id=f"soundplan-{uuid.uuid4().hex[:12]}",
        project_id=project.id,
        density=density,
        cues=cues,
        planner_provider=planner_provider,
        planner_model=planner_model,
    )
    return bind_sound_plan(plan, list(segments))
