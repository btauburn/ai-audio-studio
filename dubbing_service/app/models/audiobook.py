"""Audiobook provenance, renderer, QA and cue models (architecture v1.2).

These models extend AI Audio Studio's canonical project state; they do not
introduce a second store.  Revisions live in ``project.json``, Take-level
records live in ``segments.json`` alongside ``SegmentTake``.

Design rules enforced here:

* canonical source text, spoken text and engine representation stay distinct
  (:class:`SourceRevision`, :class:`SpokenTextRevision` and the existing
  ``SegmentTake.synthesis_text``);
* an input fingerprint (synthesis requirement) is never conflated with the
  output audio hash;
* QA results bind to ``take_id`` + ``output_audio_hash`` + ``qa_check_version``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


AttributionSource = Literal["tag", "turn", "nlp", "inline", "user", "unknown"]

CueType = Literal["dialogue", "bgm", "ambience", "sfx"]

QaStatus = Literal["passed", "failed", "skipped"]

LicenseScope = Literal["allowed", "not_allowed", "unknown", "requires_review"]


class SourceRevision(BaseModel):
    """An immutable snapshot of the canonical source that was ingested.

    ``sha256`` covers the snapshot bytes on disk, so any later edit of the
    snapshot is detectable.  ``snapshot_file`` is project-relative.
    """

    id: str
    uri: str = Field(..., description="原始来源定位符：文件路径、URL 或 fixture 名称")
    kind: Literal["file", "url", "inline"] = "file"
    media_type: Optional[str] = None
    sha256: str = Field(..., description="快照字节的 SHA256，用于校验未被篡改")
    byte_length: int = Field(0, ge=0)
    snapshot_file: Optional[str] = Field(
        None, description="项目内不可变快照相对路径"
    )
    title: Optional[str] = None
    acquired_at: datetime = Field(default_factory=utcnow)
    notes: Optional[str] = None

    def revision_key(self) -> str:
        return f"{self.sha256}:{self.byte_length}"


class SourceAnchor(BaseModel):
    """Where a live utterance came from in the canonical source snapshot."""

    source_revision_id: str
    chapter_id: Optional[str] = None
    start_offset: Optional[int] = Field(default=None, ge=0)
    end_offset: Optional[int] = Field(default=None, ge=0)
    occurrence_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="同一 anchor 在来源中第几次出现，用于重复短对白的稳定寻址",
    )


class SpokenTextRevision(BaseModel):
    """Versioned spoken text derived from canonical source text.

    Spoken text may differ from the source (pronunciation expansion, number
    normalization).  Engine-specific SSML or phoneme markup must NOT live here;
    whatever the renderer actually received stays in ``SegmentTake.synthesis_text``.
    """

    id: str
    source_revision_id: Optional[str] = None
    source_text_hash: str
    spoken_text: str
    spoken_text_hash: str
    pronunciation_revision: Optional[str] = None
    previous_revision_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    reason: Optional[str] = None


class Attribution(BaseModel):
    """Speaker attribution provenance for a live utterance."""

    speaker_id: Optional[str] = Field(None, description="角色 ID（Role.id）")
    source: AttributionSource = "unknown"
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reviewed: bool = False
    updated_at: Optional[datetime] = None
    note: Optional[str] = None

    @property
    def human_authoritative(self) -> bool:
        """True when a human decided this speaker (never auto-overwritten)."""

        return self.source == "user" and self.reviewed


class SegmentLineage(BaseModel):
    """Split/merge provenance for utterance identity.

    A semantic split or merge retires the old utterance IDs, so the original
    record keeps ``split_into`` / ``merged_into`` while each new record carries
    ``split_from`` / ``merged_from``.
    """

    split_from: Optional[str] = None
    split_into: List[str] = Field(default_factory=list)
    merged_from: List[str] = Field(default_factory=list)
    merged_into: Optional[str] = None
    retired_at: Optional[datetime] = None
    retired_reason: Optional[str] = None
    content_revision: int = Field(
        1, ge=1, description="非语义内容修订计数（普通文本编辑不改变 id）"
    )

    @property
    def retired(self) -> bool:
        return self.retired_at is not None


class ChunkRecord(BaseModel):
    """Provenance for one renderer-internal chunk reassembled into a Take."""

    index: int = Field(..., ge=0)
    text_start: int = Field(..., ge=0)
    text_end: int = Field(..., ge=0)
    audio_start_ms: int = Field(..., ge=0)
    audio_end_ms: int = Field(..., ge=0)
    chunk_sha256: Optional[str] = None


class QaCheckResult(BaseModel):
    """One deterministic or advisory QA check bound to a specific Take."""

    check_id: str
    check_version: str
    tier: int = Field(1, ge=1, le=2)
    status: QaStatus
    message: Optional[str] = None
    details: Dict = Field(default_factory=dict)
    blocking: bool = True


class QaCoverageProfile(BaseModel):
    """Identity of the duration window that governed ``coverage_plausible``.

    Only the identity is stored, never the profile itself: a re-audit needs to
    know *which* window was applied ("that Take was judged as English"), not a
    redundant copy of the numbers, which belong to the code that measured them.
    """

    language: Optional[str] = Field(
        default=None,
        description="QA 当时使用的语言（Take 上记录的解析值），为空表示未声明",
    )
    profile: str = Field(..., description="实际生效的时长窗口名，如 english/chinese")
    profile_version: str = Field(..., description="窗口实现的版本，用于判定可比性")


class TakeQaReport(BaseModel):
    """QA evaluation of one Take's exact bytes.

    ``output_audio_hash`` binds the report to the bytes that were tested, so a
    different file invalidates prior QA (see :meth:`matches`).
    """

    take_id: str
    output_audio_hash: str
    qa_check_version: str
    tier: int = Field(1, ge=1, le=2)
    results: List[QaCheckResult] = Field(default_factory=list)
    passed: bool = False
    evaluated_at: datetime = Field(default_factory=utcnow)
    notes: Optional[str] = None
    coverage_profile: Optional[QaCoverageProfile] = Field(
        default=None,
        description=(
            "coverage_plausible 所使用的语言/窗口身份；旧报告为空，"
            "表示当时没有记录，而不是使用了默认窗口"
        ),
    )

    def matches(self, take_id: str, output_audio_hash: Optional[str]) -> bool:
        """True only when this report was produced from exactly these bytes."""

        return (
            self.take_id == take_id
            and bool(output_audio_hash)
            and self.output_audio_hash == output_audio_hash
        )

    def executed_check_ids(self) -> List[str]:
        return [item.check_id for item in self.results if item.status != "skipped"]

    def blocking_failures(self) -> List[QaCheckResult]:
        return [
            item
            for item in self.results
            if item.blocking and item.status == "failed"
        ]


class AudioCue(BaseModel):
    """Future Audio Director cue anchored to an utterance, not to the timeline.

    Anchoring to ``utterance_id`` + ``relative_offset_ms`` lets a re-rendered
    Take change duration without invalidating the cue; absolute placement is
    resolved at mix time (see ``audiobook.cues.resolve_cue_window``).
    """

    id: str
    utterance_id: str = Field(..., description="AI Audio Studio Segment.id")
    source_revision_id: Optional[str] = None
    occurrence_index: int = Field(0, ge=0)
    relative_offset_ms: int = Field(0, description="相对 utterance 起点的偏移")
    relative_end_ms: Optional[int] = Field(
        None, description="相对 utterance 起点的结束偏移；为空表示到 utterance 结束"
    )
    cue_type: CueType = "sfx"
    intent: Optional[str] = None
    asset_id: Optional[str] = Field(None, description="AudioAsset.id")
    gain_db: float = 0.0
    fade_in_ms: int = Field(0, ge=0)
    fade_out_ms: int = Field(0, ge=0)
    ducking: Optional[Dict] = None
    loop: bool = False
    provenance: Dict = Field(default_factory=dict)


class EngineLicenseRecord(BaseModel):
    """Machine-readable capability/license record for one engine.

    ``commercial_use`` must never be inferred from the repository license
    alone; the four scopes are tracked separately.
    """

    engine_id: str
    display_name: str
    platform_license: str
    engine_source_license: str
    model_weight_license: str
    voice_rights: str
    commercial_use: LicenseScope = "unknown"
    is_default: bool = False
    backend_ids: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class BenchmarkCase(BaseModel):
    """One reusable benchmark fixture case (synthetic/public-safe only)."""

    case_id: str
    category: str
    title: str
    text: str
    expected_speakers: List[str] = Field(default_factory=list)
    expectations: Dict[str, object] = Field(default_factory=dict)
