"""Core data models for projects, chapters, roles and segments."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .audiobook import (
    AudioCue,
    Attribution,
    ChunkRecord,
    SegmentLineage,
    SourceAnchor,
    SourceRevision,
    SpokenTextRevision,
    TakeQaReport,
)
from .enums import EmotionMode, RoleType, SegmentReviewStatus, SegmentStatus
from .character import CharacterCastState, CharacterVoiceIdentity
from .voice_lab import VoiceLabState


class Role(BaseModel):
    """配音角色（含旁白、人物等）。"""

    id: str = Field(..., description="角色唯一 ID（内部使用）")
    voice_identity: Optional[CharacterVoiceIdentity] = None
    name: str = Field(..., description="角色名称，展示给用户")
    role_type: RoleType = Field(RoleType.CHARACTER, description="角色类型")
    description: Optional[str] = Field(
        None, description="角色简介或备注，例如人物设定、身份说明等"
    )

    # 参考音频：相对于项目根目录的路径，例如 "roles/prompts/narrator.wav"
    speaker_audio: Optional[str] = Field(
        None,
        description="角色参考音频的相对路径（用于 IndexTTS2 的 speaker_audio）",
    )

    speaker_voice_id: Optional[str] = Field(
        None,
        description="角色参考音频来源的音色库条目 ID（若通过音色库绑定则记录，用于前端展示）",
    )

    # Stable role-level TTS profile. Segment fields remain performance
    # overrides; these values prevent every Qwen role from silently falling
    # back to the same built-in speaker.
    tts_backend: Optional[str] = Field(
        None,
        description="角色默认 TTS 后端；为空时使用项目/平台默认后端",
    )
    qwen3_speaker: Optional[str] = Field(
        None,
        description="Qwen3 CustomVoice 的角色默认内置音色",
    )
    qwen3_language: Optional[str] = Field(
        None,
        description="Qwen3 的角色默认语言",
    )
    voice_description: Optional[str] = Field(
        None,
        description="Qwen3 VoiceDesign 的稳定角色音色描述",
    )
    style_instruction: Optional[str] = Field(
        None,
        description="角色默认表演/风格指令，分段情感描述可在此基础上覆盖或补充",
    )
    preferred_speed_min: float = Field(
        0.9, ge=0.5, le=2.0, description="Voice Bible 建议最低语速",
    )
    preferred_speed_max: float = Field(
        1.1, ge=0.5, le=2.0, description="Voice Bible 建议最高语速",
    )
    voice_source: Optional[str] = Field(None, description="声音/参考音频来源")
    voice_license_name: Optional[str] = Field(None, description="声音授权或许可证名称")
    voice_commercial_use: Optional[bool] = Field(None, description="是否已确认声音可商用")
    voice_rights_note: Optional[str] = Field(None, description="授权凭证或使用限制备注")

    # 角色默认情感配置：用于没有单独指定情感的段落时的默认值
    default_emotion_mode: EmotionMode = Field(
        EmotionMode.TEXT, description="角色默认情感控制模式"
    )
    default_emotion_text: Optional[str] = Field(
        None, description="角色默认情感描述文本，例如：平静、温柔、冷静理性等"
    )
    default_emotion_vector: Optional[List[float]] = Field(
        None,
        description=(
            "角色默认情感向量，[快乐,愤怒,悲伤,恐惧,厌恶,忧郁,惊讶,平静]，每项 0-1"
        ),
    )
    emotion_alpha: float = Field(
        0.65,
        ge=0.0,
        le=1.0,
        description="角色默认情感强度，0-1，默认与 REST API 推荐值保持一致",
    )


class Chapter(BaseModel):
    """章节或场景。"""

    id: str = Field(..., description="章节唯一 ID")
    title: str = Field(..., description="章节标题")
    index: int = Field(0, ge=0, description="在项目中的顺序索引，从 0 开始")
    description: Optional[str] = Field(None, description="章节描述或备注")


class SegmentEditRegion(BaseModel):

    id: Optional[str] = None
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)


class SegmentEditSilence(BaseModel):

    id: Optional[str] = None
    at: float = Field(..., ge=0.0)
    duration: float = Field(..., ge=0.0)


class SegmentEditState(BaseModel):

    speed: Optional[float] = Field(default=None, ge=0.5, le=2.0)
    volume: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    region: Optional[SegmentEditRegion] = None
    regions: List[SegmentEditRegion] = Field(default_factory=list)
    selected_region_id: Optional[str] = None
    silences: List[SegmentEditSilence] = Field(default_factory=list)
    selected_silence_id: Optional[str] = None
    silence_sec: Optional[float] = Field(default=None, ge=0.0)
    fade_in: float = Field(default=0.0, ge=0.0, le=10.0)
    fade_out: float = Field(default=0.0, ge=0.0, le=10.0)
    leading_silence: float = Field(default=0.0, ge=0.0, le=30.0)
    trailing_silence: float = Field(default=0.0, ge=0.0, le=30.0)
    processing_preset: str = Field(
        default="none",
        pattern="^(none|natural_dialogue|audio_story|clear_podcast)$",
    )


class SegmentTakeSlot(BaseModel):
    """A/B 槽位的下一次生成参数和当前 Take 引用。

    槽位参数可以继续编辑；已经生成的 Take 则保存当次生成快照，二者分离后
    重新生成不会篡改历史 Take 的可追溯信息。
    """

    take_id: Optional[str] = None
    needs_synthesis: bool = Field(
        default=True,
        description="槽位参数或共享文本已变化，需要重新生成后才与当前设置一致",
    )
    tts_backend: Optional[str] = None
    qwen3_speaker: Optional[str] = None
    qwen3_language: Optional[str] = None
    emotion_mode: EmotionMode = EmotionMode.TEXT
    emotion_text: Optional[str] = None
    emotion_vector: Optional[List[float]] = None
    emotion_audio: Optional[str] = None
    emotion_alpha: float = Field(default=0.65, ge=0.0, le=1.0)


class SegmentTake(BaseModel):
    """一次可追溯的音频生成结果。

    ``audio_file`` 指向只读原始音频；编辑只写入 ``edit_state``，需要固化时
    必须创建新的派生 Take，而不能覆盖该文件。
    """

    id: str = Field(..., description="Take 唯一 ID")
    audio_file: str = Field(..., description="项目内不可变原始音频相对路径")
    audio_sha256: Optional[str] = None
    text: Optional[str] = Field(default=None, description="本次生成所使用的文本快照")
    synthesis_text: Optional[str] = Field(
        default=None,
        description="应用项目发音词典后实际发送给 TTS 的文本快照",
    )
    generation_metadata: Dict = Field(
        default_factory=dict,
        description="模型返回的非敏感生成参数、告警和可追溯信息快照",
    )
    role_id: Optional[str] = Field(default=None, description="本次生成所使用的角色快照")
    tts_backend: Optional[str] = None
    qwen3_speaker: Optional[str] = None
    qwen3_language: Optional[str] = None
    language: Optional[str] = Field(
        default=None,
        description=(
            "本次合成实际解析出的语言快照（即呈现给渲染器的取值，见"
            " audiobook.language.resolve_synthesis_language）。为空表示当时"
            "没有任何声明，来源不可考，而不是回退成某个猜测"
        ),
    )
    emotion_mode: EmotionMode = EmotionMode.TEXT
    emotion_text: Optional[str] = None
    emotion_vector: Optional[List[float]] = None
    emotion_audio: Optional[str] = None
    emotion_alpha: float = Field(default=0.65, ge=0.0, le=1.0)
    playback_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    volume_gain: float = Field(default=1.0, ge=0.0, le=2.0)
    baked_speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="原始文件中已经固化的速度，用于兼容旧破坏性编辑",
    )
    baked_volume: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="原始文件中已经固化的增益，用于兼容旧破坏性编辑",
    )
    edit_state: Optional[SegmentEditState] = None
    edit_history: List[SegmentEditState] = Field(
        default_factory=list,
        description="该 Take 最近的非破坏编辑配方修订，旧到新，最多 20 条",
    )
    source_take_id: Optional[str] = Field(
        default=None,
        description="非破坏编辑固化时所来源的 Take ID",
    )
    missing: bool = Field(
        default=False,
        description="原始音频文件缺失或非法；保留元数据用于诊断但不可播放",
    )
    archived_at: Optional[datetime] = None
    pinned: bool = False
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Take 创建时间（UTC）",
    )

    # --- audiobook architecture v1.2 (all optional: old projects still load) ---
    input_fingerprint: Optional[str] = Field(
        default=None,
        description="本次合成需求指纹（输入），与 audio_sha256（输出）严格分离",
    )
    fingerprint_components: Optional[Dict[str, str]] = Field(
        default=None,
        description="指纹的规范化组成项，用于精确指出哪一个依赖发生了变化",
    )
    renderer_id: Optional[str] = None
    engine_id: Optional[str] = None
    model_id: Optional[str] = None
    spoken_text_revision_id: Optional[str] = Field(
        default=None, description="生成时所使用的 spoken-text 修订"
    )
    source_text_hash: Optional[str] = Field(
        default=None, description="该 Take 对应的 canonical source 文本哈希"
    )
    chunk_map: List[ChunkRecord] = Field(
        default_factory=list,
        description="渲染器内部分块与重组后的时间映射；提交的 Take 只有一个权威 utterance",
    )
    stale: bool = Field(
        default=False,
        description="生成结果晚于文本/音色/模型变化返回；可保留为历史候选，但不可自动选用",
    )
    stale_components: List[str] = Field(
        default_factory=list,
        description="导致该 Take 过期的具体依赖项名称（用于选择性重新生成）",
    )
    qa_report: Optional[TakeQaReport] = Field(
        default=None,
        description="绑定到 take_id + audio_sha256 + qa_check_version 的 QA 结果",
    )


class SegmentReviewIssue(BaseModel):
    id: str
    category: str = Field(
        ...,
        description="pronunciation/emotion/pacing/text/audio/other",
    )
    note: Optional[str] = None
    resolved: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Segment(BaseModel):
    """配音段落（角色 + 台词 + 情感 + 音频结果）。"""

    id: str = Field(..., description="段落唯一 ID")
    chapter_id: Optional[str] = Field(None, description="所属章节 ID，可为空")
    role_id: Optional[str] = Field(None, description="说话角色 ID，可为空")
    index: int = Field(0, ge=0, description="在章节中的顺序索引，从 0 开始")

    text: str = Field(..., description="要朗读的台词文本")

    emotion_mode: EmotionMode = Field(
        EmotionMode.TEXT, description="情感控制模式：文本描述 / 向量 / 无"
    )
    emotion_text: Optional[str] = Field(
        None, description="情感描述文本，例如：愤怒中带着委屈，语速略快"
    )
    emotion_vector: Optional[List[float]] = Field(
        None,
        description=(
            "8 维情感向量，[快乐,愤怒,悲伤,恐惧,厌恶,忧郁,惊讶,平静]，每项 0-1"
        ),
    )
    emotion_audio: Optional[str] = Field(
        None,
        description=(
            "情感参考音频路径（绝对路径或相对于项目根目录的路径），仅在 emotion_mode=EmotionMode.AUDIO 时使用"
        ),
    )
    emotion_alpha: float = Field(
        0.65,
        ge=0.0,
        le=1.0,
        description="当前段落的情感强度，默认 0.65",
    )

    status: SegmentStatus = Field(
        SegmentStatus.PENDING, description="生成状态：pending/generating/done/failed"
    )
    audio_file: Optional[str] = Field(
        None,
        description="当前段落对应的音频文件相对路径，例如 audio/segments/seg_0001.wav",
    )

    # B 版本（A/B 对比用）
    alt_audio_file: Optional[str] = Field(
        None, description="B 版本音频文件相对路径",
    )
    alt_emotion_text: Optional[str] = Field(
        None, description="B 版本情感描述文本",
    )
    alt_emotion_alpha: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="B 版本情感权重",
    )
    alt_tts_backend: Optional[str] = Field(
        None, description="B 版本 TTS 后端",
    )
    active_version: str = Field(
        "a", description="当前活跃版本：'a' 或 'b'，导出时使用该版本的音频",
    )

    playback_speed: float = Field(
        1.0,
        ge=0.5,
        le=2.0,
        description="当前段落的播放速度倍数，0.5-2.0，1.0 为原速",
    )
    volume_gain: float = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description="当前段落的音量放大倍数，0-2.0，1.0 为原始音量",
    )

    edit_state: Optional[SegmentEditState] = None

    # Canonical A/B + Take model. Legacy flat fields above are retained as a
    # compatibility projection while old projects and the current WebUI are
    # migrated. New code must treat take_slots/takes as the source of truth.
    take_slots: Dict[str, SegmentTakeSlot] = Field(default_factory=dict)
    takes: List[SegmentTake] = Field(default_factory=list)
    selected_take_id: Optional[str] = Field(
        default=None,
        description="最终合成采用的 Take；为空时回退到当前 A/B 槽位",
    )
    review_status: SegmentReviewStatus = Field(
        default=SegmentReviewStatus.DRAFT,
        description="人工审听状态：draft/to_review/needs_revision/approved",
    )
    review_note: Optional[str] = None
    review_issues: List[SegmentReviewIssue] = Field(default_factory=list)
    reviewed_at: Optional[datetime] = None

    tts_backend: Optional[str] = Field(
        None,
        description="TTS 后端选择：indextts/cosyvoice3/qwen3tts/qwen3tts_cv，为空时使用全局配置",
    )

    qwen3_speaker: Optional[str] = Field(
        None,
        description="Qwen3-TTS CustomVoice 内置音色选择",
    )

    qwen3_language: Optional[str] = Field(
        None,
        description="Qwen3-TTS 语言选择：Auto/Chinese/English/Japanese",
    )

    llm_raw_output: Optional[Dict] = Field(
        None, description="LLM 分析原始结构化结果，用于调试或回溯"
    )

    # --- audiobook architecture v1.2 utterance provenance ---
    source_anchor: Optional[SourceAnchor] = Field(
        None, description="该 utterance 在 canonical source 快照中的锚点"
    )
    spoken_text_revision_id: Optional[str] = Field(
        None, description="当前 spoken-text 修订 ID"
    )
    attribution: Optional[Attribution] = Field(
        None, description="说话人归属来源与置信度；人工修正不可被自动结果覆盖"
    )
    lineage: Optional[SegmentLineage] = Field(
        None, description="split/merge 身份血缘；普通文本编辑不改变 segment.id"
    )
    engine_text: Optional[str] = Field(
        None,
        description=(
            "最近一次渲染器实际收到的文本（引擎表示）。canonical source 与 "
            "spoken text 不得包含引擎专有标记；完整历史见 Take.synthesis_text。"
        ),
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="段落创建时间（UTC）"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="段落最近更新时间（UTC）"
    )


class Project(BaseModel):
    """配音项目。"""

    id: str = Field(..., description="项目唯一 ID（同时作为项目目录名）")
    character_cast: CharacterCastState = Field(default_factory=CharacterCastState)
    voice_lab: VoiceLabState = Field(default_factory=VoiceLabState)
    name: str = Field(..., description="项目名称")
    description: Optional[str] = Field(None, description="项目简介或备注")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="项目创建时间（UTC）"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="项目最近更新时间（UTC）"
    )

    # 章节与角色配置
    chapters: List[Chapter] = Field(
        default_factory=list, description="项目下的章节列表"
    )
    roles: List[Role] = Field(default_factory=list, description="项目下的角色列表")

    # --- audiobook architecture v1.2 state (single canonical store) ---
    source_revisions: List[SourceRevision] = Field(
        default_factory=list, description="canonical source 的不可变快照修订"
    )
    spoken_text_revisions: List[SpokenTextRevision] = Field(
        default_factory=list, description="spoken-text 修订历史（与 canonical source 分离）"
    )
    audio_cues: List[AudioCue] = Field(
        default_factory=list,
        description="Audio Director 预留的 utterance 锚定线索；当前不自动生成",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="非音频元数据（书名、作者、来源路径、分析工具版本等）；不参与合成指纹",
    )
