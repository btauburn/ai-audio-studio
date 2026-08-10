from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class SubtitleWord(BaseModel):
    """词级时间戳信息。"""

    index: int = Field(..., ge=0, description="在整本字幕中的全局词序索引，从 0 开始")
    text: str = Field(..., description="词文本内容")
    start: float = Field(..., ge=0.0, description="词起始时间（秒）")
    end: float = Field(..., ge=0.0, description="词结束时间（秒）")


class SubtitleSegment(BaseModel):
    """句子/分句级字幕段落。"""

    index: int = Field(..., ge=0, description="在整本字幕中的段落索引，从 0 开始")
    start: float = Field(..., ge=0.0, description="段落起始时间（秒）")
    end: float = Field(..., ge=0.0, description="段落结束时间（秒）")
    text: str = Field(..., description="该段落对应的完整文本")
    asr_text: Optional[str] = None
    words: List[SubtitleWord] = Field(default_factory=list, description="该段落内的词级时间戳列表")


class SubtitleFile(BaseModel):
    """完整的字幕文件结构，基于整书音频生成。"""

    id: str = Field(..., description="字幕文件唯一 ID（同时作为文件名的一部分）")
    project_id: str = Field(..., description="所属项目 ID")

    # 源音频信息
    audio_rel_path: str = Field(..., description="源音频相对于项目根目录的相对路径，例如 audio/final/full_xxx.wav")
    audio_filename: str = Field(..., description="源音频文件名，便于前端展示")
    audio_sha256: Optional[str] = None

    # 存储位置（相对于项目根目录）
    json_rel_path: Optional[str] = Field(
        default=None,
        description="字幕 JSON 文件相对路径，例如 subtitles/xxxx.json",
    )
    srt_rel_path: Optional[str] = Field(
        default=None,
        description="字幕 SRT 文件相对路径，例如 subtitles/xxxx.srt",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="字幕创建时间（UTC）",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="字幕最近更新时间（UTC）",
    )

    # 语种与时长
    language: Optional[str] = Field(
        default=None,
        description="识别出的语言代码，例如 zh, en, ja 等",
    )
    duration: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="整条音频的总时长（秒）",
    )

    # 引擎与模型信息，便于后续调试和迁移
    provider: Optional[str] = Field(
        default=None,
        description="字幕生成引擎/提供商名称，例如 faster-whisper",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="底层 ASR 模型名称，例如 large-v3、medium 等",
    )

    text_source: Optional[str] = None

    segments: List[SubtitleSegment] = Field(
        default_factory=list,
        description="按时间排序的字幕段落列表",
    )


class SubtitleSummary(BaseModel):
    """用于列表展示的精简字幕信息。"""

    id: str
    audio_rel_path: str
    audio_filename: str
    created_at: datetime
    updated_at: datetime
    language: Optional[str] = None
    duration: Optional[float] = None
    model_name: Optional[str] = None
    json_rel_path: str
    srt_rel_path: str


class GenerateSubtitleRequest(BaseModel):
    """从整书导出音频生成字幕的请求体。"""

    audio_filename: Optional[str] = Field(
        default=None,
        description="可选：指定要生成字幕的导出音频文件名；为空则使用最新一条导出记录",
    )
    language: Optional[str] = Field(
        default=None,
        description="可选：强制指定识别语言代码；为空则由 ASR 自动检测",
    )
    task: Optional[Literal["transcribe", "translate"]] = Field(
        default=None,
        description="可选：ASR 任务类型，transcribe 为转写，translate 为翻译到英文",
    )
    model_size: Optional[
        Literal["large-v3", "large-v3-turbo", "distil-large-v3"]
    ] = Field(
        default=None,
        description=(
            "可选：项目内 ASR 模型。默认 large-v3-turbo；"
            "distil-large-v3 仅建议用于纯英文。"
        ),
    )

    text_source: Optional[Literal["asr", "segments"]] = "segments"


class GenerateSubtitleResponse(BaseModel):
    success: bool
    subtitle_id: Optional[str] = None
    source_audio_filename: Optional[str] = None
    source_audio_rel_path: Optional[str] = None
    audio_sha256: Optional[str] = None
    json_rel_path: Optional[str] = None
    srt_rel_path: Optional[str] = None
    message: Optional[str] = None
