"""Enum definitions used across the dubbing service models."""

from __future__ import annotations

from enum import Enum


class RoleType(str, Enum):
    """角色类型。

    - narrator: 旁白/解说
    - character: 剧中角色
    - other: 其他用途
    """

    NARRATOR = "narrator"
    CHARACTER = "character"
    OTHER = "other"


class EmotionMode(str, Enum):
    """情感控制模式。

    - text: 通过情感描述文本控制（优先推荐）
    - vector: 通过 8 维情感向量控制
    - audio: 通过情感参考音频控制
    - none: 不显式控制情感（使用说话人原始情感）
    """

    TEXT = "text"
    VECTOR = "vector"
    AUDIO = "audio"
    NONE = "none"


class SegmentStatus(str, Enum):
    """配音段落的生成状态。"""

    PENDING = "pending"
    GENERATING = "generating"
    DONE = "done"
    FAILED = "failed"


class SegmentReviewStatus(str, Enum):
    """人工审听状态，与生成状态相互独立。"""

    DRAFT = "draft"
    TO_REVIEW = "to_review"
    NEEDS_REVISION = "needs_revision"
    APPROVED = "approved"
