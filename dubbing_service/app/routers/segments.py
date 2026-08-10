"""Text import and LLM-based segmentation APIs for the dubbing service."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List, Optional
import uuid
from threading import Lock
import re

from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel, Field

from dubbing_service.app.core.audio_validation import (
    AudioProbeUnavailableError,
    AudioValidationError,
    normalize_audio_extension,
    validate_decodable_audio_async,
)
from dubbing_service.app.core.helpers import (
    get_max_upload_bytes as _get_max_upload_bytes,
    get_keep_count as _get_keep_count,
    project_root as _project_root,
    sanitize_raw_for_response as _sanitize_raw_for_response,
)
from dubbing_service.app.core.path_safety import (
    SEGMENT_AUDIO_EXTENSIONS,
    resolve_scoped_project_asset,
    validate_simple_component_id,
)
from dubbing_service.app.models.enums import (
    EmotionMode,
    SegmentReviewStatus,
    SegmentStatus,
    RoleType,
)
from dubbing_service.app.models.project import (
    Project,
    Role,
    Segment,
    SegmentEditState,
    SegmentReviewIssue,
)
from dubbing_service.app.providers.llm import LLMResult, OpenAIStyleLLMProvider
from dubbing_service.app.routers.operation_guards import project_mutation_guard
from dubbing_service.app.storage.projects import (
    ProjectStorageError,
    load_project,
    touch_project_updated_at,
)
from dubbing_service.app.storage.segments import (
    SegmentStorageError,
    generate_segment_id,
    load_segments,
    save_segments,
    project_segments_lock,
)
from dubbing_service.app.storage.segmentation_commit import (
    commit_auto_segmentation,
    commit_segment_replacement,
)
from dubbing_service.app.storage.llm_configs import (
    LLMConfigStorageError,
    load_llm_configs,
)
from dubbing_service.app.storage.llm_providers import (
    LLMProviderStorageError,
    load_llm_providers,
)
from dubbing_service.app.storage.voices import (
    VoiceStorageError,
    load_voices,
    voices_audio_dir,
)
from dubbing_service.app.services.takes import (
    adopt_legacy_audio_references,
    capture_active_legacy_settings,
    clear_slot as clear_take_slot,
    ensure_take_state,
    mark_slots_dirty,
    normalize_slot,
    set_active_slot,
    sync_legacy_projection,
    get_selected_take,
)

router = APIRouter()


# 新分段默认情感权重的硬编码默认值
_DEFAULT_EMOTION_ALPHA = 0.65
_VALID_TTS_BACKENDS = {"indextts", "cosyvoice3", "qwen3tts", "qwen3tts_cv"}

def _get_default_emotion_alpha() -> float:
    """从配置文件中读取新分段的默认情感权重，如果未配置则返回 0.65。"""
    try:
        config_path = Path(__file__).resolve().parents[2] / "config.json"
        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            tts_params = cfg.get("tts_params", {})
            if isinstance(tts_params, dict):
                alpha = tts_params.get("emotion_alpha")
                if alpha is not None:
                    return float(alpha)
    except Exception:
        pass
    return _DEFAULT_EMOTION_ALPHA


def _extract_segments_payload(data: object) -> list | None:
    """兼容 LLM 返回数组或包含 segments 字段的对象。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        segments = data.get("segments")
        if isinstance(segments, list):
            return segments
    return None


class _SegmentationOutputError(Exception):
    """Internal validation error used to decide whether Pro fallback is safe."""

    def __init__(
        self,
        message: str,
        *,
        fallback_allowed: bool = True,
        raw: object | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.fallback_allowed = fallback_allowed
        self.raw = raw


def _normalize_source_content(value: str) -> str:
    """Keep ordered Unicode letters/numbers while ignoring layout/punctuation."""

    return "".join(char for char in value if char.isalnum())


def _validate_source_coverage(source_text: str, items: list) -> str | None:
    """Return a diagnostic when segmented text changes, drops, or duplicates content."""

    source = _normalize_source_content(source_text)
    rendered = _normalize_source_content(
        "".join(str(item.get("text", "")) for item in items if isinstance(item, dict))
    )
    if source == rendered:
        return None

    prefix = 0
    for expected, actual in zip(source, rendered):
        if expected != actual:
            break
        prefix += 1
    if len(rendered) < len(source):
        category = "存在遗漏"
    elif len(rendered) > len(source):
        category = "存在重复或新增内容"
    else:
        category = "存在改写或顺序变化"
    return (
        f"分段文本与原文不一致（{category}，首个差异位于规范化字符 {prefix + 1}；"
        f"原文 {len(source)} 字符，分段合计 {len(rendered)} 字符）。"
    )


def _decode_segmentation_result(result: LLMResult, source_text: str) -> tuple[str, object, list]:
    if not result.success:
        raise _SegmentationOutputError(
            "LLM 调用失败",
            fallback_allowed=bool(result.retryable),
            raw=result.raw,
        )
    if result.finish_reason in {"length", "max_tokens"}:
        raise _SegmentationOutputError("LLM 输出因长度限制被截断", raw=result.raw)
    if not result.text or not result.text.strip():
        raise _SegmentationOutputError("LLM 未返回内容", raw=result.raw)

    raw_text = result.text.strip()
    candidates = [raw_text]
    if "```" in raw_text:
        first = raw_text.find("```")
        second = raw_text.find("```", first + 3)
        if second != -1:
            inner = raw_text[first + 3 : second].lstrip()
            if inner.lower().startswith("json"):
                inner = inner[4:]
            candidates.append(inner.strip())
    if "[" in raw_text and "]" in raw_text:
        candidates.append(raw_text[raw_text.find("[") : raw_text.rfind("]") + 1])

    data: object | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        raise _SegmentationOutputError(
            "LLM 返回内容不是合法的 JSON，请检查提示词或模型输出。",
            raw={"llm_output_preview": raw_text[:1000]},
        )

    items = _extract_segments_payload(data)
    if items is None:
        raise _SegmentationOutputError(
            "LLM 返回的数据结构错误：应为包含 segments 字段的 JSON 对象。",
            raw={"llm_output_preview": raw_text[:1000]},
        )
    valid_items = [item for item in items if isinstance(item, dict) and str(item.get("text", "")).strip()]
    if not valid_items:
        raise _SegmentationOutputError("LLM 返回结果为空，未能生成有效段落。")

    coverage_error = _validate_source_coverage(source_text, valid_items)
    if coverage_error:
        raise _SegmentationOutputError(coverage_error)
    return raw_text, data, valid_items

# 记录当前正在执行 LLM 自动分段的项目 ID 集合，供前端查询恢复状态使用
_AUTO_SEGMENT_RUNNING: set[str] = set()
_AUTO_SEGMENT_LOCK = Lock()


def _begin_auto_segment(project_id: str) -> bool:
    """Atomically claim the per-project automatic-segmentation slot."""

    with _AUTO_SEGMENT_LOCK:
        if project_id in _AUTO_SEGMENT_RUNNING:
            return False
        _AUTO_SEGMENT_RUNNING.add(project_id)
        return True


def _end_auto_segment(project_id: str) -> None:
    with _AUTO_SEGMENT_LOCK:
        _AUTO_SEGMENT_RUNNING.discard(project_id)


class TextImportRequest(BaseModel):
    """导入原始文本的请求体。"""

    text: str = Field(..., min_length=1, description="要导入的整本小说/剧本文本")
    filename: str | None = Field(
        None,
        description="保存到 text/ 目录下的文件名，默认 raw.txt",
    )


class TextImportResponse(BaseModel):
    success: bool
    relative_path: str


class TextContentResponse(BaseModel):
    text: str
    relative_path: str | None = None


class AutoSegmentRequest(BaseModel):
    """基于 LLM 自动分段并生成情感描述的请求体。"""

    text: str = Field(..., min_length=1, description="需要按角色/台词拆分的文本内容")
    model: str | None = Field(
        None,
        description="可选：直接指定使用的 LLM 模型名称（不推荐与 llm_config_id 同时使用）",
    )
    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description="可选：生成温度，覆盖默认值",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="可选：采样 top_p（nucleus sampling），覆盖默认值",
    )
    llm_config_id: str | None = Field(
        None,
        description="可选：引用已保存的 LLM 配置 ID，优先级高于 model 字段",
    )


class AutoSegmentResponse(BaseModel):
    segments: List[Segment]
    new_roles_count: int = 0


class AutoSegmentStatusResponse(BaseModel):
    """LLM 自动分段运行状态响应。"""

    running: bool


class SegmentEditStateItem(BaseModel):
    """单个分段的编辑状态（前端波形编辑草稿）。"""

    segment_id: str = Field(..., description="要更新编辑状态的分段 ID")
    state: SegmentEditState = Field(..., description="该分段对应的编辑状态草稿")


class SegmentEditStateUpdatePayload(BaseModel):
    """批量更新分段编辑状态的请求体。"""

    items: List[SegmentEditStateItem] = Field(
        default_factory=list, description="要更新的所有分段编辑状态列表"
    )


class SegmentPromptTemplateResponse(BaseModel):
    """项目级分段提示词模板的读取响应。"""

    content: str


class SegmentPromptTemplateUpdatePayload(BaseModel):
    """更新项目级分段提示词模板的请求体。"""

    content: str | None = Field(
        default=None,
        description="新的模板内容；当 use_default=false 时生效。",
    )
    use_default: bool = Field(
        default=False,
        description="为 true 时删除项目自定义模板并回退到内置默认模板，此时可忽略 content。",
    )


SEG_PROMPT_DEFAULT_TEMPLATE = """请将待分段文本拆分为适合 TTS 模型演绎的 JSON 段落。请先从头到尾**阅读完整文本**，理解故事的整体情节、情绪起伏和关键转折点。然后像专业的精品有声书配音导演一样，先设计整个有声作品的演绎节奏，再决定如何拆分。

【核心任务：黄金打包策略】
TTS 模型在朗读 **3-6 句话** 的段落时效果最惊艳。
你的任务是将原文**“打包”**成这样的黄金段落。

【核心规则】
1. 原文的**汉字/实词**，禁止增删改。
2. **不要机械地按照原文的换行、空行、标点或排版位置来切分**：真正的分段边界应服务于演绎节奏和情感变化。
3. **标点优化特权**：
   - 为了优化 TTS 的语音节奏，**允许且建议**修改标点符号。
   - **长句切分**：将过长的复句，用句号或逗号断开，给 TTS 留出换气口。
   - **语气增强**：根据情感需要，可将句号改为感叹号，或添加省略号来表示犹豫或拖长音。
4. TTS 模型需要平衡"连贯性"与"可控性"：
   a. **合并（防止语气飘忽）**：
      - TTS 模型需要上下文来确定语音的语气和节奏，上下文越多效果越好。
      - 切得太碎会像机器人断断续续，缺乏情感惯性。
      - **指令**：同一角色、情绪稳定的连续内容应合并，让声音一气呵成。

   b. **切分（防止情感漂移）**：
      - 模型难以在超长段落中维持稳定情感，太长会导致后半段出错。
      - **指令**：遇到以下情况需切分：
        - 角色切换
        - 语气或情绪明显变化（如平静→惊恐）
        - 段落过长（超过6-7个完整长句）

**判断技巧**：想象真人演员在演绎有声作品，能情感、语气不变的一口气读完就合并。
5. **角色与独白**：
   - 准确识别说话人。
   - **内心独白**：归入该角色，但必须在情感描述中注明 "内心独白/混响感"。
6. **引导语吸附策略**：
   - 像 "他说道："、"李凯问：" 这种简短的引导语，归入【旁白】。
   - **关键**：请务必将这些短引导语与**前后相邻的旁白段落合并**，禁止让它们单独成为一条只有几个字的 JSON，这会破坏听感。

【已有角色（可能不全）】<ROLE_HINT>
（无法确定用"未知角色1"等表示）

【emotion_description】
用"情境状态 + 情绪 + 声音细节"组合，触发逼真的哈欠、气声、颤音等：

**戏剧性场景**：
- "像刚被吵醒意识模糊，一边打着哈欠一边含糊不清地抱怨，语速很慢"
- "像刚中奖一样兴奋的大喊"
- "像强忍剧痛，声音颤抖，从牙缝挤出"

**普通场景**：
- "像在回忆一段尘封的往事，语速从容，带有淡淡的忧伤"
- "像在讲故事，娓娓道来，从容不迫"
- "轻松随意的日常对话语气"

【输出】
仅输出顶层为对象的 JSON，不要输出解释文字：
```json
{"segments": [
  {"speaker_name": "旁白", "text": "...", "emotion_description": "..."},
  {"speaker_name": "角色A", "text": "...", "emotion_description": "..."}
]}
```

【待分段文本】
<STORY_TEXT>
{{STORY_TEXT}}
</STORY_TEXT>
"""


def _seg_prompt_template_path(project_id: str) -> Path:
    """返回项目级分段提示词模板文件路径。"""

    project_root = _project_root(project_id)
    return project_root / "text" / "seg_prompt_template.txt"


def _load_effective_seg_prompt_template(project: Project) -> str:
    """加载项目级分段提示词模板；如未自定义则返回全局配置的模板或内置默认模板。

    优先级：项目级模板 > 全局配置模板 > 内置默认模板
    """
    # 延迟导入避免循环依赖
    from dubbing_service.app.routers.settings import get_seg_prompt_template

    # 1. 尝试读取项目级模板
    path = _seg_prompt_template_path(project.id)
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except Exception:
        pass

    # 2. 返回全局配置的模板（如果没有自定义则返回内置默认模板）
    return get_seg_prompt_template()


def _prune_seg_prompt_archives(text_dir: Path, keep: int = 10) -> int:
    files: list[tuple[str, Path]] = []
    try:
        for p in text_dir.iterdir():
            if (
                p.is_file()
                and p.suffix == ".txt"
                and p.name != "seg_prompt_template.txt"
                and p.name.startswith("seg_prompt_")
            ):
                m = re.search(r"\d{8}_\d{6}", p.name)
                key = m.group(0) if m else p.name
                files.append((key, p))
    except FileNotFoundError:
        return 0
    except Exception:
        return 0

    files.sort(key=lambda x: x[0], reverse=True)
    deleted = 0
    for _, path in files[keep:]:
        try:
            path.unlink()
            deleted += 1
        except OSError:
            continue
    return deleted


def _prune_seg_llm_logs(logs_dir: Path, keep: int = 10) -> int:
    groups: dict[str, list[Path]] = {}
    try:
        for p in logs_dir.iterdir():
            if not p.is_file():
                continue
            m = re.match(r"^seg_llm_(raw|parsed)_(\d{8}_\d{6})\.(txt|json)$", p.name)
            if not m:
                continue
            ts = m.group(2)
            groups.setdefault(ts, []).append(p)
    except FileNotFoundError:
        return 0
    except Exception:
        return 0

    if not groups:
        return 0

    ts_list = sorted(groups.keys(), reverse=True)
    to_delete_ts = ts_list[keep:]
    deleted = 0
    for ts in to_delete_ts:
        for p in groups.get(ts, []):
            try:
                p.unlink()
                deleted += 1
            except OSError:
                continue
    return deleted


class CleanupSegDebugFilesResponse(BaseModel):
    deleted_text_files: int
    deleted_log_files: int


@router.post(
    "/projects/{project_id}/segments/cleanup-debug-files",
    response_model=CleanupSegDebugFilesResponse,
    dependencies=[Depends(project_mutation_guard)],
)
def api_cleanup_seg_debug_files(project_id: str) -> CleanupSegDebugFilesResponse:
    project_root = _project_root(project_id)
    text_dir = project_root / "text"
    logs_dir = project_root / "logs"

    keep_text = _get_keep_count("DUBBING_KEEP_SEG_PROMPT_ARCHIVES", 1)
    keep_logs = _get_keep_count("DUBBING_KEEP_SEG_LLM_LOGS", 1)

    deleted_text = 0
    deleted_logs = 0
    try:
        deleted_text = _prune_seg_prompt_archives(text_dir, keep=keep_text)
    except Exception:
        deleted_text = 0
    try:
        deleted_logs = _prune_seg_llm_logs(logs_dir, keep=keep_logs)
    except Exception:
        deleted_logs = 0

    return CleanupSegDebugFilesResponse(
        deleted_text_files=deleted_text,
        deleted_log_files=deleted_logs,
    )


@router.post(
    "/projects/{project_id}/segments/edit-state",
    response_model=List[Segment],
    dependencies=[Depends(project_mutation_guard)],
)
async def api_update_segments_edit_state(
    project_id: str,
    payload: SegmentEditStateUpdatePayload,
) -> List[Segment]:
    """批量更新分段的编辑状态（前端波形编辑草稿）。

    - 仅更新请求中出现的 segment_id 对应的 edit_state 字段；
    - 不修改 text/emotion/status/audio_file 等其他业务字段；
    - 持久化到 segments.json，便于跨浏览器和重启后恢复编辑草稿。
    """

    try:
        load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    with project_segments_lock(project_id):
        try:
            segments = load_segments(project_id)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        if not payload.items:
            return segments

        # 建立 segment_id -> Segment 映射，便于快速更新
        seg_map = {seg.id: seg for seg in segments}

        for item in payload.items:
            seg = seg_map.get(item.segment_id)
            if not seg:
                continue
            seg.edit_state = item.state
            if get_selected_take(seg) is not None:
                seg.review_status = SegmentReviewStatus.TO_REVIEW
                seg.reviewed_at = None

        try:
            save_segments(project_id, segments)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    touch_project_updated_at(project_id)

    return segments


class SegmentReviewRequest(BaseModel):
    review_status: SegmentReviewStatus
    note: Optional[str] = None


class SegmentReviewIssueRequest(BaseModel):
    category: str = Field(pattern="^(pronunciation|emotion|pacing|text|audio|other)$")
    note: Optional[str] = None


class ResolveReviewIssueRequest(BaseModel):
    resolved: bool = True


class BatchReviewRequest(BaseModel):
    segment_ids: List[str] = Field(min_length=1)


def _validate_segment_approval(segment: Segment) -> None:
    take = get_selected_take(segment)
    if take is None or take.missing:
        raise HTTPException(status_code=400, detail=f"分段 {segment.index + 1} 没有可用的最终 Take")
    unresolved = [issue for issue in segment.review_issues if not issue.resolved]
    if unresolved:
        raise HTTPException(
            status_code=400,
            detail=f"分段 {segment.index + 1} 仍有 {len(unresolved)} 个未解决审听问题",
        )


@router.patch(
    "/projects/{project_id}/segments/{segment_id}/review",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_update_segment_review(
    project_id: str,
    segment_id: str,
    payload: SegmentReviewRequest,
) -> Segment:
    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        segment = next((item for item in segments if item.id == segment_id), None)
        if segment is None:
            raise HTTPException(status_code=404, detail="段落不存在")
        if payload.review_status == SegmentReviewStatus.APPROVED:
            _validate_segment_approval(segment)
        segment.review_status = payload.review_status
        segment.review_note = (payload.note or "").strip() or None
        segment.reviewed_at = datetime.now(timezone.utc)
        segment.updated_at = segment.reviewed_at
        save_segments(project_id, segments)
    touch_project_updated_at(project_id)
    return segment


@router.post(
    "/projects/{project_id}/segments/{segment_id}/review-issues",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_add_segment_review_issue(
    project_id: str,
    segment_id: str,
    payload: SegmentReviewIssueRequest,
) -> Segment:
    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        segment = next((item for item in segments if item.id == segment_id), None)
        if segment is None:
            raise HTTPException(status_code=404, detail="段落不存在")
        segment.review_issues.append(
            SegmentReviewIssue(
                id=uuid.uuid4().hex[:12],
                category=payload.category,
                note=(payload.note or "").strip() or None,
            )
        )
        segment.review_status = SegmentReviewStatus.NEEDS_REVISION
        segment.reviewed_at = datetime.now(timezone.utc)
        segment.updated_at = segment.reviewed_at
        save_segments(project_id, segments)
    touch_project_updated_at(project_id)
    return segment


@router.patch(
    "/projects/{project_id}/segments/{segment_id}/review-issues/{issue_id}",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_resolve_segment_review_issue(
    project_id: str,
    segment_id: str,
    issue_id: str,
    payload: ResolveReviewIssueRequest,
) -> Segment:
    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        segment = next((item for item in segments if item.id == segment_id), None)
        if segment is None:
            raise HTTPException(status_code=404, detail="段落不存在")
        issue = next((item for item in segment.review_issues if item.id == issue_id), None)
        if issue is None:
            raise HTTPException(status_code=404, detail="审听问题不存在")
        issue.resolved = payload.resolved
        segment.review_status = (
            SegmentReviewStatus.TO_REVIEW
            if all(item.resolved for item in segment.review_issues)
            else SegmentReviewStatus.NEEDS_REVISION
        )
        segment.reviewed_at = datetime.now(timezone.utc)
        segment.updated_at = segment.reviewed_at
        save_segments(project_id, segments)
    touch_project_updated_at(project_id)
    return segment


@router.post(
    "/projects/{project_id}/segments/review/batch-approve",
    response_model=List[Segment],
    dependencies=[Depends(project_mutation_guard)],
)
async def api_batch_approve_segments(
    project_id: str,
    payload: BatchReviewRequest,
) -> List[Segment]:
    target_ids = set(payload.segment_ids)
    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        targets = [segment for segment in segments if segment.id in target_ids]
        if len(targets) != len(target_ids):
            raise HTTPException(status_code=404, detail="部分段落不存在")
        for segment in targets:
            _validate_segment_approval(segment)
        now = datetime.now(timezone.utc)
        for segment in targets:
            segment.review_status = SegmentReviewStatus.APPROVED
            segment.reviewed_at = now
            segment.updated_at = now
        save_segments(project_id, segments)
    touch_project_updated_at(project_id)
    return segments


def _ensure_llm_provider(
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> OpenAIStyleLLMProvider:
    """根据给定参数构造 LLM Provider。

    - 如果提供了 base_url / api_key，则优先使用；
    - 否则回退到全局配置 (config.json / 环境变量)。
    """

    try:
        provider = OpenAIStyleLLMProvider(
            model=model or "gpt-4o-mini",
            default_temperature=None,
            base_url=base_url,
            api_key=api_key,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return provider


def _build_segmentation_prompt(project: Project, story_text: str) -> list[dict[str, str]]:
    """根据项目配置和故事正文构造用于文本分段的 LLM 提示词。

    最终期望 LLM 返回 {"segments": [{speaker_name, text, emotion_description}, ...]}。
    """

    template = _load_effective_seg_prompt_template(project)

    role_names = [r.name for r in project.roles]
    role_hint = "，".join(role_names) if role_names else "（当前项目暂未显式创建角色）"

    rendered = template.replace("<ROLE_HINT>", role_hint)

    if "{{STORY_TEXT}}" in rendered:
        rendered = rendered.replace("{{STORY_TEXT}}", story_text)
    else:
        rendered = f"{rendered}\n\n<STORY_TEXT>\n{story_text}\n</STORY_TEXT>\n"

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "你是一个有声小说/剧本配音助手，负责根据用户提供的规则将故事文本"
                "拆分为适合配音演绎的段落，并为每段生成情感/语气/语速描述。"
                "必须返回顶层包含 segments 数组的合法 JSON 对象；除标点外，不得增删改原文内容。"
            ),
        },
        {"role": "user", "content": rendered},
    ]
    return messages


def _map_speaker_to_role(project: Project, speaker_name: str) -> Optional[str]:
    name = (speaker_name or "").strip()
    name = name.strip(" ：:,，\"'“”")

    if not name:
        for r in project.roles:
            if r.role_type == "narrator":
                return r.id
        return None

    for r in project.roles:
        if r.name == name:
            return r.id

    if name in {"旁白", "旁白君", "旁白者", "narrator", "解说"}:
        for r in project.roles:
            if r.role_type == "narrator":
                return r.id

    return None


def _get_or_create_role_for_speaker(project: Project, speaker_name: str) -> Optional[str]:
    name = (speaker_name or "").strip()
    name = name.strip(" ：:,，\"'“”")

    if not name:
        for r in project.roles:
            if r.role_type == RoleType.NARRATOR:
                return r.id
        role_id = uuid.uuid4().hex[:8]
        role = Role(id=role_id, name="旁白", role_type=RoleType.NARRATOR)
        project.roles.append(role)
        return role_id

    for r in project.roles:
        if r.name == name:
            return r.id

    narrator_names = {"旁白", "旁白君", "旁白者", "narrator", "解说"}
    if name in narrator_names:
        for r in project.roles:
            if r.role_type == RoleType.NARRATOR:
                return r.id
        role_id = uuid.uuid4().hex[:8]
        role = Role(id=role_id, name="旁白", role_type=RoleType.NARRATOR)
        project.roles.append(role)
        return role_id

    role_id = uuid.uuid4().hex[:8]
    role = Role(id=role_id, name=name, role_type=RoleType.CHARACTER)
    project.roles.append(role)
    return role_id


def _clear_segment_audio(project_id: str) -> None:
    """删除项目 audio/segments 目录下的所有分段音频文件。"""

    project_root = _project_root(project_id)
    segments_dir = project_root / "audio" / "segments"
    if not segments_dir.is_dir():
        return

    for item in segments_dir.iterdir():
        try:
            if item.is_file():
                item.unlink()
        except Exception:
            # 音频清理失败不应中断主流程
            continue


def _delete_segment_audio_file(project_id: str, rel_path: str | None) -> None:
    """删除 ``audio/segments`` 下的分段音频（如果存在）。"""

    if not rel_path:
        return

    project_root = _project_root(project_id)
    try:
        audio_path = resolve_scoped_project_asset(
            project_root,
            rel_path,
            required_subdir="audio/segments",
            allowed_extensions=SEGMENT_AUDIO_EXTENSIONS,
        )
    except Exception:
        return
    try:
        if audio_path.is_file():
            audio_path.unlink()
    except Exception:
        # 音频删除失败不影响主要逻辑
        return


def _delete_emotion_audio_file(project_id: str, rel_path: str | None) -> None:
    """删除情感参考音频文件（仅清理项目 audio/emotion 目录下的文件）。"""

    if not rel_path:
        return

    project_root = _project_root(project_id)
    try:
        audio_path = resolve_scoped_project_asset(
            project_root,
            rel_path,
            required_subdir="audio/emotion",
            allowed_extensions={
                ".wav",
                ".mp3",
                ".flac",
                ".ogg",
                ".m4a",
                ".aac",
                ".webm",
            },
        )
    except Exception:
        return
    try:
        if audio_path.is_file():
            audio_path.unlink()
    except Exception:
        return


async def _require_decodable_audio(path: Path) -> None:
    try:
        await validate_decodable_audio_async(path)
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AudioProbeUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def remap_segments_for_project(project_id: str) -> None:
    try:
        project = load_project(project_id)
    except ProjectStorageError:
        return

    with project_segments_lock(project_id):
        try:
            segments = load_segments(project_id)
        except SegmentStorageError:
            return

        changed = False

        for seg in segments:
            raw = seg.llm_raw_output or {}
            if not isinstance(raw, dict):
                continue
            speaker_name = str(raw.get("speaker_name") or raw.get("speaker") or "").strip()
            new_role_id = _map_speaker_to_role(project, speaker_name)
            if new_role_id != seg.role_id:
                seg.role_id = new_role_id
                seg.updated_at = datetime.now(timezone.utc)
                changed = True

        if not changed:
            return

        save_segments(project_id, segments)

    touch_project_updated_at(project_id)


@router.post(
    "/projects/{project_id}/text/import",
    response_model=TextImportResponse,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_import_text(project_id: str, payload: TextImportRequest) -> TextImportResponse:
    """导入原始文本到项目的 text/ 目录下。

    主要用于保存整本小说/剧本原文，便于后续分段或审阅。
    """

    project_root = _project_root(project_id)

    try:
        load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    text_dir = project_root / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    filename = payload.filename or "raw.txt"
    try:
        text_dir_resolved = text_dir.resolve()
        candidate = (text_dir / filename).resolve()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )

    if not candidate.is_relative_to(text_dir_resolved):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )

    path = candidate

    with path.open("w", encoding="utf-8") as f:
        f.write(payload.text)

    if filename.startswith("seg_prompt_") and filename.endswith(".txt"):
        keep_text = _get_keep_count("DUBBING_KEEP_SEG_PROMPT_ARCHIVES", 1)
        _prune_seg_prompt_archives(text_dir, keep=keep_text)

    touch_project_updated_at(project_id)

    rel_path = str(path.relative_to(project_root))
    return TextImportResponse(success=True, relative_path=rel_path)


@router.get("/projects/{project_id}/text", response_model=TextContentResponse)
async def api_get_text(project_id: str) -> TextContentResponse:
    """获取项目下已保存的章节文本内容。

    当前实现：
    - 优先读取 text/raw.txt；
    - 如果不存在，则返回空文本。
    """

    project_root = _project_root(project_id)

    try:
        load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    text_dir = project_root / "text"
    raw_path = text_dir / "raw.txt"

    if raw_path.is_file():
        with raw_path.open("r", encoding="utf-8") as f:
            content = f.read()
        rel_path = str(raw_path.relative_to(project_root))
        return TextContentResponse(text=content, relative_path=rel_path)

    return TextContentResponse(text="", relative_path=None)


@router.get(
    "/projects/{project_id}/segments/prompt-template",
    response_model=SegmentPromptTemplateResponse,
)
async def api_get_segment_prompt_template(project_id: str) -> SegmentPromptTemplateResponse:
    """获取项目级分段提示词模板（若未自定义则返回内置默认模板）。"""

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    content = _load_effective_seg_prompt_template(project)
    return SegmentPromptTemplateResponse(content=content)


@router.put(
    "/projects/{project_id}/segments/prompt-template",
    response_model=SegmentPromptTemplateResponse,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_update_segment_prompt_template(
    project_id: str,
    payload: SegmentPromptTemplateUpdatePayload,
) -> SegmentPromptTemplateResponse:
    """更新项目级分段提示词模板，或恢复为默认模板。"""

    try:
        load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    path = _seg_prompt_template_path(project_id)

    if payload.use_default:
        # 延迟导入避免循环依赖
        from dubbing_service.app.routers.settings import get_seg_prompt_template

        try:
            if path.is_file():
                path.unlink()
        except Exception as exc:  # pragma: no cover - 简单封装
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"恢复默认模板失败: {exc}",
            ) from exc

        # 返回全局配置的默认模板（如果用户自定义了全局模板则使用自定义的）
        content = get_seg_prompt_template()
    else:
        text = (payload.content or "").rstrip()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except Exception as exc:  # pragma: no cover - 简单封装
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"保存模板失败: {exc}",
            ) from exc

        content = text

    # 更新项目更新时间，方便前端列表看到最近编辑活动
    touch_project_updated_at(project_id)

    return SegmentPromptTemplateResponse(content=content)


@router.get(
    "/projects/{project_id}/segments/auto-generate/status",
    response_model=AutoSegmentStatusResponse,
)
async def api_auto_segment_status(project_id: str) -> AutoSegmentStatusResponse:
    """查询指定项目当前是否有 LLM 自动分段任务在执行。"""

    try:
        # 如果项目不存在，直接返回 404
        load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    with _AUTO_SEGMENT_LOCK:
        running = project_id in _AUTO_SEGMENT_RUNNING

    return AutoSegmentStatusResponse(running=running)


@router.post(
    "/projects/{project_id}/segments/auto-generate",
    response_model=AutoSegmentResponse,
    dependencies=[Depends(project_mutation_guard)],
)
def api_auto_segment(project_id: str, payload: AutoSegmentRequest) -> AutoSegmentResponse:
    if not _begin_auto_segment(project_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="该项目已有自动分段任务正在运行，请等待当前任务完成。",
        )
    try:
        return _run_auto_segment(project_id, payload)
    finally:
        _end_auto_segment(project_id)


def _run_auto_segment(project_id: str, payload: AutoSegmentRequest) -> AutoSegmentResponse:
    """基于 LLM 自动将文本拆分为段落，并为每段生成情感描述。

    当前实现：
    - 输入文本由请求体提供；
    - 输出段落会写入项目根目录下的 segments.json，并覆盖原有内容；
    - 每个段落默认使用 EmotionMode.TEXT，emotion_alpha=0.65。
    """

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    # 原文只在 LLM 输出完成校验后与 project.json/segments.json 一起提交。
    project_root = _project_root(project_id)

    text_dir = project_root / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    # 记录分段前项目已有的角色数量，用于统计本次 LLM 分段自动创建了多少新角色
    before_roles_count = len(project.roles)

    # 如果提供了 llm_config_id，则优先根据配置决定模型、max_tokens 以及提供商；否则退回到简单的 model 字符串和全局提供商
    llm_model: str | None = payload.model
    max_tokens: int | None = None
    provider_base_url: str | None = None
    provider_api_key: str | None = None
    target_cfg = None
    thinking: str | None = None
    reasoning_effort: str | None = None
    response_format_json = False
    fallback_enabled = False
    fallback_model: str | None = None

    if payload.llm_config_id:
        try:
            configs = load_llm_configs()
        except LLMConfigStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        target_cfg = next((c for c in configs if c.id == payload.llm_config_id), None)
        if target_cfg is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="指定的 LLM 配置不存在，请在 LLM 配置页面检查后重试。",
            )

        llm_model = target_cfg.model
        max_tokens = target_cfg.max_tokens
        thinking = target_cfg.thinking
        reasoning_effort = target_cfg.reasoning_effort
        response_format_json = target_cfg.response_format_json
        fallback_enabled = target_cfg.fallback_enabled
        fallback_model = target_cfg.fallback_model

        # 如果该配置关联了特定的 LLM 提供商，则优先使用该提供商的 base_url / api_key；
        # 否则如果存在默认提供商，则回退到默认提供商；再否则才使用全局配置。
        try:
            providers = load_llm_providers()
        except LLMProviderStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        if providers:
            provider = None
            if getattr(target_cfg, "provider_id", None):
                provider = next((p for p in providers if p.id == target_cfg.provider_id), None)
                if provider is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="LLM 提供商不存在，请在 LLM 提供商配置中检查后重试。",
                    )
            else:
                # 使用默认提供商（is_default=True），如果没有则使用列表中的第一个
                provider = next((p for p in providers if p.is_default), providers[0])

            provider_base_url = provider.base_url
            provider_api_key = provider.api_key

    provider = _ensure_llm_provider(
        model=llm_model,
        base_url=provider_base_url,
        api_key=provider_api_key,
    )
    llm_model = getattr(provider, "model", llm_model)
    if llm_model == "deepseek-v4-flash":
        thinking = thinking or "disabled"
        response_format_json = True
        if not payload.llm_config_id:
            fallback_enabled = True
            fallback_model = "deepseek-v4-pro"

    messages = _build_segmentation_prompt(project, payload.text)
    log_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # 将本次实际发送给 LLM 的提示词也保存一份，便于对照调试
    try:
        prompt_used_path = text_dir / f"seg_prompt_used_{log_ts}.txt"
        prompt_lines = []
        for msg in messages:
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))
            prompt_lines.append(f"[{role}]\n{content}\n")
        prompt_used_path.write_text("\n".join(prompt_lines), encoding="utf-8")
        keep_text = _get_keep_count("DUBBING_KEEP_SEG_PROMPT_ARCHIVES", 1)
        _prune_seg_prompt_archives(text_dir, keep=keep_text)
    except Exception:
        # 写入失败或清理失败都不影响主流程
        pass

    chat_kwargs = {}
    if payload.temperature is not None:
        chat_kwargs["temperature"] = payload.temperature
    if payload.top_p is not None:
        chat_kwargs["top_p"] = payload.top_p
    if max_tokens is not None:
        chat_kwargs["max_tokens"] = max_tokens

    if response_format_json:
        chat_kwargs["response_format"] = {"type": "json_object"}
    if thinking is not None:
        chat_kwargs["thinking"] = {"type": thinking}
    if reasoning_effort is not None:
        chat_kwargs["reasoning_effort"] = reasoning_effort

    try:
        result: LLMResult = provider.chat(messages, **chat_kwargs)
        raw_text, data, parsed_items = _decode_segmentation_result(result, payload.text)
    except _SegmentationOutputError as primary_error:
        should_fallback = (
            primary_error.fallback_allowed
            and fallback_enabled
            and bool(fallback_model)
            and fallback_model != llm_model
        )
        if not should_fallback:
            detail = {"message": primary_error.message}
            if primary_error.raw is not None:
                detail["llm_error"] = _sanitize_raw_for_response(primary_error.raw)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) from primary_error

        fallback_provider = _ensure_llm_provider(
            model=fallback_model,
            base_url=provider_base_url,
            api_key=provider_api_key,
        )
        fallback_kwargs = dict(chat_kwargs)
        # The fallback is one bounded request. Keep thinking disabled unless a
        # future explicit Pro preset is selected directly by the user.
        fallback_kwargs["thinking"] = {"type": "disabled"}
        fallback_kwargs.pop("reasoning_effort", None)
        fallback_kwargs["max_attempts"] = 1
        fallback_result = fallback_provider.chat(messages, **fallback_kwargs)
        try:
            raw_text, data, parsed_items = _decode_segmentation_result(fallback_result, payload.text)
        except _SegmentationOutputError as fallback_error:
            detail = {
                "message": f"主模型输出未通过校验，备用模型也失败：{fallback_error.message}",
                "primary_error": primary_error.message,
            }
            if fallback_error.raw is not None:
                detail["llm_error"] = _sanitize_raw_for_response(fallback_error.raw)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) from fallback_error

    if not parsed_items:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM 返回结果为空，未能生成有效段落。",
        )

    # 记录本次 LLM 原始输出，方便后续调试/回溯
    logs_dir = project_root / "logs"
    raw_log_path = None
    parsed_log_path = None
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        raw_log_path = logs_dir / f"seg_llm_raw_{log_ts}.txt"
        parsed_log_path = logs_dir / f"seg_llm_parsed_{log_ts}.json"
        raw_log_path.write_text(raw_text, encoding="utf-8")
        keep_logs = _get_keep_count("DUBBING_KEEP_SEG_LLM_LOGS", 1)
        _prune_seg_llm_logs(logs_dir, keep=keep_logs)
    except Exception:
        raw_log_path = None
        parsed_log_path = None

    # 将解析后的 JSON 数组也保存一份，便于对照 segments.json
    if parsed_log_path is not None:
        try:
            with parsed_log_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    segments: List[Segment] = []

    for idx, item in enumerate(parsed_items):
        if not isinstance(item, dict):
            continue
        speaker_name = str(item.get("speaker_name", "")).strip()
        text = str(item.get("text", "")).strip()
        emo_desc = str(item.get("emotion_description", "")).strip() or None

        if not text:
            continue

        role_id = _get_or_create_role_for_speaker(project, speaker_name)

        seg = Segment(
            id=generate_segment_id(),
            chapter_id=None,
            role_id=role_id,
            index=idx,
            text=text,
            emotion_mode=EmotionMode.TEXT,
            emotion_text=emo_desc,
            emotion_vector=None,
            emotion_alpha=_get_default_emotion_alpha(),
            status=SegmentStatus.PENDING,
            audio_file=None,
            llm_raw_output=item,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        segments.append(seg)

    if not segments:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM 返回结果为空，未能生成有效段落。",
        )

    # 原文、分段、角色先经过同一事务提交。任何写入失败都会恢复旧的
    # raw.txt / segments.json / project.json，旧音频此时也仍保持不动。
    new_roles = project.roles[before_roles_count:]
    try:
        commit_result = commit_auto_segmentation(
            project_id,
            raw_text=payload.text,
            segments=segments,
            new_roles=new_roles,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"提交自动分段结果失败，旧数据已保留: {exc}",
        ) from exc

    # 清理是提交后的垃圾回收：即使删除失败，也只会留下可安全清理的孤儿
    # 音频，不会让元数据指向不存在的文件。
    _clear_segment_audio(project_id)

    return AutoSegmentResponse(
        segments=segments,
        new_roles_count=commit_result.new_roles_count,
    )


@router.post(
    "/projects/{project_id}/segments/{segment_id}/emotion-audio",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_upload_segment_emotion_audio(
    project_id: str,
    segment_id: str,
    file: UploadFile = File(..., description="情感参考音频文件，建议 3-10 秒"),
) -> Segment:
    """上传并绑定指定分段的情感参考音频。

    - 音频文件会被保存到项目目录下的 `audio/emotion/` 子目录中；
    - 将该分段的 emotion_audio 字段更新为相对路径；
    - 同时将 emotion_mode 设为 AUDIO，并清空 emotion_text / emotion_vector；
    - 重置分段状态为 PENDING，清空 audio_file。
    """

    if Path(segment_id).name != segment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法分段 ID",
        )

    try:
        segments = load_segments(project_id)
    except SegmentStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    target: Segment | None = None
    for seg in segments:
        if seg.id == segment_id:
            target = seg
            break

    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"指定的分段不存在: {segment_id}",
        )

    project_root = _project_root(project_id)
    emotion_dir = project_root / "audio" / "emotion"
    emotion_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or f"{segment_id}.wav"
    try:
        ext = normalize_audio_extension(filename)
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    target_name = f"{segment_id}{ext}"
    target_path = emotion_dir / target_name
    stage_path = emotion_dir / f".{target_name}.{uuid.uuid4().hex}.upload"
    backup_path = emotion_dir / f".{target_name}.{uuid.uuid4().hex}.backup"

    max_bytes = _get_max_upload_bytes()

    written = 0
    try:
        with stage_path.open("xb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="上传文件过大",
                    )
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            stage_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    try:
        await _require_decodable_audio(stage_path)
    except HTTPException:
        try:
            stage_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    rel_path = str(Path("audio") / "emotion" / target_name)

    audio_cleanup: set[str] = set()
    promoted = False
    had_previous_target = False
    try:
        # File promotion and metadata update are rollback-coupled. Old segment
        # outputs are garbage-collected only after both have succeeded.
        with project_segments_lock(project_id):
            try:
                segments = load_segments(project_id)
            except SegmentStorageError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(exc),
                ) from exc

            target = next((seg for seg in segments if seg.id == segment_id), None)
            if target is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"指定的分段不存在: {segment_id}",
                )

            try:
                if target_path.is_file():
                    os.replace(target_path, backup_path)
                    had_previous_target = True
                os.replace(stage_path, target_path)
                promoted = True
            except Exception as exc:
                if had_previous_target and backup_path.is_file() and not target_path.exists():
                    os.replace(backup_path, target_path)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"保存情感音频失败: {exc}",
                ) from exc

            target.emotion_audio = rel_path
            target.emotion_mode = EmotionMode.AUDIO
            # 保留原有 emotion_text，便于日后切回文本模式时参考；仅清空向量
            target.emotion_vector = None
            target.status = SegmentStatus.PENDING
            mark_slots_dirty(target, target.active_version)
            target.updated_at = datetime.now(timezone.utc)

            try:
                save_segments(project_id, segments)
            except Exception as exc:
                try:
                    if promoted:
                        target_path.unlink(missing_ok=True)
                    if had_previous_target and backup_path.is_file():
                        os.replace(backup_path, target_path)
                except Exception:
                    pass
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"保存分段元数据失败: {exc}",
                ) from exc
    finally:
        try:
            stage_path.unlink(missing_ok=True)
        except OSError:
            pass

    try:
        backup_path.unlink(missing_ok=True)
    except OSError:
        pass
    for old_rel_path in audio_cleanup:
        _delete_segment_audio_file(project_id, old_rel_path)

    # Keep historical WebM recording when a file upload replaces another
    # non-WebM emotion reference; all other stale formats are now unreferenced.
    if ext != ".webm":
        for old_file in emotion_dir.glob(f"{segment_id}.*"):
            if old_file == target_path or old_file.suffix.lower() == ".webm":
                continue
            try:
                old_file.unlink()
            except Exception:
                pass

    try:
        touch_project_updated_at(project_id)
    except ProjectStorageError:
        pass

    return target


@router.get("/projects/{project_id}/segments/{segment_id}/emotion-audio")
async def api_get_segment_emotion_audio(project_id: str, segment_id: str) -> FileResponse:
    """返回指定分段绑定的情感参考音频文件用于试听。"""

    if Path(segment_id).name != segment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法分段 ID",
        )

    try:
        segments = load_segments(project_id)
    except SegmentStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    target: Segment | None = None
    for seg in segments:
        if seg.id == segment_id:
            target = seg
            break

    if target is None or not target.emotion_audio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定的分段不存在或未绑定情感参考音频",
        )

    project_root = _project_root(project_id)
    audio_path: Path | None = None

    try:
        project_root_resolved = project_root.resolve()
        candidate = (project_root / target.emotion_audio).resolve()
        if candidate.is_relative_to(project_root_resolved) and candidate.is_file():
            audio_path = candidate
    except Exception:
        audio_path = None

    if audio_path is None:
        try:
            base_dir = voices_audio_dir()
            base_dir_resolved = base_dir.resolve()
            voice_rel = str(target.emotion_audio).replace("\\", "/")
            if voice_rel.startswith("voices/"):
                voice_rel = voice_rel[len("voices/") :]
            voice_path = (base_dir / voice_rel).resolve()
            if voice_path.is_relative_to(base_dir_resolved) and voice_path.is_file():
                audio_path = voice_path
        except Exception:
            audio_path = None

    if audio_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="情感参考音频不存在",
        )

    suffix = audio_path.suffix.lower()
    mime = "audio/wav"
    if suffix == ".mp3":
        mime = "audio/mpeg"
    elif suffix == ".ogg":
        mime = "audio/ogg"
    elif suffix == ".webm":
        mime = "audio/webm"

    return FileResponse(audio_path, media_type=mime, filename=audio_path.name)


@router.get("/projects/{project_id}/segments/{segment_id}/emotion-recording")
async def api_get_segment_emotion_recording(project_id: str, segment_id: str) -> FileResponse:
    """返回按约定路径保存的情感录制音频（segment_id.webm）。

    该接口只根据约定的文件名 `audio/emotion/{segment_id}.webm` 查找文件，
    与 Segment.emotion_audio 字段当前指向无关，用于在前端恢复“之前录过”的状态。
    """

    if Path(segment_id).name != segment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法分段 ID",
        )

    project_root = _project_root(project_id)
    emotion_dir = project_root / "audio" / "emotion"
    target = emotion_dir / f"{segment_id}.webm"

    if not target.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="未找到该分段的录制情感音频",
        )

    return FileResponse(target, media_type="audio/webm", filename=target.name)


@router.post(
    "/projects/{project_id}/segments/{segment_id}/emotion-recording",
    dependencies=[Depends(project_mutation_guard)],
)
async def api_upload_segment_emotion_recording(
    project_id: str,
    segment_id: str,
    file: UploadFile = File(..., description="情感录制的原始音频文件（webm）"),
) -> dict[str, bool]:
    """上传并覆盖指定分段的原始情感录制音频。

    - 文件固定保存为项目目录下 `audio/emotion/{segment_id}.webm`；
    - 不修改 Segment.emotion_audio / emotion_mode 等配置，仅用于前端刷新后恢复“最近一次录制”的预览；
    - 录制音频通常由前端麦克风录制得到的 webm 文件。
    """

    if Path(segment_id).name != segment_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法分段 ID",
        )

    project_root = _project_root(project_id)
    emotion_dir = project_root / "audio" / "emotion"
    emotion_dir.mkdir(parents=True, exist_ok=True)

    target = emotion_dir / f"{segment_id}.webm"
    transaction_id = uuid.uuid4().hex
    stage_path = emotion_dir / f".{segment_id}.{transaction_id}.upload.webm"
    backup_path = emotion_dir / f".{segment_id}.{transaction_id}.backup"

    max_bytes = _get_max_upload_bytes()
    written = 0
    try:
        with stage_path.open("xb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="上传文件过大",
                    )
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
    except HTTPException:
        try:
            stage_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    except OSError as exc:
        try:
            stage_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存情感录制音频失败: {exc}",
        ) from exc

    try:
        await _require_decodable_audio(stage_path)
    except HTTPException:
        try:
            stage_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    promoted = False
    had_previous = False
    try:
        with project_segments_lock(project_id):
            try:
                segments = load_segments(project_id)
            except SegmentStorageError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(exc),
                ) from exc
            if not any(segment.id == segment_id for segment in segments):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"指定的分段不存在: {segment_id}",
                )

            try:
                if target.is_file():
                    os.replace(target, backup_path)
                    had_previous = True
                os.replace(stage_path, target)
                promoted = True
            except Exception as exc:
                if had_previous and backup_path.is_file() and not target.exists():
                    os.replace(backup_path, target)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"保存情感录制音频失败，旧录音已保留: {exc}",
                ) from exc
    finally:
        try:
            stage_path.unlink(missing_ok=True)
        except OSError:
            pass

    if promoted:
        try:
            backup_path.unlink(missing_ok=True)
        except OSError:
            pass
    return {"success": True}


@router.patch(
    "/projects/{project_id}/segments/{segment_id}",
    response_model=Segment,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_update_segment(
    project_id: str,
    segment_id: str,
    payload: SegmentUpdatePayload,
) -> Segment:
    """更新单个分段的基础信息（文本 / 情感描述 / 绑定角色）。

    - 支持部分更新：只修改请求体中显式提供的字段；
    - 影响生成的修改只标记槽位需要重新合成，不删除现有 Take；
    - 当显式将 role_id 设为 null 时，表示解绑角色；
    - 当设置非空 role_id 时，会校验该角色是否存在于项目中。
    """

    # 注意：整个 load → modify → save 在锁内完成，防止与并发合成冲突
    # project 的加载放在锁外（只读），减少锁持有时间
    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    audio_cleanup: set[str] = set()
    emotion_cleanup: set[str] = set()
    original_emotion_audio: str | None = None

    with project_segments_lock(project_id):
        try:
            segments = load_segments(project_id)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        target: Segment | None = None
        for seg in segments:
            if seg.id == segment_id:
                target = seg
                break

        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"指定的分段不存在: {segment_id}",
            )

        ensure_take_state(target)
        original_emotion_audio = target.emotion_audio

        updates = payload.model_dump(exclude_unset=True)

        # 文本更新
        if "text" in updates:
            new_text = updates["text"]
            if new_text is not None:
                target.text = new_text

        # 情感描述更新（空字符串或 None 代表清空）
        if "emotion_text" in updates:
            emo = updates["emotion_text"]
            if emo is None:
                target.emotion_text = None
            else:
                target.emotion_text = emo.strip() or None

        # 情感控制模式更新
        if "emotion_mode" in updates:
            mode = updates["emotion_mode"]
            if mode is None:
                # 恢复为默认的文本模式
                target.emotion_mode = EmotionMode.TEXT
            else:
                # 兼容字符串或枚举实例
                if isinstance(mode, EmotionMode):
                    target.emotion_mode = mode
                else:
                    try:
                        target.emotion_mode = EmotionMode(str(mode))
                    except ValueError as exc:  # pragma: no cover - 防御性分支
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="不支持的 emotion_mode 值",
                        ) from exc

        # 情感向量更新
        if "emotion_vector" in updates:
            vec = updates["emotion_vector"]
            if vec is None:
                target.emotion_vector = None
            else:
                if not isinstance(vec, (list, tuple)):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="emotion_vector 必须是包含 8 个数值的数组。",
                    )

                floats: list[float] = []
                for item in vec:
                    try:
                        v = float(item)
                    except (TypeError, ValueError) as exc:  # pragma: no cover - 防御性分支
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="emotion_vector 中存在无法转换为数字的元素。",
                        ) from exc

                    if v < 0.0:
                        v = 0.0
                    if v > 1.0:
                        v = 1.0
                    floats.append(v)

                if len(floats) != 8:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="emotion_vector 必须严格包含 8 个数值。",
                    )

                target.emotion_vector = floats

        # 情感参考音频更新
        if "emotion_audio" in updates:
            emo_audio = updates["emotion_audio"]
            if emo_audio is None:
                target.emotion_audio = None
            else:
                s = str(emo_audio).strip()
                if not s:
                    target.emotion_audio = None
                else:
                    rel = s.replace("\\", "/")
                    if rel.startswith("audio/emotion/"):
                        project_root = _project_root(project_id)
                        try:
                            project_root_resolved = project_root.resolve()
                            candidate = (project_root / rel).resolve()
                        except Exception as exc:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="emotion_audio 路径非法。",
                            ) from exc

                        if not candidate.is_relative_to(project_root_resolved) or not candidate.is_file():
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="emotion_audio 文件不存在或路径非法。",
                            )

                        target.emotion_audio = rel
                    else:
                        voice_rel = rel
                        if voice_rel.startswith("voices/"):
                            voice_rel = voice_rel[len("voices/") :]
                        if Path(voice_rel).name != voice_rel:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="emotion_audio 仅允许引用 audio/emotion 或音色库文件。",
                            )

                        base_dir = voices_audio_dir()
                        try:
                            base_dir_resolved = base_dir.resolve()
                            voice_path = (base_dir / voice_rel).resolve()
                        except Exception as exc:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="emotion_audio 路径非法。",
                            ) from exc

                        if not voice_path.is_relative_to(base_dir_resolved) or not voice_path.is_file():
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="音色库音频不存在或路径非法。",
                            )

                        # 若音色库中存在该文件，则允许作为情感参考音频使用
                        try:
                            voices = load_voices()
                        except VoiceStorageError:
                            voices = []
                        if not any(v and getattr(v, "audio_path", None) == voice_rel for v in voices):
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="emotion_audio 必须是 audio/emotion 或音色库中的音频文件。",
                            )

                        target.emotion_audio = voice_rel

        # 角色绑定更新
        if "role_id" in updates:
            new_role_id = updates["role_id"]
            if new_role_id is not None:
                if not any(r.id == new_role_id for r in project.roles):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"项目中不存在角色 {new_role_id}",
                    )
            target.role_id = new_role_id

        if "chapter_id" in updates:
            new_chapter_id = updates["chapter_id"]
            if new_chapter_id is not None and not any(item.id == new_chapter_id for item in project.chapters):
                raise HTTPException(status_code=400, detail=f"项目中不存在章节/场景 {new_chapter_id}")
            target.chapter_id = new_chapter_id

        # 情感强度权重更新（0-1 之间）
        if "emotion_alpha" in updates:
            alpha = updates["emotion_alpha"]
            if alpha is None:
                # 显式传入 null 时，恢复到默认值 0.65
                target.emotion_alpha = 0.65
            else:
                try:
                    v = float(alpha)
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="emotion_alpha 必须是 0-1 之间的数字。",
                    ) from exc

                if v < 0.0:
                    v = 0.0
                if v > 1.0:
                    v = 1.0

                target.emotion_alpha = v

        # TTS 后端更新
        if "tts_backend" in updates:
            backend = updates["tts_backend"]
            if backend is None or backend == "":
                target.tts_backend = None
            elif backend in ("indextts", "cosyvoice3", "qwen3tts", "qwen3tts_cv"):
                target.tts_backend = backend
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="tts_backend 必须是 indextts、cosyvoice3、qwen3tts 或 qwen3tts_cv。",
                )

        # Qwen3-TTS CustomVoice 音色更新
        if "qwen3_speaker" in updates:
            target.qwen3_speaker = updates["qwen3_speaker"] or None

        # Qwen3-TTS 语言更新
        if "qwen3_language" in updates:
            target.qwen3_language = updates["qwen3_language"] or None

        # B 版本（A/B 对比）字段更新
        if "alt_emotion_text" in updates:
            v = updates["alt_emotion_text"]
            inactive = "a" if normalize_slot(target.active_version) == "b" else "b"
            target.take_slots[inactive].emotion_text = (v.strip() if v else None) or None
        if "alt_emotion_alpha" in updates:
            v = updates["alt_emotion_alpha"]
            if v is None:
                inactive = "a" if normalize_slot(target.active_version) == "b" else "b"
                target.take_slots[inactive].emotion_alpha = target.emotion_alpha
            else:
                try:
                    val = float(v)
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="alt_emotion_alpha 必须是 0-1 之间的数字。",
                    ) from exc
                inactive = "a" if normalize_slot(target.active_version) == "b" else "b"
                target.take_slots[inactive].emotion_alpha = max(0.0, min(1.0, val))
        if "alt_tts_backend" in updates:
            v = updates["alt_tts_backend"]
            if v and v not in ("indextts", "cosyvoice3", "qwen3tts", "qwen3tts_cv"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="alt_tts_backend 值不合法。",
                )
            inactive = "a" if normalize_slot(target.active_version) == "b" else "b"
            target.take_slots[inactive].tts_backend = v or None

        alt_resynth = any(
            k in updates for k in ("alt_emotion_text", "alt_emotion_alpha", "alt_tts_backend")
        )
        if alt_resynth:
            inactive = "a" if normalize_slot(target.active_version) == "b" else "b"
            mark_slots_dirty(target, inactive)

        # 活跃版本更新（A/B 对比）
        if "active_version" in updates:
            v = updates["active_version"]
            if v in ("a", "b"):
                set_active_slot(target, v)
            elif v is None:
                set_active_slot(target, "a")
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="active_version 必须是 'a' 或 'b'。",
                )

        # 判断是否需要重新合成音频
        # 只有修改了文本、情感描述、情感模式、情感向量、情感音频、角色等关键字段时才需要重新合成
        # 修改 emotion_alpha 或 tts_backend 仅使当前活跃版本音频失效，不影响另一侧 A/B 备用音频
        shared_resynth = any(
            k in updates
            for k in (
                "text",
                "emotion_mode",
                "emotion_vector",
                "emotion_audio",
                "role_id",
                "qwen3_speaker",
                "qwen3_language",
            )
        )
        current_resynth = any(
            k in updates
            for k in (
                "emotion_text",
                "emotion_alpha",
                "tts_backend",
            )
        )
        need_resynth = any(
            k in updates
            for k in (
                "text",
                "emotion_mode",
                "emotion_vector",
                "emotion_audio",
                "role_id",
                "qwen3_speaker",
                "qwen3_language",
            )
        ) or current_resynth

        if need_resynth:
            if shared_resynth:
                mark_slots_dirty(target, "a", "b")
            else:
                mark_slots_dirty(target, target.active_version)
            target.status = SegmentStatus.PENDING

        capture_active_legacy_settings(target)
        sync_legacy_projection(target)

        target.updated_at = datetime.now(timezone.utc)

        try:
            save_segments(project_id, segments)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        if original_emotion_audio != target.emotion_audio and original_emotion_audio:
            emotion_cleanup.add(str(original_emotion_audio))

    # Metadata is now durable. Best-effort GC can only create harmless orphans;
    # it can no longer leave JSON pointing at a file deleted before a failed save.
    for rel_path in audio_cleanup:
        _delete_segment_audio_file(project_id, rel_path)
    for rel_path in emotion_cleanup:
        _delete_emotion_audio_file(project_id, rel_path)

    touch_project_updated_at(project_id)

    return target


@router.post(
    "/projects/{project_id}/segments/{segment_id}/merge-down",
    response_model=List[Segment],
    dependencies=[Depends(project_mutation_guard)],
)
async def api_merge_segment_down(project_id: str, segment_id: str) -> List[Segment]:
    """将当前分段与其下一条分段合并为一条。

    - 仅当下一条存在且两段绑定同一角色时才允许合并；
    - 合并后保留当前分段 ID，将两段文本直接拼接；
    - 情感描述按顺序用顿号/逗号拼接；
    - 两段原有 Take 会保留在合并后分段的历史记录中，新槽位重置为 PENDING。
    """

    audio_cleanup: set[str] = set()
    emotion_cleanup: set[str] = set()

    with project_segments_lock(project_id):
        try:
            segments = load_segments(project_id)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        if not segments:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="当前项目还没有任何段落，无法合并。",
            )

        target_index = -1
        for idx, seg in enumerate(segments):
            if seg.id == segment_id:
                target_index = idx
                break

        if target_index == -1:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"指定的分段不存在: {segment_id}",
            )

        if target_index >= len(segments) - 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="最后一段无法向下合并。",
            )

        current = segments[target_index]
        below = segments[target_index + 1]

        current_emotion_audio = current.emotion_audio
        below_emotion_audio = below.emotion_audio

        if current.role_id != below.role_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="仅当相邻两段绑定同一角色时才允许向下合并。",
            )

        ensure_take_state(current)
        ensure_take_state(below)

        text1 = current.text or ""
        text2 = below.text or ""
        current.text = f"{text1}{text2}"

        emo1 = (current.emotion_text or "").strip()
        emo2 = (below.emotion_text or "").strip()
        merged_emo = "，".join([x for x in [emo1, emo2] if x])
        current.emotion_text = merged_emo or None

        merged_mode = current.emotion_mode
        if current.emotion_mode == EmotionMode.NONE and below.emotion_mode == EmotionMode.NONE:
            merged_mode = EmotionMode.NONE
        elif (
            current.emotion_mode != below.emotion_mode
            or current.emotion_mode in (EmotionMode.AUDIO, EmotionMode.VECTOR)
            or below.emotion_mode in (EmotionMode.AUDIO, EmotionMode.VECTOR)
        ):
            merged_mode = EmotionMode.TEXT

        current.emotion_mode = merged_mode
        if merged_mode != EmotionMode.TEXT:
            current.emotion_text = None
        if merged_mode != EmotionMode.VECTOR:
            current.emotion_vector = None
        if merged_mode != EmotionMode.AUDIO:
            current.emotion_audio = None

        if merged_mode != EmotionMode.AUDIO and current_emotion_audio:
            emotion_cleanup.add(str(current_emotion_audio))
        if below_emotion_audio:
            emotion_cleanup.add(str(below_emotion_audio))

        existing_take_ids = {take.id for take in current.takes}
        archived_at = datetime.now(timezone.utc)
        for take in below.takes:
            if take.id not in existing_take_ids:
                take.archived_at = take.archived_at or archived_at
                current.takes.append(take)
                existing_take_ids.add(take.id)
        clear_take_slot(current, "a")
        clear_take_slot(current, "b")
        current.selected_take_id = None
        set_active_slot(current, "a")
        mark_slots_dirty(current, "a", "b")
        current.status = SegmentStatus.PENDING
        current.updated_at = datetime.now(timezone.utc)

        del segments[target_index + 1]

        for idx, seg in enumerate(segments):
            seg.index = idx

        try:
            save_segments(project_id, segments)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    for rel_path in audio_cleanup:
        _delete_segment_audio_file(project_id, rel_path)
    for rel_path in emotion_cleanup:
        _delete_emotion_audio_file(project_id, rel_path)

    try:
        touch_project_updated_at(project_id)
    except ProjectStorageError:
        pass

    return segments


@router.delete(
    "/projects/{project_id}/segments/{segment_id}",
    response_model=List[Segment],
    dependencies=[Depends(project_mutation_guard)],
)
async def api_delete_segment(project_id: str, segment_id: str) -> List[Segment]:
    """删除指定分段（以及其对应的音频文件），并重新编号 index。

    - 删除时会尝试清理该分段的 audio_file；
    - 删除后会根据列表顺序将 index 重排为 0..N-1；
    - 返回更新后的 segments 列表，方便前端直接刷新。
    """

    audio_cleanup: set[str] = set()
    emotion_cleanup: set[str] = set()

    with project_segments_lock(project_id):
        try:
            segments = load_segments(project_id)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        if not segments:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="当前项目还没有任何段落，无法删除。",
            )

        target_index = -1
        for idx, seg in enumerate(segments):
            if seg.id == segment_id:
                target_index = idx
                break

        if target_index == -1:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"指定的分段不存在: {segment_id}",
            )

        target = segments[target_index]
        ensure_take_state(target)
        audio_cleanup.update(
            str(take.audio_file)
            for take in target.takes
            if take.audio_file
        )
        if target.emotion_audio:
            emotion_cleanup.add(str(target.emotion_audio))
        emotion_cleanup.add(str(Path("audio") / "emotion" / f"{target.id}.webm"))

        del segments[target_index]

        for idx, seg in enumerate(segments):
            seg.index = idx

        try:
            save_segments(project_id, segments)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    for rel_path in audio_cleanup:
        _delete_segment_audio_file(project_id, rel_path)
    for rel_path in emotion_cleanup:
        _delete_emotion_audio_file(project_id, rel_path)

    try:
        touch_project_updated_at(project_id)
    except ProjectStorageError:
        pass

    return segments


@router.post(
    "/projects/{project_id}/segments/{segment_id}/insert-below",
    response_model=List[Segment],
    dependencies=[Depends(project_mutation_guard)],
)
async def api_insert_segment_below(project_id: str, segment_id: str) -> List[Segment]:
    """在指定分段之后插入一个新的分段。

    - 新分段默认复制当前分段的 role_id、emotion_mode 和 emotion_alpha；
    - 文本和情感描述为空，状态为 PENDING；
    - 插入后会重新顺序编号 index 字段。
    """

    with project_segments_lock(project_id):
        try:
            segments = load_segments(project_id)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        if not segments:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="当前项目还没有任何段落，无法插入新分段。",
            )

        target_index = -1
        for idx, seg in enumerate(segments):
            if seg.id == segment_id:
                target_index = idx
                break

        if target_index == -1:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"指定的分段不存在: {segment_id}",
            )

        base = segments[target_index]
        # 使用配置文件中的默认情感权重，而非从当前分段复制
        default_alpha = _get_default_emotion_alpha()
        inherit_mode = base.emotion_mode
        if inherit_mode in (EmotionMode.AUDIO, EmotionMode.VECTOR):
            inherit_mode = EmotionMode.TEXT
        new_seg = Segment(
            id=generate_segment_id(),
            chapter_id=None,
            role_id=base.role_id,
            index=base.index + 1,
            text="",
            emotion_mode=inherit_mode,
            emotion_text=None,
            emotion_vector=None,
            emotion_alpha=default_alpha,
            status=SegmentStatus.PENDING,
            audio_file=None,
            llm_raw_output=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        segments.insert(target_index + 1, new_seg)

        for idx, seg in enumerate(segments):
            seg.index = idx

        try:
            save_segments(project_id, segments)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    try:
        touch_project_updated_at(project_id)
    except ProjectStorageError:
        pass

    return segments


class ImportSegmentsFromJSONRequest(BaseModel):
    """从外部 LLM 生成的 JSON 文本导入分段的请求体。"""

    json_text: str = Field(..., min_length=2, description="外部 LLM 生成的 JSON 脚本字符串")


class SegmentUpdatePayload(BaseModel):
    """手工编辑单个分段时的部分更新请求体。"""

    text: str | None = Field(
        default=None,
        description="新的分段文本；未提供该字段则不修改，提供空字符串代表清空文本（不推荐，会影响 TTS）",
    )
    emotion_text: str | None = Field(
        default=None,
        description="新的情感描述文本；未提供该字段则不修改，提供空字符串或 null 代表清空情感描述",
    )
    role_id: str | None = Field(
        default=None,
        description="新的角色 ID；未提供该字段则不修改，显式传 null 代表解绑角色。",
    )
    chapter_id: str | None = Field(
        default=None,
        description="所属章节/场景 ID；显式传 null 表示取消归属，不触发重新合成。",
    )
    emotion_mode: EmotionMode | None = Field(
        default=None,
        description="情感控制模式：text/vector/audio/none；未提供则不修改。",
    )
    emotion_vector: list[float] | None = Field(
        default=None,
        description="8 维情感向量，[快乐,愤怒,悲伤,恐惧,厌恶,忧郁,惊讶,平静]，每项 0-1；未提供则不修改。",
    )
    emotion_audio: str | None = Field(
        default=None,
        description="情感参考音频路径；未提供则不修改，显式传 null/空字符串代表清空。",
    )
    emotion_alpha: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    tts_backend: str | None = Field(
        default=None,
        description="TTS 后端选择：indextts/cosyvoice3/qwen3tts/qwen3tts_cv；未提供则不修改。",
    )
    qwen3_speaker: str | None = Field(
        default=None,
        description="Qwen3-TTS CustomVoice 内置音色选择；未提供则不修改。",
    )
    qwen3_language: str | None = Field(
        default=None,
        description="Qwen3-TTS 语言选择：Auto/Chinese/English/Japanese；未提供则不修改。",
    )

    # B 版本（A/B 对比）字段
    alt_emotion_text: str | None = Field(
        default=None, description="B 版本情感描述文本；未提供则不修改。",
    )
    alt_emotion_alpha: float | None = Field(
        default=None, ge=0.0, le=1.0, description="B 版本情感权重；未提供则不修改。",
    )
    alt_tts_backend: str | None = Field(
        default=None, description="B 版本 TTS 后端；未提供则不修改。",
    )
    active_version: str | None = Field(
        default=None, description="当前活跃版本：'a' 或 'b'；未提供则不修改。",
    )


@router.post(
    "/projects/{project_id}/segments/import-json",
    response_model=AutoSegmentResponse,
    dependencies=[Depends(project_mutation_guard)],
)
async def api_import_segments_from_json(project_id: str, payload: ImportSegmentsFromJSONRequest) -> AutoSegmentResponse:
    """从用户提供的 JSON 文本导入分段并保存。

    JSON 结构要求与自动分段时 LLM 输出一致：
    - 根节点为数组；
    - 每个元素是包含 speaker_name, text, emotion_description 字段的对象。
    """

    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    before_roles_count = len(project.roles)

    # 注意：_clear_segment_audio 已移至解析成功后，避免解析失败时删除旧音频

    try:
        data = json.loads(payload.json_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="提供的 JSON 格式不合法，请检查后重试。",
        ) from exc

    parsed_items = _extract_segments_payload(data)
    if parsed_items is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="JSON 结构错误：根节点必须是数组或包含 segments 字段的对象。",
        )

    segments: List[Segment] = []

    for idx, item in enumerate(parsed_items):
        if not isinstance(item, dict):
            continue
        speaker_name = str(item.get("speaker_name", "")).strip()
        text = str(item.get("text", "")).strip()
        emo_desc = str(item.get("emotion_description", "")).strip() or None

        if not text:
            continue

        role_id = _get_or_create_role_for_speaker(project, speaker_name)

        seg = Segment(
            id=generate_segment_id(),
            chapter_id=None,
            role_id=role_id,
            index=idx,
            text=text,
            emotion_mode=EmotionMode.TEXT,
            emotion_text=emo_desc,
            emotion_vector=None,
            emotion_alpha=_get_default_emotion_alpha(),
            status=SegmentStatus.PENDING,
            audio_file=None,
            llm_raw_output=item,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        segments.append(seg)

    if not segments:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="JSON 解析成功但未生成有效段落，请检查 text 字段是否为空。",
        )

    # JSON 导入会同时替换 segments 并合并新增角色。必须先提交元数据，
    # 再清理不再被引用的旧音频；提交失败时旧音频保持可恢复。
    new_roles = project.roles[before_roles_count:]
    try:
        commit_result = commit_segment_replacement(
            project_id,
            segments=segments,
            new_roles=new_roles,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存导入分段失败: {exc}",
        ) from exc

    _clear_segment_audio(project_id)

    return AutoSegmentResponse(
        segments=segments,
        new_roles_count=commit_result.new_roles_count,
    )
@router.get("/projects/{project_id}/segments", response_model=List[Segment])
async def api_list_segments(project_id: str) -> List[Segment]:
    """列出项目下的所有段落（如果有）。"""

    try:
        load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    try:
        return load_segments(project_id)
    except SegmentStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post(
    "/projects/{project_id}/segments/refresh-status",
    response_model=List[Segment],
    dependencies=[Depends(project_mutation_guard)],
)
async def api_refresh_segment_status(project_id: str) -> List[Segment]:
    """刷新段落状态以反映实际音频文件存在情况。

    - 对于 status=DONE 且 audio_file 不为空的段落，检查对应音频文件是否仍然存在；
    - 如果文件已被删除，则将该段落重置为 PENDING，并清空 audio_file；
    - 保存更新后的 segments.json 和项目的更新时间，并返回最新的段落列表。
    """

    project_root = _project_root(project_id)

    now = datetime.now(timezone.utc)

    with project_segments_lock(project_id):
        try:
            segments = load_segments(project_id)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        changed = False
        for seg in segments:
            ensure_take_state(seg)
            for take in seg.takes:
                try:
                    project_root_resolved = project_root.resolve()
                    audio_path = (project_root / str(take.audio_file)).resolve()
                except Exception:
                    audio_path = None
                missing = bool(
                    not audio_path
                    or not audio_path.is_relative_to(project_root_resolved)
                    or not audio_path.is_file()
                )
                if take.missing != missing:
                    take.missing = missing
                    changed = True

            if seg.selected_take_id:
                selected = next(
                    (take for take in seg.takes if take.id == seg.selected_take_id),
                    None,
                )
                if selected is None or selected.missing:
                    seg.selected_take_id = None
                    changed = True

            sync_legacy_projection(seg)
            active_state = seg.take_slots[normalize_slot(seg.active_version)]
            if seg.status == SegmentStatus.DONE and (
                not seg.audio_file or active_state.needs_synthesis
            ):
                seg.status = SegmentStatus.PENDING
                changed = True
            if changed:
                seg.updated_at = now

        if changed:
            save_segments(project_id, segments)

    if changed:
        try:
            touch_project_updated_at(project_id)
        except ProjectStorageError:
            pass

    return segments


class ResetStaleResponse(BaseModel):
    """重置中断段落的响应。"""

    reset_count: int = Field(description="被重置的 generating 段落数")
    active_generating_count: int = Field(description="当前仍在后台真实执行的 generating 段落数")
    pending_count: int = Field(description="当前 pending 段落数（含刚重置的）")
    failed_count: int = Field(description="当前 failed 段落数")
    done_count: int = Field(description="当前 done 段落数")
    total_count: int = Field(description="段落总数")


@router.post(
    "/projects/{project_id}/segments/reset-stale",
    response_model=ResetStaleResponse,
)
async def api_reset_stale_segments(project_id: str) -> ResetStaleResponse:
    """将因中断（浏览器关闭/刷新）卡在 generating 状态的段落重置为 pending。

    用于断点续传：前端加载项目时调用此端点，恢复可续传的状态。
    """

    from dubbing_service.app.routers.ws_synthesis import is_segment_actively_synthesizing

    with project_segments_lock(project_id):
        try:
            segments = load_segments(project_id)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        reset_count = 0
        active_generating_count = 0
        now = datetime.now(timezone.utc)
        for seg in segments:
            if seg.status == SegmentStatus.GENERATING:
                if is_segment_actively_synthesizing(project_id, seg.id):
                    active_generating_count += 1
                    continue
                seg.status = SegmentStatus.PENDING
                seg.updated_at = now
                reset_count += 1

        if reset_count > 0:
            save_segments(project_id, segments)

    pending_count = sum(1 for s in segments if s.status == SegmentStatus.PENDING)
    failed_count = sum(1 for s in segments if s.status == SegmentStatus.FAILED)
    done_count = sum(1 for s in segments if s.status == SegmentStatus.DONE)

    return ResetStaleResponse(
        reset_count=reset_count,
        active_generating_count=active_generating_count,
        pending_count=pending_count,
        failed_count=failed_count,
        done_count=done_count,
        total_count=len(segments),
    )


class RestoreSegmentsRequest(BaseModel):
    """撤销/重做时批量恢复段落的请求体。"""

    segments: List[dict] = Field(..., description="完整的段落 JSON 数组快照")


@router.put(
    "/projects/{project_id}/segments/restore",
    response_model=List[Segment],
    dependencies=[Depends(project_mutation_guard)],
)
async def api_restore_segments(project_id: str, payload: RestoreSegmentsRequest) -> List[Segment]:
    """用快照数据完整替换当前段落列表（撤销/重做专用）。

    接受前端保存的段落快照 JSON 数组，直接覆盖 segments.json。
    不删除音频文件，仅替换元数据。
    """

    # 校验项目存在
    try:
        project = load_project(project_id)
    except ProjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    # 解析段落数据
    project_root = _project_root(project_id)
    role_ids = {role.id for role in project.roles}

    segments: List[Segment] = []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for idx, item in enumerate(payload.segments):
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"segments[{idx}] 不是合法的对象",
            )
        try:
            seg = Segment(**item)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"segments[{idx}] 解析失败: {exc}",
            ) from exc

        try:
            segment_id = validate_simple_component_id(seg.id, label="segment_id")
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"segments[{idx}].id 非法：必须是单一安全 ID",
            ) from exc
        if segment_id in seen_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"segments[{idx}].id 重复: {segment_id}",
            )
        seen_ids.add(segment_id)

        if seg.index in seen_indices:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"segments[{idx}].index 重复: {seg.index}",
            )
        seen_indices.add(seg.index)

        if seg.role_id is not None and seg.role_id not in role_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"segments[{idx}].role_id 引用了不存在的角色: {seg.role_id}",
            )

        for field_name in ("tts_backend", "alt_tts_backend"):
            backend = getattr(seg, field_name, None)
            if backend and backend not in _VALID_TTS_BACKENDS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"segments[{idx}].{field_name} 值不合法",
                )

        for field_name in ("audio_file", "alt_audio_file"):
            rel_path = getattr(seg, field_name, None)
            if not rel_path:
                continue
            try:
                candidate = resolve_scoped_project_asset(
                    project_root,
                    rel_path,
                    required_subdir="audio/segments",
                    allowed_extensions=SEGMENT_AUDIO_EXTENSIONS,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"segments[{idx}].{field_name} 路径越界或非法："
                        "仅允许 audio/segments 下的音频文件"
                    ),
                ) from exc

            # Undo/redo snapshots may outlive a generated file.  Restore the
            # metadata graph, but never recreate a DONE reference that already
            # resolves to 404 on disk.
            if not candidate.is_file():
                setattr(seg, field_name, None)
                seg.updated_at = datetime.now(timezone.utc)

        adopt_legacy_audio_references(seg)
        ensure_take_state(seg)
        for take in seg.takes:
            try:
                candidate = resolve_scoped_project_asset(
                    project_root,
                    take.audio_file,
                    required_subdir="audio/segments",
                    allowed_extensions=SEGMENT_AUDIO_EXTENSIONS,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"segments[{idx}].takes[{take.id}].audio_file 路径越界或非法"
                    ),
                ) from exc
            take.missing = not candidate.is_file()
        sync_legacy_projection(seg)

        if seg.status == SegmentStatus.DONE and not seg.audio_file:
            seg.status = SegmentStatus.PENDING
            seg.updated_at = datetime.now(timezone.utc)

        segments.append(seg)

    if seen_indices != set(range(len(segments))):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="segments[].index 必须唯一且连续覆盖 0..N-1",
        )

    with project_segments_lock(project_id):
        save_segments(project_id, segments)

    return segments


class BatchUpdateEmotionAlphaPayload(BaseModel):
    """批量更新情感权重的请求体。"""
    emotion_alpha: float = Field(..., ge=0.0, le=1.0, description="新的情感权重 (0-1)")


@router.post(
    "/projects/{project_id}/segments/batch-update-emotion-alpha",
    response_model=List[Segment],
    dependencies=[Depends(project_mutation_guard)],
)
async def api_batch_update_emotion_alpha(
    project_id: str,
    payload: BatchUpdateEmotionAlphaPayload,
) -> List[Segment]:
    """批量更新所有分段的情感权重。

    - 将项目中所有分段的 emotion_alpha 设置为指定值；
    - 保留已有 Take，仅将当前槽位标记为需要重新生成；
    - 保存更新后的 segments.json 并返回最新的段落列表。
    """

    _project_root(project_id)
    new_alpha = payload.emotion_alpha
    now = datetime.now(timezone.utc)
    with project_segments_lock(project_id):
        try:
            segments = load_segments(project_id)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        for seg in segments:
            seg.emotion_alpha = new_alpha
            mark_slots_dirty(seg, seg.active_version)
            seg.status = SegmentStatus.PENDING
            seg.updated_at = now

        try:
            save_segments(project_id, segments)
        except SegmentStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    try:
        touch_project_updated_at(project_id, when=now)
    except ProjectStorageError:
        pass

    return segments
