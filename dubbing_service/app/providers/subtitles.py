from __future__ import annotations

import bisect
import difflib
import gc
import logging
import os
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from faster_whisper import WhisperModel  # type: ignore

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.subtitles import (
    SubtitleFile,
    SubtitleSegment,
    SubtitleWord,
)
from service.model_path_policy import ModelPathPolicyError, require_project_model_dir

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WHISPER_MODEL_ID = "large-v3-turbo"
WHISPER_MODEL_DIRECTORIES: dict[str, str] = {
    "large-v3": "models/whisper/faster-whisper-large-v3",
    "large-v3-turbo": "models/whisper/faster-whisper-large-v3-turbo",
    "distil-large-v3": "models/whisper/faster-distil-whisper-large-v3",
}
WHISPER_MODEL_REQUIRED_FILES = (
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
)
WHISPERX_ZH_DIRECTORY = "models/whisper/whisperx-wav2vec2-zh"
WHISPERX_ZH_REQUIRED_FILES = (
    "config.json",
    "pytorch_model.bin",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "vocab.json",
)


class SubtitleModelUnavailableError(RuntimeError):
    """Raised when an allow-listed local subtitle model is not materialized."""


def resolve_whisper_model_path(model_id: str) -> Path:
    """Resolve an allow-listed UI model id to a verified project-local model."""

    normalized = str(model_id or DEFAULT_WHISPER_MODEL_ID).strip().lower()
    relative = WHISPER_MODEL_DIRECTORIES.get(normalized)
    if relative is None:
        allowed = ", ".join(sorted(WHISPER_MODEL_DIRECTORIES))
        raise SubtitleModelUnavailableError(
            f"不支持的 Whisper 模型: {model_id!r}；允许值: {allowed}"
        )
    try:
        return require_project_model_dir(
            PROJECT_ROOT,
            relative,
            label=f"Whisper model {normalized}",
            required_files=WHISPER_MODEL_REQUIRED_FILES,
        )
    except ModelPathPolicyError as exc:
        raise SubtitleModelUnavailableError(
            f"Whisper 模型 {normalized} 尚未在项目内完整配置: {exc}"
        ) from exc


def resolve_whisperx_zh_model_path() -> Path:
    """Return the verified local Chinese alignment model or fail closed."""

    try:
        return require_project_model_dir(
            PROJECT_ROOT,
            WHISPERX_ZH_DIRECTORY,
            label="WhisperX Chinese alignment model",
            required_files=WHISPERX_ZH_REQUIRED_FILES,
        )
    except ModelPathPolicyError as exc:
        raise SubtitleModelUnavailableError(
            f"WhisperX 中文对齐模型尚未在项目内完整配置: {exc}"
        ) from exc


def _detect_device_and_compute_type() -> Tuple[str, str]:
    """根据当前环境自动选择设备与精度。

    - 有 GPU 且 CUDA 可用时：使用 cuda + float16，以获得较好速度与精度平衡；
    - 否则：退回到 CPU + int8，尽量减少内存占用。
    """

    try:  # pragma: no cover - 环境探测相关逻辑
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass

    return "cpu", "int8"


@lru_cache(maxsize=1)
def _load_whisper_model(
    model_path: str,
    device: str,
    compute_type: str,
) -> WhisperModel:
    """Load one verified local model; never reinterpret the value as a repo id."""

    return WhisperModel(model_path, device=device, compute_type=compute_type)


def clear_whisper_model_cache() -> None:
    """清空 Whisper 模型缓存，并尽量释放显存。

    - 调用方可在显存紧张时手动触发；
    - 再次生成字幕时，Provider 会按需重新加载模型。
    """

    for cache_name in (
        "_load_whisper_model",
        "_load_whisperx_align_model",
        "_load_whisperx_audio",
    ):
        cached = globals().get(cache_name)
        cache_clear = getattr(cached, "cache_clear", None)
        if callable(cache_clear):
            try:
                cache_clear()
            except Exception:
                logger.warning("Failed to clear subtitle cache %s", cache_name, exc_info=True)

    gc.collect()

    try:  # pragma: no cover - 环境相关的显存释放逻辑
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # 没有 torch 或 CUDA 时，直接忽略
        pass


def is_whisper_model_loaded() -> bool:
    """检测当前进程中是否已加载过 Whisper 模型。

    这里通过 LRU 缓存的 currsize 判断是否存在已构建的 WhisperModel 实例。
    """

    try:
        info = _load_whisper_model.cache_info()  # type: ignore[attr-defined]
    except Exception:
        return False

    currsize = getattr(info, "currsize", None)
    if currsize is None:
        return False
    return currsize > 0


def _to_float(value: Optional[float]) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - 容错
        return 0.0


def _is_pure_punctuation(text: str) -> bool:
    if not text:
        return False
    for ch in text:
        if not unicodedata.category(ch).startswith("P"):
            return False
    return True


def _join_words_text(words: Sequence[SubtitleWord], *, language: Optional[str]) -> str:
    parts = [str(getattr(w, "text", "") or "") for w in (words or [])]
    parts = [p for p in parts if p]
    if not parts:
        return ""

    lang = (language or "").lower()
    if lang.startswith(("zh", "ja", "ko")):
        return "".join(parts).strip()

    contraction_suffixes = {
        "t",
        "s",
        "re",
        "ve",
        "ll",
        "d",
        "m",
    }

    out = parts[0]
    for t in parts[1:]:
        if not out:
            out = t
            continue
        if not t:
            continue

        prev_last = out[-1]
        next_first = t[0]
        insert_space = True

        if prev_last in "-–—/":
            insert_space = False
        if prev_last in ("(", "[", "{", "“", "‘"):
            insert_space = False
        if out in ('"', "'"):
            insert_space = False
        elif prev_last in ('"', "'") and (next_first.isalpha() or next_first.isdigit()):
            prev_prev = out[-2] if len(out) >= 2 else ""
            if (not prev_prev) or prev_prev.isspace() or prev_prev in ("(", "[", "{", "“", "‘"):
                insert_space = False

        if next_first in ("'", "’") and len(t) > 1:
            suffix = t[1:]
            if suffix and suffix[0].isalpha() and suffix[0].islower() and suffix in contraction_suffixes:
                insert_space = False
        if next_first in (".", ",", "!", "?", ";", ":", ")", "]", "}", "”", "’", "%"):
            insert_space = False

        if insert_space:
            out = f"{out} {t}"
        else:
            out = f"{out}{t}"

    return out.strip()


def _merge_punctuation_words(
    seg: SubtitleSegment,
    *,
    language: Optional[str] = None,
    update_asr_text: bool = False,
) -> None:
    words = list(seg.words or [])
    if not words:
        return

    open_punct = set("“‘([<{《「『")
    close_punct = set("”’)]}>》」』")
    suffix_punct = set(".,!?;:…。，！？；：")
    ambiguous_quotes = {'"', "'"}

    merged: list[SubtitleWord] = []
    changed = False
    i = 0
    while i < len(words):
        w = words[i]
        text = str(getattr(w, "text", "") or "")
        if text and _is_pure_punctuation(text):
            run_text = text
            run_start = float(getattr(w, "start", 0.0) or 0.0)
            run_end = float(getattr(w, "end", run_start) or run_start)

            j = i + 1
            while j < len(words):
                wj = words[j]
                tj = str(getattr(wj, "text", "") or "")
                if not (tj and _is_pure_punctuation(tj)):
                    break
                run_text = f"{run_text}{tj}"
                sj = float(getattr(wj, "start", run_start) or run_start)
                ej = float(getattr(wj, "end", sj) or sj)
                run_start = min(run_start, sj)
                run_end = max(run_end, ej)
                j += 1

            has_next = j < len(words)
            next_text = str(getattr(words[j], "text", "") or "") if has_next else ""
            next_is_word = bool(next_text and not _is_pure_punctuation(next_text))

            direction = "prev"
            if run_text and all(ch in ambiguous_quotes for ch in run_text):
                direction = "next" if next_is_word else "prev"
            elif run_text and run_text[0] in open_punct and not any(ch in suffix_punct for ch in run_text):
                direction = "next"
            elif any(ch in suffix_punct for ch in run_text) or (run_text and run_text[-1] in close_punct):
                direction = "prev"
            elif run_text and run_text[0] in open_punct:
                direction = "next"

            if direction == "prev" and merged:
                prev = merged[-1]
                prev.text = f"{prev.text}{run_text}"
                prev.start = min(prev.start, run_start)
                prev.end = max(prev.end, run_end)
                if prev.end < prev.start:
                    prev.end = prev.start
                changed = True
            elif has_next:
                nxt = words[j]
                nxt.text = f"{run_text}{nxt.text}"
                nxt.start = min(nxt.start, run_start)
                nxt.end = max(nxt.end, run_end)
                if nxt.end < nxt.start:
                    nxt.end = nxt.start
                changed = True
            else:
                w.text = run_text
                w.start = min(float(w.start), run_start)
                w.end = max(float(w.end), run_end)
                if w.end < w.start:
                    w.end = w.start
                merged.append(w)
                changed = True

            i = j
            continue

        merged.append(w)
        i += 1

    if changed:
        seg.words = merged

    if update_asr_text:
        seg.asr_text = _join_words_text(seg.words or [], language=language) or None


def _collect_subtitle_issues(subtitle: SubtitleFile, *, context: str) -> list[str]:
    issues: list[str] = []

    segs = list(subtitle.segments or [])
    if not segs:
        issues.append("no_segments")
        return issues

    last_seg_end: float | None = None
    word_indices: set[int] = set()
    dup_word_indices = 0

    for seg in segs:
        try:
            s0 = float(getattr(seg, "start", 0.0) or 0.0)
            s1 = float(getattr(seg, "end", s0) or s0)
        except Exception:
            issues.append("segment_time_parse_error")
            continue

        if s0 < 0.0 or s1 < 0.0:
            issues.append("segment_negative_time")
        if s1 < s0:
            issues.append("segment_end_before_start")
        if last_seg_end is not None and s0 + 1e-6 < last_seg_end:
            issues.append("segment_not_monotonic")
        last_seg_end = max(last_seg_end or s1, s1)

        ws = list(getattr(seg, "words", None) or [])
        last_w_end: float | None = None
        for w in ws:
            try:
                wi = int(getattr(w, "index", 0) or 0)
            except Exception:
                wi = 0
            if wi in word_indices:
                dup_word_indices += 1
            word_indices.add(wi)

            try:
                w0 = float(getattr(w, "start", 0.0) or 0.0)
                w1 = float(getattr(w, "end", w0) or w0)
            except Exception:
                issues.append("word_time_parse_error")
                continue

            if w0 < 0.0 or w1 < 0.0:
                issues.append("word_negative_time")
            if w1 < w0:
                issues.append("word_end_before_start")
            if w0 + 1e-6 < s0 or w1 - 1e-6 > s1:
                issues.append("word_outside_segment")
            if last_w_end is not None and w0 + 1e-6 < last_w_end:
                issues.append("word_not_monotonic")
            last_w_end = max(last_w_end or w1, w1)

    if dup_word_indices > 0:
        issues.append(f"duplicate_word_index:{dup_word_indices}")

    if issues:
        issues = [f"{context}:{i}" for i in issues]
    return issues


def _log_subtitle_issues(subtitle: SubtitleFile, *, context: str) -> None:
    issues = _collect_subtitle_issues(subtitle, context=context)
    try:
        subtitle_id = getattr(subtitle, "id", None)
    except Exception:
        subtitle_id = None
    if not issues:
        flag = (os.getenv("DUBBING_SUBTITLE_VALIDATION_LOG_OK") or "").strip().lower()
        if flag in {"1", "true", "yes", "y", "on"}:
            logger.info("Subtitle validation OK (subtitle_id=%s, context=%s)", subtitle_id, context)
        return
    logger.warning("Subtitle validation issues (subtitle_id=%s): %s", subtitle_id, "; ".join(issues[:50]))


class FastWhisperSubtitleProvider:
    """封装 faster-whisper 的字幕生成 Provider。

    仅负责：
    - 调用 WhisperModel 进行 ASR；
    - 将结果映射到 SubtitleFile / SubtitleSegment / SubtitleWord 结构。

    实际的文件存储（JSON/SRT）、项目根目录处理等逻辑在路由层完成。
    """

    @staticmethod
    def resolve_model_path(model_size: str) -> Path:
        return resolve_whisper_model_path(model_size)

    def __init__(self, model_size: str = DEFAULT_WHISPER_MODEL_ID) -> None:
        self.model_size = str(model_size or DEFAULT_WHISPER_MODEL_ID).strip().lower()
        self.model_path = self.resolve_model_path(self.model_size)
        device, compute_type = _detect_device_and_compute_type()
        self.device = device
        self.compute_type = compute_type
        self._model = _load_whisper_model(
            str(self.model_path),
            self.device,
            self.compute_type,
        )

    def generate_subtitles(
        self,
        *,
        subtitle_id: str,
        project_id: str,
        audio_path: Path,
        audio_rel_path: str,
        audio_filename: str,
        language: Optional[str] = None,
        task: Optional[str] = None,
    ) -> SubtitleFile:
        """对给定整书音频运行 ASR 并返回结构化字幕。"""

        if not audio_path.is_file():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        # faster-whisper 的 transcribe 接口会返回 segments 生成器和 info 元数据。
        # 注意：task 只能是 "transcribe" 或 "translate"，不能为 None，
        # 所以这里仅在非空时才显式传入，避免传入 None 触发校验错误。
        transcribe_kwargs: dict[str, object] = {
            "beam_size": 5,
            "word_timestamps": True,
        }
        if language:
            transcribe_kwargs["language"] = language
        if task:
            transcribe_kwargs["task"] = task

        segments_iter, info = self._model.transcribe(  # type: ignore[call-arg]
            str(audio_path),
            **transcribe_kwargs,
        )

        segments: List[SubtitleSegment] = []
        global_word_index = 0

        for seg_idx, seg in enumerate(segments_iter):
            seg_start = _to_float(getattr(seg, "start", None))
            seg_end = _to_float(getattr(seg, "end", None))
            raw_seg_text = (getattr(seg, "text", "") or "").strip()
            seg_text = raw_seg_text

            words_src: Iterable[object] = getattr(seg, "words", []) or []
            words: List[SubtitleWord] = []

            for w in words_src:
                text = (getattr(w, "word", "") or "").strip()
                if not text:
                    continue

                w_start = _to_float(getattr(w, "start", None)) or seg_start
                w_end = _to_float(getattr(w, "end", None)) or w_start
                if w_end < w_start:
                    w_end = w_start

                words.append(
                    SubtitleWord(
                        index=global_word_index,
                        text=text,
                        start=w_start,
                        end=w_end,
                    )
                )
                global_word_index += 1

            # 如果不存在词级时间戳，则仍然保证段落级时间范围有效
            if not words:
                if seg_start == 0.0 and seg_end == 0.0:
                    # 尽量避免全 0，交给后续逻辑使用 info.duration 做兜底
                    seg_end = _to_float(getattr(info, "duration", None))

            # 为了避免段落时间为 0 的情况，做一次兜底修正
            if seg_end <= seg_start:
                if words:
                    seg_start = words[0].start
                    seg_end = words[-1].end
                else:
                    seg_end = seg_start

            text_from_words = False
            lang_hint = "en" if task == "translate" else (language or getattr(info, "language", None))
            if not seg_text and words:
                seg_text = _join_words_text(words, language=lang_hint)
                text_from_words = True

            seg_obj = SubtitleSegment(
                index=seg_idx,
                start=seg_start,
                end=seg_end,
                text=seg_text,
                asr_text=seg_text,
                words=words,
            )
            if text_from_words:
                _merge_punctuation_words(seg_obj, language=lang_hint, update_asr_text=True)
                if seg_obj.asr_text is not None:
                    seg_obj.text = seg_obj.asr_text

            segments.append(seg_obj)

        # 语言与时长信息
        detected_lang = getattr(info, "language", None)
        total_duration = _to_float(getattr(info, "duration", None))
        if not total_duration and segments:
            total_duration = segments[-1].end

        subtitle_language = language or detected_lang
        if task == "translate":
            subtitle_language = "en"

        subtitle = SubtitleFile(
            id=subtitle_id,
            project_id=project_id,
            audio_rel_path=audio_rel_path,
            audio_filename=audio_filename,
            language=subtitle_language,
            duration=total_duration or None,
            provider="faster-whisper",
            model_name=self.model_size,
            text_source="asr",
            segments=segments,
        )

        for seg in subtitle.segments or []:
            _merge_punctuation_words(seg, language=subtitle.language)

        _log_subtitle_issues(subtitle, context="generate_subtitles")

        return subtitle


def subtitle_file_to_srt(subtitle: SubtitleFile) -> str:
    """将 SubtitleFile 转换为标准 SRT 文本内容。"""

    def fmt_ts(seconds: float) -> str:
        if seconds < 0:
            seconds = 0.0
        millis = int(round(seconds * 1000))
        h = millis // 3_600_000
        m = (millis % 3_600_000) // 60_000
        s = (millis % 60_000) // 1000
        ms = millis % 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: List[str] = []
    for idx, seg in enumerate(subtitle.segments, start=1):
        start_ts = fmt_ts(seg.start)
        end_ts = fmt_ts(seg.end)
        text = (seg.text or "").strip()

        lines.append(str(idx))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")

    srt = "\n".join(lines).strip() + "\n"
    return srt


def apply_timeline_items_to_subtitle(subtitle: SubtitleFile, timeline_items: Sequence[object]) -> SubtitleFile:
    items: list[tuple[float, float, str]] = []
    for raw in timeline_items or []:
        if not isinstance(raw, dict):
            continue

        raw_text = raw.get("text", "")
        text = raw_text.strip() if isinstance(raw_text, str) else ""

        try:
            start = float(raw.get("start", 0.0) or 0.0)
        except (TypeError, ValueError):
            start = 0.0

        try:
            end = float(raw.get("end", start) or start)
        except (TypeError, ValueError):
            end = start

        if end < start:
            end = start

        items.append((start, end, text))

    if not items:
        subtitle.text_source = "asr"
        return subtitle

    items.sort(key=lambda x: x[0])

    words_all: list[SubtitleWord] = []
    next_word_index = 0
    for seg in subtitle.segments or []:
        for w in seg.words or []:
            words_all.append(w)
            try:
                idx = int(getattr(w, "index", 0) or 0)
            except Exception:
                idx = 0
            if idx + 1 > next_word_index:
                next_word_index = idx + 1
    words_all.sort(key=lambda w: w.start)

    start_hints = [start for start, _, _ in items]
    words_by_item: list[list[SubtitleWord]] = [[] for _ in items]
    for w in words_all:
        anchor = w.end
        idx = bisect.bisect_left(start_hints, anchor) - 1
        if idx < 0:
            idx = 0
        if idx >= len(items):
            idx = len(items) - 1
        words_by_item[idx].append(w)

    new_segments: list[SubtitleSegment] = []

    for idx, (seg_start_hint, seg_end_hint, text) in enumerate(items):
        seg_words = words_by_item[idx]

        seg_start = seg_start_hint
        seg_end = seg_end_hint
        if seg_words:
            seg_start = seg_words[0].start
            seg_end = seg_words[-1].end
            if seg_start < 0.0:
                seg_start = 0.0
            if seg_end < seg_start:
                seg_end = seg_start

        asr_text: Optional[str] = None
        if seg_words:
            asr_text = _join_words_text(seg_words, language=subtitle.language) or None

        if not text and asr_text:
            text = asr_text

        new_segments.append(
            SubtitleSegment(
                index=idx,
                start=seg_start,
                end=seg_end,
                text=text,
                asr_text=asr_text,
                words=seg_words,
            )
        )

    subtitle.segments = new_segments
    subtitle.text_source = "segments"

    for seg in subtitle.segments or []:
        _merge_punctuation_words(seg, language=subtitle.language, update_asr_text=True)

    lang = (subtitle.language or "").lower()
    device, audio, align_model, align_metadata = _init_whisperx_alignment_resources(subtitle)

    for seg in subtitle.segments or []:
        if lang.startswith("zh") and audio is not None and align_model is not None and align_metadata is not None:
            success, next_word_index = _apply_whisperx_alignment(
                seg,
                language=subtitle.language,
                audio=audio,
                device=device,
                align_model=align_model,
                align_metadata=align_metadata,
                next_word_index=next_word_index,
            )
            if not success:
                _sync_segment_words_from_text(seg, language=subtitle.language)
        else:
            _sync_segment_words_from_text(seg, language=subtitle.language)

        _merge_punctuation_words(seg, language=subtitle.language)

    _log_subtitle_issues(subtitle, context="apply_timeline_items_to_subtitle")
    return subtitle


def _sync_segment_words_from_text(seg: SubtitleSegment, *, language: Optional[str]) -> None:
    if not (language or "").lower().startswith(("zh", "ja", "ko")):
        return

    target = (seg.text or "").strip()
    target = target.replace("\r", "").replace("\n", "")
    if (language or "").lower().startswith("zh"):
        target = "".join(target.split())
    seg.text = target

    words = seg.words or []
    if not target:
        if words:
            for w in words:
                w.text = ""
        return

    if not words:
        return

    n = len(words)
    m = len(target)
    if n <= 0:
        return
    if m <= 0:
        for w in words:
            w.text = ""
        return

    t0 = words[0].start
    t1 = words[-1].end

    cuts: List[int] = [0] * (n + 1)
    cuts[0] = 0
    cuts[-1] = m

    if t1 > t0:
        prev = 0
        for i in range(1, n):
            boundary_time = (words[i - 1].end + words[i].start) / 2.0
            r = (boundary_time - t0) / (t1 - t0)
            idx = int(round(r * m))
            if idx < 0:
                idx = 0
            if idx > m:
                idx = m
            if idx < prev:
                idx = prev
            cuts[i] = idx
            prev = idx
    else:
        base = m // n
        rem = m % n
        pos = 0
        for i in range(n):
            take = base + (1 if i < rem else 0)
            cuts[i] = pos
            pos += take
            cuts[i + 1] = pos
        cuts[-1] = m

    for i, w in enumerate(words):
        s = cuts[i]
        e = cuts[i + 1]
        if s < 0:
            s = 0
        if e < s:
            e = s
        if e > m:
            e = m
        w.text = target[s:e]

    _merge_punctuation_words(seg, language=language)


@lru_cache(maxsize=1)
def _load_whisperx_align_model(
    language_code: str,
    device: str,
    model_path: str,
) -> tuple[object, dict]:
    import whisperx  # type: ignore
    align_model, align_metadata = whisperx.load_align_model(
        language_code,
        device,
        model_name=model_path,
    )
    return align_model, align_metadata


@lru_cache(maxsize=1)
def _load_whisperx_audio(audio_path: str, mtime_ns: int) -> object:
    import whisperx  # type: ignore
    _ = mtime_ns
    return whisperx.load_audio(audio_path)


def _init_whisperx_alignment_resources(
    subtitle: SubtitleFile,
) -> tuple[str, object | None, object | None, dict | None]:
    lang = (getattr(subtitle, "language", None) or "").lower()
    if not lang.startswith("zh"):
        return "cpu", None, None, None

    try:
        try:
            project_root = safe_project_root(str(subtitle.project_id))
            project_root_resolved = project_root.resolve()
        except Exception:
            return "cpu", None, None, None

        rel = str(getattr(subtitle, "audio_rel_path", "") or "")
        try:
            audio_path = (project_root / rel).resolve()
        except Exception:
            return "cpu", None, None, None

        if not audio_path.is_relative_to(project_root_resolved):
            return "cpu", None, None, None

        if not audio_path.is_file():
            return "cpu", None, None, None

        device, _ = _detect_device_and_compute_type()
        try:
            align_model_path = resolve_whisperx_zh_model_path()
        except SubtitleModelUnavailableError as exc:
            logger.warning(
                "WhisperX Chinese alignment is unavailable; using deterministic "
                "timestamp fallback: %s",
                exc,
            )
            return "cpu", None, None, None
        mtime_ns = int(audio_path.stat().st_mtime_ns)
        audio = _load_whisperx_audio(str(audio_path), mtime_ns)
        align_model, align_metadata = _load_whisperx_align_model(
            "zh",
            device,
            str(align_model_path),
        )
        return device, audio, align_model, align_metadata
    except Exception:
        logger.exception("Failed to init WhisperX alignment resources")
        return "cpu", None, None, None


def _apply_whisperx_alignment(
    seg: SubtitleSegment,
    *,
    language: Optional[str],
    audio: object,
    device: str,
    align_model: object,
    align_metadata: dict,
    next_word_index: int,
) -> tuple[bool, int]:
    if not (language or "").lower().startswith("zh"):
        return False, next_word_index

    target = (seg.text or "").strip().replace("\r", "").replace("\n", "")
    target = "".join(target.split())
    seg.text = target

    if not target:
        if seg.words:
            for w in seg.words:
                w.text = ""
        return True, next_word_index

    if seg.end <= seg.start:
        return False, next_word_index

    try:
        import whisperx  # type: ignore
    except Exception:
        logger.exception("Failed to import whisperx")
        return False, next_word_index

    align_text = " ".join(list(target))
    transcript = [
        {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": align_text,
        }
    ]

    try:
        aligned = whisperx.align(
            transcript,
            align_model,
            align_metadata,
            audio,
            device,
            return_char_alignments=True,
            print_progress=False,
            combined_progress=False,
        )
    except Exception:
        logger.exception("whisperx.align failed")
        return False, next_word_index

    aligned_segments = []
    try:
        aligned_segments = list((aligned or {}).get("segments", []) or [])
    except Exception:
        logger.exception("Failed to parse whisperx.align result")
        aligned_segments = []

    chars: list[dict] = []
    for s in aligned_segments:
        if not isinstance(s, dict):
            continue
        cs = s.get("chars")
        if isinstance(cs, list):
            for c in cs:
                if isinstance(c, dict):
                    chars.append(c)

    if not chars:
        return False, next_word_index

    aligned_text = "".join(str(c.get("char", "")) for c in chars)
    if aligned_text != align_text:
        compact_chars = [
            c for c in chars if str(c.get("char", "")).strip() != ""
        ]
        aligned_compact = "".join(str(c.get("char", "")) for c in compact_chars)
        if aligned_compact != target:
            return False, next_word_index
        chars = compact_chars
    else:
        if not align_text:
            return False, next_word_index

        if len(chars) != len(align_text):
            return False, next_word_index

        for i in range(1, len(chars), 2):
            if str(chars[i].get("char", "")) != " ":
                return False, next_word_index

        chars = chars[0::2]
        aligned_text = "".join(str(c.get("char", "")) for c in chars)
        if aligned_text != target:
            return False, next_word_index

    try:
        import jieba  # type: ignore
    except Exception:
        logger.exception("Failed to import jieba")
        return False, next_word_index

    tokens = [t for t in jieba.cut(target, cut_all=False) if t]
    if not tokens:
        return False, next_word_index
    if "".join(tokens) != target:
        tokens = list(target)

    old_indices = [w.index for w in (seg.words or [])]
    needed = len(tokens)
    indices: list[int] = []
    if len(old_indices) >= needed:
        indices = old_indices[:needed]
    else:
        indices.extend(old_indices)
        extra = needed - len(old_indices)
        for i in range(extra):
            indices.append(next_word_index + i)
        next_word_index += extra

    if not indices:
        indices = list(range(next_word_index, next_word_index + needed))
        next_word_index += needed

    new_words: list[SubtitleWord] = []
    pos = 0
    time_cursor = float(seg.start)
    for idx, tok in zip(indices, tokens):
        span_len = len(tok)
        span = chars[pos : pos + span_len]
        pos += span_len

        starts: list[float] = []
        ends: list[float] = []
        for c in span:
            s = c.get("start")
            e = c.get("end")
            if isinstance(s, (int, float)):
                starts.append(float(s))
            if isinstance(e, (int, float)):
                ends.append(float(e))

        w_start = min(starts) if starts else time_cursor
        w_end = max(ends) if ends else w_start
        if w_start < seg.start:
            w_start = float(seg.start)
        if w_end > seg.end:
            w_end = float(seg.end)
        if w_start < time_cursor:
            w_start = time_cursor
        if w_end < w_start:
            w_end = w_start
        time_cursor = max(time_cursor, w_end)

        new_words.append(
            SubtitleWord(
                index=int(idx),
                text=str(tok),
                start=float(w_start),
                end=float(w_end),
            )
        )

    if pos != len(chars):
        return False, next_word_index

    seg.words = new_words
    _merge_punctuation_words(seg, language=language)
    return True, next_word_index


def apply_source_text_to_subtitle_by_alignment(
    subtitle: SubtitleFile,
    source_text: str,
    *,
    min_match_ratio: float = 0.45,
) -> Tuple[SubtitleFile, float]:
    src = (source_text or "").strip()
    if not src:
        subtitle.text_source = "asr"
        return subtitle, 0.0

    segs = subtitle.segments or []
    if not segs:
        subtitle.text_source = "asr"
        return subtitle, 0.0

    asr_parts: List[str] = []
    asr_spans: List[Tuple[int, int]] = []
    cursor = 0
    next_word_index = 0

    for idx, seg in enumerate(segs):
        t = (getattr(seg, "asr_text", None) or seg.text or "").strip()
        if idx > 0:
            asr_parts.append("\n")
            cursor += 1
        start = cursor
        asr_parts.append(t)
        cursor += len(t)
        end = cursor
        asr_spans.append((start, end))

        for w in (getattr(seg, "words", None) or []):
            try:
                wi = int(getattr(w, "index", 0) or 0)
            except Exception:
                wi = 0
            if wi + 1 > next_word_index:
                next_word_index = wi + 1

    asr_full = "".join(asr_parts)
    if not asr_full.strip():
        subtitle.text_source = "asr"
        return subtitle, 0.0

    src_len = len(src)

    matcher = difflib.SequenceMatcher(None, asr_full, src, autojunk=False)
    ratio = float(matcher.ratio())

    match_len = 0
    try:
        blocks = matcher.get_matching_blocks()
        match_len = sum(getattr(b, "size", 0) for b in blocks)
    except Exception:
        match_len = 0

    coverage = 0.0
    if len(asr_full) > 0:
        coverage = float(match_len) / float(len(asr_full))

    coverage_src = 0.0
    if src_len > 0:
        coverage_src = float(match_len) / float(src_len)

    match_score = max(ratio, coverage)

    if match_score < min_match_ratio or coverage_src < min_match_ratio:
        subtitle.text_source = "asr"
        for seg in segs:
            seg.text = (getattr(seg, "asr_text", None) or seg.text or "").strip()
        return subtitle, match_score

    opcodes = matcher.get_opcodes()
    opcode_idx = 0

    def _map_pos(pos: int, start_idx: int) -> Tuple[int, int]:
        if pos <= 0:
            return 0, start_idx
        if pos >= len(asr_full):
            return src_len, start_idx

        i = start_idx
        while i < len(opcodes):
            _, i1, i2, j1, j2 = opcodes[i]
            if pos < i1:
                return j1, i
            if i1 <= pos <= i2:
                i_len = i2 - i1
                j_len = j2 - j1
                if i_len <= 0:
                    mapped = j2
                else:
                    r = (pos - i1) / i_len
                    mapped = j1 + int(round(r * j_len))
                if mapped < 0:
                    mapped = 0
                if mapped > src_len:
                    mapped = src_len
                return mapped, i
            i += 1
        return src_len, len(opcodes)

    prev_end = 0

    lang = (subtitle.language or "").lower()
    device, audio, align_model, align_metadata = _init_whisperx_alignment_resources(subtitle)

    for i, seg in enumerate(segs):
        a_start, a_end = asr_spans[i]
        s_start, opcode_idx = _map_pos(a_start, opcode_idx)
        s_end, opcode_idx = _map_pos(a_end, opcode_idx)

        if s_start < prev_end:
            s_start = prev_end
        if s_end < s_start:
            s_end = s_start

        text_slice = src[s_start:s_end].strip()
        if text_slice and ("\n" in text_slice or "\r" in text_slice):
            if (subtitle.language or "").lower().startswith(("zh", "ja", "ko")):
                text_slice = text_slice.replace("\r", "").replace("\n", "")
            else:
                text_slice = text_slice.replace("\r", " ").replace("\n", " ")
                text_slice = " ".join(text_slice.split())

        if (
            i > 0
            and text_slice
            and len(text_slice) > 1
            and (subtitle.language or "").lower().startswith(("zh", "ja", "ko"))
            and text_slice[0] in "，,。！？!?；;：:、"
        ):
            prev_text = (segs[i - 1].text or "").rstrip()
            if prev_text:
                segs[i - 1].text = prev_text + text_slice[0]
                text_slice = text_slice[1:].lstrip()
        if not text_slice:
            text_slice = (getattr(seg, "asr_text", None) or seg.text or "").strip()
        seg.text = text_slice
        prev_end = max(prev_end, s_end)

    for seg in segs:
        if lang.startswith("zh") and audio is not None and align_model is not None and align_metadata is not None:
            success, next_word_index = _apply_whisperx_alignment(
                seg,
                language=subtitle.language,
                audio=audio,
                device=device,
                align_model=align_model,
                align_metadata=align_metadata,
                next_word_index=next_word_index,
            )
            if not success:
                _sync_segment_words_from_text(seg, language=subtitle.language)
        else:
            _sync_segment_words_from_text(seg, language=subtitle.language)

        _merge_punctuation_words(seg, language=subtitle.language)

    subtitle.text_source = "segments"
    _log_subtitle_issues(subtitle, context="apply_source_text_to_subtitle_by_alignment")
    return subtitle, match_score


def apply_source_text_to_subtitle(subtitle: SubtitleFile, source_text: str) -> SubtitleFile:
    src = (source_text or "").strip()
    if not src:
        subtitle.text_source = "segments"
        return subtitle

    segs = subtitle.segments or []
    if not segs:
        subtitle.text_source = "segments"
        return subtitle

    src_len = len(src)
    punct = set("。！？!?…；;，,、\n")

    weights: List[int] = []
    for seg in segs:
        t = (getattr(seg, "asr_text", None) or seg.text or "").strip()
        w = 0
        for ch in t:
            if not ch.isspace():
                w += 1
        weights.append(max(w, 1))

    total = sum(weights) or len(segs)
    pos = 0
    n = len(segs)

    for i, seg in enumerate(segs):
        if pos >= src_len:
            seg.text = (seg.asr_text or seg.text or "").strip()
            continue

        if i == n - 1:
            seg.text = src[pos:].strip()
            break

        remaining_len = src_len - pos
        remaining_segs = n - i
        min_keep = max(0, remaining_segs - 1)
        max_take = max(0, remaining_len - min_keep)

        desired = int(round(src_len * weights[i] / total))
        desired = max(0, min(max_take, desired))
        if desired == 0 and max_take > 0:
            desired = 1

        end = min(src_len, pos + desired)
        max_end = min(src_len, pos + max_take)

        window = 20
        cut = end

        forward_limit = min(max_end, end + window)
        found = None
        for j in range(end, forward_limit):
            if src[j] in punct:
                found = j + 1
                break

        if found is None:
            back_limit = max(pos, end - window)
            for j in range(end - 1, back_limit - 1, -1):
                if src[j] in punct:
                    found = j + 1
                    break

        if found is not None:
            cut = found

        if cut < pos:
            cut = pos
        if cut > max_end:
            cut = max_end

        seg.text = src[pos:cut].strip()
        pos = cut

    for seg in segs:
        _sync_segment_words_from_text(seg, language=subtitle.language)
        _merge_punctuation_words(seg, language=subtitle.language)

    subtitle.text_source = "segments"
    _log_subtitle_issues(subtitle, context="apply_source_text_to_subtitle")
    return subtitle
