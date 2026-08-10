"""TTS provider abstraction and IndexTTS2/CosyVoice3 REST implementation."""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dubbing_service.app.core.config import (
    get_cosyvoice_base_url,
    get_config,
    get_qwen3tts_base_url,
    get_qwen3tts_cv_base_url,
    safe_project_root,
)
from dubbing_service.app.core.helpers import resolve_role_reference_audio
from dubbing_service.app.models.enums import EmotionMode
from dubbing_service.app.models.project import Role, Segment
from dubbing_service.app.storage.voices import voices_audio_dir

logger = logging.getLogger(__name__)


def _get_tts_backend() -> str:
    """从配置文件读取当前 TTS 后端类型。"""
    from dubbing_service.app.routers.settings import _load_raw_file_config
    file_cfg = _load_raw_file_config()
    return file_cfg.get("tts_backend", "indextts")


def _load_tts_params() -> Dict[str, Any]:
    """从配置文件读取 TTS 参数，缺失的使用默认值。"""
    from dubbing_service.app.routers.settings import TTS_PARAMS_DEFAULTS, _load_raw_file_config

    file_cfg = _load_raw_file_config()
    tts_params = file_cfg.get("tts_params", {})
    if not isinstance(tts_params, dict):
        tts_params = {}

    result = {}
    for key, default_val in TTS_PARAMS_DEFAULTS.items():
        result[key] = tts_params.get(key, default_val)
    return result


@dataclass
class TTSResult:
    """语音合成结果。"""

    success: bool
    # 由 IndexTTS REST 返回的原始 JSON
    raw: Dict[str, Any]
    # 如果返回中包含生成文件路径，这里尝试提取（字段名依服务实现而定）
    audio_path: Optional[str] = None


class TTSProvider:
    """TTS Provider 抽象基类。"""

    def synthesize_segment(self, project_id: str, role: Role, segment: Segment) -> TTSResult:  # pragma: no cover - interface
        raise NotImplementedError


class IndexTTSTTSProvider(TTSProvider):
    """基于 IndexTTS2 REST 服务的 TTS Provider 实现。
    
    默认调用 `POST {tts_base_url}/api/v1/tts/synthesize` 接口，
    传入文本、参考音频和情感控制参数。
    """
    
    def __init__(self) -> None:
        cfg = get_config()
        self.base_url = cfg.tts_base_url.rstrip("/")
        # 可选的 TTS API Key（例如通过网关鉴权），为空则不附加鉴权头
        self.api_key = getattr(cfg, "tts_api_key", None)

    # 映射 EmotionMode 到 REST API 的 emotion_mode 参数
    @staticmethod
    def _emotion_mode_to_int(mode: EmotionMode) -> int:
        if mode == EmotionMode.TEXT:
            return 3
        if mode == EmotionMode.VECTOR:
            return 2
        if getattr(EmotionMode, "AUDIO", None) is not None and mode == EmotionMode.AUDIO:
            return 1
        # NONE 或其他情况，使用说话人自身情感
        return 0

    def _build_payload(
        self,
        project_id: str,
        role: Role,
        segment: Segment,
        previous_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            project_root = safe_project_root(project_id)
            project_root_resolved = project_root.resolve()
        except Exception:
            project_root = None
            project_root_resolved = None

        speaker_audio_path = resolve_role_reference_audio(
            project_id,
            getattr(role, "speaker_audio", None),
            getattr(role, "speaker_voice_id", None),
        )
        speaker_audio_abs = str(speaker_audio_path) if speaker_audio_path is not None else None

        emotion_mode_int = self._emotion_mode_to_int(segment.emotion_mode)

        # 从配置文件读取 TTS 参数
        tts_params = _load_tts_params()

        payload: Dict[str, Any] = {
            "text": segment.text,
            "speaker_audio": speaker_audio_abs,
            "emotion_mode": emotion_mode_int,
            "emotion_alpha": segment.emotion_alpha,
            "use_random": False,
            # 长文本分段与生成参数，从配置文件读取
            "max_text_tokens_per_segment": tts_params["max_text_tokens_per_segment"],
            "max_mel_tokens": tts_params["max_mel_tokens"],
            "interval_silence": tts_params["interval_silence"],
            "do_sample": True,
            "temperature": tts_params["temperature"],
            "top_p": tts_params["top_p"],
            "top_k": tts_params["top_k"],
            "num_beams": tts_params["num_beams"],
            "repetition_penalty": tts_params["repetition_penalty"],
            "length_penalty": tts_params["length_penalty"],
            # 输出：保存到服务器文件系统，不返回 base64
            "return_base64": False,
            "save_file": True,
        }

        if segment.emotion_mode == EmotionMode.TEXT:
            # 上下文注入：将前一分段文本与当前情感描述组合，提升 TTS 稳定性
            enhanced_emo_text = self._build_enhanced_emotion_text(
                segment.emotion_text, previous_text
            )
            if enhanced_emo_text:
                payload["emotion_text"] = enhanced_emo_text
        elif segment.emotion_mode == EmotionMode.VECTOR:
            if segment.emotion_vector:
                payload["emotion_vector"] = segment.emotion_vector
        elif getattr(EmotionMode, "AUDIO", None) is not None and segment.emotion_mode == EmotionMode.AUDIO:
            if segment.emotion_audio:
                emo_path = Path(segment.emotion_audio)
                resolved_path: Optional[Path] = None

                if emo_path.is_absolute():
                    if emo_path.is_file():
                        resolved_path = emo_path
                else:
                    if project_root is not None and project_root_resolved is not None:
                        try:
                            cand = (project_root / str(segment.emotion_audio)).resolve()
                        except Exception:
                            cand = None
                        if (
                            cand is not None
                            and cand.is_relative_to(project_root_resolved)
                            and cand.is_file()
                        ):
                            resolved_path = cand

                    if resolved_path is None:
                        voice_rel = str(segment.emotion_audio).replace("\\", "/")
                        if voice_rel.startswith("voices/"):
                            voice_rel = voice_rel[len("voices/") :]
                        try:
                            base_dir = voices_audio_dir()
                            base_dir_resolved = base_dir.resolve()
                            cand = (base_dir / voice_rel).resolve()
                        except Exception:
                            cand = None
                        if (
                            cand is not None
                            and cand.is_relative_to(base_dir_resolved)
                            and cand.is_file()
                        ):
                            resolved_path = cand

                if resolved_path is not None:
                    payload["emotion_audio"] = str(resolved_path)
                    logger.debug(
                        "[TTS] 情感参考音频模式: emotion_mode=%s, emotion_audio=%s",
                        payload["emotion_mode"], payload["emotion_audio"],
                    )
                else:
                    payload["emotion_audio"] = None
            else:
                payload["emotion_audio"] = None

        return payload

    def _build_enhanced_emotion_text(
        self,
        current_emotion: Optional[str],
        previous_text: Optional[str],
    ) -> Optional[str]:
        """构建增强版情感描述文本（含上下文注入）。

        格式：前情提要：{前文最后N字}。当前状态：{当前情感描述}
        N 由配置项 context_injection_length 控制，默认50，设为0则禁用上下文注入。
        """
        # 第一段（或未提供前文）时，无论 context_injection_length 配置多少，
        # 都直接返回原始情感描述文本，不做任何包装。
        if not previous_text:
            return current_emotion or None

        # 从配置读取上下文注入长度
        tts_params = _load_tts_params()
        context_length = tts_params.get("context_injection_length", 50)

        # context_injection_length 为 0 时，不做任何包装，直接返回原始情感文本
        if context_length == 0:
            return current_emotion or None

        parts = []

        # 前情提要：取前一分段的后N字作为上下文
        if previous_text and context_length > 0:
            context = previous_text[-context_length:] if len(previous_text) > context_length else previous_text
            parts.append(f"前情提要：{context}")

        # 当前情感描述
        if current_emotion:
            parts.append(f"当前状态：{current_emotion}")

        if not parts:
            return None

        return "。".join(parts)

    def synthesize_segment(
        self,
        project_id: str,
        role: Role,
        segment: Segment,
        previous_text: Optional[str] = None,
    ) -> TTSResult:
        """调用 IndexTTS2 REST 服务为单个段落进行语音合成。"""

        url = f"{self.base_url}/api/v1/tts/synthesize"
        payload = self._build_payload(
            project_id=project_id,
            role=role,
            segment=segment,
            previous_text=previous_text,
        )

        # 在调用 TTS 前先校验参考音频路径是否存在，避免把无效路径传给 IndexTTS2 导致 500。
        speaker_audio_path = payload.get("speaker_audio")
        if not speaker_audio_path:
            return TTSResult(
                success=False,
                raw={"error": "角色未配置可用的参考音频，无法合成。"},
                audio_path=None,
            )
        try:
            if not Path(speaker_audio_path).is_file():
                return TTSResult(
                    success=False,
                    raw={"error": f"参考音频文件不存在: {speaker_audio_path}"},
                    audio_path=None,
                )
        except Exception:
            # 如果校验本身异常，不阻断后续调用，交由下游处理
            pass

        if payload.get("emotion_mode") == 1 and not payload.get("emotion_audio"):
            return TTSResult(
                success=False,
                raw={"error": "情感参考音频不存在或路径非法，无法合成。"},
                audio_path=None,
            )

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            # 增加超时，防止 TTS 服务长时间无响应挂死调用方。
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:  # pragma: no cover - 简单错误封装
            body = exc.read().decode("utf-8", errors="replace")
            logger.warning("TTS HTTPError body: %s", body)

            # 优先从 TTS 返回体中解析更友好的 detail 信息（如 "语音合成失败: ..."）
            detail = None
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict) and parsed.get("detail"):
                    detail = str(parsed["detail"])
            except json.JSONDecodeError:
                parsed = None

            reason = getattr(exc, "reason", None)
            if isinstance(reason, ConnectionResetError):
                msg = (
                    "与 TTS 服务的连接被远端重置 (WinError 10054)，通常是 TTS 服务重启或网络闪断导致，"
                    "可稍后重试。"
                )
            elif detail:
                # 直接使用 TTS 提供的 detail，通常包含“语音合成失败: ...”等信息
                msg = detail
            else:
                msg = f"TTS 网络错误: {exc}"

            return TTSResult(success=False, raw={"error": msg, "status": exc.code, "body": body}, audio_path=None)
        except Exception as exc:  # pragma: no cover - 简单错误封装
            return TTSResult(success=False, raw={"error": str(exc)}, audio_path=None)

        try:
            raw = json.loads(resp_text)
        except json.JSONDecodeError:
            raw = {"raw": resp_text}

        # 如果返回体中包含显式的失败标记或错误信息，则认为本次调用失败。
        success = True
        if isinstance(raw, dict):
            if raw.get("success") is False:
                success = False
            if "error" in raw and raw.get("error"):
                success = False

        audio_path = None
        # 根据 REST 实现尝试提取生成音频路径（字段名可能需要根据实际返回调整）。
        # IndexTTS2 默认返回字段名为 audio_file，这里放在首位。
        for key in ("audio_file", "audio_path", "output_path", "file_path", "path"):
            if isinstance(raw, dict) and key in raw and isinstance(raw[key], str):
                audio_path = raw[key]
                break

        return TTSResult(success=success, raw=raw, audio_path=audio_path)


class CosyVoice3TTSProvider(TTSProvider):
    """基于 CosyVoice3 REST 服务的 TTS Provider 实现。
    
    与 IndexTTS2 使用相同的端点，但支持自然语言指令控制。
    """
    
    def __init__(self) -> None:
        cfg = get_config()
        self.base_url = get_cosyvoice_base_url().rstrip("/")
        self.api_key = getattr(cfg, "tts_api_key", None)

    def _build_payload(
        self,
        project_id: str,
        role: Role,
        segment: Segment,
        previous_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建 CosyVoice3 API 请求参数。"""
        speaker_audio_path = resolve_role_reference_audio(
            project_id,
            getattr(role, "speaker_audio", None),
            getattr(role, "speaker_voice_id", None),
        )
        speaker_audio_abs = str(speaker_audio_path) if speaker_audio_path is not None else None

        # CosyVoice3 默认使用 instruct 模式
        mode = "instruct"
        instruct = getattr(role, "style_instruction", None)
        
        if segment.emotion_mode == EmotionMode.TEXT and segment.emotion_text:
            # 用户提供了指令文本
            instruct = segment.emotion_text
        # 如果没有指令，服务端会使用默认值

        tts_params = _load_tts_params()
        speed = tts_params.get("cosyvoice_speed", 1.0)
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 1.0
        speed = max(0.5, min(2.0, speed))

        payload: Dict[str, Any] = {
            "text": segment.text,
            "speaker_audio": speaker_audio_abs,
            "mode": mode,
            "instruct": instruct,
            # 兼容 IndexTTS2 格式，让 CosyVoice3 服务能识别
            "emotion_mode": 3 if instruct else 0,
            "emotion_text": instruct,
            "speed": speed,
            "return_base64": False,
            "save_file": True,
        }

        return payload

    def synthesize_segment(
        self,
        project_id: str,
        role: Role,
        segment: Segment,
        previous_text: Optional[str] = None,
    ) -> TTSResult:
        """调用 CosyVoice3 REST 服务为单个段落进行语音合成。"""

        url = f"{self.base_url}/api/v1/tts/synthesize"
        payload = self._build_payload(
            project_id=project_id,
            role=role,
            segment=segment,
            previous_text=previous_text,
        )

        speaker_audio_path = payload.get("speaker_audio")
        if not speaker_audio_path:
            return TTSResult(
                success=False,
                raw={"error": "角色未配置可用的参考音频，无法合成。"},
                audio_path=None,
            )
        try:
            if not Path(speaker_audio_path).is_file():
                return TTSResult(
                    success=False,
                    raw={"error": f"参考音频文件不存在: {speaker_audio_path}"},
                    audio_path=None,
                )
        except Exception:
            pass

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            detail = None
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict) and parsed.get("detail"):
                    detail = str(parsed["detail"])
            except json.JSONDecodeError:
                pass

            msg = detail if detail else f"TTS 网络错误: {exc}"
            return TTSResult(success=False, raw={"error": msg, "status": exc.code, "body": body}, audio_path=None)
        except Exception as exc:
            return TTSResult(success=False, raw={"error": str(exc)}, audio_path=None)

        try:
            raw = json.loads(resp_text)
        except json.JSONDecodeError:
            raw = {"raw": resp_text}

        success = True
        if isinstance(raw, dict):
            if raw.get("success") is False:
                success = False
            if "error" in raw and raw.get("error"):
                success = False

        audio_path = None
        for key in ("audio_file", "audio_path", "output_path", "file_path", "path"):
            if isinstance(raw, dict) and key in raw and isinstance(raw[key], str):
                audio_path = raw[key]
                break

        return TTSResult(success=success, raw=raw, audio_path=audio_path)


class Qwen3TTSProvider(TTSProvider):
    """基于 Qwen3-TTS VoiceDesign REST 服务的 TTS Provider 实现。
    
    使用文本描述生成音色并合成语音，无需参考音频。
    适用于临时角色或快速原型。
    """
    
    def __init__(self) -> None:
        cfg = get_config()
        self.base_url = get_qwen3tts_base_url()
        self.api_key = getattr(cfg, "tts_api_key", None)

    def _build_payload(
        self,
        project_id: str,
        role: Role,
        segment: Segment,
        previous_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建 Qwen3-TTS VoiceDesign API 请求参数。"""
        
        # VoiceDesign requires a stable role identity. Performance instructions
        # are appended instead of replacing that identity for every segment.
        voice_description = (
            getattr(role, "voice_description", None)
            or role.description
            or "自然、清晰、适合长篇朗读的声音"
        )
        performance_parts = [
            getattr(role, "style_instruction", None),
            segment.emotion_text,
        ]
        performance = "；".join(str(item).strip() for item in performance_parts if item and str(item).strip())
        instruct = str(voice_description).strip()
        if performance:
            instruct = f"{instruct}；表演要求：{performance}"
        
        tts_params = _load_tts_params()
        default_language = tts_params.get("qwen3_default_language") or "Chinese"

        # 获取语言设置（优先使用分段设置，否则使用默认配置）
        language = (
            getattr(segment, "qwen3_language", None)
            or getattr(role, "qwen3_language", None)
            or default_language
        )
        if language == "Auto":
            # 自动检测语言
            text = segment.text or ""
            ascii_count = sum(1 for c in text if ord(c) < 128 and c.isalpha())
            total_alpha = sum(1 for c in text if c.isalpha())
            if total_alpha > 0 and ascii_count / total_alpha > 0.7:
                language = "English"
            else:
                language = "Chinese"

        payload: Dict[str, Any] = {
            "text": segment.text,
            "instruct": instruct,
            "language": language,
            "return_base64": False,
            "save_file": True,
        }

        return payload

    def synthesize_segment(
        self,
        project_id: str,
        role: Role,
        segment: Segment,
        previous_text: Optional[str] = None,
    ) -> TTSResult:
        """调用 Qwen3-TTS VoiceDesign REST 服务为单个段落进行语音合成。"""

        url = f"{self.base_url}/api/v1/tts/synthesize"
        payload = self._build_payload(
            project_id=project_id,
            role=role,
            segment=segment,
            previous_text=previous_text,
        )

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                resp_text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            detail = None
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict) and parsed.get("detail"):
                    detail = str(parsed["detail"])
            except json.JSONDecodeError:
                pass

            msg = detail if detail else f"TTS 网络错误: {exc}"
            return TTSResult(success=False, raw={"error": msg, "status": exc.code, "body": body}, audio_path=None)
        except Exception as exc:
            return TTSResult(success=False, raw={"error": str(exc)}, audio_path=None)

        try:
            raw = json.loads(resp_text)
        except json.JSONDecodeError:
            raw = {"raw": resp_text}

        success = True
        if isinstance(raw, dict):
            if raw.get("success") is False:
                success = False
            if "error" in raw and raw.get("error"):
                success = False

        audio_path = None
        for key in ("audio_file", "audio_path", "output_path", "file_path", "path"):
            if isinstance(raw, dict) and key in raw and isinstance(raw[key], str):
                audio_path = raw[key]
                break

        return TTSResult(success=success, raw=raw, audio_path=audio_path)


class Qwen3TTSCustomVoiceProvider(TTSProvider):
    """基于 Qwen3-TTS CustomVoice REST 服务的 TTS Provider 实现。
    
    使用内置音色 + 文本指令控制情感。
    """
    
    def __init__(self) -> None:
        cfg = get_config()
        self.base_url = get_qwen3tts_cv_base_url()
        self.api_key = getattr(cfg, "tts_api_key", None)

    def _build_payload(
        self,
        project_id: str,
        role: Role,
        segment: Segment,
        previous_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建 Qwen3-TTS CustomVoice API 请求参数。"""
        
        # Segment is a one-off override; the role profile is the stable default.
        speaker = (
            getattr(segment, "qwen3_speaker", None)
            or getattr(role, "qwen3_speaker", None)
            or "vivian"
        )
        
        # 获取语言（从 segment 的扩展字段或使用默认）
        tts_params = _load_tts_params()
        default_language = tts_params.get("qwen3_default_language") or "Chinese"
        language = (
            getattr(segment, "qwen3_language", None)
            or getattr(role, "qwen3_language", None)
            or default_language
        )
        
        # 情感指令使用 emotion_text
        instruct = segment.emotion_text or getattr(role, "style_instruction", None) or ""
        
        payload: Dict[str, Any] = {
            "text": segment.text,
            "speaker": speaker,
            "language": language,
            "instruct": instruct if instruct else None,
            "return_base64": False,
            "save_file": True,
        }

        return payload

    def synthesize_segment(
        self,
        project_id: str,
        role: Role,
        segment: Segment,
        previous_text: Optional[str] = None,
    ) -> TTSResult:
        """调用 Qwen3-TTS CustomVoice REST 服务为单个段落进行语音合成。"""

        url = f"{self.base_url}/api/v1/tts/synthesize"
        payload = self._build_payload(
            project_id=project_id,
            role=role,
            segment=segment,
            previous_text=previous_text,
        )

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                resp_text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            detail = None
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict) and parsed.get("detail"):
                    detail = str(parsed["detail"])
            except json.JSONDecodeError:
                pass

            msg = detail if detail else f"TTS 网络错误: {exc}"
            return TTSResult(success=False, raw={"error": msg, "status": exc.code, "body": body}, audio_path=None)
        except Exception as exc:
            return TTSResult(success=False, raw={"error": str(exc)}, audio_path=None)

        try:
            raw = json.loads(resp_text)
        except json.JSONDecodeError:
            raw = {"raw": resp_text}

        success = True
        if isinstance(raw, dict):
            if raw.get("success") is False:
                success = False
            if "error" in raw and raw.get("error"):
                success = False

        audio_path = None
        for key in ("audio_file", "audio_path", "output_path", "file_path", "path"):
            if isinstance(raw, dict) and key in raw and isinstance(raw[key], str):
                audio_path = raw[key]
                break

        return TTSResult(success=success, raw=raw, audio_path=audio_path)


class TTSProviderFactory:
    """TTS Provider 工厂，根据配置创建对应的 Provider。"""

    @staticmethod
    def create(backend: Optional[str] = None) -> TTSProvider:
        """创建 TTS Provider。
        
        Args:
            backend: TTS 后端类型，可选 "indextts", "cosyvoice3" 或 "qwen3tts"。
                    如果为 None，则从配置文件读取。
        """
        if backend is None:
            backend = _get_tts_backend()
        
        if backend == "cosyvoice3":
            return CosyVoice3TTSProvider()
        if backend == "qwen3tts":
            return Qwen3TTSProvider()
        if backend == "qwen3tts_cv":
            return Qwen3TTSCustomVoiceProvider()
        return IndexTTSTTSProvider()


def get_tts_provider() -> TTSProvider:
    """获取当前配置的 TTS Provider 单例。"""
    return TTSProviderFactory.create()
