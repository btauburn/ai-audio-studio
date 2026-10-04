"""TTS provider abstraction and IndexTTS2/CosyVoice3 REST implementation."""

from __future__ import annotations

import json
import logging
import time
import uuid
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
from dubbing_service.app.audiobook.provider_capabilities import resolve_customvoice_speaker
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

    @staticmethod
    def _resolve_stable_reference(project_id: str, role: Role, backend: str) -> Optional[Path]:
        """Reference backends share one fail-closed persisted identity contract."""
        identity = getattr(role, "voice_identity", None)
        if identity is None:
            return resolve_role_reference_audio(project_id, role.speaker_audio, role.speaker_voice_id)
        if identity.status != "LOCKED" or identity.mechanism != "existing_reference" or identity.backend != backend:
            raise ValueError("character identity is incompatible with reference provider; explicit casting revision required")
        relative, expected = identity.reference_audio, identity.reference_audio_sha256
        if not relative or not expected or len(expected) != 64:
            raise ValueError("locked reference identity lacks an immutable artifact/checksum; explicit migration required")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not relative.replace("\\", "/").startswith(f"casting/voices/{identity.id}/"):
            raise ValueError("locked reference artifact does not belong to its project identity")
        root = safe_project_root(project_id).resolve()
        candidate = root / path
        candidate.resolve().relative_to(root)
        cursor = candidate
        while cursor != root:
            if cursor.is_symlink() or (cursor.exists() and getattr(cursor.lstat(), "st_file_attributes", 0) & 0x400):
                raise ValueError("linked locked reference artifact is forbidden")
            cursor = cursor.parent
        if not candidate.is_file():
            raise ValueError("locked reference artifact is missing")
        import hashlib
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("locked reference artifact checksum mismatch")
        return candidate

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

        speaker_audio_path = self._resolve_stable_reference(project_id, role, "indextts")
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
        try:
            payload = self._build_payload(project_id=project_id, role=role, segment=segment, previous_text=previous_text)
        except (OSError, ValueError) as exc:
            return TTSResult(False, {"error": str(exc)})

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
        speaker_audio_path = IndexTTSTTSProvider._resolve_stable_reference(project_id, role, "cosyvoice3")
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
        try:
            payload = self._build_payload(project_id=project_id, role=role, segment=segment, previous_text=previous_text)
        except (OSError, ValueError) as exc:
            return TTSResult(False, {"error": str(exc)})

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

    def runtime_identity(self) -> Dict[str, Any]:
        from dubbing_service.app.audiobook.runtime_binding import validate_worker
        request = urllib.request.Request(f"{self.base_url}/api/v1/tts/runtime-identity", method="GET")
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(request, timeout=15) as response:
            worker = validate_worker(json.loads(response.read().decode("utf-8")))
            profile = getattr(self, "_project_runtime_profile", None)
            if profile:
                from dubbing_service.app.audiobook.project_qwen_runtime import require_profile
                require_profile(worker, profile)
            return worker

    def _bind_project_runtime(self, project_id: str) -> None:
        from dubbing_service.app.audiobook.project_qwen_runtime import project_profile, require_profile
        from dubbing_service.app.audiobook.runtime_binding import RuntimeBindingError
        profile = project_profile(project_id)
        worker = self.runtime_identity()
        require_profile(worker, profile)
        expected = getattr(self, "_expected_runtime_identity", None)
        if expected is not None and expected != worker:
            raise RuntimeBindingError("Project worker differs from the bound rendition; no runtime switch or fallback allowed")
        self._expected_runtime_identity = worker

    def _character_request(self, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("project_id"):
            self._bind_project_runtime(payload["project_id"])
        expected = getattr(self, "_expected_runtime_identity", None)
        if operation == "synthesize" and expected is not None:
            payload = {**payload, "expected_runtime_sha256": expected["sha256"]}
        request = urllib.request.Request(
            f"{self.base_url}/api/v1/tts/character/{operation}",
            data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST",
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"character voice {operation} failed: {detail}") from exc
        if not isinstance(result, dict) or result.get("success") is False:
            raise RuntimeError(f"character voice {operation} returned no successful result")
        if operation == "synthesize" and expected is not None:
            if result.get("worker_runtime_identity") != expected or result.get("fallback") is not None:
                raise RuntimeError("Worker returned mismatched runtime or fallback audio; refusing result")
        return result

    def design_character_reference(self, project_id: str, identity_id: str, instruction: str,
                                   text: str, seed: int = 0,
                                   conditioning: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate one ORIGINAL project-owned reference; no outside audio input.

        ``conditioning`` is the numeric acoustic target derived from the
        candidate's performer reference. It travels with the request so the
        worker can shape the original reference onto it before the Base prompt
        is extracted; the performer's name alone would change nothing acoustically.
        """
        payload: Dict[str, Any] = {"project_id": project_id, "identity_id": identity_id,
                                   "instruction": instruction, "text": text, "seed": seed}
        if conditioning:
            payload["conditioning"] = conditioning
        return self._character_request("design", payload)

    def condition_character_reference(self, project_id: str, identity_id: str,
                                      conditioning: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply the reference's acoustic targets to the generated original reference."""
        payload: Dict[str, Any] = {"project_id": project_id, "identity_id": identity_id}
        if conditioning:
            payload["conditioning"] = conditioning
        return self._character_request("acoustics", payload)

    def prepare_character_voice(self, project_id: str, identity_id: str,
                                reference_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract or reuse the generated reference's safe, project-owned Base prompt."""
        return self._character_request("prepare", {"project_id": project_id, "identity_id": identity_id})

    def design_character_voice(self, project_id: str, identity_id: str, instruction: str,
                               text: str, seed: int = 0,
                               conditioning: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        reference = self.design_character_reference(project_id, identity_id, instruction, text, seed,
                                                    conditioning)
        if conditioning:
            # Between design and prepare, on purpose: the Base prompt must encode
            # the conditioned reference, never the unconditioned design output.
            self.condition_character_reference(project_id, identity_id, conditioning)
        return self.prepare_character_voice(project_id, identity_id, reference)

    def import_character_reference(self, project_id: str, identity_id: str,
                                   acquisition_id: str, acquisition_sha256: str) -> Dict[str, Any]:
        """Import a verified project-local eligible acquisition, then prepare Base.

        The worker revalidates the source/rights/audio receipt. No raw URL or
        arbitrary path crosses this boundary; CustomVoice is not used for clones.
        """
        reference = self._character_request("import-reference", {
            "project_id": project_id, "identity_id": identity_id,
            "acquisition_id": acquisition_id, "acquisition_sha256": acquisition_sha256,
        })
        return self.prepare_character_voice(project_id, identity_id, reference)

    def _synthesize_character_identity(self, project_id: str, role: Role, segment: Segment) -> TTSResult:
        identity = role.voice_identity
        if identity.status != "LOCKED":
            return TTSResult(False, {"error": "character voice is not locked"})
        performance = {"emotion": segment.emotion_text, "intensity": segment.emotion_alpha,
                       "pace": segment.playback_speed, "volume": segment.volume_gain,
                       "delivery": role.style_instruction}
        try:
            result = self._character_request("synthesize", {
                "project_id": project_id, "identity_id": identity.id,
                "prompt_artifact": identity.prompt_artifact, "prompt_sha256": identity.prompt_sha256,
                "reference_audio": identity.reference_audio, "reference_audio_sha256": identity.reference_audio_sha256,
                "text": segment.text, "language": segment.qwen3_language or role.qwen3_language or "English",
                "performance": performance,
            })
            return TTSResult(True, result, result.get("audio_file"))
        except Exception as exc:
            # Never fall back to fresh VoiceDesign and silently change identity.
            return TTSResult(False, {"error": str(exc), "identity_id": identity.id})

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

        identity = getattr(role, "voice_identity", None)
        if identity is not None:
            if identity.backend != "qwen3tts" or identity.mechanism != "qwen_reference":
                return TTSResult(False, {"error": "locked character identity is incompatible with VoiceDesign provider; explicit casting revision required",
                                         "identity_id": identity.id})
            return self._synthesize_character_identity(project_id, role, segment)

        try:
            self._bind_project_runtime(project_id)
        except Exception as exc:
            return TTSResult(False, {"error": str(exc)})

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

    def _post_request_control(self, endpoint: str, request_id: str) -> Dict[str, Any] | None:
        control = urllib.request.Request(
            f"{self.base_url}/api/v1/tts/{endpoint}",
            data=json.dumps({"request_id": request_id}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            control.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(control, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            return payload if isinstance(payload, dict) else None
        except Exception as exc:  # noqa: BLE001 - controls are best-effort
            logger.debug("Qwen request %s failed: %s", endpoint, exc)
            return None

    def _request_cancel(self, request_id: str) -> Dict[str, Any] | None:
        """Ask the isolated CUDA-graph worker to stop a timed-out request."""
        return self._post_request_control("cancel", request_id)

    def _request_status(self, request_id: str) -> Dict[str, Any] | None:
        return self._post_request_control("status", request_id)

    def _resolve_interrupted_request(self, request_id: str) -> TTSResult:
        cancel_state = self._request_cancel(request_id)
        state = self._request_status(request_id)
        for _ in range(12):
            if not state or state.get("state") != "active":
                break
            time.sleep(0.25)
            state = self._request_status(request_id)
        if state and state.get("state") == "completed":
            raw = state.get("response")
            if isinstance(raw, dict):
                path = raw.get("audio_file")
                if isinstance(path, str) and path:
                    return TTSResult(success=True, raw=raw, audio_path=path)
        if state and state.get("state") == "failed":
            return TTSResult(
                success=False,
                raw={
                    "error": str(state.get("error") or "worker confirmed request failure"),
                    "outcome": "confirmed_failure",
                    "request_id": request_id,
                    "cancel_state": (cancel_state or {}).get("state"),
                },
                audio_path=None,
            )
        return TTSResult(
            success=False,
            raw={
                "error": "TTS request was interrupted; worker outcome is not yet confirmed",
                "outcome": "uncertain",
                "request_id": request_id,
                "cancel_state": (cancel_state or {}).get("state"),
                "worker_status": (state or {}).get("state", "unavailable"),
            },
            audio_path=None,
        )

    def _build_payload(
        self,
        project_id: str,
        role: Role,
        segment: Segment,
        previous_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建 Qwen3-TTS CustomVoice API 请求参数。"""
        
        # Segment is a one-off override; the role profile is the stable default.
        requested_speaker = (
            getattr(segment, "qwen3_speaker", None)
            or getattr(role, "qwen3_speaker", None)
        )
        # Canonicalize before issuing a request so aliases have one stable
        # identity and unknown values fail closed at the provider boundary.
        speaker = resolve_customvoice_speaker(requested_speaker)
        
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
            "request_id": f"{segment.id}-{uuid.uuid4().hex[:12]}",
            "utterance_id": segment.id,
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

        identity = getattr(role, "voice_identity", None)
        if identity is not None:
            if identity.status != "LOCKED" or identity.backend != "qwen3tts_cv" or identity.mechanism != "builtin" or not identity.builtin_speaker:
                return TTSResult(False, {"error": "character identity is incompatible with CustomVoice provider; explicit casting revision required",
                                         "identity_id": identity.id})
            if segment.qwen3_speaker and segment.qwen3_speaker.casefold() != identity.builtin_speaker.casefold():
                return TTSResult(False, {"error": "segment speaker conflicts with locked character identity; explicit casting revision required",
                                         "identity_id": identity.id})
            # The accepted identity, rather than an independently edited role
            # setting, supplies the stable built-in speaker. Do not mutate the
            # canonical role while constructing one rendering request.
            role = role.model_copy(update={"qwen3_speaker": identity.builtin_speaker})

        url = f"{self.base_url}/api/v1/tts/synthesize"
        try:
            payload = self._build_payload(
                project_id=project_id,
                role=role,
                segment=segment,
                previous_text=previous_text,
            )
        except ValueError as exc:
            return TTSResult(success=False, raw={"error": str(exc), "outcome": "invalid_voice_selection"})

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
        except BaseException as exc:
            if not isinstance(exc, Exception):
                self._request_cancel(str(payload["request_id"]))
                raise
            return self._verify_effective_voice(self._resolve_interrupted_request(str(payload["request_id"])), payload)

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

        return self._verify_effective_voice(TTSResult(success=success, raw=raw, audio_path=audio_path), payload)

    def _verify_effective_voice(self, result: TTSResult, payload: Dict[str, Any]) -> TTSResult:
        """Apply the same identity check to direct and recovered worker replies."""
        raw = result.raw
        if result.success and isinstance(raw, dict):
            returned_speaker = raw.get("effective_speaker")
            if returned_speaker is not None:
                try:
                    returned_speaker = resolve_customvoice_speaker(str(returned_speaker))
                except ValueError:
                    returned_speaker = None
                if returned_speaker is None or returned_speaker != payload["speaker"]:
                    return TTSResult(success=False, raw={
                        "error": "TTS worker effective speaker does not match the requested speaker",
                        "outcome": "effective_voice_mismatch",
                        "requested_speaker": payload["speaker"],
                        "effective_speaker": returned_speaker,
                        "request_id": payload["request_id"],
                    })
                raw["effective_speaker"] = returned_speaker
            raw.setdefault("provider_voice_resolution", {
                "requested_speaker": payload["speaker"],
                "effective_speaker": returned_speaker,
                "verified": returned_speaker is not None and returned_speaker == payload["speaker"],
            })

        return result


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
