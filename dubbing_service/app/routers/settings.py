"""Platform-level configuration APIs (LLM/TTS service settings)."""

from __future__ import annotations

import json
import logging
import os
import queue as queue_module
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from dubbing_service.app.core.config import (
    clear_config_caches,
    get_cosyvoice_base_url,
    get_config,
    get_qwen3tts_base_url,
    get_qwen3tts_cv_base_url,
)
from dubbing_service.app.core.tts_backends import (
    BACKEND_SPECS,
    build_backend_command,
    public_backend_definitions,
)
from dubbing_service.app.core.tts_operations import (
    release_tts_operation,
    try_acquire_tts_operation,
)
from dubbing_service.app.core.secret_store import (
    SecretReference,
    SecretStoreError,
    delete_secret,
    get_secret,
    set_secret,
)
from dubbing_service.app.providers.subtitles import clear_whisper_model_cache
from dubbing_service.app.providers.llm import OpenAIStyleLLMProvider
from dubbing_service.app.routers.segments import SEG_PROMPT_DEFAULT_TEMPLATE
from dubbing_service.app.storage.llm_configs import (
    LLMConfigStorageError,
    load_llm_configs,
)
from dubbing_service.app.storage.llm_providers import (
    LLMProviderStorageError,
    load_llm_providers,
)


from dubbing_service.app.core.helpers import (
    mask_secret_value as _mask_secret,
    mask_secrets_in_text as _mask_secrets_in_text,
    sanitize_raw_for_response as _sanitize_raw_for_response,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class PlatformSettingsPayload(BaseModel):
    """平台级配置：TTS/LLM 服务信息。

    - tts_service_name: 仅用于展示的服务名称。
    - tts_base_url: TTS REST 服务地址。
    - cosyvoice_base_url: CosyVoice3 REST 服务地址。
    - qwen3tts_base_url: Qwen3-TTS VoiceDesign REST 服务地址。
    - qwen3tts_cv_base_url: Qwen3-TTS CustomVoice REST 服务地址。
    - tts_api_key: 可选的 TTS API Key。
    - llm_service_name: 仅用于展示的服务名称。
    - llm_base_url: LLM 服务 Base URL。
    - llm_api_key: LLM API Key。
    """

    tts_service_name: str | None = Field(default=None, description="TTS 服务名称，仅用于展示")
    tts_base_url: str | None = Field(default=None, description="TTS REST 服务地址")
    cosyvoice_base_url: str | None = Field(default=None, description="CosyVoice3 REST 服务地址")
    qwen3tts_base_url: str | None = Field(default=None, description="Qwen3-TTS VoiceDesign REST 服务地址")
    qwen3tts_cv_base_url: str | None = Field(default=None, description="Qwen3-TTS CustomVoice REST 服务地址")
    tts_api_key: str | None = Field(default=None, description="可选：TTS API Key")
    llm_service_name: str | None = Field(default=None, description="LLM 服务名称，仅用于展示")
    llm_base_url: str | None = Field(default=None, description="LLM Base URL")
    llm_api_key: str | None = Field(default=None, description="LLM API Key")


class PlatformSettingsResponse(BaseModel):
    tts_service_name: str | None = None
    tts_base_url: str | None = None
    cosyvoice_base_url: str | None = None
    qwen3tts_base_url: str | None = None
    qwen3tts_cv_base_url: str | None = None
    tts_api_key: str | None = None
    llm_service_name: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None


class LLMTestResponse(BaseModel):
    """测试 LLM 调用的返回结果。"""

    success: bool
    message: str
    raw: dict | None = None


# TTS 语音合成参数的默认值（来自 IndexTTS2 REST 推荐值）
TTS_PARAMS_DEFAULTS = {
    "max_text_tokens_per_segment": 120,
    "max_mel_tokens": 1500,
    "interval_silence": 200,
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": 30,
    "num_beams": 3,
    "repetition_penalty": 1.8,
    "length_penalty": 0.0,
    "emotion_alpha": 0.65,
    "context_injection_length": 50,  # 上下文注入：取前一分段后N个字符作为前情提要
    "cosyvoice_speed": 1.0,  # CosyVoice3 语速，0.5-2.0
    "qwen3_default_language": "Chinese",  # Qwen3 默认语言：Chinese/English/.../Auto
}


QWEN3_ALLOWED_LANGUAGES = {
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
}


class TTSParamsPayload(BaseModel):
    """TTS 语音合成参数配置。"""

    max_text_tokens_per_segment: int | None = Field(default=None, description="每段最大文本 token 数")
    max_mel_tokens: int | None = Field(default=None, description="最大音频 token 数")
    interval_silence: int | None = Field(default=None, description="分段间静音时长(ms)")
    temperature: float | None = Field(default=None, description="生成温度，越高越随机")
    top_p: float | None = Field(default=None, description="Nucleus 采样阈值")
    top_k: int | None = Field(default=None, description="Top-K 采样数量")
    num_beams: int | None = Field(default=None, description="Beam search 数量，越大质量越高但更慢")
    repetition_penalty: float | None = Field(default=None, description="重复惩罚，防止重复")
    length_penalty: float | None = Field(default=None, description="长度惩罚")
    emotion_alpha: float | None = Field(default=None, ge=0.0, le=1.0, description="新分段默认情感权重(0-1)")
    context_injection_length: int | None = Field(default=None, ge=0, le=200, description="上下文注入长度：取前一分段后N个字符(0=禁用)")
    cosyvoice_speed: float | None = Field(default=None, ge=0.5, le=2.0, description="CosyVoice3 语速，0.5-2.0")
    qwen3_default_language: str | None = Field(
        default=None,
        description="Qwen3 默认语言：Auto/Chinese/English/Japanese/Korean/German/French/Russian/Portuguese/Spanish/Italian",
    )

    @field_validator("qwen3_default_language")
    @classmethod
    def _validate_qwen3_default_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        for option in QWEN3_ALLOWED_LANGUAGES:
            if text.lower() == option.lower():
                return option
        raise ValueError("qwen3_default_language 必须是 Auto/Chinese/English/Japanese/Korean/German/French/Russian/Portuguese/Spanish/Italian")


class TTSParamsResponse(BaseModel):
    """TTS 参数响应，包含当前值和默认值。"""

    max_text_tokens_per_segment: int
    max_mel_tokens: int
    interval_silence: int
    temperature: float
    top_p: float
    top_k: int
    num_beams: int
    repetition_penalty: float
    length_penalty: float
    emotion_alpha: float
    context_injection_length: int
    cosyvoice_speed: float
    qwen3_default_language: str
    # 默认值，方便前端显示"恢复默认"
    defaults: dict


def _config_file_path() -> Path:
    """返回 dubbing_service/config.json 的路径。"""

    return Path(__file__).resolve().parents[2] / "config.json"


_CONFIG_FILE_LOCK = threading.RLock()


def _load_raw_file_config_unlocked() -> dict:
    path = _config_file_path()
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_raw_file_config_unlocked(data: dict) -> None:
    """原子写入配置文件，防止并发写入导致数据损坏。"""
    path = _config_file_path()

    # 原子写入：先写临时文件，再重命名
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix="config_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_raw_file_config() -> dict:
    with _CONFIG_FILE_LOCK:
        return _load_raw_file_config_unlocked()


def _save_raw_file_config(data: dict) -> None:
    with _CONFIG_FILE_LOCK:
        _save_raw_file_config_unlocked(data)


def _update_raw_file_config(mutator: Callable[[dict], None]) -> dict:
    """在同一把锁内执行 load → mutate → save，避免并发覆盖。"""
    with _CONFIG_FILE_LOCK:
        data = _load_raw_file_config_unlocked()
        mutator(data)
        _save_raw_file_config_unlocked(data)
        return data


@router.get("/settings/platform", response_model=PlatformSettingsResponse)
async def api_get_platform_settings() -> PlatformSettingsResponse:
    """获取当前平台级 TTS/LLM 配置。

    返回值以 AppConfig 中的生效值为主，展示名称等仅从 config.json 读取。
    """

    cfg = get_config()
    file_cfg = _load_raw_file_config()

    return PlatformSettingsResponse(
        tts_service_name=file_cfg.get("tts_service_name") or None,
        tts_base_url=cfg.tts_base_url,
        cosyvoice_base_url=cfg.cosyvoice_base_url,
        qwen3tts_base_url=cfg.qwen3tts_base_url,
        qwen3tts_cv_base_url=cfg.qwen3tts_cv_base_url,
        tts_api_key=_mask_secret(cfg.tts_api_key),
        llm_service_name=file_cfg.get("llm_service_name") or None,
        llm_base_url=cfg.llm_base_url,
        llm_api_key=_mask_secret(cfg.llm_api_key),
    )


@router.put("/settings/platform", response_model=PlatformSettingsResponse)
async def api_update_platform_settings(payload: PlatformSettingsPayload) -> PlatformSettingsResponse:
    """更新平台级 TTS/LLM 配置。

    会把配置写入 dubbing_service/config.json，并刷新全局 AppConfig 缓存。
    注意：如果同时设置了环境变量（优先级更高），可能会覆盖这里的文件配置。
    """

    updates = payload.model_dump(exclude_unset=True)

    before = _load_raw_file_config()
    secret_changes: dict[str, tuple[str | None, str | None, str | None]] = {}
    for key, account in (
        ("tts_api_key", "platform-tts-api-key"),
        ("llm_api_key", "platform-llm-api-key"),
    ):
        if key not in updates:
            continue
        raw_value = updates[key]
        if isinstance(raw_value, str) and raw_value.strip() == "********":
            continue
        old_ref = before.get(f"{key}_ref")
        old_secret: str | None = None
        parsed = SecretReference.parse(old_ref if isinstance(old_ref, str) else None)
        if parsed is not None and parsed.kind == "keyring":
            try:
                old_secret = get_secret(old_ref)
            except SecretStoreError:
                old_secret = None
        try:
            value = str(raw_value or "").strip()
            new_ref = set_secret(account, value) if value else None
        except SecretStoreError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        secret_changes[key] = (
            new_ref,
            old_ref if isinstance(old_ref, str) else None,
            old_secret,
        )

    def _mutate(data: dict) -> None:
        # 仅在用户显式提供字段时更新对应键
        for key in (
            "tts_service_name",
            "tts_base_url",
            "cosyvoice_base_url",
            "qwen3tts_base_url",
            "qwen3tts_cv_base_url",
            "llm_service_name",
            "llm_base_url",
        ):
            if key in updates:
                data[key] = updates[key]
        for key, (new_ref, _old_ref, _old_secret) in secret_changes.items():
            data.pop(key, None)
            ref_key = f"{key}_ref"
            if new_ref:
                data[ref_key] = new_ref
            else:
                data.pop(ref_key, None)

    try:
        _update_raw_file_config(_mutate)
    except Exception as exc:  # pragma: no cover - 简单封装
        for _key, (new_ref, old_ref, old_secret) in secret_changes.items():
            try:
                if old_ref and old_secret:
                    parsed = SecretReference.parse(old_ref)
                    if parsed is not None and parsed.kind == "keyring":
                        set_secret(parsed.account, old_secret)
                    if new_ref and new_ref != old_ref:
                        delete_secret(new_ref)
                elif new_ref:
                    delete_secret(new_ref)
            except SecretStoreError:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存配置失败: {exc}",
        ) from exc

    for _key, (new_ref, old_ref, _old_secret) in secret_changes.items():
        if old_ref and old_ref != new_ref:
            try:
                delete_secret(old_ref)
            except SecretStoreError:
                logger.warning("旧平台系统凭据清理失败", exc_info=True)

    # 刷新配置缓存，让后续 LLM/TTS Provider 使用新的配置
    try:
        clear_config_caches()
    except Exception:
        # 失败时忽略，让下次进程重启后至少能生效
        logger.warning("刷新配置缓存失败（platform settings）", exc_info=True)

    # 使用刷新后的配置构造响应
    cfg = get_config()
    refreshed_file_cfg = _load_raw_file_config()

    return PlatformSettingsResponse(
        tts_service_name=refreshed_file_cfg.get("tts_service_name") or None,
        tts_base_url=cfg.tts_base_url,
        cosyvoice_base_url=cfg.cosyvoice_base_url,
        qwen3tts_base_url=cfg.qwen3tts_base_url,
        qwen3tts_cv_base_url=cfg.qwen3tts_cv_base_url,
        tts_api_key=_mask_secret(cfg.tts_api_key),
        llm_service_name=refreshed_file_cfg.get("llm_service_name") or None,
        llm_base_url=cfg.llm_base_url,
        llm_api_key=_mask_secret(cfg.llm_api_key),
    )


@router.post("/settings/test-llm", response_model=LLMTestResponse)
def api_test_llm() -> LLMTestResponse:
    """测试默认 LLM 是否可用。

    优先使用 **默认 LLM 配置**（llm_models.json 中 is_default=True 的配置）来确定：
    - 使用的模型名称（model）；
    - 首选的 LLM 提供商（provider_id 绑定的提供商）。

    如果未配置任何 LLM 配置，则回退到：
    - LLM 提供商列表中的默认提供商；
    - 再回退到全局 LLM 配置 (config.json / 环境变量)。
    """

    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None

    # 1. 读取默认 LLM 配置，用于确定模型和首选提供商
    try:
        configs = load_llm_configs()
    except LLMConfigStorageError:
        configs = []

    default_config = None
    if configs:
        default_config = next((c for c in configs if c.is_default), configs[0])
        model = default_config.model

    # 2. 读取 LLM 提供商，根据默认配置的 provider_id 优先选择对应 provider
    try:
        providers = load_llm_providers()
    except LLMProviderStorageError:
        providers = []

    if providers:
        target_provider = None
        if default_config and default_config.provider_id:
            target_provider = next(
                (p for p in providers if p.id == default_config.provider_id),
                None,
            )

        if target_provider is None:
            target_provider = next((p for p in providers if p.is_default), providers[0])

        base_url = target_provider.base_url
        api_key = target_provider.api_key

    # 3. 初始化 Provider：如果有默认配置，则显式传入 model；否则使用 Provider 内部默认模型
    try:
        if model:
            provider = OpenAIStyleLLMProvider(model=model, base_url=base_url, api_key=api_key)
        else:
            provider = OpenAIStyleLLMProvider(base_url=base_url, api_key=api_key)
    except Exception as exc:  # pragma: no cover - 简单封装
        return LLMTestResponse(
            success=False,
            message=f"初始化 LLM Provider 失败: {exc}",
            raw=None,
        )

    messages = [
        {"role": "system", "content": "你是一个用于连通性检测的助手。"},
        {"role": "user", "content": "请简要回复：pong"},
    ]

    result = provider.chat(messages, max_tokens=16)

    if not result.success:
        raw = result.raw or {}
        safe_raw = _sanitize_raw_for_response(raw)
        msg = None
        if isinstance(safe_raw, dict):
            msg = safe_raw.get("error") or safe_raw.get("message")
        msg_out = str(msg) if msg else "LLM 调用失败"
        return LLMTestResponse(
            success=False,
            message=_mask_secrets_in_text(msg_out),
            raw=safe_raw,
        )

    text = (result.text or "").strip() or "LLM 调用成功，但返回内容为空。"
    safe_raw = _sanitize_raw_for_response(result.raw)
    return LLMTestResponse(success=True, message=_mask_secrets_in_text(text), raw=safe_raw)


@router.post("/settings/gpu/unload-whisper")
async def api_unload_whisper_model() -> dict[str, object]:
    """卸载 Whisper 模型缓存，尽量释放显存。

    仅影响当前配音平台进程中的 faster-whisper 模型，不会终止正在进行的字幕生成任务。
    """

    clear_whisper_model_cache()
    return {
        "success": True,
        "message": "已请求卸载 Whisper 模型缓存并释放显存",
    }


@router.post("/settings/gpu/unload-tts")
def api_unload_tts_model() -> dict[str, object]:
    """通过 REST 调用 TTS 服务的卸载接口，释放 TTS 模型显存。"""

    target_backend = _get_tts_target_backend()

    try:
        data = _call_tts_endpoint_json(
            target_backend,
            "/api/v1/tts/unload-model",
            method="POST",
            timeout=5,
        )
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"调用 TTS 卸载接口失败: {exc}",
        ) from exc

    return {
        "success": True,
        "message": "已请求 TTS 服务卸载模型并释放显存",
        "response": data,
    }


@router.post("/settings/gpu/unload-sfx")
def api_unload_stable_audio_model() -> dict[str, object]:
    """Unload the managed Stable Audio model without stopping the platform."""

    from dubbing_service.app.core.sfx_worker import SFXWorkerError, unload_model

    try:
        data = unload_model()
    except SFXWorkerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "success": True,
        "message": "已请求 Stable Audio SFX worker 卸载模型并释放显存",
        "response": data,
    }


def _get_tts_params_from_config() -> dict:
    """从配置文件读取 TTS 参数，缺失的使用默认值。"""
    file_cfg = _load_raw_file_config()
    tts_params = file_cfg.get("tts_params", {})
    if not isinstance(tts_params, dict):
        tts_params = {}

    # 合并默认值
    result = {}
    for key, default_val in TTS_PARAMS_DEFAULTS.items():
        result[key] = tts_params.get(key, default_val)
    return result


@router.get("/settings/tts-params", response_model=TTSParamsResponse)
async def api_get_tts_params() -> TTSParamsResponse:
    """获取当前 TTS 语音合成参数配置。"""
    params = _get_tts_params_from_config()
    return TTSParamsResponse(**params, defaults=TTS_PARAMS_DEFAULTS)


@router.put("/settings/tts-params", response_model=TTSParamsResponse)
async def api_update_tts_params(payload: TTSParamsPayload) -> TTSParamsResponse:
    """更新 TTS 语音合成参数配置。"""
    # 仅更新用户显式提供的字段
    updates = payload.model_dump(exclude_unset=True)

    def _mutate(file_cfg: dict) -> None:
        tts_params = file_cfg.get("tts_params", {})
        if not isinstance(tts_params, dict):
            tts_params = {}
        for key, val in updates.items():
            if val is not None:
                tts_params[key] = val
        file_cfg["tts_params"] = tts_params

    try:
        _update_raw_file_config(_mutate)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存配置失败: {exc}",
        ) from exc

    params = _get_tts_params_from_config()
    return TTSParamsResponse(**params, defaults=TTS_PARAMS_DEFAULTS)


@router.post("/settings/tts-params/reset", response_model=TTSParamsResponse)
async def api_reset_tts_params() -> TTSParamsResponse:
    """重置 TTS 参数为默认值。"""
    try:
        _update_raw_file_config(
            lambda file_cfg: file_cfg.__setitem__("tts_params", TTS_PARAMS_DEFAULTS.copy())
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存配置失败: {exc}",
        ) from exc

    return TTSParamsResponse(**TTS_PARAMS_DEFAULTS, defaults=TTS_PARAMS_DEFAULTS)


# ─────────────────────────────────────────────────────────────────────────────
# LLM 分段提示词模板配置
# ─────────────────────────────────────────────────────────────────────────────

class SegPromptTemplatePayload(BaseModel):
    """LLM 分段提示词模板更新请求。"""

    template: str = Field(..., description="提示词模板内容")


class SegPromptTemplateResponse(BaseModel):
    """LLM 分段提示词模板响应。"""

    template: str
    is_default: bool = Field(description="是否为默认模板")


def get_seg_prompt_template() -> str:
    """获取当前全局 LLM 分段提示词模板。"""
    file_cfg = _load_raw_file_config()
    template = file_cfg.get("seg_prompt_template")
    if template and isinstance(template, str) and template.strip():
        return template
    return SEG_PROMPT_DEFAULT_TEMPLATE


@router.get("/settings/seg-prompt-template", response_model=SegPromptTemplateResponse)
async def api_get_seg_prompt_template() -> SegPromptTemplateResponse:
    """获取当前 LLM 分段提示词模板。"""
    file_cfg = _load_raw_file_config()
    template = file_cfg.get("seg_prompt_template")
    if template and isinstance(template, str) and template.strip():
        return SegPromptTemplateResponse(template=template, is_default=False)
    return SegPromptTemplateResponse(template=SEG_PROMPT_DEFAULT_TEMPLATE, is_default=True)


@router.put("/settings/seg-prompt-template", response_model=SegPromptTemplateResponse)
async def api_update_seg_prompt_template(payload: SegPromptTemplatePayload) -> SegPromptTemplateResponse:
    """更新 LLM 分段提示词模板。"""
    # 如果保存的内容与内置默认模板完全相同，则删除字段（视为使用默认）
    template_normalized = payload.template.strip()
    default_normalized = SEG_PROMPT_DEFAULT_TEMPLATE.strip()

    is_default = template_normalized == default_normalized

    def _mutate(file_cfg: dict) -> None:
        if is_default:
            # 内容与默认相同，删除字段
            file_cfg.pop("seg_prompt_template", None)
        else:
            file_cfg["seg_prompt_template"] = payload.template

    try:
        _update_raw_file_config(_mutate)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存配置失败: {exc}",
        ) from exc

    return SegPromptTemplateResponse(template=payload.template, is_default=is_default)


@router.post("/settings/seg-prompt-template/reset", response_model=SegPromptTemplateResponse)
async def api_reset_seg_prompt_template() -> SegPromptTemplateResponse:
    """重置 LLM 分段提示词模板为默认值。"""
    try:
        _update_raw_file_config(lambda file_cfg: file_cfg.pop("seg_prompt_template", None))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存配置失败: {exc}",
        ) from exc

    return SegPromptTemplateResponse(template=SEG_PROMPT_DEFAULT_TEMPLATE, is_default=True)


# ─────────────────────────────────────────────────────────────────────────────
# TTS 后端配置
# ─────────────────────────────────────────────────────────────────────────────

TTS_BACKENDS = public_backend_definitions()

TTS_BACKEND_IDS = set(BACKEND_SPECS)


def _normalize_tts_backend(value: object) -> str:
    """将后端值规范化为受支持的后端 ID。"""
    backend = str(value or "").strip()
    if backend in TTS_BACKEND_IDS:
        return backend
    return "indextts"


class TTSBackendPayload(BaseModel):
    """TTS 后端切换请求。"""
    backend: str = Field(..., description="TTS 后端类型: indextts | cosyvoice3 | qwen3tts | qwen3tts_cv")


class TTSBackendResponse(BaseModel):
    """TTS 后端响应。"""
    backend: str
    available_backends: list


@router.get("/settings/tts-backend", response_model=TTSBackendResponse)
async def api_get_tts_backend() -> TTSBackendResponse:
    """获取当前 TTS 后端配置。"""
    file_cfg = _load_raw_file_config()
    backend = _normalize_tts_backend(file_cfg.get("tts_backend"))
    return TTSBackendResponse(backend=backend, available_backends=TTS_BACKENDS)


@router.put("/settings/tts-backend", response_model=TTSBackendResponse)
async def api_update_tts_backend(payload: TTSBackendPayload) -> TTSBackendResponse:
    """切换 TTS 后端。
    
    注意：切换后端后需要重启 TTS 服务才能生效。
    """
    valid_backends = sorted(TTS_BACKEND_IDS)
    if payload.backend not in valid_backends:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无效的后端类型: {payload.backend}，可选: {valid_backends}",
        )

    try:
        _update_raw_file_config(
            lambda file_cfg: file_cfg.__setitem__("tts_backend", payload.backend)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存配置失败: {exc}",
        ) from exc

    # 刷新配置缓存
    try:
        clear_config_caches()
    except Exception:
        logger.warning("刷新配置缓存失败（tts backend）", exc_info=True)

    return TTSBackendResponse(backend=payload.backend, available_backends=TTS_BACKENDS)


# ==================== TTS 服务进程管理 ====================

_tts_process: subprocess.Popen | None = None
_tts_process_lock = threading.Lock()
_tts_log_queue: "queue_module.Queue[str]" = queue_module.Queue(maxsize=1000)
_tts_reader_thread: threading.Thread | None = None
_tts_runtime_backend: str | None = None
_tts_instance_token: str | None = None


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_loopback_port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex(("127.0.0.1", int(port))) == 0


def _get_tts_runtime_state() -> tuple[bool, int | None, str | None]:
    global _tts_process, _tts_runtime_backend, _tts_instance_token

    with _tts_process_lock:
        running = _tts_process is not None and _tts_process.poll() is None
        if not running:
            _tts_process = None
            _tts_runtime_backend = None
            _tts_instance_token = None
            return False, None, None
        return True, _tts_process.pid if _tts_process else None, _tts_runtime_backend


def _get_configured_tts_backend() -> str:
    file_cfg = _load_raw_file_config()
    return _normalize_tts_backend(file_cfg.get("tts_backend"))


def _get_tts_backend_base_url(backend: str | None) -> str:
    cfg = get_config()
    backend_id = _normalize_tts_backend(backend)
    base_urls = {
        "indextts": cfg.tts_base_url,
        "cosyvoice3": get_cosyvoice_base_url(),
        "qwen3tts": get_qwen3tts_base_url(),
        "qwen3tts_cv": get_qwen3tts_cv_base_url(),
    }
    return str(base_urls.get(backend_id) or cfg.tts_base_url).rstrip("/")


def _call_tts_endpoint_json(
    backend: str | None,
    path: str,
    *,
    method: str = "GET",
    timeout: float = 45,
) -> dict[str, object]:
    url = f"{_get_tts_backend_base_url(backend)}{path if path.startswith('/') else '/' + path}"
    req = urllib.request.Request(
        url,
        data=None if method.upper() in {"GET", "HEAD"} else b"",
        headers={},
        method=method.upper(),
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(body)
    except Exception:
        data = {"raw": body}
    return data if isinstance(data, dict) else {"raw": data}


def unload_managed_tts_model_for_gpu_handoff() -> dict[str, object] | None:
    """Unload an idle managed TTS model before an in-process GPU workload.

    The caller must already own the global TTS/GPU operation lease.  Keeping
    the worker process alive makes the next synthesis lazily reload its model,
    while avoiding concurrent TTS and Whisper residency on smaller GPUs.
    """

    running, _pid, runtime_backend = _get_tts_runtime_state()
    if not running or not runtime_backend:
        return None
    return _call_tts_endpoint_json(
        runtime_backend,
        "/api/v1/tts/unload-model",
        method="POST",
        timeout=60,
    )


def _get_tts_target_backend(requested_backend: str | None = None) -> str:
    running, _, runtime_backend = _get_tts_runtime_state()
    if running and runtime_backend:
        return runtime_backend
    if requested_backend:
        return _normalize_tts_backend(requested_backend)
    return _get_configured_tts_backend()


def _tts_reader_loop(proc: subprocess.Popen) -> None:
    """读取 TTS 进程输出到日志队列"""
    if proc.stdout is None:
        return
    try:
        for line in proc.stdout:
            try:
                _tts_log_queue.put_nowait(line)
            except queue_module.Full:
                # 队列满了，丢弃旧日志
                try:
                    _tts_log_queue.get_nowait()
                    _tts_log_queue.put_nowait(line)
                except Exception:
                    logger.debug("写入 TTS 日志队列失败", exc_info=True)
    except Exception:
        logger.debug("TTS 日志读取线程异常退出", exc_info=True)


def _wait_for_tts_ready(
    proc: subprocess.Popen,
    backend: str,
    *,
    expected_instance_token: str,
    timeout_seconds: float = 45.0,
) -> dict[str, object]:
    """Wait until the worker has bound its health endpoint or exited."""
    import time

    url = f"{_get_tts_backend_base_url(backend)}/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            raise RuntimeError(f"TTS 服务启动后退出，exit_code={exit_code}")
        try:
            with opener.open(url, timeout=0.8) as response:
                body = response.read().decode("utf-8", errors="replace")
            payload = json.loads(body) if body else {}
            if not isinstance(payload, dict):
                raise RuntimeError("TTS 健康检查返回了非对象 JSON")
            if payload.get("backend") != backend:
                raise RuntimeError(
                    f"TTS 健康检查后端不匹配: {payload.get('backend')} != {backend}"
                )
            if payload.get("instance_token") != expected_instance_token:
                raise RuntimeError("TTS 健康检查实例令牌不匹配，端口可能属于旧进程")
            return payload
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise TimeoutError(f"等待 TTS 服务就绪超时: {url}; last_error={last_error}")


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Gracefully stop the worker group, then hard-kill only on timeout."""
    if proc.poll() is not None:
        return
    pid = proc.pid
    if pid is None:
        return
    if sys.platform.startswith("win"):
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=3)
            if proc.poll() is not None:
                return
        except (OSError, subprocess.SubprocessError):
            pass
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            # 向整个进程组发送 SIGTERM（start_new_session=True 使子进程在独立会话中）
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            # 进程已退出或无权限，回退到只终止主进程
            try:
                proc.terminate()
            except Exception:
                logger.debug("终止主进程失败（fallback）", exc_info=True)


class TTSServiceStatusResponse(BaseModel):
    """TTS 服务状态响应"""
    running: bool
    backend: str | None = None
    runtime_backend: str | None = None
    config_backend: str | None = None
    pid: int | None = None
    logs: list[str] = Field(default_factory=list)


class TTSServiceStartPayload(BaseModel):
    """启动 TTS 服务请求"""
    backend: str = Field("indextts", description="TTS 后端: indextts/cosyvoice3/qwen3tts/qwen3tts_cv")
    fp32: bool = Field(False, description="是否使用 FP32（仅 IndexTTS2）")
    persist_default: bool = Field(
        True,
        description="是否同时将本次启动的后端保存为平台默认后端；自动任务切换 worker 时应设为 false",
    )


class TTSServiceLoadModelPayload(BaseModel):
    """通过平台代理触发 TTS 服务加载模型。"""
    backend: str | None = Field(default=None, description="可选：指定目标后端。默认优先作用于当前运行中的后端")


@router.get("/settings/tts-service/status", response_model=TTSServiceStatusResponse)
async def api_get_tts_service_status() -> TTSServiceStatusResponse:
    """获取 TTS 服务运行状态"""
    running, pid, runtime_backend = _get_tts_runtime_state()
    
    # 获取最近的日志
    logs = []
    while not _tts_log_queue.empty() and len(logs) < 100:
        try:
            logs.append(_tts_log_queue.get_nowait())
        except queue_module.Empty:
            break
    
    # 获取当前配置的后端
    file_cfg = _load_raw_file_config()
    config_backend = _normalize_tts_backend(file_cfg.get("tts_backend"))
    
    return TTSServiceStatusResponse(
        running=running,
        backend=runtime_backend if running else None,
        runtime_backend=runtime_backend if running else None,
        config_backend=config_backend,
        pid=pid,
        logs=logs,
    )


@router.post("/settings/tts-service/start", response_model=TTSServiceStatusResponse)
def api_start_tts_service(payload: TTSServiceStartPayload) -> TTSServiceStatusResponse:
    """Start or switch the managed TTS worker under an exclusive GPU lease."""

    if payload.backend not in TTS_BACKEND_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无效的后端类型: {payload.backend}，可选: {sorted(TTS_BACKEND_IDS)}",
        )

    operation_lease = try_acquire_tts_operation(kind="tts_service_start")
    if operation_lease is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="批量合成正在执行，暂时不能启动或切换 TTS 服务。",
        )
    try:
        return _start_tts_service_exclusive(payload)
    finally:
        release_tts_operation(operation_lease)


def _start_tts_service_exclusive(
    payload: TTSServiceStartPayload,
) -> TTSServiceStatusResponse:
    """启动 TTS 服务"""
    global _tts_process, _tts_reader_thread, _tts_runtime_backend, _tts_instance_token
    
    if payload.backend not in TTS_BACKEND_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"无效的后端类型: {payload.backend}，可选: {sorted(TTS_BACKEND_IDS)}",
        )

    project_root = _get_project_root()
    config_backend = _get_configured_tts_backend()
    try:
        cmd = build_backend_command(
            project_root,
            payload.backend,
            Path(sys.executable),
            fp32=payload.fp32,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"TTS 后端尚未配置完整: {exc}",
        ) from exc

    with _tts_process_lock:
        # 如果已经在运行，先停止
        if _tts_process is not None and _tts_process.poll() is None:
            _tts_log_queue.put("[TTS 服务] 正在停止旧服务...\n")
            _kill_process_tree(_tts_process)
            try:
                _tts_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    _tts_process.kill()
                    _tts_process.wait(timeout=3)
                except Exception:
                    logger.warning("强制结束旧 TTS 服务失败", exc_info=True)
            except Exception:
                logger.warning("等待旧 TTS 服务退出时异常", exc_info=True)
            _tts_process = None
            _tts_runtime_backend = None
            _tts_instance_token = None
            # 等待端口释放
            time.sleep(1)
            _tts_log_queue.put("[TTS 服务] 旧服务已停止\n")

        target_port = BACKEND_SPECS[payload.backend].port
        if _is_loopback_port_listening(target_port):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"端口 {target_port} 已被未受当前平台管理的进程占用。"
                    "请先确认并停止旧 TTS 工作进程。"
                ),
            )
        
        # 用户主动启动/切换时同步保存平台默认值；合成队列和 AI 预览等
        # 内部自动切换只改变运行中的 worker，不应悄悄改写用户默认配置。
        if payload.persist_default:
            try:
                _update_raw_file_config(
                    lambda file_cfg: file_cfg.__setitem__("tts_backend", payload.backend)
                )
                clear_config_caches()
                config_backend = payload.backend
            except Exception:
                logger.warning("启动 TTS 服务前写入后端配置失败", exc_info=True)
        
        # 启动进程
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("MODELSCOPE_CACHE", str(project_root / "models" / ".modelscope_cache"))
        env.setdefault("COSYVOICE_WETEXT_DIR", str(project_root / "models" / "cosyvoice3" / "wetext"))
        env.setdefault("NO_PROXY", "127.0.0.1,localhost")
        env.setdefault("no_proxy", "127.0.0.1,localhost")
        instance_token = secrets.token_urlsafe(32)
        env["DUBBING_WORKER_INSTANCE_TOKEN"] = instance_token
        
        popen_kwargs: dict = {}
        if sys.platform.startswith("win"):
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        
        try:
            # Do not discard the worker's first diagnostics after the reader
            # thread has already started.
            while not _tts_log_queue.empty():
                try:
                    _tts_log_queue.get_nowait()
                except queue_module.Empty:
                    break

            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **popen_kwargs,
            )
            _tts_process = proc
            _tts_runtime_backend = payload.backend
            _tts_instance_token = instance_token
            
            # 启动日志读取线程
            _tts_reader_thread = threading.Thread(target=_tts_reader_loop, args=(proc,), daemon=True)
            _tts_reader_thread.start()
            _tts_log_queue.put(f"[TTS 服务] 进程已创建，等待健康检查 (PID={proc.pid}, backend={payload.backend})\n")

            health = _wait_for_tts_ready(
                proc,
                payload.backend,
                expected_instance_token=instance_token,
            )
            _tts_log_queue.put(
                f"[TTS 服务] 已就绪 (PID={proc.pid}, backend={payload.backend}, status={health.get('status')})\n"
            )
            
        except Exception as exc:
            if _tts_process is not None and _tts_process.poll() is None:
                _kill_process_tree(_tts_process)
            _tts_process = None
            _tts_runtime_backend = None
            _tts_instance_token = None
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"启动 TTS 服务失败: {exc}",
            ) from exc
    
    return TTSServiceStatusResponse(
        running=True,
        backend=payload.backend,
        runtime_backend=payload.backend,
        config_backend=config_backend,
        pid=_tts_process.pid if _tts_process else None,
        logs=[],
    )


@router.post("/settings/tts-service/load-model")
def api_load_tts_model(payload: TTSServiceLoadModelPayload | None = None) -> dict[str, object]:
    """通过平台代理触发当前 TTS 服务加载模型。"""
    requested_backend = None
    if payload and payload.backend is not None:
        requested_backend = str(payload.backend).strip()
        if requested_backend and requested_backend not in TTS_BACKEND_IDS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的后端类型: {requested_backend}，可选: {sorted(TTS_BACKEND_IDS)}",
            )

    target_backend = _get_tts_target_backend(requested_backend or None)

    try:
        data = _call_tts_endpoint_json(
            target_backend,
            "/api/v1/tts/load-model",
            method="POST",
            timeout=900,
        )
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"调用 TTS 加载模型接口失败: {exc}",
        ) from exc

    return {
        "success": True,
        "message": "已请求 TTS 服务加载模型",
        "backend": target_backend,
        "response": data,
    }


@router.post("/settings/tts-service/stop", response_model=TTSServiceStatusResponse)
def api_stop_tts_service() -> TTSServiceStatusResponse:
    """Stop the managed worker without racing an in-flight synthesis batch."""

    operation_lease = try_acquire_tts_operation(kind="tts_service_stop")
    if operation_lease is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="批量合成正在执行，暂时不能停止 TTS 服务。",
        )
    try:
        return _stop_tts_service_exclusive()
    finally:
        release_tts_operation(operation_lease)


def _stop_tts_service_exclusive() -> TTSServiceStatusResponse:
    """停止 TTS 服务"""
    global _tts_process, _tts_runtime_backend, _tts_instance_token

    file_cfg = _load_raw_file_config()
    config_backend = _normalize_tts_backend(file_cfg.get("tts_backend"))
    
    with _tts_process_lock:
        if _tts_process is None or _tts_process.poll() is not None:
            _tts_process = None
            _tts_runtime_backend = None
            _tts_instance_token = None
            return TTSServiceStatusResponse(running=False, config_backend=config_backend, logs=[])
        
        _tts_log_queue.put("[TTS 服务] 正在停止...\n")
        _kill_process_tree(_tts_process)
        
        try:
            _tts_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # SIGTERM 超时，强制杀死
            try:
                _tts_process.kill()
                _tts_process.wait(timeout=3)
            except Exception:
                logger.warning("停止 TTS 服务时强制 kill 失败", exc_info=True)
        except Exception:
            logger.warning("等待 TTS 服务停止时异常", exc_info=True)
        
        _tts_log_queue.put("[TTS 服务] 已停止\n")
        _tts_process = None
        _tts_runtime_backend = None
        _tts_instance_token = None
    
    return TTSServiceStatusResponse(running=False, config_backend=config_backend, logs=[])


def shutdown_tts_process() -> None:
    """应用退出时清理 TTS 子进程（由 lifespan shutdown 调用）。"""
    global _tts_process, _tts_runtime_backend, _tts_instance_token

    with _tts_process_lock:
        if _tts_process is None or _tts_process.poll() is not None:
            _tts_process = None
            _tts_runtime_backend = None
            _tts_instance_token = None
            return

        _kill_process_tree(_tts_process)
        try:
            _tts_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                _tts_process.kill()
                _tts_process.wait(timeout=3)
            except Exception:
                logger.warning("应用退出清理 TTS 进程：强制 kill 失败", exc_info=True)
        except Exception:
            logger.warning("应用退出清理 TTS 进程：等待退出时异常", exc_info=True)
        finally:
            _tts_process = None
            _tts_runtime_backend = None
            _tts_instance_token = None
