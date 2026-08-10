"""LLM configuration management APIs."""

from __future__ import annotations

from typing import List, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from dubbing_service.app.core.config import get_config
from dubbing_service.app.core.helpers import (
    mask_secrets_in_text as _mask_secrets_in_text,
    sanitize_raw_for_response as _sanitize_raw_for_response,
)
from dubbing_service.app.providers.llm import OpenAIStyleLLMProvider
from dubbing_service.app.storage.llm_configs import (
    LLMConfig,
    LLMConfigStorageError,
    generate_llm_config_id,
    llm_configs_lock,
    load_llm_configs,
    save_llm_configs,
)
from dubbing_service.app.storage.llm_providers import (
    LLMProviderStorageError,
    load_llm_providers,
    save_llm_providers,
)


router = APIRouter()


class LLMConfigPayload(BaseModel):
    """创建/更新 LLM 配置的请求体。"""

    name: str = Field(..., min_length=1, max_length=100, description="展示名称")
    provider_id: str | None = Field(
        default=None,
        description="可选：关联的 LLM 提供商 ID；为空时使用默认提供商",
    )
    model: str = Field(..., min_length=1, max_length=100, description="LLM 模型名称，如 deepseek-v4-flash")
    max_tokens: int | None = Field(
        None,
        ge=1,
        le=65536,
        description="可选：覆盖模型默认的 max_tokens；不填则使用模型默认值（当前全局上限 65536）",
    )
    description: str | None = Field(
        None,
        max_length=500,
        description="可选：说明该配置的用途，例如默认 4K / 8K 上限等",
    )
    response_format_json: bool = Field(
        default=False,
        description="是否在调用时启用 response_format={\"type\": \"json_object\"}",
    )
    thinking: Literal["enabled", "disabled"] | None = Field(
        default=None,
        description="DeepSeek V4 thinking 模式；为空时不覆盖服务端默认值",
    )
    reasoning_effort: Literal["high", "max"] | None = Field(
        default=None,
        description="DeepSeek V4 思考强度，仅在 thinking=enabled 时使用",
    )
    fallback_model: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="主模型校验失败或遇到临时错误时使用的一次性备用模型",
    )
    fallback_enabled: bool = False


class LLMConfigResponse(BaseModel):
    id: str
    name: str
    provider_id: str | None = None
    model: str
    max_tokens: int | None = None
    description: str | None = None
    response_format_json: bool = False
    thinking: Literal["enabled", "disabled"] | None = None
    reasoning_effort: Literal["high", "max"] | None = None
    fallback_model: str | None = None
    fallback_enabled: bool = False
    is_default: bool = False


class LLMConfigTestResponse(BaseModel):
    """测试单个 LLM 配置连通性的返回结果。"""

    success: bool
    message: str
    raw: dict | None = None


def _ensure_default(configs: List[LLMConfig]) -> List[LLMConfig]:
    """确保至少有一个配置被标记为默认。

    如果当前没有任何 is_default=True，则将第一个配置标记为默认。
    """

    if not configs:
        return configs

    if not any(c.is_default for c in configs):
        configs[0].is_default = True

    return configs


def _seed_default_if_empty(configs: List[LLMConfig]) -> List[LLMConfig]:
    """如果当前没有任何配置，则根据全局配置自动生成一个默认项。"""

    if configs:
        return configs

    cfg = get_config()

    default_model = "gpt-4o-mini"
    is_deepseek = bool(cfg.llm_base_url and "deepseek.com" in cfg.llm_base_url.lower())
    if is_deepseek:
        default_model = "deepseek-v4-flash"

    default_cfg = LLMConfig(
        id=generate_llm_config_id(),
        name=f"{default_model} (默认)",
        model=default_model,
        max_tokens=64000 if is_deepseek else None,
        description="DeepSeek V4 Flash 默认分段模型。" if is_deepseek else "从全局 LLM 配置推断出的默认模型。",
        response_format_json=is_deepseek,
        thinking="disabled" if is_deepseek else None,
        fallback_model="deepseek-v4-pro" if is_deepseek else None,
        fallback_enabled=is_deepseek,
        is_default=True,
    )
    save_llm_configs([default_cfg])
    return [default_cfg]


@router.get("/llm-configs", response_model=list[LLMConfigResponse])
async def api_list_llm_configs() -> list[LLMConfigResponse]:
    """列出所有可用的 LLM 配置。

    如果当前没有配置，则自动根据全局 LLM 设置生成一个默认配置。
    """

    try:
        configs = load_llm_configs()
    except LLMConfigStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    configs = _seed_default_if_empty(configs)
    configs = _ensure_default(configs)

    return [LLMConfigResponse(**c.model_dump()) for c in configs]


@router.post("/llm-configs/{config_id}/test", response_model=LLMConfigTestResponse)
async def api_test_llm_config(config_id: str) -> LLMConfigTestResponse:
    """测试指定的 LLM 配置是否可用。

    - 优先使用该配置绑定的 LLM 提供商（provider_id）。
    - 如果未绑定提供商，则回退到默认提供商；
    - 如果仍未找到任何提供商，则回退到全局 LLM 配置 (config.json / 环境变量)。
    """

    try:
        configs = load_llm_configs()
    except LLMConfigStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    target: LLMConfig | None = None
    for cfg in configs:
        if cfg.id == config_id:
            target = cfg
            break

    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="指定的 LLM 配置不存在",
        )

    base_url: str | None = None
    api_key: str | None = None

    # 根据配置选择合适的 Provider（显式 provider_id > 默认提供商 > 全局配置）
    try:
        providers = load_llm_providers()
    except LLMProviderStorageError:
        providers = []

    if target.provider_id:
        provider = next((p for p in providers if p.id == target.provider_id), None)
        if provider is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="该配置关联的 LLM 提供商不存在",
            )
        base_url = provider.base_url
        api_key = provider.api_key
    elif providers:
        default_provider = next((p for p in providers if p.is_default), providers[0])
        base_url = default_provider.base_url
        api_key = default_provider.api_key
    else:
        # 保持 base_url / api_key 为 None，交给全局配置决定
        _ = get_config()  # 触发一次加载，便于后续 Provider 使用

    try:
        provider = OpenAIStyleLLMProvider(model=target.model, base_url=base_url, api_key=api_key)
    except Exception as exc:  # pragma: no cover - 简单封装
        return LLMConfigTestResponse(
            success=False,
            message=_mask_secrets_in_text(f"初始化 LLM Provider 失败: {exc}"),
            raw=None,
        )

    messages = [
        {"role": "system", "content": "你是一个用于连通性检测的助手。"},
        {"role": "user", "content": "请简要回复：pong"},
    ]

    # 显式限制 max_tokens，确保测试请求尽可能省 token
    test_kwargs = {"max_tokens": 16}
    if target.thinking is not None:
        test_kwargs["thinking"] = {"type": target.thinking}
    if target.reasoning_effort is not None:
        test_kwargs["reasoning_effort"] = target.reasoning_effort
    result = provider.chat(messages, **test_kwargs)

    if not result.success:
        raw = result.raw or {}
        safe_raw = _sanitize_raw_for_response(raw)
        msg = None
        if isinstance(safe_raw, dict):
            msg = safe_raw.get("error") or safe_raw.get("message")
        msg_out = str(msg) if msg else "LLM 调用失败"
        return LLMConfigTestResponse(success=False, message=_mask_secrets_in_text(msg_out), raw=safe_raw) 

    text = (result.text or "").strip() or "LLM 调用成功，但返回内容为空。"
    safe_raw = _sanitize_raw_for_response(result.raw)
    return LLMConfigTestResponse(success=True, message=_mask_secrets_in_text(text), raw=safe_raw)


@router.post("/llm-configs", response_model=LLMConfigResponse)
async def api_create_llm_config(payload: LLMConfigPayload) -> LLMConfigResponse:
    """创建新的 LLM 配置项。"""

    with llm_configs_lock():
        try:
            configs = load_llm_configs()
        except LLMConfigStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        model = payload.model.strip()
        fields_set = payload.model_fields_set
        is_flash = model == "deepseek-v4-flash"
        cfg = LLMConfig(
            id=generate_llm_config_id(),
            name=payload.name.strip(),
            provider_id=payload.provider_id.strip() if payload.provider_id else None,
            model=model,
            max_tokens=payload.max_tokens,
            description=payload.description.strip() if payload.description else None,
            response_format_json=payload.response_format_json or is_flash,
            thinking=(
                payload.thinking
                if "thinking" in fields_set
                else ("disabled" if is_flash else None)
            ),
            reasoning_effort=payload.reasoning_effort,
            fallback_model=(
                payload.fallback_model.strip() if payload.fallback_model
                else ("deepseek-v4-pro" if is_flash and "fallback_model" not in fields_set else None)
            ),
            fallback_enabled=(
                payload.fallback_enabled
                if "fallback_enabled" in fields_set
                else is_flash
            ),
            is_default=False,
        )

        configs.append(cfg)
        configs = _ensure_default(configs)

        try:
            save_llm_configs(configs)
        except LLMConfigStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    return LLMConfigResponse(**cfg.model_dump())


@router.put("/llm-configs/{config_id}", response_model=LLMConfigResponse)
async def api_update_llm_config(config_id: str, payload: LLMConfigPayload) -> LLMConfigResponse:
    """更新已有的 LLM 配置。"""

    with llm_configs_lock():
        try:
            configs = load_llm_configs()
        except LLMConfigStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        target = None
        for cfg in configs:
            if cfg.id == config_id:
                target = cfg
                break

        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定的 LLM 配置不存在",
            )

        previous_model = target.model
        next_model = payload.model.strip()
        fields_set = payload.model_fields_set
        is_flash = next_model == "deepseek-v4-flash"

        target.name = payload.name.strip()
        target.provider_id = payload.provider_id.strip() if payload.provider_id else None
        target.model = next_model
        target.max_tokens = payload.max_tokens
        target.description = payload.description.strip() if payload.description else None
        target.response_format_json = payload.response_format_json or is_flash
        if "thinking" in fields_set:
            target.thinking = payload.thinking
        elif previous_model != next_model:
            target.thinking = "disabled" if is_flash else None
        if "reasoning_effort" in fields_set:
            target.reasoning_effort = payload.reasoning_effort
        elif previous_model != next_model:
            target.reasoning_effort = None
        if "fallback_model" in fields_set:
            target.fallback_model = payload.fallback_model.strip() if payload.fallback_model else None
        elif previous_model != next_model:
            target.fallback_model = "deepseek-v4-pro" if is_flash else None
        if "fallback_enabled" in fields_set:
            target.fallback_enabled = payload.fallback_enabled
        elif previous_model != next_model:
            target.fallback_enabled = is_flash

        configs = _ensure_default(configs)

        try:
            save_llm_configs(configs)
        except LLMConfigStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    return LLMConfigResponse(**target.model_dump())


@router.delete("/llm-configs/{config_id}")
async def api_delete_llm_config(config_id: str) -> dict[str, bool]:
    """删除指定的 LLM 配置。

    如果删除后不存在任何配置，则下次访问列表时会重新生成一个默认配置。
    """

    with llm_configs_lock():
        try:
            configs = load_llm_configs()
        except LLMConfigStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        new_configs: List[LLMConfig] = []
        found = False
        for cfg in configs:
            if cfg.id == config_id:
                found = True
                continue
            new_configs.append(cfg)

        if not found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定的 LLM 配置不存在",
            )

        if new_configs:
            new_configs = _ensure_default(new_configs)
            try:
                save_llm_configs(new_configs)
            except LLMConfigStorageError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(exc),
                ) from exc
        else:
            save_llm_configs([])

    return {"success": True}


@router.post("/llm-configs/{config_id}/set-default", response_model=list[LLMConfigResponse])
async def api_set_default_llm_config(config_id: str) -> list[LLMConfigResponse]:
    """将指定配置设置为默认，并取消其他配置的默认标记。"""

    with llm_configs_lock():
        try:
            configs = load_llm_configs()
        except LLMConfigStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        found = False
        provider_id: str | None = None
        for cfg in configs:
            if cfg.id == config_id:
                cfg.is_default = True
                found = True
                provider_id = cfg.provider_id
            else:
                cfg.is_default = False

        if not found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定的 LLM 配置不存在",
            )

        try:
            save_llm_configs(configs)
        except LLMConfigStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    # 当某个 LLM 配置被设为默认且绑定了提供商时，自动同步更新默认 LLM 提供商
    if provider_id:
        try:
            providers = load_llm_providers()
        except LLMProviderStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        if providers:
            provider_found = False
            for p in providers:
                if p.id == provider_id:
                    p.is_default = True
                    provider_found = True
                else:
                    p.is_default = False

            if provider_found:
                try:
                    save_llm_providers(providers)
                except LLMProviderStorageError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(exc),
                    ) from exc

    return [LLMConfigResponse(**c.model_dump()) for c in configs]
