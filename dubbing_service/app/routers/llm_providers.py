"""LLM provider (service) management APIs."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from dubbing_service.app.core.config import get_config
from dubbing_service.app.core.helpers import mask_secret_value as _mask_api_key
from dubbing_service.app.core.secret_store import SecretStoreError, delete_secret
from dubbing_service.app.storage.llm_providers import (
    LLMProvider,
    LLMProviderStorageError,
    generate_llm_provider_id,
    llm_providers_lock,
    load_llm_providers,
    save_llm_providers,
)


router = APIRouter()


def _to_response(provider: LLMProvider) -> LLMProviderResponse:
    resp = LLMProviderResponse(
        **provider.model_dump(exclude={"api_key", "credential_ref", "credential_error"}),
        api_key=provider.api_key,
    )
    resp.api_key = _mask_api_key(resp.api_key)
    return resp


class LLMProviderPayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="展示名称")
    base_url: str = Field(..., min_length=1, max_length=200, description="LLM 服务 Base URL")
    api_key: str | None = Field(
        default=None,
        max_length=512,
        description="可选：LLM API Key，如果服务不需要鉴权可以留空",
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="可选：说明该提供商的用途，例如 DeepSeek 正式环境等",
    )


class LLMProviderResponse(BaseModel):
    id: str
    name: str
    base_url: str
    api_key: str | None = None
    description: str | None = None
    is_default: bool = False


def _ensure_default(providers: List[LLMProvider]) -> List[LLMProvider]:
    if not providers:
        return providers

    if not any(p.is_default for p in providers):
        providers[0].is_default = True

    return providers


def _seed_default_if_empty(providers: List[LLMProvider]) -> List[LLMProvider]:
    if providers:
        return providers

    cfg = get_config()
    if not cfg.llm_base_url or not cfg.llm_api_key:
        return providers

    default_provider = LLMProvider(
        id=generate_llm_provider_id(),
        name="默认 LLM 提供商",
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
        description="从全局 LLM 配置推断出的默认提供商。",
        is_default=True,
    )
    save_llm_providers([default_provider])
    return [default_provider]


@router.get("/llm-providers", response_model=list[LLMProviderResponse])
async def api_list_llm_providers() -> list[LLMProviderResponse]:
    try:
        providers = load_llm_providers()
    except LLMProviderStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    providers = _seed_default_if_empty(providers)
    providers = _ensure_default(providers)

    return [_to_response(p) for p in providers]


@router.post("/llm-providers", response_model=LLMProviderResponse)
async def api_create_llm_provider(payload: LLMProviderPayload) -> LLMProviderResponse:
    with llm_providers_lock():
        try:
            providers = load_llm_providers()
        except LLMProviderStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        provider = LLMProvider(
            id=generate_llm_provider_id(),
            name=payload.name.strip(),
            base_url=payload.base_url.strip(),
            api_key=(
                None
                if not payload.api_key
                else (
                    None
                    if payload.api_key.strip() == "********"
                    else payload.api_key.strip()
                )
            ),
            description=payload.description.strip() if payload.description else None,
            is_default=False,
        )

        providers.append(provider)
        providers = _ensure_default(providers)

        try:
            save_llm_providers(providers)
        except LLMProviderStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    return _to_response(provider)


@router.put("/llm-providers/{provider_id}", response_model=LLMProviderResponse)
async def api_update_llm_provider(provider_id: str, payload: LLMProviderPayload) -> LLMProviderResponse:
    credential_to_delete: str | None = None
    with llm_providers_lock():
        try:
            providers = load_llm_providers()
        except LLMProviderStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        target = None
        for p in providers:
            if p.id == provider_id:
                target = p
                break

        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定的 LLM 提供商不存在",
            )

        target.name = payload.name.strip()
        target.base_url = payload.base_url.strip()

        if payload.api_key is None:
            credential_to_delete = target.credential_ref
            target.api_key = None
            target.credential_ref = None
        else:
            s = payload.api_key.strip()
            if s != "********":
                if s:
                    target.api_key = s
                else:
                    credential_to_delete = target.credential_ref
                    target.api_key = None
                    target.credential_ref = None
        target.description = payload.description.strip() if payload.description else None

        providers = _ensure_default(providers)

        try:
            save_llm_providers(providers)
        except LLMProviderStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    if credential_to_delete:
        try:
            delete_secret(credential_to_delete)
        except SecretStoreError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="提供商已更新，但旧系统凭据清理失败",
            ) from exc

    return _to_response(target)


@router.delete("/llm-providers/{provider_id}")
async def api_delete_llm_provider(provider_id: str) -> dict[str, bool]:
    credential_to_delete: str | None = None
    with llm_providers_lock():
        try:
            providers = load_llm_providers()
        except LLMProviderStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        new_providers: List[LLMProvider] = []
        found = False
        for p in providers:
            if p.id == provider_id:
                found = True
                credential_to_delete = p.credential_ref
                continue
            new_providers.append(p)

        if not found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定的 LLM 提供商不存在",
            )

        if new_providers:
            new_providers = _ensure_default(new_providers)
            try:
                save_llm_providers(new_providers)
            except LLMProviderStorageError as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(exc),
                ) from exc
        else:
            save_llm_providers([])

    if credential_to_delete:
        try:
            delete_secret(credential_to_delete)
        except SecretStoreError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="提供商已删除，但系统凭据清理失败",
            ) from exc

    return {"success": True}


@router.post("/llm-providers/{provider_id}/set-default", response_model=list[LLMProviderResponse])
async def api_set_default_llm_provider(provider_id: str) -> list[LLMProviderResponse]:
    with llm_providers_lock():
        try:
            providers = load_llm_providers()
        except LLMProviderStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        found = False
        for p in providers:
            if p.id == provider_id:
                p.is_default = True
                found = True
            else:
                p.is_default = False

        if not found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定的 LLM 提供商不存在",
            )

        try:
            save_llm_providers(providers)
        except LLMProviderStorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

        providers = _ensure_default(providers)

    return [_to_response(p) for p in providers]
