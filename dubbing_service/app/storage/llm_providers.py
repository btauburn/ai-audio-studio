"""File-based helpers for managing LLM providers (base URL / API key)."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Generator, List

from pydantic import BaseModel, Field, ValidationError

from dubbing_service.app.core.secret_store import (
    SecretReference,
    SecretStoreError,
    delete_secret,
    get_secret,
    provider_account,
    set_secret,
)


class LLMProviderStorageError(Exception):
    """与 LLM 提供商存储相关的异常。"""


_LLM_PROVIDERS_LOCK = Lock()


@contextmanager
def llm_providers_lock() -> Generator[None, None, None]:
    """LLM 提供商读写锁的上下文管理器，保护 llm_providers.json 的读-改-写操作。"""
    _LLM_PROVIDERS_LOCK.acquire()
    try:
        yield
    finally:
        _LLM_PROVIDERS_LOCK.release()


class LLMProvider(BaseModel):
    id: str
    name: str
    base_url: str
    # ``api_key`` exists only in memory.  New writes persist a reference to the
    # OS credential store, never the plaintext value.
    api_key: str | None = Field(default=None, exclude=True, repr=False)
    credential_ref: str | None = None
    credential_error: str | None = Field(default=None, exclude=True, repr=False)
    description: str | None = None
    is_default: bool = False


def _llm_providers_path() -> Path:
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "llm_providers.json"


def load_llm_providers() -> List[LLMProvider]:
    path = _llm_providers_path()
    if not path.is_file():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise LLMProviderStorageError(
            f"llm_providers.json 内容损坏，无法解析 JSON: {exc}"
        ) from exc

    if not isinstance(raw, list):
        raise LLMProviderStorageError("llm_providers.json 内容格式错误：应为列表")

    providers: List[LLMProvider] = []
    for item in raw:
        try:
            provider = LLMProvider(**item)
        except ValidationError as exc:
            raise LLMProviderStorageError("解析 LLM 提供商配置失败") from exc
        else:
            if provider.credential_ref:
                try:
                    provider.api_key = get_secret(provider.credential_ref)
                except SecretStoreError as exc:
                    provider.api_key = None
                    provider.credential_error = str(exc)
            providers.append(provider)

    return providers


def save_llm_providers(providers: List[LLMProvider]) -> None:
    path = _llm_providers_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Promote every in-memory plaintext key into the OS credential store first.
    # Keep enough state to restore the credential if the atomic JSON promotion
    # fails afterwards.
    credential_changes: list[tuple[LLMProvider, str | None, str | None, str]] = []
    try:
        for provider in providers:
            if not provider.api_key:
                continue
            parsed = SecretReference.parse(provider.credential_ref)
            if parsed is not None and parsed.kind == "env":
                continue

            old_ref = provider.credential_ref
            old_secret: str | None = None
            if parsed is not None and parsed.kind == "keyring":
                old_secret = get_secret(old_ref)
            new_ref = set_secret(provider_account(provider.id), provider.api_key)
            provider.credential_ref = new_ref
            provider.credential_error = None
            credential_changes.append((provider, old_ref, old_secret, new_ref))
    except SecretStoreError as exc:
        raise LLMProviderStorageError(str(exc)) from exc

    data = [
        p.model_dump(
            mode="json",
            exclude={"api_key", "credential_error"},
        )
        for p in providers
    ]

    tmp_path = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    promoted = False
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
        promoted = True
    except Exception as exc:
        for provider, old_ref, old_secret, new_ref in reversed(credential_changes):
            try:
                if old_ref and old_secret:
                    old_account = SecretReference.parse(old_ref)
                    if old_account is not None and old_account.kind == "keyring":
                        set_secret(old_account.account, old_secret)
                    if new_ref != old_ref:
                        delete_secret(new_ref)
                else:
                    delete_secret(new_ref)
            except SecretStoreError:
                pass
            provider.credential_ref = old_ref
        raise LLMProviderStorageError(f"保存 LLM 提供商配置失败: {exc}") from exc
    finally:
        if tmp_path.is_file():
            try:
                tmp_path.unlink()
            except Exception:
                pass

    if not promoted:  # pragma: no cover - defensive; exception path returns above
        raise LLMProviderStorageError("保存 LLM 提供商配置失败")


def migrate_plaintext_llm_provider_credentials() -> int:
    """Move legacy JSON ``api_key`` values into the OS credential store.

    The returned count is safe to log; secret values are never returned.
    """

    path = _llm_providers_path()
    if not path.is_file():
        return 0
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMProviderStorageError("无法读取 LLM 提供商配置以迁移凭据") from exc
    if not isinstance(raw, list):
        raise LLMProviderStorageError("llm_providers.json 内容格式错误：应为列表")
    plaintext_count = sum(
        1
        for item in raw
        if isinstance(item, dict) and bool(str(item.get("api_key") or "").strip())
    )
    if not plaintext_count:
        return 0
    providers = load_llm_providers()
    save_llm_providers(providers)
    return plaintext_count


def generate_llm_provider_id() -> str:
    return uuid.uuid4().hex[:8]
