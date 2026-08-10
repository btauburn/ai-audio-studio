"""Small wrapper around the operating-system credential store.

Secrets must not be serialized into project JSON.  On Windows this uses the
Credential Locker backend supplied by ``keyring``; macOS uses Keychain and
desktop Linux uses the available Secret Service/KWallet backend.
"""

from __future__ import annotations

from dataclasses import dataclass


SERVICE_NAME = "IndexTTS Dubbing Platform"


class SecretStoreError(RuntimeError):
    """Raised when the operating-system credential store is unavailable."""


@dataclass(frozen=True, slots=True)
class SecretReference:
    kind: str
    account: str

    def serialize(self) -> str:
        return f"{self.kind}:{self.account}"

    @classmethod
    def parse(cls, value: str | None) -> "SecretReference | None":
        raw = str(value or "").strip()
        if not raw or ":" not in raw:
            return None
        kind, account = raw.split(":", 1)
        if kind not in {"keyring", "env"} or not account:
            return None
        return cls(kind=kind, account=account)


def provider_account(provider_id: str) -> str:
    return f"llm-provider-{provider_id}"


def _keyring_module():
    try:
        import keyring  # type: ignore
        from keyring.errors import KeyringError  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency/runtime specific
        raise SecretStoreError("系统凭据存储组件不可用") from exc

    try:
        backend = keyring.get_keyring()
        priority = float(getattr(backend, "priority", 0) or 0)
    except Exception as exc:  # pragma: no cover - backend specific
        raise SecretStoreError("无法初始化系统凭据存储") from exc
    if priority <= 0:
        raise SecretStoreError("当前系统没有可用的安全凭据存储后端")
    return keyring, KeyringError


def get_secret(reference: str | None) -> str | None:
    ref = SecretReference.parse(reference)
    if ref is None:
        return None
    if ref.kind == "env":
        import os

        return os.getenv(ref.account) or None

    keyring, keyring_error = _keyring_module()
    try:
        value = keyring.get_password(SERVICE_NAME, ref.account)
    except keyring_error as exc:  # pragma: no cover - backend specific
        raise SecretStoreError("读取系统凭据失败") from exc
    return str(value) if value else None


def set_secret(account: str, value: str) -> str:
    secret = str(value or "").strip()
    if not account or not secret:
        raise SecretStoreError("凭据账号和密钥不能为空")
    keyring, keyring_error = _keyring_module()
    try:
        keyring.set_password(SERVICE_NAME, account, secret)
    except keyring_error as exc:  # pragma: no cover - backend specific
        raise SecretStoreError("写入系统凭据失败") from exc
    return SecretReference("keyring", account).serialize()


def delete_secret(reference: str | None) -> None:
    ref = SecretReference.parse(reference)
    if ref is None or ref.kind != "keyring":
        return
    keyring, keyring_error = _keyring_module()
    try:
        keyring.delete_password(SERVICE_NAME, ref.account)
    except keyring_error as exc:
        # A missing credential is already the desired state.  Backends expose
        # different concrete exceptions, so verify by reading once.
        try:
            if keyring.get_password(SERVICE_NAME, ref.account) is None:
                return
        except Exception:
            pass
        raise SecretStoreError("删除系统凭据失败") from exc
