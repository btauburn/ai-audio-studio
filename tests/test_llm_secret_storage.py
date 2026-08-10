from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from dubbing_service.app.storage.llm_providers import (
    LLMProvider,
    load_llm_providers,
    migrate_plaintext_llm_provider_credentials,
    save_llm_providers,
)


def _fake_vault():
    values: dict[str, str] = {}

    def set_secret(account: str, value: str) -> str:
        values[account] = value
        return f"keyring:{account}"

    def get_secret(reference: str | None) -> str | None:
        if not reference or not reference.startswith("keyring:"):
            return None
        return values.get(reference.split(":", 1)[1])

    def delete_secret(reference: str | None) -> None:
        if reference and reference.startswith("keyring:"):
            values.pop(reference.split(":", 1)[1], None)

    return values, set_secret, get_secret, delete_secret


def test_save_provider_never_serializes_plaintext_api_key(tmp_path: Path) -> None:
    path = tmp_path / "llm_providers.json"
    vault, fake_set, fake_get, fake_delete = _fake_vault()
    provider = LLMProvider(
        id="provider1",
        name="Test",
        base_url="https://example.invalid",
        api_key="unit-test-secret",
    )

    with patch(
        "dubbing_service.app.storage.llm_providers._llm_providers_path",
        return_value=path,
    ), patch(
        "dubbing_service.app.storage.llm_providers.set_secret",
        side_effect=fake_set,
    ), patch(
        "dubbing_service.app.storage.llm_providers.get_secret",
        side_effect=fake_get,
    ), patch(
        "dubbing_service.app.storage.llm_providers.delete_secret",
        side_effect=fake_delete,
    ):
        save_llm_providers([provider])
        raw_text = path.read_text(encoding="utf-8")
        raw = json.loads(raw_text)
        loaded = load_llm_providers()

    assert "unit-test-secret" not in raw_text
    assert "api_key" not in raw[0]
    assert raw[0]["credential_ref"] == "keyring:llm-provider-provider1"
    assert vault["llm-provider-provider1"] == "unit-test-secret"
    assert loaded[0].api_key == "unit-test-secret"


def test_legacy_plaintext_provider_migration_is_lossless(tmp_path: Path) -> None:
    path = tmp_path / "llm_providers.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "legacy1",
                    "name": "Legacy",
                    "base_url": "https://example.invalid",
                    "api_key": "legacy-unit-secret",
                    "is_default": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    _vault, fake_set, fake_get, fake_delete = _fake_vault()

    with patch(
        "dubbing_service.app.storage.llm_providers._llm_providers_path",
        return_value=path,
    ), patch(
        "dubbing_service.app.storage.llm_providers.set_secret",
        side_effect=fake_set,
    ), patch(
        "dubbing_service.app.storage.llm_providers.get_secret",
        side_effect=fake_get,
    ), patch(
        "dubbing_service.app.storage.llm_providers.delete_secret",
        side_effect=fake_delete,
    ):
        assert migrate_plaintext_llm_provider_credentials() == 1
        loaded = load_llm_providers()

    raw_text = path.read_text(encoding="utf-8")
    assert "legacy-unit-secret" not in raw_text
    assert loaded[0].api_key == "legacy-unit-secret"
    assert loaded[0].is_default is True


def test_environment_secret_reference_is_not_rewritten_to_keyring(tmp_path: Path) -> None:
    path = tmp_path / "llm_providers.json"
    provider = LLMProvider(
        id="env1",
        name="Environment",
        base_url="https://example.invalid",
        api_key="runtime-value",
        credential_ref="env:TEST_LLM_KEY",
    )

    with patch(
        "dubbing_service.app.storage.llm_providers._llm_providers_path",
        return_value=path,
    ), patch("dubbing_service.app.storage.llm_providers.set_secret") as set_secret:
        save_llm_providers([provider])

    set_secret.assert_not_called()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw[0]["credential_ref"] == "env:TEST_LLM_KEY"
    assert "api_key" not in raw[0]
