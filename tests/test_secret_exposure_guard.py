from __future__ import annotations

from dubbing_service.app.core.helpers import (
    expose_secrets,
    mask_secret_value,
    sanitize_raw_for_response,
)


def test_legacy_expose_secrets_environment_cannot_unmask(monkeypatch) -> None:
    monkeypatch.setenv("DUBBING_EXPOSE_SECRETS", "1")

    assert expose_secrets() is False
    assert mask_secret_value("unit-test-key") == "********"
    assert sanitize_raw_for_response({"api_key": "unit-test-key"}) == {
        "api_key": "********"
    }
