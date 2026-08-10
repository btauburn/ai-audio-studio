"""Move legacy plaintext API keys into the operating-system credential store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dubbing_service.app.core.config import clear_config_caches  # noqa: E402
from dubbing_service.app.core.secret_store import SecretStoreError, set_secret  # noqa: E402
from dubbing_service.app.routers.settings import (  # noqa: E402
    _load_raw_file_config,
    _save_raw_file_config,
)
from dubbing_service.app.storage.llm_providers import (  # noqa: E402
    LLMProviderStorageError,
    migrate_plaintext_llm_provider_credentials,
)


def _plaintext_counts() -> tuple[int, int]:
    providers_path = PROJECT_ROOT / "dubbing_service" / "llm_providers.json"
    provider_count = 0
    if providers_path.is_file():
        try:
            raw = json.loads(providers_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = []
        if isinstance(raw, list):
            provider_count = sum(
                1
                for item in raw
                if isinstance(item, dict) and bool(str(item.get("api_key") or "").strip())
            )

    config = _load_raw_file_config()
    platform_count = sum(
        1
        for key in ("llm_api_key", "tts_api_key")
        if bool(str(config.get(key) or "").strip())
    )
    return provider_count, platform_count


def _migrate_platform_keys() -> int:
    config = _load_raw_file_config()
    migrated = 0
    for key, account in (
        ("llm_api_key", "platform-llm-api-key"),
        ("tts_api_key", "platform-tts-api-key"),
    ):
        value = str(config.get(key) or "").strip()
        if not value:
            continue
        config[f"{key}_ref"] = set_secret(account, value)
        config.pop(key, None)
        migrated += 1
    if migrated:
        _save_raw_file_config(config)
        clear_config_caches()
    return migrated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report whether plaintext keys remain",
    )
    args = parser.parse_args()

    before_provider, before_platform = _plaintext_counts()
    if args.check:
        total = before_provider + before_platform
        print(
            f"Plaintext secret check: {total} remaining "
            f"(providers={before_provider}, platform={before_platform})"
        )
        return 1 if total else 0

    try:
        migrated_provider = migrate_plaintext_llm_provider_credentials()
        migrated_platform = _migrate_platform_keys()
    except (LLMProviderStorageError, SecretStoreError, OSError) as exc:
        print(f"Secret migration failed: {exc}", file=sys.stderr)
        return 1

    remaining_provider, remaining_platform = _plaintext_counts()
    remaining = remaining_provider + remaining_platform
    print(
        "Secret migration complete: "
        f"providers={migrated_provider}, platform={migrated_platform}, "
        f"plaintext_remaining={remaining}"
    )
    return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
