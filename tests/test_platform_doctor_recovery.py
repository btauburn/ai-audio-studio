from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import platform_doctor


def _write_manifest(root: Path, *, include_recovery_file: bool) -> Path:
    payload = b"verified-local-only-model"
    target = root / "models" / "runtime" / "model.bin"
    recovery = root / "models" / "recovery_blobs" / "model.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    if include_recovery_file:
        recovery.parent.mkdir(parents=True)
        recovery.write_bytes(payload)
    manifest = root / "models" / "manifests" / "test.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "sources": {"local_only": {"provider": "provenance_only"}},
                "files": [
                    {
                        "path": "models/runtime/model.bin",
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "source": "local_only",
                        "recovery": [
                            {
                                "provider": "project_cache",
                                "path": "models/recovery_blobs/model.bin",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_doctor_verifies_non_modelscope_recovery_copy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(platform_doctor, "PROJECT_ROOT", tmp_path)
    manifest = _write_manifest(tmp_path, include_recovery_file=True)

    results = platform_doctor._check_manifest(manifest, full_hash=True)

    assert all(result.ok for result in results)
    assert any(result.label.startswith("recovery:") for result in results)


def test_doctor_fails_when_non_modelscope_recovery_copy_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(platform_doctor, "PROJECT_ROOT", tmp_path)
    manifest = _write_manifest(tmp_path, include_recovery_file=False)

    results = platform_doctor._check_manifest(manifest, full_hash=False)

    assert any(not result.ok and result.label.startswith("recovery:") for result in results)


def test_doctor_plaintext_credential_check_reports_counts_not_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(platform_doctor, "PROJECT_ROOT", tmp_path)
    config_root = tmp_path / "dubbing_service"
    config_root.mkdir(parents=True)
    sentinel = "unit-test-secret-that-must-not-be-printed"
    (config_root / "config.json").write_text(
        json.dumps(
            {
                "llm_api_key": sentinel,
                "tts_api_key": "",
                "llm_api_key_ref": "keyring://index-tts/platform-llm-api-key",
            }
        ),
        encoding="utf-8",
    )
    (config_root / "llm_providers.json").write_text(
        json.dumps(
            [
                {"id": "one", "api_key": sentinel},
                {"id": "two", "credential_ref": "env://DEEPSEEK_API_KEY"},
            ]
        ),
        encoding="utf-8",
    )

    result = platform_doctor._check_plaintext_api_credentials()

    assert not result.ok
    assert result.detail == (
        "config fields=1; provider records=1; migrate to credential references"
    )
    assert sentinel not in result.detail


def test_doctor_plaintext_credential_check_allows_reference_only_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(platform_doctor, "PROJECT_ROOT", tmp_path)
    config_root = tmp_path / "dubbing_service"
    config_root.mkdir(parents=True)
    (config_root / "config.json").write_text(
        json.dumps(
            {
                "llm_api_key_ref": "keyring://index-tts/platform-llm-api-key",
                "tts_api_key_ref": "env://DUBBING_TTS_API_KEY",
            }
        ),
        encoding="utf-8",
    )
    (config_root / "llm_providers.json").write_text(
        json.dumps(
            [
                {"id": "one", "credential_ref": "keyring://index-tts/provider-one"},
            ]
        ),
        encoding="utf-8",
    )

    result = platform_doctor._check_plaintext_api_credentials()

    assert result.ok
    assert result.detail == "config fields=0; provider records=0"


def test_doctor_plaintext_credential_check_allows_missing_provider_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(platform_doctor, "PROJECT_ROOT", tmp_path)
    config_root = tmp_path / "dubbing_service"
    config_root.mkdir(parents=True)
    (config_root / "config.json").write_text(
        json.dumps({"llm_api_key": "", "tts_api_key": ""}),
        encoding="utf-8",
    )

    result = platform_doctor._check_plaintext_api_credentials()

    assert result.ok
    assert result.detail == "config fields=0; provider records=0"
