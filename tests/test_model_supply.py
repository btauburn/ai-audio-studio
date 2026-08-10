from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from scripts.model_preflight import (
    ManifestError,
    load_manifest,
    resolve_within,
    verify_manifest,
)
from scripts.model_provision import ProvisionError, provision_artifact


def _manifest_for(payload: bytes, *, recovery_path: str | None = None) -> dict:
    artifact = {
        "group": "runtime",
        "path": "models/test/model.bin",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source": "model",
        "source_path": "model.bin",
        "source_revision": "pinned-revision",
        "source_file_revision": "a" * 40,
    }
    if recovery_path:
        artifact["recovery"] = [
            {"provider": "project_cache", "path": recovery_path}
        ]
    return {
        "schema_version": 1,
        "name": "Test model",
        "source": "unit test",
        "revision": "pinned-revision",
        "model_id": "test_model",
        "sources": {
            "model": {
                "provider": "modelscope",
                "model_id": "example/test-model",
            }
        },
        "files": [artifact],
    }


def test_preflight_reports_hash_mismatch(tmp_path: Path) -> None:
    expected = b"expected model bytes"
    manifest = _manifest_for(expected)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target = tmp_path / "models" / "test" / "model.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * len(expected))

    _, statuses = verify_manifest(manifest_path, project_root=tmp_path)

    assert len(statuses) == 1
    assert statuses[0].state == "hash_mismatch"


def test_provision_restores_verified_cache_and_quarantines_target(tmp_path: Path) -> None:
    expected = b"verified local model payload"
    cache = tmp_path / "models" / "cache" / "blob.bin"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(expected)
    manifest = _manifest_for(expected, recovery_path="models/cache/blob.bin")
    artifact = manifest["files"][0]
    target = tmp_path / artifact["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupt")

    state, quarantine = provision_artifact(
        manifest,
        artifact,
        project_root=tmp_path,
    )

    assert state == "restored_from_project_cache"
    assert target.read_bytes() == expected
    assert quarantine is not None
    assert quarantine.read_bytes() == b"corrupt"
    assert cache.read_bytes() == expected


def test_provision_requires_explicit_network_permission(tmp_path: Path) -> None:
    payload = b"missing"
    manifest = _manifest_for(payload)

    with pytest.raises(ProvisionError, match="allow-modelscope-download"):
        provision_artifact(
            manifest,
            manifest["files"][0],
            project_root=tmp_path,
        )


def test_provision_uses_branch_after_verifying_pinned_file_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = b"verified downloaded payload"
    manifest = _manifest_for(expected)
    artifact = manifest["files"][0]
    artifact["source_revision"] = "master"
    artifact["source_file_revision"] = "c" * 40
    downloaded = tmp_path / "downloaded.bin"
    downloaded.write_bytes(expected)
    calls: dict[str, str] = {}

    class FakeHubApi:
        def get_model_files(
            self,
            model_id: str,
            *,
            revision: str,
            recursive: bool,
        ) -> list[dict[str, object]]:
            calls["metadata_model_id"] = model_id
            calls["metadata_revision"] = revision
            assert recursive is True
            return [
                {
                    "Path": artifact["source_path"],
                    "Revision": artifact["source_file_revision"],
                    "Size": artifact["size"],
                    "Sha256": artifact["sha256"],
                }
            ]

    def fake_model_file_download(**kwargs: object) -> str:
        calls["download_model_id"] = str(kwargs["model_id"])
        calls["download_revision"] = str(kwargs["revision"])
        return str(downloaded)

    modelscope_api = importlib.import_module("modelscope.hub.api")
    modelscope_download = importlib.import_module("modelscope.hub.file_download")
    monkeypatch.setattr(modelscope_api, "HubApi", FakeHubApi)
    monkeypatch.setattr(modelscope_download, "model_file_download", fake_model_file_download)

    state, quarantine = provision_artifact(
        manifest,
        artifact,
        project_root=tmp_path,
        allow_modelscope_download=True,
    )

    assert state == "restored_from_modelscope"
    assert quarantine is None
    assert calls == {
        "metadata_model_id": "example/test-model",
        "metadata_revision": "master",
        "download_model_id": "example/test-model",
        "download_revision": "master",
    }
    assert (tmp_path / artifact["path"]).read_bytes() == expected


def test_failed_promotion_restores_quarantined_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = b"verified replacement"
    original = b"original corrupt target"
    cache = tmp_path / "models" / "cache" / "blob.bin"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(expected)
    manifest = _manifest_for(expected, recovery_path="models/cache/blob.bin")
    artifact = manifest["files"][0]
    target = tmp_path / artifact["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(original)

    def fail_promote(source: str, destination: str) -> None:
        raise OSError(f"injected replace failure: {source} -> {destination}")

    monkeypatch.setattr("scripts.model_provision.os.replace", fail_promote)

    with pytest.raises(ProvisionError, match="original target restored"):
        provision_artifact(
            manifest,
            artifact,
            project_root=tmp_path,
        )

    assert target.read_bytes() == original
    assert cache.read_bytes() == expected
    assert not list(target.parent.glob(f".{target.name}.part-*"))


def test_manifest_rejects_paths_outside_project(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="escapes"):
        resolve_within(tmp_path, "../outside.bin", "artifact path")


def test_load_manifest_requires_doctor_compatible_fields(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"schema_version": 1, "files": []}),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="manifest name"):
        load_manifest(path)


def test_modelscope_manifest_rejects_file_revision_as_download_revision(
    tmp_path: Path,
) -> None:
    manifest = _manifest_for(b"payload")
    manifest["files"][0]["source_revision"] = "b" * 40
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match="downloadable branch/tag"):
        load_manifest(path)


def test_modelscope_manifest_requires_pinned_file_revision(tmp_path: Path) -> None:
    manifest = _manifest_for(b"payload")
    manifest["files"][0].pop("source_file_revision")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match="source_file_revision"):
        load_manifest(path)


def test_all_project_modelscope_manifests_separate_repo_and_file_revisions() -> None:
    manifest_root = Path(__file__).resolve().parents[1] / "models" / "manifests"

    for manifest_path in sorted(manifest_root.glob("*.json")):
        load_manifest(manifest_path)
