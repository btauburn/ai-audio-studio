from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.model_preflight import (
    ManifestError,
    has_license_acceptance,
    has_passed_quality_gate,
    license_receipt_path,
    load_manifest,
    record_license_acceptance,
    resolve_within,
    runtime_code_revision_status,
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


def _huggingface_manifest(payload: bytes) -> dict:
    revision = "d" * 40
    confirmation_text = "I accept both pinned test licenses and confirm registration."
    confirmation_hash = hashlib.sha256(confirmation_text.encode()).hexdigest()
    return {
        "schema_version": 2,
        "name": "Gated test model",
        "source": "unit test",
        "revision": revision,
        "model_id": "gated_test_model",
        "sources": {
            "model": {
                "provider": "huggingface",
                "repo_id": "example/gated-model",
                "revision": revision,
                "gated": True,
                "token_env": "HF_TOKEN",
            }
        },
        "license_policy": {
            "requires_explicit_acceptance": True,
            "confirmation_text": confirmation_text,
            "confirmation_sha256": confirmation_hash,
            "receipt_path": f"models/.license_acceptance/gated/{confirmation_hash}.json",
            "documents": [
                {
                    "name": "License one",
                    "url": "https://example.invalid/one",
                    "path": "models/gated/LICENSE.md",
                    "sha256": hashlib.sha256(b"one").hexdigest(),
                },
                {
                    "name": "License two",
                    "url": "https://example.invalid/two",
                    "path": "models/gated/LICENSE_TWO.md",
                    "sha256": hashlib.sha256(b"two").hexdigest(),
                },
            ],
        },
        "quality_gate": {
            "path": "models/gated/gate-fixture.json",
            "categories_total": 12,
            "cases_total": 24,
            "minimum_categories_passed": 10,
            "technical_rounds": 5,
            "technical_runs": 30,
        },
        "files": [
            {
                "group": "runtime",
                "path": "models/gated/model.bin",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source": "model",
                "source_path": "model.bin",
                "source_revision": revision,
                "source_blob_id": "1" * 40,
            },
            {
                "group": "license",
                "path": "models/gated/LICENSE.md",
                "size": 3,
                "sha256": hashlib.sha256(b"one").hexdigest(),
                "source": "model",
                "source_path": "LICENSE.md",
                "source_revision": revision,
                "source_blob_id": "2" * 40,
            },
            {
                "group": "license",
                "path": "models/gated/LICENSE_TWO.md",
                "size": 3,
                "sha256": hashlib.sha256(b"two").hexdigest(),
                "source": "model",
                "source_path": "LICENSE_TWO.md",
                "source_revision": revision,
                "source_blob_id": "3" * 40,
            },
        ],
    }


def test_stable_audio_manifest_is_pinned_to_expected_project_root() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "manifests"
        / "stable_audio_3_small_sfx.json"
    )
    manifest = load_manifest(manifest_path)

    assert manifest["revision"] == "ae12755283df9d62ca39a9b050a39a0b607b8c20"
    assert manifest["code_revision"] == "a0b57f5483c4588f827f3552b7d5c6ca2a9687be"
    assert manifest["code_source"] == {
        "distribution": "stable-audio-3",
        "url": "https://github.com/Stability-AI/stable-audio-3.git",
    }
    assert manifest["runtime_policy"]["model_root"] == "models/stable_audio_3/small-sfx"
    assert sum(item["size"] for item in manifest["files"]) == 3_492_094_029
    assert all(item["source_revision"] == manifest["revision"] for item in manifest["files"])


def test_huggingface_provision_requires_explicit_permission_and_hash(tmp_path: Path) -> None:
    manifest = _huggingface_manifest(b"payload")
    artifact = manifest["files"][0]

    with pytest.raises(ProvisionError, match="allow-huggingface-gated-download"):
        provision_artifact(manifest, artifact, project_root=tmp_path)

    with pytest.raises(ProvisionError, match="accept-license-hash"):
        provision_artifact(
            manifest,
            artifact,
            project_root=tmp_path,
            allow_huggingface_gated_download=True,
        )


def test_huggingface_token_is_read_only_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"verified gated payload"
    manifest = _huggingface_manifest(payload)
    artifact = manifest["files"][0]
    confirmation_hash = manifest["license_policy"]["confirmation_sha256"]
    downloaded = tmp_path / "downloaded.bin"
    downloaded.write_bytes(payload)
    calls: dict[str, object] = {}

    class FakeHfApi:
        def __init__(self, *, token: str) -> None:
            calls["api_token"] = token

        def model_info(self, **kwargs: object) -> SimpleNamespace:
            calls["metadata"] = kwargs
            return SimpleNamespace(
                sha=manifest["revision"],
                siblings=[
                    SimpleNamespace(
                        rfilename=artifact["source_path"],
                        size=artifact["size"],
                        blob_id=artifact["source_blob_id"],
                    )
                ],
            )

    def fake_hf_hub_download(**kwargs: object) -> str:
        calls["download"] = kwargs
        return str(downloaded)

    hub = importlib.import_module("huggingface_hub")
    monkeypatch.setattr(hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ProvisionError, match="HF_TOKEN is not set"):
        provision_artifact(
            manifest,
            artifact,
            project_root=tmp_path,
            allow_huggingface_gated_download=True,
            accepted_license_hash=confirmation_hash,
        )

    monkeypatch.setenv("HF_TOKEN", "environment-only-test-token")
    state, _ = provision_artifact(
        manifest,
        artifact,
        project_root=tmp_path,
        allow_huggingface_gated_download=True,
        accepted_license_hash=confirmation_hash,
    )

    assert state == "restored_from_huggingface"
    assert calls["api_token"] == "environment-only-test-token"
    assert calls["download"]["token"] == "environment-only-test-token"  # type: ignore[index]
    assert calls["download"]["revision"] == manifest["revision"]  # type: ignore[index]
    assert (tmp_path / artifact["path"]).read_bytes() == payload


def test_license_receipt_contains_required_confirmations_but_no_identity(
    tmp_path: Path,
) -> None:
    manifest = _huggingface_manifest(b"payload")
    receipt_path = record_license_acceptance(
        manifest,
        license_tier="community",
        stability_registered=True,
        stability_terms_accepted=True,
        gemma_terms_accepted=True,
        project_root=tmp_path,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert has_license_acceptance(manifest, tmp_path)
    assert receipt_path == license_receipt_path(manifest, tmp_path)
    assert receipt["license_tier"] == "community"
    assert receipt["stability_registered"] is True
    assert receipt["stability_terms_accepted"] is True
    assert receipt["gemma_terms_accepted"] is True
    assert "token" not in json.dumps(receipt).lower()
    assert "account" not in json.dumps(receipt).lower()


def test_quality_gate_must_match_current_model_hashes_and_all_technical_runs(
    tmp_path: Path,
) -> None:
    manifest = _huggingface_manifest(b"payload")
    manifest["code_revision"] = "a" * 40
    manifest["code_source"] = {
        "distribution": "stable-audio-3",
        "url": "https://github.com/Stability-AI/stable-audio-3.git",
    }
    gate_path = tmp_path / manifest["quality_gate"]["path"]
    gate_path.parent.mkdir(parents=True)
    model_files = [
        {
            "path": item["path"],
            "size": item["size"],
            "sha256": item["sha256"],
        }
        for item in manifest["files"]
        if item["group"] == "runtime"
    ]
    gate = {
        "schema_version": 1,
        "passed": True,
        "model_id": manifest["model_id"],
        "model_revision": manifest["revision"],
        "source_revisions": {"model": manifest["revision"]},
        "code_revision": manifest["code_revision"],
        "code_distribution": manifest["code_source"]["distribution"],
        "code_source_url": manifest["code_source"]["url"],
        "model_files": model_files,
        "quality": {
            "categories_total": 12,
            "cases_total": 24,
            "minimum_categories_passed": 10,
            "categories_passed": 10,
            "categories": [
                {
                    "category": f"c{index}",
                    "cases_total": 2,
                    "cases_passed": 2 if index < 10 else 0,
                    "passed": index < 10,
                }
                for index in range(12)
            ],
        },
        "technical": {
            "rounds_total": 5,
            "runs_total": 30,
            "runs_passed": 30,
            "handoffs_total": 5,
            "handoffs_passed": 5,
            "rounds": [
                {"round": index, "runs_total": 6, "runs_passed": 6}
                for index in range(1, 6)
            ],
            "handoffs": [
                {
                    "round": index,
                    "passed": True,
                    "memory_release_passed": True,
                    "tts_unload": {"success": True},
                    "sfx_load": {"success": True},
                    "sfx_unload": {"success": True},
                    "tts_load": {"success": True},
                }
                for index in range(1, 6)
            ],
            "warm_latency": {
                "runs_total": 25,
                "p95_passed": True,
                "p95_budget_ratio": 0.8,
            },
        },
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    assert has_passed_quality_gate(manifest, tmp_path)
    gate["code_revision"] = "b" * 40
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    assert not has_passed_quality_gate(manifest, tmp_path)
    gate["code_revision"] = manifest["code_revision"]
    gate["code_source_url"] = "https://github.com/example/polluted.git"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    assert not has_passed_quality_gate(manifest, tmp_path)
    gate["code_source_url"] = manifest["code_source"]["url"]
    gate["model_files"][0]["sha256"] = "0" * 64
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    assert not has_passed_quality_gate(manifest, tmp_path)


def _write_fake_stableaudio_install(
    root: Path,
    *,
    revision: str,
    url: str = "https://github.com/Stability-AI/stable-audio-3.git",
) -> Path:
    runtime_python = (
        root
        / "environments"
        / "stableaudio"
        / ".venv"
        / "Scripts"
        / "python.exe"
    )
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.write_bytes(b"unit-test-python")
    metadata = (
        runtime_python.parent.parent
        / "Lib"
        / "site-packages"
        / "stable_audio_3-0.1.0.dist-info"
    )
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "direct_url.json").write_text(
        json.dumps(
            {
                "url": url,
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": revision,
                    "requested_revision": revision,
                },
            }
        ),
        encoding="utf-8",
    )
    return runtime_python


def test_runtime_code_revision_reads_and_strictly_checks_direct_url(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    manifest = {
        "code_revision": revision,
        "code_source": {
            "distribution": "stable-audio-3",
            "url": "https://github.com/Stability-AI/stable-audio-3.git",
        },
    }
    runtime_python = _write_fake_stableaudio_install(tmp_path, revision=revision)

    assert runtime_code_revision_status(manifest, runtime_python) == (
        True,
        f"verified:{revision}",
    )

    direct_url = next(
        runtime_python.parent.parent.glob(
            "Lib/site-packages/stable_audio_3-*.dist-info/direct_url.json"
        )
    )
    payload = json.loads(direct_url.read_text(encoding="utf-8"))
    payload["url"] = "https://github.com/example/polluted.git"
    direct_url.write_text(json.dumps(payload), encoding="utf-8")
    assert runtime_code_revision_status(manifest, runtime_python) == (
        False,
        "code_source_url_mismatch",
    )


def test_runtime_code_revision_rejects_missing_or_duplicate_metadata(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    manifest = {
        "code_revision": revision,
        "code_source": {
            "distribution": "stable-audio-3",
            "url": "https://github.com/Stability-AI/stable-audio-3.git",
        },
    }
    runtime_python = _write_fake_stableaudio_install(tmp_path, revision=revision)
    direct_url = next(
        runtime_python.parent.parent.glob(
            "Lib/site-packages/stable_audio_3-*.dist-info/direct_url.json"
        )
    )
    direct_url.unlink()
    assert runtime_code_revision_status(manifest, runtime_python) == (
        False,
        "direct_url_missing_or_linked",
    )

    _write_fake_stableaudio_install(tmp_path, revision=revision)
    duplicate = (
        runtime_python.parent.parent
        / "Lib"
        / "site-packages"
        / "stable_audio_3-9.9.9.dist-info"
    )
    duplicate.mkdir()
    (duplicate / "direct_url.json").write_text("{}", encoding="utf-8")
    assert runtime_code_revision_status(manifest, runtime_python) == (
        False,
        "distribution_metadata_count:2",
    )


def test_start_worker_fails_closed_before_launch_on_code_revision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dubbing_service.app.core import sfx_worker

    monkeypatch.setattr(sfx_worker, "_read_quality_gate", lambda: (True, None))
    monkeypatch.setattr(
        sfx_worker, "_supply_state", lambda **_kwargs: (True, True, None)
    )
    monkeypatch.setattr(
        sfx_worker,
        "_runtime_revision_state",
        lambda: (False, None, "revision mismatch from direct_url.json"),
    )

    with pytest.raises(sfx_worker.SFXWorkerError, match="direct_url.json"):
        sfx_worker.start_worker()
