from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKUP_SCRIPT = ROOT / "scripts" / "backup_project.ps1"


def _powershell() -> str:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is not available")
    return executable


def _run_backup_script(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *arguments,
        ],
        cwd=script.parents[1],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=60,
    )


def test_default_backup_plan_keeps_only_required_model_recovery_material() -> None:
    bigvgan_blob = (
        ROOT
        / "models"
        / "recovery_blobs"
        / "index_tts"
        / "bigvgan"
        / "bigvgan_generator.pt"
    )
    if not bigvgan_blob.is_file():
        pytest.skip("local non-downloadable recovery blob is absent in this source-only checkout")

    completed = _run_backup_script(BACKUP_SCRIPT, "-ValidateOnly")
    assert completed.returncode == 0, completed.stdout + completed.stderr

    marker = "BACKUP_VALIDATION_JSON="
    report_line = next(
        (line for line in completed.stdout.splitlines() if line.startswith(marker)),
        None,
    )
    assert report_line is not None, completed.stdout
    report = json.loads(report_line[len(marker) :])

    assert report["mode"] == "recoverable-default"
    assert report["includes_models"] is False
    assert report["credentials_excluded"] is True
    assert report["credential_scan_file_count"] > 0
    assert report["secret_paths_selected"] == []
    assert report["ordinary_model_paths_selected"] == []
    assert report["ordinary_manifest_model_path_count"] > 0
    assert report["ordinary_manifest_model_paths_selected"] == []
    assert {
        ".env",
        "dubbing_service/config.json",
        "dubbing_service/llm_providers.json",
    }.issubset(set(report["excluded_sensitive_paths"]))
    assert "models/manifests/index_tts2.json" in report["manifest_paths"]
    assert set(report["recovery_blob_paths"]) == {
        "models/recovery_blobs/index_tts/bigvgan/config.json",
        "models/recovery_blobs/index_tts/bigvgan/bigvgan_generator.pt",
    }
    assert report["recovery_blob_count"] == 2
    assert report["validated_recovery_bytes"] == 449_229_576


def test_small_fixture_backup_is_hash_verified_and_excludes_secrets_and_weight(
    tmp_path: Path,
) -> None:
    project = tmp_path / "fixture-project"
    script = project / "scripts" / "backup_project.ps1"
    script.parent.mkdir(parents=True)
    shutil.copy2(BACKUP_SCRIPT, script)

    recovery_content = b"fixture non-downloadable recovery blob\n"
    recovery_hash = hashlib.sha256(recovery_content).hexdigest()
    files = {
        "models/recovery_blobs/index_tts/bigvgan/model.pt": recovery_content,
        "models/index_tts/ordinary-weight.pt": b"must not be copied\n",
        "models/README.md": b"fixture model recovery instructions\n",
        "docs/operations.md": b"fixture operations\n",
        "dubbing_projects/demo/project.json": b'{"name":"private fixture"}\n',
        ".env": b"DUBBING_LLM_API_KEY=must-not-be-copied\n",
        "dubbing_service/config.json": b'{"llm_api_key":"must-not-be-copied"}\n',
        "dubbing_service/llm_providers.json": b'{"api_key":"must-not-be-copied"}\n',
    }
    for relative, content in files.items():
        target = project / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    manifest_path = project / "models" / "manifests" / "index_tts2.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "path": "models/index_tts/ordinary-weight.pt",
                        "size": len(recovery_content),
                        "sha256": recovery_hash,
                        "recovery": [
                            {
                                "provider": "project_cache",
                                "path": "models/recovery_blobs/index_tts/bigvgan/model.pt",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    unsafe_destination = project / "unsafe-backup"
    rejected = _run_backup_script(script, "-DestinationRoot", str(unsafe_destination))
    assert rejected.returncode != 0
    assert "must be outside the project directory" in rejected.stdout + rejected.stderr
    assert not unsafe_destination.exists()

    backup_root = tmp_path / "backups"
    completed = _run_backup_script(script, "-DestinationRoot", str(backup_root))
    assert completed.returncode == 0, completed.stdout + completed.stderr

    backups = [path for path in backup_root.iterdir() if path.is_dir()]
    assert len(backups) == 1
    backup = backups[0]
    assert not backup.name.endswith(".partial")

    manifest = json.loads((backup / "backup-manifest.json").read_text(encoding="utf-8-sig"))
    archived_paths = {entry["path"] for entry in manifest["files"]}
    assert "models/recovery_blobs/index_tts/bigvgan/model.pt" in archived_paths
    assert "models/manifests/index_tts2.json" in archived_paths
    assert "dubbing_projects/demo/project.json" in archived_paths
    assert "models/index_tts/ordinary-weight.pt" not in archived_paths
    assert ".env" not in archived_paths
    assert "dubbing_service/config.json" not in archived_paths
    assert "dubbing_service/llm_providers.json" not in archived_paths

    for entry in manifest["files"]:
        archived_file = backup / Path(entry["path"])
        assert archived_file.stat().st_size == entry["size"]
        assert hashlib.sha256(archived_file.read_bytes()).hexdigest() == entry["sha256"]

    summary = json.loads((backup / "backup.json").read_text(encoding="utf-8-sig"))
    assert summary["contains_credentials"] is False
    assert summary["backup_manifest_sha256"] == hashlib.sha256(
        (backup / "backup-manifest.json").read_bytes()
    ).hexdigest()


def test_modelscope_recovery_blob_is_optional_for_source_only_backup(
    tmp_path: Path,
) -> None:
    project = tmp_path / "fixture-project"
    script = project / "scripts" / "backup_project.ps1"
    script.parent.mkdir(parents=True)
    shutil.copy2(BACKUP_SCRIPT, script)

    (project / "models" / "manifests").mkdir(parents=True)
    (project / "models" / "README.md").write_text("models\n", encoding="utf-8")
    (project / "docs").mkdir(parents=True)
    (project / "docs" / "operations.md").write_text("operations\n", encoding="utf-8")
    (project / "models" / "manifests" / "downloadable.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    "download": {
                        "provider": "modelscope",
                        "model_id": "example/model",
                    }
                },
                "files": [
                    {
                        "path": "models/runtime/model.bin",
                        "size": 1,
                        "sha256": "0" * 64,
                        "source": "download",
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

    completed = _run_backup_script(script, "-ValidateOnly")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '"recovery_blob_count":0' in completed.stdout.replace(" ", "")


def test_disclaimer_has_explicit_release_blockers_instead_of_placeholders() -> None:
    disclaimer = (ROOT / "DISCLAIMER").read_text(encoding="utf-8")

    assert "[开源许可证类型]" not in disclaimer
    assert "[国家/地区]" not in disclaimer
    assert "尚未为“整个配音平台”的对外发布选择并确认单一许可证" in disclaimer
    assert "本声明尚未选择特定国家或地区的管辖法律" in disclaimer
    assert "对外发布前必须填写" in disclaimer
