#!/usr/bin/env python3
"""Recover pinned model artifacts into ordinary project-local files.

Recovery order is intentionally conservative:

1. a manifest-pinned blob already inside this project;
2. an explicitly enabled download from a manifest-pinned ModelScope source;
3. an explicitly enabled, license-confirmed gated Hugging Face source.

No Hugging Face cache or network endpoint is used implicitly.  Every candidate
is checked for exact byte size and SHA256 before it is atomically promoted.
Invalid existing targets are quarantined rather than deleted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

try:
    from .model_preflight import (
        DEFAULT_MANIFEST,
        PROJECT_ROOT,
        ManifestError,
        artifact_target,
        load_manifest,
        record_license_acceptance,
        required_license_confirmation,
        resolve_within,
        selected_artifacts,
        sha256_file,
        verify_artifact,
    )
except ImportError:  # Direct execution: python scripts/model_provision.py
    from model_preflight import (
        DEFAULT_MANIFEST,
        PROJECT_ROOT,
        ManifestError,
        artifact_target,
        load_manifest,
        record_license_acceptance,
        required_license_confirmation,
        resolve_within,
        selected_artifacts,
        sha256_file,
        verify_artifact,
    )


class ProvisionError(RuntimeError):
    """Raised when an artifact cannot be safely recovered."""


def _candidate_matches(candidate: Path, artifact: dict[str, Any]) -> bool:
    return (
        candidate.is_file()
        and not candidate.is_symlink()
        and candidate.stat().st_size == artifact["size"]
        and sha256_file(candidate) == artifact["sha256"].lower()
    )


def _cache_candidates(
    artifact: dict[str, Any], project_root: Path
) -> Iterable[tuple[str, Path]]:
    for source in artifact.get("recovery", []):
        if source.get("provider") != "project_cache":
            continue
        raw_path = source.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        yield "project cache", resolve_within(project_root, raw_path, "recovery path")


def _download_from_modelscope(
    manifest: dict[str, Any],
    artifact: dict[str, Any],
    project_root: Path,
) -> Path:
    source_name = artifact.get("source")
    source = manifest.get("sources", {}).get(source_name, {})
    if source.get("provider") != "modelscope":
        raise ProvisionError(
            f"{artifact['path']}: no approved ModelScope download source; "
            "restore its pinned project-cache blob first"
        )
    model_id = source.get("model_id")
    file_path = artifact.get("source_path")
    revision = artifact.get("source_revision")
    if not all(isinstance(value, str) and value for value in (model_id, file_path, revision)):
        raise ProvisionError(f"{artifact['path']}: incomplete pinned ModelScope source")

    try:
        from modelscope.hub.api import HubApi
        from modelscope.hub.file_download import model_file_download
    except ImportError as exc:
        raise ProvisionError(
            "ModelScope is not installed; run provisioning from the IndexTTS environment"
        ) from exc

    cache_dir = project_root / "models" / ".download_cache" / "modelscope"
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected_file_revision = artifact.get("source_file_revision")
    if expected_file_revision:
        try:
            entries = HubApi().get_model_files(
                model_id,
                revision=revision,
                recursive=True,
            )
        except Exception as exc:
            raise ProvisionError(
                f"{artifact['path']}: failed to verify ModelScope metadata: {exc}"
            ) from exc
        matches = [entry for entry in entries if entry.get("Path") == file_path]
        if len(matches) != 1:
            raise ProvisionError(
                f"{artifact['path']}: expected exactly one ModelScope metadata entry"
            )
        entry = matches[0]
        metadata_matches = (
            entry.get("Revision") == expected_file_revision
            and entry.get("Size") == artifact["size"]
            and str(entry.get("Sha256", "")).lower() == artifact["sha256"].lower()
        )
        if not metadata_matches:
            raise ProvisionError(
                f"{artifact['path']}: ModelScope metadata changed; refusing download"
            )
    try:
        downloaded = model_file_download(
            model_id=model_id,
            file_path=file_path,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=False,
        )
    except Exception as exc:  # ModelScope uses several transport exception types.
        raise ProvisionError(
            f"{artifact['path']}: ModelScope download failed "
            f"({model_id}@{revision}:{file_path}): {exc}"
        ) from exc
    if not downloaded:
        raise ProvisionError(f"{artifact['path']}: ModelScope returned no local file")
    return Path(downloaded)


def _download_from_huggingface(
    manifest: dict[str, Any],
    artifact: dict[str, Any],
    project_root: Path,
) -> Path:
    source_name = artifact.get("source")
    source = manifest.get("sources", {}).get(source_name, {})
    if source.get("provider") != "huggingface" or source.get("gated") is not True:
        raise ProvisionError(f"{artifact['path']}: no approved gated Hugging Face source")
    repo_id = source.get("repo_id")
    revision = source.get("revision")
    file_path = artifact.get("source_path")
    token_env = source.get("token_env")
    if not all(
        isinstance(value, str) and value
        for value in (repo_id, revision, file_path, token_env)
    ):
        raise ProvisionError(f"{artifact['path']}: incomplete gated Hugging Face source")
    token = os.environ.get(token_env)
    if not token:
        raise ProvisionError(
            f"{artifact['path']}: {token_env} is not set. Accept access on the official "
            "Hugging Face model page and provide a read token through the environment only."
        )
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ProvisionError(
            "huggingface-hub is not installed; run provisioning from the platform environment"
        ) from exc

    cache_dir = project_root / "models" / ".download_cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        info = HfApi(token=token).model_info(
            repo_id=repo_id,
            revision=revision,
            files_metadata=True,
        )
        if info.sha != revision:
            raise ProvisionError(
                f"{artifact['path']}: Hugging Face resolved a different commit"
            )
        matches = [item for item in info.siblings if item.rfilename == file_path]
        if len(matches) != 1:
            raise ProvisionError(
                f"{artifact['path']}: expected exactly one Hugging Face metadata entry"
            )
        entry = matches[0]
        if (
            entry.size != artifact["size"]
            or str(entry.blob_id).lower()
            != str(artifact.get("source_blob_id", "")).lower()
        ):
            raise ProvisionError(
                f"{artifact['path']}: Hugging Face metadata changed; refusing download"
            )
    except ProvisionError:
        raise
    except Exception as exc:
        raise ProvisionError(
            f"{artifact['path']}: failed to verify gated Hugging Face metadata: {exc}"
        ) from exc
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="model",
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=False,
        )
    except Exception as exc:  # huggingface_hub exposes several transport errors.
        raise ProvisionError(
            f"{artifact['path']}: gated Hugging Face download failed "
            f"({repo_id}@{revision}:{file_path}): {exc}"
        ) from exc
    if not downloaded:
        raise ProvisionError(f"{artifact['path']}: Hugging Face returned no local file")
    return Path(downloaded)


def _quarantine_existing(
    target: Path,
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    project_root: Path,
) -> Path | None:
    if not target.exists() and not target.is_symlink():
        return None
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_id = str(manifest.get("model_id", manifest["name"]))
    relative = Path(artifact["path"])
    quarantine = (
        project_root
        / "models"
        / ".quarantine"
        / model_id
        / timestamp
        / relative
    )
    counter = 1
    original = quarantine
    while quarantine.exists() or quarantine.is_symlink():
        quarantine = original.with_name(f"{original.name}.{counter}")
        counter += 1
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(quarantine))
    return quarantine


def _atomic_promote(
    source: Path,
    target: Path,
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    project_root: Path,
) -> Path | None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part-{os.getpid()}-{uuid.uuid4().hex}")
    quarantine: Path | None = None
    try:
        shutil.copy2(source, temporary)
        if not _candidate_matches(temporary, artifact):
            raise ProvisionError(f"copied candidate failed verification: {artifact['path']}")
        quarantine = _quarantine_existing(target, artifact, manifest, project_root)
        try:
            os.replace(temporary, target)
        except Exception as promote_error:
            if quarantine is not None:
                try:
                    if target.exists() or target.is_symlink():
                        raise ProvisionError(
                            f"promotion failed but target unexpectedly exists: {artifact['path']}"
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(quarantine), str(target))
                except Exception as rollback_error:
                    raise ProvisionError(
                        f"promotion and rollback both failed for {artifact['path']}: "
                        f"promote={promote_error}; rollback={rollback_error}"
                    ) from promote_error
                raise ProvisionError(
                    f"promotion failed; original target restored for {artifact['path']}: "
                    f"{promote_error}"
                ) from promote_error
            raise ProvisionError(
                f"promotion failed for new target {artifact['path']}: {promote_error}"
            ) from promote_error
        return quarantine
    finally:
        if temporary.exists():
            temporary.unlink()


def provision_artifact(
    manifest: dict[str, Any],
    artifact: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    allow_modelscope_download: bool = False,
    allow_huggingface_gated_download: bool = False,
    accepted_license_hash: str | None = None,
    dry_run: bool = False,
) -> tuple[str, Path | None]:
    current = verify_artifact(
        manifest,
        artifact,
        project_root=project_root,
        verify_hash=True,
    )
    target = artifact_target(manifest, artifact, project_root)
    if current.ok:
        return "already_valid", None

    for label, candidate in _cache_candidates(artifact, project_root):
        if not _candidate_matches(candidate, artifact):
            continue
        if dry_run:
            return f"would_restore_from_{label.replace(' ', '_')}", None
        quarantine = _atomic_promote(candidate, target, artifact, manifest, project_root)
        return "restored_from_project_cache", quarantine

    source = manifest.get("sources", {}).get(artifact.get("source"), {})
    provider = source.get("provider")
    if provider == "huggingface":
        if not allow_huggingface_gated_download:
            raise ProvisionError(
                f"{artifact['path']}: {current.state}; no valid pinned project-cache copy. "
                "Re-run with --allow-huggingface-gated-download to use the approved source."
            )
        if dry_run:
            return "would_download_from_huggingface", None
        expected_license_hash = required_license_confirmation(manifest)
        if expected_license_hash is None or accepted_license_hash != expected_license_hash:
            raise ProvisionError(
                f"{artifact['path']}: --accept-license-hash must exactly match "
                f"{expected_license_hash or 'the manifest confirmation hash'}"
            )
        downloaded = _download_from_huggingface(manifest, artifact, project_root)
        if not _candidate_matches(downloaded, artifact):
            raise ProvisionError(
                f"{artifact['path']}: downloaded bytes do not match pinned size/SHA256"
            )
        quarantine = _atomic_promote(downloaded, target, artifact, manifest, project_root)
        return "restored_from_huggingface", quarantine

    if not allow_modelscope_download:
        raise ProvisionError(
            f"{artifact['path']}: {current.state}; no valid pinned project-cache copy. "
            "Re-run with --allow-modelscope-download to use the approved source."
        )
    if dry_run:
        source = manifest.get("sources", {}).get(artifact.get("source"), {})
        if source.get("provider") != "modelscope":
            raise ProvisionError(f"{artifact['path']}: no approved ModelScope source")
        return "would_download_from_modelscope", None

    downloaded = _download_from_modelscope(manifest, artifact, project_root)
    if not _candidate_matches(downloaded, artifact):
        raise ProvisionError(
            f"{artifact['path']}: downloaded bytes do not match pinned size/SHA256"
        )
    quarantine = _atomic_promote(downloaded, target, artifact, manifest, project_root)
    return "restored_from_modelscope", quarantine


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--group",
        action="append",
        help="recover only a named artifact group; may be repeated",
    )
    parser.add_argument(
        "--allow-huggingface-gated-download",
        action="store_true",
        help="allow pinned gated Hugging Face downloads; requires HF_TOKEN in the environment",
    )
    parser.add_argument(
        "--accept-license-hash",
        help="exact manifest confirmation hash; never accepts a token or account identifier",
    )
    parser.add_argument(
        "--license-tier",
        choices=("community", "enterprise"),
        default="community",
        help="license tier represented by the explicit confirmation",
    )
    parser.add_argument(
        "--allow-modelscope-download",
        action="store_true",
        help="allow pinned ModelScope downloads when no valid project-cache blob exists",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = load_manifest(args.manifest.resolve())
        artifacts = selected_artifacts(manifest, args.group)
    except (ManifestError, OSError) as exc:
        print(f"model provision error: {exc}", file=sys.stderr)
        return 2

    failures = 0
    changes = 0
    for artifact in artifacts:
        try:
            state, quarantine = provision_artifact(
                manifest,
                artifact,
                project_root=PROJECT_ROOT,
                allow_modelscope_download=args.allow_modelscope_download,
                allow_huggingface_gated_download=args.allow_huggingface_gated_download,
                accepted_license_hash=args.accept_license_hash,
                dry_run=args.dry_run,
            )
            if state != "already_valid":
                changes += 1
                suffix = f"; previous target saved to {quarantine}" if quarantine else ""
                print(f"{state}: {artifact['path']}{suffix}")
        except (ManifestError, OSError, ProvisionError) as exc:
            failures += 1
            print(f"FAILED: {exc}", file=sys.stderr)

    if failures:
        print(f"Provisioning failed for {failures} artifact(s)", file=sys.stderr)
        return 1
    if (
        args.allow_huggingface_gated_download
        and not args.dry_run
        and args.accept_license_hash
    ):
        try:
            receipt = record_license_acceptance(
                manifest,
                license_tier=args.license_tier,
                stability_registered=True,
                stability_terms_accepted=True,
                gemma_terms_accepted=True,
                project_root=PROJECT_ROOT,
            )
            print(f"license_acceptance_recorded: {receipt}")
        except (ManifestError, OSError) as exc:
            print(f"FAILED: license acceptance receipt: {exc}", file=sys.stderr)
            return 1
    verb = "would recover" if args.dry_run else "recovered"
    print(f"OK: {len(artifacts)} artifact(s) checked, {changes} {verb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
