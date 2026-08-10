#!/usr/bin/env python3
"""Verify project-local model artifacts against a pinned manifest.

This command is deliberately network-free.  It is safe to call from startup
scripts and reports missing, truncated, symlinked, or checksum-mismatched model
files before a GPU worker is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "models" / "manifests" / "index_tts2.json"
_REPARSE_POINT = 0x400


class ManifestError(ValueError):
    """Raised when a model manifest is malformed or unsafe."""


@dataclass(frozen=True)
class ArtifactStatus:
    path: str
    group: str
    state: str
    expected_size: int
    actual_size: int | None = None
    expected_sha256: str | None = None
    actual_sha256: str | None = None

    @property
    def ok(self) -> bool:
        return self.state == "ok"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON manifest {path}: {exc}") from exc

    if payload.get("schema_version") != 1:
        raise ManifestError("unsupported manifest schema_version")
    for field in ("name", "source", "revision"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ManifestError(f"manifest {field} must be a non-empty string")
    artifacts = payload.get("files")
    if not isinstance(artifacts, list) or not artifacts:
        raise ManifestError("manifest files must be a non-empty list")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ManifestError("manifest sources must be a non-empty object")

    seen: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ManifestError("each artifact must be an object")
        rel = artifact.get("path")
        size = artifact.get("size")
        checksum = artifact.get("sha256")
        if not isinstance(rel, str) or not rel:
            raise ManifestError("artifact path must be a non-empty string")
        if rel in seen:
            raise ManifestError(f"duplicate artifact path: {rel}")
        seen.add(rel)
        if not isinstance(size, int) or size < 0:
            raise ManifestError(f"invalid artifact size: {rel}")
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(char not in "0123456789abcdef" for char in checksum.lower())
        ):
            raise ManifestError(f"invalid SHA256 for artifact: {rel}")
        source_name = artifact.get("source")
        source = sources.get(source_name) if isinstance(source_name, str) else None
        if not isinstance(source, dict):
            raise ManifestError(f"unknown artifact source for: {rel}")
        if source.get("provider") == "modelscope":
            source_path = artifact.get("source_path")
            repo_revision = artifact.get("source_revision")
            file_revision = artifact.get("source_file_revision")
            if not isinstance(source_path, str) or not source_path:
                raise ManifestError(
                    f"ModelScope artifact source_path must be non-empty: {rel}"
                )
            if not isinstance(repo_revision, str) or not repo_revision:
                raise ManifestError(
                    f"ModelScope artifact source_revision must be a branch/tag: {rel}"
                )
            if re.fullmatch(r"[0-9a-fA-F]{40}", repo_revision):
                raise ManifestError(
                    f"ModelScope artifact source_revision looks like a file revision, "
                    f"not a downloadable branch/tag: {rel}"
                )
            if not isinstance(file_revision, str) or not re.fullmatch(
                r"[0-9a-fA-F]{40}", file_revision
            ):
                raise ManifestError(
                    f"ModelScope artifact source_file_revision must be pinned: {rel}"
                )
    return payload


def resolve_within(base: Path, relative: str, label: str = "path") -> Path:
    candidate_rel = Path(relative)
    if candidate_rel.is_absolute():
        raise ManifestError(f"{label} must be relative: {relative}")
    resolved_base = base.resolve()
    candidate = Path(os.path.abspath(resolved_base / candidate_rel))
    # Resolve parent components so a directory junction cannot escape the
    # project, but keep the final component intact so it can be rejected when
    # it is itself a symlink/reparse point.
    resolved = candidate.parent.resolve(strict=False) / candidate.name
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise ManifestError(f"{label} escapes its root: {relative}") from exc
    return resolved


def artifact_target(
    manifest: dict[str, Any],
    artifact: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> Path:
    return resolve_within(project_root, artifact["path"], "artifact path")


def _is_reparse_point(path: Path) -> bool:
    stat_result = path.lstat()
    return bool(getattr(stat_result, "st_file_attributes", 0) & _REPARSE_POINT)


def verify_artifact(
    manifest: dict[str, Any],
    artifact: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    verify_hash: bool = True,
) -> ArtifactStatus:
    target = artifact_target(manifest, artifact, project_root)
    relative = artifact["path"]
    group = str(artifact.get("group", "default"))
    expected_size = artifact["size"]
    expected_sha256 = artifact["sha256"].lower()

    if not target.exists():
        return ArtifactStatus(relative, group, "missing", expected_size, expected_sha256=expected_sha256)
    if not target.is_file():
        return ArtifactStatus(relative, group, "not_file", expected_size, expected_sha256=expected_sha256)
    if target.is_symlink() or _is_reparse_point(target):
        return ArtifactStatus(relative, group, "linked", expected_size, expected_sha256=expected_sha256)

    actual_size = target.stat().st_size
    if actual_size != expected_size:
        return ArtifactStatus(
            relative,
            group,
            "size_mismatch",
            expected_size,
            actual_size,
            expected_sha256,
        )
    if not verify_hash:
        return ArtifactStatus(
            relative,
            group,
            "ok",
            expected_size,
            actual_size,
            expected_sha256,
        )

    actual_sha256 = sha256_file(target)
    state = "ok" if actual_sha256 == expected_sha256 else "hash_mismatch"
    return ArtifactStatus(
        relative,
        group,
        state,
        expected_size,
        actual_size,
        expected_sha256,
        actual_sha256,
    )


def selected_artifacts(
    manifest: dict[str, Any], groups: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    selected_groups = set(groups or ())
    artifacts = manifest["files"]
    if not selected_groups:
        return list(artifacts)
    available = {str(item.get("group", "default")) for item in artifacts}
    unknown = selected_groups - available
    if unknown:
        raise ManifestError(f"unknown artifact group(s): {', '.join(sorted(unknown))}")
    return [item for item in artifacts if str(item.get("group", "default")) in selected_groups]


def verify_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    project_root: Path = PROJECT_ROOT,
    verify_hash: bool = True,
    groups: Iterable[str] | None = None,
) -> tuple[dict[str, Any], list[ArtifactStatus]]:
    manifest = load_manifest(manifest_path)
    statuses = [
        verify_artifact(
            manifest,
            artifact,
            project_root=project_root,
            verify_hash=verify_hash,
        )
        for artifact in selected_artifacts(manifest, groups)
    ]
    return manifest, statuses


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="check existence, regular-file status, and size without reading full SHA256",
    )
    parser.add_argument(
        "--group",
        action="append",
        help="verify only a named artifact group; may be repeated",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest, statuses = verify_manifest(
            args.manifest.resolve(),
            project_root=PROJECT_ROOT,
            verify_hash=not args.quick,
            groups=args.group,
        )
    except (ManifestError, OSError) as exc:
        print(f"model preflight error: {exc}", file=sys.stderr)
        return 2

    failures = [status for status in statuses if not status.ok]
    if args.json:
        print(
            json.dumps(
                {
                    "model_id": manifest.get("model_id", manifest["name"]),
                    "ok": not failures,
                    "verification": "size" if args.quick else "sha256",
                    "artifacts": [asdict(status) for status in statuses],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        mode = "size-only" if args.quick else "SHA256"
        print(f"[{manifest.get('model_id', manifest['name'])}] checking {len(statuses)} artifacts ({mode})")
        for failure in failures:
            details = ""
            if failure.actual_size is not None:
                details = f" (expected {failure.expected_size}, got {failure.actual_size} bytes)"
            print(f"  FAIL {failure.path}: {failure.state}{details}")
        if failures:
            print(f"FAILED: {len(failures)} artifact(s) require recovery")
        else:
            total = sum(status.expected_size for status in statuses)
            print(f"OK: {len(statuses)} artifacts, {total / (1024 ** 3):.2f} GiB verified")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
