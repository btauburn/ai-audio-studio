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
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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

    schema_version = payload.get("schema_version")
    if schema_version not in (1, 2):
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
        if source.get("provider") == "huggingface":
            repo_id = source.get("repo_id")
            source_path = artifact.get("source_path")
            repo_revision = source.get("revision")
            file_revision = artifact.get("source_revision")
            blob_id = artifact.get("source_blob_id")
            if not isinstance(repo_id, str) or not re.fullmatch(
                r"[^/\s]+/[^/\s]+", repo_id
            ):
                raise ManifestError(f"Hugging Face repo_id is invalid: {rel}")
            if not isinstance(source_path, str) or not source_path:
                raise ManifestError(
                    f"Hugging Face artifact source_path must be non-empty: {rel}"
                )
            if not isinstance(repo_revision, str) or not re.fullmatch(
                r"[0-9a-fA-F]{40}", repo_revision
            ):
                raise ManifestError(
                    f"Hugging Face source revision must be a pinned commit: {rel}"
                )
            if manifest_revision := payload.get("revision"):
                if manifest_revision != repo_revision:
                    raise ManifestError(
                        f"Hugging Face source revision must equal manifest revision: {rel}"
                    )
            if file_revision != repo_revision:
                raise ManifestError(
                    f"Hugging Face artifact source_revision must equal the pinned "
                    f"source commit: {rel}"
                )
            if not isinstance(blob_id, str) or not re.fullmatch(
                r"[0-9a-fA-F]{40,64}", blob_id
            ):
                raise ManifestError(
                    f"Hugging Face artifact source_blob_id must be pinned: {rel}"
                )
            gated = source.get("gated")
            token_env = source.get("token_env")
            if gated is not True:
                raise ManifestError(
                    f"Hugging Face source must declare gated=true: {rel}"
                )
            if token_env != "HF_TOKEN":
                raise ManifestError(
                    f"Hugging Face gated source token_env must be HF_TOKEN: {rel}"
                )

    if schema_version == 2:
        _validate_license_policy(payload)
        _validate_quality_gate_policy(payload)
    _validate_code_source_policy(payload)
    return payload


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", value))


def _validate_code_source_policy(manifest: dict[str, Any]) -> None:
    """Validate an optional, pinned Python distribution provenance policy.

    Older manifests intentionally remain valid.  Once either field is present,
    however, both the exact Git commit and the expected direct-install source
    must be declared so runtime verification can fail closed.
    """

    revision = manifest.get("code_revision")
    source = manifest.get("code_source")
    if revision is None and source is None:
        return
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ManifestError("code_revision must be a lowercase pinned Git commit")
    if not isinstance(source, dict):
        raise ManifestError("code_source must be an object when code_revision is set")
    distribution = source.get("distribution")
    if not isinstance(distribution, str) or not re.fullmatch(
        r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", distribution
    ):
        raise ManifestError("code_source distribution is invalid")
    url = source.get("url")
    if (
        not isinstance(url, str)
        or not re.fullmatch(r"https://[^\s?#]+\.git", url)
        or "@" in url.removeprefix("https://")
    ):
        raise ManifestError("code_source url must be an uncredentialed pinned HTTPS Git URL")


def _validate_license_policy(manifest: dict[str, Any]) -> None:
    policy = manifest.get("license_policy")
    if not isinstance(policy, dict):
        raise ManifestError("schema v2 manifest license_policy must be an object")
    if policy.get("requires_explicit_acceptance") is not True:
        raise ManifestError("schema v2 manifest must require explicit license acceptance")
    confirmation_text = policy.get("confirmation_text")
    confirmation_sha256 = policy.get("confirmation_sha256")
    if not isinstance(confirmation_text, str) or not confirmation_text.strip():
        raise ManifestError("license confirmation_text must be non-empty")
    if not _is_sha256(confirmation_sha256):
        raise ManifestError("license confirmation_sha256 must be a SHA256")
    actual = hashlib.sha256(confirmation_text.encode("utf-8")).hexdigest()
    if actual != str(confirmation_sha256).lower():
        raise ManifestError("license confirmation_sha256 does not match confirmation_text")
    receipt_path = policy.get("receipt_path")
    if not isinstance(receipt_path, str) or not receipt_path:
        raise ManifestError("license receipt_path must be non-empty")
    documents = policy.get("documents")
    if not isinstance(documents, list) or len(documents) < 2:
        raise ManifestError("license_policy must list the Stability and Gemma terms")
    artifacts = {item["path"]: item for item in manifest["files"]}
    for document in documents:
        if not isinstance(document, dict):
            raise ManifestError("each license document must be an object")
        for field in ("name", "url", "path"):
            if not isinstance(document.get(field), str) or not document[field]:
                raise ManifestError(f"license document {field} must be non-empty")
        if not _is_sha256(document.get("sha256")):
            raise ManifestError("license document sha256 must be pinned")
        artifact = artifacts.get(document["path"])
        if artifact is None or artifact["sha256"].lower() != document["sha256"].lower():
            raise ManifestError("license document must match a pinned manifest artifact")


def _validate_quality_gate_policy(manifest: dict[str, Any]) -> None:
    policy = manifest.get("quality_gate")
    if not isinstance(policy, dict):
        raise ManifestError("schema v2 manifest quality_gate must be an object")
    required_values = {
        "categories_total": 12,
        "cases_total": 24,
        "minimum_categories_passed": 10,
        "technical_rounds": 5,
        "technical_runs": 30,
    }
    if not isinstance(policy.get("path"), str) or not policy["path"]:
        raise ManifestError("quality_gate path must be non-empty")
    for field, expected in required_values.items():
        if policy.get(field) != expected:
            raise ManifestError(f"quality_gate {field} must be {expected}")


def required_license_confirmation(manifest: dict[str, Any]) -> str | None:
    policy = manifest.get("license_policy")
    if not isinstance(policy, dict) or policy.get("requires_explicit_acceptance") is not True:
        return None
    value = policy.get("confirmation_sha256")
    return str(value).lower() if _is_sha256(value) else None


def license_receipt_path(
    manifest: dict[str, Any], project_root: Path = PROJECT_ROOT
) -> Path | None:
    policy = manifest.get("license_policy")
    if not isinstance(policy, dict):
        return None
    raw_path = policy.get("receipt_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    return resolve_within(project_root, raw_path, "license receipt path")


def _valid_operator_confirmation_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return True


def _valid_utc_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def license_acceptance_status(
    manifest: dict[str, Any], project_root: Path = PROJECT_ROOT
) -> tuple[bool, str]:
    expected = required_license_confirmation(manifest)
    if expected is None:
        return True, "not_required"
    path = license_receipt_path(manifest, project_root)
    if (
        path is None
        or not path.is_file()
        or path.is_symlink()
        or _is_reparse_point(path)
    ):
        return False, "missing"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "invalid_json"
    checks = (
        receipt.get("schema_version") == 1,
        receipt.get("model_id") == manifest.get("model_id"),
        receipt.get("model_revision") == manifest.get("revision"),
        receipt.get("confirmation_sha256") == expected,
        receipt.get("license_tier") in {"community", "enterprise"},
        receipt.get("stability_registered") is True,
        receipt.get("stability_terms_accepted") is True,
        receipt.get("gemma_terms_accepted") is True,
        _valid_operator_confirmation_id(receipt.get("operator_confirmation_id")),
        _valid_utc_timestamp(receipt.get("accepted_at_utc")),
    )
    return (True, "accepted") if all(checks) else (False, "stale_or_incomplete")


def has_license_acceptance(
    manifest: dict[str, Any], project_root: Path = PROJECT_ROOT
) -> bool:
    return license_acceptance_status(manifest, project_root)[0]


def record_license_acceptance(
    manifest: dict[str, Any],
    *,
    license_tier: str,
    stability_registered: bool,
    stability_terms_accepted: bool,
    gemma_terms_accepted: bool,
    operator_confirmation_id: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    expected = required_license_confirmation(manifest)
    path = license_receipt_path(manifest, project_root)
    if expected is None or path is None:
        raise ManifestError("manifest does not define an explicit license acceptance")
    if license_tier not in {"community", "enterprise"}:
        raise ManifestError("license_tier must be community or enterprise")
    if not all(
        value is True
        for value in (
            stability_registered,
            stability_terms_accepted,
            gemma_terms_accepted,
        )
    ):
        raise ManifestError(
            "Stability registration and both Stability/Gemma term acceptances are required"
        )
    confirmation_id = operator_confirmation_id or str(uuid.uuid4())
    if not _valid_operator_confirmation_id(confirmation_id):
        raise ManifestError("operator_confirmation_id must be a UUID and contain no identity data")
    receipt = {
        "schema_version": 1,
        "model_id": manifest.get("model_id"),
        "model_revision": manifest.get("revision"),
        "confirmation_sha256": expected,
        "license_tier": license_tier,
        "stability_registered": True,
        "stability_terms_accepted": True,
        "gemma_terms_accepted": True,
        "operator_confirmation_id": confirmation_id,
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def quality_gate_path(
    manifest: dict[str, Any], project_root: Path = PROJECT_ROOT
) -> Path | None:
    policy = manifest.get("quality_gate")
    if not isinstance(policy, dict) or not isinstance(policy.get("path"), str):
        return None
    return resolve_within(project_root, policy["path"], "quality gate path")


def quality_gate_status(
    manifest: dict[str, Any], project_root: Path = PROJECT_ROOT
) -> tuple[bool, str]:
    policy = manifest.get("quality_gate")
    path = quality_gate_path(manifest, project_root)
    if not isinstance(policy, dict):
        return True, "not_required"
    if (
        path is None
        or not path.is_file()
        or path.is_symlink()
        or _is_reparse_point(path)
    ):
        return False, "missing"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "invalid_json"
    expected_files = {
        item["path"]: (item["size"], item["sha256"].lower())
        for item in manifest["files"]
        if item.get("group") == "runtime"
    }
    actual_files = {}
    for item in receipt.get("model_files", []):
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            actual_files[item["path"]] = (
                item.get("size"),
                str(item.get("sha256", "")).lower(),
            )
    quality = receipt.get("quality", {})
    technical = receipt.get("technical", {})
    categories = quality.get("categories", [])
    rounds = technical.get("rounds", [])
    handoffs = technical.get("handoffs", [])
    warm_latency = technical.get("warm_latency", {})
    category_passes = sum(
        item.get("passed") is True for item in categories if isinstance(item, dict)
    )
    technical_runs = sum(
        item.get("runs_total", 0) for item in rounds if isinstance(item, dict)
    )
    technical_passes = sum(
        item.get("runs_passed", 0) for item in rounds if isinstance(item, dict)
    )
    expected_revisions = {
        key: value.get("revision")
        for key, value in manifest.get("sources", {}).items()
        if isinstance(value, dict) and isinstance(value.get("revision"), str)
    }
    expected_code_revision = manifest.get("code_revision")
    expected_code_source = manifest.get("code_source")
    expected_code_distribution = (
        expected_code_source.get("distribution")
        if isinstance(expected_code_source, dict)
        else None
    )
    expected_code_source_url = (
        expected_code_source.get("url")
        if isinstance(expected_code_source, dict)
        else None
    )
    category_names = [
        item.get("category") for item in categories if isinstance(item, dict)
    ]
    category_cases = sum(
        item.get("cases_total", 0) for item in categories if isinstance(item, dict)
    )
    rounds_are_complete = {
        item.get("round") for item in rounds if isinstance(item, dict)
    } == {1, 2, 3, 4, 5} and all(
        item.get("runs_total") == 6 and item.get("runs_passed") == 6
        for item in rounds
        if isinstance(item, dict)
    )
    checks = (
        receipt.get("schema_version") == 1,
        receipt.get("passed") is True,
        receipt.get("model_id") == manifest.get("model_id"),
        receipt.get("model_revision") == manifest.get("revision"),
        receipt.get("source_revisions") == expected_revisions,
        expected_code_revision is None
        or receipt.get("code_revision") == expected_code_revision,
        expected_code_distribution is None
        or receipt.get("code_distribution") == expected_code_distribution,
        expected_code_source_url is None
        or receipt.get("code_source_url") == expected_code_source_url,
        actual_files == expected_files,
        quality.get("categories_total") == policy.get("categories_total"),
        quality.get("cases_total") == policy.get("cases_total"),
        quality.get("minimum_categories_passed")
        == policy.get("minimum_categories_passed"),
        len(categories) == policy.get("categories_total"),
        len(set(category_names)) == policy.get("categories_total"),
        category_cases == policy.get("cases_total"),
        all(
            isinstance(item, dict)
            and item.get("cases_total") == 2
            and item.get("cases_passed") in {0, 1, 2}
            and item.get("passed") is (item.get("cases_passed", 0) >= 1)
            for item in categories
        ),
        category_passes == quality.get("categories_passed"),
        category_passes >= policy.get("minimum_categories_passed", 10),
        technical.get("rounds_total") == policy.get("technical_rounds"),
        technical.get("runs_total") == policy.get("technical_runs"),
        technical.get("runs_passed") == policy.get("technical_runs"),
        technical.get("handoffs_total") == policy.get("technical_rounds"),
        technical.get("handoffs_passed") == policy.get("technical_rounds"),
        len(rounds) == policy.get("technical_rounds"),
        rounds_are_complete,
        technical_runs == policy.get("technical_runs"),
        technical_passes == policy.get("technical_runs"),
        len(handoffs) == policy.get("technical_rounds"),
        {
            item.get("round") for item in handoffs if isinstance(item, dict)
        }
        == {1, 2, 3, 4, 5},
        all(
            isinstance(item, dict)
            and item.get("round") in {1, 2, 3, 4, 5}
            and item.get("passed") is True
            and item.get("memory_release_passed") is True
            and item.get("tts_unload", {}).get("success") is True
            and item.get("sfx_load", {}).get("success") is True
            and item.get("sfx_unload", {}).get("success") is True
            and item.get("tts_load", {}).get("success") is True
            for item in handoffs
        ),
        warm_latency.get("runs_total") == 25,
        warm_latency.get("p95_passed") is True,
        isinstance(warm_latency.get("p95_budget_ratio"), (int, float)),
        warm_latency.get("p95_budget_ratio", float("inf")) <= 1.0,
    )
    return (True, "passed") if all(checks) else (False, "stale_or_failed")


def has_passed_quality_gate(
    manifest: dict[str, Any], project_root: Path = PROJECT_ROOT
) -> bool:
    return quality_gate_status(manifest, project_root)[0]


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


def runtime_code_revision_status(
    manifest: dict[str, Any], runtime_python: Path
) -> tuple[bool, str]:
    """Verify the installed distribution's PEP 610 provenance without importing it.

    The worker environment is isolated from the platform environment, so the
    authoritative install provenance is the distribution's ``direct_url.json``.
    Reading it directly also avoids executing package or ``.pth`` code during a
    safety preflight.
    """

    revision = manifest.get("code_revision")
    source = manifest.get("code_source")
    if revision is None and source is None:
        return True, "not_required"
    try:
        _validate_code_source_policy(manifest)
    except ManifestError:
        return False, "invalid_manifest_code_source"
    assert isinstance(revision, str)
    assert isinstance(source, dict)
    distribution = str(source["distribution"])
    expected_url = str(source["url"])

    if (
        not runtime_python.is_file()
        or runtime_python.is_symlink()
        or _is_reparse_point(runtime_python)
    ):
        return False, "runtime_python_missing_or_linked"

    environment_root = runtime_python.parent.parent
    site_roots = [environment_root / "Lib" / "site-packages"]
    for library_root_name in ("lib", "lib64"):
        library_root = environment_root / library_root_name
        if library_root.is_dir():
            site_roots.extend(library_root.glob("python*/site-packages"))

    normalized_distribution = re.sub(r"[-_.]+", "_", distribution)
    metadata_dirs: list[Path] = []
    for site_root in site_roots:
        if site_root.is_dir():
            metadata_dirs.extend(site_root.glob(f"{normalized_distribution}-*.dist-info"))
    # Multiple records can make import selection path-dependent, so treat a
    # partially upgraded or otherwise polluted environment as untrusted.
    if len(metadata_dirs) != 1:
        return False, f"distribution_metadata_count:{len(metadata_dirs)}"
    metadata_dir = metadata_dirs[0]
    direct_url_path = metadata_dir / "direct_url.json"
    if (
        metadata_dir.is_symlink()
        or _is_reparse_point(metadata_dir)
        or not direct_url_path.is_file()
        or direct_url_path.is_symlink()
        or _is_reparse_point(direct_url_path)
    ):
        return False, "direct_url_missing_or_linked"
    try:
        if direct_url_path.stat().st_size > 64 * 1024:
            return False, "direct_url_too_large"
        direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False, "direct_url_invalid_json"
    if not isinstance(direct_url, dict) or direct_url.get("url") != expected_url:
        return False, "code_source_url_mismatch"
    vcs = direct_url.get("vcs_info")
    if not isinstance(vcs, dict) or vcs.get("vcs") != "git":
        return False, "direct_url_not_git"
    if vcs.get("commit_id") != revision:
        return False, "code_revision_mismatch"
    if vcs.get("requested_revision") != revision:
        return False, "requested_code_revision_mismatch"
    return True, f"verified:{revision}"


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
    parser.add_argument(
        "--require-license-acceptance",
        action="store_true",
        help="also require a complete receipt for the manifest's current license hash",
    )
    parser.add_argument(
        "--require-quality-gate",
        action="store_true",
        help="also require a 12-category/30-run quality receipt for the pinned model",
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
    license_ok, license_state = license_acceptance_status(manifest, PROJECT_ROOT)
    quality_ok, quality_state = quality_gate_status(manifest, PROJECT_ROOT)
    policy_failures = (
        (args.require_license_acceptance and not license_ok)
        or (args.require_quality_gate and not quality_ok)
    )
    if args.json:
        print(
            json.dumps(
                {
                    "model_id": manifest.get("model_id", manifest["name"]),
                    "ok": not failures and not policy_failures,
                    "verification": "size" if args.quick else "sha256",
                    "artifacts": [asdict(status) for status in statuses],
                    "license_acceptance": {
                        "required": args.require_license_acceptance,
                        "ok": license_ok,
                        "state": license_state,
                    },
                    "quality_gate": {
                        "required": args.require_quality_gate,
                        "ok": quality_ok,
                        "state": quality_state,
                    },
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
        if args.require_license_acceptance and not license_ok:
            print(f"  FAIL license acceptance: {license_state}")
        if args.require_quality_gate and not quality_ok:
            print(f"  FAIL quality gate: {quality_state}")
        if failures or policy_failures:
            print(
                f"FAILED: {len(failures)} artifact(s) require recovery; "
                f"policy checks {'failed' if policy_failures else 'passed'}"
            )
        else:
            total = sum(status.expected_size for status in statuses)
            print(f"OK: {len(statuses)} artifacts, {total / (1024 ** 3):.2f} GiB verified")
    return 1 if failures or policy_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
