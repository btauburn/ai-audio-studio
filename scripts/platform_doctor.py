"""Deterministic local preflight for the dubbing platform.

The startup mode is intentionally read-only and does not install dependencies
or download models.  Use ``--full`` when validating release artifacts to hash
every file listed by the model manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from .model_preflight import (
        license_acceptance_status,
        load_manifest,
        quality_gate_status,
        runtime_code_revision_status,
    )
except ImportError:  # Direct execution: python scripts/platform_doctor.py
    from model_preflight import (
        license_acceptance_status,
        load_manifest,
        quality_gate_status,
        runtime_code_revision_status,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class CheckResult:
    label: str
    ok: bool
    detail: str
    required: bool = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_command(name: str) -> CheckResult:
    executable = shutil.which(name)
    if not executable:
        return CheckResult(name, False, "not found on PATH")
    try:
        completed = subprocess.run(
            [executable, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        first_line = completed.stdout.splitlines()[0] if completed.stdout else "no version output"
        return CheckResult(name, completed.returncode == 0, first_line)
    except Exception as exc:
        return CheckResult(name, False, str(exc))


def _check_port(port: int) -> CheckResult:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        in_use = probe.connect_ex(("127.0.0.1", port)) == 0
    return CheckResult(f"port {port}", not in_use, "free" if not in_use else "already in use")


def _check_json(path: Path, *, required: bool = True) -> CheckResult:
    if not path.is_file():
        return CheckResult(str(path.relative_to(PROJECT_ROOT)), False, "missing", required)
    try:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
        return CheckResult(str(path.relative_to(PROJECT_ROOT)), True, "valid JSON", required)
    except Exception as exc:
        return CheckResult(str(path.relative_to(PROJECT_ROOT)), False, f"invalid JSON: {exc}", required)


def _check_stableaudio_runtime_revision() -> CheckResult:
    label = "Stable Audio runtime revision (optional)"
    runtime_python = (
        PROJECT_ROOT
        / "environments"
        / "stableaudio"
        / ".venv"
        / "Scripts"
        / "python.exe"
    )
    manifest_path = (
        PROJECT_ROOT
        / "models"
        / "manifests"
        / "stable_audio_3_small_sfx.json"
    )
    if not manifest_path.is_file():
        return CheckResult(label, False, "manifest not installed", required=False)
    try:
        manifest = load_manifest(manifest_path)
        ok, detail = runtime_code_revision_status(manifest, runtime_python)
    except Exception as exc:
        return CheckResult(label, False, f"unreadable: {exc}", required=False)
    return CheckResult(label, ok, detail, required=False)


def _is_nonempty_plaintext_secret(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _check_plaintext_api_credentials() -> CheckResult:
    """Fail when legacy plaintext API keys remain in JSON configuration.

    Only aggregate counts are returned.  Secret values and provider identity
    fields must never become doctor output.
    """

    config_path = PROJECT_ROOT / "dubbing_service" / "config.json"
    providers_path = PROJECT_ROOT / "dubbing_service" / "llm_providers.json"
    unreadable_files = 0
    config_fields = 0
    provider_records = 0

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            unreadable_files += 1
        else:
            config_fields = sum(
                1
                for key in ("llm_api_key", "tts_api_key")
                if _is_nonempty_plaintext_secret(config.get(key))
            )
    except (OSError, json.JSONDecodeError):
        unreadable_files += 1

    if providers_path.is_file():
        try:
            providers = json.loads(providers_path.read_text(encoding="utf-8"))
            if not isinstance(providers, list):
                unreadable_files += 1
            else:
                provider_records = sum(
                    1
                    for item in providers
                    if isinstance(item, dict)
                    and _is_nonempty_plaintext_secret(item.get("api_key"))
                )
        except (OSError, json.JSONDecodeError):
            unreadable_files += 1

    if unreadable_files:
        return CheckResult(
            "plaintext API credentials",
            False,
            f"inspection unavailable for {unreadable_files} JSON file(s)",
        )
    if config_fields or provider_records:
        return CheckResult(
            "plaintext API credentials",
            False,
            (
                f"config fields={config_fields}; "
                f"provider records={provider_records}; migrate to credential references"
            ),
        )
    return CheckResult(
        "plaintext API credentials",
        True,
        "config fields=0; provider records=0",
    )


def _iter_project_json() -> Iterable[Path]:
    root = PROJECT_ROOT / "dubbing_projects"
    if root.is_dir():
        yield from sorted(root.rglob("*.json"))


def _check_manifest(path: Path, *, full_hash: bool) -> list[CheckResult]:
    results = [_check_json(path)]
    if not results[0].ok:
        return results
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list):
        results.append(CheckResult(path.name, False, "manifest has no files list"))
        return results
    optional = data.get("optional") is True
    required = not optional
    if optional:
        present_count = sum(
            1
            for item in files
            if isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and (PROJECT_ROOT / item["path"]).is_file()
        )
        if present_count == 0:
            results.append(
                CheckResult(
                    f"optional model:{data.get('model_id', path.stem)}",
                    False,
                    "not installed; feature remains disabled",
                    required=False,
                )
            )
            return results
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            results.append(CheckResult(path.name, False, "invalid file entry", required))
            continue
        candidate = (PROJECT_ROOT / item["path"]).resolve()
        try:
            candidate.relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            results.append(
                CheckResult(item["path"], False, "path escapes project root", required)
            )
            continue
        if not candidate.is_file():
            results.append(CheckResult(item["path"], False, "missing", required))
            continue
        expected_size = item.get("size")
        if isinstance(expected_size, int) and candidate.stat().st_size != expected_size:
            results.append(
                CheckResult(
                    item["path"],
                    False,
                    f"size mismatch: {candidate.stat().st_size} != {expected_size}",
                    required,
                )
            )
            continue
        if full_hash and isinstance(item.get("sha256"), str):
            actual = _sha256(candidate)
            if actual.lower() != item["sha256"].lower():
                results.append(
                    CheckResult(
                        item["path"], False, f"SHA256 mismatch: {actual}", required
                    )
                )
                continue
        results.append(
            CheckResult(
                item["path"], True, "verified" if full_hash else "present", required
            )
        )

        source = data.get("sources", {}).get(item.get("source"), {})
        if isinstance(source, dict) and source.get("provider") in {
            "modelscope",
            "huggingface",
        }:
            continue
        recoveries = [
            recovery
            for recovery in item.get("recovery", [])
            if isinstance(recovery, dict) and recovery.get("provider") == "project_cache"
        ]
        if not recoveries:
            results.append(
                CheckResult(
                    f"recovery:{item['path']}",
                    False,
                    "non-ModelScope artifact has no verified project recovery copy",
                    required,
                )
            )
            continue
        for recovery in recoveries:
            recovery_path = recovery.get("path")
            if not isinstance(recovery_path, str) or not recovery_path:
                results.append(
                    CheckResult(
                        f"recovery:{item['path']}",
                        False,
                        "invalid recovery path",
                        required,
                    )
                )
                continue
            candidate_recovery = (PROJECT_ROOT / recovery_path).resolve()
            try:
                candidate_recovery.relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                results.append(
                    CheckResult(
                        f"recovery:{recovery_path}",
                        False,
                        "path escapes project root",
                        required,
                    )
                )
                continue
            if not candidate_recovery.is_file():
                results.append(
                    CheckResult(f"recovery:{recovery_path}", False, "missing", required)
                )
                continue
            if isinstance(expected_size, int) and candidate_recovery.stat().st_size != expected_size:
                results.append(
                    CheckResult(
                        f"recovery:{recovery_path}",
                        False,
                        f"size mismatch: {candidate_recovery.stat().st_size} != {expected_size}",
                        required,
                    )
                )
                continue
            if full_hash and isinstance(item.get("sha256"), str):
                actual_recovery = _sha256(candidate_recovery)
                if actual_recovery.lower() != item["sha256"].lower():
                    results.append(
                        CheckResult(
                            f"recovery:{recovery_path}",
                            False,
                            f"SHA256 mismatch: {actual_recovery}",
                            required,
                        )
                    )
                    continue
            results.append(
                CheckResult(
                    f"recovery:{recovery_path}",
                    True,
                    "verified" if full_hash else "present",
                    required,
                )
            )
    if isinstance(data.get("license_policy"), dict):
        try:
            license_ok, license_state = license_acceptance_status(data, PROJECT_ROOT)
            quality_ok, quality_state = quality_gate_status(data, PROJECT_ROOT)
        except Exception as exc:
            results.append(
                CheckResult(
                    f"model policy:{data.get('model_id', path.stem)}",
                    False,
                    f"unreadable: {exc}",
                    required,
                )
            )
        else:
            results.append(
                CheckResult(
                    f"license:{data.get('model_id', path.stem)}",
                    license_ok,
                    license_state,
                    required,
                )
            )
            results.append(
                CheckResult(
                    f"quality gate:{data.get('model_id', path.stem)}",
                    quality_ok,
                    quality_state,
                    required,
                )
            )
    return results


def run_checks(*, startup: bool, full_hash: bool) -> list[CheckResult]:
    results: list[CheckResult] = []
    required_paths = {
        ".venv/Scripts/python.exe": PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        "Qwen runtime": PROJECT_ROOT / "environments" / "qwen3tts" / ".venv" / "Scripts" / "python.exe",
        "CosyVoice runtime": PROJECT_ROOT / "environments" / "cosyvoice" / ".venv" / "Scripts" / "python.exe",
        "IndexTTS2 model": PROJECT_ROOT / "models" / "index_tts",
        "Qwen CustomVoice model": PROJECT_ROOT / "models" / "qwen3_tts" / "Qwen3-TTS-CustomVoice",
        "Qwen VoiceDesign model": PROJECT_ROOT / "models" / "qwen3_tts" / "Qwen3-TTS-VoiceDesign",
        "CosyVoice3 model": PROJECT_ROOT / "models" / "cosyvoice3" / "Fun-CosyVoice3-0.5B",
    }
    for label, path in required_paths.items():
        results.append(CheckResult(label, path.exists(), str(path) if path.exists() else "missing"))

    stableaudio_python = (
        PROJECT_ROOT
        / "environments"
        / "stableaudio"
        / ".venv"
        / "Scripts"
        / "python.exe"
    )
    results.append(
        CheckResult(
            "Stable Audio runtime (optional)",
            stableaudio_python.is_file(),
            str(stableaudio_python) if stableaudio_python.is_file() else "not installed; SFX generation disabled",
            required=False,
        )
    )
    results.append(_check_stableaudio_runtime_revision())

    results.extend((_check_command("ffmpeg"), _check_command("ffprobe")))
    for port in (9000, 8001, 8002, 8003, 8004, 8005):
        results.append(_check_port(port))

    results.append(_check_json(PROJECT_ROOT / "dubbing_service" / "config.json"))
    results.append(_check_json(PROJECT_ROOT / "dubbing_service" / "llm_models.json"))
    results.append(
        _check_json(
            PROJECT_ROOT / "dubbing_service" / "llm_providers.json",
            required=False,
        )
    )
    results.append(_check_plaintext_api_credentials())
    for project_json in _iter_project_json():
        results.append(_check_json(project_json))

    manifest_root = PROJECT_ROOT / "models" / "manifests"
    manifests = sorted(manifest_root.glob("*.json")) if manifest_root.is_dir() else []
    if not manifests:
        results.append(CheckResult("model manifests", False, "no manifests found"))
    else:
        for manifest in manifests:
            results.extend(_check_manifest(manifest, full_hash=full_hash))

    if startup:
        # A startup preflight checks presence and sizes from the manifest. Full
        # hashing is reserved for provisioning/release verification.
        return results
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local dubbing platform prerequisites")
    parser.add_argument("--startup", action="store_true", help="fast read-only startup checks")
    parser.add_argument("--full", action="store_true", help="hash all manifest files")
    parser.add_argument("--verbose", action="store_true", help="print every successful check")
    args = parser.parse_args()

    results = run_checks(startup=args.startup, full_hash=args.full)
    for result in results:
        if result.ok and not args.verbose:
            continue
        marker = "OK" if result.ok else ("FAIL" if result.required else "WARN")
        print(f"[{marker}] {result.label}: {result.detail}")
    failures = [item for item in results if item.required and not item.ok]
    print(f"Doctor summary: {len(results) - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
