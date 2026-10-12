"""CPU-only contracts for the runtime delivery-capability probe.

Every case builds a fake repository tree under ``tmp_path`` and reads it as
text and bytes.  No model is imported, no environment is entered and the real
``environments/`` tree is only read (never written) by the one round-trip case.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dubbing_service.app.audiobook.provider_capabilities import (
    capability_record,
    performance_capabilities,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROFILE = "faster-base-032"
EXPERIMENTAL_PROFILE = "faster-base-032-direction-x1"
FASTER_VENV = REPO_ROOT / "environments" / "faster_qwen_base032" / ".venv"

SITE_PACKAGES = Path("environments/faster_qwen_base032/.venv/Lib/site-packages")
MODEL_PY = SITE_PACKAGES / "faster_qwen3_tts" / "model.py"
GENERATE_PY = SITE_PACKAGES / "faster_qwen3_tts" / "generate.py"
DIST_INFO = SITE_PACKAGES / "faster_qwen3_tts-0.3.2.dist-info" / "METADATA"
BASE_ADAPTER = Path("service/faster_qwen3tts_base_adapter.py")
DIRECTION_ADAPTER = Path("service/faster_qwen3tts_base_direction_adapter.py")

SIGNATURE_WITH_INSTRUCT = (
    "class Model:\n"
    "    def _prepare_generation(self, text: str, ref_audio=None,\n"
    "                            instruct: Optional[str] = None):\n"
    "        return text, instruct\n"
)
SIGNATURE_WITHOUT_INSTRUCT = (
    "class Model:\n"
    "    def _prepare_generation(self, text: str, ref_audio=None):\n"
    "        return text\n"
)

CONTROL_NAMES = ("native_instruction", "emotion_label", "intensity", "emphasis", "role_delivery",
                 "pace", "volume", "numeric_rate", "seed")


def write(root: Path, relative_path: Path, text: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def build_tree(root: Path, *, signature: str | None = SIGNATURE_WITH_INSTRUCT,
               adapter: str | None = "base adapter without the directive path\n",
               direction_adapter: str | None = None,
               version: str = "0.3.2") -> Path:
    if signature is not None:
        write(root, MODEL_PY, signature)
        write(root, GENERATE_PY, "# generation entry point\n")
        write(root, DIST_INFO, "Metadata-Version: 2.1\nName: faster-qwen3-tts\n"
                               f"Version: {version}\n")
    if adapter is not None:
        write(root, BASE_ADAPTER, adapter)
    if direction_adapter is not None:
        write(root, DIRECTION_ADAPTER, direction_adapter)
    return root


def reference_identity(**overrides) -> dict:
    payload = {"mechanism": "qwen_reference"}
    payload.update(overrides)
    return payload


def native_instruction(root: Path, profile: str = DEFAULT_PROFILE, identity: dict | None = None) -> dict:
    return performance_capabilities(profile, identity, repo_root=root)["controls"]["native_instruction"]


def test_default_profile_reports_the_library_gap_truthfully(tmp_path: Path) -> None:
    root = build_tree(tmp_path)
    record = performance_capabilities(DEFAULT_PROFILE, reference_identity(), repo_root=root)
    assert record["schema_version"] == "performance-capability-v1"
    assert record["backend"] == "qwen3tts"
    assert record["identity_mechanism"] == "qwen_reference"
    assert record["conditioning_mode"] == "icl"
    assert record["evidence"]["faster_version"] == "0.3.2"
    assert record["evidence"]["library_accepts_instruct"] is True
    assert record["evidence"]["adapter_forwards_instruct"] is False
    assert record["evidence"]["faster_model_sha256"] == hashlib.sha256(
        (root / MODEL_PY).read_bytes()).hexdigest()
    control = record["controls"]["native_instruction"]
    assert control["state"] == "available_not_forwarded"
    assert control["stage"] == "native"
    assert control["review_copy_eligible"] is True
    assert "does not forward" in control["reason"]


def test_default_profile_without_instruct_in_the_signature_is_unsupported(tmp_path: Path) -> None:
    root = build_tree(tmp_path, signature=SIGNATURE_WITHOUT_INSTRUCT)
    control = native_instruction(root)
    assert control["state"] == "unsupported"
    assert control["review_copy_eligible"] is False


def test_missing_faster_environment_is_unsupported_and_never_raises(tmp_path: Path) -> None:
    root = build_tree(tmp_path, signature=None)
    record = performance_capabilities(DEFAULT_PROFILE, reference_identity(), repo_root=root)
    assert record["evidence"]["faster_version"] is None
    assert record["evidence"]["faster_model_sha256"] is None
    assert record["evidence"]["library_accepts_instruct"] is False
    assert record["controls"]["native_instruction"]["state"] == "unsupported"


def test_adapter_that_already_forwards_instruct_is_not_the_direction_path(tmp_path: Path) -> None:
    root = build_tree(tmp_path, adapter="def prepare(text, instruct=None):\n    return instruct\n")
    record = performance_capabilities(DEFAULT_PROFILE, reference_identity(), repo_root=root)
    assert record["evidence"]["adapter_forwards_instruct"] is True
    assert record["controls"]["native_instruction"]["state"] == "unsupported"


def test_experimental_profile_needs_the_direction_adapter(tmp_path: Path) -> None:
    root = build_tree(tmp_path)
    assert native_instruction(root, EXPERIMENTAL_PROFILE, reference_identity())["state"] == "unsupported"


def test_experimental_profile_forwards_for_an_icl_reference_identity(tmp_path: Path) -> None:
    root = build_tree(tmp_path, direction_adapter="# experimental direction adapter\n")
    record = performance_capabilities(EXPERIMENTAL_PROFILE, reference_identity(), repo_root=root)
    assert record["controls"]["native_instruction"]["state"] == "forwarded_experimental"
    assert record["controls"]["native_instruction"]["review_copy_eligible"] is True
    assert record["evidence"]["direction_adapter_sha256"] == hashlib.sha256(
        (root / DIRECTION_ADAPTER).read_bytes()).hexdigest()


@pytest.mark.parametrize("identity", [None, {"mechanism": "builtin"},
                                      {"mechanism": "qwen_reference", "mode": "x_vector_only"}])
def test_experimental_profile_needs_a_qwen_reference_icl_identity(tmp_path: Path, identity) -> None:
    root = build_tree(tmp_path, direction_adapter="# experimental direction adapter\n")
    assert native_instruction(root, EXPERIMENTAL_PROFILE, identity)["state"] == "unsupported"


@pytest.mark.parametrize("profile", ["native-current", "native-base"])
def test_native_profiles_never_forward_an_instruction(tmp_path: Path, profile: str) -> None:
    root = build_tree(tmp_path, direction_adapter="# experimental direction adapter\n")
    assert native_instruction(root, profile, reference_identity())["state"] == "unsupported"


def test_unknown_profile_marks_every_control_unsupported(tmp_path: Path) -> None:
    root = build_tree(tmp_path, direction_adapter="# experimental direction adapter\n")
    controls = performance_capabilities("mystery-profile", reference_identity(), repo_root=root)["controls"]
    assert tuple(controls) == CONTROL_NAMES
    for name, control in controls.items():
        assert control["state"] == "unsupported", name
        assert control["reason"] == "unknown runtime profile"
        assert control["review_copy_eligible"] is False


@pytest.mark.parametrize("profile", [DEFAULT_PROFILE, EXPERIMENTAL_PROFILE, "native-base",
                                     "native-current"])
def test_numeric_rate_and_seed_are_never_available(tmp_path: Path, profile: str) -> None:
    root = build_tree(tmp_path, direction_adapter="# experimental direction adapter\n")
    controls = performance_capabilities(profile, reference_identity(), repo_root=root)["controls"]
    assert controls["numeric_rate"]["state"] == "unsupported"
    assert "no numeric rate parameter" in controls["numeric_rate"]["reason"]
    assert controls["seed"]["state"] == "unsupported"
    assert "derives" in controls["seed"]["reason"]


def test_an_unknown_profile_never_promises_a_rate_or_a_seed(tmp_path: Path) -> None:
    root = build_tree(tmp_path, direction_adapter="# experimental direction adapter\n")
    controls = performance_capabilities("mystery-profile", reference_identity(), repo_root=root)["controls"]
    assert controls["numeric_rate"]["state"] == "unsupported"
    assert controls["seed"]["state"] == "unsupported"


def test_postprocess_controls_need_a_reference_identity(tmp_path: Path) -> None:
    root = build_tree(tmp_path)
    available = performance_capabilities(DEFAULT_PROFILE, reference_identity(), repo_root=root)["controls"]
    assert available["pace"]["state"] == "available"
    assert available["pace"]["stage"] == "postprocess"
    assert available["pace"]["review_copy_eligible"] is False
    assert available["volume"]["state"] == "available"
    assert available["volume"]["stage"] == "postprocess"

    unsupported = performance_capabilities(DEFAULT_PROFILE, {"mechanism": "builtin"},
                                           repo_root=root)["controls"]
    assert unsupported["pace"]["state"] == "unsupported"
    assert unsupported["volume"]["state"] == "unsupported"
    assert performance_capabilities(DEFAULT_PROFILE, None, repo_root=root)["conditioning_mode"] == "not_applicable"


def test_record_hash_is_stable_and_tracks_the_adapter_bytes(tmp_path: Path) -> None:
    root = build_tree(tmp_path)
    first = performance_capabilities(DEFAULT_PROFILE, reference_identity(), repo_root=root)
    second = performance_capabilities(DEFAULT_PROFILE, reference_identity(), repo_root=root)
    assert first["sha256"] == second["sha256"]
    assert len(first["sha256"]) == 64

    hashed = {key: value for key, value in first.items() if key != "sha256"}
    assert hashlib.sha256(json.dumps(hashed, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest() == first["sha256"]

    write(root, BASE_ADAPTER, "base adapter without the directive path, but longer\n")
    third = performance_capabilities(DEFAULT_PROFILE, reference_identity(), repo_root=root)
    assert third["sha256"] != first["sha256"]


@pytest.mark.skipif(not FASTER_VENV.is_dir(), reason="faster_qwen_base032 virtualenv is not installed")
def test_real_checkout_reports_the_library_gap() -> None:
    record = performance_capabilities(DEFAULT_PROFILE, reference_identity())
    assert record["controls"]["native_instruction"]["state"] == "available_not_forwarded"


def test_existing_capability_record_is_unchanged() -> None:
    record = capability_record("qwen3tts")
    assert record["schema"] == "provider-capabilities-v1"
    assert {"performance", "mechanisms", "adapter"} <= set(record)
