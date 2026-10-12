"""Pin the working-tree bytes that existing audio identities were built from.

A single changed byte in any of these files, including a line ending, changes
the runtime identity of audio that already exists.  These pins are evidence
about the checkout, not a lock on future work: a change to one of them is a
decision for Brian, not for whoever is running the tests.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

#: The repository root, resolved the way the other test modules resolve it.
REPO_ROOT = Path(__file__).resolve().parents[1]

_MESSAGE = (
    "These are the bytes existing runtime identities were built from; a different line ending "
    "or any edit changes them; do not update a pin without Brian's decision."
)

PINNED_SOURCES = (
    ("dubbing_service/app/audiobook/renderer.py",
     "db7e4ba3423f90dbcab8a3b7fad7362efc1f0c513cf3ded3ced655eea0d1f891"),
    ("service/qwen3tts_server.py",
     "f152a5a0f074157177f28f6d6f37dd33a9eb6cb865f29d7370fc0ae895e5e49d"),
    ("service/qwen_character_voice.py",
     "ed128bc16f048d19143719b775c806ff8c7a745cfec89652688d43a581c93b3a"),
    ("service/qwen_runtime_identity.py",
     "413ead27b31170ea3ad7bca676131bcb7a87dff1762b2265eb60a0aa44351d42"),
    ("service/faster_qwen3tts_base_server.py",
     "3a021419a7719e7dc43c1ef3ca027fd1f19814a5d4a51246bd429576a79a262b"),
    ("service/faster_qwen3tts_base_adapter.py",
     "20d5547dee16809c289b058fa1e82fb158989032057cad7f3b41f97737162615"),
    ("dubbing_service/app/providers/tts.py",
     "03a673cdfd49125e9e09a50cac0a671b5e114b1412a05c306720a61ab0f80d5f"),
    ("dubbing_service/app/audiobook/reference_transcripts.py",
     "fea431f01fb06ef19679c1f0dc5cb66359a7bca071dc545bc933283915231cb6"),
    ("dubbing_service/app/audiobook/reference_quality.py",
     "d319422e6452b7fb7b90b2b0ddb4b6d4a8588e1b378e2b773d5fe1ee8ec25b46"),
    ("dubbing_service/app/audiobook/reference_source_qc.py",
     "07b90df2984732b357561ef49df92b6dd53ef5f497110477ef780ea77794d8c5"),
    ("dubbing_service/app/audiobook/reference_preparation.py",
     "fc70a953e873ddc8a180b2f5a91af78c67ef277fe8eec0fafa34fbc0caf10908"),
    ("dubbing_service/app/audiobook/reference_separation.py",
     "ea119777376c9f6adce64ebca7bbe60465bd6cc25072720ce47c5ac55b25b214"),
    ("dubbing_service/app/audiobook/reference_delivery_fit.py",
     "63038dad48da7ff3e87f34dae26fbfc1804663932b09cb648524b6e309082ffc"),
)

OPTIONAL_PINNED_SOURCES = (
    ("environments/qwen3tts/uv.lock",
     "a5dd7d14609cd87de9cdc4bfa492ce6ff257458c87d45d0572e97277c9769eab"),
    ("models/manifests/qwen3_tts_base.json",
     "4089876d4a5d9966571d7c354925b64a6656eb2ab02631823fe15b94b2c67408"),
)

if os.environ.get("AAS_FROZEN_SOURCE_PINS", "").strip().casefold() == "off":
    pytest.skip("frozen source pins are switched off by AAS_FROZEN_SOURCE_PINS=off",
                allow_module_level=True)


def _digest(relative_path: str) -> str:
    return hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()


@pytest.mark.parametrize(("relative_path", "expected"), PINNED_SOURCES,
                         ids=[path for path, _ in PINNED_SOURCES])
def test_pinned_source_bytes_are_unchanged(relative_path: str, expected: str) -> None:
    path = REPO_ROOT / relative_path
    assert path.is_file(), f"{relative_path} is missing. {_MESSAGE}"
    assert _digest(relative_path) == expected, f"{relative_path} changed. {_MESSAGE}"


@pytest.mark.parametrize(("relative_path", "expected"), OPTIONAL_PINNED_SOURCES,
                         ids=[path for path, _ in OPTIONAL_PINNED_SOURCES])
def test_optional_pinned_source_bytes_are_unchanged(relative_path: str, expected: str) -> None:
    path = REPO_ROOT / relative_path
    if not path.is_file():
        pytest.skip(f"{relative_path} is not present in this checkout")
    assert _digest(relative_path) == expected, f"{relative_path} changed. {_MESSAGE}"
