"""Execute the shipped browser controller with real JS, alongside the Python suite."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_voice_lab_player_javascript_contracts():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to execute browser controller tests")
    completed = subprocess.run(
        [node, "--test", "tests/voice_lab_player.test.cjs"],
        cwd=ROOT, capture_output=True, text=True, timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_voice_lab_is_linked_into_existing_surfaces_and_uses_one_player():
    page = (ROOT / "dubbing_service/webui/voice-lab.html").read_text(encoding="utf-8")
    assert page.count("<audio ") == 1
    assert "<video" not in page
    studio = (ROOT / "dubbing_service/webui/index.html").read_text(encoding="utf-8")
    private = (ROOT / "dubbing_service/app/audiobook/delivery/player.html").read_text(encoding="utf-8")
    assert "'/web/voice-lab.html'" in studio
    assert 'href="/voice-lab"' in private
