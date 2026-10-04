"""The Voice Lab stylesheet carries its textures inside itself; every one it uses must be there and be a real image."""
from __future__ import annotations

import base64
import pathlib
import re

CSS = pathlib.Path(__file__).resolve().parents[1] / "dubbing_service" / "webui" / "voice-lab.css"


def _css():
    return CSS.read_text("utf-8")


def _defined():
    block = re.search(r"/\* textures:begin.*?/\* textures:end \*/", _css(), re.S)
    assert block, "the generated textures block is missing; run tools/voice_lab_textures.py"
    return dict(re.findall(r"--((?:fluid-[a-z]+)|(?:wisp-[a-z]+)|canvas-texture):url\(\"data:image/[a-z]+;base64,([A-Za-z0-9+/=]+)\"\)", block.group(0)))


def test_every_texture_the_page_uses_is_defined():
    used = set(re.findall(r"var\(--((?:fluid-[a-z]+)|(?:wisp-[a-z]+)|canvas-texture)\)", _css()))
    assert used, "the buttons and the canvas should use the generated textures"
    assert used <= set(_defined()), f"used but not generated: {sorted(used - set(_defined()))}"


def test_each_texture_is_a_real_image_of_the_expected_kind():
    for name, payload in _defined().items():
        data = base64.b64decode(payload)
        if name == "canvas-texture":
            assert data[:3] == b"\xff\xd8\xff", "the page background tile is a JPEG"
        else:
            assert data[:4] == b"RIFF" and data[8:12] == b"WEBP", f"{name} is a WebP (it needs transparency)"
        assert 4_000 < len(data) < 150_000, f"{name} is an unexpected size: {len(data)} bytes"


def test_every_button_family_has_its_liquid_and_its_wisps():
    families = ("aurora", "pearl", "ember", "mint")
    assert {f"fluid-{n}" for n in families} | {f"wisp-{n}" for n in families} <= set(_defined())
