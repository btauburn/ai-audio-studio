"""Keep every test app and maintenance lifespan on a disposable project root."""
from __future__ import annotations

import pytest

from dubbing_service.app.core.config import clear_config_caches


@pytest.fixture(autouse=True)
def isolate_projects_root(tmp_path, monkeypatch):
    """Prevent TestClient lifespan recovery from scanning or claiming live data."""
    root = tmp_path / "studio-projects"
    root.mkdir()
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(root))
    clear_config_caches()
    # Keeper-first casting must never read Brian's live generic voice shelf.
    # Patch both the storage owner and legacy imported directory aliases.
    from dubbing_service.app.storage import voices
    from dubbing_service.app.providers import tts
    from dubbing_service.app.routers import voices as voices_router, roles, segments
    library_root = tmp_path / "studio-voice-library"
    audio_root = library_root / "voices"
    audio_root.mkdir(parents=True)
    monkeypatch.setattr(voices, "_voices_file_path", lambda: library_root / "voices.json")
    for module in (voices, tts, voices_router, roles, segments):
        monkeypatch.setattr(module, "voices_audio_dir", lambda: audio_root)
    try:
        yield root
    finally:
        # Clear while the isolated environment is still in force. The
        # monkeypatch fixture then restores the caller's environment after any
        # app lifespan contexts have completed.
        clear_config_caches()
