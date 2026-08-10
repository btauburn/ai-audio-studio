from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from dubbing_service.app.routers import qwen3tts as qwen3tts_router
from dubbing_service.app.core.audio_path_policy import (
    AudioFileNotFoundError,
    InvalidAudioPathError,
    managed_audio_roots,
    resolve_existing_audio_file,
    resolve_existing_audio_name,
)


def _write_audio(path: Path, content: bytes = b"RIFF-test") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_resolve_existing_audio_accepts_managed_relative_and_absolute_paths(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    root = repo / "dubbing_projects"
    audio = _write_audio(root / "story" / "roles" / "narrator.WAV")

    relative = resolve_existing_audio_file(
        "dubbing_projects/story/roles/narrator.WAV",
        roots=(root,),
        relative_bases=(repo,),
    )
    absolute = resolve_existing_audio_file(
        str(audio),
        roots=(root,),
        relative_bases=(repo,),
    )

    assert relative == audio.resolve()
    assert absolute == audio.resolve()


@pytest.mark.parametrize(
    "value",
    [
        "../outside.wav",
        "project/../../outside.wav",
        "project/reference.txt",
        "project/reference.wav/",
        "project//reference.wav",
    ],
)
def test_resolve_existing_audio_rejects_traversal_and_invalid_shapes(
    tmp_path: Path,
    value: str,
) -> None:
    root = tmp_path / "managed"
    root.mkdir()

    with pytest.raises(InvalidAudioPathError):
        resolve_existing_audio_file(
            value,
            roots=(root,),
            relative_bases=(tmp_path,),
        )


def test_resolve_existing_audio_rejects_absolute_path_outside_allowed_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "managed"
    root.mkdir()
    outside = _write_audio(tmp_path / "outside.wav")

    with pytest.raises(InvalidAudioPathError, match="outside allowed roots"):
        resolve_existing_audio_file(str(outside), roots=(root,))


def test_resolve_existing_audio_reports_missing_managed_file(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    root.mkdir()

    with pytest.raises(AudioFileNotFoundError):
        resolve_existing_audio_file(
            str(root / "missing.wav"),
            roots=(root,),
        )


def test_resolve_existing_audio_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "managed"
    root.mkdir()
    outside = _write_audio(tmp_path / "outside.wav")
    link = root / "linked.wav"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(InvalidAudioPathError, match="link escapes"):
        resolve_existing_audio_file(str(link), roots=(root,))


def test_resolve_existing_audio_rejects_simulated_reparse_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "managed"
    link = _write_audio(root / "linked.wav")
    outside = _write_audio(tmp_path / "outside.wav")
    outside_resolved = outside.resolve()
    original_resolve = Path.resolve

    def _resolve(path: Path, strict: bool = False) -> Path:
        if path == link:
            return outside_resolved
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", _resolve)

    with pytest.raises(InvalidAudioPathError, match="link escapes"):
        resolve_existing_audio_file(str(link), roots=(root,))


def test_resolve_existing_audio_name_only_accepts_direct_audio_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outputs"
    audio = _write_audio(root / "result.wav")

    assert resolve_existing_audio_name(root, "result.wav") == audio.resolve()
    with pytest.raises(InvalidAudioPathError):
        resolve_existing_audio_name(root, "../result.wav")
    with pytest.raises(InvalidAudioPathError):
        resolve_existing_audio_name(root, str(audio))
    with pytest.raises(InvalidAudioPathError):
        resolve_existing_audio_name(root, "result.json")


def test_managed_audio_roots_include_external_project_and_explicit_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    projects = tmp_path / "project-data"
    extra = tmp_path / "trusted-references"
    monkeypatch.setenv("DUBBING_PROJECTS_ROOT", str(projects))
    monkeypatch.setenv("DUBBING_TTS_ALLOWED_AUDIO_ROOTS", str(extra))

    roots = managed_audio_roots(repo)

    assert projects.resolve() in roots
    assert extra.resolve() in roots
    assert (repo / "dubbing_service" / "voices").resolve() in roots
    assert (repo / "VioceTimbre").resolve() in roots
    assert (repo / "examples").resolve() in roots
    assert (repo / "cosyvoice" / "asset").resolve() in roots


def test_managed_audio_roots_honor_file_config_projects_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    configured = tmp_path / "configured-projects"
    config_path = repo / "dubbing_service" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        '{"projects_root": "' + str(configured).replace("\\", "\\\\") + '"}',
        encoding="utf-8",
    )
    monkeypatch.delenv("DUBBING_PROJECTS_ROOT", raising=False)

    roots = managed_audio_roots(repo)

    assert configured.resolve() in roots


def test_qwen_audio_proxy_only_serves_managed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _write_audio(tmp_path / "outputs" / "preview.wav")
    _write_audio(tmp_path / "outside.wav")
    monkeypatch.setattr(qwen3tts_router, "_repo_root", lambda: tmp_path)

    response = qwen3tts_router.api_get_audio("outputs/preview.wav")

    assert Path(response.path) == output.resolve()
    assert qwen3tts_router._normalize_audio_path(str(output)) == "outputs/preview.wav"


@pytest.mark.parametrize(
    ("value", "status_code"),
    [
        ("../outside.wav", 400),
        ("outside.txt", 400),
        ("outputs/missing.wav", 404),
    ],
)
def test_qwen_audio_proxy_rejects_unsafe_or_missing_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    status_code: int,
) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "service" / "outputs").mkdir(parents=True)
    monkeypatch.setattr(qwen3tts_router, "_repo_root", lambda: tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        qwen3tts_router.api_get_audio(value)

    assert exc_info.value.status_code == status_code


def test_worker_output_names_do_not_include_user_text() -> None:
    repo = Path(__file__).resolve().parents[1]
    workers = (
        repo / "service" / "server.py",
        repo / "service" / "cosyvoice_server.py",
        repo / "service" / "qwen3tts_server.py",
        repo / "service" / "qwen3tts_customvoice_server.py",
    )

    for worker in workers:
        source = worker.read_text(encoding="utf-8")
        assert "_sanitize_filename_component" not in source
        assert "resolve_existing_audio_name" in source
