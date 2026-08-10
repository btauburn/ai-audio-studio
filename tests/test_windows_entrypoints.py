from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OWNED_ENTRYPOINT_ROOTS = (
    PROJECT_ROOT / "service",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "tools",
    PROJECT_ROOT / "dubbing_service",
)
WINDOWS_ENTRYPOINTS = tuple(
    sorted(
        (*PROJECT_ROOT.glob("*.bat"), *PROJECT_ROOT.glob("*.cmd")),
        key=lambda path: str(path).lower(),
    )
    + sorted(
        (
            path
            for root in OWNED_ENTRYPOINT_ROOTS
            for pattern in ("*.bat", "*.cmd")
            for path in root.rglob(pattern)
        ),
        key=lambda path: str(path).lower(),
    )
)


def test_windows_batch_entrypoints_are_utf8_crlf() -> None:
    assert WINDOWS_ENTRYPOINTS
    for path in WINDOWS_ENTRYPOINTS:
        payload = path.read_bytes()
        assert not payload.startswith(b"\xef\xbb\xbf"), path
        payload.decode("utf-8")
        assert b"\n" not in payload.replace(b"\r\n", b""), path
        assert payload.count(b"\r\n") > 0, path


def test_git_attributes_preserve_windows_batch_line_endings() -> None:
    attributes = (PROJECT_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.bat text eol=crlf" in attributes
    assert "*.cmd text eol=crlf" in attributes


def test_platform_batch_has_noninteractive_preflight_mode() -> None:
    script = (PROJECT_ROOT / "start_platform.bat").read_text(encoding="utf-8")
    assert 'if /I "%~1"=="--check" (' in script
    assert 'if /I "%~1"=="--lan" set "PLATFORM_HOST=0.0.0.0"' in script
    assert '"%CD%\\tools\\platform_launcher.py" --host "%PLATFORM_HOST%"' in script


def test_mcp_batch_uses_the_locked_platform_interpreter() -> None:
    script = (PROJECT_ROOT / "service" / "start_mcp.bat").read_text(encoding="utf-8")
    assert '"%PROJECT_ROOT%\\.venv\\Scripts\\python.exe"' in script
    assert "uv run" not in script
    assert 'if /I "%~1"=="--check" (' in script


def test_legacy_environment_check_delegates_to_platform_doctor() -> None:
    script = (PROJECT_ROOT / "service" / "check_environment.py").read_text(encoding="utf-8")
    assert '"scripts" / "platform_doctor.py"' in script
    assert "uv sync" not in script
