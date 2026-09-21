"""End-to-end tests for the ``make-audiobook`` CLI boundary.

These tests exercise the CLI the way a caller does -- as a real subprocess with
its own projects root -- because the CLI's job is boundary behaviour: argument
parsing, exit codes, JSON payload shape, and the fact that every operation goes
through the on-disk authority. A subprocess also catches bugs that an in-process
call hides, such as writing project state only in memory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import adapter

ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = ROOT / "scripts" / "make_audiobook.py"

SAMPLE_TEXT = (
    "Chapter 1\n\n"
    "The lighthouse keeper counted the ships before dawn.\n\n"
    "[Ilse] You are late again.\n\n"
    "[Soren] The tide was against me.\n"
)


def _run(args, *, projects_root: Path, expect: int | None = 0, real_audiobooker: bool = False):
    """Run the CLI in a child process and return the parsed JSON payload."""
    env = dict(os.environ)
    env["DUBBING_PROJECTS_ROOT"] = str(projects_root)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if not real_audiobooker:
        env["DUBBING_AUDIOBOOKER_DISABLE"] = "1"
    else:
        env.pop("DUBBING_AUDIOBOOKER_DISABLE", None)
    completed = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    if expect is not None:
        assert completed.returncode == expect, (
            f"exit {completed.returncode} for {args}\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
    return completed


def _payload(completed: subprocess.CompletedProcess) -> dict:
    return json.loads(completed.stdout)


@pytest.fixture()
def projects_root(tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    source = tmp_path / "book.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")
    return source


@pytest.fixture()
def ingested(projects_root: Path, sample_file: Path) -> dict:
    completed = _run(
        ["ingest", "--file", str(sample_file), "--offline"], projects_root=projects_root
    )
    return _payload(completed)


def test_no_command_prints_help_and_exits_2(projects_root: Path) -> None:
    completed = _run([], projects_root=projects_root, expect=2)
    assert "ingest" in completed.stdout
    assert "benchmark" in completed.stdout


def test_script_and_package_share_one_main() -> None:
    """The checkout shim and the installed entry point must not drift apart."""
    import importlib

    from dubbing_service.app.audiobook import cli as cli_module

    assert callable(cli_module.main)
    assert callable(cli_module.build_parser)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        shim = importlib.import_module("make_audiobook")
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    assert shim.main is cli_module.main


def test_ingest_builds_a_project_from_a_file(ingested: dict, projects_root: Path) -> None:
    assert ingested["ok"] is True
    assert ingested["operation"] == "ingest"
    project_id = ingested["project_id"]
    assert (projects_root / project_id).is_dir()
    assert ingested["utterances_created"] >= 3
    assert ingested["chapters_created"] == 1
    # The offline analyzer is allowed, but only if it says so out loud.
    assert ingested["analysis_source"] == "offline_splitter"
    assert any("offline_splitter" in warning for warning in ingested["warnings"])


def test_ingest_refuses_an_unknown_project_id_without_creating_one(
    projects_root: Path, sample_file: Path
) -> None:
    """A caller that names a project must never be handed a different one."""

    completed = _run(
        [
            "ingest",
            "--file",
            str(sample_file),
            "--project-id",
            "no_such_project",
            "--offline",
        ],
        projects_root=projects_root,
        expect=1,
    )
    payload = _payload(completed)
    assert payload["ok"] is False
    assert "no_such_project" in payload["error"]
    assert list(projects_root.glob("*/project.json")) == []


@pytest.mark.skipif(
    not adapter.availability().available, reason="audiobooker is not provisioned"
)
def test_ingest_uses_the_real_analyzer_when_it_is_installed(
    projects_root: Path, sample_file: Path
) -> None:
    completed = _run(
        ["ingest", "--file", str(sample_file)],
        projects_root=projects_root,
        real_audiobooker=True,
    )
    payload = _payload(completed)
    assert payload["ok"] is True
    assert payload["analysis_source"] == "audiobooker"
    assert not any("offline_splitter" in warning for warning in payload["warnings"])
    assert payload["utterances_created"] >= 3


def test_render_reports_take_ids_and_success_counts(
    ingested: dict, projects_root: Path
) -> None:
    completed = _run(
        ["render", "--project-id", ingested["project_id"], "--renderer", "fixture"],
        projects_root=projects_root,
    )
    payload = _payload(completed)
    assert payload["ok"] is True
    assert payload["renderer"] == "fixture"
    assert payload["succeeded"] == ingested["utterances_created"]
    assert payload["rendered"] == ingested["utterances_created"]
    assert payload["failed"] == []
    takes = {record["segment_id"]: record for record in payload["takes"]}
    assert len(takes) == ingested["utterances_created"]
    for record in takes.values():
        assert record["take_id"]
        assert record["qa_passed"] is True
        assert record["selected"] is True
        assert record["duration_ms"] > 0
        assert record["stale_components"] == []
    assert payload["status"]["coverage"]["ready"] is True


def test_status_and_export_never_leak_a_traceback_for_an_unknown_project(
    projects_root: Path,
) -> None:
    """Every command must answer with JSON, never with a Python traceback."""

    for args in (
        ["status", "--project-id", "nope"],
        ["export", "--project-id", "nope"],
        ["render", "--project-id", "nope"],
    ):
        completed = _run(args, projects_root=projects_root, expect=1)
        payload = _payload(completed)
        assert payload["ok"] is False
        assert payload["error"]
        assert "Traceback" not in completed.stdout


def test_status_and_export_agree_on_release_readiness(
    ingested: dict, projects_root: Path, tmp_path: Path
) -> None:
    project_id = ingested["project_id"]
    _run(["render", "--project-id", project_id], projects_root=projects_root)

    status = _payload(_run(["status", "--project-id", project_id], projects_root=projects_root))
    assert status["project"]["coverage"]["ready"] is True
    assert status["project"]["release_ready"] is True
    assert status["project"]["stale_utterances"] == []

    manifest = tmp_path / "release.json"
    export = _payload(
        _run(
            ["export", "--project-id", project_id, "--output", str(manifest)],
            projects_root=projects_root,
        )
    )
    assert export["ready"] is True
    assert export["coverage"]["ready"] is True
    assert export["coverage"]["missing"] == 0
    assert export["written_to"] == str(manifest)
    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert written["ready"] is True
    assert written["coverage"]["issues"] == []


def test_verify_recovers_and_confirms_a_clean_state(
    ingested: dict, projects_root: Path
) -> None:
    project_id = ingested["project_id"]
    _run(["render", "--project-id", project_id], projects_root=projects_root)
    payload = _payload(
        _run(["verify", "--project-id", project_id, "--recover"], projects_root=projects_root)
    )
    assert payload["ok"] is True
    assert payload["problems"] == []
    assert payload["pending_markers"] == []
    assert payload["recovery"]["ok"] is True


def test_verify_reports_a_missing_project_as_a_problem(projects_root: Path) -> None:
    completed = _run(["verify", "--project-id", "does_not_exist"], projects_root=projects_root, expect=1)
    payload = _payload(completed)
    assert payload["ok"] is False
    assert payload["problems"]


def test_select_rejects_an_unknown_take_with_a_clean_error(
    ingested: dict, projects_root: Path
) -> None:
    project_id = ingested["project_id"]
    render = _payload(
        _run(["render", "--project-id", project_id], projects_root=projects_root)
    )
    segment_id = render["takes"][0]["segment_id"]
    completed = _run(
        [
            "select",
            "--project-id",
            project_id,
            "--segment-id",
            segment_id,
            "--take-id",
            "take_missing",
        ],
        projects_root=projects_root,
        expect=1,
    )
    payload = _payload(completed)
    assert payload["ok"] is False
    assert payload["operation"] == "select"
    assert "take_missing" in payload["error"]
    assert "Traceback" not in completed.stderr


def test_select_accepts_a_real_take(ingested: dict, projects_root: Path) -> None:
    project_id = ingested["project_id"]
    render = _payload(
        _run(["render", "--project-id", project_id], projects_root=projects_root)
    )
    record = render["takes"][0]
    payload = _payload(
        _run(
            [
                "select",
                "--project-id",
                project_id,
                "--segment-id",
                record["segment_id"],
                "--take-id",
                record["take_id"],
            ],
            projects_root=projects_root,
        )
    )
    assert payload["ok"] is True
    assert payload["operation"] == "select_take"
    assert payload["segment_id"] == record["segment_id"]
    assert payload["take_id"] == record["take_id"]


def test_cli_writes_no_second_canonical_database(
    ingested: dict, projects_root: Path
) -> None:
    project_id = ingested["project_id"]
    _run(["render", "--project-id", project_id], projects_root=projects_root)
    suffixes = {path.suffix for path in (projects_root / project_id).rglob("*")}
    assert ".db" not in suffixes
    assert ".sqlite" not in suffixes
    assert ".sqlite3" not in suffixes


def test_backend_render_never_crashes_the_boundary(
    ingested: dict, projects_root: Path
) -> None:
    """A missing engine runtime must produce JSON, never a traceback.

    Whether the production engine is installed is an environment fact, so the
    test asserts the boundary contract (JSON + exit code) rather than
    success or failure of the engine itself.
    """

    completed = _run(
        [
            "render",
            "--project-id",
            ingested["project_id"],
            "--renderer",
            "backend",
            "--backend",
            "qwen3tts",
        ],
        projects_root=projects_root,
        expect=None,
    )
    assert completed.returncode in (0, 1)
    payload = _payload(completed)
    assert "Traceback" not in completed.stderr
    assert payload["requested_renderer"] == "backend"
    if completed.returncode == 1:
        assert payload["ok"] is False
        assert payload["operation"] == "render"
        assert payload["error"]
        for record in payload.get("failed", []):
            assert record["segment_id"]
            assert record["error"]
    else:
        assert payload["ok"] is True


def test_benchmark_run_is_reproducible_and_covered(projects_root: Path) -> None:
    payload = _payload(
        _run(["benchmark", "--render", "--renderer", "fixture"], projects_root=projects_root)
    )
    assert payload["ok"] is True
    assert len(payload["cases"]) == 13
    assert payload["utterances"] == 36
    assert payload["rendered"] == 36
    assert payload["corpus_fingerprint"] == (
        "5a5624a739be056cf5c841feca0ccce87d1da3f274983a1f329c10f48f840c76"
    )
    report = payload["report"]
    assert report["coverage"]["ready"] is True
    assert report["corpus_fingerprint"] == payload["corpus_fingerprint"]
    # A fixture run must never be presented as hardware evidence.
    blob = json.dumps(payload).lower()
    for claim in ("rtx", "cuda", "vram", "latency_ms"):
        assert claim not in blob
