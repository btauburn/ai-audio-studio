"""A blocking QA refusal must survive every later write, reload and restart.

Reproduction for the durability defect.  ``promote_take`` correctly refuses a
Take whose blocking Tier-1 QA failed, but the refusal only lives in memory: the
rejected Take still occupies the active slot, and ``ensure_take_state`` refills
``selected_take_id`` from that slot.  Because that function runs on *every*
project load, the next unrelated write persists the rejected audio as the
selected result — the refusal is not durable.

Nothing here is mocked: real storage, real ``ensure_take_state``, real
``commit_project_state``, and a real second interpreter for every reload.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import persistence, pipeline, renderer, takes
from dubbing_service.app.storage.segments import load_segments, save_segments
from tests.audiobook_helpers import build_project, build_segments, use_projects_root

REPO_ROOT = Path(persistence.__file__).resolve().parents[3]

# Long enough that a one-millisecond-per-character render is unambiguously
# truncated, and long enough that a normal render passes the same check.
TEXT = (
    "The ledger stayed open on the desk for three days before anyone dared to "
    "read the last entry."
)
OTHER_TEXT = "She counted the stairwells twice before the lights came back on in the hallway."

RELOAD_CHILD = '''
import json
import sys

from dubbing_service.app.storage.segments import load_segments

payload = {}
for segment in load_segments(sys.argv[1]):
    payload[segment.id] = {
        "selected_take_id": segment.selected_take_id,
        "active_slot": segment.active_version,
        "slots": {name: state.take_id for name, state in segment.take_slots.items()},
        "takes": [
            {
                "id": take.id,
                "missing": take.missing,
                "stale": take.stale,
                "archived": take.archived_at is not None,
                "qa_passed": None if take.qa_report is None else take.qa_report.passed,
                "blocking": (
                    []
                    if take.qa_report is None
                    else [item.check_id for item in take.qa_report.blocking_failures()]
                ),
            }
            for take in segment.takes
        ],
    }
print(json.dumps(payload))
'''


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, [TEXT, OTHER_TEXT])
    return root, project, segments


def _child_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH") or ""
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    env["PYTHONUTF8"] = "1"
    return env


def _reload(projects_root: Path, tmp_path: Path, project_id: str) -> dict:
    """Reload the project in a fresh interpreter and report what it selected."""

    script = Path(tmp_path) / "reload_segments.py"
    script.write_text(RELOAD_CHILD.strip() + "\n", encoding="utf-8")
    env = _child_env()
    env["DUBBING_PROJECTS_ROOT"] = str(projects_root)
    completed = subprocess.run(
        [sys.executable, str(script), project_id],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _raw_selection(projects_root: Path, project_id: str) -> dict:
    """What the commit on disk recorded — no service code involved."""

    document = json.loads(
        (Path(projects_root) / project_id / "segments.json").read_text(encoding="utf-8")
    )
    segments = document["segments"] if isinstance(document, dict) else document
    return {segment["id"]: segment.get("selected_take_id") for segment in segments}


def _fixture(project_id: str, *, ms_per_char: float | None = None):
    kwargs = {} if ms_per_char is None else {"ms_per_char": ms_per_char}
    return renderer.FixtureRenderer(
        project_root=pipeline.project_root(project_id), renderer_id="fixture", **kwargs
    )


def _render(project_id: str, segment_id: str, *, ms_per_char: float | None = None):
    return pipeline.render_utterances(
        project_id, [segment_id], renderer=_fixture(project_id, ms_per_char=ms_per_char)
    )[0]


def test_blocking_qa_refusal_survives_reload_and_unrelated_write(studio, tmp_path):
    projects_root, project, segments = studio
    target, other = segments[0], segments[1]

    record = _render(project.id, target.id, ms_per_char=1.0)

    # The render itself succeeds: the Take exists and its audio is on disk.
    assert record.success is True
    assert record.qa_passed is False
    assert record.selected is False
    assert record.error and "coverage_plausible" in record.error

    segments = load_segments(project.id)
    rejected = takes.find_take(segments[0], record.take_id)
    assert rejected.qa_report is not None
    assert rejected.qa_report.passed is False
    assert [item.check_id for item in rejected.qa_report.blocking_failures()] == [
        "coverage_plausible"
    ]
    assert (pipeline.project_root(project.id) / rejected.audio_file).is_file()
    assert rejected.archived_at is None

    # Refused in memory and refused on disk: the candidate occupies the active
    # slot, but nothing has selected it yet.
    assert segments[0].selected_take_id is None
    assert _raw_selection(projects_root, project.id)[target.id] is None

    # (A) A plain reload must not turn the refusal into a selection.
    first = _reload(projects_root, tmp_path, project.id)
    assert first[target.id]["selected_take_id"] is None, (
        "ensure_take_state selected a Take that blocking QA refused"
    )
    assert first[target.id]["slots"]["a"] == record.take_id  # candidate retained
    assert first[target.id]["takes"][0]["blocking"] == ["coverage_plausible"]

    # (B) An unrelated committing write must not launder the refusal either.
    other_record = _render(project.id, other.id)
    assert other_record.qa_passed is True
    assert other_record.selected is True

    second = _reload(projects_root, tmp_path, project.id)
    assert second[target.id]["selected_take_id"] is None, (
        "an unrelated write persisted the QA-rejected Take as the selection"
    )
    assert second[target.id]["takes"][0]["qa_passed"] is False
    assert _raw_selection(projects_root, project.id)[target.id] is None

    # And the aggregate still refuses to claim a release.
    assert pipeline.release_report(project.id).ready is False
    status = pipeline.status_report(project.id)
    assert status["qa_blocked"] == 1
    blocked = {item["segment_id"]: item for item in status["blocked_utterances"]}[
        target.id
    ]
    assert blocked["selected_take_id"] is None
    assert record.take_id in blocked["candidate_take_ids"]
    assert blocked["blocking_qa"]["checks"] == ["coverage_plausible"]


def test_good_selected_take_is_not_replaced_by_a_bad_candidate(studio, tmp_path):
    projects_root, project, segments = studio
    target = segments[0]

    for segment in segments:
        record = _render(project.id, segment.id)
        assert record.qa_passed is True and record.selected is True

    bad = _render(project.id, target.id, ms_per_char=1.0)
    assert bad.qa_passed is False and bad.selected is False

    state = _reload(projects_root, tmp_path, project.id)
    assert state[target.id]["selected_take_id"] == state[target.id]["takes"][0]["id"]
    assert state[target.id]["selected_take_id"] != bad.take_id
    assert state[target.id]["slots"]["a"] == bad.take_id
    assert state[target.id]["takes"][1]["qa_passed"] is False

    segments = load_segments(project.id)
    good_id = segments[0].selected_take_id
    assert good_id and good_id != bad.take_id
    good = takes.find_take(segments[0], good_id)
    assert (pipeline.project_root(project.id) / good.audio_file).is_file()
    assert (pipeline.project_root(project.id) / takes.find_take(segments[0], bad.take_id).audio_file).is_file()

    assert pipeline.release_report(project.id).ready is True
    status = pipeline.status_report(project.id)
    assert status["qa_blocked"] == 0
    assert status["release_ready"] is True


def test_explicit_selection_refuses_a_qa_blocked_candidate(studio):
    _, project, segments = studio
    record = _render(project.id, segments[0].id, ms_per_char=1.0)

    segment = load_segments(project.id)[0]
    with pytest.raises(takes.PromotionError):
        takes.promote_take(segment, record.take_id)
    assert segment.selected_take_id is None


def test_refusal_survives_an_unrelated_storage_write(studio, tmp_path):
    """The refusal must not be undone by a write that never mentions it."""

    projects_root, project, segments = studio
    target = segments[0]

    record = _render(project.id, target.id, ms_per_char=1.0)
    assert record.selected is False

    # One unrelated edit, committed through the ordinary storage path.
    reloaded = load_segments(project.id)
    reloaded[1].text = OTHER_TEXT + " Then the corridor went quiet again."
    save_segments(project.id, reloaded)

    assert _raw_selection(projects_root, project.id)[target.id] is None
    state = _reload(projects_root, tmp_path, project.id)
    assert state[target.id]["selected_take_id"] is None
    assert state[target.id]["takes"][0]["blocking"] == ["coverage_plausible"]
    assert pipeline.release_report(project.id).ready is False