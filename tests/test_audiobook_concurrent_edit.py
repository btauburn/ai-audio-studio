"""A render must never overwrite an edit that landed while it was running.

The race is reproduced **deterministically**: a gated renderer blocks after the
render pass has read canonical state and before it commits its result, which
lets the test commit a real user edit through the normal storage path in
between.  Ordering is fixed by events, never by ``sleep``.

The invariant under test is the ownership boundary:

    A worker may append the artifact it produced, but it may not overwrite
    newer canonical metadata it did not own.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import pytest

from dubbing_service.app.audiobook import (
    pipeline,
    renderer as renderer_module,
    takes as takes_module,
)
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.segments import (
    load_segments,
    project_segments_lock,
    save_segments,
)
from tests.audiobook_helpers import build_project, build_segments, use_projects_root

# Generous: only bounds a hang, it never participates in the ordering.
GATE_TIMEOUT = 60.0


class GatedRenderer:
    """A real ``FixtureRenderer`` that announces its start and waits for a go-ahead.

    The gate sits *inside* the render call, i.e. after the render pass has loaded
    canonical state and before it commits anything, which is exactly the window a
    concurrent user edit occupies.
    """

    def __init__(
        self,
        inner: renderer_module.FixtureRenderer,
        *,
        gate_segment_id: str,
        entered: threading.Event,
        proceed: threading.Event,
    ) -> None:
        self._inner = inner
        self.gate_segment_id = gate_segment_id
        self.entered = entered
        self.proceed = proceed
        self.renderer_id = inner.renderer_id
        self.engine_id = inner.engine_id

    def render(self, request: renderer_module.RenderRequest) -> renderer_module.RenderOutcome:
        if request.segment.id == self.gate_segment_id and not self.proceed.is_set():
            self.entered.set()
            if not self.proceed.wait(GATE_TIMEOUT):  # pragma: no cover - hang guard
                raise AssertionError("render gate was never opened")
        return self._inner.render(request)


@dataclass
class BackgroundRender:
    """Runs ``render_utterances`` in a thread and re-raises its failure."""

    thread: threading.Thread
    records: Optional[list] = None
    error: Optional[BaseException] = None

    def join(self) -> list:
        self.thread.join(timeout=GATE_TIMEOUT)
        assert not self.thread.is_alive(), "background render did not finish"
        if self.error is not None:
            raise self.error
        assert self.records is not None
        return self.records


def start_render(project_id: str, utterance_ids, renderer) -> BackgroundRender:
    run = BackgroundRender(thread=threading.Thread(target=lambda: None))

    def target() -> None:
        try:
            run.records = pipeline.render_utterances(
                project_id, utterance_ids, renderer=renderer
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced by join()
            run.error = exc

    run.thread = threading.Thread(target=target, daemon=True)
    run.thread.start()
    return run


def render_now(project_id: str, utterance_ids) -> list:
    """A complete, ungated render pass."""

    renderer = renderer_module.FixtureRenderer(
        project_root=pipeline.project_root(project_id), renderer_id="fixture"
    )
    return pipeline.render_utterances(project_id, utterance_ids, renderer=renderer)


def edit_segment(project_id: str, segment_id: str, **changes: object) -> None:
    """Commit a segment edit the way the normal storage path does: under the lock."""

    with project_segments_lock(project_id):
        segments = load_segments(project_id)
        target = next(item for item in segments if item.id == segment_id)
        for name, value in changes.items():
            setattr(target, name, value)
        save_segments(project_id, segments)


def edit_text(project_id: str, segment_id: str, text: str) -> None:
    edit_segment(project_id, segment_id, text=text)


def edit_project_metadata(project_id: str, **entries: object) -> None:
    with project_segments_lock(project_id):
        project = load_project(project_id)
        project.metadata.update(entries)
        save_project(project)


def segments_by_id(project_id: str) -> dict:
    return {item.id: item for item in load_segments(project_id)}


@pytest.fixture()
def studio(monkeypatch, tmp_path):
    root = use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    segments = build_segments(project, ["Version A text.", "Untouched segment."])
    return root, project, segments


def gate(root, project_id, segment_id, entered, proceed):
    inner = renderer_module.FixtureRenderer(
        project_root=pipeline.project_root(project_id), renderer_id="fixture"
    )
    return GatedRenderer(
        inner, gate_segment_id=segment_id, entered=entered, proceed=proceed
    )


def test_text_edit_landing_mid_render_survives_the_render_commit(studio):
    """The headline race: the render's commit must not resurrect the old text."""

    root, project, segments = studio
    project_id = project.id
    target_id = segments[0].id

    entered, proceed = threading.Event(), threading.Event()
    run = start_render(
        project_id,
        [target_id],
        gate(root, project_id, target_id, entered, proceed),
    )

    assert entered.wait(GATE_TIMEOUT), "render never reached its gate"
    edit_text(project_id, target_id, "Version B text, typed by a human.")
    proceed.set()
    records = run.join()

    reloaded = load_segments(project_id)
    reloaded_target = next(item for item in reloaded if item.id == target_id)

    # 1. The user's newer edit is canonical.
    assert reloaded_target.text == "Version B text, typed by a human."
    # 2. The render still produced its artifact, retained as history.
    assert any(take.id == records[0].take_id for take in reloaded_target.takes)
    assert records[0].candidate_created is True
    # 3. The candidate lost the race: it must not be selected, and it must say so.
    assert reloaded_target.selected_take_id != records[0].take_id
    assert records[0].selected is False
    assert records[0].stale is True


def test_unrelated_segment_edit_landing_mid_render_survives(studio):
    """Segment B's newer edit must not be clobbered by segment A's render."""

    root, project, segments = studio
    project_id = project.id
    rendered_id, other_id = segments[0].id, segments[1].id

    entered, proceed = threading.Event(), threading.Event()
    run = start_render(
        project_id,
        [rendered_id],
        gate(root, project_id, rendered_id, entered, proceed),
    )

    assert entered.wait(GATE_TIMEOUT), "render never reached its gate"
    edit_text(project_id, other_id, "An unrelated edit.")
    proceed.set()
    run.join()

    reloaded = {item.id: item for item in load_segments(project_id)}
    assert reloaded[other_id].text == "An unrelated edit."
    assert reloaded[rendered_id].text == "Version A text."


def test_stale_candidate_is_retained_as_history_and_reports_what_changed(studio):
    """A refused result is history with a diagnosis, not silence and not a deletion."""

    root, project, segments = studio
    project_id, target_id = project.id, segments[0].id

    entered, proceed = threading.Event(), threading.Event()
    run = start_render(
        project_id,
        [target_id],
        gate(root, project_id, target_id, entered, proceed),
    )
    assert entered.wait(GATE_TIMEOUT)
    edit_text(project_id, target_id, "Rewritten mid-flight.")
    proceed.set()
    records = run.join()

    candidate_id = records[0].take_id
    reloaded = segments_by_id(project_id)[target_id]
    candidate = takes_module.find_take(reloaded, candidate_id)

    assert candidate.stale is True
    assert candidate.stale_components == ["spoken_text_hash"]
    # The artifact exists and is addressable even though it was not selected.
    assert candidate.audio_file
    assert (pipeline.project_root(project_id) / candidate.audio_file).is_file()
    assert any(take.id == candidate_id for take in reloaded.takes)


def test_stale_candidate_cannot_be_selected_even_on_explicit_request(studio):
    """Staleness lives in committed state, not in one caller's return value."""

    root, project, segments = studio
    project_id, target_id = project.id, segments[0].id

    entered, proceed = threading.Event(), threading.Event()
    run = start_render(
        project_id,
        [target_id],
        gate(root, project_id, target_id, entered, proceed),
    )
    assert entered.wait(GATE_TIMEOUT)
    edit_text(project_id, target_id, "Rewritten mid-flight.")
    proceed.set()
    records = run.join()

    reloaded = segments_by_id(project_id)[target_id]
    assert reloaded.selected_take_id is None

    with pytest.raises(takes_module.PromotionError):
        takes_module.promote_take(reloaded, records[0].take_id)

    # The refusal must not have moved anything on its way out.
    assert reloaded.selected_take_id is None
    assert takes_module.find_take(reloaded, records[0].take_id).stale is True


def test_review_note_edit_mid_render_does_not_strand_the_candidate(studio):
    """A non-audio edit is not an audio dependency: the result is still good."""

    root, project, segments = studio
    project_id, target_id = project.id, segments[0].id

    entered, proceed = threading.Event(), threading.Event()
    run = start_render(
        project_id,
        [target_id],
        gate(root, project_id, target_id, entered, proceed),
    )
    assert entered.wait(GATE_TIMEOUT)
    edit_segment(project_id, target_id, review_note="Checked by a human.")
    proceed.set()
    records = run.join()

    reloaded = segments_by_id(project_id)[target_id]
    assert reloaded.review_note == "Checked by a human."
    assert records[0].stale is False
    assert records[0].selected is True
    assert reloaded.selected_take_id == records[0].take_id
    assert takes_module.find_take(reloaded, records[0].take_id).stale is False


def test_synthesis_affecting_edit_mid_render_stales_the_candidate(studio):
    """An emotion edit is an audio dependency, so the running render is out of date."""

    root, project, segments = studio
    project_id, target_id = project.id, segments[0].id

    entered, proceed = threading.Event(), threading.Event()
    run = start_render(
        project_id,
        [target_id],
        gate(root, project_id, target_id, entered, proceed),
    )
    assert entered.wait(GATE_TIMEOUT)
    edit_segment(project_id, target_id, emotion_text="whispered")
    proceed.set()
    records = run.join()

    reloaded = segments_by_id(project_id)[target_id]
    candidate = takes_module.find_take(reloaded, records[0].take_id)
    assert reloaded.emotion_text == "whispered"
    assert records[0].selected is False
    assert candidate.stale is True
    assert "emotion_text" in candidate.stale_components


def test_existing_selected_audio_is_not_replaced_by_a_stale_candidate(studio):
    """Promotion is a pointer change: the good recording is never destroyed."""

    root, project, segments = studio
    project_id, target_id = project.id, segments[0].id

    first = render_now(project_id, [target_id])
    assert first[0].selected is True
    good_take_id = first[0].take_id
    good_audio = takes_module.find_take(
        segments_by_id(project_id)[target_id], good_take_id
    ).audio_file

    entered, proceed = threading.Event(), threading.Event()
    run = start_render(
        project_id,
        [target_id],
        gate(root, project_id, target_id, entered, proceed),
    )
    assert entered.wait(GATE_TIMEOUT)
    edit_text(project_id, target_id, "Rewritten while rendering.")
    proceed.set()
    second = run.join()

    reloaded = segments_by_id(project_id)[target_id]
    assert second[0].selected is False
    assert reloaded.selected_take_id != second[0].take_id

    good = takes_module.find_take(reloaded, good_take_id)
    assert good.audio_file == good_audio
    assert (pipeline.project_root(project_id) / good_audio).is_file()


def test_rerender_after_the_edit_produces_a_selectable_take(studio):
    """The edit is recoverable: a fresh render satisfies the new requirement."""

    root, project, segments = studio
    project_id, target_id = project.id, segments[0].id

    entered, proceed = threading.Event(), threading.Event()
    run = start_render(
        project_id,
        [target_id],
        gate(root, project_id, target_id, entered, proceed),
    )
    assert entered.wait(GATE_TIMEOUT)
    edit_text(project_id, target_id, "Regenerated text.")
    proceed.set()
    lost = run.join()
    assert lost[0].selected is False

    retry = render_now(project_id, [target_id])
    assert retry[0].selected is True
    assert retry[0].stale is False

    reloaded = segments_by_id(project_id)[target_id]
    assert reloaded.text == "Regenerated text."
    assert reloaded.selected_take_id == retry[0].take_id
    # Both artifacts survive: the retry does not delete the lost candidate.
    assert {lost[0].take_id, retry[0].take_id} <= {
        take.id for take in reloaded.takes
    }


def test_project_metadata_edit_mid_render_survives_and_does_not_stale(studio):
    """Only the artifact's own segment is written, and metadata is not audio."""

    root, project, segments = studio
    project_id, target_id = project.id, segments[0].id

    entered, proceed = threading.Event(), threading.Event()
    run = start_render(
        project_id,
        [target_id],
        gate(root, project_id, target_id, entered, proceed),
    )
    assert entered.wait(GATE_TIMEOUT)
    edit_project_metadata(project_id, book_title="Renamed mid-render")
    proceed.set()
    records = run.join()

    assert load_project(project_id).metadata["book_title"] == "Renamed mid-render"
    assert records[0].selected is True
    assert records[0].stale is False


def test_a_second_worker_cannot_overwrite_the_slot_the_first_one_claimed(studio):
    """Two renders of one utterance: the loser is history, the winner keeps the slot."""

    root, project, segments = studio
    project_id, target_id = project.id, segments[0].id

    first_entered, first_go = threading.Event(), threading.Event()
    second_entered, second_go = threading.Event(), threading.Event()
    first = start_render(
        project_id,
        [target_id],
        gate(root, project_id, target_id, first_entered, first_go),
    )
    second = start_render(
        project_id,
        [target_id],
        gate(root, project_id, target_id, second_entered, second_go),
    )

    # Both workers must have read the same canonical state before either commits.
    assert first_entered.wait(GATE_TIMEOUT) and second_entered.wait(GATE_TIMEOUT)

    first_go.set()
    first_records = first.join()
    second_go.set()
    second_records = second.join()

    assert first_records[0].selected is True
    assert second_records[0].selected is False
    assert second_records[0].candidate_created is True
    assert second_records[0].error, "a refused claim must say why"

    reloaded = segments_by_id(project_id)[target_id]
    assert reloaded.take_slots["a"].take_id == first_records[0].take_id
    assert reloaded.selected_take_id == first_records[0].take_id
    assert {take.id for take in reloaded.takes} >= {
        first_records[0].take_id,
        second_records[0].take_id,
    }


def test_each_utterance_in_a_pass_merges_against_live_state(studio):
    """A multi-utterance pass is not one snapshot: later units see earlier edits."""

    root, project, segments = studio
    project_id = project.id
    first_id, second_id = segments[0].id, segments[1].id

    entered, proceed = threading.Event(), threading.Event()
    run = start_render(
        project_id,
        [first_id, second_id],
        gate(root, project_id, first_id, entered, proceed),
    )

    assert entered.wait(GATE_TIMEOUT)
    edit_text(project_id, second_id, "Edited before its own render began.")
    proceed.set()
    records = run.join()

    reloaded = segments_by_id(project_id)
    assert reloaded[second_id].text == "Edited before its own render began."
    assert reloaded[first_id].selected_take_id == records[0].take_id
    # The second unit rendered the edited text, so it is selectable, not stale.
    assert records[1].selected is True
    assert records[1].stale is False
    assert reloaded[second_id].selected_take_id == records[1].take_id
