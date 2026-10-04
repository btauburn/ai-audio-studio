"""Recovery and approval-boundary checks against isolated canonical projects."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.test_casting_review import world, prepare, ready_candidate
from dubbing_service.app.audiobook import casting_review as casting, voice_lab
from dubbing_service.app.storage.projects import load_project, save_project
from dubbing_service.app.storage.jobs import mark_job_active, get_job


def test_simultaneous_dispatch_ticks_claim_exactly_one_round(world):
    _, project_id = world
    prepare(world)
    scheduled = []
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: casting.maintenance_tick(
            project_ids=[project_id], schedule=lambda *args: scheduled.append(args)), range(2)))
    assert len(scheduled) == 1
    assert sum(len(result) for result in results) == 1
    mark_job_active(project_id, scheduled[0][2], False)


def test_resolved_round_stops_remaining_work_before_synthesis(world, monkeypatch):
    _, project_id = world
    prepare(world)
    body = ready_candidate(world)
    casting.feedback(project_id, {**body, "stars": 5, "action": "use"})
    project = load_project(project_id)
    intent = project.voice_lab.casting["roles"]["role-a"]["round"]["intent_id"]
    from dubbing_service.app.audiobook import voice_lab_audition
    monkeypatch.setattr(voice_lab_audition, "audition", lambda *a, **kw: pytest.fail("resolved round must not synthesize"))
    casting._round_worker(project_id, intent)
    project = load_project(project_id)
    assert project.voice_lab.casting["roles"]["role-a"]["round"]["status"] == "resolved"
    assert project.voice_lab.casting["intents"][intent]["status"] == "complete"
    assert get_job(project_id, intent).status.value == "succeeded"


def test_legacy_assignment_cannot_bypass_casting_approval(world):
    _, project_id = world
    prepare(world)
    body = ready_candidate(world)
    before = load_project(project_id).roles[0].model_dump()
    with pytest.raises(voice_lab.VoiceLabError, match="casting inbox"):
        voice_lab.assign(project_id, body["key"], "role-a", None)
    assert load_project(project_id).roles[0].model_dump() == before


def test_duplicate_effective_identity_cannot_be_approved_for_distinct_roles(world):
    from dubbing_service.app.models.project import Role, Segment
    from dubbing_service.app.storage.segments import save_segments
    _, project_id = world
    project = load_project(project_id)
    project.roles.append(Role(id="role-b", name="Distinct person", tts_backend="qwen3tts"))
    save_project(project)
    save_segments(project_id, [Segment(id="a", index=0, text="A.", role_id="role-a"),
                               Segment(id="b", index=1, text="B.", role_id="role-b")])
    prepare(world)
    first = ready_candidate(world, key="same-reference")
    casting.feedback(project_id, {**first, "stars": 5, "action": "use"})
    second = ready_candidate(world, role_id="role-b", key="same-reference")
    with pytest.raises(casting.CastingError, match="already approved"):
        casting.feedback(project_id, {**second, "event_id": "other-event", "stars": 5, "action": "use"})
    assert casting.approved(load_project(project_id), "role-a")
    assert not casting.approved(load_project(project_id), "role-b")
