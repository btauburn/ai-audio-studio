"""Real planner fixtures: target preservation, override, role isolation and reuse."""
from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook import named_reference_feedback as named
from dubbing_service.app.storage.projects import load_project
from tests import test_casting_review as casting_fixtures

casting_world = casting_fixtures.world


def target(project, role_id='role-a'):
    project.metadata['voice_lab_source_targets'] = {role_id: {
        'performer': 'Richard Harris', 'requested_by_event': 'explicit-owner-event',
        'source_status': 'eligible_reference_required'}}


def test_blank_replacement_honors_stored_target_and_holds_without_source(casting_world):
    _, pid = casting_world
    project = load_project(pid)
    target(project)
    current = casting._new_round(project, 'role-a', replacement=True)
    assert current['direction']['requested_performer'] == 'Richard Harris'
    assert current['status'] == 'blocked'
    assert len(current['candidates']) == 1
    row = current['candidates'][0]
    assert row['inspiration']['performer'] == 'Richard Harris'
    assert row['provenance'] == 'reference_clone' and row['reuse_key'] is None
    assert project.voice_lab.casting['intents'][current['intent_id']]['status'] == 'complete'


def test_stored_target_cannot_reuse_an_unrelated_performance_key(casting_world):
    _, pid = casting_world
    project = load_project(pid)
    target(project)
    current = casting._new_round(project, 'role-a', replacement=True,
        direction=casting._direction('Slower'), reuse_key='unrelated-heard-voice')
    assert not current['direction']['performance_only']
    assert current['candidates'][0]['status'] == 'blocked'
    assert current['candidates'][0]['reuse_key'] is None


def test_shared_keeper_gate_excludes_unrelated_rated_voice(casting_world, monkeypatch):
    from dubbing_service.app.audiobook import preferred_voices
    _, pid = casting_world
    project = load_project(pid)
    target(project)
    monkeypatch.setattr(preferred_voices, 'match_preferred', lambda *a, **kw: [
        {'actual_source_performer': 'John Lithgow'}])
    assert casting._keeper_proposals(project, 'role-a', None, None) == []


def test_later_explicit_performer_overrides_target_without_rewriting_it(casting_world):
    _, pid = casting_world
    project = load_project(pid)
    target(project)
    assert named.requested_performer(project, 'role-a', 'Clone the voice of Alan Rickman') == 'Alan Rickman'
    assert named.requested_performer(project, 'role-a', explicit='Daniel Radcliffe') == 'Daniel Radcliffe'
    assert project.metadata['voice_lab_source_targets']['role-a']['performer'] == 'Richard Harris'


def test_target_is_role_specific_and_requires_owner_event(casting_world):
    _, pid = casting_world
    project = load_project(pid)
    target(project)
    assert named.requested_performer(project, 'other-role') is None
    project.metadata['voice_lab_source_targets']['role-a'].pop('requested_by_event')
    assert named.requested_performer(project, 'role-a') is None
