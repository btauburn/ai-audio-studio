"""Character check/heart semantics over isolated real-store fixtures; no TTS."""
import copy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook import voice_lab as lab
from dubbing_service.app.routers.voice_lab import router
from dubbing_service.app.storage.projects import load_project, save_project
from tests import test_casting_review as fixtures

world = fixtures.world
prepare = fixtures.prepare
ready_candidate = fixtures.ready_candidate


def test_explicit_check_selects_without_inventing_rating_and_replays(world):
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    project = load_project(pid)
    project.voice_lab.preferences[body['key']] = {
        'favorite': True, 'rating': 3, 'audio_sha256': body['audio_sha256']}
    save_project(project)
    before = copy.deepcopy(load_project(pid).voice_lab.preferences)
    body.update(action='use', selection_confirmed=True)
    result = casting.feedback(pid, body)
    assert casting.feedback(pid, body) == result
    project = load_project(pid)
    assert casting.approved(project, 'role-a')
    item = project.voice_lab.casting['roles']['role-a']
    assert item['assignment']['decision'] == 'explicit_selection'
    assert item['assignment']['stars'] is None
    assert item['round']['candidates'][0]['rating'] is None
    assert project.voice_lab.preferences == before
    view = casting.view(pid)['roles'][0]
    assert view['chosen'] and view['selected_candidate_id'] == body['candidate_id']
    assert view['candidates'][0]['favorite'] is True


@pytest.mark.parametrize('confirmation', [False, None, 1, 'true'])
def test_check_cannot_be_inferred_or_coerced(world, confirmation):
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    body.update(action='use')
    if confirmation is not None:
        body['selection_confirmed'] = confirmation
    before = load_project(pid).model_dump(mode='json')
    with pytest.raises(casting.CastingError, match='explicit|boolean'):
        casting.feedback(pid, body)
    assert load_project(pid).model_dump(mode='json') == before


def test_low_rating_with_check_is_rejected_as_conflicting_instruction(world):
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    body.update(action='use', selection_confirmed=True, stars=2)
    with pytest.raises(casting.CastingError, match='explicit selection'):
        casting.feedback(pid, body)
    assert not casting.approved(load_project(pid), 'role-a')


def test_save_or_heart_does_not_approve_even_with_confirmation(world):
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    body['selection_confirmed'] = True
    casting.feedback(pid, body)
    lab.save_preference(pid, {'key': body['key'], 'audio_sha256': body['audio_sha256'], 'favorite': True})
    assert not casting.approved(load_project(pid), 'role-a')


def test_stale_check_remains_historical_without_assigning(world):
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    project = load_project(pid)
    item = project.voice_lab.casting['roles']['role-a']
    item['assignment_revision'] += 1
    save_project(project)
    before_roles = load_project(pid).model_dump(mode='json')['roles']
    body.update(action='use', selection_confirmed=True)
    result = casting.feedback(pid, body)
    assert result['conflict']
    assert load_project(pid).model_dump(mode='json')['roles'] == before_roles
    assert not casting.approved(load_project(pid), 'role-a')


def test_current_sample_is_verified_and_does_not_create_candidates_or_approval(world):
    root, pid = world
    prepare(world)
    body = ready_candidate(world)
    project = load_project(pid)
    record = project.voice_lab.voices[body['key']]
    record.update(role_id='role-a', name='Current original sample', kind='original_inspired', synthetic=True)
    project.voice_lab.casting['roles']['role-a'].pop('round')
    save_project(project)
    meta = root / pid / 'project.json'
    before = meta.read_bytes()
    row = casting.view(pid, base_url='/api/voice-lab/' + pid)['roles'][0]
    assert row['current_audio']['audio_available']
    assert row['current_audio']['provenance'] == 'generated_original_voice'
    assert row['current_audio']['audio_url'] == '/api/voice-lab/' + pid + '/audio/' + body['key']
    assert row['candidates'] == [] and not row['chosen']
    assert meta.read_bytes() == before
    (root / pid / record['audio_file']).write_bytes(b'corrupt')
    row = casting.view(pid)['roles'][0]
    assert row['current_audio']['audio_available'] is False
    assert row['current_audio']['audio_url'] is None


def test_current_audio_cannot_be_misattributed_or_inherit_wrong_version_favorite(world):
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    project = load_project(pid)
    record = project.voice_lab.voices[body['key']]
    assert casting._current_audio(project, 'role-a', body['key'], {body['key']: record}, '/base') is None
    record['role_id'] = 'role-a'
    project.voice_lab.preferences[body['key']] = {'favorite': True, 'audio_sha256': '0' * 64}
    assert not casting._current_audio(project, 'role-a', body['key'], {body['key']: record}, '/base')['favorite']


def test_real_route_accepts_boolean_check_and_rejects_coercion(world, monkeypatch):
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    monkeypatch.setattr(casting, 'maintenance_tick', lambda **kwargs: None)
    app = FastAPI()
    app.include_router(router, prefix='/api/v1')
    client = TestClient(app)
    url = f'/api/v1/projects/{pid}/voice-lab/casting/feedback'
    body.update(action='use', selection_confirmed='true')
    assert client.post(url, json=body).status_code == 422
    body['selection_confirmed'] = True
    response = client.post(url, json=body)
    assert response.status_code == 200, response.text
    assert casting.approved(load_project(pid), 'role-a')


def test_stale_assignment_audio_cannot_be_labelled_current(world):
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    project = load_project(pid)
    record = project.voice_lab.voices[body['key']]
    record['role_id'] = 'role-a'
    stale = {**record, 'key': 'old-voice', 'version': 'old-revision'}
    project.voice_lab.voices['old-voice'] = stale
    project.voice_lab.casting['roles']['role-a']['assignment'] = {
        'key': 'old-voice', 'voice_revision': 'old-revision', 'approved': True}
    save_project(project)
    view = casting.view(pid)['roles'][0]
    assert view['current_key'] == body['key']
    assert view['current_audio']['key'] == body['key']
    assert not view['chosen'] and view['selected_candidate_id'] is None
    assert casting._current_audio(project, 'role-a', 'old-voice', {'old-voice': stale}, '/base') is None


def test_confirmation_change_cannot_replay_same_event(world):
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    body.update(action='use', selection_confirmed=True)
    casting.feedback(pid, body)
    body['selection_confirmed'] = False
    with pytest.raises(casting.CastingError, match='different feedback'):
        casting.feedback(pid, body)


@pytest.mark.parametrize('fixture_only', [False, True])
def test_human_confirmation_keeps_fixture_exclusion_and_never_writes_rating(world, fixture_only):
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    project = load_project(pid)
    project.voice_lab.scope = 'production'
    # Only this disposable fixture exercises the human-origin branch. No live
    # data or actual human approval is created by the test.
    project.voice_lab.voices[body['key']]['identity']['provenance']['fixture_only'] = fixture_only
    save_project(project)
    before_roles = load_project(pid).model_dump(mode='json')['roles']
    body.update(action='use', selection_confirmed=True)
    if fixture_only:
        with pytest.raises(casting.CastingError, match='Test-generated'):
            casting.feedback(pid, body)
        assert load_project(pid).model_dump(mode='json')['roles'] == before_roles
        assert not casting.approved(load_project(pid), 'role-a')
    else:
        casting.feedback(pid, body)
        project = load_project(pid)
        event = project.voice_lab.casting['events'][body['event_id']]
        assert event['origin'] == 'human' and 'stars' not in event
        assert casting.approved(project, 'role-a')
        assert project.voice_lab.preferences == {}
        assert project.voice_lab.preference_history == []
        assert project.voice_lab.casting['roles']['role-a']['round']['candidates'][0]['rating'] is None
