# ruff: noqa: F811 -- pytest fixture import is reused as test arguments.
"""Disposable fixture audio only; no human stars, real worker or acquisition."""
from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook import named_reference_feedback as named
from dubbing_service.app.storage.projects import load_project
from tests.test_casting_review import world, ready_candidate, prepare  # noqa: F401


def test_named_performer_is_explicit_and_no_rating_is_inferred():
    direction = casting._direction("The narrator wasn't great. Deeper, more like David Attenborough")
    assert direction['requested_performer'] == 'David Attenborough'
    assert direction['reference_mode'] == 'verified_recording'
    assert direction['dissatisfied'] and direction['actionable']
    assert 'stars' not in direction and 'rating' not in direction


def test_star_free_save_persists_and_creates_one_bounded_named_request(world):
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    body.update(raw_direction='Deeper, more like David Attenborough')
    result = casting.feedback(pid, body)
    state = load_project(pid).voice_lab.casting
    current = state['roles']['role-a']
    assert result['saved'] and not result['conflict']
    assert 'without adding a rating' in result['reason']
    assert current['round_revision'] == 2
    assert current['history'][0]['candidates'][0]['rating'] is None
    assert current['assignment'] is None
    assert len(current['round']['candidates']) == 1
    candidate = current['round']['candidates'][0]
    assert candidate['status'] == 'blocked' and candidate['rating'] is None
    assert candidate['provenance'] == 'reference_clone'
    assert candidate['inspiration']['performer'] == 'David Attenborough'
    assert 'not loaded' in candidate['inspiration']['performance']
    assert current['round']['direction']['requested_performer'] == 'David Attenborough'
    assert state['events'][body['event_id']]['raw_direction'] == body['raw_direction']
    assert state['intents'][current['round']['intent_id']]['status'] == 'complete'
    assert casting.feedback(pid, body) == result
    assert load_project(pid).voice_lab.casting['roles']['role-a']['round_revision'] == 2


def test_star_free_plain_note_saves_without_automatic_generation(world):
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    body.update(raw_direction='The third sentence has an odd pause.')
    result = casting.feedback(pid, body)
    assert result['saved']
    assert load_project(pid).voice_lab.casting['roles']['role-a']['round_revision'] == 1


def test_named_request_has_no_unrelated_or_trait_only_fallback(world):
    _, pid = world
    project = load_project(pid)
    plan = named.proposals(project, 'role-a', 'David Attenborough')
    assert len(plan) == 1 and plan[0]['blocked']
    assert plan[0]['design_instruction'] == '' and not plan[0].get('conditioning')
    assert plan[0]['reference']['characterization'] == 'not_loaded'


def test_later_text_retains_named_request_in_the_same_feedback_direction():
    previous = named.interpret('More like David Attenborough')
    assert named.interpret('Deeper', previous)['requested_performer'] == 'David Attenborough'


def test_later_deeper_feedback_cannot_lose_a_stored_named_request(world):
    from dubbing_service.app.storage.projects import save_project
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    project = load_project(pid)
    project.voice_lab.casting['roles']['role-a']['round']['direction'] = casting._direction('More like David Attenborough')
    save_project(project)
    body.update(raw_direction='Deeper')
    casting.feedback(pid, body)
    current = load_project(pid).voice_lab.casting['roles']['role-a']['round']
    assert current['direction']['requested_performer'] == 'David Attenborough'
    assert len(current['candidates']) == 1 and current['candidates'][0]['status'] == 'blocked'
    assert current['candidates'][0]['provenance'] == 'reference_clone'


def test_named_request_takes_precedence_over_performance_reuse(world):
    _, pid = world
    prepare(world)
    body = ready_candidate(world)
    body.update(raw_direction='Slower, more like David Attenborough')
    result = casting.feedback(pid, body)
    assert result['saved']
    current = load_project(pid).voice_lab.casting['roles']['role-a']['round']
    assert not current['direction']['performance_only']
    assert current['candidates'][0]['status'] == 'blocked'
    assert current['candidates'][0]['provenance'] == 'reference_clone'
    assert current['candidates'][0]['reuse_key'] is None
