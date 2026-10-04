"""Original owner-design intent and real audition lifecycle, isolated signals only."""
import copy

import pytest

from dubbing_service.app.audiobook import casting_review as casting
from dubbing_service.app.audiobook import original_inspired_design as original
from dubbing_service.app.audiobook import voice_lab as lab
from dubbing_service.app.storage.projects import load_project, save_project
from tests import test_casting_auditions, test_casting_review

casting_world = test_casting_review.world
audition_world = test_casting_auditions.casting_project


def install_target(project, rid):
    project.metadata['voice_lab_source_targets'] = {rid: {
        'performer': 'Richard Harris', 'requested_by_event': 'older-recording-request'}}
    project.metadata['voice_lab_original_design_targets'] = {rid: {
        'schema': original.SCHEMA, 'performer': 'Richard Harris',
        'performance': 'Older gentle British storyteller; descriptive original inspiration',
        'requested_by_event': 'new-owner-original-design-request', 'actor_audio_used': False,
        'options': [
            {'label': 'Warm rounded elder', 'instruction': 'Older male British English voice, warm rounded low-mid resonance and gentle patient pauses.'},
            {'label': 'Airy reflective elder', 'instruction': 'Older male British English voice, airy lighter middle resonance and reflective rising inflection.'},
            {'label': 'Textured storyteller', 'instruction': 'Older male British English voice, soft gravel in low register, spacious storytelling and quiet amusement.'}]}}


def test_original_route_preserves_recording_history_and_creates_three_fresh_designs(casting_world):
    _, pid = casting_world
    p = load_project(pid)
    install_target(p, 'role-a')
    old_targets = copy.deepcopy(p.metadata['voice_lab_source_targets'])
    r = casting._new_round(p, 'role-a', reuse_key='unrelated', replacement=False)
    assert r['direction']['reference_mode'] == 'original_inspired_design'
    assert len(r['candidates']) == 3 and r['status'] == 'pending'
    assert len({c['concept']['emphasis'] for c in r['candidates']}) == 3
    assert all(c['provenance'] == 'generated' and c['reuse_key'] is None for c in r['candidates'])
    assert all(c['conditioning'] is None and c['reference']['actor_audio_used'] is False for c in r['candidates'])
    assert p.metadata['voice_lab_source_targets'] == old_targets
    assert r['candidates'][0]['inspiration']['performer'] == 'Richard Harris'
    assert all(isinstance(t, dict) and t.get('dimension') and t.get('value')
               for c in r['candidates'] for t in c['inferred_traits'])


def test_later_recording_request_still_holds_and_never_uses_original_route(casting_world):
    _, pid = casting_world
    p = load_project(pid)
    install_target(p, 'role-a')
    r = casting._new_round(p, 'role-a', direction=casting._direction('Clone the voice of Alan Rickman'))
    assert r['status'] == 'blocked' and len(r['candidates']) == 1
    assert r['candidates'][0]['provenance'] == 'reference_clone'
    assert r['candidates'][0]['inspiration']['performer'] == 'Alan Rickman'


@pytest.mark.parametrize('change', ['missing-event', 'actor-audio', 'duplicates', 'bad-schema'])
def test_invalid_original_intent_does_not_bypass_recording_gate(casting_world, change):
    _, pid = casting_world
    p = load_project(pid)
    install_target(p, 'role-a')
    row = p.metadata['voice_lab_original_design_targets']['role-a']
    if change == 'missing-event':
        row.pop('requested_by_event')
    elif change == 'actor-audio':
        row['actor_audio_used'] = True
    elif change == 'duplicates':
        row['options'][1] = copy.deepcopy(row['options'][0])
    else:
        row['schema'] = 'unknown'
    assert original.target(p, 'role-a') is None
    assert casting._new_round(p, 'role-a')['status'] == 'blocked'
    assert original.target(p, 'other-role') is None


def test_original_audition_saves_reference_and_uses_production_path_for_all_sections(audition_world, monkeypatch):
    p, role, provider = audition_world
    install_target(p, role.id)
    r = casting._new_round(p, role.id, replacement=True)
    save_project(p)
    monkeypatch.setattr('dubbing_service.app.audiobook.performer_reference.resolve',
                        lambda *a, **k: pytest.fail('Original design must not infer celebrity measurements'))
    result = test_casting_auditions.run(p, role, provider, 'original-inspired-audition',
        candidate_id=r['candidates'][0]['candidate_id'], script_version='casting-30-v1')
    assert result['status'] == 'succeeded', result
    assert len(provider.designs) == 1 and len(provider.calls) == 6
    assert provider.conditioning is None
    record = load_project(p.id).voice_lab.voices[result['key']]
    assert record['kind'] == 'original_inspired'
    assert record['identity']['provenance']['origin'] == 'generated_original_voice'
    assert record['identity']['status'] == 'LOCKED'
    assert record['inspiration']['performer'] == 'Richard Harris'
    assert record['technical_qa'] == 'passed' and record['human_review'] == 'pending'
    # Later audition reuse uses the exact same conditioning; never re-designs it.
    again = test_casting_auditions.run(p, role, provider, 'same-original-reused',
        candidate_id=r['candidates'][0]['candidate_id'], reuse_key=result['key'])
    assert again['status'] == 'succeeded'
    reused = load_project(p.id).voice_lab.voices[again['key']]
    assert len(provider.designs) == 1
    assert reused['identity']['prompt_sha256'] == record['identity']['prompt_sha256']


def test_modified_prompt_loses_original_exception(casting_world):
    _, pid = casting_world
    p = load_project(pid)
    install_target(p, 'role-a')
    casting._new_round(p, 'role-a')
    c = p.voice_lab.shortlists['role-a']['matches'][0]
    assert original.authorized_candidate(c)
    c['design_instruction'] += ' changed'
    assert not original.authorized_candidate(c)
    assert lab.candidate_conditioning(c) is None
