"""Clone-only policy is enforced by the shipped discovery and render owners."""
import hashlib
import pytest
from dubbing_service.app.audiobook import casting_director as director, pipeline
from dubbing_service.app.models.project import Role, Segment
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import save_segments, load_segments


def project():
    p = create_project('Clone policy fixture')
    p.voice_lab.scope = 'regression_fixture'
    p.roles = [Role(id='role', name='Erick', tts_backend='qwen3tts')]
    p.metadata['director_voice_policy'] = {'mode': 'cloned_only'}
    save_project(p)
    save_segments(p.id, [Segment(id='line', role_id='role', index=0, text='There is still time.')])
    return p


def candidate(origin='eligible_reference_clone'):
    return {'candidate_id': origin, 'name': origin, 'origin': 'generic_library',
            'eligible': True, 'holds': [], 'traits': {'texture': 'clear'}, 'tags': ['natural'],
            'reference_conditioned': True,
            'conditioning': {'mechanism': 'qwen_reference', 'backend': 'qwen3tts',
                             'reference_audio_sha256': hashlib.sha256(b'actual recorded fixture bytes').hexdigest(),
                             'prompt_sha256': hashlib.sha256(b'actual prompt fixture bytes').hexdigest()},
            'provenance': {'stored_origin': origin, 'actor_audio_used': origin == 'eligible_reference_clone'}}


def test_policy_change_invalidates_cached_recommendations():
    p=project(); p.metadata.pop('director_voice_policy'); save_project(p)
    before=director._binding(p,load_segments(p.id))
    p.metadata['director_voice_policy']={'mode':'cloned_only'}; save_project(p)
    assert director._binding(p,load_segments(p.id)) != before
    assert director.view(p.id)['voice_policy']=='cloned_only'


def test_real_compute_excludes_synthetic_conditioned_voice(monkeypatch):
    from dubbing_service.app.audiobook import casting_director_discovery as discovery
    p=project()
    monkeypatch.setattr(discovery,'discover_candidates',lambda *args: {'candidates':[candidate(),candidate('generated_original_voice')],'holds':[]})
    r=director.analyze(p.id)
    assert r['voice_policy']=='cloned_only'
    assert {c['candidate_id'] for role in r['roles'] for c in role['candidates']}=={'eligible_reference_clone'}
    assert any(h['code']=='clone_policy_generated_original_voice' for h in r['holds'])
    assert not any(role['current_voice']['approved'] for role in r['roles'])


def test_render_refuses_missing_clone_before_provider_work():
    p=project()
    class NeverCalled:
        backend='qwen3tts'
        def render(self,*args,**kwargs): pytest.fail('Blocked role must never synthesize')
        def _get_provider(self): pytest.fail('Blocked role must never initialize inference')
    with pytest.raises(pipeline.PipelineError,match='Cloned-only synthesis'):
        pipeline.render_utterances(p.id,['line'],renderer=NeverCalled())
    assert not load_segments(p.id)[0].takes


def test_unknown_policy_fails_before_any_render():
    p=project();p.metadata['director_voice_policy']={'mode':'cloned_ish'};save_project(p)
    from dubbing_service.app.audiobook.cloned_voice_policy import ClonePolicyError
    with pytest.raises(ClonePolicyError): pipeline.render_utterances(p.id,['line'],renderer=object())


def test_protected_synthetic_voice_is_saved_but_not_reported_as_policy_coverage(monkeypatch):
    from dubbing_service.app.models.character import CharacterVoiceIdentity
    p=project()
    p.roles[0].voice_identity=CharacterVoiceIdentity(id='synthetic',role_id='role',mechanism='qwen_reference',
        model='fixture',dossier_revision='fixture',casting_profile_revision='fixture',design_instruction='original',
        reference_audio='ref.wav',prompt_artifact='prompt.bin',
        reference_audio_sha256=hashlib.sha256(b'synthetic reference fixture').hexdigest(),
        prompt_sha256=hashlib.sha256(b'synthetic prompt fixture').hexdigest(),
        provenance={'origin':'generated_original_voice','actor_audio_used':False})
    save_project(p)
    monkeypatch.setattr(director,'_current',lambda *a:{'identity_id':'synthetic','key':'original','rating':5,'approved':True,'protected':True})
    before=load_project(p.id).model_dump_json()
    r=director.analyze(p.id)['roles'][0]
    assert r['protected'] and r['current_voice']['approved']
    assert r['needs_casting'] and r['policy_conflict'] and r['coverage']=='blocked_by_voice_policy'
    assert load_project(p.id).model_dump_json()==before


def test_ordinary_casting_round_cannot_escape_into_synthetic_design():
    from dubbing_service.app.audiobook import casting_review
    p=project()
    with pytest.raises(casting_review.CastingError,match='requires cloned voices'):
        casting_review._proposal_plan(p,'role',None,None,'',None)


def test_direct_audition_cannot_escape_into_synthetic_design(monkeypatch):
    from dubbing_service.app.audiobook import voice_lab_audition as auditions
    p=project()
    monkeypatch.setattr(auditions,'_candidate_for_intent',lambda *a:{'candidate_id':'design','design_instruction':'original design'})
    with pytest.raises(auditions.AuditionError,match='recorded reference'):
        auditions._bound_intent(p,'role','design')


def test_exact_clone_audition_freezes_book_policy_into_child_binding(monkeypatch):
    from dubbing_service.app.audiobook import voice_lab_audition as auditions
    p=project()
    clone={'kind':'reference_clone','identity':{'mechanism':'qwen_reference','backend':'qwen3tts',
        'reference_audio':'reference.wav','prompt_artifact':'prompt.bin',
        'reference_audio_sha256':hashlib.sha256(b'recorded reference fixture').hexdigest(),
        'prompt_sha256':hashlib.sha256(b'prompt fixture').hexdigest(),
        'provenance':{'origin':'eligible_reference_clone','actor_audio_used':True}}}
    monkeypatch.setattr(auditions,'_candidate_for_intent',lambda *a:{'candidate_id':'reuse','reuse_key':'saved'})
    monkeypatch.setattr(auditions,'_reuse_record',lambda *a:clone)
    frozen_hash,binding=auditions._bound_intent(p,'role','reuse',reuse_key='saved')
    assert binding['director_voice_policy']=={'mode':'cloned_only'}
    p.metadata.pop('director_voice_policy');save_project(p)
    unrestricted_hash,unrestricted=auditions._bound_intent(p,'role','reuse',reuse_key='saved')
    assert 'director_voice_policy' not in unrestricted and unrestricted_hash!=frozen_hash
