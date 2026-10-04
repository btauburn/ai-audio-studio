"""Reference ASR uses synthetic PCM and a fake CPU recognizer only."""
# ruff: noqa: F811 -- pytest fixture imports are reused as test arguments.
import hashlib
import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from dubbing_service.app.audiobook import voice_lab_audition  # noqa: F401 - load store adapters before path mocks
from dubbing_service.app.audiobook import reference_transcripts as asr
from dubbing_service.app.audiobook import performer_discovery as discovery
from tests.test_voice_lab_reference_clone import fixture, acquire, import_voice  # noqa: F401


@pytest.fixture
def recognition(tmp_path, monkeypatch):
    calls = []
    words = [SimpleNamespace(start=.2, end=.8, word=' Fixture', probability=.99)]
    rows = [SimpleNamespace(start=.2, end=1., text='Fixture speech.', avg_logprob=-.1, no_speech_prob=.01, words=words)]
    class Model:
        def __init__(self, path, **kwargs):
            assert kwargs == dict(device='cpu', compute_type='int8', cpu_threads=2, local_files_only=True)
            calls.append(('load', path))
        def transcribe(self, path, **kwargs):
            calls.append(('transcribe', hashlib.sha256(open(path,'rb').read()).hexdigest()))
            assert kwargs['word_timestamps'] and not kwargs['condition_on_previous_text']
            return iter(rows), None
    monkeypatch.setattr(asr, '_model', lambda: (tmp_path, 'm'*64))
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=Model))
    output=io.BytesIO()
    sf.write(output, np.sin(np.arange(64000)/17).astype(np.float32)*.1, 16000, format='WAV', subtype='FLOAT')
    return output.getvalue(), calls, rows


def test_exact_crop_hash_cache_and_no_approval(tmp_path, recognition):
    audio, calls, _ = recognition
    result=asr.transcribe_bytes(tmp_path,audio)
    assert result['status']=='draft' and not result['human_verified'] and result['machine_generated']
    assert result['audio_sha256']==hashlib.sha256(audio).hexdigest()
    assert result['segments'][0]['words'][0]['start']==.2
    assert asr.transcribe_bytes(tmp_path,audio)['cache_hit']
    assert len(calls)==2
    path=next((tmp_path/'transcripts').glob('*.json'))
    blob=json.loads(path.read_text()); blob['result']['text']='tampered'; path.write_text(json.dumps(blob))
    assert asr.transcribe_bytes(tmp_path,audio)['status']=='failed'
    assert len(calls)==2


def test_missing_local_model_has_no_recognizer_load(tmp_path, recognition, monkeypatch):
    audio,calls,_=recognition
    def missing(): raise ValueError('Local ASR model absent')
    monkeypatch.setattr(asr,'_model',missing)
    result=asr.transcribe_bytes(tmp_path,audio)
    assert result['status']=='unavailable' and calls==[]


@pytest.mark.parametrize('field,value',[('end',99),('start',float('nan'))])
def test_invalid_timestamps_fail_not_cache(tmp_path,recognition,field,value):
    audio,_,rows=recognition
    setattr(rows[0],field,value)
    result=asr.transcribe_bytes(tmp_path,audio)
    assert result['status']=='failed' and not list((tmp_path/'transcripts').glob('*.json'))


def test_empty_recognition_failure_is_correctable(tmp_path,recognition):
    audio,_,rows=recognition
    rows.clear()
    assert asr.transcribe_bytes(tmp_path,audio)['status']=='failed'


def test_automatic_transcript_binds_exact_crop_and_keeps_review_held(fixture,recognition):
    f=fixture
    acquisition=acquire(f,transcript=None,audio_review=None)
    evidence=acquisition['transcript_evidence']
    assert evidence['audio_sha256']==acquisition['reference_audio_sha256']
    assert acquisition['reference_text']=='Fixture speech.' and not acquisition['clone_ready']
    assert not evidence['human_verified']
    assert acquire(f,transcript=None,audio_review=None)['acquisition_sha256']==acquisition['acquisition_sha256']
    with pytest.raises(ValueError,match='QA'): import_voice(f,acquisition)
    assert f.calls==[]


def test_review_only_allows_licensed_inspection_but_never_import(fixture,recognition):
    f=fixture
    result=acquire(f,transcript=None,audio_review=None,intended_use_evidence={},review_only=True)
    assert result['eligibility']['download_eligible'] and not result['eligibility']['intended_use_eligible']
    assert not result['clone_ready'] and any('Review-only' in s for s in result['technical_qa']['holds'])
    with pytest.raises(ValueError): import_voice(f,result)
    assert f.calls==[]


def test_stereo_exact_mean_crop_and_missing_provenance_rejected(fixture,monkeypatch):
    from service import qwen_character_voice as voice
    f=fixture
    left=np.sin(np.arange(80000)/23).astype(np.float32)*.15
    wave=np.column_stack((left,left*.7))
    output=io.BytesIO(); sf.write(output,wave,16000,format='WAV',subtype='FLOAT')
    raw=output.getvalue()
    monkeypatch.setattr(discovery,'_download_media',lambda *a,**k:raw)
    f.source['upstream_media_sha1']=hashlib.sha1(raw).hexdigest()
    review=f.review|{'original_sha256':hashlib.sha256(raw).hexdigest()}
    acquired=acquire(f,audio_review=review)
    assert acquired['clone_ready']
    ref,_=sf.read(f.directory/acquired['reference_path'],dtype='float32')
    assert np.array_equal(ref,wave[:64000].mean(axis=1))
    assert import_voice(f,acquired)['provenance']['origin']=='eligible_reference_clone'
    path=f.directory/'acquisitions'/acquired['acquisition_id']/'acquisition.json'
    receipt=json.loads(path.read_text());receipt['preprocessing']=receipt['preprocessing'][1:]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='stereo|conversion|preprocessing'):
        voice._eligible_acquisition(f.project,acquired['acquisition_id'],hashlib.sha256(path.read_bytes()).hexdigest())


def test_correction_is_bound_separate_and_stale_rejected(tmp_path,monkeypatch):
    from dubbing_service.app.core import config
    from dubbing_service.app.audiobook import reference_transcripts
    monkeypatch.setattr(config,'safe_project_root',lambda _:tmp_path)
    result={'audio_sha256':'a'*64,'text_sha256':'b'*64}
    monkeypatch.setattr(reference_transcripts,'acquisition_transcript',lambda *a:result)
    with pytest.raises(ValueError,match='changed'):
        asr.correct_acquisition_transcript('p','a','h','Correct words.','c'*64)
    saved=asr.correct_acquisition_transcript('p','a','h','Correct words.','b'*64)
    assert saved['status']=='corrected_unreviewed' and not saved['human_verified']
    assert saved['alignment_method']=='not aligned after correction'
    assert len(list((tmp_path/'voice_lab/discovery/transcripts/corrections').glob('*.json')))==1
    found=asr._correction(tmp_path/'voice_lab/discovery','a','h','a'*64,'b'*64)
    assert found==saved
    with pytest.raises(ValueError,match='stale'):
        asr._correction(tmp_path/'voice_lab/discovery','a','h','changed','b'*64)


def test_prepared_clone_import_registration_first_audition_and_child_provenance(fixture,monkeypatch,tmp_path):
    import shutil
    from service import qwen_character_voice as voice
    from dubbing_service.app.audiobook import voice_lab as lab, voice_lab_audition as auditions, named_reference_feedback as named
    from dubbing_service.app.audiobook.fingerprint import voice_revision_for_role
    from dubbing_service.app.core.config import safe_project_root
    from dubbing_service.app.models.project import Role
    from dubbing_service.app.storage.projects import create_project, save_project, load_project
    from tests.audiobook_helpers import use_projects_root
    f=fixture
    f.source['performer']='Synthetic Fixture Speaker'
    acquisition=acquire(f)
    root=use_projects_root(monkeypatch,tmp_path/'studio')
    parent=create_project('Eligible synthetic reference fixture')
    parent.roles=[Role(id='narrator',name='Narrator',tts_backend='qwen3tts')]
    save_project(parent)
    shutil.copytree(f.directory,root/parent.id/'voice_lab/discovery')
    monkeypatch.setattr(voice,'safe_project_root',safe_project_root)
    class Provider:
        def import_character_reference(self,pid,identity,aid,sha):
            f.runtime.import_reference(pid,identity,aid,sha)
            return f.runtime.prepare(pid,identity)
    before=parent.roles[0].model_dump(mode='json')
    imported=lab.import_reference(parent.id,'narrator',acquisition['acquisition_id'],acquisition['acquisition_sha256'],
                                  voice_revision_for_role(parent.roles[0]),'first-reference',provider=Provider())
    parent=load_project(parent.id)
    assert parent.roles[0].model_dump(mode='json')==before
    assert not parent.voice_lab.preferences
    assert parent.metadata['voice_lab_named_references']['narrator'][0]['key']==imported['key']
    record=auditions._reuse_record(parent,imported['key'])
    assert record['audio_file'] is None and record['kind']=='reference_clone'
    plan=named.proposals(parent,'narrator','Synthetic Fixture Speaker')
    assert len(plan)==1 and not plan[0]['blocked'] and plan[0]['reuse_key']==imported['key']
    child=create_project('Bounded first reference audition fixture')
    child.roles=[Role(id='narrator',name='Narrator',tts_backend='qwen3tts')]
    save_project(child)
    auditions._clone_reusable_voice(parent.id,child,{'reuse_key':imported['key'],'role_id':'narrator','binding':{}})
    receipt,_,_=voice._eligible_acquisition(root/child.id,acquisition['acquisition_id'],acquisition['acquisition_sha256'])
    assert receipt['reference_audio_sha256']==acquisition['reference_audio_sha256']
    assert load_project(parent.id).roles[0].model_dump(mode='json')==before
    assert f.calls==['prepare']


def test_reference_routes_reject_wrong_receipt_hash_and_audio_tampering(fixture,recognition,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from dubbing_service.app.core import config
    from dubbing_service.app.routers.voice_lab import router
    f=fixture
    acquisition=acquire(f,transcript=None,audio_review=None)
    monkeypatch.setattr(config,'safe_project_root',lambda _:f.project)
    app=FastAPI();app.include_router(router,prefix='/api/v1');client=TestClient(app)
    base=f'/api/v1/projects/fixture/voice-lab/references/{acquisition["acquisition_id"]}'
    payload={'acquisition_sha256':acquisition['acquisition_sha256']}
    result=client.post(base+'/transcript',json=payload)
    assert result.status_code==200 and not result.json()['human_verified']
    assert client.post(base+'/transcript',json={'acquisition_sha256':'0'*64}).status_code==409
    audio=client.get(base+'/audio',params=payload,headers={'Range':'bytes=0-15'})
    assert audio.status_code==206 and len(audio.content)==16
    corrected=client.post(base+'/transcript/correction',json=payload|{'text':'Fixture corrected.', 'expected_text_sha256':result.json()['text_sha256']})
    assert corrected.status_code==200,corrected.text
    again=client.post(base+'/transcript',json=payload).json()
    assert again['correction']['text']=='Fixture corrected.'
    assert client.post(base+'/transcript/correction',json=payload|{'text':'Overwrites stale.', 'expected_text_sha256':result.json()['text_sha256']}).status_code==409
    (f.directory/acquisition['reference_path']).write_bytes(b'tampered')
    assert client.get(base+'/audio',params=payload).status_code==409
    assert client.post(base+'/transcript',json=payload).status_code==409


def test_manual_reference_then_asr_correction_remains_listable(fixture,recognition,monkeypatch):
    from dubbing_service.app.core import config
    f=fixture
    acquired=acquire(f)
    assert 'transcript_evidence' not in acquired
    monkeypatch.setattr(config,'safe_project_root',lambda _:f.project)
    draft=asr.acquisition_transcript('fixture',acquired['acquisition_id'],acquired['acquisition_sha256'])
    correction=asr.correct_acquisition_transcript('fixture',acquired['acquisition_id'],acquired['acquisition_sha256'],
                                                 'Corrected fixture words.',draft['text_sha256'])
    result=asr.list_acquisitions('fixture')
    assert len(result['references'])==1
    assert result['references'][0]['correction']==correction
    assert not correction['human_verified']


def test_asr_cache_invalidation_after_dependency_identity_change(tmp_path, recognition, monkeypatch):
    audio, calls, _ = recognition
    first = asr.transcribe_bytes(tmp_path, audio)
    assert len(calls) == 2
    newer = dict(asr._LOADED_DEPENDENCIES) | {'faster-whisper': 'new-process-fixture-version'}
    monkeypatch.setattr(asr, '_LOADED_DEPENDENCIES', newer)
    second = asr.transcribe_bytes(tmp_path, audio)
    assert not second['cache_hit'] and len(calls) == 4
    assert first['dependency_versions'] != second['dependency_versions']
    assert len(list((tmp_path/'transcripts').glob('*.json'))) == 2


def test_installed_dependency_drift_requires_restart(monkeypatch):
    monkeypatch.setattr(asr, '_installed_dependencies', lambda: {'faster-whisper': 'changed-on-disk'})
    with pytest.raises(ValueError, match='dependencies changed; restart'):
        asr._model()



def test_quality_scipy_dependency_drift_requires_restart(monkeypatch):
    from dubbing_service.app.audiobook import reference_quality
    assert "scipy" in asr._LOADED_DEPENDENCIES
    changed = dict(asr._LOADED_DEPENDENCIES) | {"scipy": "changed-on-disk"}
    monkeypatch.setattr(asr, "_installed_dependencies", lambda: changed)
    with pytest.raises(ValueError, match="dependencies changed; restart"):
        reference_quality._pipeline()
