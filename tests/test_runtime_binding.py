"""Runtime provenance tests use synthetic signals, never checkpoints/GPU."""
import copy
import io
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from dubbing_service.app.audiobook import pipeline, renderer
from dubbing_service.app.audiobook.runtime_binding import (
    RuntimeBindingError, bind_job, digest, validate_worker,
)
from dubbing_service.app.models.jobs import SynthesisJob, SynthesisJobStatus
from dubbing_service.app.providers.tts import Qwen3TTSProvider, TTSResult
from dubbing_service.app.storage.jobs import get_job, upsert_job, mutate_job
from dubbing_service.app.storage.projects import save_project
from dubbing_service.app.storage.segments import save_segments, load_segments
from tests.audiobook_helpers import build_project, build_segments, make_tone_wav, use_projects_root


def worker(profile='native-base'):
    contract = {'schema': 'qwen-base-worker-identity-v1', 'runtime_profile': profile,
        'fallback': 'disabled', 'model_id': 'TEST-SIGNAL-NOT-TTS',
        'checkpoint_manifest_sha256': 'a'*64, 'generation_policy': {'top_p': 1.},
        'packages': {'qwen-tts': '0.1.1'}, 'implementation_sha256': {'test': 'b'*64}}
    return {'contract': contract, 'sha256': digest(contract)}


class SignalProvider:
    def __init__(self, root):
        self.root, self.identity, self.calls, self.fail = root, worker(), 0, None
    def runtime_identity(self):
        if self.identity is None:
            raise OSError('worker unavailable')
        return validate_worker(copy.deepcopy(self.identity))
    def synthesize_segment(self, project_id, role, segment):
        self.calls += 1
        if self.fail:
            return TTSResult(False, {'error': self.fail})
        path = make_tone_wav(self.root/f'{self.calls}.wav', seconds=1.2)
        return TTSResult(True, {'worker_runtime_identity': self.identity, 'fallback': None}, str(path))


@pytest.fixture
def studio(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    project = build_project()
    project.roles[0].tts_backend = 'qwen3tts'
    save_project(project)
    rows = build_segments(project, ['A short signal test.', 'Another signal test.'])
    for row in rows:
        row.tts_backend = 'qwen3tts'
    save_segments(project.id, rows)
    provider = SignalProvider(tmp_path/'signals')
    def fresh():
        return renderer.BackendRenderer(backend='qwen3tts', engine_id='qwen3tts', renderer_id='qwen3tts',
            project_root=pipeline.project_root(project.id), provider=provider)
    upsert_job(project.id, SynthesisJob(id='runtime-job', project_id=project.id, job_type='batch_synthesize',
        segment_ids=[r.id for r in rows], segment_backends={r.id:'qwen3tts' for r in rows}))
    return project, rows, provider, fresh


def test_actual_runtime_enters_job_and_take_fingerprint_and_resumes(studio):
    project, rows, provider, fresh = studio
    first = pipeline.render_utterances(project.id, [rows[0].id], renderer=fresh(), resume=True,
        promote=True, run_qa=False, job_id='runtime-job')[0]
    assert first.success and first.selected
    selected = load_segments(project.id)[0]
    take = selected.takes[0]
    assert take.fingerprint_components['runtime_version'] == take.generation_metadata['runtime_binding']['sha256']
    assert pipeline._requirement_for_take(project, selected, take).fingerprint() == take.input_fingerprint
    job = get_job(project.id, 'runtime-job')
    assert job.runtime_bindings['qwen3tts']['contract']['utterance_ids'] == [r.id for r in rows]
    # QA can be missing in this fixture; exact runtime is validated before re-audit.
    from dubbing_service.app.audiobook.runtime_binding import check_scope
    check_scope([selected], fresh().runtime_contract())
    assert provider.calls == 1


@pytest.mark.parametrize('change', ['profile', 'checkpoint', 'settings', 'implementation', 'chunking', 'reference'])
def test_job_mismatch_refuses_before_append_and_preserves_completed(studio, change, monkeypatch):
    project, rows, provider, fresh = studio
    bind_job(project.id, 'runtime-job', fresh(), rows, project.roles)
    mutate_job(project.id, 'runtime-job', lambda job: job.completed_segment_ids.append(rows[0].id))
    if change == 'chunking':
        monkeypatch.setattr(renderer, 'CHUNK_MAX_CHARS', 200)
    elif change == 'reference':
        project.roles[0].speaker_audio = 'changed-reference.wav'
        save_project(project)
    else:
        key = {'profile':'runtime_profile','checkpoint':'checkpoint_manifest_sha256',
            'settings':'generation_policy','implementation':'implementation_sha256'}[change]
        provider.identity['contract'][key] = 'faster-base-032' if change == 'profile' else ('c'*64 if change == 'checkpoint' else {'changed': True})
        provider.identity['sha256'] = digest(provider.identity['contract'])
    before = get_job(project.id, 'runtime-job').model_dump()
    with pytest.raises(RuntimeBindingError, match='changed'):
        pipeline.render_utterances(project.id, [rows[1].id], renderer=fresh(), resume=True,
            promote=True, run_qa=False, job_id='runtime-job')
    assert provider.calls == 0
    assert get_job(project.id, 'runtime-job').model_dump() == before
    assert not load_segments(project.id)[1].takes


def test_legacy_partial_job_fails_without_rewriting_progress(studio):
    project, rows, provider, fresh = studio
    mutate_job(project.id, 'runtime-job', lambda j: j.completed_segment_ids.append(rows[0].id))
    before = get_job(project.id, 'runtime-job').model_dump()
    with pytest.raises(RuntimeBindingError, match='Legacy'):
        pipeline.render_utterances(project.id, [rows[1].id], renderer=fresh(), resume=True, job_id='runtime-job')
    assert get_job(project.id, 'runtime-job').model_dump() == before
    assert provider.calls == 0


def test_mixed_job_non_qwen_completion_does_not_block_first_qwen_binding(studio):
    project, rows, provider, fresh = studio
    rows[0].tts_backend = 'cosyvoice3';save_segments(project.id, rows)
    def other_complete(j):
        j.segment_backends[rows[0].id]='cosyvoice3';j.completed_segment_ids=[rows[0].id]
    mutate_job(project.id, 'runtime-job', other_complete)
    bind_job(project.id, 'runtime-job', fresh(), rows, project.roles)
    scope = get_job(project.id, 'runtime-job').runtime_bindings['qwen3tts']['contract']
    assert scope['utterance_ids'] == [rows[1].id]


def test_backend_alias_is_bound_to_same_worker(studio):
    project, rows, provider, fresh = studio
    r = fresh();r.backend='qwen3tts_voicedesign'
    rows[1].tts_backend='qwen3tts_voicedesign'
    bind_job(project.id, 'runtime-job', r, rows, project.roles)
    assert set(get_job(project.id, 'runtime-job').runtime_bindings) == {'qwen3tts'}


def test_cancelled_job_and_unavailable_worker_do_not_dispatch(studio):
    project, rows, provider, fresh = studio
    mutate_job(project.id, 'runtime-job', lambda j: setattr(j, 'status', SynthesisJobStatus.CANCELLED))
    with pytest.raises(RuntimeBindingError, match='Cancelled'):
        pipeline.render_utterances(project.id, [rows[0].id], renderer=fresh(), job_id='runtime-job')
    provider.identity=None
    with pytest.raises(OSError, match='unavailable'):
        fresh().runtime_contract()
    assert provider.calls == 0


def test_bound_failure_is_not_automatically_retried(studio):
    project, rows, provider, fresh = studio
    provider.fail='generation cancelled'
    record = pipeline.render_utterances(project.id, [rows[0].id], renderer=fresh(), job_id='runtime-job')[0]
    assert not record.success and 'cancelled' in record.error
    assert provider.calls == 1
    assert not load_segments(project.id)[0].takes


def test_provider_rejects_changed_response_and_explicit_fallback(monkeypatch):
    provider=Qwen3TTSProvider();provider.base_url='http://127.0.0.1:9999'
    provider._expected_runtime_identity=worker()
    sent=[]
    for result in [{'success':True,'worker_runtime_identity':worker('faster-base-032')},
                   {'success':True,'worker_runtime_identity':worker(),'fallback':'native'}]:
        def response(request, **kwargs):
            sent.append(json.loads(request.data));return io.BytesIO(json.dumps(result).encode())
        monkeypatch.setattr('urllib.request.urlopen',response)
        with pytest.raises(RuntimeError, match='mismatched runtime or fallback'):
            provider._character_request('synthesize', {'text':'Test'})
    assert all(p['expected_runtime_sha256']==worker()['sha256'] for p in sent)


def test_native_worker_guard_checks_before_any_synthesis(monkeypatch):
    import service.qwen3tts_server as native
    monkeypatch.setattr(native,'config',None)
    calls=[]
    def synthesize(**kwargs):
        calls.append(kwargs);return {'success':True,'audio_file':'TEST-SIGNAL'}
    monkeypatch.setattr(native._character_runtime,'synthesize',synthesize)
    with TestClient(native.app) as client:
        identity=client.get('/api/v1/tts/runtime-identity').json()
        payload={'project_id':'test','identity_id':'test','prompt_artifact':'test','prompt_sha256':'a'*64,
            'reference_audio':'test','reference_audio_sha256':'b'*64,'text':'Test', 'expected_runtime_sha256':'0'*64}
        assert client.post('/api/v1/tts/character/synthesize',json=payload).status_code==409
        assert calls==[]
        payload['expected_runtime_sha256']=identity['sha256']
        result=client.post('/api/v1/tts/character/synthesize',json=payload)
        assert result.status_code==200 and result.json()['worker_runtime_identity']==identity
        assert len(calls)==1 and 'expected_runtime_sha256' not in calls[0]


def test_process_snapshot_refuses_changed_manifest_packages_or_path(monkeypatch):
    import service.qwen_runtime_identity as identity
    current=worker();monkeypatch.setattr(identity,'worker_identity',lambda *args:copy.deepcopy(current))
    snapshot=identity.WorkerIdentitySnapshot(None,None,{},'native-base')
    assert snapshot.read(None)==worker()
    current['contract']['packages']['qwen-tts']='changed';current['sha256']=digest(current['contract'])
    with pytest.raises(ValueError, match='restart'):
        snapshot.read(None)
