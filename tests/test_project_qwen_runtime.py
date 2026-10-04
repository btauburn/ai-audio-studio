from types import SimpleNamespace

import pytest

from dubbing_service.app.audiobook import project_qwen_runtime as policy, voice_lab_runtime as runtime
from dubbing_service.app.audiobook.runtime_binding import RuntimeBindingError
from dubbing_service.app.core import config, tts_operations
from dubbing_service.app.routers import settings
from dubbing_service.app.storage import jobs, projects
from tests.test_runtime_binding import worker


@pytest.mark.parametrize('profile,actual,allowed', [('native-current','native-base',True), ('faster-base-032','faster-base-032',True), ('faster-base-032','native-base',False), ('native-current','faster-base-032',False)])
def test_required_profile_never_falls_back(profile, actual, allowed):
    if allowed:
        policy.require_profile(worker(actual), profile)
    else:
        with pytest.raises(RuntimeBindingError, match='Waiting for'):
            policy.require_profile(worker(actual), profile)


def test_profile_is_separate_metadata_and_approved_faster_is_default(monkeypatch):
    project = SimpleNamespace(metadata={})
    monkeypatch.setattr(projects, 'load_project', lambda _: project)
    assert policy.project_profile('p') == 'faster-base-032'
    project.metadata['requested_runtime'] = 'native-current'
    assert policy.project_profile('p') == 'native-current'
    project.metadata['requested_runtime'] = 'faster-base-032'
    assert policy.project_profile('p') == 'faster-base-032'
    project.metadata['requested_runtime'] = 'unapproved-other-model'
    with pytest.raises(RuntimeBindingError): policy.project_profile('p')


@pytest.mark.parametrize('ownership,busy,pending,allowed', [(False,False,False,False), (True,True,False,False), (True,False,True,False), (True,False,False,True)])
def test_only_owned_idle_worker_can_switch_without_discarding_other_production(tmp_path, monkeypatch, ownership, busy, pending, allowed):
    (tmp_path / 'p').mkdir()
    (tmp_path / 'p/project.json').write_text('{}')
    monkeypatch.setattr(policy, 'project_profile', lambda _: 'faster-base-032')
    monkeypatch.setattr(config, 'get_config', lambda: SimpleNamespace(projects_root_abs=tmp_path))
    monkeypatch.setattr(settings, '_get_tts_runtime_state', lambda: (True, 123, 'qwen3tts'))
    monkeypatch.setattr(settings, '_tts_process', SimpleNamespace(poll=lambda: None) if ownership else None)
    monkeypatch.setattr(tts_operations, 'has_active_tts_operation', lambda: busy)
    other = SimpleNamespace(id='other', job_type='batch_synthesize', status=SimpleNamespace(value='queued'))
    monkeypatch.setattr(jobs, 'load_jobs', lambda _: [other] if pending else [])
    ready = False
    monkeypatch.setattr(runtime, '_healthy', lambda backend, profile=None: ready)
    calls = []
    def start(payload):
        nonlocal ready
        calls.append(payload)
        ready = True
    monkeypatch.setattr(settings, 'api_start_tts_service', start)
    if allowed:
        runtime.ensure_backend_ready('qwen3tts', project_id='p', job_id='own')
        assert len(calls) == 1
        assert calls[0].qwen_execution_profile == 'faster-base-032'
        assert calls[0].persist_default is False and calls[0].preserve_outputs is True
    else:
        with pytest.raises(ValueError, match='Waiting for Faster Qwen'):
            runtime.ensure_backend_ready('qwen3tts', project_id='p', job_id='own')
        assert calls == []


@pytest.mark.parametrize('requested,actual', [('faster-base-032','native-base'), ('native-current','faster-base-032')])
def test_generic_factory_character_boundary_rejects_wrong_worker_before_http(monkeypatch, requested, actual):
    from dubbing_service.app.providers.tts import TTSProviderFactory
    monkeypatch.setattr(policy, 'project_profile', lambda _: requested)
    provider = TTSProviderFactory.create('qwen3tts')
    monkeypatch.setattr(provider, 'runtime_identity', lambda: worker(actual))
    calls = []
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: calls.append(a))
    with pytest.raises(RuntimeBindingError, match='Waiting for'):
        provider._character_request('synthesize', {'project_id':'p', 'text':'Original text'})
    assert not calls


def test_direct_provider_preserves_prebound_digest_and_rejects_worker_change(monkeypatch):
    from dubbing_service.app.providers.tts import TTSProviderFactory
    monkeypatch.setattr(policy, 'project_profile', lambda _: 'faster-base-032')
    provider = TTSProviderFactory.create('qwen3tts')
    original = worker('faster-base-032')
    provider._expected_runtime_identity = original
    monkeypatch.setattr(provider, 'runtime_identity', lambda: original)
    provider._bind_project_runtime('p')
    assert provider._expected_runtime_identity is original
    changed = worker('faster-base-032')
    changed['contract']['implementation_sha256']['test'] = 'c'*64
    from dubbing_service.app.audiobook.runtime_binding import digest
    changed['sha256'] = digest(changed['contract'])
    monkeypatch.setattr(provider, 'runtime_identity', lambda: changed)
    with pytest.raises(RuntimeBindingError, match='bound rendition'):
        provider._bind_project_runtime('p')
    assert provider._expected_runtime_identity is original


def test_direct_legacy_synthesis_rejects_project_mismatch_before_http(monkeypatch):
    from dubbing_service.app.models.project import Role, Segment
    from dubbing_service.app.providers.tts import TTSProviderFactory
    monkeypatch.setattr(policy, 'project_profile', lambda _: 'faster-base-032')
    provider = TTSProviderFactory.create('qwen3tts')
    monkeypatch.setattr(provider, 'runtime_identity', lambda: worker('native-base'))
    calls = []
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: calls.append(a))
    result = provider.synthesize_segment('p', Role(id='r',name='R'), Segment(id='s',index=0,text='Original text',role_id='r'))
    assert not result.success and 'Waiting for' in result.raw['error']
    assert not calls
