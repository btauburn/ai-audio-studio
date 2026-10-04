"""Default promotion boundaries; no live worker, audio or project data is used."""
import pytest
from fastapi import HTTPException

from dubbing_service.app.audiobook import project_qwen_runtime as policy
from dubbing_service.app.audiobook import voice_lab_audition as preview
from dubbing_service.app.routers import settings
from dubbing_service.app.storage import projects
from tests.test_voice_lab_audition import setup  # isolated fixture


@pytest.mark.parametrize('backend,explicit,stored,expected', [
    ('qwen3tts', None, 'native-current', 'faster-base-032'),
    ('qwen3tts', None, 'native-sdpa', 'faster-base-032'),
    ('qwen3tts', None, 'qwen-cudagraph', 'faster-base-032'),
    ('qwen3tts', 'native-current', 'qwen-cudagraph', 'native-current'),
    ('qwen3tts_cv', None, 'native-current', 'native-current'),
    ('qwen3tts_cv', None, 'qwen-cudagraph', 'qwen-cudagraph'),
])
def test_service_start_resolves_backend_specific_default_before_any_process_change(
        monkeypatch, backend, explicit, stored, expected):
    monkeypatch.setattr(settings, '_get_tts_params_from_config',
                        lambda: {'qwen3_execution_profile': stored})
    seen = []
    def capture(*args, **kwargs):
        seen.append(kwargs['qwen_execution_profile'])
        raise FileNotFoundError('Disposable fixture; no process may start')
    monkeypatch.setattr(settings, 'build_backend_command', capture)
    with pytest.raises(HTTPException) as error:
        settings._start_tts_service_exclusive(settings.TTSServiceStartPayload(
            backend=backend, qwen_execution_profile=explicit))
    assert error.value.status_code == 503
    assert seen == [expected]


@pytest.mark.parametrize('explicit,expected', [(None, 'faster-base-032'), ('native-current', 'native-current')])
def test_new_base_audition_freezes_runtime_and_child_preserves_it(setup, monkeypatch, explicit, expected):
    _, project, role, _, _, _ = setup
    if explicit:
        project.metadata['requested_runtime'] = explicit
        projects.save_project(project)
    intent_hash, binding = preview._bound_intent(project, role.id, 'test-candidate')
    assert binding['requested_runtime'] == expected
    project.metadata[preview._META] = {'schema': 'voice-lab-auditions-v1', 'requests': {},
        'intents': {intent_hash: {'role_id': role.id, 'candidate_id': 'test-candidate',
            'binding': binding, 'candidate_snapshot': project.voice_lab.shortlists[role.id]['matches'][0],
            'reuse_key': None, 'job_id': 'fixture-job', 'attempts': 0}}}
    projects.save_project(project)
    monkeypatch.setattr(policy, 'DEFAULT_QWEN_BASE_PROFILE', 'native-current')
    child_id = preview._prepare_project(project.id, intent_hash)
    assert projects.load_project(child_id).metadata['requested_runtime'] == expected
    retained = projects.load_project(project.id).metadata[preview._META]['intents'][intent_hash]
    assert retained['binding'] == binding
    assert intent_hash in projects.load_project(project.id).metadata[preview._META]['intents']


def test_legacy_stored_request_is_reused_without_rehash_or_new_dispatch(setup, monkeypatch):
    from dubbing_service.app.models.jobs import SynthesisJob, SynthesisJobStatus
    from dubbing_service.app.storage.jobs import upsert_job
    _, project, role, _, body, _ = setup
    _, binding = preview._bound_intent(project, role.id, 'test-candidate')
    binding.pop('requested_runtime', None)  # saved before default promotion
    old_hash = preview.lab.digest(binding)
    job_id = 'audition_' + old_hash[:24]
    project.metadata[preview._META] = {'schema': 'voice-lab-auditions-v1',
        'requests': {body['request_id']: {'intent': old_hash,
            'payload_hash': preview.lab.digest([role.id, 'test-candidate', body['expected_identity_id'], None, None]),
            'expected_identity_id': body['expected_identity_id']}},
        'intents': {old_hash: {'role_id': role.id, 'candidate_id': 'test-candidate',
            'binding': binding, 'job_id': job_id, 'attempts': 0}}}
    projects.save_project(project)
    upsert_job(project.id, SynthesisJob(id=job_id, project_id=project.id,
        job_type='voice_lab_listening', status=SynthesisJobStatus.CANCELLED))
    def forbidden(*args, **kwargs):
        raise AssertionError('A stored request must not be rehashed or dispatched')
    monkeypatch.setattr(preview, '_bound_intent', forbidden)
    result = preview.audition(project.id, **body, schedule=forbidden)
    assert result['status'] == 'cancelled'
    assert result['job_id'] == job_id
    assert projects.load_project(project.id).metadata[preview._META]['intents'][old_hash]['binding'] == binding


def test_builtin_customvoice_reuse_does_not_acquire_base_profile(setup, monkeypatch):
    _, project, role, _, _, _ = setup
    monkeypatch.setattr(preview, '_candidate_for_intent', lambda *args:
        {'candidate_id': 'builtin', 'original_casting': True, 'reuse_key': 'heard-builtin'})
    monkeypatch.setattr(preview, '_reuse_record', lambda *args:
        {'builtin_snapshot': {'settings': {'tts_backend': 'qwen3tts_cv'}}})
    _, binding = preview._bound_intent(project, role.id, 'builtin', reuse_key='heard-builtin')
    assert 'requested_runtime' not in binding


def test_base_faster_profile_cannot_be_launched_as_customvoice(tmp_path):
    from dubbing_service.app.core.tts_backends import build_backend_command
    with pytest.raises(ValueError, match='unsupported Qwen execution profile'):
        build_backend_command(tmp_path, 'qwen3tts_cv', tmp_path / 'python',
                              qwen_execution_profile='faster-base-032')
