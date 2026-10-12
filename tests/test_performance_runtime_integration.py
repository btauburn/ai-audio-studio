"""CPU composition: real M3 pilot, M4 lookup, M5 transport and registration."""
import copy
from pathlib import Path

import pytest
from dubbing_service.app.audiobook import performance_director as pd, performance_provider as provider
from dubbing_service.app.audiobook import project_qwen_runtime, provider_capabilities, pipeline
from dubbing_service.app.models.project import Role, SourceAnchor, Chapter
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.storage.projects import load_project, save_project
from tests.audiobook_helpers import build_segments, build_project
from tests.test_performance_pilot_assembly import studio
from tests.test_performance_provider import FakeWorker, identity_for


@pytest.fixture
def directed(studio):
    root = Path(__file__).resolve().parents[1]
    if not (root / 'environments/faster_qwen_base032/.venv/Lib/site-packages/faster_qwen3_tts/model.py').is_file():
        pytest.skip('Pinned local Faster library is not installed in this source-only checkout')
    project = load_project(studio['pilot_id'])
    identity = CharacterVoiceIdentity(id='synthetic-voice', role_id='character', status='LOCKED',
        backend='qwen3tts', mechanism='qwen_reference', reference_audio_sha256='a'*64,
        prompt_sha256='b'*64, prompt_artifact='synthetic/prompt.npz', reference_audio='synthetic/reference.wav',
        dossier_revision='synthetic', casting_profile_revision='synthetic', design_instruction='synthetic CPU only', model='synthetic-not-TTS')
    role = Role(id='character', name='Synthetic Character', role_type='character', voice_identity=identity)
    project.roles = [role]
    project.chapters = [Chapter(id='chap0', title='Synthetic CPU Chapter', index=0)]
    project.metadata['requested_runtime'] = provider.PROFILE
    save_project(project)
    segment = build_segments(project, ['A synthetic CPU protocol line.'], role_id=role.id)[0]
    capability = pd._live_capability(project, segment)
    assert capability['controls']['seed']['state'] == 'forwarded_experimental'
    row = pd.UtteranceDirection(utterance_id=segment.id, role_id=role.id,
        text_sha256=pd.text_sha256(segment.text), voice_revision=pd.voice_revision_for_role(role),
        resolved=pd.ResolvedControls(native_instruction='Calm and warm.', seed_salt=7),
        dispositions=[pd.ControlDisposition(control='native_instruction', state='forward_native')],
        sources=pd.DirectionSources(capability_sha256=capability['sha256'], resolver_version='synthetic-integration'))
    stored = pd.write_direction_set(project.id, pd.DirectionSet(project_id=project.id, directions=[row]))
    pd.commit_index(project.id, 0, lambda index: index.update(enabled=True) or index['active'].update(direction_set=stored['sha256']),
                    actor='synthetic-test', event_id='synthetic-direction')
    return load_project(project.id), role, segment, row


def test_canonical_provider_sends_the_exact_requirement_component(directed, monkeypatch):
    project, role, segment, row = directed
    worker = FakeWorker(identity_for(provider.PROFILE))
    monkeypatch.setattr(provider.urllib.request, 'urlopen', worker)
    client = provider.DirectedQwen3TTSProvider()
    result = client.synthesize_segment(project.id, role, segment)
    assert result.success, result.raw
    sent = worker.synth_posts()[0][2]['performance']
    requirement = pipeline.requirement_for(project, segment, role=role)
    assert sent['direction_sha256'] == requirement.performance_direction == pd.direction_component(row)
    assert sent['native_instruction'] == 'Calm and warm.' and sent['seed_salt'] == 7


def test_a_held_canonical_direction_sends_no_http(directed, monkeypatch):
    project, role, segment, row = directed
    index = project.metadata[pd.INDEX_KEY]
    path = pd._project_root(project.id) / pd.DIRECTION_DIRECTORY / (index['active']['direction_set'] + '.json')
    path.write_text('{broken', encoding='utf-8')
    calls = []
    monkeypatch.setattr(provider.urllib.request, 'urlopen', lambda *args, **kw: calls.append(args))
    result = provider.DirectedQwen3TTSProvider().synthesize_segment(project.id, role, segment)
    assert not result.success and not calls


def test_ordinary_projects_cannot_select_the_experimental_runtime(studio):
    ordinary = load_project(studio['project'].id)
    ordinary.metadata['requested_runtime'] = provider.PROFILE
    save_project(ordinary)
    with pytest.raises((RuntimeError, ValueError)):
        project_qwen_runtime.project_profile(ordinary.id)


def test_a_real_synthetic_pilot_can_select_the_profile(directed):
    project, *_ = directed
    assert project_qwen_runtime.project_profile(project.id) == provider.PROFILE


def test_new_launch_mapping_uses_existing_environment_without_default_change():
    from dubbing_service.app.core.tts_backends import build_backend_command, DEFAULT_QWEN_BASE_PROFILE
    from dubbing_service.app.routers.settings import TTSServiceStartPayload
    root = Path(__file__).resolve().parents[1]
    if not (root / 'environments/faster_qwen_base032/.venv/Scripts/python.exe').is_file():
        pytest.skip('Launch mapping requires the separately installed pinned Faster environment')
    command = build_backend_command(root, 'qwen3tts', root / '.venv/Scripts/python.exe',
                                    qwen_execution_profile=provider.PROFILE, preserve_outputs=True)
    assert 'faster_qwen_base032' in command[0]
    assert str(root / 'service/faster_qwen3tts_base_direction_server.py') in command
    assert DEFAULT_QWEN_BASE_PROFILE == 'faster-base-032'
    request = TTSServiceStartPayload(qwen_execution_profile=provider.PROFILE, persist_default=False)
    assert request.persist_default is False


def test_m1_capability_probe_is_unchanged_and_only_m5_extension_reports_seed(directed):
    project, role, segment, _ = directed
    identity = role.voice_identity.model_dump(mode='json')
    old = provider_capabilities.performance_capabilities(provider.PROFILE, identity)
    new = provider_capabilities.experimental_direction_capabilities(provider.PROFILE, identity)
    assert old['controls']['seed']['state'] == 'unsupported'
    assert new['controls']['seed']['state'] == 'forwarded_experimental'
    default = provider_capabilities.experimental_direction_capabilities('faster-base-032', identity)
    assert default['controls']['seed']['state'] == 'unsupported'


def test_m5_provider_preserves_actual_pilot_gate_with_injected_lookup(directed, monkeypatch):
    project, role, segment, row = directed
    ordinary = build_project(name='Ordinary synthetic project')
    forged = provider.ResolvedDirection(ordinary.id, role.id, role.voice_identity.id, pd.text_sha256(segment.text),
                                       row.resolved.native_instruction, row.resolved.seed_salt, pd.direction_component(row))
    client = provider.DirectedQwen3TTSProvider(lambda *_: forged)
    worker = FakeWorker(identity_for(provider.PROFILE))
    monkeypatch.setattr(provider.urllib.request, 'urlopen', worker)
    result = client.synthesize_segment(ordinary.id, role, segment)
    assert not result.success and not worker.requests


@pytest.mark.parametrize('instruction', ['Speak 25% faster.', 'Imitate another performer.', 'Sound like another voice.'])
def test_transport_gates_refuse_voice_or_numeric_rate_requests(instruction):
    from service.faster_qwen3tts_base_direction_adapter import resolve_controls, DirectionError
    with pytest.raises((ValueError, RuntimeError)):
        provider.resolve_controls(instruction)
    with pytest.raises(DirectionError):
        resolve_controls(instruction)
