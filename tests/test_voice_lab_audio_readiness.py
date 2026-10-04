from hashlib import sha256
from types import SimpleNamespace

import pytest

from dubbing_service.app.audiobook import voice_lab as lab, voice_lab_audition as auditions


@pytest.mark.parametrize('state', ['valid', 'missing', 'changed', 'conditioning_only'])
def test_ready_audio_requires_actual_hash_verified_bytes(tmp_path, monkeypatch, state):
    sample = tmp_path / 'sample.wav'
    sample.write_bytes(b'RIFF preserved audio bytes')
    checksum = sha256(sample.read_bytes()).hexdigest()
    record = {'key': 'v_test', 'name': 'Harry', 'audio_file': 'sample.wav', 'audio_sha256': checksum}
    if state == 'missing':
        sample.unlink()
    elif state == 'changed':
        sample.write_bytes(b'RIFF changed audio bytes')
    elif state == 'conditioning_only':
        record.pop('audio_file')
        record.pop('audio_sha256')
    voice_lab = SimpleNamespace(revision=1, scope='production', preferences={}, casting={}, shortlists={}, review_position={}, automation={}, excerpt=None, issues=[])
    project = SimpleNamespace(id='test', name='Test', voice_lab=voice_lab, metadata={}, roles=[])
    monkeypatch.setattr(lab, 'load_project', lambda _: project)
    monkeypatch.setattr(lab, 'voice_records', lambda _: {'v_test': record})
    monkeypatch.setattr(lab, 'safe_project_root', lambda _: tmp_path)
    monkeypatch.setattr(auditions, 'audition_summaries', lambda _: [])
    result = lab.view('test')['voices'][0]
    assert result['audio_available'] is (state == 'valid')
    assert bool(result['audio_url']) is (state == 'valid')
    if state in {'missing', 'changed'}:
        assert result['audio_error']
