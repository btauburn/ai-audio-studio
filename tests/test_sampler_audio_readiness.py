from hashlib import sha256
from types import SimpleNamespace
import pytest
from dubbing_service.app.audiobook import voice_lab as lab, voice_lab_audition as auditions

@pytest.mark.parametrize('state', ['valid', 'missing', 'changed'])
def test_sampler_play_requires_verified_actual_bytes(tmp_path, monkeypatch, state):
    path = tmp_path / 'sampler.mp3'
    path.write_bytes(b'preserved sampler bytes')
    record = {'audio_file': 'sampler.mp3', 'audio_sha256': sha256(path.read_bytes()).hexdigest(), 'book_not_supplied': True}
    if state == 'missing': path.unlink()
    if state == 'changed': path.write_bytes(b'changed sampler')
    project = SimpleNamespace(id='p', name='Test', metadata={}, roles=[], voice_lab=SimpleNamespace(revision=1,scope='production',preferences={},casting={},shortlists={},review_position={},automation={},issues=[],excerpt=record))
    monkeypatch.setattr(lab, 'load_project', lambda _: project)
    monkeypatch.setattr(lab, 'safe_project_root', lambda _: tmp_path)
    monkeypatch.setattr(lab, 'voice_records', lambda _: {})
    monkeypatch.setattr(lab, 'comparison_records', lambda _: {})
    monkeypatch.setattr(auditions, 'audition_summaries', lambda _: [])
    excerpt = lab.view('p')['excerpt']
    assert excerpt['audio_available'] is (state == 'valid')
    assert bool(excerpt['audio_url']) is (state == 'valid')
