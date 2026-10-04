"""Quality decisions, cache integrity, fixed acquisition API, and no source promotion."""
import hashlib
import io
import json

import numpy as np
import pytest
import soundfile as sf

from dubbing_service.app.audiobook import reference_quality as quality
from dubbing_service.app.audiobook import reference_transcripts as asr
from tests.test_voice_lab_reference_clone import fixture, acquire  # noqa: F401


def pcm(wave=None):
    wave = np.sin(np.arange(64000) / 17).astype(np.float32) * .1 if wave is None else wave
    stream = io.BytesIO()
    sf.write(stream, wave, 16000, format='WAV', subtype='FLOAT')
    return stream.getvalue()


def recognition(audio, text='Clear fixture words.', *, probability=.99, start=.2, end=3.8):
    return {'status': 'draft', 'machine_generated': True, 'audio_sha256': hashlib.sha256(audio).hexdigest(), 'text': text,
            'segments': [{'start': start, 'end': end, 'text': text, 'avg_logprob': -.1,
                          'no_speech_prob': .01, 'words': [{'start': start, 'end': end,
                                                          'word': text, 'probability': probability}]}]}


@pytest.fixture
def pipeline(monkeypatch):
    identity = {'version': 'fixture-cpu-v1'}
    calls = []
    monkeypatch.setattr(quality, '_pipeline', lambda: dict(identity))
    monkeypatch.setattr(quality, '_speech', lambda wave, rate: (wave, [{'start': 0., 'end': 4.}]))
    def transcribe(directory, audio, **kwargs):
        calls.append(hashlib.sha256(audio).hexdigest())
        return recognition(audio)
    monkeypatch.setattr(asr, 'transcribe_bytes', transcribe)
    return identity, calls


def test_clean_technical_pass_explicitly_leaves_isolation_identity_and_rights_unverified(tmp_path, pipeline):
    result = quality.verify_bytes(tmp_path, pcm(), 'Clear fixture words.')
    assert result['status'] == 'pass'
    assert result['metrics']['word_error_rate'] == 0
    assert not result['automatic_source_approval'] and not result['human_verified']
    assert not result['clone_eligibility_changed'] and result['source_isolation_status'] == 'unverified'
    assert result['checks']['music']['status'] == 'unavailable'
    assert result['checks']['overlap_speakers']['status'] == 'unavailable'
    assert result['device'] == 'cpu'


@pytest.mark.parametrize('wave', [np.zeros(64000, np.float32), np.ones(64000, np.float32)])
def test_silence_and_clipping_fail_even_with_confident_matching_asr(tmp_path, pipeline, wave):
    assert quality.verify_bytes(tmp_path, pcm(wave), 'Clear fixture words.')['status'] == 'fail'


@pytest.mark.parametrize('audio', [b'not audio', pcm(np.full(64000, np.nan, np.float32)), pcm(np.zeros((64000, 2), np.float32)), pcm(np.zeros(16000, np.float32))], ids=['malformed', 'nonfinite', 'stereo', 'short'])
def test_malformed_nonfinite_stereo_and_short_inputs_fail_without_inference(tmp_path, pipeline, audio):
    result = quality.verify_bytes(tmp_path, audio, 'Clear fixture words.')
    assert result['status'] == 'fail' and pipeline[1] == []


def test_missing_expected_text_and_low_confidence_are_review_not_pass(tmp_path, pipeline, monkeypatch):
    assert quality.verify_bytes(tmp_path, pcm())['status'] == 'review'
    monkeypatch.setattr(asr, 'transcribe_bytes', lambda d, a, **kw: recognition(a, probability=.2))
    result = quality.verify_bytes(tmp_path, pcm(), 'Clear fixture words.')
    assert result['status'] == 'review' and result['checks']['asr']['status'] == 'review'


def test_unrelated_transcript_fails_and_word_normalization_handles_case_punctuation(tmp_path, pipeline):
    assert quality.word_error_rate('CLEAR, fixture words!', 'Clear fixture words.') == 0
    assert quality.verify_bytes(tmp_path, pcm(), 'Completely different narration here.')['status'] == 'fail'


def test_tail_omission_and_timestamps_outside_crop_are_never_pass(tmp_path, pipeline, monkeypatch):
    monkeypatch.setattr(asr, 'transcribe_bytes', lambda d, a, **kw: recognition(a, end=2.))
    result = quality.verify_bytes(tmp_path, pcm(), 'Clear fixture words.')
    assert result['checks']['crop_coverage']['status'] == 'review'
    monkeypatch.setattr(asr, 'transcribe_bytes', lambda d, a, **kw: recognition(a, end=100.))
    assert quality.verify_bytes(tmp_path, pcm(), 'Clear fixture words altered.')['status'] == 'review'


def test_missing_model_is_review_and_no_speech_is_fail(tmp_path, pipeline, monkeypatch):
    def missing():
        raise ValueError('Local CPU model unavailable')
    monkeypatch.setattr(quality, '_pipeline', missing)
    assert quality.verify_bytes(tmp_path, pcm(), 'Clear fixture words.')['status'] == 'review'
    monkeypatch.setattr(quality, '_pipeline', lambda: {})
    monkeypatch.setattr(quality, '_speech', lambda wave, rate: (wave, []))
    assert quality.verify_bytes(tmp_path, pcm(), 'Clear fixture words.')['status'] == 'fail'


def test_cache_binds_exact_audio_transcript_pipeline_and_detects_tampering(tmp_path, pipeline):
    audio = pcm()
    first = quality.verify_bytes(tmp_path, audio, 'Clear fixture words.')
    assert quality.verify_bytes(tmp_path, audio, 'Clear fixture words.')['cache_hit']
    assert len(pipeline[1]) == 1
    quality.verify_bytes(tmp_path, audio, 'Changed transcript.')
    quality.verify_bytes(tmp_path, pcm(np.sin(np.arange(64000) / 19).astype(np.float32) * .1), 'Clear fixture words.')
    pipeline[0]['version'] = 'changed-cpu-v2'
    quality.verify_bytes(tmp_path, audio, 'Clear fixture words.')
    assert len(pipeline[1]) == 4
    pipeline[0]['version'] = 'fixture-cpu-v1'
    path = tmp_path / 'reference-quality' / (first['cache_key'] + '.json')
    blob = json.loads(path.read_text()); blob['result']['status'] = 'fabricated'; path.write_text(json.dumps(blob))
    assert quality.verify_bytes(tmp_path, audio, 'Clear fixture words.')['status'] == 'fail'
    assert len(pipeline[1]) == 4


def test_api_uses_fixed_hashes_and_preserves_immutable_receipt_and_clone_hold(fixture, pipeline, monkeypatch):  # noqa: F811
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from dubbing_service.app.core import config
    from dubbing_service.app.routers.voice_lab import router
    f = fixture
    acquired = acquire(f, audio_review=None)
    receipt = f.directory / 'acquisitions' / acquired['acquisition_id'] / 'acquisition.json'
    before = receipt.read_bytes()
    monkeypatch.setattr(config, 'safe_project_root', lambda _: f.project)
    app = FastAPI(); app.include_router(router, prefix='/api/v1'); client = TestClient(app)
    url = f'/api/v1/projects/fixture/voice-lab/references/{acquired["acquisition_id"]}/quality'
    payload = {'acquisition_sha256': acquired['acquisition_sha256'], 'expected_text': acquired['reference_text']}
    monkeypatch.setattr(asr, 'transcribe_bytes', lambda d, a, **kw: recognition(a, text=acquired['reference_text']))
    result = client.post(url, json=payload)
    assert result.status_code == 200 and result.json()['status'] == 'pass'
    assert receipt.read_bytes() == before and not acquired['clone_ready']
    listed = quality.current_quality(f.directory, acquired['acquisition_id'], acquired['acquisition_sha256'], acquired['reference_audio_sha256'], acquired['reference_text'])
    assert listed['status'] == 'pass'
    assert quality.current_quality(f.directory, acquired['acquisition_id'], acquired['acquisition_sha256'], acquired['reference_audio_sha256'], 'Changed words.')['status'] == 'stale'
    assert client.post(url, json=payload | {'acquisition_sha256': '0' * 64}).status_code == 409
    assert client.post(url, json=payload | {'expected_text': '', 'stars': 5}).status_code == 422
    (f.directory / acquired['reference_path']).write_bytes(b'tampered')
    assert quality.current_quality(f.directory, acquired['acquisition_id'], acquired['acquisition_sha256'], acquired['reference_audio_sha256'], 'Clear fixture words.')['status'] == 'fail'
    assert client.post(url, json=payload).status_code == 409


def test_asr_draft_is_self_consistency_not_independent_alignment(tmp_path, pipeline):
    result = quality.verify_bytes(tmp_path, pcm(), 'Clear fixture words.', expected_origin='machine_draft')
    assert result['status'] == 'review' and result['metrics']['word_error_rate'] == 0
    assert 'self-consistency' in result['checks']['transcript_alignment']['reason']


def test_changed_loaded_source_requires_restart(monkeypatch):
    monkeypatch.setattr(quality, '_LOADED_SOURCE_SHA', '0' * 64)
    with pytest.raises(ValueError, match='restart'):
        quality._pipeline()
    monkeypatch.setattr(asr, '_LOADED_SOURCE_SHA', '0' * 64)
    with pytest.raises(ValueError, match='restart'):
        asr._model()


def test_high_non_speech_energy_is_review_not_music_detection():
    wave = np.sin(np.arange(64000) / 17).astype(np.float32) * .1
    checks, _ = quality._evaluate(wave, 16000, [{'start': 1., 'end': 3.}], recognition(pcm()), 'Clear fixture words.')
    assert checks['background_energy']['status'] == 'review'
    assert checks['music']['status'] == 'unavailable'


def test_machine_draft_whitespace_and_later_generated_words_remain_unverified(fixture, pipeline, monkeypatch):  # noqa: F811
    from dubbing_service.app.core import config
    f = fixture
    acquired = acquire(f, transcript=None, audio_review=None)
    monkeypatch.setattr(config, 'safe_project_root', lambda _: f.project)
    result = quality.acquisition_quality('fixture', acquired['acquisition_id'], acquired['acquisition_sha256'], 'Clear fixture words.  ')
    assert result['status'] == 'review' and result['expected_text_origin'] == 'machine_draft'
    manual = acquire(f, transcript='Original supplied transcript.', audio_review=f.review)
    result = quality.acquisition_quality('fixture', manual['acquisition_id'], manual['acquisition_sha256'], 'Clear fixture words.')
    assert result['status'] == 'review' and result['expected_text_origin'] == 'unverified_entry'
