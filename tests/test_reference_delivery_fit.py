"""Fixed delivery policy fixtures with explicit bound listening judgments.

Synthetic tones verify measurement and provenance mechanics, not a trained
semantic detector. Delivery fixture labels represent hearing evidence.
"""
import hashlib

import numpy as np
import pytest

from dubbing_service.app.audiobook.reference_delivery_fit import evaluate


def wave(gain=.1):
    rate = 16000
    time = np.arange(rate*6)/rate
    envelope = np.where((time % 2) < 1.5,1,.001)
    return (gain*np.sin(2*np.pi*180*time)*envelope).astype(np.float32), rate


def assessment(audio, delivery):
    return {'method':'human_listening','human_verified':True,'actor':'fixture-listener',
        'evidence_ref':'disposable-fixture-listening-label', 'source_id':'fixture-source',
        'start_seconds':0,'end_seconds':6,'input_pcm_sha256':hashlib.sha256(audio.tobytes()).hexdigest(),
        'delivery':delivery,'observations':['Observable delivery label from fixed fixture.']}


@pytest.mark.parametrize('label,delivery,technical,gain,expected', [
    ('clean crowd-facing yelling','audience_directed_shouting','pass',.1,'reject'),
    ('calm clean reading','calm_reading','pass',.1,'accept'),
    ('expressive close-miked narration','expressive_narration','pass',.1,'accept'),
    ('clean stage interview','stage_interview','pass',.1,'review'),
    ('clipped compressed reading','calm_reading','pass',1.2,'reject'),
    ('low-volume noisy insufficient speech','calm_reading','reject',.001,'reject'),
])
def test_fixed_delivery_fixtures(label,delivery,technical,gain,expected):
    audio, rate = wave(gain)
    result = evaluate(audio,rate,source={'source_id':'fixture-source','title':label},
        start_seconds=0,end_seconds=6,technical_status=technical,
        audio_assessment=assessment(audio,delivery))
    assert result['delivery_decision'] == expected
    assert result['selected_reference'] is (expected == 'accept')
    assert result['identity_claim'] is None
    if delivery == 'audience_directed_shouting':
        assert result['technical_decision'] == 'pass'


def test_metadata_and_large_dynamic_range_cannot_waive_missing_audio_assessment():
    audio,rate = wave()
    result = evaluate(audio,rate,source={'source_id':'fixture-source','title':'Calm audiobook narration'},
        start_seconds=0,end_seconds=6,technical_status='pass')
    assert result['delivery_decision'] == 'review' and result['outcome'] == 'no_suitable_reference'
    assert result['metrics']['semantic_shouting_probability'] is None and result['score'] is None


@pytest.mark.parametrize('field,value', [('input_pcm_sha256','f'*64),('source_id','wrong'),
    ('start_seconds',1),('end_seconds',5),('human_verified',False),('actor',''),('observations',[])])
def test_changed_span_or_unbound_assessment_cannot_accept(field,value):
    audio,rate = wave()
    heard = assessment(audio,'calm_reading')
    heard[field] = value
    result = evaluate(audio,rate,source={'source_id':'fixture-source'},start_seconds=0,end_seconds=6,
        technical_status='pass',audio_assessment=heard)
    assert not result['selected_reference'] and result['audio_assessment'] is None


@pytest.mark.parametrize('gain',[.001,.1,.8])
def test_gain_and_clean_compression_do_not_reject_expressive_narration(gain):
    audio,rate = wave(gain)
    audio = (.8*np.tanh(audio)).astype(np.float32)
    result = evaluate(audio,rate,source={'source_id':'fixture-source','context':'Live event reading'},
        start_seconds=0,end_seconds=6,technical_status='pass',audio_assessment=assessment(audio,'expressive_narration'))
    assert result['delivery_decision'] == 'accept'


def test_actual_clipping_cannot_be_waived_by_calm_metadata_or_listening_label():
    audio,rate = wave(1.4)
    audio = np.clip(audio,-1,1)
    result = evaluate(audio,rate,source={'source_id':'fixture-source','title':'Quiet reading'},
        start_seconds=0,end_seconds=6,technical_status='pass',audio_assessment=assessment(audio,'calm_reading'))
    assert result['technical_decision'] == 'reject' and 'ACTUAL_CLIPPING' in result['reason_codes']
