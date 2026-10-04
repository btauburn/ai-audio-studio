"""Bounded narration delivery evidence; acoustic proxies never certify shouting."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

SCHEMA = 'narration-delivery-fit-v1'
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
TARGET = 'sustained_conversational_storytelling'
_ACCEPT = {'calm_reading', 'expressive_narration', 'conversational_storytelling'}
_REJECT = {'audience_directed_shouting', 'sustained_yelling', 'exaggerated_crowd_address'}


def evaluate(wave, sample_rate, *, source, start_seconds, end_seconds,
             technical_status, transcript=None, audio_assessment=None):
    """Return a separate delivery decision for this exact PCM/span.

    No installed, calibrated semantic delivery detector is currently available.
    Unknown audio delivery is held for listening, even if source metadata says
    audiobook and technical QA passes. Human judgments must bind these frames.
    """
    if (isinstance(sample_rate, bool) or not isinstance(sample_rate, int)
            or not 8000 <= sample_rate <= 96000 or not isinstance(wave, np.ndarray)
            or wave.dtype != np.float32 or wave.ndim not in (1, 2)
            or not np.isfinite(wave).all() or not 3 <= len(wave) / sample_rate <= 30):
        raise ValueError('Invalid delivery-fit audio')
    if (wave.ndim == 2 and wave.shape[1] != 2) or not isinstance(source, dict) or not source.get('source_id'):
        raise ValueError('Invalid delivery-fit source')
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (start_seconds, end_seconds)) or start_seconds < 0
            or abs(end_seconds - start_seconds - len(wave) / sample_rate) > .02):
        raise ValueError('Delivery-fit span differs from assessed audio')
    if technical_status not in {'pass', 'hold', 'reject'}:
        raise ValueError('Unknown technical quality decision')
    pcm_sha = hashlib.sha256(wave.tobytes()).hexdigest()
    mono = wave.mean(1) if wave.ndim == 2 else wave
    # Preserve raw frames. The comparison copy normalizes recording gain only;
    # relative energy is not VAD, a crowd detector, or a semantic prosody model.
    peak = float(np.max(np.abs(wave)))
    rms = float(np.sqrt(np.mean(np.asarray(mono, dtype=np.float64) ** 2)))
    clipping = float(np.mean(np.abs(wave) >= .999))
    frame = max(1, round(.02 * sample_rate))
    trimmed = mono[:len(mono)//frame*frame].reshape(-1, frame)
    levels = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    analysis = levels / max(rms, np.finfo(float).tiny)
    db = 20 * np.log10(np.maximum(analysis, 1e-6))
    active = analysis > .15
    starts = np.flatnonzero(np.diff(np.r_[False, active, False].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, active, False].astype(int)) == -1)
    pauses = (starts[1:] - ends[:-1]) * frame / sample_rate
    metrics = {'raw_peak': peak, 'raw_rms': rms, 'clipped_fraction': clipping,
        'normalized_intensity_range_db_p95_p10': float(np.percentile(db,95)-np.percentile(db,10)),
        'relative_energy_active_fraction': float(np.mean(active)),
        'relative_energy_phrase_seconds': ((ends-starts)*frame/sample_rate).tolist(),
        'relative_energy_pause_seconds': pauses.tolist(),
        'pitch_hz': None, 'speech_rate_wpm': None, 'semantic_shouting_probability': None,
        'missing': ['reliable_pitch', 'aligned_speech_rate', 'calibrated_semantic_delivery_detector'],
        'limitations': 'Energy/dynamics are weak gain/EQ/compression/content-confounded proxies; no yelling, overlap, age, accent, or identity inference.'}
    reasons = []
    technical = 'reject' if peak > 1 or clipping > .001 else technical_status
    if technical == 'reject':
        reasons.append('TECHNICAL_FAILURE')
    if clipping > .001 or peak > 1:
        reasons.append('ACTUAL_CLIPPING')
    # Low gain and clean compression are deliberately not delivery rejection.
    context = ' '.join(str(source.get(k) or '') for k in ('title','source_label','context','description')).casefold()
    context_flags = [term for term in ('crowd','stage','rally','interview','live event') if term in context]
    if context_flags:
        reasons.append('SOURCE_CONTEXT_REQUIRES_AUDIO_ASSESSMENT')
    assessment = audio_assessment if isinstance(audio_assessment, dict) else {}
    bound = (assessment.get('method') == 'human_listening' and assessment.get('human_verified') is True
        and bool(assessment.get('actor')) and bool(assessment.get('evidence_ref'))
        and assessment.get('input_pcm_sha256') == pcm_sha
        and assessment.get('source_id') == source['source_id']
        and assessment.get('start_seconds') == start_seconds and assessment.get('end_seconds') == end_seconds
        and isinstance(assessment.get('observations'), list) and bool(assessment['observations']))
    delivery = assessment.get('delivery') if bound else None
    if technical != 'pass':
        decision = 'reject' if technical == 'reject' else 'review'
    elif bound and delivery in _REJECT:
        decision = 'reject'
        reasons.append('OBSERVED_NARRATION_DELIVERY_MISMATCH')
    elif bound and delivery in _ACCEPT:
        decision = 'accept'
        reasons.append('BOUND_LISTENING_ASSESSMENT_SUPPORTS_NARRATION')
    else:
        decision = 'review'
        reasons.append('INSUFFICIENT_AUDIO_DELIVERY_EVIDENCE')
    return {'schema': SCHEMA, 'pipeline_source_sha256': SOURCE_SHA256,
        'target_profile': TARGET, 'input_pcm_sha256': pcm_sha, 'sample_rate': sample_rate,
        'source': {k: source.get(k) for k in ('source_id','source_page','title','source_label','context')},
        'start_seconds': start_seconds, 'end_seconds': end_seconds,
        'duration_seconds': len(wave)/sample_rate,
        'transcript_sha256': hashlib.sha256(transcript.encode()).hexdigest() if transcript else None,
        'technical_decision': technical, 'delivery_decision': decision,
        'selected_reference': decision == 'accept' and technical == 'pass',
        'outcome': 'suitable_reference' if decision == 'accept' and technical == 'pass' else 'no_suitable_reference',
        'reason_codes': reasons, 'context_flags': context_flags, 'metrics': metrics,
        'score': None, 'evidence_strength': 'bound_human_listening' if bound else 'delivery_unknown',
        'audio_assessment': assessment if bound else None,
        'identity_claim': None, 'production_approved': False}
