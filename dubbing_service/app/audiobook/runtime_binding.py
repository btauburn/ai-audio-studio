"""Fail-closed worker, Take cache and durable-job execution binding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from .fingerprint import voice_revision_for_role


class RuntimeBindingError(RuntimeError):
    pass


def canonical_backend(value):
    return 'qwen3tts' if value in {'qwen3tts', 'qwen3tts_voicedesign'} else value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def validate_worker(value):
    if (not isinstance(value, dict) or not isinstance(value.get('contract'), dict)
            or value.get('sha256') != digest(value['contract'])
            or value['contract'].get('schema') != 'qwen-base-worker-identity-v1'
            or value['contract'].get('runtime_profile') not in {'native-base', 'faster-base-032'}
            or value['contract'].get('fallback') != 'disabled'):
        raise RuntimeBindingError('Worker runtime identity is missing or invalid; no synthesis or cache reuse is allowed')
    return value


def engine_components(binding):
    worker = binding['worker']['contract']
    return {'model_id': worker['model_id'], 'model_revision': worker['checkpoint_manifest_sha256'],
            'runtime_version': binding['sha256'], 'decoder_version': worker['packages'].get('qwen-tts') or 'unknown'}


def require_take(take, binding):
    actual = (take.generation_metadata or {}).get('runtime_binding')
    if not actual or actual.get('sha256') != binding['sha256'] or digest(actual['contract']) != actual['sha256']:
        raise RuntimeBindingError('Selected Take has missing or different runtime provenance; start a new complete rendition explicitly, preserving the prior audio')


def check_scope(segments, binding):
    for segment in segments:
        take = next((t for t in segment.takes if t.id == segment.selected_take_id), None)
        if take is not None:
            require_take(take, binding)


def bind_job(project_id, job_id, renderer, segments, roles, aliases=None, storage_project_id=None):
    """Persist before dispatch; old/mismatched jobs cannot append mixed output."""
    binding = renderer.runtime_contract() if hasattr(renderer, 'runtime_contract') else None
    if binding is None:
        return None  # Explicit fixture/other-engine seam; never a real Qwen fallback.
    from dubbing_service.app.storage.jobs import mutate_job
    from dubbing_service.app.models.jobs import SynthesisJobStatus
    aliases = aliases or {}
    by_role = {r.id: r for r in roles}
    segments = [s for s in segments if (s.tts_backend or getattr(by_role.get(aliases.get(s.role_id, s.role_id)), 'tts_backend', None)) is not None and canonical_backend(s.tts_backend or getattr(by_role.get(aliases.get(s.role_id, s.role_id)), 'tts_backend', None)) == canonical_backend(renderer.backend)]
    role_ids = {aliases.get(s.role_id, s.role_id) for s in segments}
    scope = {'renderer': binding, 'utterance_ids': [s.id for s in segments],
        'voices': {r.id: {'voice_revision': voice_revision_for_role(r),
                         'identity': r.voice_identity.model_dump(mode='json') if r.voice_identity else None}
                   for r in roles if r.id in role_ids}}
    receipt = {'contract': scope, 'sha256': digest(scope)}
    check_scope(segments, binding)
    def commit(job):
        if job.stop_requested or job.status == SynthesisJobStatus.CANCELLED:
            raise RuntimeBindingError('Cancelled job cannot dispatch synthesis')
        key = canonical_backend(renderer.backend)
        prior = job.runtime_bindings.get(key)
        relevant_completed = [sid for sid in job.completed_segment_ids
            if not job.segment_backends.get(sid) or canonical_backend(job.segment_backends[sid]) == key]
        if prior is None and relevant_completed:
            raise RuntimeBindingError('Legacy job has completed output but no runtime binding; explicit whole-job restart is required')
        if prior is not None:
            old = prior.get('contract', {})
            compatible = (prior.get('sha256') == digest(old)
                and old.get('renderer') == binding
                and set(scope['utterance_ids']).issubset(old.get('utterance_ids', []))
                and all(old.get('voices', {}).get(k) == v for k, v in scope['voices'].items()))
            if not compatible:
                raise RuntimeBindingError('Job renderer, checkpoint, settings, chunking or reference version changed; resume is refused before synthesis')
            # Existing governed link/skip handling may reduce scope; it never
            # adds new work or changes the pinned runtime/reference versions.
            return
        job.runtime_bindings[key] = receipt
    mutate_job(storage_project_id or project_id, job_id, commit)
    return binding


RENDERER_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).with_name('renderer.py').read_bytes()).hexdigest()
