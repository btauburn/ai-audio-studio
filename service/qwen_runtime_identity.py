"""Stable identity of the actual Base worker, without loading weights."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def source_snapshot(root, paths):
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def worker_identity(root, args, sources, profile='native-base'):
    expected = (root / 'models/qwen3_tts/Qwen3-TTS-Base').resolve()
    if (root / args.base_model_dir).resolve() != expected:
        raise ValueError('Runtime binding requires the exact approved Base snapshot; alternate checkpoints require a separate explicit integration')
    manifest = root / 'models/manifests/qwen3_tts_base.json'
    payload = json.loads(manifest.read_text('utf-8'))
    versions = {}
    for package in ['qwen-tts', 'torch', 'transformers', 'faster-qwen3-tts']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    # Faster is not a dependency of the native worker even if installed nearby.
    if profile == 'native-base':
        versions.pop('faster-qwen3-tts')
    contract = {
        'schema': 'qwen-base-worker-identity-v1', 'runtime_profile': profile,
        'model_id': payload['model_id'],
        'checkpoint_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'checkpoint_files': payload['files'],
        'base_model_path': str((root / args.base_model_dir).resolve()),
        'dtype': args.dtype, 'device': args.device, 'attention': 'sdpa',
        'packages': versions, 'implementation_sha256': sources,
        'inference_policy_revision': 'qwen-base-sampled-v1',
        'generation_policy': {'do_sample': True, 'subtalker_dosample': True,
            'temperature': .9, 'top_k': 50, 'top_p': 1., 'repetition_penalty': 1.05,
            'subtalker_top_k': 50, 'subtalker_top_p': 1., 'subtalker_temperature': .9,
            'seed_rule': 'sha256(identity_id + text) first 8 hex digits',
            'budget': 'min(2048,max(128,ceil((word_count*.7+4)*12.5)))',
            'non_streaming_mode': True, 'append_silence': False},
        'fallback': 'disabled',
    }
    return {'contract': contract, 'sha256': digest(contract)}


class WorkerIdentitySnapshot:
    """Freeze at process import; reject changed disk/config provenance."""
    def __init__(self, root, args, sources, profile):
        self.root, self.sources, self.profile = root, sources.copy(), profile
        self.frozen = worker_identity(root, args, sources, profile)

    def read(self, args):
        current = worker_identity(self.root, args, self.sources, self.profile)
        if current != self.frozen:
            raise ValueError('Worker checkpoint, dependencies or configuration changed on disk; restart the worker before continuing')
        import copy
        return copy.deepcopy(self.frozen)
