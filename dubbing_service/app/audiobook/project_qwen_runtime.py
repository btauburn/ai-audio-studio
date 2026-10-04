"""Project runtime choice is independent of voice identity and saved defaults."""
from __future__ import annotations

from .runtime_binding import RuntimeBindingError
from ..core.tts_backends import DEFAULT_QWEN_BASE_PROFILE

ALLOWED = {'native-current', 'faster-base-032'}


def project_profile(project_id):
    from dubbing_service.app.storage.projects import load_project
    project = load_project(project_id)
    profile = project.metadata.get('requested_runtime') or DEFAULT_QWEN_BASE_PROFILE
    if profile not in ALLOWED:
        raise RuntimeBindingError('Unsupported project Qwen runtime; choose native-current or faster-base-032 explicitly')
    return profile


def actual_profile(health):
    return health.get('runtime_profile') or 'native-current'


def require_profile(worker, profile):
    actual = worker['contract']['runtime_profile']
    expected = 'native-base' if profile == 'native-current' else profile
    if actual != expected:
        raise RuntimeBindingError(f'Waiting for {"Faster Qwen" if profile == "faster-base-032" else "native Qwen"} worker: requested {profile}, active {actual}. No fallback or mixed-runtime generation is allowed.')
