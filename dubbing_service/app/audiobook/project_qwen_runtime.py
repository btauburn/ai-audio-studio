"""Project runtime choice is independent of voice identity and saved defaults."""
from __future__ import annotations

from .runtime_binding import RuntimeBindingError, DIRECTION_RUNTIME_PROFILE
from ..core.tts_backends import DEFAULT_QWEN_BASE_PROFILE

ALLOWED = {'native-current', 'faster-base-032', DIRECTION_RUNTIME_PROFILE}


def require_direction_pilot(project_id):
    from dubbing_service.app.storage.projects import load_project
    from dubbing_service.app.core.config import get_config
    from .performance_pilot import read_pilot, PILOT_INDEX_SCHEMA, LEDGER_SCHEMA
    from pathlib import Path
    project = load_project(project_id)
    state = read_pilot(project_id)
    index, ledger = state['index'], state['ledger']
    if (project.voice_lab.scope != 'isolated_listening_test'
            or index.get('schema_version') != PILOT_INDEX_SCHEMA
            or ledger.get('schema_version') != LEDGER_SCHEMA
            or index.get('pilot_id') != project_id or ledger.get('pilot_id') != project_id
            or index.get('kind') != 'review_only' or index.get('production_enablement') is not False
            or Path(index.get('authorized_destination_root') or '').resolve() != get_config().projects_root_abs.resolve()
            or not index.get('scope_sha256')):
        raise RuntimeBindingError('Direction runtime requires an isolated review-only performance pilot')
    return project


def project_profile(project_id):
    from dubbing_service.app.storage.projects import load_project
    project = load_project(project_id)
    profile = project.metadata.get('requested_runtime') or DEFAULT_QWEN_BASE_PROFILE
    if profile not in ALLOWED:
        raise RuntimeBindingError('Unsupported project Qwen runtime; choose native-current or faster-base-032 explicitly')
    if profile == DIRECTION_RUNTIME_PROFILE:
        require_direction_pilot(project_id)
    return profile


def actual_profile(health):
    return health.get('runtime_profile') or 'native-current'


def require_profile(worker, profile):
    actual = worker['contract']['runtime_profile']
    expected = 'native-base' if profile == 'native-current' else profile
    if actual != expected:
        raise RuntimeBindingError(f'Waiting for {"Faster Qwen" if profile == "faster-base-032" else "native Qwen"} worker: requested {profile}, active {actual}. No fallback or mixed-runtime generation is allowed.')
