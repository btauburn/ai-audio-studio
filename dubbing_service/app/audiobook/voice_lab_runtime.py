"""On-demand worker readiness through Studio's existing process owner.

Starting the library never calls this. Explicit generation dispatch calls it
before taking its synthesis lease; another worker is never silently stopped.
"""
from __future__ import annotations

import threading
import time

import httpx

from dubbing_service.app.core.tts_backends import get_backend_spec
from .runtime_binding import RuntimeBindingError

_start_lock = threading.Lock()


def _healthy(backend: str, profile: str | None = None) -> bool:
    spec = get_backend_spec(backend)
    try:
        with httpx.Client(timeout=2, trust_env=False) as client:
            response = client.get(spec.base_url + "/health")
        data = response.json()
        if not response.is_success or data.get("backend") != backend:
            return False
        if profile is not None:
            from .runtime_binding import validate_worker
            from .project_qwen_runtime import require_profile
            with httpx.Client(timeout=3, trust_env=False) as client:
                identity_response = client.get(spec.base_url + "/api/v1/tts/runtime-identity")
            if not identity_response.is_success:
                return False
            require_profile(validate_worker(identity_response.json()), profile)
        return True
    except (httpx.HTTPError, ValueError, RuntimeBindingError):
        return False


def ensure_backend_ready(backend: str, *, job_id: str | None = None, project_id: str | None = None) -> None:
    """Reuse a healthy worker, or start an absent worker without changing defaults."""
    get_backend_spec(backend)
    from .project_qwen_runtime import project_profile
    profile = project_profile(project_id) if project_id and backend == "qwen3tts" else None
    healthy = lambda: _healthy(backend, profile) if profile is not None else _healthy(backend)
    if healthy():
        return
    from dubbing_service.app.routers import settings
    from dubbing_service.app.core.config import get_config
    from dubbing_service.app.core.tts_operations import has_active_tts_operation
    from dubbing_service.app.storage.jobs import load_jobs
    from .book_creator_recovery import recoverable_idle_book_job

    with _start_lock:
        if healthy():
            return
        running, _, current = settings._get_tts_runtime_state()
        if running and current != backend:
            raise ValueError("Another voice worker is active. Finish its work before starting this preview or book.")
        if running and profile and not healthy():
            # The existing owner/lease can switch its own idle worker only.
            # Never kill an external process or unload a competing model here.
            pending = any(job.id != job_id and job.job_type in {"batch_synthesize", "single_synthesize"}
                          and job.status.value in {"queued", "running"}
                          and not recoverable_idle_book_job(path.parent.name, job)
                          for path in get_config().projects_root_abs.glob("*/project.json")
                          for job in load_jobs(path.parent.name))
            owned = settings._tts_process is not None and settings._tts_process.poll() is None
            if not owned or has_active_tts_operation() or pending:
                label = "Faster Qwen" if profile == "faster-base-032" else "native Qwen"
                raise ValueError(f"Waiting for {label} worker. The active worker is busy or externally owned; no fallback was used. Resume this saved request after the operator activates the selected runtime.")
            settings.api_start_tts_service(settings.TTSServiceStartPayload(
                backend=backend, persist_default=False, preserve_outputs=True,
                qwen_execution_profile=profile))
        if not running:
            pending = any(job.id != job_id and job.job_type in {"batch_synthesize", "single_synthesize"}
                          and job.status.value in {"queued", "running"}
                          and not recoverable_idle_book_job(path.parent.name, job)
                          for path in get_config().projects_root_abs.glob("*/project.json")
                          for job in load_jobs(path.parent.name))
            if has_active_tts_operation() or pending:
                raise ValueError("Existing production has priority. Resume after it finishes.")
            settings.api_start_tts_service(settings.TTSServiceStartPayload(
                backend=backend, persist_default=False, only_if_absent=True, preserve_outputs=True, qwen_execution_profile=profile))
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if healthy():
                return
            time.sleep(.5)
        raise ValueError("The voice worker has not become ready. Check Studio's worker status and retry this request.")
