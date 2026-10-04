"""Read-only global voice library access using the existing library owner."""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request

from dubbing_service.app.audiobook import voice_lab_library as lab_library
from dubbing_service.app.audiobook.delivery.server import verified_audio_response
from dubbing_service.app.core.config import get_config
from dubbing_service.app.storage import voices as library

router = APIRouter(prefix="/voice-library")


@router.get("/recordings")
def recordings():
    from dubbing_service.app.audiobook.recordings_library import list_recordings
    return list_recordings()


@router.api_route("/recordings/{project_id}/{export_id}/audio", methods=["GET", "HEAD"])
def recording_audio(project_id: str, export_id: str, request: Request, download: bool = False):
    from dubbing_service.app.audiobook.recordings_library import audio_artifact
    try:
        path, checksum = audio_artifact(project_id, export_id)
        return verified_audio_response(path, checksum, request, export_id, download)
    except (OSError, ValueError):
        raise HTTPException(409, "This recording is unavailable or changed") from None


@router.get("/status")
def status():
    return {"service": "ai-audio-studio-voice-library", "pid": os.getpid(),
            "projects_root": str(get_config().projects_root_abs)}


@router.get("")
def versions():
    rows = lab_library.list_versions()
    for voice in rows:
        for version in voice["versions"]:
            version["audio_url"] = (f"/api/v1/voice-library/versions/{voice['library_voice_id']}/"
                                    f"{version['version']}/audio")
    return {"voices": rows}


@router.api_route("/versions/{library_voice_id}/{version}/audio", methods=["GET", "HEAD"])
def audio(library_voice_id: str, version: str, request: Request):
    try:
        with library.voices_lock():
            voice = next((v for v in library.load_voices() if v.id == library_voice_id), None)
            if voice is None or version not in voice.voice_lab_versions:
                raise HTTPException(404, "Saved voice version not found")
            entry = lab_library._validated_entry(version, voice.voice_lab_versions[version])
            sample = entry["assets"]["sample"]
            path = lab_library._safe_path(library.voices_audio_dir(), sample["path"])
        return verified_audio_response(path, sample["sha256"], request, 0, False)
    except (OSError, ValueError, KeyError):
        raise HTTPException(409, "This saved voice sample is unavailable or changed") from None
