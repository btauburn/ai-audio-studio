"""Versioned Voice Lab profiles in Studio's existing global voice library.

Call only inside the Studio owner process, like the existing voices router.
No model loading, network, cast assignment, or preference transfer. Existing bounded local decoder checks acquired references.
Audio artifacts are exclusive-created and never replaced/deleted. Metadata is
published through the existing library/project owners and their normal locks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.storage import voices as library
from dubbing_service.app.storage.projects import load_project, project_meta_lock, save_project
from .voice_lab import VoiceLabError, digest, voice_records

MAX_ASSET_BYTES = 128 * 1024 * 1024
MAX_VERSION_BYTES = 256 * 1024 * 1024
MAX_VERSIONS_PER_VOICE = 100
MAX_LIBRARY_VERSIONS = 1000
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}
REVIEW_FIELDS = {"favorite", "rating", "suitability", "audio_url", "human_review"}


class VoiceLibraryError(VoiceLabError):
    pass


def _safe_path(root: Path, relative: str) -> Path:
    """Reject lexical traversal, ADS, absolute paths and existing reparse hops."""
    if not isinstance(relative, str) or not relative or ":" in relative:
        raise VoiceLibraryError("Invalid library artifact path")
    parts = relative.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts) or Path(relative).is_absolute():
        raise VoiceLibraryError("Artifact path traversal is refused")
    root = root.resolve(strict=True)
    candidate = root.joinpath(*parts)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink() or (current.exists() and getattr(current.lstat(), "st_file_attributes", 0) & 0x400):
            raise VoiceLibraryError("Linked library artifacts are refused")
    if not candidate.resolve().is_relative_to(root):
        raise VoiceLibraryError("Artifact is outside its owner")
    return candidate


def _read_verified(path: Path, expected: str) -> bytes:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise VoiceLibraryError("Artifact has no valid exact hash")
    if not path.is_file() or path.stat().st_size > MAX_ASSET_BYTES:
        raise VoiceLibraryError("Artifact is missing or exceeds the library size limit")
    with path.open("rb") as handle:
        content = handle.read(MAX_ASSET_BYTES + 1)
    if not content or len(content) > MAX_ASSET_BYTES or hashlib.sha256(content).hexdigest() != expected:
        raise VoiceLibraryError("Artifact bytes no longer match this voice version")
    return content


def _publish_blob(root: Path, relative: str, content: bytes, expected: str) -> str:
    path = _safe_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _read_verified(path, expected)
        return relative
    # The owner locks serialize normal calls; exclusive creation also protects
    # against accidental path reuse. A partial failed write is never overwritten.
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        _read_verified(path, expected)
    return relative


def _assets(record: dict) -> dict[str, tuple[str, str]]:
    if not record.get("audio_file") or not record.get("audio_sha256"):
        raise VoiceLibraryError("Publish a voice after its production-path sample exists")
    if Path(record["audio_file"]).suffix.lower() not in AUDIO_EXTENSIONS:
        raise VoiceLibraryError("Production sample has an unsupported format")
    assets = {"sample": (record["audio_file"], record["audio_sha256"])}
    identity = record.get("identity") or {}
    if identity:
        parsed = CharacterVoiceIdentity.model_validate(identity)
        if parsed.mechanism == "builtin":
            if not parsed.builtin_speaker:
                raise VoiceLibraryError("Builtin voice has no provider voice identity")
        elif parsed.mechanism == "qwen_reference":
            for field, hash_field, extensions in (("reference_audio", "reference_audio_sha256", AUDIO_EXTENSIONS),
                                                  ("prompt_artifact", "prompt_sha256", {".npz"})):
                relative = identity.get(field)
                if not relative or Path(relative).suffix.lower() not in extensions or not identity.get(hash_field):
                    raise VoiceLibraryError("Reusable conditioning is incomplete")
                assets[field] = (relative, identity[hash_field])
        else:
            raise VoiceLibraryError("This voice mechanism cannot yet be reused across projects")
    elif not record.get("builtin_speaker"):
        raise VoiceLibraryError("Historical audio has no reusable conditioning identity")
    details = record.get("details") or {}
    if details.get("take_proof_file") or details.get("take_proof_sha256"):
        if (not details.get("take_proof_file") or Path(details["take_proof_file"]).suffix.lower() != ".json"
                or not re.fullmatch(r"[a-f0-9]{64}", str(details.get("take_proof_sha256") or ""))):
            raise VoiceLibraryError("Audition evidence mapping is incomplete")
        assets["audition_proof"] = (details["take_proof_file"], details["take_proof_sha256"])
    return assets



def _valid_version_key(record: dict) -> bool:
    key = record.get("key")
    if key == "v_" + digest([record.get("voice_id"), record.get("version")])[:28]:
        return True
    intent = record.get("audition_intent_hash")
    if not isinstance(intent, str) or not re.fullmatch(r"[a-f0-9]{64}", intent):
        return False
    identity = (record.get("identity") or {}).get("id") or record.get("builtin_speaker")
    return bool(identity and key == "v_" + digest([identity, record.get("version"), intent])[:28])


def _audition_proof_binding(record: dict, content: bytes) -> str | None:
    if len(content) > 16 * 1024 * 1024:
        raise VoiceLibraryError("Audition evidence exceeds its size bound")
    try:
        proof = json.loads(content)
    except (ValueError, TypeError) as exc:
        raise VoiceLibraryError("Audition evidence is malformed") from exc
    if not isinstance(proof, dict) or proof.get("schema") != "voice-lab-audition-takes-v2":
        raise VoiceLibraryError("Audition evidence schema is unsupported")
    takes = proof.get("takes") or []
    identity_id = (record.get("identity") or {}).get("id")
    if (not takes or proof.get("assembled_audio_sha256") != record.get("audio_sha256")
            or proof.get("project_id") != record.get("audition_project_id")
            or [item.get("id") for item in takes] != record.get("take_ids")
            or not (proof.get("assembled_qa") or {}).get("passed")):
        raise VoiceLibraryError("Audition proof does not bind the heard sample")
    languages = set()
    language_unknown = False
    for take in takes:
        fingerprint = take.get("fingerprint_components") or {}
        stamp = (take.get("generation_metadata") or {}).get("character_voice_identity") or {}
        qa = take.get("qa_report") or {}
        if (fingerprint.get("voice_revision") != record.get("version")
                or (identity_id and stamp.get("id") != identity_id)
                or qa.get("take_id") != take.get("id")
                or qa.get("output_audio_hash") != take.get("audio_sha256") or not qa.get("passed")):
            raise VoiceLibraryError("Audition proof has an unbound identity or Take")
        language = fingerprint.get("language")
        if language and take.get("language") == language:
            languages.add(language)
        else:
            language_unknown = True
    if language_unknown:
        return None
    if len(languages) != 1:
        raise VoiceLibraryError("Audition proof contains conflicting generation languages")
    return next(iter(languages))


def _publication_record(project_id: str, source: dict) -> dict:
    record = copy.deepcopy(source)
    details = record.get("details") or {}
    proof_file, checksum = details.get("take_proof_file"), details.get("take_proof_sha256")
    if proof_file or checksum:
        match = re.fullmatch(r"voice_lab/auditions/([a-f0-9]{64})/take\.json", proof_file or "")
        if not match:
            # Imported immutable copies retain their proven intent explicitly.
            if not record.get("audition_intent_hash"):
                raise VoiceLibraryError("Audition evidence has no stable intent identity")
        else:
            record["audition_intent_hash"] = match.group(1)
        content = _read_verified(_safe_path(safe_project_root(project_id), proof_file), checksum)
        language = _audition_proof_binding(record, content)
        if record.get("generation_language") and language != record["generation_language"]:
            raise VoiceLibraryError("Recorded generation language conflicts with Take evidence")
        if language:
            record["generation_language"] = language
            record["generation_language_evidence"] = {"method": "immutable_selected_take_fingerprints",
                "take_proof_sha256": checksum, "sample_audio_sha256": record["audio_sha256"]}
    if record.get("audition_intent_hash") and not proof_file:
        raise VoiceLibraryError("Audition identity requires its immutable Take proof")
    if not _valid_version_key(record):
        raise VoiceLibraryError("Voice has no stable version identity")
    return record


def _validated_entry(version: str, entry: dict) -> dict:
    payload = {k: v for k, v in entry.items() if k != "manifest_sha256"}
    if digest(payload) != entry.get("manifest_sha256") or entry.get("record", {}).get("version") != version:
        raise VoiceLibraryError("Library voice metadata no longer matches its immutable version")
    record = entry["record"]
    if not _valid_version_key(record):
        raise VoiceLibraryError("Library voice identity does not match its version key")
    expected = _assets(record)
    assets = entry.get("assets") or {}
    if set(assets) != set(expected) or any(
        assets[name].get("sha256") != checksum or
        Path(assets[name].get("path", "")).suffix.lower() != Path(relative).suffix.lower()
        for name, (relative, checksum) in expected.items()
    ):
        raise VoiceLibraryError("Library artifact mapping does not match the voice identity")
    if record.get("audition_intent_hash"):
        asset = assets.get("audition_proof")
        if not asset:
            raise VoiceLibraryError("Audition identity requires its immutable Take proof")
        root = library.voices_audio_dir()
        language = _audition_proof_binding(record, _read_verified(_safe_path(root, asset["path"]), asset["sha256"]))
        if record.get("generation_language") and record["generation_language"] != language:
            raise VoiceLibraryError("Library generation language conflicts with immutable evidence")
    return copy.deepcopy(entry)


def _profile_fingerprint(record: dict) -> str:
    # Artifact locations and role binding vary across projects; the actual
    # voice/model/design/inspiration and content hashes must remain identical.
    identity = copy.deepcopy(record.get("identity") or {})
    for field in ("role_id", "reference_audio", "prompt_artifact"):
        identity.pop(field, None)
    return digest({**{k: record.get(k) for k in (
        "voice_id", "version", "name", "kind", "tags", "inspiration", "builtin_speaker", "backend")},
        "identity": identity})


def _entry_summary(library_id: str, version: str, entry: dict) -> dict:
    record = entry["record"]
    return {"library_voice_id": library_id, "version": version,
            **{k: copy.deepcopy(record.get(k)) for k in (
                "key", "name", "kind", "inspiration", "tags", "audio_sha256", "duration_seconds")}}


def list_versions() -> list[dict]:
    """Read-only list for the Studio/private proxy; no local paths or preferences."""
    with library.voices_lock():
        return [{"library_voice_id": voice.id, "name": voice.name,
                 "versions": [_entry_summary(voice.id, version, _validated_entry(version, entry))
                              for version, entry in voice.voice_lab_versions.items()]}
                for voice in library.load_voices() if voice.voice_lab_versions]


def publish_version(project_id: str, key: str) -> dict:
    """Append one immutable heard voice version to the existing voice library.

    Repeating the same publish verifies and returns the first saved sample.
    Changing identity/profile requires a new version; no existing blob changes.
    """
    # Never nest voices_lock -> project_meta_lock: the existing legacy voice
    # router also scans projects while holding voices_lock.
    with project_meta_lock(project_id):
        project = load_project(project_id)
        record = copy.deepcopy(voice_records(project).get(key))
    if not record:
        raise VoiceLibraryError("Voice version is unavailable")
    version = record.get("version")
    if not isinstance(version, str) or not version or len(version) > 128:
        raise VoiceLibraryError("Voice has no stable version identity")
    record = _publication_record(project_id, record)
    source_assets = _assets(record)
    source_root = safe_project_root(project_id)
    content = {name: _read_verified(_safe_path(source_root, relative), checksum)
               for name, (relative, checksum) in source_assets.items()}
    if sum(map(len, content.values())) > MAX_VERSION_BYTES:
        raise VoiceLibraryError("Voice version exceeds the library size limit")
    for field in REVIEW_FIELDS:
        record.pop(field, None)
    # Keep source/eligibility/model metadata; strip only mutable listening state.
    library_id = "vl_" + digest(record["voice_id"])[:24]
    version_folder = digest([library_id, version])[:24]
    with library.voices_lock():
        voices = library.load_voices()
        voice = next((v for v in voices if v.id == library_id), None)
        if voice and version in voice.voice_lab_versions:
            existing = _validated_entry(version, voice.voice_lab_versions[version])
            if existing["record"]["audio_sha256"] != record["audio_sha256"]:
                raise VoiceLibraryError("This version is already bound to a different production sample")
            if _profile_fingerprint(existing["record"]) != _profile_fingerprint(record):
                raise VoiceLibraryError("Profile edits require a new voice version")
            root = library.voices_audio_dir()
            for asset in existing["assets"].values():
                _read_verified(_safe_path(root, asset["path"]), asset["sha256"])
            return {**_entry_summary(library_id, version, existing), "published": False}
        if (voice and len(voice.voice_lab_versions) >= MAX_VERSIONS_PER_VOICE) or sum(len(v.voice_lab_versions) for v in voices) >= MAX_LIBRARY_VERSIONS:
            raise VoiceLibraryError("Voice library version budget reached")
        root = library.voices_audio_dir()
        assets = {}
        for name, (relative, checksum) in source_assets.items():
            destination = f"voice_lab/{library_id}/{version_folder}/{name}-{checksum}{Path(relative).suffix.lower()}"
            assets[name] = {"path": _publish_blob(root, destination, content[name], checksum), "sha256": checksum, "bytes": len(content[name])}
        entry = {"schema_version": 1, "record": record, "assets": assets,
                 "source_project_id": project_id, "published_at": datetime.now(timezone.utc).isoformat()}
        entry["manifest_sha256"] = digest(entry)
        if voice is None:
            stamp = datetime.now(timezone.utc)
            voice = library.Voice(id=library_id, name=record["name"], tags=record.get("tags", []),
                description="Versioned Voice Lab profile; import through Voice Lab to preserve its conditioning.",
                audio_path=None, created_at=stamp, updated_at=stamp)
            voices.append(voice)
        voice.voice_lab_versions[version] = entry
        voice.updated_at = datetime.now(timezone.utc)
        library.save_voices(voices)
    return {**_entry_summary(library_id, version, entry), "published": True}


def import_version(target_project_id: str, library_voice_id: str, version: str) -> dict:
    """Copy a version into a project, leaving roles/Takes/preferences untouched."""
    with library.voices_lock():
        voice = next((v for v in library.load_voices() if v.id == library_voice_id), None)
        if voice is None or version not in voice.voice_lab_versions:
            raise VoiceLibraryError("Library voice version is unavailable")
        entry = _validated_entry(version, voice.voice_lab_versions[version])
        source_root = library.voices_audio_dir()
        content = {name: _read_verified(_safe_path(source_root, asset["path"]), asset["sha256"])
                   for name, asset in entry["assets"].items()}
    if sum(map(len, content.values())) > MAX_VERSION_BYTES:
        raise VoiceLibraryError("Voice version exceeds the import size limit")
    from dubbing_service.app.core.tts_operations import try_acquire_project_mutation, release_project_mutation
    lease = try_acquire_project_mutation(target_project_id, kind="voice_lab_library_import")
    if lease is None:
        raise VoiceLibraryError("This project is rendering; voice import must wait")
    try:
        from . import reference_portability
        try:
            bundle = reference_portability.prepare(entry,library_voice_id,content.get("prompt_artifact"),
                library.voices_audio_dir(),safe_project_root(entry["source_project_id"]))
        except (ValueError,OSError,KeyError) as exc:
            raise VoiceLibraryError("External reference portability held: " + str(exc)[:300]) from exc
        # Revalidation can replay existing CPU source processing. The mutation
        # lease serializes imports/renders; metadata reads/feedback remain free.
        portability = reference_portability.install(bundle,safe_project_root(target_project_id),library_voice_id,version,
            retain_lineage=target_project_id != entry["source_project_id"])
        with project_meta_lock(target_project_id):
            project = load_project(target_project_id)
            record = copy.deepcopy(entry["record"])
            key = record["key"]
            existing = voice_records(project).get(key)
            result = {"library_voice_id": library_voice_id, "version": version, "key": key,
                      "existing_cast_unchanged": True}
            root = safe_project_root(target_project_id)
            if existing and existing.get("audio_sha256"):
                if existing["audio_sha256"] != record["audio_sha256"]:
                    raise VoiceLibraryError("Project already has another heard sample for this version")
                if _profile_fingerprint(existing) != _profile_fingerprint(record):
                    raise VoiceLibraryError("Project voice identity conflicts with this library version")
                for _, (relative, checksum) in _assets(existing).items():
                    _read_verified(_safe_path(root, relative), checksum)
                if portability is not None and (existing.get("details") or {}).get("reference_portability") != portability:
                    existing["details"] = {**(existing.get("details") or {}),"reference_portability":portability}
                    project.voice_lab.voices[key] = existing
                    project.voice_lab.revision += 1
                    save_project(project)
                return {**result,"imported":False,"reference_portability":portability}
            if existing and _profile_fingerprint(existing) != _profile_fingerprint(record):
                raise VoiceLibraryError("Conditioning profile changed; import requires a new version")
            folder = digest([library_voice_id, version])[:24]
            for name, asset in entry["assets"].items():
                suffix = Path(asset["path"]).suffix.lower()
                if name == "sample":
                    destination = f"voice_lab/library/{library_voice_id}/{folder}/sample-{asset['sha256']}{suffix}"
                    record["audio_file"] = _publish_blob(root, destination, content[name], asset["sha256"])
                elif name == "audition_proof":
                    destination = f"voice_lab/library/{library_voice_id}/{folder}/take-proof-{asset['sha256']}{suffix}"
                    record["details"]["take_proof_file"] = _publish_blob(root, destination, content[name], asset["sha256"])
                else:
                    identity_id = record["identity"]["id"]
                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", identity_id):
                        raise VoiceLibraryError("Reusable conditioning identity is invalid")
                    destination = f"casting/voices/{identity_id}/{name}-{asset['sha256']}{suffix}"
                    record["identity"][name] = _publish_blob(root, destination, content[name], asset["sha256"])
            if portability is not None:
                record["details"] = {**(record.get("details") or {}),"reference_portability":portability}
            record.update(role_id=None, role_name="Library voice (unassigned)", human_review="pending",
                          library_voice_id=library_voice_id, library_version=version,
                          origin_project_id=entry["source_project_id"], origin_take_id=record.get("take_id"))
            record["details"] = {**record.get("details", {}), "library_voice_id": library_voice_id,
                                  "origin_project_id": entry["source_project_id"], "origin_take_id": record.get("take_id")}
            project.voice_lab.voices[key] = record
            project.voice_lab.revision += 1
            save_project(project)
            return {**result,"imported":True,"reference_portability":portability}
    finally:
        release_project_mutation(lease)
