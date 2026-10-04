"""Evidence-bound migration of existing human ratings; no project writes or synthesis."""
from __future__ import annotations

import copy
import json
import re
import threading

from . import preferred_voices as preferred
from . import voice_lab as lab
from . import voice_lab_library as versions
from ..core.config import safe_project_root
from ..core.tts_operations import release_project_mutation, try_acquire_project_mutation
from ..storage import voices as library
from ..storage.projects import load_project, project_meta_lock

_LOCK = threading.Lock()
SCHEMA = "preferred-voice-migration-v1"


def _candidate(project, event):
    item = (project.voice_lab.casting.get("roles") or {}).get(event.get("role_id")) or {}
    matches = [(round_, candidate) for round_ in [*item.get("history", []), item.get("round") or {}]
               for candidate in round_.get("candidates", [])
               if candidate.get("candidate_id") == event.get("candidate_id")
               and candidate.get("key") == event.get("key")
               and candidate.get("audio_sha256") == event.get("audio_sha256")]
    if not matches:
        raise versions.VoiceLibraryError("Rating has no exact historical audition candidate")
    round_, candidate = matches[0]
    return {**copy.deepcopy(candidate), "_historical_round_direction": copy.deepcopy(round_.get("direction") or {})}



def _traits(record, candidate, event):
    traits = preferred.traits_from_record(record)
    evidence = []
    for row in [*(candidate.get("inferred_traits") or []), *(candidate.get("supported_traits") or [])]:
        if isinstance(row, dict) and row.get("dimension") and row.get("value"):
            traits[row["dimension"]] = row["value"]
            evidence.append(copy.deepcopy(row))
    # Use direction frozen before this heard round, never later requested edits.
    retained = " ".join((candidate.get("_historical_round_direction") or {}).get("retained") or [])
    affirmative = [clause for clause in re.split(r"[.!?;\n]", retained.replace("’", "'"))
                   if not re.search(r"\b(?:not|no|never|don't|without|avoid)\b", clause, re.I)]
    for dimension, value, pattern in (("accent", "British", r"\bbritish\b"),
                                       ("presentation", "masculine", r"\bmale\b")):
        if any(re.search(pattern, clause, re.I) for clause in affirmative):
            traits[dimension] = value
            evidence.append({"dimension": dimension, "value": value,
                             "level": "HISTORICAL_REQUESTED_DIRECTION", "source": retained,
                             "measured_acoustic_fact": False})
    return traits, evidence


def plan(project_id: str) -> dict:
    with project_meta_lock(project_id):
        project = load_project(project_id)
        before = lab.digest(project.model_dump(mode="json"))
        if project.voice_lab.scope == "regression_fixture":
            raise versions.VoiceLibraryError("Fixture approvals cannot enter the real keeper library")
        events = copy.deepcopy(project.voice_lab.casting.get("events") or {})
        rows, held, skipped = [], [], []
        if len(events) > 2000:
            raise versions.VoiceLibraryError("Migration event budget exceeded")
        for event_id, event in sorted(events.items()):
            stars = event.get("stars")
            if type(stars) is not int or stars not in (4, 5):
                skipped.append({"event_id": event_id, "reason": "No explicit four/five-star rating; selection and favorite are not ratings"})
                continue
            try:
                if event.get("event_id") != event_id or event.get("origin") != "human" or not event.get("actor"):
                    raise versions.VoiceLibraryError("Rating has no original human actor provenance")
                candidate = _candidate(project, event)
                source = lab._verified_record(project, event["key"], event["audio_sha256"])
                if (event.get("voice_id") != source.get("voice_id")
                        or event.get("voice_version") != source.get("version")
                        or event.get("role_id") != source.get("role_id")
                        or not event.get("created_at")):
                    raise versions.VoiceLibraryError("Rating identity, role, timestamp or version differs from heard sample")
                record = versions._publication_record(project_id, source)
                assets = versions._assets(record)
                root = safe_project_root(project_id)
                for relative, checksum in assets.values():
                    versions._read_verified(versions._safe_path(root, relative), checksum)
                traits, trait_evidence = _traits(record, candidate, event)
                rows.append({"event_id": event_id, "source_event": event,
                    "source_event_sha256": lab.digest(event), "rating": stars,
                    "role_id": event["role_id"], "character": next(role.name for role in project.roles if role.id == event["role_id"]),
                    "key": source["key"], "voice_id": source["voice_id"], "version": source["version"],
                    "library_voice_id": "vl_" + lab.digest(source["voice_id"])[:24],
                    "sample_audio_sha256": source["audio_sha256"],
                    "reference_audio_sha256": (source.get("identity") or {}).get("reference_audio_sha256"),
                    "prompt_sha256": (source.get("identity") or {}).get("prompt_sha256"),
                    "source_record_sha256": lab.digest(source), "publication_record_sha256": lab.digest(record),
                    "take_proof_sha256": (record.get("details") or {}).get("take_proof_sha256"),
                    "generation_language": record.get("generation_language"), "traits": traits,
                    "trait_evidence": trait_evidence})
            except (ValueError, OSError, KeyError, StopIteration) as exc:
                held.append({"event_id": event_id, "rating": stars, "reason": str(exc)[:500]})
        if lab.digest(project.model_dump(mode="json")) != before:
            raise versions.VoiceLibraryError("Migration planning changed source project")
    frozen = {"schema": SCHEMA, "project_id": project_id, "source_project_sha256": before, "rows": rows, "held": held, "skipped": skipped}
    return {**frozen, "plan_sha256": lab.digest(frozen), "dry_run": True, "eligible_events": len(rows), "source_project_unchanged": True}


def migrate(project_id: str, *, dry_run: bool = True, expected_plan_sha256: str | None = None) -> dict:
    if dry_run:
        return plan(project_id)
    with _LOCK:
        lease = try_acquire_project_mutation(project_id, kind="preferred_voice_migration")
        if lease is None:
            raise versions.VoiceLibraryError("Project is rendering or mutating; keeper migration must wait")
        try:
            frozen = plan(project_id)
            if not expected_plan_sha256 or expected_plan_sha256 != frozen["plan_sha256"]:
                raise versions.VoiceLibraryError("Keeper migration plan changed; review a fresh dry run")
            if not frozen["rows"]:
                return {"schema": SCHEMA, "project_id": project_id, "plan_sha256": frozen["plan_sha256"],
                        "status": "no_op", "completed": [], "held": frozen["held"], "skipped": frozen["skipped"], "tts_calls": 0}
            migration_id = "migration_" + frozen["plan_sha256"][:32]
            for _attempt in range(5):
                prior_path = library.voices_audio_dir() / "voice_lab/migrations" / migration_id / "receipt.json"
                if not prior_path.exists():
                    break
                prior = json.loads(prior_path.read_text("utf-8"))
                if prior.get("status") != "rolled_back":
                    break
                # Keep each recovery journal and backup immutable. Re-applying
                # the same reviewed plan gets a deterministic new attempt id.
                migration_id = "migration_" + lab.digest({"plan": frozen["plan_sha256"], "recovered_receipt": prior})[:32]
            else:
                raise versions.VoiceLibraryError("Migration recovery attempt budget exceeded")
            directory = library.voices_audio_dir() / "voice_lab/migrations" / migration_id
            directory.mkdir(parents=True, exist_ok=True)
            receipt_path = directory / "receipt.json"
            if receipt_path.exists():
                previous = json.loads(receipt_path.read_text("utf-8"))
                if previous.get("status") == "complete":
                    return {**previous, "already_applied": True}
                raise versions.VoiceLibraryError("A previous incomplete or rolled-back migration needs explicit recovery")
            with library.voices_lock():
                path = library._voices_file_path()
                original = path.read_bytes() if path.exists() else b"[]"
                (directory / "voices.before.json").write_bytes(original)
            (directory / "plan.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), "utf-8")
            result = {"schema": SCHEMA, "project_id": project_id, "migration_id": migration_id,
                      "plan_sha256": frozen["plan_sha256"], "status": "applying", "completed": [], "published_scopes": [],
                      "held": frozen["held"], "skipped": frozen["skipped"], "tts_calls": 0,
                      "library_before_sha256": _bytes_hash(original)}
            receipt_path.write_text(json.dumps(result, indent=2), "utf-8")
            try:
                for row in frozen["rows"]:
                    published = versions.publish_version(project_id, row["key"])
                    # The publication is already durable. Journal its ownership
                    # before rating, whose save/readback may fail independently.
                    result["published_scopes"].append({**{key: row[key] for key in
                        ("event_id", "rating", "key", "voice_id", "version", "library_voice_id", "sample_audio_sha256", "source_event_sha256")},
                        "published_owned": published["published"]})
                    receipt_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
                    preferred.record_rating(library_voice_id=published["library_voice_id"], version=published["version"],
                        rating=row["rating"], project_id=project_id, role_id=row["role_id"], character=row["character"],
                        key=row["key"], audio_sha256=row["sample_audio_sha256"], feedback=row["source_event"].get("raw_direction") or "",
                        action=row["source_event"].get("action") or "save", traits=row["traits"], evidence_id=row["event_id"],
                        source_event=row["source_event"], trait_evidence=row["trait_evidence"])
                    result["completed"].append({**{key: row[key] for key in ("event_id", "rating", "key", "voice_id", "version", "library_voice_id", "sample_audio_sha256", "source_event_sha256")},
                                               "published_owned": published["published"]})
                if lab.digest(load_project(project_id).model_dump(mode="json")) != frozen["source_project_sha256"]:
                    raise versions.VoiceLibraryError("Source project changed during migration; retain evidence and review")
                result.update(status="complete", source_project_unchanged=True, ratings_invented=False, favorites_promoted=False,
                              original_voices_preserved=True, already_applied=False)
            except Exception as exc:
                result.update(status="incomplete", reason=str(exc)[:500])
                raise
            finally:
                with library.voices_lock():
                    result["library_after_sha256"] = _bytes_hash(library._voices_file_path().read_bytes())
                    original_rows = {item["id"]: item for item in json.loads(original)}
                    current_rows = {item.id: item.model_dump(mode="json") for item in library.load_voices()}
                    scopes = sorted({(row["library_voice_id"], row["version"]) for row in result["published_scopes"]})
                    result["owned_versions"] = []
                    for voice_id, version in scopes:
                        previous = (original_rows.get(voice_id, {}).get("voice_lab_preferred") or {}).get(version)
                        actual = current_rows[voice_id]["voice_lab_preferred"].get(version)
                        ours = [row for row in result["published_scopes"] if row["library_voice_id"] == voice_id and row["version"] == version]
                        allowed = {(row.get("project_id"), row.get("event_id")) for row in (previous or {}).get("ratings") or []}
                        allowed.update((project_id, row["event_id"]) for row in ours)
                        ratings_safe = all((row.get("project_id"), row.get("event_id")) in allowed for row in (actual or {}).get("ratings") or [])
                        retired_safe = (actual or {}).get("retired") == (previous or {}).get("retired")
                        result["owned_versions"].append({"library_voice_id": voice_id, "version": version,
                            "preferred_after_sha256": lab.digest(actual),
                            "version_after_sha256": lab.digest(current_rows[voice_id]["voice_lab_versions"].get(version)),
                            "published_owned": any(row["published_owned"] for row in ours),
                            "rollback_safe": ratings_safe and retired_safe})
                    result["ownership_finalized"] = True
                receipt_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
            return result
        finally:
            release_project_mutation(lease)


def _bytes_hash(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def rollback(project_id: str, migration_id: str) -> dict:
    if not re.fullmatch(r"migration_[a-f0-9]{32}", migration_id):
        raise versions.VoiceLibraryError("Invalid migration identity")
    with _LOCK, library.voices_lock():
        directory = library.voices_audio_dir() / "voice_lab/migrations" / migration_id
        receipt = json.loads((directory / "receipt.json").read_text("utf-8"))
        if receipt.get("project_id") != project_id:
            raise versions.VoiceLibraryError("Migration belongs to another project")
        if receipt.get("status") == "rolled_back":
            return receipt
        if receipt.get("ownership_finalized") is not True:
            raise versions.VoiceLibraryError("Interrupted ownership journal needs recovery; rollback has not changed the library")
        backup = (directory / "voices.before.json").read_bytes()
        if _bytes_hash(backup) != receipt.get("library_before_sha256"):
            raise versions.VoiceLibraryError("Migration backup integrity failed; library was not changed")
        original_values = json.loads(backup)
        if not isinstance(original_values, list):
            raise versions.VoiceLibraryError("Migration backup is not a valid voice library")
        for value in original_values:
            library.Voice.model_validate(value)
        original = {row["id"]: row for row in original_values}
        voices = library.load_voices()
        current = {voice.id: voice for voice in voices}
        # Rollback is scoped to exact owned versions/preferences. Other voices,
        # versions and metadata survive concurrent and subsequent owner writes.
        for owned in receipt.get("owned_versions") or []:
            voice = current.get(owned["library_voice_id"])
            version = owned["version"]
            if (not owned.get("rollback_safe") or voice is None
                    or lab.digest(voice.voice_lab_preferred.get(version)) != owned["preferred_after_sha256"]
                    or lab.digest(voice.voice_lab_versions.get(version)) != owned["version_after_sha256"]):
                raise versions.VoiceLibraryError("An owned version changed or gained another rating; rollback would lose newer work")
        for owned in receipt.get("owned_versions") or []:
            voice = current[owned["library_voice_id"]]
            version = owned["version"]
            previous_voice = original.get(voice.id) or {}
            previous_preference = (previous_voice.get("voice_lab_preferred") or {}).get(version)
            if previous_preference is None:
                voice.voice_lab_preferred.pop(version, None)
            else:
                voice.voice_lab_preferred[version] = copy.deepcopy(previous_preference)
            if owned["published_owned"] and version not in (previous_voice.get("voice_lab_versions") or {}):
                voice.voice_lab_versions.pop(version, None)
        library.save_voices(voices)
        receipt.update(status="rolled_back", original_preferences_restored=True,
                       unrelated_library_changes_preserved=True, audio_assets_retained=True,
                       empty_voice_metadata_retained=True)
        (directory / "receipt.json").write_text(json.dumps(receipt, indent=2), "utf-8")
        return receipt
