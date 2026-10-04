"""Persistent cross-project preferred-voice memory over the existing voice library.

The canonical home is the global voice library already used by Voice Lab
(:mod:`dubbing_service.app.storage.voices`): each immutable version gained a
``voice_lab_preferred`` record holding the ratings, feedback and traits that
explain *why* the voice worked. Nothing here is a second preference store, and
nothing here assigns a voice: a prior voice is only ever *proposed* as another
candidate in the normal audition process.

Three separate concepts stay separate:

* a character identity is a role in one project;
* a voice identity is the immutable ``library_voice_id`` + ``version``;
* a casting assignment is a human decision recorded on the project role.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone

from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.storage import voices as library
from .voice_lab import digest

PREFERRED_SCHEMA = "preferred-voice-v1"

#: A rating at or above this value makes a voice eligible for reuse elsewhere.
REUSE_MIN_RATING = 4
#: A rating below the reuse threshold never promotes a voice into the pool.
PROMOTION_BLOCKED_BELOW = REUSE_MIN_RATING
#: Trait similarity a proposal needs before it is worth a human's attention.
MATCH_THRESHOLD = 45.0

#: Trait fit is scaled below 100 so the rating itself always contributes the
#: remaining headroom: a five-star prior voice must outrank a four-star one by a
#: visible margin even when their traits match equally well.
_TRAIT_SCALE = 92.0
_RATING_BONUS = 4.0

#: Dimension weights for fit scoring. Vocal identity dimensions dominate, because
#: they are what makes one character's approved voice suit another character.
_DIMENSION_WEIGHT = {
    "presentation": 22.0, "perceived_vocal_age": 18.0, "accent": 20.0,
    "pitch": 8.0, "weight": 7.0, "resonance": 7.0, "texture": 6.0,
    "cadence": 7.0, "tempo": 5.0, "articulation": 5.0, "warmth": 6.0,
    "intensity": 4.0, "confidence": 5.0, "formality": 5.0, "theatricality": 4.0,
    "emotional_range": 4.0, "breathiness": 3.0, "roughness": 3.0,
}
#: Values that mean different things when they differ, rather than merely
#: differing strings (a deliberate cadence against a brisk one is a real clash).
_OPPOSED = (("masculine", "feminine"), ("youthful", "older adult"), ("child", "older adult"),
            ("deliberate", "brisk"), ("warm", "cold"), ("restrained", "high energy"),
            ("formal", "conversational"), ("clear", "rough"), ("rounded", "nasal"))


class PreferredVoiceError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value, limit) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise PreferredVoiceError("Preference text must be a string")
    text = " ".join(value.split())
    if len(text) > limit:
        raise PreferredVoiceError("Preference text is too long")
    return text


def _signal_from_ratings(ratings: list[dict]) -> int | None:
    """The strongest reusable signal among the recorded ratings, if any.

    A later low rating does not erase an earlier approval, and a low rating on
    its own never promotes a voice: the history stays, the signal is the best
    4- or 5-star rating present.
    """
    signals = [row["rating"] for row in ratings
               if isinstance(row.get("rating"), int) and not isinstance(row.get("rating"), bool)
               and row["rating"] >= REUSE_MIN_RATING]
    return max(signals) if signals else None


def _record(library_id: str, version: str) -> tuple[dict, dict]:
    with library.voices_lock():
        voices = library.load_voices()
        voice = next((item for item in voices if item.id == library_id), None)
        if voice is None or version not in voice.voice_lab_versions:
            raise PreferredVoiceError("This library voice version is unavailable")
        return voice.model_dump(mode="json").get("voice_lab_preferred", {}).get(version, {}), \
            voice.voice_lab_versions[version]["record"]


def normalized_language(value) -> str | None:
    value = str(value or "").strip().casefold()
    aliases = {"english": "en", "french": "fr", "german": "de", "spanish": "es",
               "chinese": "zh", "japanese": "ja", "korean": "ko", "italian": "it",
               "portuguese": "pt", "russian": "ru"}
    return None if value in {"", "auto", "unknown", "unspecified"} else aliases.get(value, value)


def generation_language(record: dict) -> str | None:
    # This owner-authored field records the actual audition role's synthesis
    # language. Do not infer it from performer nationality or the source book.
    return normalized_language(record.get("generation_language"))


def traits_from_record(record: dict) -> dict:
    """The vocal traits a stored voice already documents about itself.

    Read from the immutable voice identity's design instruction inputs, so the
    matching vocabulary is the same one the profile and the anchor ranker use.
    """
    traits: dict = {}
    if generation_language(record):
        traits["language"] = generation_language(record)
    identity = record.get("identity") or {}
    if identity.get("design_instruction"):
        traits["design_instruction"] = identity["design_instruction"]
    inspiration = identity.get("inspiration") or record.get("inspiration") or {}
    if inspiration.get("performer"):
        traits["performer"] = inspiration["performer"]
    if inspiration.get("performance"):
        traits["performance"] = inspiration["performance"]
    return traits


def record_rating(*, library_voice_id: str, version: str, rating: int, project_id: str,
                  role_id: str, character: str | None = None, key: str | None = None,
                  audio_sha256: str | None = None, feedback: str = "", action: str = "save",
                  traits: dict | None = None, evidence_id: str | None = None,
                  source_event: dict | None = None, trait_evidence: list | None = None) -> dict:
    """Attach one human rating to the immutable voice version it was given to.

    Called for every rating, including a 1-3 star rating: the history is the
    record of what happened, while ``promoted`` states whether the voice may be
    offered to future projects.
    """
    if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
        raise PreferredVoiceError("A preference rating must be an integer from 1 to 5")
    if not isinstance(library_voice_id, str) or not library_voice_id or len(library_voice_id) > 128:
        raise PreferredVoiceError("A stable library voice id is required")
    if not isinstance(version, str) or not version or len(version) > 128:
        raise PreferredVoiceError("A stable voice version is required")
    entry = {"rating": rating, "project_id": _safe_text(project_id, 100),
             "role_id": _safe_text(role_id, 100), "character": _safe_text(character or "", 200),
             "key": _safe_text(key or "", 200), "audio_sha256": _safe_text(audio_sha256 or "", 64),
             "feedback": _safe_text(feedback, 4000), "action": _safe_text(action, 20),
             "event_id": _safe_text(evidence_id or "", 200), "recorded_at": _now()}
    source_record = None
    if source_event is not None:
        # A historical audition can have a different intent key while retaining
        # the exact same version/audio. Verify its original owner evidence before
        # entering voices_lock; never reverse the project/library lock order.
        from dubbing_service.app.storage.projects import load_project, project_meta_lock
        from . import voice_lab as lab, voice_lab_library as shelf
        try:
            with project_meta_lock(project_id):
                original = load_project(project_id)
                if (original.voice_lab.casting.get("events") or {}).get(evidence_id) != source_event:
                    raise ValueError("Original rating event is not stored by its project owner")
                source_record = lab._verified_record(original, key, audio_sha256)
            source_record = shelf._publication_record(project_id, source_record)
            for relative, checksum in shelf._assets(source_record).values():
                shelf._read_verified(shelf._safe_path(shelf.safe_project_root(project_id), relative), checksum)
        except (ValueError, OSError, KeyError) as exc:
            raise PreferredVoiceError("Original human rating event differs from the exact heard voice") from exc
    with library.voices_lock():
        voices = library.load_voices()
        voice = next((item for item in voices if item.id == library_voice_id), None)
        if voice is None or version not in voice.voice_lab_versions:
            raise PreferredVoiceError("This library voice version is unavailable")
        record = voice.voice_lab_versions[version]["record"]
        if source_event is not None:
            if (source_event.get("origin") != "human" or not source_event.get("actor")
                    or source_event.get("event_id") != evidence_id or type(source_event.get("stars")) is not int
                    or source_event["stars"] != rating or source_event.get("key") != key
                    or source_event.get("audio_sha256") != audio_sha256 or source_event.get("role_id") != role_id
                    or audio_sha256 != record.get("audio_sha256")
                    or source_record.get("voice_id") != record.get("voice_id")
                    or source_record.get("version") != version
                    or shelf._profile_fingerprint(source_record) != shelf._profile_fingerprint(record)
                    or source_event.get("voice_id") != record.get("voice_id")
                    or source_event.get("voice_version") != version or not source_event.get("created_at")):
                raise PreferredVoiceError("Original human rating event differs from the exact heard voice")
            entry.update(source_event=copy.deepcopy(source_event), source_event_sha256=digest(source_event),
                         actor=source_event["actor"], origin=source_event["origin"], original_recorded_at=source_event["created_at"], source_event_project_id=project_id)
            if key != record.get("key"):
                binding = {"schema": "verified-historical-heard-key-v1", "project_id": project_id, "key": key,
                    "published_key": record["key"], "voice_id": record["voice_id"], "version": version,
                    "audio_sha256": audio_sha256, "source_event_sha256": digest(source_event),
                    "source_record_sha256": digest(source_record),
                    "profile_sha256": shelf._profile_fingerprint(record),
                    "audition_intent_hash": source_record.get("audition_intent_hash"),
                    "take_proof_sha256": (source_record.get("details") or {}).get("take_proof_sha256")}
                entry["heard_key_binding"] = {**binding, "binding_sha256": digest(binding)}

        current = copy.deepcopy(voice.voice_lab_preferred.get(version) or {})
        ratings = list(current.get("ratings") or [])
        if evidence_id and any(row.get("event_id") == entry["event_id"] and row.get("project_id") == entry["project_id"]
                               for row in ratings):
            return current
        ratings.append(entry)
        characters = list(current.get("characters") or [])
        identity_row = {"project_id": entry["project_id"], "role_id": entry["role_id"],
                        "character": entry["character"]}
        if identity_row not in characters:
            characters.append(identity_row)
        projects = list(dict.fromkeys([*(current.get("projects") or []), entry["project_id"]]))
        # The heard record names the reference at the top level; the identity form
        # stores the same performance under `character`. Accept both so a stored
        # proposal never carries a half-empty inspiration.
        top = record.get("inspiration") or {}
        nested = (record.get("identity") or {}).get("inspiration") or {}
        inspiration = {"performer": top.get("performer") or nested.get("performer"),
                       "performance": top.get("performance") or nested.get("character")}
        signal = _signal_from_ratings(ratings)
        voice.voice_lab_preferred[version] = {
            "schema": PREFERRED_SCHEMA, "version": version, "library_voice_id": library_voice_id,
            "name": record.get("name"), "signal": signal,
            "promoted": bool(signal and not current.get("retired")),
            "highest_rating": max((row["rating"] for row in ratings), default=None),
            "ratings": ratings, "characters": characters, "projects": projects,
            "traits": {**(current.get("traits") or {}), **(traits or traits_from_record(record))},
            "trait_evidence": copy.deepcopy(trait_evidence if trait_evidence is not None else current.get("trait_evidence") or []),
            "inspiration": {"performer": inspiration.get("performer"),
                            "performance": inspiration.get("performance")},
            "engine": {"backend": record.get("backend"), "version": record.get("version")},
            "generation": {"key": record.get("key"), "voice_id": record.get("voice_id"),
                           "identity_id": (record.get("identity") or {}).get("id"),
                           "design_instruction": (record.get("identity") or {}).get("design_instruction")},
            "retired": current.get("retired"),
            "created_at": current.get("created_at") or _now(), "updated_at": _now(),
        }
        voice.updated_at = datetime.now(timezone.utc)
        library.save_voices(voices)
    return library_preferred(library_voice_id, version)


def library_preferred(library_voice_id: str, version: str) -> dict:
    record, _ = _record(library_voice_id, version)
    return record or {}


def retire(*, library_voice_id: str, version: str, reason: str) -> dict:
    """Mark a version unavailable for future proposals without erasing history."""
    with library.voices_lock():
        voices = library.load_voices()
        voice = next((item for item in voices if item.id == library_voice_id), None)
        if voice is None or version not in voice.voice_lab_preferred:
            raise PreferredVoiceError("This preferred voice is unavailable")
        entry = voice.voice_lab_preferred[version]
        entry["retired"] = {"at": _now(), "reason": _safe_text(reason, 400)}
        entry["promoted"] = False
        entry["updated_at"] = _now()
        voice.updated_at = datetime.now(timezone.utc)
        library.save_voices(voices)
    return library_preferred(library_voice_id, version)


def preferred_entries(*, include_retired: bool = False) -> list[dict]:
    """Every promoted preferred voice, strongest signal first."""
    rows = []
    with library.voices_lock():
        voices = library.load_voices()
        for voice in voices:
            for version, entry in (voice.voice_lab_preferred or {}).items():
                if entry.get("schema") != PREFERRED_SCHEMA or not entry.get("ratings"):
                    continue
                if not entry.get("promoted") and not include_retired:
                    continue
                rows.append({"library_voice_id": voice.id, "version": version, **entry})
    rows.sort(key=lambda row: (-(row.get("signal") or 0), -len(row.get("ratings") or []),
                               row["library_voice_id"], row["version"]))
    return rows


def _score(traits: dict, entry: dict) -> tuple[float, float, list[str]]:
    """Trait fit plus the rating signal, and the reasons a human can read.

    A single opposed pair is disqualifying, because two voices that differ in
    vocal weight or perceived age are not interchangeable however many other
    labels they share.
    """
    wanted = {key: str(value).casefold() for key, value in (traits or {}).items()
              if value and key in _DIMENSION_WEIGHT}
    if not wanted:
        return 0.0, 0.0, []
    wanted_display = {key: str(value) for key, value in (traits or {}).items()
                      if value and key in _DIMENSION_WEIGHT}
    stored = {key: str(value).casefold() for key, value in (entry.get("traits") or {}).items()}
    # Known identity constraints are compatibility gates, not soft discounts.
    for key in ("presentation", "perceived_vocal_age", "accent", "language"):
        desired = str((traits or {}).get(key) or "").strip().casefold()
        actual = stored.get(key, "").strip()
        if key == "language":
            desired, actual = normalized_language(desired), normalized_language(actual)
            if desired and not actual:
                return 0.0, 0.0, []
        if desired and actual and desired != actual:
            return 0.0, 0.0, []
    for left, right in _OPPOSED:
        for key, value in wanted.items():
            if value == left and stored.get(key) == right:
                return 0.0, 0.0, []
            if value == right and stored.get(key) == left:
                return 0.0, 0.0, []
    available = sum(_DIMENSION_WEIGHT[key] for key in wanted)
    earned = 0.0
    reasons = []
    for key, value in wanted.items():
        if stored.get(key) == value:
            earned += _DIMENSION_WEIGHT[key]
            reasons.append(f"{key.replace('_', ' ')}: {wanted_display[key]}")
    inspiration = entry.get("inspiration") or {}
    if inspiration.get("performer"):
        reasons.append(f"previously inspired by {inspiration['performer']}"
                       + (f", {inspiration['performance']}" if inspiration.get("performance") else ""))
    trait_fit = 100.0 * earned / available
    # Requirement: a 5-star prior voice carries a stronger reuse signal than a
    # 4-star one, with every other term held equal.
    signal = entry.get("signal") or 0
    percent = min(100.0, round(_TRAIT_SCALE * earned / available + (signal - REUSE_MIN_RATING) * _RATING_BONUS, 1))
    return percent, round(trait_fit, 1), reasons



def _rating_key_bound(row: dict, sample: dict) -> bool:
    event = row.get("source_event")
    if event is not None:
        if (not isinstance(event, dict) or type(event.get("stars")) is not int
                or row.get("rating") != event["stars"]
                or row.get("event_id") != event.get("event_id")
                or row.get("role_id") != event.get("role_id")
                or row.get("key") != event.get("key")
                or row.get("audio_sha256") != event.get("audio_sha256")
                or event.get("voice_id") != sample.get("voice_id")
                or event.get("voice_version") != sample.get("version")
                or row.get("origin") != event.get("origin") or event.get("origin") != "human"
                or row.get("actor") != event.get("actor") or not event.get("actor")
                or row.get("original_recorded_at") != event.get("created_at") or not event.get("created_at")
                or row.get("source_event_sha256") != digest(event)
                or row.get("source_event_project_id") != row.get("project_id")):
            return False
    if row.get("key") == sample.get("key"):
        return True
    from .voice_lab_library import _profile_fingerprint, _valid_version_key
    binding = row.get("heard_key_binding") or {}
    payload = {key: value for key, value in binding.items() if key != "binding_sha256"}
    event = row.get("source_event") or {}
    alias = {**sample, "key": row.get("key"), "audition_intent_hash": binding.get("audition_intent_hash")}
    return (binding.get("schema") == "verified-historical-heard-key-v1"
        and binding.get("binding_sha256") == digest(payload)
        and binding.get("project_id") == row.get("project_id")
        and binding.get("published_key") == sample.get("key")
        and binding.get("key") == row.get("key") == event.get("key")
        and binding.get("voice_id") == sample.get("voice_id") == event.get("voice_id")
        and binding.get("version") == sample.get("version") == event.get("voice_version")
        and binding.get("audio_sha256") == sample.get("audio_sha256") == event.get("audio_sha256")
        and binding.get("source_event_sha256") == digest(event)
        and binding.get("profile_sha256") == _profile_fingerprint(sample)
        and bool(binding.get("take_proof_sha256")) and _valid_version_key(alias))


def match_preferred(traits: dict, *, limit: int = 3, threshold: float = MATCH_THRESHOLD,
                    exclude_library_voice_ids: tuple = ()) -> list[dict]:
    """Propose prior approved voices that genuinely fit a new character.

    Returns evidence, never an assignment: each row states its prior rating, its
    fit percentage, the traits that matched and where the voice was used before.
    """
    limit = max(1, int(limit))
    excluded = {value for value in exclude_library_voice_ids if value}
    proposals = []
    for entry in preferred_entries():
        if entry["library_voice_id"] in excluded:
            continue
        try:
            _, sample = _record(entry["library_voice_id"], entry["version"])
        except (PreferredVoiceError, ValueError, OSError):
            continue
        promoting = [row for row in entry.get("ratings") or []
                     if type(row.get("rating")) is int and 4 <= row["rating"] <= 5
                     and row.get("audio_sha256") == sample.get("audio_sha256")
                     and _rating_key_bound(row, sample)
                     and row.get("project_id") and row.get("role_id") and row.get("event_id")]
        if not promoting:
            continue
        event = max(promoting, key=lambda row: (row["rating"], row.get("recorded_at", "")))
        bound_entry = {**entry, "signal": event["rating"], "highest_rating": event["rating"],
                       "traits": {**(entry.get("traits") or {}), "language": generation_language(sample)}}
        percent, trait_fit, reasons = _score(traits, bound_entry)
        if percent < threshold:
            continue
        characters = [row for row in entry.get("characters") or [] if row.get("character")]
        proposals.append({
            "schema": PREFERRED_SCHEMA, "prior_match": True,
            "library_voice_id": entry["library_voice_id"], "library_version": entry["version"],
            "name": entry.get("name"), "signal": event["rating"],
            "highest_rating": event["rating"], "match_percent": percent,
            "trait_match_percent": trait_fit, "match_reasons": reasons, "matched_traits": [row.split(": ", 1)[1] for row in reasons
                                                          if ": " in row and "previously inspired" not in row],
            "traits": bound_entry["traits"], "inspiration": entry.get("inspiration") or {},
            "previously_used": characters[:5], "projects": list(entry.get("projects") or [])[:5],
            "ratings": list(entry.get("ratings") or [])[-5:],
            "rated_roles": [{"project_id": row.get("project_id"), "role_id": row.get("role_id")}
                            for row in entry.get("ratings") or []],
            "promoting_rating": copy.deepcopy(event),
            "sample_audio_sha256": sample["audio_sha256"],
            "trait_evidence": copy.deepcopy(entry.get("trait_evidence") or []),
            "actual_source_performer": (((sample.get("identity") or {}).get("provenance") or {})
                                        .get("source", {}).get("performer")
                                        if ((sample.get("identity") or {}).get("provenance") or {})
                                        .get("origin") == "eligible_reference_clone" else None),
            "assignment": None,
        })
    proposals.sort(key=lambda row: (-row["match_percent"], -row["signal"], row["library_voice_id"]))
    return proposals[:limit]


def prepare_prior_candidate(project_id: str, library_voice_id: str, version: str) -> dict:
    """Make one proposed prior voice auditable inside ``project_id``.

    Never call this while holding the project meta lock: the import acquires the
    same non-reentrant lock plus the existing project mutation lease. The caller
    records the returned key as the candidate's ``reuse_key``, so the heard voice
    is the *exact* approved voice rather than a new voice described like it.
    """
    from .voice_lab_library import VoiceLibraryError, import_version
    try:
        result = import_version(project_id, library_voice_id, version)
    except (VoiceLibraryError, ValueError, OSError) as exc:
        return {"state": "blocked", "reason": str(exc)[:400]}
    return {"state": "ready", "key": result.get("key"), "imported": bool(result.get("imported")),
            "library_voice_id": library_voice_id, "library_version": version}


def evidence(row: dict) -> dict:
    """The review-surface evidence for a proposed prior voice, and nothing more."""
    characters = ", ".join(f"{item.get('character') or item.get('role_id')}" for item in row.get("previously_used") or [])
    return {"schema": PREFERRED_SCHEMA, "prior_match": True,
            "library_voice_id": row.get("library_voice_id"), "library_version": row.get("library_version"),
            "prior_rating": row.get("signal"), "highest_rating": row.get("highest_rating"),
            "match_percent": row.get("match_percent"), "trait_match_percent": row.get("trait_match_percent"),
            "match_reasons": list(row.get("match_reasons") or []),
            "previously_used": characters, "projects": list(row.get("projects") or []),
            "inspiration": row.get("inspiration") or {},
            "promoting_rating": copy.deepcopy(row.get("promoting_rating")),
            "sample_audio_sha256": row.get("sample_audio_sha256"),
            "sample_origin": "historical_approved_sample",
            "rationale": (f"Previously approved voice · prior rating {row.get('signal')} stars · "
                          f"{row.get('match_percent')}% match")}


def stable_voice_id(library_voice_id: str, version: str) -> str:
    """The character- and project-independent identity of one reusable voice."""
    return digest({"library_voice_id": library_voice_id, "version": version})


def identity_snapshot(record: dict) -> dict:
    """The stored identity's own evidence, used to explain a proposal later."""
    raw = record.get("identity")
    identity = CharacterVoiceIdentity.model_validate(raw) if raw else None
    return {"identity_id": identity.id if identity else None,
            "mechanism": identity.mechanism if identity else None,
            "design_instruction": identity.design_instruction if identity else None,
            "inspiration": (identity.inspiration.model_dump(mode="json")
                            if identity and identity.inspiration else None)}


__all__ = [
    "MATCH_THRESHOLD", "PREFERRED_SCHEMA", "PROMOTION_BLOCKED_BELOW", "REUSE_MIN_RATING",
    "PreferredVoiceError", "evidence", "identity_snapshot", "library_preferred",
    "match_preferred", "preferred_entries", "prepare_prior_candidate", "record_rating",
    "retire", "stable_voice_id", "traits_from_record",
]
