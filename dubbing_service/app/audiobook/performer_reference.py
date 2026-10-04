"""Reference evidence and acoustic conditioning for one performer anchor.

The name of a performer is not a condition. This module turns one anchor into the
numeric acoustic target that generation is actually steered by, and records where
every number came from.

Two characterization states are possible, and they are never conflated:

* **measured** — permitted audio for the reference was ingested and analysed with
  :mod:`service.voice_acoustics`. The measured values carry the analysis method.
* **inferred** — no permitted audio could be ingested (no rights evidence, no
  local copy, or the source is not machine-readable). The acoustic target is then
  a documented prior derived from the catalogue's own descriptors and published
  speaking-voice ranges. It is still numeric, so generation is still steered by
  acoustics rather than by a name — but it is labelled as an estimate.

In both states the anchor keeps its performer and specific performance, and the
concept's acoustic bias is applied on top so candidates for one character differ
audibly rather than only nominally.
"""
from __future__ import annotations

from typing import Mapping

SCHEMA = "performer-reference-v1"

#: Prior acoustic estimates for descriptor families. These are documented
#: speaking-voice ranges and casting conventions, not measurements of any person.
_PITCH_MEDIAN_HZ = {"masculine": 118.0, "feminine": 205.0, "androgynous": 165.0}
_PITCH_RANGE_SEMITONES = {"masculine": 6.0, "feminine": 8.0, "androgynous": 7.0}
_AGE_PITCH_FACTOR = {"young": 1.12, "adult": 1.0, "older": 0.91}
_WORDS_PER_MINUTE = (("deliberate", 135.0), ("measured", 135.0), ("unhurried", 138.0),
                     ("brisk", 175.0), ("energetic", 172.0), ("lively", 168.0), ("quick", 170.0))
_DEFAULT_WORDS_PER_MINUTE = 150.0
_PAUSE_MS = (("deliberate", 380.0), ("measured", 360.0), ("brisk", 220.0), ("energetic", 230.0))
_DEFAULT_PAUSE_MS = 300.0
_BRIGHTNESS_FACTOR = (("bright", 1.12), ("clear", 1.06), ("resonant", 0.90),
                      ("rounded", 0.92), ("warm", 0.96), ("dark", 0.86))
_BREATHINESS = (("airy", 0.22), ("breathy", 0.24), ("soft", 0.20), ("warm", 0.14))
_DEFAULT_BREATHINESS = 0.10
_RASP = (("gravel", 0.34), ("rough", 0.32), ("rasp", 0.30), ("mature", 0.18), ("dry", 0.12))
_DEFAULT_RASP = 0.08
_ARTICULATION = (("precise", 0.72), ("clear", 0.70), ("clipped", 0.74), ("crisp", 0.72))
_DEFAULT_ARTICULATION = 0.62
_DYNAMIC_RANGE_DB = (("theatrical", 14.0), ("expressive", 12.5), ("lively", 12.0),
                     ("measured", 8.5), ("restrained", 8.0), ("dry", 9.5))
_DEFAULT_DYNAMIC_RANGE_DB = 11.0

#: Bounds the worker may shape within. Beyond these the artefacts cost more than
#: the acoustic correction is worth, and the remainder is left to the instruction.
SHAPING_BOUNDS = {"max_semitones": 3.0, "rate_range": [0.85, 1.18]}

#: Fields a profile derived from descriptors can carry; everything else stays
#: absent rather than invented.
_PRIOR_FIELDS = ("f0_median_hz", "f0_mean_hz", "f0_p10_hz", "f0_p90_hz", "f0_range_semitones",
                 "syllables_per_minute", "words_per_minute", "pause_count_per_minute",
                 "mean_pause_ms", "pause_fraction", "rms_dbfs", "peak_dbfs", "dynamic_range_db",
                 "brightness_hz", "breathiness", "rasp", "articulation_index",
                 "sentence_end_f0_slope", "sentence_end_energy_drop_db", "voiced_fraction")


def _pick(tags, table, default):
    for key, value in table:
        if any(key in tag for tag in tags):
            return value
    return default


def anchor_tags(anchor: Mapping) -> list[str]:
    return [str(tag).casefold() for tag in (anchor.get("tags") or [])]


def _presentation(traits: Mapping) -> str:
    """Which published voice range this character's own evidence falls into."""
    for name in ("presentation", "gender", "pronouns", "sex"):
        value = str((traits.get(name) or {}).get("value") or "").casefold()
        if not value:
            continue
        if any(word in value for word in ("male", "masculine", "he/him", "man")):
            if "female" in value or "woman" in value:
                return "feminine"
            return "masculine"
        if any(word in value for word in ("female", "feminine", "she/her", "woman")):
            return "feminine"
    return "androgynous"


def acoustic_prior(anchor: Mapping, concept: Mapping) -> dict:
    """A documented numeric prior for one anchor, with every field labelled inferred.

    Nothing here is a measurement of a person. Values come from the catalogue's
    own descriptors and published speaking-voice ranges, and the returned profile
    lists them under ``inferred`` so no consumer can present them as measured.
    """
    tags = anchor_tags(anchor)
    traits = dict(concept.get("supported") or {})
    traits.update({name: row for name, row in (concept.get("inferred") or {}).items()})
    presentation = _presentation(traits)
    age_value = " ".join(str((traits.get(name) or {}).get("value") or "").casefold()
                         for name in ("perceived_vocal_age", "age", "age_range"))
    age = ("young" if any(word in age_value for word in ("young", "youth", "teen", "adolesc", "child"))
           else "older" if any(word in age_value for word in ("older", "elder", "senior", "mature", "aged"))
           else "adult")
    pitch = _PITCH_MEDIAN_HZ[presentation] * _AGE_PITCH_FACTOR[age]
    spread = _PITCH_RANGE_SEMITONES[presentation] * (1.25 if "theatrical" in tags else 1.0)
    words = _pick(tags, _WORDS_PER_MINUTE, _DEFAULT_WORDS_PER_MINUTE)
    pause = _pick(tags, _PAUSE_MS, _DEFAULT_PAUSE_MS)
    brightness = pitch * 13.0 * _pick(tags, _BRIGHTNESS_FACTOR, 1.0)
    profile = {
        "schema": "voice-acoustics-v1",
        "method": "descriptor-prior-v1",
        "f0_median_hz": round(pitch, 1),
        "f0_mean_hz": round(pitch * 1.02, 1),
        "f0_p10_hz": round(pitch / (2 ** (spread / 24)), 1),
        "f0_p90_hz": round(pitch * (2 ** (spread / 24)), 1),
        "f0_range_semitones": round(spread, 2),
        "syllables_per_minute": round(words * 1.5, 1),
        "words_per_minute": words,
        "pause_count_per_minute": round(60000.0 / (words / 1.5 * 1.35 + pause), 1),
        "mean_pause_ms": pause,
        "rms_dbfs": -21.0,
        "peak_dbfs": -3.5,
        "dynamic_range_db": _pick(tags, _DYNAMIC_RANGE_DB, _DEFAULT_DYNAMIC_RANGE_DB),
        "brightness_hz": round(brightness, 1),
        "breathiness": _pick(tags, _BREATHINESS, _DEFAULT_BREATHINESS),
        "rasp": _pick(tags, _RASP, _DEFAULT_RASP),
        "articulation_index": _pick(tags, _ARTICULATION, _DEFAULT_ARTICULATION),
        "sentence_end_f0_slope": -1.4 if "narrator" in tags else -1.0,
        "sentence_end_energy_drop_db": -6.0,
        "voiced_fraction": 0.62,
        "measured": [],
        "inferred": [],
        "notes": [],
    }
    profile["pause_fraction"] = round(min(0.6, pause / 1000.0 / 4.2), 3)
    profile["inferred"] = sorted(name for name in _PRIOR_FIELDS if profile.get(name) is not None)
    profile["notes"].append(
        "No permitted reference audio was ingested for this anchor; every value is a documented "
        "prior from the catalogue's own descriptors and published speaking-voice ranges, not a "
        "measurement of any person.")
    return profile


def measured_profile(path, *, at: str | None = None) -> dict:
    """Measure permitted reference audio, falling back to nothing if unreadable."""
    from service import voice_acoustics
    profile = voice_acoustics.measure(path)
    profile["notes"] = list(profile.get("notes") or [])
    profile["measured_at"] = at
    profile["source_path_name"] = str(getattr(path, "name", path))
    return profile


def _reference_holds(anchor: Mapping, concept: Mapping) -> list[str]:
    holds = []
    if not anchor.get("performer") or not anchor.get("performance"):
        holds.append("The anchor has no specific performer and performance")
    if not (anchor.get("source_urls") or anchor.get("verified_facts")):
        holds.append("The anchor carries no source evidence for its performance claim")
    if not (concept.get("supported") or concept.get("inferred")):
        holds.append("The concept establishes no casting dimensions")
    return holds


def resolve(anchor: Mapping, concept: Mapping, *, audio_path=None, at: str | None = None,
            retrieval_state: Mapping | None = None) -> dict:
    """One anchor's complete reference evidence and acoustic conditioning target.

    ``audio_path`` is permitted audio for this reference when one exists; the
    characterization is then measured rather than estimated. ``retrieval_state``
    records why acquisition did or did not happen, so a later reader can tell an
    un-ingestible source from an unattempted one.
    """
    from service import voice_acoustics
    holds = _reference_holds(anchor, concept)
    state = dict(retrieval_state or {})
    notes = []
    profile = None
    characterization = "inferred"
    if audio_path is not None:
        try:
            profile = measured_profile(audio_path, at=at)
            if profile.get("measured"):
                characterization = "measured"
                state.setdefault("state", "measured")
            else:
                notes.append("Permitted audio was analysed but no acoustic property could be measured; "
                             "the descriptor prior is used and labelled as inferred.")
                profile = None
        except Exception as exc:  # noqa: BLE001 - an unusable file must not fail casting
            notes.append(f"Permitted audio could not be analysed ({type(exc).__name__}); "
                         "the descriptor prior is used and labelled as inferred.")
            state.setdefault("state", "held")
            state.setdefault("reason", "Reference audio was not machine-readable")
            profile = None
    if profile is None:
        profile = acoustic_prior(anchor, concept)
        state.setdefault("state", "not_ingested" if audio_path is None else "held")
        if audio_path is None:
            state.setdefault("reason", (state.get("reason")
                                        or "No permitted reference audio is available for this anchor"))
    if holds:
        notes.extend(holds)
    bias = dict(concept.get("bias") or {})
    target = voice_acoustics.merge(profile, bias) if bias else dict(profile)
    parameters = voice_acoustics.parameters(target)
    instruction = voice_acoustics.instruction_fragment(target)
    return {
        "schema": SCHEMA,
        "performer": anchor.get("performer"),
        "performance": anchor.get("performance"),
        "source_urls": list(anchor.get("source_urls") or []),
        "source_type": state.get("source_type") or (
            "catalogue performance metadata + measured reference audio" if characterization == "measured"
            else "catalogue performance metadata + documented acoustic prior"),
        "verified_facts": list(anchor.get("verified_facts") or []),
        "catalogue_version": anchor.get("catalog_version"),
        "retrieval": {"attempted": audio_path is not None, "at": at,
                      "state": state.get("state"), "reason": state.get("reason"),
                      "source_urls": list(anchor.get("source_urls") or [])},
        "characterization": {
            "state": characterization,
            "method": profile.get("method"),
            "measured": list(profile.get("measured") or []),
            "inferred": list(profile.get("inferred") or []),
            "profile": profile,
            "notes": list(dict.fromkeys(notes + list(profile.get("notes") or []))),
        },
        "conditioning": {
            "schema": "voice-acoustics-v1",
            "target_profile": target,
            "parameters": parameters,
            "instruction": instruction,
            "bias": bias,
            "shaping_bounds": dict(SHAPING_BOUNDS),
            "characterization": characterization,
            "reference": {"performer": anchor.get("performer"), "performance": anchor.get("performance"),
                          "source_type": state.get("source_type"),
                          "source_urls": list(anchor.get("source_urls") or [])},
        },
        "clamped": abs(float(bias.get("pitch_semitones", 0.0))) > SHAPING_BOUNDS["max_semitones"],
        "rationale": _rationale(anchor, concept, characterization, parameters),
    }


def _rationale(anchor, concept, characterization, parameters) -> str:
    parts = [f"{anchor.get('performer')}, {anchor.get('performance')}",
             f"cast for {concept.get('label')} ({concept.get('emphasis')})"]
    if parameters.get("pitch_target_hz"):
        parts.append(f"target median pitch {parameters['pitch_target_hz']:.0f} Hz")
    if parameters.get("rate_words_per_minute"):
        parts.append(f"{parameters['rate_words_per_minute']:.0f} wpm")
    parts.append(f"acoustics {characterization}")
    return "; ".join(parts) + "."


def conditioning_payload(evidence: Mapping) -> dict:
    """Exactly what the generation request carries for one candidate.

    A numeric conditioning block, never a performer name on its own: the worker
    applies the numbers to the original reference it generates.
    """
    conditioning = dict(evidence.get("conditioning") or {})
    if not conditioning.get("parameters") or not conditioning.get("target_profile"):
        raise ValueError("Reference evidence has no acoustic conditioning")
    conditioning["reference"] = {"performer": evidence.get("performer"),
                                  "performance": evidence.get("performance"),
                                  "source_type": evidence.get("source_type"),
                                  "source_urls": list(evidence.get("source_urls") or [])}
    return conditioning


def describe(evidence: Mapping) -> dict:
    """The review-surface summary: where the numbers came from, plainly."""
    characterization = evidence.get("characterization") or {}
    parameters = (evidence.get("conditioning") or {}).get("parameters") or {}
    retrieval = evidence.get("retrieval") or {}
    return {
        "schema": SCHEMA,
        "performer": evidence.get("performer"),
        "performance": evidence.get("performance"),
        "source_urls": list(evidence.get("source_urls") or []),
        "source_type": evidence.get("source_type"),
        "retrieved_at": retrieval.get("at"),
        "retrieval_state": retrieval.get("state"),
        "retrieval_reason": retrieval.get("reason"),
        "characterization": characterization.get("state"),
        "characterization_method": characterization.get("method"),
        "measured_properties": list(characterization.get("measured") or []),
        "inferred_properties": list(characterization.get("inferred") or []),
        "notes": list(characterization.get("notes") or []),
        "parameters": parameters,
        "rationale": evidence.get("rationale"),
    }
