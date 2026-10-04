"""Bounded, local-first performance matching and official recording metadata.

No project/cast writes, synthesis, credentials, arbitrary URL fetches or crawler.
Only catalog performer names leave the machine. See docs/voice_lab_references.md.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dubbing_service.app.storage.atomic import replace_metadata_file

from . import official_recording_sources, performer_catalog

CATALOG_VERSION = "2026-09-29.2"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/audio/"
TERMS_URL = "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"
USER_AGENT = "AI-Audio-Studio-VoiceLab/1.0 (private bounded metadata research; https://github.com/tadashijlu/ai-audio-studio)"


def _seed(key, performer, performance, tags, direction, source_url, fact):
    return {
        "candidate_id": key, "performer": performer, "performance": performance,
        "tags": tags.split(","), "creative_direction": direction,
        "source_urls": [source_url], "verified_facts": [fact],
        "facts_verified_at": "2026-09-26", "language": "en",
        "evidence_level": "metadata_only_exploratory", "audio_observations": [],
        "creative_interpretation": "Delivery, vocal age and accent tags are creative casting directions, not acoustic measurements or biographical facts.",
    }


# Credit facts are from the official pages cited. No source audio was measured.
_SEEDS = (
    _seed("attenborough-planet-earth", "David Attenborough", "Planet Earth (2006), UK narration",
          "male,mature,narrator,warm,curious,measured,british,documentary",
          "Mature documentary warmth, unhurried curiosity and measured phrasing.",
          "https://downloads.bbc.co.uk/mediacentre/the-hunt-media-pack.pdf",
          "BBC's The Hunt media pack credits Attenborough with narrating Planet Earth in 2006."),
    _seed("watson-meg-march", "Emma Watson", "Meg March, Little Women (2019)",
          "female,young,adult,warm,clear,thoughtful,american,conversational",
          "Youthful adult clarity, thoughtful directness and conversational warmth.",
          "https://www.sonypictures.com/movies/littlewomen",
          "Sony lists Emma Watson as Meg March in its 2019 Little Women cast."),
    _seed("fry-sherlock-holmes", "Stephen Fry", "Sherlock Holmes: The Definitive Collection (2017), narration",
          "male,mature,narrator,resonant,confident,dry,expressive,british",
          "Resonant storytelling, relaxed confidence, dry humor and expressive phrasing.",
          "https://www.audible.com/pd/Sherlock-Holmes-Audiobook/B06WLMWF2S",
          "Audible credits Stephen Fry as narrator of its 2017 Sherlock Holmes collection."),
    _seed("dale-harry-potter", "Jim Dale", "Harry Potter, original US single-narrator audiobooks",
          "male,mature,narrator,theatrical,lively,clear,versatile,british",
          "Nimble theatrical storytelling and lively, intelligible character contrasts.",
          "https://www.harrypotter.com/news/cast-revealed-for-new-harry-potter-audiobook-series",
          "The official Harry Potter announcement identifies Jim Dale's earlier single-voice narrations."),
    _seed("holland-peter-parker", "Tom Holland", "Peter Parker, Spider-Man: Homecoming (2017)",
          "male,young,teen,earnest,energetic,conversational,american,vulnerable",
          "An original youthful male voice with earnest energy and vulnerable conversational turns.",
          "https://www.sonypictures.com/movies/spidermanhomecoming",
          "Sony identifies Tom Holland as young Peter Parker in Spider-Man: Homecoming (2017)."),
    _seed("chalamet-laurie", "Timothée Chalamet", "Laurie, Little Women (2019)",
          "male,young,adult,thoughtful,soft,conversational,american,playful",
          "An original young male voice with reflective softness and light playful changes.",
          "https://www.sonypictures.com/movies/littlewomen",
          "Sony credits Timothée Chalamet as Laurie in Little Women (2019)."),
    _seed("radcliffe-harry-potter", "Daniel Radcliffe", "Harry Potter, Harry Potter and the Deathly Hallows: Part 1 (2010)",
          "male,young,teen,earnest,restrained,vulnerable,british,clear",
          "An original young male voice with restrained resolve and emotional vulnerability.",
          "https://www.peacocktv.com/collections/harry-potter-movies/harry-potter-and-the-deathly-hallows-part-1",
          "Peacock's indexed official synopsis identifies Daniel Radcliffe as Harry; page access is region dependent."),
    _seed("ronan-jo-march", "Saoirse Ronan", "Jo March, Little Women (2019)",
          "female,young,adult,energetic,direct,expressive,american,determined",
          "An original young adult voice with direct intent and broad expressive energy.",
          "https://www.sonypictures.com/movies/littlewomen",
          "Sony credits Saoirse Ronan as Jo March in Little Women (2019)."),
    _seed("dern-marmee", "Laura Dern", "Marmee, Little Women (2019)",
          "female,mature,adult,warm,measured,reassuring,american,grounded",
          "An original adult voice with grounded warmth and calm reassurance.",
          "https://www.sonypictures.com/movies/littlewomen",
          "Sony credits Laura Dern as Marmee in Little Women (2019)."),
    _seed("burton-reading-rainbow", "LeVar Burton", "Reading Rainbow, The Tin Forest, host passages",
          "male,adult,narrator,warm,curious,clear,american,reassuring",
          "An original storyteller with welcoming warmth, clear explanation and curiosity.",
          "https://www.pbs.org/video/the-tin-forest-xew5hj/",
          "PBS's episode transcript identifies LeVar Burton as host; the reference is his hosting, not the guest book reader."),
)


def _seeds():
    """The hand-written catalogue rows, still the base of the pool.

    Kept as a function so the pool can be assembled after import without the
    module-level constant and the merged pool drifting apart.
    """
    return _SEEDS


#: The complete casting pool: the seeds above plus every sourced and adopted row.
#: Assembled once at import, with the reasons any row was refused.
_POOL = performer_catalog.load_pool(_seeds())
CATALOG = _POOL["performers"]
CATALOG_POOL = _POOL


@dataclass(frozen=True)
class DiscoveryLimits:
    max_requests: int = 12
    max_candidates: int = 8
    max_recordings_per_performer: int = 2
    max_response_bytes: int = 2_000_000
    max_storage_bytes: int = 24_000_000
    max_retries: int = 1
    max_downloads: int = 2
    max_download_bytes: int = 8_000_000
    timeout_seconds: float = 8.0
    backoff_seconds: float = 0.25
    cache_ttl_seconds: float = 604800.0
    retry_cooldown_seconds: float = 300.0

    def __post_init__(self):
        bounds = {"max_requests": (0, 32), "max_candidates": (1, 10),
                  "max_recordings_per_performer": (1, 3), "max_response_bytes": (256, 4_000_000),
                  "max_storage_bytes": (1024, 100_000_000), "max_retries": (0, 2),
                  "max_downloads": (0, 4), "max_download_bytes": (1024, 32_000_000),
                  "timeout_seconds": (0.1, 15), "backoff_seconds": (0, 2),
                  "cache_ttl_seconds": (0, 2_592_000), "retry_cooldown_seconds": (1, 86400)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
            if name.startswith("max_") and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")


class DiscoveryHold(RuntimeError):
    """Bound or upstream condition stops this metadata request, not matching."""


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _tokens(value):
    if isinstance(value, (list, tuple, set)):
        value = " ".join(str(x) for x in value)
    return set(re.findall(r"[a-z]+", str(value or "").casefold()))


def _query_tags(description, traits):
    # Appearance and names never determine voice, gender or accent. Only explicit
    # vocal/delivery fields in the structured dossier contribute.
    words = _tokens(description)
    for field in ("gender", "pronouns", "age", "age_range", "life_stage", "language", "accent",
                  "voice_description", "personality", "temperament", "role", "delivery", "tags",
                  # Studio's existing evidence-based CastingProfile vocabulary.
                  "presentation", "perceived_vocal_age", "pitch", "weight", "resonance", "texture",
                  "articulation", "cadence", "tempo", "formality", "warmth", "intensity",
                  "confidence", "breathiness", "roughness", "theatricality", "emotional_range"):
        value = (traits or {}).get(field)
        if isinstance(value, dict):
            value = value.get("value") if value.get("confidence") != "UNKNOWN" and value.get("level") != "UNKNOWN" else None
        words |= _tokens(value)
    synonyms = {"man": "male", "boy": "male", "he": "male", "woman": "female", "girl": "female",
                "she": "female", "youthful": "young", "teenager": "teen", "teenage": "teen",
                "masculine": "male", "feminine": "female", "adolescent": "teen", "child": "young",
                "elderly": "mature", "older": "mature", "narration": "narrator", "storyteller": "narrator",
                "warmth": "warm", "gentle": "soft", "quiet": "soft", "curiosity": "curious",
                "humor": "dry", "humour": "dry", "theatricality": "theatrical", "english": "en"}
    words |= {synonyms[w] for w in words if w in synonyms}
    if words & {"teen", "boy", "girl"}:
        words.add("young")
    return words


def _valid_feedback(row):
    version = row.get("version")
    version_valid = (isinstance(version, int) and not isinstance(version, bool) and version >= 1) or (
        isinstance(version, str) and bool(re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", version)))
    return bool(row.get("voice_id") and version_valid and re.fullmatch(r"[a-fA-F0-9]{64}", str(row.get("audio_sha256", ""))))


def rank_candidates(description: str = "", *, traits: Mapping | None = None,
                    preferences: Iterable[Mapping] = (), limit: int = 5,
                    role_id: str | None = None) -> list[dict]:
    """Pure explainable ranking. Feedback must retain voice/version/audio linkage.

    Missing ratings are neutral. Suitability applies only to the matching role.
    Returned candidates always have favorite=False/rating=None: no inheritance.
    """
    limit = max(1, min(int(limit), len(CATALOG)))
    requested = _query_tags(description, traits)
    rows = [dict(row) for row in preferences if _valid_feedback(row)]
    candidates = []
    weights = {"young": 4, "teen": 2, "adult": 1, "mature": 4,
               "male": 3, "female": 3, "narrator": 4, "british": 4, "american": 4}
    for original in CATALOG:
        candidate = copy.deepcopy(original)
        tags = set(candidate["tags"])
        matches = sorted(requested & tags)
        character_fit = sum(weights.get(tag, 1.5) for tag in matches)
        mismatch = 0
        for a, b in (("male", "female"), ("young", "mature"), ("british", "american")):
            if (a in requested and b not in requested and b in tags) or (b in requested and a not in requested and a in tags):
                mismatch -= 12 if a == "male" else 6
        language_value = (traits or {}).get("language", "")
        if isinstance(language_value, Mapping):
            language_value = language_value.get("value", "")
        language = str(language_value or "").casefold()
        language_penalty = -8 if language and language not in {"en", "english", "unknown"} else 0
        preference_score = 0.0
        feedback_keys = []
        for row in rows:
            inspiration = row.get("inspiration") or {}
            exact = row.get("candidate_id") == candidate["candidate_id"] or (
                inspiration.get("performer") == candidate["performer"] and inspiration.get("performance") == candidate["performance"])
            related = inspiration.get("performer") == candidate["performer"]
            overlap = len(_tokens(row.get("tags", [])) & tags)
            affinity = 1.0 if exact else (0.6 if related else min(0.3, overlap * 0.08))
            rating = row.get("rating")
            signal = 2.0 if row.get("favorite") is True else 0.0
            if isinstance(rating, (int, float)) and not isinstance(rating, bool) and 1 <= rating <= 5:
                signal += (rating - 3) * 1.5
            suitability = row.get("suitability")
            suitability_role_matches = role_id and row.get("role_id") == role_id
            if isinstance(suitability, Mapping):
                suitability = suitability.get(role_id) if role_id else None
                suitability_role_matches = bool(role_id)
            if suitability_role_matches and isinstance(suitability, (int, float)) and not isinstance(suitability, bool) and 1 <= suitability <= 5:
                signal += (suitability - 3) * 1.5
            contribution = affinity * signal
            preference_score += contribution
            if contribution:
                feedback_keys.append({"voice_id": row["voice_id"], "version": row["version"],
                                      "audio_sha256": row["audio_sha256"], "contribution": round(contribution, 3)})
        preference_score = max(-8, min(8, preference_score))
        factors = {"character_fit": character_fit, "compatibility": mismatch + language_penalty,
                   "evidence": 1.0, "preference": round(preference_score, 3), "diversity": 0.0}
        candidate.update(id=candidate["candidate_id"], score=round(sum(factors.values()), 3), score_factors=factors,
                         matched_traits=matches, preference_evidence=feedback_keys,
                         favorite=False, rating=None, suitability=None,
                         explanation=("Creative fit: " + ", ".join(matches) if matches else "Exploratory performance reference; role traits remain unspecified.")
                         + (f"; recorded preference contribution {preference_score:+.2f}." if preference_score else ".")
                         + " Metadata only; listen to the original synthetic production voice before judging it.",
                         generation_direction=candidate["creative_direction"], design_instruction=candidate["creative_direction"],
                         inspiration={"performer": candidate["performer"], "performance": candidate["performance"], "source_urls": candidate["source_urls"]})
        candidates.append(candidate)
    # Small deterministic novelty penalty avoids a shortlist dominated by one
    # performance family. No randomness makes refresh/resume reproducible.
    selected = []
    while candidates and len(selected) < limit:
        for c in candidates:
            similarity = max(len(set(c["tags"]) & set(s["tags"])) / max(1, len(set(c["tags"]) | set(s["tags"]))) for s in selected) if selected else 0
            c["score_factors"]["diversity"] = round(-1.5 * similarity, 3)
            c["score"] = round(sum(c["score_factors"].values()), 3)
        candidates.sort(key=lambda c: (-c["score"], c["candidate_id"]))
        selected.append(candidates.pop(0))
    return selected


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DiscoveryHold("Redirect held: official metadata requests never follow redirects")


def _request_json(url: str, *, timeout: float, max_bytes: int) -> tuple[dict, str]:
    # Keep the canonical fixed-endpoint contract and injected test transports.
    parts = urlsplit(url)
    if (parts.scheme, parts.netloc, parts.path) not in {
        ("https", "www.wikidata.org", "/w/api.php"),
        ("https", "commons.wikimedia.org", "/w/api.php"),
        ("https", "api.openverse.org", "/v1/audio/"),
    }:
        raise DiscoveryHold("Endpoint is outside the fixed metadata allowlist")
    from .public_discovery_tools import PublicDiscoveryClient
    return PublicDiscoveryClient().metadata_transport(url, timeout=timeout, max_bytes=max_bytes)


def _retryable_public_error(error) -> bool:
    from .public_discovery_tools import DiscoveryError
    if not isinstance(error, DiscoveryError):
        return False
    status = getattr(error, "http_status", None)
    return ((error.code == "http" and type(status) is int and (status == 429 or 500 <= status <= 599))
            or error.code == "upstream" and getattr(error, "api_error_code", None) == "maxlag"
            or error.code in {"network", "deadline", "dns", "dns_busy", "dns_abandoned"})


@contextmanager
def _cache_lock(directory):
    """OS lock has no stale process ownership after crash; this is not a queue."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".discovery.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DiscoveryHold("Discovery is already running for this project") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class MetadataClient:
    """Durable per-request cache, budget, backoff and resume; no background work."""
    def __init__(self, directory: Path, *, limits: DiscoveryLimits | None = None,
                 online: bool = True, transport: Callable | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limits = limits or DiscoveryLimits()
        self.online = online
        self.transport = transport or _request_json
        self.stats = {"requests": 0, "cache_hits": 0, "bytes_received": 0, "downloads": 0, "generated_audio_seconds": 0}

    def save(self, path: Path, value):
        raw = _json_bytes(value)
        used = sum(p.stat().st_size for p in self.directory.rglob("*") if p.is_file())
        # Include temporary stage, so even peak storage remains bounded.
        if used + len(raw) > self.limits.max_storage_bytes:
            raise DiscoveryHold("Discovery storage budget exhausted; retained existing evidence")
        stage = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with stage.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            replace_metadata_file(stage, path)
        finally:
            stage.unlink(missing_ok=True)

    def get(self, endpoint: str, params: Mapping) -> dict:
        if endpoint not in {WIKIDATA_API, COMMONS_API, OPENVERSE_API}:
            raise DiscoveryHold("Endpoint is outside the fixed metadata allowlist")
        url = endpoint + "?" + urlencode(sorted(params.items()))
        key = hashlib.sha256(url.encode()).hexdigest()
        pointer = self.directory / ("request-" + key + ".json")
        if pointer.exists():
            cached = json.loads(pointer.read_text("utf-8"))
            if cached.get("retry_after", 0) > time.time():
                raise DiscoveryHold(cached["hold"])
            if "payload" in cached and (not self.online or time.time() - cached["retrieved_epoch"] <= self.limits.cache_ttl_seconds):
                if _digest(cached["payload"]) != cached["payload_sha256"]:
                    raise DiscoveryHold("Cached metadata integrity mismatch")
                self.stats["cache_hits"] += 1
                return cached
        if not self.online:
            raise DiscoveryHold("Offline: official metadata not cached")
        last_error = "Request budget exhausted"
        for attempt in range(self.limits.max_retries + 1):
            if self.stats["requests"] >= self.limits.max_requests:
                raise DiscoveryHold("Request budget exhausted; resume uses durable successful requests")
            self.stats["requests"] += 1
            try:
                payload, content_sha256 = self.transport(url, timeout=self.limits.timeout_seconds, max_bytes=self.limits.max_response_bytes)
                # A ``maxlag`` reply is the official API saying "busy, come back",
                # which is a retryable condition rather than a metadata error. It
                # arrives as HTTP 200 with an error body, so it has to be detected
                # here; treating it as a hard failure would report the upstream's
                # congestion as a missing source and quietly shrink the pool.
                lag = (payload.get("error") or {}).get("code") if isinstance(payload, dict) else None
                if lag == "maxlag":
                    detail = str((payload.get("error") or {}).get("info") or "")[:200]
                    raise DiscoveryHold(f"Official API is lagging and asked for a retry: {detail}")
                size = len(_json_bytes(payload))
                if size > self.limits.max_response_bytes:
                    raise DiscoveryHold("Metadata response exceeds the byte limit")
                self.stats["bytes_received"] += size
                result = {"url": url, "retrieved_at": _now(), "retrieved_epoch": time.time(),
                          "content_sha256": content_sha256, "payload_sha256": _digest(payload), "payload": payload}
                # Content-addressed immutable snapshot plus small mutable request cache.
                snapshot = self.directory / ("evidence-" + result["payload_sha256"] + ".json")
                if not snapshot.exists():
                    self.save(snapshot, result)
                self.save(pointer, result)
                return result
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, DiscoveryHold) as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                retryable = isinstance(exc, (URLError, TimeoutError)) and not isinstance(exc, HTTPError)
                retryable |= isinstance(exc, HTTPError) and (exc.code == 429 or exc.code >= 500)
                retryable |= _retryable_public_error(exc)
                # An upstream that asked us to wait is a retryable condition too,
                # and it is the common one during a lagging replica.
                retryable |= isinstance(exc, DiscoveryHold) and "asked for a retry" in str(exc)
                if not retryable or attempt == self.limits.max_retries:
                    break
                time.sleep(self.limits.backoff_seconds * 2 ** attempt)
        hold = {"url": url, "hold": last_error, "retrieved_at": _now(),
                "retry_after": time.time() + self.limits.retry_cooldown_seconds}
        self.save(pointer, hold)
        raise DiscoveryHold(last_error)


def _plain(value):
    value = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", str(value or ""), flags=re.I | re.S)
    return html.unescape(re.sub(r"<[^>]*>", "", value))[:4000]


def reference_eligibility(source: Mapping, *, intended_use_evidence: Mapping | None = None) -> dict:
    """Independent download/intended-use decisions. Never infer voice consent.

    Evidence must be a trusted local owner decision, not fields extracted from
    an external metadata page. Callers must not populate it from fetched HTML.
    """
    holds = []
    official = official_recording_sources.reviewed(source)
    media = urlsplit(str(source.get("media_url") or ""))
    license_url = str(source.get("license_url") or "")
    open_license = bool(re.fullmatch(r"https?://creativecommons\.org/(?:licenses/(?:by|by-sa)/[1-4]\.0|publicdomain/(?:zero/1\.0|mark/1\.0))/?", license_url))
    if not open_license and official is None:
        holds.append("Recording license missing or outside the reviewed download licenses")
    if not source.get("attribution"):
        holds.append("Attribution metadata missing")
    if not source.get("source_page") or not source.get("terms_url"):
        holds.append("Source page or terms evidence missing")
    if official is None and (media.scheme != "https" or media.netloc != "upload.wikimedia.org"
                             or not media.path.startswith("/wikipedia/commons/") or media.fragment):
        holds.append("Media URL missing or outside permitted Commons storage")
    if source.get("restrictions"):
        holds.append("Source reports additional restrictions requiring review")
    download = not holds
    evidence = dict(intended_use_evidence or {})
    authorities = {"performer_consent", "voice_license", "owner_original_voice"}
    # An explicit private owner request is distinct from performer consent and
    # applies only to this reviewed source; never infer it from public metadata.
    if official is not None and official.get("provider") == "reviewed_official_recording":
        authorities.add("owner_private_use_request")
    intended = bool(evidence.get("status") == "approved" and evidence.get("use") == "private_synthetic_voice"
                    and evidence.get("authority") in authorities
                    and evidence.get("evidence_ref") and evidence.get("source_id") == source.get("source_id"))
    if not intended:
        holds.append("No source-bound evidence authorizing private synthetic-voice use; recording license does not establish voice/personality rights")
    if not source.get("language"):
        holds.append("Recording language unknown")
    duration = source.get("duration_seconds")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
        holds.append("Recording duration unknown")
    return {"download_eligible": download, "intended_use_eligible": intended,
            "acquisition_eligible": download and intended and not holds,
            "clone_ready": False, "technical_review": "not_analyzed",
            "hold_reasons": holds, "intended_use": "private_synthetic_voice",
            "intended_use_evidence": evidence or None}


def _commons_source(client: MetadataClient, match: dict, entity_id: str, claim: dict) -> dict:
    filename = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
    if not isinstance(filename, str) or len(filename) > 500 or not re.search(r"\.(ogg|oga|opus|flac|mp3|wav|webm)$", filename, re.I):
        raise DiscoveryHold("P990 recording filename is missing or invalid")
    response = client.get(COMMONS_API, {"action": "query", "format": "json", "formatversion": 2,
            "prop": "imageinfo", "titles": "File:" + filename, "iiprop": "url|size|sha1|mime|extmetadata|metadata",
            "iiextmetadatalanguage": "en", "maxlag": 5})
    pages = response["payload"].get("query", {}).get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    info = (pages[0].get("imageinfo") or [{}])[0] if pages else {}
    ext = info.get("extmetadata") or {}
    metadata = {item.get("name"): item.get("value") for item in info.get("metadata", []) if isinstance(item, dict)}
    def field(name):
        return _plain((ext.get(name) or {}).get("value"))
    language_ids = [q.get("datavalue", {}).get("value", {}).get("id") for q in claim.get("qualifiers", {}).get("P407", [])]
    known_languages = {"Q1860": "en", "Q150": "fr", "Q188": "de", "Q1321": "es", "Q7850": "zh", "Q5287": "ja"}
    duration = info.get("duration") or metadata.get("length") or metadata.get("duration") or metadata.get("playtime_seconds")
    try:
        duration = float(duration) if duration is not None else None
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            duration = None
    except (TypeError, ValueError):
        duration = None
    source = {"source_id": "commons-" + _digest(filename)[:24], "provider": "wikimedia_commons",
              "performer": match["performer"], "entity_id": entity_id, "performance": None,
              "inspiration_performance": match["performance"],
              "performance_linkage": "P990 is a spoken-voice recording; it is not verified as the named performance",
              "filename": filename, "source_page": info.get("descriptionurl"), "media_url": info.get("url"),
              "attribution": field("Artist") or field("Attribution"), "license": field("LicenseShortName"),
              "license_url": field("LicenseUrl"), "terms_url": TERMS_URL,
              "restrictions": field("Restrictions"), "usage_terms": field("UsageTerms"),
              "recording_description": field("ImageDescription"), "credit": field("Credit"),
              "recording_license_evidence": field("Permission"),
              "retrieved_at": response["retrieved_at"], "metadata_content_sha256": response["content_sha256"],
              "content_sha256": None, "upstream_media_sha1": info.get("sha1"),
              "language": next((known_languages[q] for q in language_ids if q in known_languages), None),
              "language_entity_ids": language_ids, "duration_seconds": duration, "size_bytes": info.get("size"),
              "mime": info.get("mime"), "evidence_url": response["url"],
              "preprocessing": [], "audio_observations": [], "original_path": None}
    source.update(reference_eligibility(source))
    return source


def _performer_sources(client, match):
    reviewed = official_recording_sources.records_for(match["performer"])
    if reviewed:
        for source in reviewed:
            source.update(reference_eligibility(source))
        match["recording_discovery"] = "reviewed_official_metadata"
        return reviewed[:client.limits.max_recordings_per_performer]
    search = client.get(WIKIDATA_API, {"action": "wbsearchentities", "format": "json", "language": "en",
            "uselang": "en", "search": match["performer"], "type": "item", "limit": 8, "maxlag": 5})
    entities = search["payload"].get("search", [])
    exact = [e for e in entities if _plain(e.get("label")).casefold() == match["performer"].casefold()
             and re.fullmatch(r"Q[1-9][0-9]*", str(e.get("id", "")))]
    # Use explicit performer occupation metadata to distinguish a credited
    # performer from same-name books, athletes, historians, and other homonyms.
    # More than one plausible performer stays held; never choose by ordering.
    if len(exact) > 1:
        performing = [e for e in exact if _tokens(e.get("description")) &
                      {"actor", "actress", "narrator", "comedian", "broadcaster", "presenter"}]
        if len(performing) == 1:
            exact = performing
    if len(exact) != 1:
        raise DiscoveryHold("Performer entity unresolved or ambiguous in official search")
    qid = exact[0]["id"]
    entity = client.get(WIKIDATA_API, {"action": "wbgetentities", "format": "json", "ids": qid,
            "props": "claims|labels|descriptions", "languages": "en", "maxlag": 5})
    data = entity["payload"].get("entities", {}).get(qid, {})
    if not any(c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") == "Q5" for c in data.get("claims", {}).get("P31", [])):
        raise DiscoveryHold("Resolved entity is not explicitly identified as a human")
    match["entity_id"] = qid
    match["entity_source_url"] = f"https://www.wikidata.org/wiki/{qid}"
    match["entity_evidence_sha256"] = entity["content_sha256"]
    claims = [c for c in data.get("claims", {}).get("P990", []) if c.get("rank") != "deprecated"]
    sources = []
    for claim in claims[:client.limits.max_recordings_per_performer]:
        sources.append(_commons_source(client, match, qid, claim))
    match["recording_discovery"] = "metadata_found" if sources else "no_P990_recording"
    return sources


def discover(directory: str | Path, description: str = "", *, traits: Mapping | None = None,
             preferences: Iterable[Mapping] = (), limit: int = 5, online: bool = True,
             limits: DiscoveryLimits | None = None, role_id: str | None = None) -> dict:
    """Rank locally, retrieve bounded official metadata, save resumable evidence.

    directory is the existing owner's project/voice_lab/discovery directory.
    Scheduler owns pause/resume; calling again resumes cached requests. The
    result never grants cloning consent or starts audio generation/acquisition.
    """
    limits = limits or DiscoveryLimits()
    matches = rank_candidates(description, traits=traits, preferences=preferences,
                              limit=min(limit, limits.max_candidates), role_id=role_id)
    directory = Path(directory)
    with _cache_lock(directory):
        client = MetadataClient(directory, limits=limits, online=online)
        sources, errors, completed = [], [], []
        for match in matches:
            try:
                records = _performer_sources(client, match)
                sources.extend(records)
                match["recording_source_ids"] = [s["source_id"] for s in records]
                completed.append(match["candidate_id"])
            except (DiscoveryHold, OSError, ValueError, KeyError, TypeError) as exc:
                match["recording_discovery"] = "held"
                match["recording_hold"] = str(exc)[:400]
                errors.append({"candidate_id": match["candidate_id"], "reason": str(exc)[:400]})
        unique_sources = list({s["source_id"]: s for s in sources}.values())
        result = {"catalog_version": CATALOG_VERSION, "matches": matches, "sources": unique_sources,
                  "status": "complete" if not errors else "partial", "errors": errors,
                  "stats": client.stats, "limits": asdict(limits), "retrieved_at": _now(),
                  "cursor": {"completed_candidate_ids": completed,
                             "pending_candidate_ids": [m["candidate_id"] for m in matches if m["candidate_id"] not in completed]},
                  "synthetic_audio_replenishment": "idle", "audio_generation_triggered": False}
        # Store a bounded stable query slot. Private request text is not persisted
        # here; its hash ties resume state to the owner's request.
        result["request_hash"] = _digest({"description": description, "traits": traits or {}, "role_id": role_id})
        try:
            client.save(directory / "latest.json", result)
        except DiscoveryHold as exc:
            result["status"] = "partial"
            result["errors"].append({"reason": str(exc)})
        return result


def probe_openverse(directory: str | Path, *, limits: DiscoveryLimits | None = None) -> dict:
    """Optional one-page metadata probe; never adopts results as voice identity."""
    with _cache_lock(Path(directory)):
        client = MetadataClient(Path(directory), limits=limits or DiscoveryLimits(max_requests=1, max_retries=0))
        try:
            response = client.get(OPENVERSE_API, {"q": "spoken voice", "page_size": 3})
            rows = response["payload"].get("results", [])
            return {"status": "reachable", "relevant_performer_recordings_verified": 0,
                    "adopted_as_provider": False, "reason": "Search results do not establish performer identity or intended-use eligibility",
                    "results_received": len(rows), "response_sha256": response["content_sha256"], "stats": client.stats}
        except (DiscoveryHold, OSError, ValueError) as exc:
            return {"status": "unavailable", "adopted_as_provider": False, "reason": str(exc), "stats": client.stats}


def _download_media(url: str, *, timeout: float, max_bytes: int) -> bytes:
    parts = urlsplit(url)
    official = official_recording_sources.reviewed_url(url)
    if official is None and (parts.scheme != "https" or parts.netloc != "upload.wikimedia.org"
                             or not parts.path.startswith("/wikipedia/commons/") or parts.fragment):
        raise DiscoveryHold("Media URL is outside approved Commons storage")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with build_opener(_NoRedirects()).open(request, timeout=timeout) as response:
        raw = response.read(max_bytes + 1)
    if not raw or len(raw) > max_bytes:
        raise DiscoveryHold("Recording is empty or exceeds the download byte limit")
    if official is not None:
        try:
            official_recording_sources.verify_original(official, raw)
        except ValueError as exc:
            raise DiscoveryHold(str(exc)) from exc
    return raw


def _save_media(client: MetadataClient, path: Path, raw: bytes):
    used = sum(p.stat().st_size for p in client.directory.rglob("*") if p.is_file())
    if used + len(raw) > client.limits.max_storage_bytes:
        raise DiscoveryHold("Discovery storage budget exhausted")
    if path.exists():
        if path.read_bytes() != raw:
            raise DiscoveryHold("Immutable reference bytes already exist with different content")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with stage.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        replace_metadata_file(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def validate_audio_review(review: Mapping | None, *, original_sha256: str, transcript: str,
                          start_seconds: float, end_seconds: float, source_id: str) -> list[str]:
    """ASR/speaker/music checks require measured or human-reviewed local evidence.

    These fields must come from the owner's local analysis/review adapter, never
    external web metadata. PCM amplitude is insufficient to infer these facts.
    """
    review = review or {}
    expected = {"original_sha256": original_sha256, "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                "start_seconds": start_seconds, "end_seconds": end_seconds, "source_id": source_id}
    holds = []
    if any(review.get(k) != v for k, v in expected.items()):
        holds.append("Transcript/speaker review is not bound to these exact bytes, transcript and selection")
    if not review.get("evidence_ref") or review.get("method") not in {"local_tools", "human_review"}:
        holds.append("Local transcript/speaker review evidence unavailable")
    for key, required in {"single_speaker": True, "transcript_aligned": True, "overlapping_speech": False,
                          "heavy_music": False, "unsuitable_noise": False}.items():
        if review.get(key) is not required:
            holds.append(f"Audio review missing or failed: {key}")
    return holds


def acquire_reference(directory: str | Path, source: Mapping, *, intended_use_evidence: Mapping,
                      transcript: str | None = None, start_seconds: float, end_seconds: float,
                      audio_review: Mapping | None = None, limits: DiscoveryLimits | None = None,
                      review_only: bool = False, isolate_speech: bool = False,
                      automatic_preparation: bool | None = None, transcript_origin: str = "provided",
                      cached_original_only: bool = False, audition_draft: bool = False) -> dict:
    """Acquire an eligible reviewed original and prepare one bounded clean segment.

    Missing transcripts use installed offline CPU ASR for this exact crop.
    No GPU is used. Missing alignment, music or speaker evidence
    produces a durable held receipt, never a clone-ready declaration. The worker
    accepts only an acquisition id + hash, scoped to this project's fixed path.
    """
    import io
    import numpy as np
    import soundfile as sf

    if not isinstance(audition_draft, bool):
        raise DiscoveryHold("Audition-draft flag must be a boolean")
    if not isinstance(isolate_speech, bool):
        raise DiscoveryHold("isolate_speech must be a boolean")
    if automatic_preparation is None:
        # New unreviewed source acquisitions use bounded machine preparation.
        # Existing explicit reviews and manual isolation retain their exact recipe.
        automatic_preparation = not (audio_review or review_only or isolate_speech)
    if not isinstance(automatic_preparation, bool) or not isinstance(cached_original_only, bool):
        raise DiscoveryHold("Automatic preparation and cached-original flags must be booleans")
    if automatic_preparation and isolate_speech:
        raise DiscoveryHold("Choose automatic preparation or explicit isolation, not both")
    if transcript_origin not in {"provided", "machine_draft", "user_correction", "unverified_entry"}:
        raise DiscoveryHold("Unsupported transcript evidence origin")
    source = dict(source)
    # Persist registry claims, never extra caller/model assertions about rights,
    # identity or the named performance. Eligibility is computed independently.
    canonical = official_recording_sources.reviewed(source)
    if canonical is not None:
        source = canonical
        if canonical.get("requires_cached_original"):
            cached_original_only = True  # Corpus archive URLs must never become full GETs.
    eligibility = reference_eligibility(source, intended_use_evidence=intended_use_evidence)
    if not eligibility["acquisition_eligible"] and not (review_only and eligibility["download_eligible"]):
        raise DiscoveryHold("Reference acquisition held: " + "; ".join(eligibility["hold_reasons"]))
    if transcript is not None and (not isinstance(transcript, str) or not transcript.strip() or len(transcript) > 1000):
        raise DiscoveryHold("Reference transcript must contain 1-1000 characters")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (start_seconds, end_seconds)):
        raise DiscoveryHold("Reference selection times must be finite numbers")
    if start_seconds < 0 or not 3 <= end_seconds - start_seconds <= 30:
        raise DiscoveryHold("Reference segment must be 3-30 seconds inside the original")
    directory = Path(directory)
    limits = limits or DiscoveryLimits()
    with _cache_lock(directory):
        client = MetadataClient(directory, limits=limits)
        original_key = _digest({"source_id": source["source_id"], "media_url": source["media_url"],
                                "sha1": source.get("upstream_media_sha1")})
        download_folder = directory / "downloads"
        original = download_folder / (original_key + ".audio")
        original_record = download_folder / (original_key + ".json")
        if original.exists():
            raw = original.read_bytes()
            if len(raw) > limits.max_download_bytes:
                raise DiscoveryHold("Cached original exceeds the byte limit")
            original_sha = hashlib.sha256(raw).hexdigest()
            if original_record.exists() and json.loads(original_record.read_text("utf-8"))["content_sha256"] != original_sha:
                raise DiscoveryHold("Original recording checksum mismatch")
        else:
            if cached_original_only:
                raise DiscoveryHold("Automatic preparation requires the existing verified original; no download attempted")
            if len(list(download_folder.glob("*.audio"))) >= limits.max_downloads:
                raise DiscoveryHold("Reference download count limit reached")
            # A single bounded attempt. Explicit resume reuses any completed file.
            raw = _download_media(source["media_url"], timeout=limits.timeout_seconds, max_bytes=limits.max_download_bytes)
            original_sha = hashlib.sha256(raw).hexdigest()
            upstream_sha1 = source.get("upstream_media_sha1")
            if upstream_sha1:
                expected = str(upstream_sha1).lower()
                if len(expected) != 40:
                    try:
                        expected = f"{int(expected, 36):040x}"
                    except ValueError as exc:
                        raise DiscoveryHold("Invalid upstream media checksum") from exc
                if hashlib.sha1(raw).hexdigest() != expected:
                    raise DiscoveryHold("Downloaded recording differs from Commons media checksum")
            try:
                official_recording_sources.verify_original(source, raw)
            except ValueError as exc:
                raise DiscoveryHold(str(exc)) from exc
            _save_media(client, original, raw)
            client.stats["downloads"] = 1
        try:
            official_recording_sources.verify_original(source, raw)
        except ValueError as exc:
            raise DiscoveryHold(str(exc)) from exc
        if not original_record.exists():
            client.save(original_record, {"source": source, "content_sha256": original_sha, "retrieved_at": _now()})
        from .reference_codec import decode_reference
        try:
            decoded = decode_reference(original, start_seconds, end_seconds)
        except ValueError as exc:
            raise DiscoveryHold(str(exc)) from exc
        wave, rate = decoded.wave, decoded.sample_rate
        start_frame, end_frame = round(start_seconds * rate), round(end_seconds * rate)
        source_processing = None
        automatic_evidence = None
        automatic_cache_hit = False
        narration_assessment = (audio_review or {}).get("narration_delivery")
        if automatic_preparation:
            from .reference_preparation import prepare
            wave, rate, source_processing, automatic_evidence, automatic_cache_hit = prepare(
                directory, wave, rate, transcript, language=source.get("language") or "en",
                expected_origin=transcript_origin if transcript else "machine_draft",
                storage_budget_bytes=limits.max_storage_bytes)
            if automatic_evidence["status"] == "pass":
                audio_review = {
                    "original_sha256": original_sha,
                    "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                    "start_seconds": start_seconds, "end_seconds": end_seconds, "source_id": source["source_id"],
                    "evidence_ref": "automatic-source-qc:" + _digest(automatic_evidence),
                    "method": "local_tools", "single_speaker": True, "transcript_aligned": True,
                    "overlapping_speech": False, "heavy_music": False, "unsuitable_noise": False,
                    "human_verified": False, "automatic_preparation": automatic_evidence,
                    **({"narration_delivery": narration_assessment} if narration_assessment is not None else {}),
                    **({"source_processing": source_processing} if source_processing else {})}
        elif isolate_speech:
            from .reference_separation import isolate_speech as prepare_speech
            wave, rate, source_processing = prepare_speech(wave, rate)
        elif decoded.channels == 2:
            wave = wave.mean(axis=1)
        peak = float(np.max(np.abs(wave)))
        clipping_fraction = float(np.mean(np.abs(wave) >= 0.999))
        silent_fraction = float(np.mean(np.abs(wave) < 0.001))
        output = io.BytesIO()
        # Stereo mean, contiguous crop and lossless float WAV; no pitch/gain/resampling.
        sf.write(output, wave, rate, format="WAV", subtype="FLOAT")
        reference_raw = output.getvalue()
        transcript_evidence = None
        if transcript is None:
            from .reference_transcripts import transcribe_bytes
            transcript_evidence = transcribe_bytes(directory, reference_raw, language=source.get("language", "en"))
            transcript_evidence.pop("cache_hit", None)  # Immutable receipt is independent of cache hits.
            transcript = transcript_evidence.get("text", "")
        holds = validate_audio_review(audio_review, original_sha256=original_sha, transcript=transcript,
                                       start_seconds=start_seconds, end_seconds=end_seconds, source_id=source["source_id"])
        if source_processing is not None and (
                not audio_review or audio_review.get("source_processing") != source_processing):
            holds.append("Audio review is not bound to this exact speech-isolation recipe/output")
        if automatic_evidence is not None and automatic_evidence["status"] != "pass":
            holds.append(automatic_evidence["reason"])
        if review_only:
            holds.append("Review-only preparation; source-use approval and audio review remain required before cloning")
        if peak > 1 or clipping_fraction > 0.001:
            holds.append("Selected reference is clipped")
        if silent_fraction > 0.65 or float(np.sqrt(np.mean(wave ** 2))) < 0.005:
            holds.append("Selected reference contains unsuitable silence or low signal")
        if not transcript:
            holds.append("Automatic transcript unavailable; supply/correct transcript before use")
        technical_holds = list(holds)
        delivery_fit = None
        draft_policy = None
        if automatic_preparation or source.get("requires_narration_delivery_review"):
            # Technical cleanliness cannot certify calm/sustained narration.
            # This applies to new automatic acquisitions; existing immutable
            # references/identities and explicit historical recipes are retained.
            from .reference_delivery_fit import evaluate
            delivery_fit = evaluate(wave, rate, source=source,
                start_seconds=start_seconds, end_seconds=end_seconds,
                technical_status="hold" if holds else "pass", transcript=transcript,
                audio_assessment=narration_assessment)
            if audition_draft and not technical_holds and delivery_fit["delivery_decision"] == "review":
                from .recorded_audition_policy import policy_for
                try:
                    draft_policy = policy_for(wave, rate, source=source, original_sha256=original_sha,
                        transcript=transcript, start_seconds=start_seconds, end_seconds=end_seconds)
                except ValueError as exc:
                    holds.append(str(exc))
            if not delivery_fit["selected_reference"] and draft_policy is None:
                holds.append("Narration delivery fit requires suitable audio evidence: "
                    + ", ".join(delivery_fit["reason_codes"]))
        reference_sha = hashlib.sha256(reference_raw).hexdigest()
        request = {"source_id": source["source_id"], "original_sha256": original_sha, "transcript": transcript,
                   "start_seconds": start_seconds, "end_seconds": end_seconds,
                   "intended_use_evidence": dict(intended_use_evidence), "audio_review": dict(audio_review or {}),
                   **({"source_processing": source_processing} if source_processing else {}),
                   **({"automatic_preparation": automatic_evidence} if automatic_evidence else {}),
                   **({"narration_delivery_fit": delivery_fit} if delivery_fit else {}),
                   **({"audition_draft_policy": draft_policy} if draft_policy else {}),
                   **({"review_only": True} if review_only else {})}
        acquisition_id = _digest(request)[:32]
        folder = directory / "acquisitions" / acquisition_id
        reference = folder / "reference.wav"
        receipt_path = folder / "acquisition.json"
        # FLOAT WAV PEAK headers can contain wall-clock timestamps. Reuse the
        # existing immutable bytes only after exact PCM/format comparison; the
        # receipt hash below still rejects any changed existing artifact.
        if reference.exists():
            existing_raw = reference.read_bytes()
            existing_wave, existing_rate = sf.read(io.BytesIO(existing_raw), dtype="float32")
            if existing_rate != rate or existing_wave.shape != wave.shape or not np.array_equal(existing_wave, wave):
                raise DiscoveryHold("Immutable reference PCM already exists with different content")
            reference_raw = existing_raw
            reference_sha = hashlib.sha256(reference_raw).hexdigest()
            if transcript_evidence and transcript_evidence["audio_sha256"] != reference_sha:
                # A regenerated FLOAT WAV may differ only in its PEAK timestamp.
                # ASR must describe the preserved bytes, not that temporary header.
                transcript_evidence = transcribe_bytes(directory, reference_raw,
                                                       language=source.get("language", "en"))
                transcript_evidence.pop("cache_hit", None)
                if transcript_evidence.get("text", "") != transcript:
                    raise DiscoveryHold("Preserved reference transcript differs on exact-byte replay")
        _save_media(client, reference, reference_raw)
        receipt = {"schema": "voice-lab-acquisition-v1", "acquisition_id": acquisition_id,
                   "source": source, "eligibility": eligibility, "original_sha256": original_sha,
                   "original_path": original.relative_to(directory).as_posix(),
                   "reference_path": reference.relative_to(directory).as_posix(),
                   "reference_audio_sha256": reference_sha, "reference_text": transcript,
                   "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                   "start_seconds": start_seconds, "end_seconds": end_seconds,
                   "duration_seconds": len(wave) / rate, "sample_rate": rate,
                   "audio_review": dict(audio_review or {}),
                   **({"transcript_evidence": transcript_evidence} if transcript_evidence else {}),
                   "technical_qa": {"status": "pass" if not technical_holds else "hold", "peak": peak,
                                    "clipping_fraction": clipping_fraction, "silent_fraction": silent_fraction,
                                    "holds": technical_holds},
                   "readiness_holds": holds,
                   "clone_ready": not holds,
                   **({"audition_draft_policy": draft_policy, "audition_draft_ready": True,
                       "source_fit_review_required": True, "production_approved": False} if draft_policy else {}),
                   **({"automatic_preparation": automatic_evidence} if automatic_evidence else {}),
                   **({"narration_delivery_fit": delivery_fit} if delivery_fit else {}),
                   **({"source_decode": decoded.decoder} if decoded.decoder is not None else {}),
                   **({"source_processing": source_processing} if source_processing else {}),
                   "preprocessing": (
                       [{"operation": "contiguous_crop", "start_frame": start_frame, "end_frame": end_frame},
                        {"operation": "reference_vocals_isolation", "recipe": source_processing},
                        {"operation": "lossless_float_wav", "sample_rate": rate}] if source_processing else
                       ([{"operation": "stereo_to_mono_mean", "input_channels": 2}] if decoded.channels == 2 else []) +
                       [{"operation": "contiguous_crop", "start_frame": start_frame, "end_frame": end_frame},
                        {"operation": "lossless_float_wav", "sample_rate": rate}]
                   ),
                   "retrieved_at": json.loads(original_record.read_text("utf-8"))["retrieved_at"]}
        if receipt_path.exists():
            if json.loads(receipt_path.read_text("utf-8")) != receipt:
                raise DiscoveryHold("Immutable acquisition receipt changed")
        else:
            client.save(receipt_path, receipt)
        return {**receipt, "acquisition_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(), "stats": client.stats,
                **({"automatic_preparation_cache_hit": automatic_cache_hit} if automatic_evidence else {})}
