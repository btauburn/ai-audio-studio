"""No live network: exercise privacy, eligibility, bounds, resume and ranking."""

import copy
import hashlib
import json
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from dubbing_service.app.audiobook import performer_discovery as discovery


def eligible_metadata():
    return {"source_id": "source-1", "source_page": "https://commons.wikimedia.org/wiki/File:Test.ogg",
            "media_url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Test.ogg",
            "attribution": "Fixture artist", "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "terms_url": discovery.TERMS_URL, "language": "en", "duration_seconds": 12}


def permission():
    return {"source_id": "source-1", "status": "approved", "use": "private_synthetic_voice",
            "authority": "performer_consent", "evidence_ref": "local-consent:fixture-only"}


def test_recording_license_does_not_grant_synthetic_voice_permission():
    result = discovery.reference_eligibility(eligible_metadata())
    assert result["download_eligible"] is True
    assert result["intended_use_eligible"] is False
    assert result["acquisition_eligible"] is False
    assert any("voice/personality" in hold for hold in result["hold_reasons"])


def test_explicit_source_bound_permission_is_separate_from_audio_qa():
    result = discovery.reference_eligibility(eligible_metadata(), intended_use_evidence=permission())
    assert result["acquisition_eligible"] is True
    assert result["clone_ready"] is False
    wrong = permission() | {"source_id": "different-recording"}
    assert not discovery.reference_eligibility(eligible_metadata(), intended_use_evidence=wrong)["intended_use_eligible"]


@pytest.mark.parametrize("missing", ["attribution", "license_url", "media_url", "source_page", "terms_url", "language", "duration_seconds"])
def test_missing_metadata_never_acquires(missing):
    source = eligible_metadata()
    del source[missing]
    assert not discovery.reference_eligibility(source, intended_use_evidence=permission())["acquisition_eligible"]


@pytest.mark.parametrize("media", ["http://127.0.0.1/a.wav", "https://upload.wikimedia.org.evil.test/wikipedia/commons/a",
                                   "https://user@upload.wikimedia.org/wikipedia/commons/a", "https://upload.wikimedia.org:443/wikipedia/commons/a"])
def test_media_hosts_and_recording_restrictions_are_held(media):
    assert not discovery.reference_eligibility(eligible_metadata() | {"media_url": media})["download_eligible"]
    assert not discovery.reference_eligibility(eligible_metadata() | {"restrictions": "personality rights"})["download_eligible"]


def test_sparse_young_male_discovers_specific_performances_beyond_user_seeds():
    """A sparse query must reach well past the handful of original catalogue names.

    The exact winners depend on the pool, so the assertion is on the properties
    that matter: every match is genuinely a young male performance, each carries a
    specific credited performance, and the result is not drawn from the small
    original seed set.
    """
    matches = discovery.rank_candidates("young male", limit=3)
    assert len(matches) == 3
    assert all("young" in m["tags"] and "male" in m["tags"] for m in matches)
    assert all(m["source_urls"] and m["performance"] and m["audio_observations"] == [] for m in matches)
    assert all(m["evidence_level"] == "metadata_only_exploratory" for m in matches)
    assert all(m["matched_traits"] for m in matches), "a match must state the trait it matched on"
    # The pool is much larger than the seeds, so a young-male query has real choice.
    young_male = [r for r in discovery.CATALOG if {"young", "male"} <= set(r["tags"])]
    assert len(young_male) >= 6, f"the pool needs several young-male references, got {len(young_male)}"
    assert len({m["performer"] for m in matches}) == 3


def test_appearance_alone_is_never_a_voice_constraint():
    assert discovery.rank_candidates(traits={"appearance": "young male British pale tall"}) == discovery.rank_candidates()


def test_studio_casting_profile_dimensions_drive_explainable_ranking():
    rows = discovery.rank_candidates(traits={"presentation": "masculine", "perceived_vocal_age": "adolescent",
                                            "texture": "clear", "warmth": "warm"}, limit=3)
    assert len(rows) == 3
    assert all({"male", "young"} <= set(m["matched_traits"]) for m in rows)
    assert any("clear" in m["matched_traits"] for m in rows)
    # Each candidate explains itself and names a specific credited performance.
    assert all(m["performance"] and m["explanation"] and m["source_urls"] for m in rows)
    assert len({m["performer"] for m in rows}) == 3, "one round must not repeat a performer"
    unknown = {"presentation": {"value": "feminine", "level": "UNKNOWN"},
               "perceived_vocal_age": {"value": "young", "confidence": "UNKNOWN"}}
    assert discovery.rank_candidates(traits=unknown) == discovery.rank_candidates()


def test_exact_version_feedback_changes_ranking_without_inheritance_or_input_mutation():
    before = discovery.rank_candidates("young male", limit=3)
    target = before[-1]
    preference = {"voice_id": "local-original", "version": 2, "audio_sha256": "a" * 64,
                  "candidate_id": target["candidate_id"], "favorite": True, "rating": 5}
    snapshot = copy.deepcopy(preference)
    after = discovery.rank_candidates("young male", preferences=[preference], limit=3)
    assert after[0]["candidate_id"] == target["candidate_id"]
    assert after[0]["score_factors"]["preference"] > 0
    assert after[0]["preference_evidence"][0]["version"] == 2
    assert all(m["rating"] is None and not m["favorite"] for m in after)
    assert preference == snapshot
    for field in ("voice_id", "version", "audio_sha256"):
        incomplete = preference.copy()
        del incomplete[field]
        assert discovery.rank_candidates("young male", preferences=[incomplete], limit=3) == before


def test_unrated_and_unrelated_role_feedback_are_neutral():
    before = discovery.rank_candidates("young male", limit=3)
    preference = {"voice_id": "local-original", "version": 1, "audio_sha256": "a" * 64,
                  "candidate_id": before[-1]["candidate_id"], "rating": None,
                  "favorite": False, "suitability": 5, "role_id": "other"}
    assert discovery.rank_candidates("young male", preferences=[preference], role_id="current", limit=3) == before
    current = discovery.rank_candidates("young male", preferences=[preference], role_id="other", limit=3)
    assert current[0]["candidate_id"] == before[-1]["candidate_id"]


def test_studio_revision_strings_and_role_suitability_mapping_are_supported():
    before = discovery.rank_candidates("young male", limit=3)
    row = {"voice_id": "saved-voice", "version": "2e55ac9a", "audio_sha256": "d" * 64,
           "candidate_id": before[-1]["candidate_id"], "suitability": {"this-role": 5}}
    after = discovery.rank_candidates("young male", role_id="this-role", preferences=[row], limit=3)
    assert after[0]["candidate_id"] == before[-1]["candidate_id"]
    assert discovery.rank_candidates("young male", role_id="other-role", preferences=[row], limit=3) == before


def fixture_transport(calls, *, p990=True):
    def request(url, **kwargs):
        calls.append(url)
        query = parse_qs(urlsplit(url).query)
        if query.get("action") == ["wbsearchentities"]:
            result = {"search": [{"id": "Q1", "label": query["search"][0]}]}
        elif query.get("action") == ["wbgetentities"]:
            claims = {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}]}
            if p990:
                claims["P990"] = [{"mainsnak": {"datavalue": {"value": "Fixture.ogg"}},
                                    "qualifiers": {"P407": [{"datavalue": {"value": {"id": "Q1860"}}}]}}]
            result = {"entities": {"Q1": {"claims": claims}}}
        else:
            result = {"query": {"pages": [{"imageinfo": [{
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:Fixture.ogg",
                "url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Fixture.ogg", "sha1": "abc", "size": 500,
                "metadata": [{"name": "length", "value": "12.3"}],
                "extmetadata": {"LicenseShortName": {"value": "CC BY 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"},
                                "Artist": {"value": "<a href='https://bad.example'>Fixture Artist</a>"}}
            }]}]}}
        return result, hashlib.sha256(json.dumps(result).encode()).hexdigest()
    return request


def test_official_discovery_cache_resume_never_uploads_description_or_duplicates(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(discovery, "_request_json", fixture_transport(calls))
    private = "PRIVATE CHAPTER TEXT young male"
    first = discovery.discover(tmp_path, private, limit=1)
    assert first["status"] == "complete"
    assert first["stats"]["requests"] == 3
    assert all("PRIVATE" not in url for url in calls)
    source = first["sources"][0]
    assert source["download_eligible"] and not source["intended_use_eligible"]
    assert source["duration_seconds"] == 12.3 and source["language"] == "en"
    assert source["attribution"] == "Fixture Artist"
    assert source["performance"] is None and source["content_sha256"] is None
    assert source["metadata_content_sha256"] and source["upstream_media_sha1"]
    second = discovery.discover(tmp_path, private, limit=1)
    assert second["stats"]["requests"] == 0
    assert second["stats"]["cache_hits"] == 3
    assert second["sources"] == first["sources"]
    assert len(calls) == 3
    assert second["stats"]["downloads"] == 0
    assert not second["audio_generation_triggered"]
    assert "PRIVATE" not in (tmp_path / "latest.json").read_text()


def test_request_budget_and_incremental_resume(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(discovery, "_request_json", fixture_transport(calls))
    limits = discovery.DiscoveryLimits(max_requests=1, max_retries=0)
    result = discovery.discover(tmp_path, "young male", limit=1, limits=limits)
    assert result["status"] == "partial" and result["stats"]["requests"] == 1
    for _ in range(2):
        result = discovery.discover(tmp_path, "young male", limit=1, limits=limits)
        assert result["stats"]["requests"] <= 1
    assert result["status"] == "complete" and len(calls) == 3


def test_an_upstream_maxlag_reply_is_retried_instead_of_reported_as_a_missing_source(monkeypatch, tmp_path):
    """``maxlag`` means "busy, come back", and carries HTTP 200 with an error body.

    Reading it as a hard failure would report the official API's own congestion as
    a performer having no usable recording — a wrong negative that quietly shrinks
    the pool, which is the pool size existing to prevent.
    """
    calls = []
    sleeps = []
    lagged = []

    def lagging(url, **kwargs):
        calls.append(url)
        # Only the first Wikidata search is made to lag, so the assertion is about
        # the retry of one request rather than about the whole discovery's volume.
        if len(lagged) == 0:
            lagged.append(url)
            return ({"error": {"code": "maxlag",
                               "info": "Waiting for wdqs1011: 9.5 seconds lagged."}}, "a" * 64)
        return fixture_transport([])(url, **kwargs)

    monkeypatch.setattr(discovery, "_request_json", lagging)
    monkeypatch.setattr(discovery.time, "sleep", sleeps.append)
    result = discovery.discover(tmp_path, limit=1, limits=discovery.DiscoveryLimits(max_retries=1))
    assert len(lagged) == 1, "the lagging reply is a single request that must be retried"
    assert calls.count(lagged[0]) == 2, "the lagging URL must be attempted twice"
    assert sleeps == [0.25], "the retry must back off rather than hammer the API"
    assert result["status"] == "complete", result["errors"]
    assert result["sources"], "the retry must recover the real metadata"

    # An exhausted budget reports the lag as the reason, never as "no recording".
    def always_lagging(url, **kwargs):
        return ({"error": {"code": "maxlag", "info": "Waiting for wdqs1011."}}, "a" * 64)

    monkeypatch.setattr(discovery, "_request_json", always_lagging)
    exhausted = discovery.discover(tmp_path / "second", limit=1,
                                   limits=discovery.DiscoveryLimits(max_retries=0))
    assert exhausted["status"] == "partial"
    assert exhausted["errors"] and "asked for a retry" in exhausted["errors"][0]["reason"]
    assert not exhausted["sources"]


def test_retry_and_durable_cooldown(monkeypatch, tmp_path):
    calls = []
    sleeps = []
    def failure(url, **kwargs):
        calls.append(url)
        raise HTTPError(url, 429, "rate limited", {}, None)
    monkeypatch.setattr(discovery, "_request_json", failure)
    monkeypatch.setattr(discovery.time, "sleep", sleeps.append)
    result = discovery.discover(tmp_path, limit=1)
    assert result["status"] == "partial" and len(calls) == 2 and sleeps == [0.25]
    resumed = discovery.discover(tmp_path, limit=1)
    assert resumed["stats"]["requests"] == 0 and len(calls) == 2


def test_missing_or_ambiguous_entities_held_without_guessing(monkeypatch, tmp_path):
    def ambiguity(url, **kwargs):
        performer = parse_qs(urlsplit(url).query)["search"][0]
        return {"search": [{"id": "Q1", "label": performer}, {"id": "Q2", "label": performer}]}, "f" * 64
    monkeypatch.setattr(discovery, "_request_json", ambiguity)
    result = discovery.discover(tmp_path, limit=1)
    assert result["status"] == "partial" and not result["sources"]
    assert "ambiguous" in result["errors"][0]["reason"]


def test_performer_occupation_disambiguates_same_name_nonperformer(monkeypatch, tmp_path):
    transport = fixture_transport([], p990=False)
    def same_name(url, **kwargs):
        response, sha = transport(url, **kwargs)
        if response.get("search"):
            response["search"][0]["description"] = "British actor"
            response["search"].insert(0, {"id": "Q2", "label": response["search"][0]["label"], "description": "historian"})
        return response, sha
    monkeypatch.setattr(discovery, "_request_json", same_name)
    assert discovery.discover(tmp_path, "young male", limit=1)["status"] == "complete"


def test_commons_top_level_duration_is_preserved(monkeypatch, tmp_path):
    transport = fixture_transport([])
    def top_level_duration(url, **kwargs):
        response, sha = transport(url, **kwargs)
        if "query" in response:
            info = response["query"]["pages"][0]["imageinfo"][0]
            info["duration"] = 8.91646
            del info["metadata"]
        return response, sha
    monkeypatch.setattr(discovery, "_request_json", top_level_duration)
    result = discovery.discover(tmp_path, "warm narrator", limit=1)
    assert result["sources"][0]["duration_seconds"] == 8.91646


def test_cache_lock_rejects_parallel_discovery_and_does_not_grow(tmp_path):
    with discovery._cache_lock(tmp_path):
        with pytest.raises(discovery.DiscoveryHold, match="already running"):
            with discovery._cache_lock(tmp_path):
                pytest.fail("must not acquire competing discovery lock")
    before = (tmp_path / ".discovery.lock").stat().st_size
    for _ in range(5):
        with discovery._cache_lock(tmp_path):
            pass
    assert (tmp_path / ".discovery.lock").stat().st_size == before == 1


def test_storage_and_response_limits_preserve_existing_evidence(tmp_path):
    sentinel = tmp_path / "evidence-existing.json"
    sentinel.write_bytes(b"x" * 800)
    client = discovery.MetadataClient(tmp_path, limits=discovery.DiscoveryLimits(max_storage_bytes=1024))
    with pytest.raises(discovery.DiscoveryHold, match="storage"):
        client.save(tmp_path / "new.json", {"value": "a" * 500})
    assert sentinel.read_bytes() == b"x" * 800
    assert not list(tmp_path.glob("*.tmp"))
    oversized = discovery.MetadataClient(tmp_path / "other", limits=discovery.DiscoveryLimits(max_response_bytes=256),
                                        transport=lambda *a, **kw: ({"data": "x" * 300}, "a" * 64))
    with pytest.raises(discovery.DiscoveryHold, match="byte limit"):
        oversized.get(discovery.WIKIDATA_API, {"action": "x"})


def test_offline_mode_and_fixed_endpoint_do_not_contact_network(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("network must not run")
    client = discovery.MetadataClient(tmp_path, online=False, transport=forbidden)
    with pytest.raises(discovery.DiscoveryHold, match="Offline"):
        client.get(discovery.WIKIDATA_API, {"action": "x"})
    with pytest.raises(discovery.DiscoveryHold, match="allowlist"):
        client.get("http://127.0.0.1/secret", {})
    with pytest.raises(discovery.DiscoveryHold, match="allowlist"):
        discovery._request_json("https://evil.example/", timeout=1, max_bytes=500)


def test_cached_metadata_is_hash_checked(tmp_path):
    client = discovery.MetadataClient(tmp_path, transport=fixture_transport([]))
    params = {"action": "wbsearchentities", "search": "Tom Holland"}
    client.get(discovery.WIKIDATA_API, params)
    pointer = next(tmp_path.glob("request-*.json"))
    cached = json.loads(pointer.read_text())
    cached["payload"]["search"][0]["label"] = "altered"
    pointer.write_text(json.dumps(cached))
    with pytest.raises(discovery.DiscoveryHold, match="integrity"):
        client.get(discovery.WIKIDATA_API, params)


@pytest.mark.parametrize("kwargs", [{"max_requests": 100}, {"max_retries": -1}, {"max_candidates": 3.1},
                                   {"timeout_seconds": float("nan")}, {"max_response_bytes": True}])
def test_limits_cannot_disable_bounds(kwargs):
    with pytest.raises(ValueError):
        discovery.DiscoveryLimits(**kwargs)
