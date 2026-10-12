"""Casting-director candidate discovery: isolated stores, no GPU, no network, no writes."""
from __future__ import annotations

import copy
import json
import socket
from datetime import datetime, timezone

import pytest

from dubbing_service.app.audiobook import casting_director_discovery as discovery
from dubbing_service.app.audiobook import preferred_voices
from dubbing_service.app.audiobook.voice_lab import digest
from dubbing_service.app.models.character import CastingDecision, CastingProfile
from dubbing_service.app.models.enums import RoleType
from dubbing_service.app.models.project import Role, Segment
from dubbing_service.app.storage import voices as library
from dubbing_service.app.storage.projects import create_project
from tests.audiobook_helpers import use_projects_root

SHA = {name: (name * 64)[:64] for name in "abcdef0123456789"}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Every store is a temp path before anything runs, and the network is unreachable."""
    use_projects_root(monkeypatch, tmp_path)
    voices_root = tmp_path / "library" / "voices"
    voices_root.mkdir(parents=True)
    monkeypatch.setattr(library, "voices_audio_dir", lambda: voices_root)
    monkeypatch.setattr(library, "_voices_file_path", lambda: voices_root.parent / "voices.json")

    def refuse(*args, **kwargs):
        raise AssertionError("discovery must never touch the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    return voices_root.parent / "voices.json"


def _identity(voice_id, *, mechanism="qwen_reference", provenance=None, **extra):
    row = {"id": voice_id, "role_id": "r", "status": "LOCKED", "mechanism": mechanism,
           "dossier_revision": "d", "casting_profile_revision": "p", "design_instruction": "An original warm voice",
           "backend": "qwen3tts", "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "provenance": provenance or {}}
    if mechanism == "qwen_reference":
        row.update(reference_audio="ref.wav", reference_audio_sha256=SHA["1"], prompt_artifact="p.npz",
                   prompt_sha256=SHA["2"])
    if mechanism == "builtin":
        row.update(builtin_speaker="ryan", backend="qwen3tts_cv")
    row.update(extra)
    return row


def _clone_provenance(source_id="cstr-vctk-p226", *, evidenced=True):
    provenance = {"origin": "eligible_reference_clone", "actor_audio_used": True,
                  "acquisition_sha256": digest(source_id),
                  "source": {"source_id": source_id, "performer": "p226", "license": "CC BY 4.0",
                             "license_url": "https://creativecommons.org/licenses/by/4.0/",
                             "source_page": "https://datashare.example.org/handle/1", "attribution": "CSTR"}}
    if evidenced:
        provenance["eligibility"] = {"acquisition_eligible": True, "intended_use_evidence": {
            "status": "approved", "authority": "voice_license", "evidence_ref": "publisher TTS purpose",
            "scope": "private audition", "performer_consent_claimed": False}}
    return provenance


def publish(voice_id, name, *, kind="reference_clone", identity=None, builtin=False, audio="a", version="v1"):
    """Write one immutable library version straight into the isolated library."""
    key = "v_" + digest([voice_id, version])[:28]
    record = {"voice_id": voice_id, "version": version, "key": key, "name": name, "kind": kind, "tags": ["fixture"],
              "inspiration": None, "backend": "qwen3tts", "audio_file": "sample.wav", "audio_sha256": SHA[audio],
              "duration_seconds": 3.0, "identity": identity or {}}
    assets = {"sample": {"path": "voice_lab/x/sample.wav", "sha256": SHA[audio], "bytes": 1}}
    if builtin:
        record.update(builtin_speaker="ryan", backend="qwen3tts_cv")
    if identity and identity.get("mechanism") == "qwen_reference":
        assets["reference_audio"] = {"path": "voice_lab/x/ref.wav", "sha256": SHA["1"], "bytes": 1}
        assets["prompt_artifact"] = {"path": "voice_lab/x/p.npz", "sha256": SHA["2"], "bytes": 1}
    entry = {"schema_version": 1, "record": record, "assets": assets, "source_project_id": "src",
             "published_at": "2026-10-01T00:00:00+00:00"}
    entry["manifest_sha256"] = digest(entry)
    library_id = "vl_" + digest(voice_id)[:24]
    with library.voices_lock():
        voices = library.load_voices()
        stamp = datetime.now(timezone.utc)
        voice = next((v for v in voices if v.id == library_id), None)
        if voice is None:
            voice = library.Voice(id=library_id, name=name, created_at=stamp, updated_at=stamp)
            voices.append(voice)
        voice.voice_lab_versions[version] = entry
        library.save_voices(voices)
    return library_id, version, record


def rate(library_id, version, record, stars, *, role="r1", event="e1", traits=None):
    return preferred_voices.record_rating(
        library_voice_id=library_id, version=version, rating=stars, project_id="src", role_id=role,
        character="Fixture", key=record["key"], audio_sha256=record["audio_sha256"], evidence_id=event, traits=traits)


def clone(voice_id, name, *, audio="a", stars=None, evidenced=True, traits=None, source_id=None):
    lib, version, record = publish(
        voice_id, name, audio=audio,
        identity=_identity(voice_id, provenance=_clone_provenance(source_id or "cstr-" + voice_id, evidenced=evidenced)))
    if stars:
        rate(lib, version, record, stars, event="ev-" + voice_id, traits=traits)
    return lib, version, record


def make_project(*, female_role=True):
    project = create_project("Discovery fixture")
    project.roles = [Role(id="narrator", name="Narrator", role_type=RoleType.NARRATOR),
                     Role(id="hermione", name="Hermione", role_type=RoleType.CHARACTER)]
    if female_role:
        project.character_cast.profiles["hermione"] = CastingProfile(
            role_id="hermione", revision="r", dossier_revision="d", design_instruction="Create an original voice.",
            decisions=[CastingDecision(dimension="presentation", value="feminine", reason="gender evidence",
                                       evidence_ids=["f1"], level="EXPLICIT")])
    segments = [Segment(id="s1", index=0, role_id="narrator", text="It was a bright day."),
                Segment(id="s2", index=1, role_id="hermione", text="Hello there!")]
    return project, segments


def by_name(result):
    return {c["name"]: c for c in result["candidates"]}


# ------------------------------------------------------------------------------ order and reuse

def test_approved_library_leads_and_clone_beats_design(isolated):
    project, segments = make_project()
    clone("approved-voice", "Approved clone", audio="a", stars=5)
    clone("generic-voice", "Generic clone", audio="b", stars=None)
    result = discovery.discover_candidates(project, segments)
    names = by_name(result)
    assert names["Approved clone"]["origin"] == "approved_library"
    assert names["Generic clone"]["origin"] == "generic_library"
    order = [c["name"] for c in result["candidates"]]
    assert order.index("Approved clone") < order.index("Generic clone")
    # Reference-conditioned voices exist for every role, so no design direction is offered.
    assert not any(c["origin"] == "model_design" for c in result["candidates"])
    assert names["Approved clone"]["reference_conditioned"] is True
    assert names["Approved clone"]["verification"]["approval_binding_verified"] is True
    assert names["Generic clone"]["verification"]["approval_binding_verified"] is False
    assert all(c["recommendation_only"] and c["approved"] is False for c in result["candidates"])
    assert result["recommendations_are_approvals"] is False and result["human_review"] == "pending"
    for shortlist in result["shortlists"].values():
        assert shortlist[0]["origin"] == "approved_library"


def test_library_reuse_exposes_exact_immutable_identity_without_local_paths():
    project, segments = make_project()
    lib, version, record = clone("reuse-voice", "Reusable", stars=4)
    candidate = by_name(discovery.discover_candidates(project, segments))["Reusable"]
    assert (candidate["library_voice_id"], candidate["library_version"], candidate["key"]) == (lib, version, record["key"])
    assert candidate["sample"]["audio_sha256"] == record["audio_sha256"]
    assert candidate["conditioning"]["prompt_sha256"] == SHA["2"]
    assert candidate["prior_use"]["projects"] == ["src"]
    text = json.dumps(candidate)
    assert "ref.wav" not in text and "p.npz" not in text and "voices.json" not in text
    json.dumps(discovery.discover_candidates(project, segments))  # JSON-compatible


def test_design_directions_only_fill_roles_without_a_viable_clone():
    project, segments = make_project()
    clone("masc-voice", "Masculine clone", stars=5, traits={"presentation": "masculine"})
    result = discovery.discover_candidates(project, segments)
    designs = [c for c in result["candidates"] if c["origin"] == "model_design"]
    assert designs, "the feminine role has no viable clone, so original directions are offered"
    assert {c["designed_for_role_id"] for c in designs} == {"hermione"}
    assert len({c["candidate_id"] for c in designs}) == len(designs)
    assert 3 <= len(designs) <= 5
    assert all("accent" not in c["traits"] for c in designs)
    for design in designs:
        assert design["generated_reference"] is False and design["reference_conditioned"] is False
        assert design["provenance"]["actor_audio_used"] is False and design["provenance"]["consent_evidence"] is None
        assert design["design"]["design_instruction"] and "explicit_audition_request" in design["requires"]
        assert design["traits"]["presentation"] == "feminine"
    # A direction is a labelled recommendation, so the clone still sorts before it.
    order = [c["origin"] for c in result["candidates"]]
    assert order.index("approved_library") < order.index("model_design")
    gap_codes = {(g["role_id"], g["code"]) for g in result["coverage_gaps"]}
    assert ("hermione", "no_reference_conditioned_candidate") in gap_codes


# ------------------------------------------------------------------------------ protection and ratings

def test_four_and_five_star_voices_are_protected_and_low_stars_flagged_and_held():
    project, segments = make_project()
    clone("five", "Five star", audio="a", stars=5)
    clone("four", "Four star", audio="b", stars=4)
    clone("three", "Three star", audio="c", stars=3)
    clone("one", "One star", audio="d", stars=1)
    result = discovery.discover_candidates(project, segments)
    names = by_name(result)
    assert names["Five star"]["protected"] and names["Four star"]["protected"]
    assert names["Five star"]["rating"] == 5 and names["Four star"]["rating"] == 4
    for name, stars in (("Three star", 3), ("One star", 1)):
        row = names[name]
        assert row["rating"] == stars and row["low_rating_flag"] and not row["protected"]
        assert not row["eligible"] and "rated_below_reuse_threshold" in {h["code"] for h in row["holds"]}
        assert row["origin"] == "generic_library"
    listed = {item["name"] for shortlist in result["shortlists"].values() for item in shortlist}
    assert {"Five star", "Four star"} <= listed and not ({"Three star", "One star"} & listed)
    assert result["counts"]["protected"] == 2 and result["counts"]["flagged_low_rating"] == 2


def test_later_low_rating_does_not_remove_protection():
    project, segments = make_project()
    lib, version, record = clone("history", "Rated twice", stars=5)
    rate(lib, version, record, 2, event="ev-later")
    row = by_name(discovery.discover_candidates(project, segments))["Rated twice"]
    assert row["rating"] == 5 and row["protected"] and row["eligible"]
    assert any("lower recorded rating" in c for c in row["cautions"])


def test_protected_candidates_survive_the_limit():
    project, segments = make_project()
    clone("keep", "Protected keeper", audio="a", stars=5)
    for index, (name, audio) in enumerate((("Generic one", "b"), ("Generic two", "c"), ("Generic three", "d"))):
        clone(f"generic-{index}", name, audio=audio)
    result = discovery.discover_candidates(project, segments, limit=1)
    assert "Protected keeper" in by_name(result)
    assert result["truncated"] == 3 and len(result["candidates"]) == 1
    selected = discovery.discover_candidates(project, segments, limit=1)
    assert len(selected["candidates"]) == 1


def test_weak_generic_presets_are_never_a_fallback_but_a_rated_one_is_protected():
    project, segments = make_project(female_role=False)
    publish("preset-unrated", "Unrated preset", kind="provider_builtin", builtin=True,
            identity=_identity("preset-unrated", mechanism="builtin"), audio="a")
    lib, version, record = publish("preset-loved", "Loved preset", kind="provider_builtin", builtin=True,
                                   identity=_identity("preset-loved", mechanism="builtin"), audio="b")
    rate(lib, version, record, 5, event="ev-preset")
    result = discovery.discover_candidates(project, segments)
    names = by_name(result)
    assert "Unrated preset" not in names
    assert any(row["code"] == "weak_generic_preset" for row in result["excluded"])
    assert names["Loved preset"]["protected"] and names["Loved preset"]["reference_conditioned"] is False
    assert not any(item["name"] == "Unrated preset" for s in result["shortlists"].values() for item in s)


# ------------------------------------------------------------------------------ rights

def external_source(source_id="wiki-ext-1", **overrides):
    source = {"source_id": source_id, "performer": "Example Reader", "language": "en", "duration_seconds": 42.0,
              "attribution": "Uploader, CC BY", "license": "CC BY 4.0",
              "license_url": "https://creativecommons.org/licenses/by/4.0/",
              "source_page": "https://commons.wikimedia.org/wiki/File:Example.ogg",
              "terms_url": "https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia",
              "media_url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Example.ogg"}
    source.update(overrides)
    return source


def test_external_recording_license_is_not_voice_consent():
    project, segments = make_project(female_role=False)
    result = discovery.discover_candidates(project, segments, external_sources=[external_source()])
    row = by_name(result)["Example Reader"]
    assert row["origin"] == "authorized_external" and not row["eligible"]
    assert row["provenance"]["download_eligible"] is True and row["provenance"]["intended_use_eligible"] is False
    assert row["provenance"]["rights_status"] == "held"
    assert row["provenance"]["consent_evidence"] is None and row["provenance"]["performer_consent_claimed"] is None
    assert any("source-bound evidence" in hold["detail"] for hold in row["holds"])
    assert row["reference_conditioned"] is False and row["generated_reference"] is False
    assert "https://commons.wikimedia.org/wiki/File:Example.ogg" in row["provenance"]["source_urls"]
    assert not any(item["candidate_id"] == row["candidate_id"] for s in result["shortlists"].values() for item in s)


def test_external_source_with_owner_evidence_is_eligible_for_acquisition_but_never_cloned_or_approved():
    project, segments = make_project(female_role=False)
    evidence = {"status": "approved", "use": "private_synthetic_voice", "authority": "voice_license",
                "evidence_ref": "owner decision 1", "source_id": "wiki-ext-2", "performer_consent_claimed": False}
    result = discovery.discover_candidates(
        project, segments, external_sources=[external_source("wiki-ext-2", intended_use_evidence=evidence)])
    row = by_name(result)["Example Reader"]
    assert row["eligible"] and row["holds"] == []
    assert row["provenance"]["rights_status"] == "acquisition_authorized_pending_acquisition"
    assert row["provenance"]["performer_consent_claimed"] is False  # as supplied, never upgraded
    assert row["reference_conditioned"] is False and not row["approved"]
    assert "source_download" in row["requires"] and "human_audition" in row["requires"]


def test_external_source_with_foreign_evidence_or_bad_metadata_stays_held():
    project, segments = make_project(female_role=False)
    other = {"status": "approved", "use": "private_synthetic_voice", "authority": "voice_license",
             "evidence_ref": "elsewhere", "source_id": "someone-else"}
    sources = [external_source("wiki-ext-3", intended_use_evidence=other),
               external_source("wiki-ext-4", license_url="https://example.com/all-rights-reserved", performer="Closed"),
               "not a mapping", {"performer": "No identity"}]
    result = discovery.discover_candidates(project, segments, external_sources=sources)
    names = by_name(result)
    assert not names["Example Reader"]["eligible"] and not names["Closed"]["eligible"]
    assert any("license" in h["detail"].lower() for h in names["Closed"]["holds"])
    assert sum(h["code"] == "external_source_unidentifiable" for h in result["holds"]) == 2


def test_unresolved_provenance_is_retained_only_for_human_approval_without_asserting_rights():
    project, segments = make_project(female_role=False)
    clone("old-selected", "Rated historical", audio="a", stars=5, evidenced=False)
    clone("old-unrated", "Unrated historical", audio="b", evidenced=False)
    clone("evidenced", "Evidenced clone", audio="c", evidenced=True)
    names = by_name(discovery.discover_candidates(project, segments))
    kept, dropped, good = names["Rated historical"], names["Unrated historical"], names["Evidenced clone"]
    assert kept["eligible"] and kept["provenance"]["rights_status"] == "historical_provenance_unresolved"
    assert kept["provenance"]["fresh_acquisition_rights_asserted"] is False
    assert any("no fresh acquisition" in c for c in kept["cautions"])
    assert not dropped["eligible"] and "provenance_unresolved" in {h["code"] for h in dropped["holds"]}
    assert good["provenance"]["rights_status"] == "recorded_source_evidenced"
    assert good["provenance"]["consent_evidence"]["performer_consent_claimed"] is False
    assert good["provenance"]["fresh_acquisition_rights_asserted"] is False


def test_contradictory_provenance_is_held():
    project, segments = make_project(female_role=False)
    bad = _clone_provenance()
    bad["actor_audio_used"] = False
    publish("contradiction", "Contradictory", identity=_identity("contradiction", provenance=bad))
    row = by_name(discovery.discover_candidates(project, segments))["Contradictory"]
    assert not row["eligible"] and "provenance_contradiction" in {h["code"] for h in row["holds"]}


# ------------------------------------------------------------------------------ project voices, dedup, gates

def test_human_selected_project_voice_is_retained_and_protected_with_its_role():
    project, segments = make_project(female_role=False)
    key = "v_projectonly"
    project.voice_lab.voices[key] = {
        "key": key, "voice_id": "voice_projectonly", "version": "pv1", "name": "Selected recorded draft",
        "kind": "reference_clone", "tags": [], "audio_sha256": SHA["e"], "duration_seconds": 5.0,
        "identity": _identity("voice_projectonly", provenance=_clone_provenance("cstr-projectonly", evidenced=False))}
    project.voice_lab.casting["events"]["ev1"] = {"origin": "human", "actor": "brian", "key": key, "action": "use"}
    project.voice_lab.casting["roles"]["narrator"] = {"assignment": {"key": key, "approved": True, "event_id": "ev1"}}
    project.voice_lab.voices["v_leftover"] = {**project.voice_lab.voices[key], "key": "v_leftover", "name": "Leftover",
                                              "audio_sha256": SHA["f"], "identity": _identity("voice_left")}
    result = discovery.discover_candidates(project, segments)
    names = by_name(result)
    row = names["Selected recorded draft"]
    assert row["eligible"] and row["protected"] and row["protection_reasons"] == ["human_selected_assignment"]
    assert row["selected_for_roles"] == ["narrator"] and row["source_tier"] == "project_voice_unpublished"
    assert row["provenance"]["fresh_acquisition_rights_asserted"] is False
    assert "Leftover" not in names, "an unrated, unselected audition leftover is not a candidate"


def test_same_voice_through_several_sources_is_deduplicated_to_the_library_row():
    project, segments = make_project(female_role=False)
    lib, version, record = clone("dup", "Library original", stars=5, source_id="cstr-dup")
    # The project already imported this exact version, and the same recording appears as external metadata.
    project.voice_lab.voices[record["key"]] = {**copy.deepcopy(record), "library_voice_id": lib, "library_version": version}
    project.voice_lab.casting["events"]["ev"] = {"origin": "human", "actor": "brian", "key": record["key"]}
    project.voice_lab.casting["roles"]["narrator"] = {"assignment": {"key": record["key"], "approved": True, "event_id": "ev"}}
    result = discovery.discover_candidates(project, segments, external_sources=[external_source("cstr-dup")])
    matches = [c for c in result["candidates"] if c["name"] in {"Library original", "Example Reader"}]
    assert len(matches) == 1 and matches[0]["origin"] == "approved_library"
    assert matches[0]["selected_for_roles"] == ["narrator"]
    assert matches[0]["duplicates"] and matches[0]["duplicates"][0]["origin"] == "authorized_external"
    assert {"rated_4_5_stars", "human_selected_assignment"} <= set(matches[0]["protection_reasons"])


def test_identity_gate_excludes_a_voice_whose_presentation_contradicts_the_role():
    project, segments = make_project()
    clone("fem", "Feminine clone", audio="a", stars=5, traits={"presentation": "feminine"})
    clone("masc", "Masculine clone", audio="b", stars=5, traits={"presentation": "masculine"})
    result = discovery.discover_candidates(project, segments)
    hermione = [item["name"] for item in result["shortlists"]["hermione"]]
    assert "Feminine clone" in hermione and "Masculine clone" not in hermione
    narrator = {item["name"]: item for item in result["shortlists"]["narrator"]}
    assert narrator["Masculine clone"]["fit"] == "unscored_no_book_traits"
    assert hermione[0] == "Feminine clone"
    first = result["shortlists"]["hermione"][0]
    assert first["fit"] == "gates_passed" and first["matched_dimensions"] >= 1
    assert not any(g["role_id"] == "hermione" and g["code"] == "no_reference_conditioned_candidate"
                   for g in result["coverage_gaps"])
    assert any(g["role_id"] == "narrator" and g["code"] == "book_trait_evidence_missing" for g in result["coverage_gaps"])


def test_shortlist_is_bounded_and_coverage_gaps_report_empty_libraries():
    project, segments = make_project()
    for index in range(4):
        clone(f"many-{index}", f"Voice {index}", audio="abcd"[index], stars=5)
    result = discovery.discover_candidates(project, segments, shortlist_size=2, design_per_role=0)
    assert all(len(rows) == 2 for rows in result["shortlists"].values())


def test_empty_library_reports_coverage_gaps_instead_of_inventing_voices():
    empty = discovery.discover_candidates(*make_project(), design_per_role=0)
    assert empty["candidates"] == []
    assert all(rows == [] for rows in empty["shortlists"].values())
    assert {g["code"] for g in empty["coverage_gaps"]} >= {"no_viable_candidate", "no_approved_library_candidate"}


# ------------------------------------------------------------------------------ integrity and purity

def test_invalid_library_version_is_reported_not_trusted_and_good_ones_survive():
    project, segments = make_project(female_role=False)
    lib, version, _ = clone("good", "Good clone", stars=5)
    broken_lib, broken_version, _ = clone("broken", "Broken clone", audio="b", stars=5)
    with library.voices_lock():
        voices = library.load_voices()
        target = next(v for v in voices if v.id == broken_lib)
        target.voice_lab_versions[broken_version]["record"]["name"] = "Tampered after manifest"
        library.save_voices(voices)
    result = discovery.discover_candidates(project, segments)
    assert "Good clone" in by_name(result) and "Tampered after manifest" not in by_name(result)
    hold = next(h for h in result["holds"] if h["code"] == "library_version_invalid")
    assert hold["library_voice_id"] == broken_lib and hold["protected"] is True


def test_discovery_writes_nothing_and_is_deterministic(isolated):
    project, segments = make_project()
    clone("det", "Deterministic", stars=5)
    before = isolated.read_bytes()
    first = discovery.discover_candidates(project, segments, external_sources=[external_source()])
    second = discovery.discover_candidates(project, segments, external_sources=[external_source()])
    assert isolated.read_bytes() == before
    assert first == second
    assert first["side_effects"] == {"downloads": 0, "inference": 0, "writes": 0, "approvals": 0}
    assert json.loads(json.dumps(first)) == first
