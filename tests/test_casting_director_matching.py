"""Evidence-based analysis and explainable ranking: synthetic, CPU-only, no stores or models."""
from __future__ import annotations

import copy
import json

import pytest

from dubbing_service.app.audiobook import casting_director_matching as matching
from dubbing_service.app.models.character import CharacterDossier, StoryScope
from dubbing_service.app.models.enums import RoleType
from dubbing_service.app.models.project import Chapter, Project, Role, Segment
from tests.test_character_casting import fact


def build(role_facts, *, count=16, narrator=False, spread=None):
    """Roles r0.. alternate lines (adjacent contact); ``role_facts`` maps index -> facts."""
    names = [f"Role{i}" for i in range(len(role_facts))]
    project = Project(id="p", name="Synthetic", chapters=[Chapter(id="ch1", title="One", index=0)],
                      roles=[Role(id=f"r{i}", name=name) for i, name in enumerate(names)])
    for index, role in enumerate(project.roles):
        project.character_cast.dossiers[role.id] = CharacterDossier(
            role_id=role.id, canonical_name=role.name, revision="d1", facts=list(role_facts[index]))
    if narrator:
        project.roles.append(Role(id="nar", name="Narrator", role_type=RoleType.NARRATOR))
    order = spread or [i % len(names) for i in range(count)]
    segments = [Segment(id=f"s{n}", chapter_id="ch1", role_id=f"r{r}", index=n, text="I will be ready.")
                for n, r in enumerate(order)]
    if narrator:
        segments += [Segment(id=f"n{n}", chapter_id="ch1", role_id="nar", index=100 + n, text="He said.")
                     for n in range(3)]
    return project, segments


def analysis_for(project, segments, role_id="r0"):
    return next(a for a in matching.analyze_characters(project, segments) if a["role_id"] == role_id)


def voice(identifier, traits=None, tags=(), **extra):
    return {"candidate_id": identifier, "name": identifier.title(), "origin": "approved_library",
            "library_voice_id": f"lib-{identifier}", "library_version": 1, "reference_conditioned": True,
            "protected": False, "rating": None, "traits": traits or {}, "tags": list(tags),
            "provenance": {"rights_status": "recorded"}, "eligible": True, "holds": [], **extra}


FEMALE_YOUNG = [fact("gender", "female"), fact("age", 17, identifier="age17"),
                fact("voice_description", "a clear, high-pitched voice", identifier="vd")]


# ---------------------------------------------------------------- analysis
def test_analysis_reports_only_evidence_backed_traits_and_open_dimensions():
    project, segments = build([FEMALE_YOUNG, []])
    a = analysis_for(project, segments)
    assert a["role_id"] == "r0" and a["role_name"] == "Role0" and a["role_type"] == "character"
    assert a["traits"]["presentation"] == "feminine"
    assert a["traits"]["pitch"] == "high"
    assert a["trait_detail"]["pitch"]["evidence_ids"] == ["vd"]
    assert a["importance"] == "major" and 0 < a["importance_score"] < 1
    assert 0 < a["confidence"] <= 1
    assert "accent" in a["missing_evidence"]
    assert "accent" not in a["traits"]  # never invented
    json.dumps(a)  # JSON-compatible
    bare = analysis_for(project, segments, "r1")
    assert bare["traits"] == {} and bare["confidence"] == 0.0
    assert set(bare["missing_evidence"]) == set(matching._KEY_DIMENSIONS)


def test_names_and_inferred_gender_do_not_create_traits():
    project, segments = build([[fact("gender", "male", level="INFERRED")]])
    a = analysis_for(project, segments)
    assert "presentation" not in a["traits"] and "gender" not in a["traits"]


def test_relationships_temperament_and_habits_are_evidence_linked():
    project, segments = build([
        [fact("temperament", "shy and cautious", identifier="temp"),
         fact("speech_patterns", "clipped", level="SUPPORTED", identifier="habit"),
         fact("relationships", "mentor of Role1", identifier="rel")],
        [fact("gender", "male")]])
    a = analysis_for(project, segments)
    assert [r["role_id"] for r in a["relationships"]] == ["r1"]
    assert a["relationships"][0]["role_name"] == "Role1" and a["relationships"][0]["needs_contrast"]
    assert a["temperament"][0]["fact_id"] == "temp"
    assert a["habits"][0]["fact_id"] == "habit" and a["habits"][0]["level"] == "SUPPORTED"
    assert a["relationship_facts"][0]["fact_id"] == "rel"
    assert a["traits"]["confidence"] == "cautious"
    assert a["trait_detail"]["confidence"]["evidence_ids"] == ["temp"]


def test_spoiler_and_story_boundary_evidence_is_not_used():
    later = fact("accent", "British", scope=StoryScope(chapter_start=9), identifier="later")
    project, segments = build([[later]])
    project.character_cast.policy.max_chapter = 0
    assert "accent" not in analysis_for(project, segments)["traits"]


def test_narrator_and_unattributed_roles():
    project, segments = build([FEMALE_YOUNG], narrator=True)
    by_id = {a["role_id"]: a for a in matching.analyze_characters(project, segments)}
    assert by_id["nar"]["functions"] == ["narration"] and by_id["nar"]["role_type"] == "narrator"
    assert by_id["nar"]["traits"] == {} and by_id["nar"]["importance"] == "recurring"
    assert any("No character dossier" in r for r in by_id["nar"]["review_reasons"])


def test_changed_evidence_changes_fingerprint_and_ranking():
    project, segments = build([FEMALE_YOUNG])
    first = analysis_for(project, segments)
    deep = voice("deep", {"presentation": "male", "pitch": "low"})
    bright = voice("bright", {"presentation": "female", "pitch": "high", "perceived_vocal_age": "adolescent"})
    ranked = matching.rank_candidates(first, [deep, bright])
    assert [c["candidate_id"] for c in ranked] == ["bright"]  # the male voice contradicts evidence

    project.character_cast.dossiers["r0"].facts = [fact("gender", "male"), fact("age", 70),
                                                   fact("voice_description", "a deep, rough voice")]
    second = analysis_for(project, segments)
    assert second["evidence_fingerprint"] != first["evidence_fingerprint"]
    ranked = matching.rank_candidates(second, [deep, bright])
    assert [c["candidate_id"] for c in ranked][0] == "deep"
    assert ranked[0]["analysis_fingerprint"] == second["evidence_fingerprint"]


# ---------------------------------------------------------------- ranking
def five_varied():
    base = {"presentation": "female", "perceived_vocal_age": "adolescent"}
    return [
        voice("amy", {**base, "pitch": "high", "texture": "clear"}, ["bright"]),
        voice("bea", {**base, "pitch": "mid-high", "texture": "airy"}, ["soft"]),
        voice("cat", {**base, "pitch": "mid", "texture": "rough"}, ["wry"]),
        voice("dot", {**base, "pitch": "high", "texture": "rough"}, ["sharp"]),
        voice("eve", {**base, "pitch": "low", "texture": "clear"}, ["grave"]),
    ]


def test_ranks_three_to_five_with_complete_explanations_and_no_approval():
    project, segments = build([FEMALE_YOUNG])
    a = analysis_for(project, segments)
    ranked = matching.rank_candidates(a, five_varied())
    assert 3 <= len(ranked) <= 5 and [c["rank"] for c in ranked] == list(range(1, len(ranked) + 1))
    assert ranked[0]["candidate_id"] == "amy"
    scores = [c["score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)
    top = ranked[0]
    assert set(top["dimensions"]) == {"fit", "distinctiveness", "fidelity", "clarity", "naturalness",
                                      "stability", "human_rating"}
    assert top["recommendation_only"] is True and top["approved"] is False and top["human_review"] == "pending"
    assert top["needs_casting"] is False and top["viable_count"] == len(ranked)
    assert top["explanation"]["summary"] and "matches pitch: high" in top["explanation"]["strengths"]
    assert top["library_voice_id"] == "lib-amy" and top["origin"] == "approved_library"  # candidate retained
    assert len(matching.rank_candidates(a, five_varied(), limit=3)) == 3
    assert len(matching.rank_candidates(a, five_varied(), limit=99)) <= 5


def test_unmeasured_quality_is_missing_not_fabricated_and_scores_conservatively():
    project, segments = build([FEMALE_YOUNG])
    a = analysis_for(project, segments)
    plain = voice("plain", {"presentation": "female", "pitch": "high"}, ["bright"])
    measured = voice("measured", {"presentation": "female", "pitch": "high"}, ["calm"],
                     quality={"fidelity": 0.9, "clarity": {"value": 0.8, "source": "snr-probe"},
                              "naturalness": True, "stability": 1.7})
    by_id = {c["candidate_id"]: c for c in matching.rank_candidates(a, [plain, measured])}
    p, m = by_id["plain"], by_id["measured"]
    for name in matching.QUALITY_DIMENSIONS:
        assert p["dimensions"][name]["score"] is None and p["dimensions"][name]["status"] == "unmeasured"
    assert p["missing_quality"] == list(matching.QUALITY_DIMENSIONS)
    assert set(p["explanation"]["unmeasured"]) >= set(matching.QUALITY_DIMENSIONS)
    assert m["dimensions"]["fidelity"]["score"] == 0.9 and m["dimensions"]["fidelity"]["status"] == "measured"
    assert m["dimensions"]["clarity"]["basis"].endswith("snr-probe")
    # bool and out-of-range values are not measurements
    assert m["dimensions"]["naturalness"]["score"] is None and m["dimensions"]["stability"]["score"] is None
    assert m["missing_quality"] == ["naturalness", "stability"]
    assert m["score"] > p["score"] and p["confidence"] < m["confidence"]


def test_human_rating_is_recorded_evidence_not_approval():
    project, segments = build([FEMALE_YOUNG])
    a = analysis_for(project, segments)
    rated = matching.rank_candidates(a, [voice("r", {"presentation": "female"}, rating=5)])[0]
    unrated = matching.rank_candidates(a, [voice("u", {"presentation": "female"}, rating=None)])[0]
    assert rated["dimensions"]["human_rating"]["score"] == 1.0 and rated["approved"] is False
    assert unrated["dimensions"]["human_rating"]["score"] is None


def test_cast_similarity_and_scene_contact_penalize_candidates():
    project, segments = build([FEMALE_YOUNG, [fact("gender", "female")]])
    a = analysis_for(project, segments)
    twin = voice("twin", {"presentation": "female", "pitch": "high", "texture": "clear"}, ["bright"])
    fresh = voice("fresh", {"presentation": "female", "pitch": "high", "texture": "airy"}, ["soft"])
    cast = [{"role_id": "r1", "candidate_id": "other", "library_voice_id": "lib-other",
             "traits": {"presentation": "female", "pitch": "high", "texture": "clear"}, "tags": ["bright"]}]
    ranked = {c["candidate_id"]: c for c in matching.rank_candidates(a, [twin, fresh], cast=cast)}
    assert ranked["twin"]["dimensions"]["distinctiveness"]["score"] < ranked["fresh"]["dimensions"]["distinctiveness"]["score"]
    assert ranked["fresh"]["rank"] < ranked["twin"]["rank"]
    assert any("similar" in c for c in ranked["twin"]["explanation"]["concerns"])

    # Without scene contact the same similarity is penalized less.
    far = copy.deepcopy(a)
    far["relationships"] = []
    far_twin = matching.rank_candidates(far, [twin], cast=cast)[0]
    assert far_twin["dimensions"]["distinctiveness"]["score"] > ranked["twin"]["dimensions"]["distinctiveness"]["score"]


def test_identical_voice_for_cooccurring_role_is_excluded_but_noncooccurring_only_penalized():
    project, segments = build([FEMALE_YOUNG, [fact("gender", "female")]])
    a = analysis_for(project, segments)
    cast = [{"role_id": "r1", "library_voice_id": "lib-amy", "traits": {"presentation": "female"}}]
    result = matching.casting_recommendation(a, five_varied(), cast=cast)
    assert "amy" not in [c["candidate_id"] for c in result["ranked"]]
    excluded = {e["candidate_id"]: e for e in result["excluded"]}
    assert excluded["amy"]["category"] == "same_voice_cooccurring"

    lone = copy.deepcopy(a)
    lone["relationships"] = []
    again = matching.casting_recommendation(lone, five_varied(), cast=cast)
    amy = next(c for c in again["ranked"] if c["candidate_id"] == "amy")
    assert amy["dimensions"]["distinctiveness"]["score"] == pytest.approx(0.6)


def test_current_voice_of_the_same_role_is_flagged_not_penalized():
    project, segments = build([FEMALE_YOUNG])
    a = analysis_for(project, segments)
    cast = [{"role_id": "r0", "library_voice_id": "lib-amy", "traits": {"presentation": "female"}}]
    ranked = matching.rank_candidates(a, five_varied(), cast=cast)
    amy = next(c for c in ranked if c["candidate_id"] == "amy")
    assert amy["is_current_voice"] is True and amy["dimensions"]["distinctiveness"]["score"] == 1.0


def test_distinct_variants_duplicates_and_near_duplicates_are_not_separate_auditions():
    project, segments = build([FEMALE_YOUNG])
    a = analysis_for(project, segments)
    base = {"presentation": "female", "pitch": "high", "texture": "clear"}
    candidates = [
        voice("v1", base, ["bright"], library_voice_id="lib-same", library_version=1),
        voice("v2", base, ["bright"], library_voice_id="lib-same", library_version=2),  # same library voice
        voice("clone", base, ["bright"], library_voice_id="lib-clone"),  # identical traits
        voice("other", {"presentation": "female", "pitch": "mid", "texture": "airy"}, ["soft"]),
        voice("third", {"presentation": "female", "pitch": "mid-high", "texture": "rough"}, ["wry"]),
        voice("fourth", {"presentation": "female", "pitch": "high", "texture": "airy"}, ["calm"]),
    ]
    result = matching.casting_recommendation(a, candidates)
    ids = [c["candidate_id"] for c in result["ranked"]]
    assert len(ids) == len(set(ids)) and not ({"v1", "v2", "clone"} <= set(ids))
    assert sum(i in ids for i in ("v1", "v2", "clone")) == 1
    dropped = {e["candidate_id"] for e in result["excluded"] if e["category"] == "near_duplicate"}
    assert dropped == {"v1", "v2", "clone"} - set(ids)
    assert result["needs_casting"] is False


def test_insufficient_distinct_candidates_marks_needs_casting():
    project, segments = build([FEMALE_YOUNG])
    a = analysis_for(project, segments)
    traits = {"presentation": "female", "pitch": "high"}
    candidates = [voice("a", traits, ["x"], library_voice_id="same"), voice("b", traits, ["x"], library_voice_id="same")]
    result = matching.casting_recommendation(a, candidates)
    assert len(result["ranked"]) == 1 and result["needs_casting"] is True
    assert "distinct" in result["needs_casting_reason"]
    assert result["ranked"][0]["needs_casting"] is True
    empty = matching.casting_recommendation(a, [])
    assert empty["ranked"] == [] and empty["needs_casting"] is True and empty["viable_count"] == 0


def test_ineligible_held_featureless_duplicate_and_invalid_candidates_are_excluded():
    project, segments = build([FEMALE_YOUNG])
    a = analysis_for(project, segments)
    good = voice("good", {"presentation": "female"})
    candidates = [
        good, voice("held", {"presentation": "female"}, holds=["rights unresolved"]),
        voice("off", {"presentation": "female"}, eligible=False),
        {k: v for k, v in voice("silent", {"presentation": "female"}).items() if k != "eligible"},
        voice("blank"), dict(good), "not-a-dict",
    ]
    result = matching.casting_recommendation(a, candidates)
    assert [c["candidate_id"] for c in result["ranked"]] == ["good"]
    categories = {e["candidate_id"]: e["category"] for e in result["excluded"] if e["candidate_id"]}
    assert categories == {"held": "held", "off": "ineligible", "silent": "ineligible",
                          "blank": "unassessable", "good": "duplicate"}
    assert any(e["category"] == "invalid" for e in result["excluded"])
    assert "rights unresolved" in next(e for e in result["excluded"] if e["candidate_id"] == "held")["reasons"][0]


def test_missing_rights_evidence_is_flagged_never_assumed():
    project, segments = build([FEMALE_YOUNG])
    a = analysis_for(project, segments)
    c = voice("norights", {"presentation": "female"}, provenance={})
    ranked = matching.rank_candidates(a, [c])[0]
    assert any("rights status not recorded" in x for x in ranked["explanation"]["concerns"])
    assert ranked["provenance"] == {}  # untouched


def test_no_evidence_means_fit_is_unmeasured_not_defaulted():
    project, segments = build([[], FEMALE_YOUNG])
    a = analysis_for(project, segments, "r0")
    ranked = matching.rank_candidates(a, five_varied())
    assert ranked and all(c["dimensions"]["fit"]["score"] is None for c in ranked)
    assert ranked[0]["dimensions"]["fit"]["status"] == "unmeasured"
    assert ranked[0]["confidence"] < 0.5


def test_unstated_traits_count_as_partial_miss_and_conflicts_disqualify():
    project, segments = build([FEMALE_YOUNG])
    a = analysis_for(project, segments)
    wide = voice("wide", {"presentation": "female", "pitch": "high", "texture": "clear",
                          "perceived_vocal_age": "adolescent"})
    thin = voice("thin", {"presentation": "female"})
    clash = voice("clash", {"presentation": "male", "pitch": "high"})
    old = voice("old", {"presentation": "female", "perceived_vocal_age": "older adult"})
    result = matching.casting_recommendation(a, [wide, thin, clash, old])
    by_id = {c["candidate_id"]: c for c in result["ranked"]}
    assert by_id["wide"]["dimensions"]["fit"]["score"] > by_id["thin"]["dimensions"]["fit"]["score"]
    assert {e["candidate_id"]: e["category"] for e in result["excluded"]} == {"clash": "conflict", "old": "conflict"}


def test_accent_is_matched_only_from_evidence_and_candidate_statements():
    project, segments = build([[fact("accent", "British")]])
    a = analysis_for(project, segments)
    uk = voice("uk", {"accent": "British (neutral RP)"})
    us = voice("us", {"accent": "American (General American)"})
    silent = voice("silent", {"pitch": "mid"})
    result = matching.casting_recommendation(a, [uk, us, silent])
    assert [c["candidate_id"] for c in result["ranked"]][0] == "uk"
    assert {e["candidate_id"] for e in result["excluded"]} == {"us"}
    open_ = next(c for c in result["ranked"] if c["candidate_id"] == "silent")
    assert open_["dimensions"]["fit"]["score"] is None  # no stated accent -> unknown, not a pass


def test_ranking_is_deterministic_pure_and_does_not_mutate_inputs():
    project, segments = build([FEMALE_YOUNG, [fact("gender", "male")]])
    a = analysis_for(project, segments)
    candidates, cast = five_varied(), [{"role_id": "r1", "traits": {"presentation": "male"}}]
    before = copy.deepcopy((a, candidates, cast))
    first = matching.rank_candidates(a, candidates, cast=cast)
    second = matching.rank_candidates(a, list(reversed(candidates)), cast=cast)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert (a, candidates, cast) == before
    first[0]["traits"]["pitch"] = "changed"
    assert candidates[0]["traits"]["pitch"] == "high"


def test_rank_requires_a_single_role_analysis():
    with pytest.raises(ValueError):
        matching.rank_candidates({}, [])
    project, segments = build([FEMALE_YOUNG, []])
    with pytest.raises(ValueError):
        matching.rank_candidates(matching.analyze_characters(project, segments), [])
    one = matching.analyze_characters(project, segments)[:1]
    assert matching.rank_candidates(one, five_varied())
