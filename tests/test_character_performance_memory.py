"""CPU tests for the book-scoped character performance memory seam.

Everything here is a labeled synthetic fixture: invented prose, invented role ids and a hand-written stand-in
for a model response.  No real book, voice, model, GPU, network or live project is read or generated, and
nothing here is evidence about audio quality, production readiness or human approval.
"""
from __future__ import annotations

import copy
import json

import pytest

from dubbing_service.app.audiobook import character_performance_memory as cpm
from dubbing_service.app.audiobook import scene_coaching
from dubbing_service.app.audiobook.scene_dialogue_plan import _sha

BOOK = "b" * 64
OTHER_BOOK = "c" * 64
FIXTURE_PROVENANCE = {"origin": "model", "model_requested": "synthetic-test-fixture-not-a-real-model",
                      "receipt_sha256": "a1" * 32, "fixture": True}

# (utterance suffix, role id, role name, narrator, text)
CH_A = [("00", "narrator", "Narrator", True, "The gate was shut and the road behind them was dark."),
        ("01", "erick", "Erick", False, "“Open it,”"),
        ("02", "narrator", "Narrator", True, "said Erick."),
        ("03", "jane", "Jane", False, "“I will not,” she said."),
        ("04", "narrator", "Narrator", True, "Jane folded her arms.")]
CH_B = [("00", "narrator", "Narrator", True, "Morning came slowly over the wall."),
        ("01", "erick", "Erick", False, "“Now?”"),
        ("02", "jane", "Jane", False, "“We walk together,”")]
# A fresh chapter of the same book: same stable role ids, a different project.
CH_FRESH = [("00", "narrator", "Narrator", True, "A new road began beyond the gate."),
            ("01", "jane", "Jane", False, "“We walk,”"),
            ("02", "erick", "Erick", False, "“Together,”")]


def chapter_request(project_id, rows, *, extra_constraints=()):
    context = [{"utterance_id": f"{project_id}-{suffix}", "text": text, "role_id": rid, "narrator": narrator,
                "role_name": name} for suffix, rid, name, narrator, text in rows]
    selected = [c["utterance_id"] for c in context if not c["narrator"]]
    return {"instruction": "Fixture coaching instruction.", "constraints": ["Fixture constraint one.", *extra_constraints],
            "response_contract": {"schema_version": scene_coaching.RESPONSE_SCHEMA, "scenes": "fixture"},
            "scenes": [{"scene_id": f"{project_id}-scene-1", "project_id": project_id, "context": context,
                        "selected_dialogue_ids": selected}]}


@pytest.fixture
def chapters():
    return [chapter_request("fix-a", CH_A), chapter_request("fix-b", CH_B)]


def row_of(request, role_id):
    for source in request["sources"]:
        for row in source["context"]:
            if row["role_id"] == role_id and not row["narrator"]:
                return source["project_id"], row
    raise AssertionError(role_id)


def span(request, role_id, needle=None):
    project_id, row = row_of(request, role_id)
    text = row["text"]
    start = text.index(needle) if needle else 0
    end = start + len(needle) if needle else len(text)
    return {"project_id": project_id, "utterance_id": row["utterance_id"], "start": start, "end": end,
            "text": text[start:end]}


def make_response(request, *, mutate=None):
    """A hand-written stand-in for a captured model response; each field is derived from the request."""
    habits = {"erick": ["Short, firm phrases with clean word endings.", "Leans on the final word of a demand."],
              "jane": ["Steady, measured phrasing that keeps endings open.", "Warmer and slower to commit than the other speaker."]}
    contrast = {"erick": "Blunt and brief next to the other speaker's steadier replies.",
                "jane": "Calmer and fuller than the other speaker's clipped demands."}
    profiles = []
    for role in request["roles"]:
        rid = role["role_id"]
        profiles.append({"role_id": rid, "role_name": role["role_name"], "delivery_habits": list(habits[rid]),
                         "contrast_notes": contrast[rid], "confidence": "moderate", "evidence": [span(request, rid)]})
    response = {"schema_version": cpm.RESPONSE_SCHEMA, "profiles": profiles}
    if mutate:
        mutate(response)
    return response


def make_memory(chapters, *, book=BOOK, mutate=None, provenance=None):
    request = cpm.build_memory_request(chapters, book_sha256=book)
    return cpm.capture_bundle(request, make_response(request, mutate=mutate), provenance or FIXTURE_PROVENANCE)


def rehash(bundle):
    """Re-bind every hash after a deliberate edit, so only the semantic check can refuse it."""
    bundle["request_sha256"], bundle["response_sha256"] = _sha(bundle["request"]), _sha(bundle["response"])
    bundle["memory_sha256"] = cpm._memory_sha(bundle)
    return bundle


# --------------------------------------------------------------------------- request building


def test_request_is_canonical_voice_independent_and_excludes_narrators(chapters):
    request = cpm.build_memory_request(chapters, book_sha256=BOOK)
    assert request["schema_version"] == "character-performance-memory-request-v1"
    assert request["scope"] == "book" and request["voice_independent"] is True
    assert request["interpretation_status"] == "unapproved_model_interpretation"
    assert request["roles"] == [{"role_id": "erick", "role_name": "Erick"}, {"role_id": "jane", "role_name": "Jane"}]
    assert request["narrator_role_ids"] == ["narrator"]
    assert [s["project_id"] for s in request["sources"]] == ["fix-a", "fix-b"]
    assert cpm.build_memory_request(copy.deepcopy(chapters), book_sha256=BOOK) == request
    assert cpm.build_memory_request({"fix-a": chapters[0], "fix-b": chapters[1]}, book_sha256=BOOK) == request
    assert "voice" not in json.dumps(sorted(request)).replace("voice_independent", "")  # no voice-id fields


def test_building_never_mutates_inputs_and_shares_no_objects(chapters):
    before = copy.deepcopy(chapters)
    request = cpm.build_memory_request(chapters, book_sha256=BOOK)
    assert chapters == before
    request["sources"][0]["context"][0]["text"] = "changed"
    request["roles"].clear()
    assert chapters == before


def test_only_known_context_fields_are_copied(chapters):
    chapters[0]["scenes"][0]["context"][1]["voice_id"] = "voice_secret_clone"
    request = cpm.build_memory_request(chapters, book_sha256=BOOK)
    assert "voice_secret_clone" not in json.dumps(request)


@pytest.mark.parametrize("book", ["", "B" * 64, "b" * 63, None, 7])
def test_book_hash_must_be_a_lowercase_digest(chapters, book):
    with pytest.raises(cpm.PerformanceMemoryError, match="sha256"):
        cpm.build_memory_request(chapters, book_sha256=book)


def edit(fn):
    def apply(chapters):
        fn(chapters)
        return chapters
    return apply


REQUEST_REFUSALS = {
    "narrator-selected": (edit(lambda c: c[0]["scenes"][0]["selected_dialogue_ids"].append("fix-a-00")), "narration"),
    "unknown-selected": (edit(lambda c: c[0]["scenes"][0]["selected_dialogue_ids"].append("nope")), "unknown dialogue"),
    "duplicate-scene": (edit(lambda c: c.append(copy.deepcopy(c[0]))), "supplied twice"),
    "duplicate-utterance": (edit(lambda c: c[0]["scenes"][0]["context"].append(copy.deepcopy(c[0]["scenes"][0]["context"][0]))),
                            "repeats utterance"),
    "one-id-two-names": (edit(lambda c: c[1]["scenes"][0]["context"][1].update(role_name="Eric")), "more than one name"),
    "one-name-two-ids": (edit(lambda c: c[1]["scenes"][0]["context"][1].update(role_id="erick2")), "more than one role id"),
    "narrator-and-character": (edit(lambda c: c[1]["scenes"][0]["context"][1].update(role_id="narrator", role_name="Narrator")),
                               "ambiguous"),
    "nameless-character": (edit(lambda c: c[0]["scenes"][0]["context"][1].update(role_name=None)), "role_name"),
    "narrator-flag-not-bool": (edit(lambda c: c[0]["scenes"][0]["context"][1].update(narrator=0)), "incomplete context row"),
    "empty-text": (edit(lambda c: c[0]["scenes"][0]["context"][1].update(text="")), "incomplete context row"),
    "bad-project-id": (edit(lambda c: c[0]["scenes"][0].update(project_id="bad id")), "project_id"),
    "no-scenes": (edit(lambda c: c[0].update(scenes=[])), "needs scenes"),
}


@pytest.mark.parametrize("name", sorted(REQUEST_REFUSALS))
def test_malformed_or_ambiguous_sources_are_refused(chapters, name):
    mutate, message = REQUEST_REFUSALS[name]
    with pytest.raises(cpm.PerformanceMemoryError, match=message):
        cpm.build_memory_request(mutate(chapters), book_sha256=BOOK)


def test_conflicting_duplicate_rows_across_scenes_are_refused(chapters):
    other = copy.deepcopy(chapters[0])
    other["scenes"][0]["scene_id"] = "fix-a-scene-2"
    other["scenes"][0]["context"][1]["text"] = "“Open it now,”"
    with pytest.raises(cpm.PerformanceMemoryError, match="differs between scenes"):
        cpm.build_memory_request([chapters[0], other], book_sha256=BOOK)


def test_request_with_only_narration_selected_has_nothing_to_profile():
    request = chapter_request("fix-n", [("00", "narrator", "Narrator", True, "Only narration here.")])
    request["scenes"][0]["selected_dialogue_ids"] = []
    with pytest.raises(cpm.PerformanceMemoryError, match="selected dialogue"):
        cpm.build_memory_request([request], book_sha256=BOOK)


# --------------------------------------------------------------------------- capture and validation


def test_capture_binds_request_response_provenance_and_validates(chapters):
    bundle = make_memory(chapters)
    assert bundle["schema_version"] == "character-performance-memory-bundle-v1"
    assert bundle["request_sha256"] == _sha(bundle["request"]) and bundle["response_sha256"] == _sha(bundle["response"])
    assert bundle["response"]["schema_version"] == "character-performance-memory-proposals-v1"
    assert (bundle["approved"], bundle["author_instruction"], bundle["delivery_profile"]) == (False, False, False)
    assert bundle["interpretation_status"] == "unapproved_model_interpretation" and bundle["human_review"] == "pending"
    summary = cpm.validate_bundle(bundle)
    assert summary["memory_sha256"] == bundle["memory_sha256"] and summary["book_sha256"] == BOOK
    assert sorted(summary["profiles"]) == ["erick", "jane"] and summary["narrator_role_ids"] == ["narrator"]
    assert json.loads(json.dumps(bundle)) == bundle                   # plain JSON round trip keeps every hash
    assert cpm.validate_bundle(json.loads(json.dumps(bundle)))["memory_sha256"] == bundle["memory_sha256"]


def test_validation_summary_and_capture_are_independent_copies(chapters):
    request = cpm.build_memory_request(chapters, book_sha256=BOOK)
    response = make_response(request)
    provenance = dict(FIXTURE_PROVENANCE)
    before = copy.deepcopy((request, response, provenance))
    bundle = cpm.capture_bundle(request, response, provenance)
    assert (request, response, provenance) == before
    response["profiles"][0]["delivery_habits"].append("mutated afterwards")
    provenance["model_requested"] = "mutated afterwards"
    cpm.validate_bundle(bundle)                                        # the bundle did not follow the caller's objects
    snapshot = copy.deepcopy(bundle)
    summary = cpm.validate_bundle(bundle)
    summary["profiles"]["erick"]["delivery_habits"].append("mutated")
    summary["provenance"]["origin"] = "human"
    assert bundle == snapshot


def test_memory_hash_changes_with_source_model_roles_and_response(chapters):
    base = make_memory(chapters)["memory_sha256"]
    assert make_memory(chapters)["memory_sha256"] == base
    assert make_memory(chapters, book=OTHER_BOOK)["memory_sha256"] != base
    assert make_memory(chapters, provenance={**FIXTURE_PROVENANCE, "model_requested": "other-fixture"})["memory_sha256"] != base
    assert make_memory(chapters, provenance={**FIXTURE_PROVENANCE, "receipt_sha256": "e2" * 32})["memory_sha256"] != base
    edited = copy.deepcopy(chapters)
    edited[1]["scenes"][0]["context"][1]["text"] = "“Now, then?”"
    assert make_memory(edited)["memory_sha256"] != base                # source prose
    assert make_memory(chapters[:1])["memory_sha256"] != base          # source/role set
    assert make_memory(chapters, mutate=lambda r: r["profiles"][0].update(confidence="high"))["memory_sha256"] != base


PROFILE_REFUSALS = {
    "narrator-profile": (lambda r: r["profiles"].append({**copy.deepcopy(r["profiles"][0]), "role_id": "narrator",
                                                          "role_name": "Narrator"}), "Narrator narrator"),
    "missing-role": (lambda r: r["profiles"].pop(), "exactly cover"),
    "unrequested-role": (lambda r: r["profiles"].append({**copy.deepcopy(r["profiles"][0]), "role_id": "mog"}), "unrequested"),
    "duplicate-role": (lambda r: r["profiles"].append(copy.deepcopy(r["profiles"][0])), "Duplicate profile"),
    "wrong-name": (lambda r: r["profiles"][0].update(role_name="Eric"), "differs from the requested name"),
    "extra-profile-field": (lambda r: r["profiles"][0].update(voice_id="voice_x"), "fields differ"),
    "extra-response-field": (lambda r: r.update(approved=True), "fields differ"),
    "no-habits": (lambda r: r["profiles"][0].update(delivery_habits=[]), "delivery habits"),
    "four-habits": (lambda r: r["profiles"][0].update(delivery_habits=["a one", "b two", "c three", "d four"]), "delivery habits"),
    "long-habit": (lambda r: r["profiles"][0].update(delivery_habits=["F" + "irm" * 45]), "at most 120"),
    "duplicate-habits": (lambda r: r["profiles"][0].update(delivery_habits=["Firm and clear.", "firm and clear."]), "duplicate"),
    "habit-not-string": (lambda r: r["profiles"][0].update(delivery_habits=[7]), "delivery habit"),
    "habit-padded": (lambda r: r["profiles"][0].update(delivery_habits=[" Firm and clear."]), "surrounding whitespace"),
    "identity-term-age": (lambda r: r["profiles"][0].update(delivery_habits=["Sounds younger when excited."]), "identity"),
    "identity-term-accent": (lambda r: r["profiles"][0].update(delivery_habits=["A light accent on endings."]), "identity"),
    "identity-term-gender": (lambda r: r["profiles"][0].update(contrast_notes="A deeper, more male reading."), "identity"),
    "numeric-rate": (lambda r: r["profiles"][0].update(delivery_habits=["Speaks at 140 wpm."]), "numeric rate"),
    "narration-term": (lambda r: r["profiles"][0].update(contrast_notes="Reads like the narration around him."), "narration"),
    "persona-term": (lambda r: r["profiles"][0].update(delivery_habits=["Keeps a gruff persona."]), "persona"),
    "voice-id": (lambda r: r["profiles"][0].update(delivery_habits=["Keeps voice_8f3a2c clear."]), "voice id"),
    "borrowed-voice": (lambda r: r["profiles"][0].update(delivery_habits=["Delivered in the style of a quiet hush."]), "not safe"),
    "names-a-character": (lambda r: r["profiles"][0].update(contrast_notes="Blunter than Jane."), "names a character"),
    "control-character": (lambda r: r["profiles"][0].update(contrast_notes="Blunt.\nBrief."), "one line"),
    "long-contrast": (lambda r: r["profiles"][0].update(contrast_notes="B" + "lunt" * 70), "at most 240"),
    "bad-confidence": (lambda r: r["profiles"][0].update(confidence="certain"), "confidence"),
    "no-evidence": (lambda r: r["profiles"][0].update(evidence=[]), "evidence spans"),
    "too-much-evidence": (lambda r: r["profiles"][0].update(evidence=[{**r["profiles"][0]["evidence"][0], "end": e}
                                                                       for e in range(2, 11)]), "evidence spans"),
    "evidence-text-mismatch": (lambda r: r["profiles"][0]["evidence"][0].update(text="“Open its,”"), "differs from the supplied"),
    "evidence-offset-mismatch": (lambda r: r["profiles"][0]["evidence"][0].update(start=1), "differs from the supplied"),
    "evidence-past-end": (lambda r: r["profiles"][0]["evidence"][0].update(end=999), "differs from the supplied"),
    "evidence-bool-offset": (lambda r: r["profiles"][0]["evidence"][0].update(start=False), "differs from the supplied"),
    "evidence-unknown-row": (lambda r: r["profiles"][0]["evidence"][0].update(utterance_id="nope"), "differs from the supplied"),
    "evidence-wrong-project": (lambda r: r["profiles"][0]["evidence"][0].update(project_id="fix-zzz"), "differs from the supplied"),
    "evidence-extra-field": (lambda r: r["profiles"][0]["evidence"][0].update(role_id="erick"), "evidence fields"),
    "evidence-duplicate": (lambda r: r["profiles"][0]["evidence"].append(copy.deepcopy(r["profiles"][0]["evidence"][0])),
                           "duplicate evidence"),
    "profile-not-object": (lambda r: r["profiles"].append("jane"), "fields differ"),
    "schema": (lambda r: r.update(schema_version="scene-coaching-proposals-v1"), "response schema"),
}


@pytest.mark.parametrize("name", sorted(PROFILE_REFUSALS))
def test_unsafe_unsupported_or_uncovering_responses_are_refused_before_a_bundle_exists(chapters, name):
    mutate, message = PROFILE_REFUSALS[name]
    with pytest.raises(cpm.PerformanceMemoryError, match=message):
        make_memory(chapters, mutate=mutate)


def test_evidence_from_another_role_alone_does_not_support_a_profile(chapters):
    def only_the_other_role(response):
        project_id, row = row_of(cpm.build_memory_request(chapters, book_sha256=BOOK), "jane")
        response["profiles"][0]["evidence"] = [{"project_id": project_id, "utterance_id": row["utterance_id"], "start": 0,
                                                 "end": 3, "text": row["text"][:3]}]
    with pytest.raises(cpm.PerformanceMemoryError, match="spoken by the role"):
        make_memory(chapters, mutate=only_the_other_role)


def test_narration_may_be_extra_evidence_beside_the_roles_own_speech(chapters):
    def add_narration(response):
        request = cpm.build_memory_request(chapters, book_sha256=BOOK)
        narration = request["sources"][0]["context"][2]
        response["profiles"][0]["evidence"].append({"project_id": "fix-a", "utterance_id": narration["utterance_id"],
                                                    "start": 0, "end": 4, "text": narration["text"][:4]})
    assert len(make_memory(chapters, mutate=add_narration)["response"]["profiles"][0]["evidence"]) == 2


def test_two_roles_may_not_share_identical_habits(chapters):
    def same(response):
        response["profiles"][1]["delivery_habits"] = list(response["profiles"][0]["delivery_habits"])
    with pytest.raises(cpm.PerformanceMemoryError, match="identical habits"):
        make_memory(chapters, mutate=same)


def test_role_metadata_names_are_not_delivery_text(chapters):
    chapters[0]["scenes"][0]["context"][1]["role_name"] = "Male incani mugger"      # would trip the identity-term guard
    chapters[1]["scenes"][0]["context"][1]["role_name"] = "Male incani mugger"
    bundle = make_memory(chapters)
    assert bundle["response"]["profiles"][0]["role_name"] == "Male incani mugger"


BUNDLE_REFUSALS = {
    "unknown-schema": (lambda b: b.update(schema_version="scene-coaching-bundle-v1"), "Unknown"),
    "extra-field": (lambda b: b.update(voice_id="voice_x"), "fields or version"),
    "missing-field": (lambda b: b.pop("limitations"), "fields or version"),
    "version": (lambda b: b.update(version="character-performance-memory-0"), "fields or version"),
    "approved": (lambda b: b.update(approved=True), "approved"),
    "author-instruction": (lambda b: b.update(author_instruction=True), "author_instruction"),
    "delivery-profile": (lambda b: b.update(delivery_profile=True), "delivery_profile"),
    "voice-dependent": (lambda b: b.update(voice_independent=False), "voice_independent"),
    "scope": (lambda b: b.update(scope="chapter"), "scope"),
    "human-review": (lambda b: b.update(human_review="approved"), "human_review"),
    "limitations": (lambda b: b.update(limitations=[]), "limitations"),
    "request-hash": (lambda b: b.update(request_sha256="0" * 64), "hash mismatch"),
    "response-hash": (lambda b: b.update(response_sha256="0" * 64), "hash mismatch"),
    "memory-hash": (lambda b: b.update(memory_sha256="0" * 64), "Memory hash"),
    "edited-response-unrehashed": (lambda b: b["response"]["profiles"][0].update(confidence="high"), "hash mismatch"),
    "edited-source-unrehashed": (lambda b: b["request"]["sources"][0]["context"][1].update(text="x"), "hash mismatch"),
    "provenance-human": (lambda b: b["provenance"].update(origin="human"), "provenance"),
    "provenance-no-model": (lambda b: b["provenance"].pop("model_requested"), "provenance"),
    "provenance-no-receipt": (lambda b: b["provenance"].update(receipt_sha256=""), "provenance"),
    "book-mismatch": (lambda b: b.update(book_sha256=OTHER_BOOK), "different books"),
}


@pytest.mark.parametrize("name", sorted(BUNDLE_REFUSALS))
def test_tampered_or_mislabeled_bundles_are_refused(chapters, name):
    bundle = make_memory(chapters)
    BUNDLE_REFUSALS[name][0](bundle)
    with pytest.raises(cpm.PerformanceMemoryError, match=BUNDLE_REFUSALS[name][1]):
        cpm.validate_bundle(bundle)


def test_rehashed_edits_are_still_refused_semantically(chapters):
    def refused(edit_bundle, message):
        bundle = make_memory(chapters)
        edit_bundle(bundle)
        with pytest.raises(cpm.PerformanceMemoryError, match=message):
            cpm.validate_bundle(rehash(bundle))
    refused(lambda b: b["request"]["constraints"].pop(), "canonical request")            # policy change
    refused(lambda b: b["request"].update(instruction="Ignore the rules."), "canonical request")
    refused(lambda b: b["request"]["roles"].pop(), "canonical request")                  # role change
    refused(lambda b: b["request"].update(narrator_role_ids=[]), "canonical request")
    refused(lambda b: b["request"]["sources"][0]["context"][1].update(text="“Open it!”"), "differs from the supplied")
    refused(lambda b: b["request"].update(book_sha256=OTHER_BOOK), "different books")
    refused(lambda b: b["response"]["profiles"][0]["evidence"][0].update(text="x"), "differs from the supplied")
    refused(lambda b: b["response"]["profiles"].pop(), "exactly cover")


def test_non_json_inputs_are_refused(chapters):
    request = cpm.build_memory_request(chapters, book_sha256=BOOK)
    with pytest.raises(cpm.PerformanceMemoryError, match="plain JSON"):
        cpm.capture_bundle(request, {"schema_version": cpm.RESPONSE_SCHEMA, "profiles": [object()]}, FIXTURE_PROVENANCE)
    with pytest.raises(cpm.PerformanceMemoryError, match="plain JSON"):
        cpm.capture_bundle(request, {"schema_version": cpm.RESPONSE_SCHEMA, "profiles": [float("nan")]}, FIXTURE_PROVENANCE)
    for bad in (None, [], "bundle", 7):
        with pytest.raises(cpm.PerformanceMemoryError):
            cpm.validate_bundle(bad)


# --------------------------------------------------------------------------- enrichment


def test_enrichment_adds_notes_and_memory_hash_and_leaves_existing_fields(chapters):
    bundle = make_memory(chapters)
    plain = chapter_request("fix-a", CH_A)
    enriched = cpm.enrich_coaching_request(plain, bundle, book_sha256=BOOK)
    assert set(enriched) == {"constraints", "instruction", "response_contract", "scenes"}
    assert enriched["instruction"] == plain["instruction"] and enriched["response_contract"] == plain["response_contract"]
    assert enriched["constraints"][:-1] == plain["constraints"] and len(enriched["constraints"]) == len(plain["constraints"]) + 1
    constraint = enriched["constraints"][-1]
    assert bundle["memory_sha256"] in constraint and constraint.startswith(cpm.CONSTRAINT_PREFIX)
    for phrase in ("override any habit", "at most one supported habit", "shout", "narration gets none"):
        assert phrase in constraint
    scene = enriched["scenes"][0]
    assert {k: v for k, v in scene.items() if k != cpm.SCENE_KEY} == plain["scenes"][0]
    memory = scene[cpm.SCENE_KEY]
    assert memory["memory_sha256"] == bundle["memory_sha256"] and memory["interpretation"] == cpm.STATUS
    assert [n["role_id"] for n in memory["notes"]] == ["erick", "jane"]
    assert all(set(n) == {"role_id", "role_name", "delivery_habits", "contrast_notes", "confidence"} for n in memory["notes"])
    assert "narrator" not in {n["role_id"] for n in memory["notes"]}
    assert cpm.enriched_memory_sha256(enriched) == bundle["memory_sha256"] and cpm.enriched_memory_sha256(plain) is None


def test_enrichment_is_a_pure_deep_copy(chapters):
    bundle = make_memory(chapters)
    plain = chapter_request("fix-a", CH_A)
    plain_before, bundle_before = copy.deepcopy(plain), copy.deepcopy(bundle)
    enriched = cpm.enrich_coaching_request(plain, bundle, book_sha256=BOOK)
    assert plain == plain_before and bundle == bundle_before
    enriched["scenes"][0]["context"][0]["text"] = "changed"
    enriched["scenes"][0][cpm.SCENE_KEY]["notes"][0]["delivery_habits"].append("mutated")
    enriched["constraints"].append("mutated")
    enriched["response_contract"]["scenes"] = "mutated"
    assert plain == plain_before and bundle == bundle_before
    again = cpm.enrich_coaching_request(plain, bundle, book_sha256=BOOK)
    assert again["scenes"][0][cpm.SCENE_KEY]["notes"][0]["delivery_habits"] == bundle["response"]["profiles"][0]["delivery_habits"]


def test_a_fresh_chapter_of_the_same_book_is_enriched_by_role_id(chapters):
    bundle = make_memory(chapters)
    fresh = chapter_request("fix-fresh", CH_FRESH)
    enriched = cpm.enrich_coaching_request(fresh, bundle, book_sha256=BOOK)
    assert [n["role_id"] for n in enriched["scenes"][0][cpm.SCENE_KEY]["notes"]] == ["jane", "erick"]


def test_every_scene_is_enriched_with_only_its_relevant_roles(chapters):
    bundle = make_memory(chapters)
    plain = chapter_request("fix-two", CH_A)
    second = chapter_request("fix-two", CH_FRESH)["scenes"][0]
    second.update(scene_id="fix-two-scene-2")
    second["context"] = [r for r in second["context"] if r["role_id"] != "erick"]
    second["selected_dialogue_ids"] = [u for u in second["selected_dialogue_ids"] if u in {r["utterance_id"] for r in second["context"]}]
    plain["scenes"].append(second)
    enriched = cpm.enrich_coaching_request(plain, bundle, book_sha256=BOOK)
    assert [[n["role_id"] for n in s[cpm.SCENE_KEY]["notes"]] for s in enriched["scenes"]] == [["erick", "jane"], ["jane"]]
    assert {s[cpm.SCENE_KEY]["memory_sha256"] for s in enriched["scenes"]} == {bundle["memory_sha256"]}


ENRICH_REFUSALS = {
    "role-without-profile": (lambda r: r["scenes"][0]["context"].append(
        {"utterance_id": "fix-a-09", "text": "“Hm.”", "role_id": "mog", "narrator": False, "role_name": "Mog"})
        or r["scenes"][0]["selected_dialogue_ids"].append("fix-a-09"), "no profile for role mog"),
    "memory-role-is-narrator-here": (lambda r: r["scenes"][0]["context"][1].update(narrator=True)
                                     or r["scenes"][0]["selected_dialogue_ids"].remove("fix-a-01"), "narrator here"),
    "memory-narrator-is-character-here": (lambda r: r["scenes"][0].update(
        context=[{"utterance_id": "fix-a-00", "text": "“Hm.”", "role_id": "narrator", "narrator": False, "role_name": "Narrator"}],
        selected_dialogue_ids=["fix-a-00"]), "narrator in the memory"),
    "name-changed": (lambda r: [row.update(role_name="Eric") for row in r["scenes"][0]["context"] if row["role_id"] == "erick"],
                     "named 'Eric'"),
    "name-belongs-to-another-id": (lambda r: [row.update(role_id="erick-b") for row in r["scenes"][0]["context"]
                                              if row["role_id"] == "erick"], "never mapped by name"),
    "ambiguous-in-request": (lambda r: r["scenes"][0]["context"].append(
        {"utterance_id": "fix-a-08", "text": "“Hm.”", "role_id": "erick", "narrator": False, "role_name": "Eric"}),
        "more than one name"),
    "already-bound-constraint": (lambda r: r["constraints"].append(cpm.CONSTRAINT_PREFIX + "x"), "already carry"),
    "already-bound-scene": (lambda r: r["scenes"][0].update(character_performance_memory={}), "already enriched"),
    "extra-top-level-field": (lambda r: r.update(performance_memory="x"), "four-field"),
    "missing-top-level-field": (lambda r: r.pop("instruction"), "four-field"),
    "no-scenes": (lambda r: r.update(scenes=[]), "no scenes"),
}


@pytest.mark.parametrize("name", sorted(ENRICH_REFUSALS))
def test_enrichment_refuses_role_name_book_and_shape_conflicts(chapters, name):
    bundle = make_memory(chapters)
    request = chapter_request("fix-a", CH_A)
    ENRICH_REFUSALS[name][0](request)
    before = copy.deepcopy(request)
    with pytest.raises(cpm.PerformanceMemoryError, match=ENRICH_REFUSALS[name][1]):
        cpm.enrich_coaching_request(request, bundle, book_sha256=BOOK)
    assert request == before


def test_enrichment_refuses_another_book_and_a_bad_memory(chapters):
    bundle = make_memory(chapters)
    request = chapter_request("fix-a", CH_A)
    with pytest.raises(cpm.PerformanceMemoryError, match="book differs"):
        cpm.enrich_coaching_request(request, bundle, book_sha256=OTHER_BOOK)
    with pytest.raises(cpm.PerformanceMemoryError, match="sha256"):
        cpm.enrich_coaching_request(request, bundle, book_sha256="nope")
    bundle["response"]["profiles"][0]["confidence"] = "high"
    with pytest.raises(cpm.PerformanceMemoryError, match="hash mismatch"):
        cpm.enrich_coaching_request(request, bundle, book_sha256=BOOK)


def test_inconsistent_memory_carriage_is_detected(chapters):
    enriched = cpm.enrich_coaching_request(chapter_request("fix-a", CH_A), make_memory(chapters), book_sha256=BOOK)
    for edit_request, message in (
            (lambda r: r["constraints"].pop(), "inconsistently"),
            (lambda r: r["scenes"][0].pop(cpm.SCENE_KEY), "inconsistently"),
            (lambda r: r["scenes"][0][cpm.SCENE_KEY].update(memory_sha256="d" * 64), "differs from its scenes"),
            (lambda r: r["scenes"][0][cpm.SCENE_KEY].update(memory_sha256="short"), "sha256")):
        damaged = copy.deepcopy(enriched)
        edit_request(damaged)
        with pytest.raises(cpm.PerformanceMemoryError, match=message):
            cpm.enriched_memory_sha256(damaged)


def test_changed_memory_changes_the_scene_context_and_policy_hashes_scene_coaching_caches_on(chapters):
    plain = chapter_request("fix-a", CH_A)
    first = cpm.enrich_coaching_request(plain, make_memory(chapters), book_sha256=BOOK)

    def keys(request):
        policy = _sha({k: request[k] for k in ["instruction", "constraints", "response_contract"]})
        return policy, [_sha(scene) for scene in request["scenes"]]

    plain_keys = keys(plain)
    assert keys(first) != plain_keys and all(a != b for a, b in zip(keys(first)[1], plain_keys[1]))
    assert keys(cpm.enrich_coaching_request(plain, make_memory(chapters), book_sha256=BOOK)) == keys(first)  # stable
    for variant in (
            make_memory(chapters, provenance={**FIXTURE_PROVENANCE, "model_requested": "other-fixture"}),
            make_memory(chapters, mutate=lambda r: r["profiles"][0].update(confidence="high")),
            make_memory(chapters[:1]),
            make_memory([chapters[0], chapter_request("fix-b", CH_B[:2] + [("02", "jane", "Jane", False, "“We stay,”")])])):
        second = cpm.enrich_coaching_request(plain, variant, book_sha256=BOOK)
        assert keys(second)[0] != keys(first)[0], "policy hash must carry the memory hash"
        assert keys(second)[1] != keys(first)[1], "scene context hash must carry the memory hash"


def test_an_enriched_request_is_accepted_by_the_existing_scene_coaching_validator(chapters):
    enriched = cpm.enrich_coaching_request(chapter_request("fix-a", CH_A), make_memory(chapters), book_sha256=BOOK)
    scenes = []
    for scene in enriched["scenes"]:
        by_uid = {r["utterance_id"]: r for r in scene["context"]}
        directions = [{"utterance_id": uid, "objective": "Fixture objective", "emotion": "fixture tension",
                       "intensity": "moderate", "cadence": "steady", "emphasis": [], "ending": "settle",
                       "confidence": "moderate",
                       "native_instruction": "Deliver this line with controlled urgency; keep every word clear.",
                       "evidence": [{"utterance_id": uid, "start": 0, "end": len(by_uid[uid]["text"]),
                                     "text": by_uid[uid]["text"]}]} for uid in scene["selected_dialogue_ids"]]
        scenes.append({"scene_id": scene["scene_id"], "directions": directions})
    response = {"schema_version": scene_coaching.RESPONSE_SCHEMA, "scenes": scenes}
    bundle = {"schema_version": scene_coaching.BUNDLE_SCHEMA, "request": enriched, "response": response,
              "request_sha256": _sha(enriched), "response_sha256": _sha(response),
              "provenance": {"origin": "model", "model_requested": "synthetic-test-fixture-not-a-real-model",
                             "receipt_sha256": "a1" * 32, "fixture": True}}
    assert set(scene_coaching.validate_bundle(bundle)) == {s["scene_id"] for s in enriched["scenes"]}
