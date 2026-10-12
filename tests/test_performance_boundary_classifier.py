"""Boundary-classification contracts for section 9.2's taxonomy.

Every case is synthetic and CPU-only.  The classifier is pure: it reads two
input rows and returns one class with the features that decided it.
"""

from __future__ import annotations

import pytest

from dubbing_service.app.audiobook import performance_boundaries as boundaries

ROLE_TYPES = {"narrator": boundaries.NARRATOR, "rosa": boundaries.CHARACTER,
              "miles": boundaries.CHARACTER, "other-purpose": "other"}
ALIASES = {"rosa-alias": "rosa", "rosa-alias-2": "rosa-alias"}


def line(utterance_id: str, role_id: str, text: str, start: int, end: int, *,
         revision: str = "chapter-rev-1", is_tag: bool = False, separator: bool = False,
         previous_tag=None) -> dict:
    return {
        "utterance_id": utterance_id,
        "role_id": role_id,
        "text": text,
        "source_revision_id": revision,
        "start_offset": start,
        "end_offset": end,
        "is_tag": is_tag,
        "separator": separator,
        "previous_tag": previous_tag,
    }


def classify(a: dict, b: dict) -> dict:
    return boundaries.classify_boundary(a, b, role_types=ROLE_TYPES, aliases=ALIASES)


def adjacent(a_text: str, a_role: str, b_text: str, b_role: str, *, gap: int = 0,
             a_tag: bool = False, separator: bool = False, previous_tag=None) -> dict:
    """Two lines in one source block; gap 0 is the same paragraph."""

    a = line("a", a_role, a_text, 100, 120, is_tag=a_tag)
    b = line("b", b_role, b_text, 120 + gap, 140 + gap, separator=separator,
             previous_tag=previous_tag)
    return classify(a, b)


def test_same_and_new_paragraph_pairs_are_separated() -> None:
    assert adjacent("The room was quiet.", "narrator", "She waited.", "narrator")["class"] == "N_N_SAME_PARA"
    assert adjacent("The room was quiet.", "narrator", "She waited.", "narrator", gap=2)["class"] == "N_N_NEW_PARA"
    assert adjacent("The room was quiet.", "narrator", "I never said that.", "rosa")["class"] == "N_D_SAME_PARA"
    assert adjacent("The room was quiet.", "narrator", "I never said that.", "rosa", gap=2)["class"] == "N_D_NEW_PARA"
    assert adjacent("I never said that.", "rosa", "She looked away.", "narrator")["class"] == "D_N_BEAT_SAME_PARA"
    assert adjacent("I never said that.", "rosa", "She looked away.", "narrator", gap=2)["class"] == "D_N_NEW_PARA"


def test_a_source_gap_of_nine_is_recorded_and_not_promoted_to_a_separator() -> None:
    result = adjacent("The room was quiet.", "narrator", "She waited.", "narrator", gap=9)
    assert result["class"] == "N_N_NEW_PARA"
    assert result["features"]["source_gap"] == 9
    assert result["features"]["para_break"] is True


def test_dialogue_then_an_attached_tag_is_a_tag() -> None:
    comma = adjacent("I never said that,", "rosa", "she said.", "narrator")
    assert comma["class"] == "D_TAG"
    lower = adjacent("I never said that.", "rosa", "he said.", "narrator")
    assert lower["class"] == "D_TAG"
    assert lower["features"]["b_lower"] is True
    pattern = adjacent("I never said that.", "rosa", "The woman said nothing.", "narrator")
    assert pattern["class"] == "D_TAG"


def test_a_dialogue_then_narration_pair_without_a_tag_is_a_beat() -> None:
    result = adjacent("I never said that.", "rosa", "She looked away.", "narrator")
    assert result["class"] == "D_N_BEAT_SAME_PARA"
    assert result["features"]["b_lower"] is False
    assert result["features"]["a_end"] == "."


def test_a_tag_keeps_continuation_metadata_only_with_explicit_context() -> None:
    tag = line("tag", "narrator", "she said quietly.", 100, 110, is_tag=True)
    resumed = line("b", "rosa", "and I never will.", 110, 120,
                   previous_tag={"role_id": "rosa", "class": "D_TAG"})
    result = classify(tag, resumed)
    assert result["class"] == "TAG_D_CONTINUATION"
    assert result["features"]["previous_tag"] == {"role_id": "rosa", "class": "D_TAG"}
    assert result["features"]["same_role"] is True

    aliased = line("b", "rosa-alias", "and I never will.", 110, 120,
                   previous_tag={"role_id": "rosa", "class": "D_TAG"})
    assert classify(tag, aliased)["class"] == "TAG_D_CONTINUATION"


def test_a_tag_without_supplied_context_is_unknown_rather_than_guessed() -> None:
    tag = line("tag", "narrator", "she said quietly.", 100, 110, is_tag=True)
    resumed = line("b", "rosa", "and I never will.", 110, 120)
    result = classify(tag, resumed)
    assert result["class"] == "UNKNOWN"
    assert "continuation context" in result["reason"]

    other_speaker = line("b", "miles", "and I never will.", 110, 120,
                         previous_tag={"role_id": "rosa", "class": "D_TAG"})
    assert classify(tag, other_speaker)["class"] == "UNKNOWN"

    unknown_class = line("b", "rosa", "and I never will.", 110, 120,
                         previous_tag={"role_id": "rosa", "class": "D_TAG_V2"})
    ambiguous = classify(tag, unknown_class)
    assert ambiguous["class"] == "UNKNOWN"
    assert "ambiguous tag" in ambiguous["reason"]

    not_a_mapping = line("b", "rosa", "and I never will.", 110, 120, previous_tag="D_TAG")
    assert classify(tag, not_a_mapping)["class"] == "UNKNOWN"


def test_a_paragraph_break_after_a_tag_is_not_a_continuation() -> None:
    tag = line("tag", "narrator", "she said quietly.", 100, 110, is_tag=True)
    resumed = line("b", "rosa", "and I never will.", 112, 120,
                   previous_tag={"role_id": "rosa", "class": "D_TAG"})
    assert classify(tag, resumed)["class"] == "N_D_NEW_PARA"


def test_dialogue_to_dialogue_families() -> None:
    assert adjacent("I never said that.", "rosa", "You did.", "miles")["class"] == "D_D_REPLY"
    interrupted = adjacent("I never said that\u2014", "rosa", "You did.", "miles")
    assert interrupted["class"] == "D_D_INTERRUPTION"
    assert adjacent("I never said that-", "rosa", "You did.", "miles")["class"] == "D_D_INTERRUPTION"
    trailoff = adjacent("I never said that\u2026", "rosa", "You did.", "miles")
    assert trailoff["class"] == "D_D_AFTER_TRAILOFF"
    assert adjacent("I never said anything...", "rosa", "You did.", "miles")["class"] == "D_D_AFTER_TRAILOFF"
    assert adjacent("I never said that.", "rosa", "And I never will.", "rosa")["class"] == "D_D_SAME_SPEAKER"


def test_narration_then_an_ellipsis_is_not_a_dialogue_trailoff() -> None:
    result = adjacent("She waited...", "narrator", "Then she spoke.", "narrator")
    assert result["class"] == "N_N_SAME_PARA"


def test_alias_equivalence_uses_the_canonical_role() -> None:
    direct = adjacent("I never said that.", "rosa", "And I never will.", "rosa")
    aliased = adjacent("I never said that.", "rosa", "And I never will.", "rosa-alias")
    chained = adjacent("I never said that.", "rosa-alias-2", "And I never will.", "rosa")
    assert direct["class"] == aliased["class"] == chained["class"] == "D_D_SAME_SPEAKER"
    assert chained["a_role_id"] == chained["b_role_id"] == "rosa"


def test_alias_cycles_are_refused_rather_than_followed_forever() -> None:
    a = line("a", "loop-a", "One.", 100, 120)
    b = line("b", "loop-b", "Two.", 120, 140)
    result = boundaries.classify_boundary(a, b, role_types={"loop-a": "character", "loop-b": "character"},
                                          aliases={"loop-a": "loop-b", "loop-b": "loop-a"})
    assert result["class"] == "UNKNOWN"
    assert "cycle" in result["reason"]


def test_an_explicit_separator_takes_precedence_and_is_reported_once() -> None:
    result = adjacent("I never said that.", "rosa", "Later that night.", "narrator", separator=True)
    assert result["class"] == "SCENE_SEPARATOR"
    assert result["reason"] == "explicit source separator between the lines"
    assert result["features"]["explicit_separator"] is True
    # An explicit separator wins even where a tag rule would otherwise apply.
    tagged = adjacent("I never said that,", "rosa", "she said.", "narrator", separator=True)
    assert tagged["class"] == "SCENE_SEPARATOR"


def test_missing_anchors_unknown_roles_and_cross_revisions_are_unknown() -> None:
    a = line("a", "narrator", "One.", 100, 120)
    missing_start = line("b", "rosa", "Two.", 0, 140)
    missing_start["start_offset"] = None
    result = classify(a, missing_start)
    assert result["class"] == "UNKNOWN" and "anchor" in result["reason"]

    missing_end = line("a", "narrator", "One.", 100, 120)
    missing_end["end_offset"] = None
    assert classify(missing_end, line("b", "rosa", "Two.", 120, 140))["class"] == "UNKNOWN"

    no_revision = line("a", "narrator", "One.", 100, 120, revision="")
    assert classify(no_revision, line("b", "rosa", "Two.", 120, 140))["class"] == "UNKNOWN"

    assert adjacent("I never said that.", "rosa", "You did.", "miles", gap=0)["class"] == "D_D_REPLY"
    cross = boundaries.classify_boundary(
        line("a", "rosa", "One.", 100, 120, revision="rev-1"),
        line("b", "miles", "Two.", 120, 140, revision="rev-2"),
        role_types=ROLE_TYPES, aliases=ALIASES)
    assert cross["class"] == "UNKNOWN" and "revisions" in cross["reason"]

    unknown_role = classify(a, line("b", "ghost", "Two.", 120, 140))
    assert unknown_role["class"] == "UNKNOWN" and "role type" in unknown_role["reason"]
    assert unknown_role["features"]["b_role_type"] is None

    other_role = classify(a, line("b", "other-purpose", "Two.", 120, 140))
    assert other_role["class"] == "UNKNOWN"


def test_overlapping_anchors_are_unknown() -> None:
    result = classify(line("a", "narrator", "One.", 100, 120), line("b", "rosa", "Two.", 100, 140))
    assert result["class"] == "UNKNOWN"
    assert "overlap" in result["reason"]


def test_the_classification_record_is_self_describing() -> None:
    result = adjacent("The room was quiet.", "narrator", "I never said that.", "rosa")
    assert result["schema_version"] == "boundary-classification-v1"
    assert result["classifier_version"] == boundaries.CLASSIFIER_VERSION
    assert result["a_role_id"] == "narrator" and result["b_role_id"] == "rosa"
    assert result["machine_generated"] is True and result["human_verified"] is False
    assert set(result["features"]) >= {"a_end", "b_lower", "b_words", "para_break", "source_gap",
                                       "same_role", "a_role_type", "b_role_type", "a_is_tag"}
    assert result["class"] in boundaries.BOUNDARY_CLASSES


def test_input_rows_must_be_mappings_with_identifiers_and_text() -> None:
    with pytest.raises(boundaries.PerformanceDirectorError) as refused:
        boundaries.classify_boundary("not-a-row", line("b", "rosa", "Two.", 120, 140),
                                     role_types=ROLE_TYPES, aliases=ALIASES)
    assert refused.value.code == "input_ineligible"

    with pytest.raises(boundaries.PerformanceDirectorError):
        boundaries.classify_boundary(line("../escape", "narrator", "One.", 100, 120),
                                     line("b", "rosa", "Two.", 120, 140),
                                     role_types=ROLE_TYPES, aliases=ALIASES)

    with pytest.raises(boundaries.PerformanceDirectorError):
        boundaries.classify_boundary(line("a", "narrator", None, 100, 120),
                                     line("b", "rosa", "Two.", 120, 140),
                                     role_types=ROLE_TYPES, aliases=ALIASES)
