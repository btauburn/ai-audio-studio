"""CPU-only tests for Performance Director scene-aware analysis (node C).

Nothing here touches a project store, a model, a network or a GPU.  Caches are
written only under ``tmp_path``.  The text fixtures are synthetic prose written
for these tests; no assertion is evidence about how a voice sounds.
"""

from __future__ import annotations

import copy
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from dubbing_service.app.audiobook import performance_director as director
from dubbing_service.app.audiobook import performance_scene_analysis as psa
from dubbing_service.app.audiobook import provider_capabilities
from dubbing_service.app.models.audiobook import Attribution, SourceAnchor, SourceRevision
from dubbing_service.app.models.character import (
    CharacterDossier,
    CharacterFact,
    CharacterVoiceIdentity,
)
from dubbing_service.app.models.project import Chapter, Project, Role, Segment

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "a" * 64
SOURCE_ID = "src_fixture0001"  # a registry id, deliberately not a hash


def identity(role_id: str, prompt: str = "p1") -> CharacterVoiceIdentity:
    return CharacterVoiceIdentity(
        id=f"voice-{role_id}", role_id=role_id, mechanism="qwen_reference", dossier_revision="d1",
        casting_profile_revision="c1", design_instruction="synthetic fixture", model="fixture-model",
        prompt_sha256=prompt * 32 if len(prompt) == 2 else hashlib.sha256(prompt.encode()).hexdigest())


def make_roles(with_identity: bool = True) -> list[Role]:
    roles = [Role(id="narr", name="Narrator", role_type="narrator"),
             Role(id="mara", name="Mara Quill"), Role(id="tom", name="Tom")]
    if with_identity:
        for role in roles:
            role.voice_identity = identity(role.id)
    return roles


def make_segments(rows: list[tuple[str, str]], *, chapter: str = "c1", start: int = 0,
                  source: str | None = SOURCE_ID) -> list[Segment]:
    return [Segment(id=f"u{start + i}", chapter_id=chapter, role_id=role, index=start + i, text=text,
                    source_anchor=SourceAnchor(source_revision_id=source) if source else None,
                    attribution=Attribution(speaker_id=role, source="user", reviewed=True))
            for i, (role, text) in enumerate(rows)]


def make_registry(digest: str = SOURCE, revision_id: str = SOURCE_ID) -> list[SourceRevision]:
    return [SourceRevision(id=revision_id, uri="fixture://scene-analysis", kind="inline", sha256=digest)]


def make_project(roles: list[Role] | None = None, **extra) -> Project:
    extra.setdefault("source_revisions", make_registry())
    return Project(id="proj-1", name="Fixture", roles=roles or make_roles(),
                   chapters=[Chapter(id="c1", title="One", index=1), Chapter(id="c2", title="Two", index=2)],
                   **extra)


DRAMA = [
    ("narr", "Tom looked at the door and waited."),
    ("mara", '"Where were you?"'),
    ("tom", '"Out."'),
    ("narr", "Nobody spoke."),
    ("mara", '"Fine."'),
]


def by_id(result: dict, utterance_id: str) -> dict:
    return next(row for row in result["directions"] if row["utterance_id"] == utterance_id)


def fake_capability(profile: str, identity_dump: dict | None = None, *, forwards: bool) -> dict:
    mechanism = (identity_dump or {}).get("mechanism")
    state = "forwarded_experimental" if forwards else "available_not_forwarded"
    record = {
        "schema_version": "performance-capability-v1", "runtime_profile": profile, "backend": "qwen3tts",
        "identity_mechanism": mechanism, "conditioning_mode": "icl" if mechanism == "qwen_reference" else "not_applicable",
        "evidence": {"faster_version": "fixture"},
        "controls": {name: {"state": "unsupported", "stage": "native", "review_copy_eligible": False,
                            "reason": "fixture"} for name in
                     ("emotion_label", "intensity", "emphasis", "role_delivery", "numeric_rate", "seed")}
        | {"native_instruction": {"state": state, "stage": "native", "review_copy_eligible": True, "reason": "fixture"},
           "pace": {"state": "available", "stage": "postprocess", "review_copy_eligible": False, "reason": "fixture"},
           "volume": {"state": "available", "stage": "postprocess", "review_copy_eligible": False, "reason": "fixture"}},
    }
    record["sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode("utf-8")).hexdigest()
    return record


@pytest.fixture()
def forwarding_runtime(monkeypatch):
    """A fake capability owner that forwards on the experimental profile only (no repo file reads)."""

    def probe(profile, identity_dump=None, **_):
        return fake_capability(profile, identity_dump, forwards=profile == psa.EXPERIMENTAL_RUNTIME_PROFILE
                               and (identity_dump or {}).get("mechanism") == "qwen_reference")

    monkeypatch.setattr(provider_capabilities, "performance_capabilities", probe)
    monkeypatch.setattr(provider_capabilities, "experimental_direction_capabilities", probe)


def evidence_items(value):
    """Every evidence-like record (it has utterance_id, start, end and text) in a nested structure."""

    if isinstance(value, dict):
        if {"utterance_id", "start", "end", "text"} <= set(value):
            yield value
        for item in value.values():
            yield from evidence_items(item)
    elif isinstance(value, list):
        for item in value:
            yield from evidence_items(item)


# ------------------------------------------------------------ source binding


def test_exact_prose_is_preserved_and_every_evidence_span_is_source_bound():
    rows = [("narr", 'Mara said angrily, "Get out!"'), ("mara", '"Get OUT of my *house*, you liar!"'),
            ("tom", '"I... I suppose you are right."'), ("narr", "Nobody spoke for a long moment.")]
    segments = make_segments(rows)
    before = [segment.model_dump(mode="json") for segment in segments]
    project = make_project()
    snapshot = project.model_dump(mode="json")
    result = psa.analyze_scenes(project, segments)

    assert [segment.model_dump(mode="json") for segment in segments] == before
    assert project.model_dump(mode="json") == snapshot
    texts = {segment.id: segment.text for segment in segments}
    scene = result["scenes"][0]
    assert [(u["utterance_id"], u["text"]) for u in scene["utterances"]] == list(texts.items())
    for item in scene["utterances"]:
        assert item["text_sha256"] == hashlib.sha256(texts[item["utterance_id"]].encode()).hexdigest()
    spans = list(evidence_items(result["directions"]))
    assert spans, "the fixture must produce evidence"
    for span in spans:
        assert texts[span["utterance_id"]][span["start"]:span["end"]] == span["text"]
    assert all(row["human_review"] == "pending" and row["approved"] is False for row in result["directions"])
    assert result["human_review"] == "pending"


def test_result_is_json_compatible_and_deterministic():
    segments = make_segments(DRAMA)
    first = psa.analyze_scenes(make_project(), segments)
    second = psa.analyze_scenes(make_project(), segments)
    assert json.loads(json.dumps(first)) == first
    assert first["analysis_sha256"] == second["analysis_sha256"]
    assert first == second


# --------------------------------------------------------------------- cache


def test_cache_miss_then_hit_never_rewrites_the_entry(tmp_path):
    segments = make_segments(DRAMA)
    first = psa.analyze_scenes(make_project(), segments, cache_root=tmp_path)
    assert [entry["status"] for entry in first["cache"]["scenes"]] == ["miss_written"]
    assert first["side_effects"]["cache_written"] is True
    key = first["scenes"][0]["analysis_key"]
    entry = tmp_path / psa.CACHE_DIRECTORY / f"{key}.json"
    stored = entry.read_bytes()
    assert json.loads(stored)["texts"] == [segment.text for segment in segments]

    second = psa.analyze_scenes(make_project(), segments, cache_root=tmp_path)
    assert [entry_["status"] for entry_ in second["cache"]["scenes"]] == ["hit"]
    assert second["side_effects"]["cache_written"] is False
    assert second["cache"]["hits"] == 1 and second["cache"]["identity"] == first["cache"]["identity"]
    assert entry.read_bytes() == stored
    assert second["directions"] == first["directions"]
    assert sorted(path.name for path in tmp_path.rglob("*") if path.is_file()) == [f"{key}.json"]


def test_changed_prose_is_stale_and_old_entry_is_left_alone(tmp_path):
    segments = make_segments(DRAMA)
    first = psa.analyze_scenes(make_project(), segments, cache_root=tmp_path)
    old_key = first["scenes"][0]["analysis_key"]
    old_bytes = (tmp_path / psa.CACHE_DIRECTORY / f"{old_key}.json").read_bytes()

    edited = copy.deepcopy(segments)
    edited[1].text = '"Where on earth were you?!"'
    second = psa.analyze_scenes(make_project(), edited, cache_root=tmp_path)
    new = second["scenes"][0]
    assert new["analysis_key"] != old_key
    assert second["cache"]["scenes"][0]["status"] == "miss_written"
    assert new["utterances"][1]["text"] == '"Where on earth were you?!"'
    assert by_id(second, "u1")["ending"]["terminal"] == "question"
    assert (tmp_path / psa.CACHE_DIRECTORY / f"{old_key}.json").read_bytes() == old_bytes


def test_role_voice_change_reuses_the_scene_but_changes_the_voice_binding(tmp_path):
    segments = make_segments(DRAMA)
    first = psa.analyze_scenes(make_project(), segments, cache_root=tmp_path)
    roles = make_roles()
    roles[1].voice_identity = identity("mara", prompt="a-different-voice-prompt")
    second = psa.analyze_scenes(make_project(roles), segments, cache_root=tmp_path)

    assert second["scenes"][0]["analysis_key"] == first["scenes"][0]["analysis_key"]
    assert second["cache"]["scenes"][0]["status"] == "hit"
    assert by_id(second, "u1")["voice_revision"] != by_id(first, "u1")["voice_revision"]
    assert by_id(second, "u2")["voice_revision"] == by_id(first, "u2")["voice_revision"]
    assert by_id(second, "u1")["preservation"]["voice_unchanged"] is True


def test_role_type_or_attribution_change_invalidates_the_scene(tmp_path):
    segments = make_segments(DRAMA)
    base = psa.analyze_scenes(make_project(), segments, cache_root=tmp_path)["scenes"][0]["analysis_key"]

    roles = make_roles()
    roles[0].role_type = "character"  # the narrator becomes a character
    retyped = psa.analyze_scenes(make_project(roles), segments, cache_root=tmp_path)
    assert retyped["scenes"][0]["analysis_key"] != base
    assert by_id(retyped, "u0")["status"] == "recommendation"

    reattributed = copy.deepcopy(segments)
    reattributed[2].role_id = "mara"
    reattributed[2].attribution = Attribution(speaker_id="mara", source="user", reviewed=True)
    other = psa.analyze_scenes(make_project(), reattributed, cache_root=tmp_path)
    assert other["scenes"][0]["analysis_key"] != base


def test_tampered_or_mismatched_entry_is_ignored_and_not_overwritten(tmp_path):
    segments = make_segments(DRAMA)
    key = psa.analyze_scenes(make_project(), segments, cache_root=tmp_path)["scenes"][0]["analysis_key"]
    path = tmp_path / psa.CACHE_DIRECTORY / f"{key}.json"
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["texts"][1] = "tampered prose"
    path.write_text(json.dumps(entry), encoding="utf-8")
    tampered = path.read_bytes()

    result = psa.analyze_scenes(make_project(), segments, cache_root=tmp_path)
    assert result["cache"]["scenes"][0]["status"].startswith("invalid_entry_ignored")
    assert result["scenes"][0]["utterances"][1]["text"] == segments[1].text  # live prose, not the cache's
    assert path.read_bytes() == tampered
    assert result["side_effects"]["cache_written"] is False

    path.write_text("not json", encoding="utf-8")
    again = psa.analyze_scenes(make_project(), segments, cache_root=tmp_path)
    assert again["cache"]["scenes"][0]["status"].startswith("invalid_entry_ignored")
    assert path.read_text(encoding="utf-8") == "not json"


def test_cache_root_must_exist_and_no_cache_means_no_files(tmp_path):
    with pytest.raises(psa.SceneAnalysisError) as excinfo:
        psa.analyze_scenes(make_project(), make_segments(DRAMA), cache_root=tmp_path / "missing")
    assert excinfo.value.code == "input_ineligible"
    assert not (tmp_path / "missing").exists()
    watched = tmp_path / "watched"
    watched.mkdir()
    result = psa.analyze_scenes(make_project(), make_segments(DRAMA))
    assert result["cache"]["enabled"] is False
    assert result["cache"]["scenes"][0]["status"] == "uncached"
    assert list(watched.iterdir()) == []


# --------------------------------------------------------------- narrator


def test_narrator_is_preserved_and_never_gets_an_instruction(forwarding_runtime):
    rows = [("narr", 'Mara whispered nervously, "Please don\'t go."'), ("mara", '"Please don\'t go."')]
    result = psa.analyze_scenes(make_project(), make_segments(rows), runtime_profile=psa.EXPERIMENTAL_RUNTIME_PROFILE,
                                explicit_directions={"u0": {"delivery": ["warm and slow"]}})
    narrator = by_id(result, "u0")
    assert narrator["status"] == "narrator_preserved"
    assert narrator["preservation"]["narrator"] is True and narrator["preservation"]["forwarding"] == "never"
    assert narrator["native_instruction_candidate"]["text"] is None
    for profile, state in narrator["what_if"].items():
        assert state["would_forward"] is False
    assert narrator["resolution"]["resolved"]["native_instruction"] is None
    assert not any(d["state"] == "forward_native" for d in narrator["resolution"]["dispositions"])
    # The same words in a character's mouth can be forwarded on the experimental runtime.
    mara = by_id(result, "u1")
    assert mara["what_if"][psa.EXPERIMENTAL_RUNTIME_PROFILE]["would_forward"] is True
    assert mara["what_if"][psa.EXPERIMENTAL_RUNTIME_PROFILE]["applied"] is False


def test_narrator_still_receives_assembly_pacing_advice():
    rows = [("mara", '"Well?"'), ("narr", "Tom waited."), ("tom", '"No."')]
    result = psa.analyze_scenes(make_project(), make_segments(rows))
    assert by_id(result, "u1")["boundary"]["after"]["value"] == "beat_after"
    assert by_id(result, "u1")["native_instruction_candidate"]["text"] is None


# ----------------------------------------------- explicit versus inferred


def test_explicit_source_tags_are_labelled_apart_from_inference():
    rows = [("narr", "Tom answered sharply."), ("tom", '"You will not touch it."'),
            ("mara", '"I hate this, I hate all of it."')]
    result = psa.analyze_scenes(make_project(), make_segments(rows))
    tagged = by_id(result, "u1")
    assert tagged["emotion"]["label"] == "anger" and tagged["emotion"]["basis"] == psa.BASIS_SOURCE
    assert tagged["emotion"]["confidence"] == "moderate"  # a named-role binding, not a pronoun guess
    assert {cue["binding"] for cue in tagged["tag_cues"]} == {"named_role"}
    assert tagged["delivery_phrases"][0]["basis"] == psa.BASIS_SOURCE
    inferred = by_id(result, "u2")
    assert inferred["emotion"]["label"] == "anger" and inferred["emotion"]["basis"] == psa.BASIS_INFERRED
    assert inferred["effective"]["emotion"]["basis"] == psa.BASIS_INFERRED
    for row in result["directions"]:
        assert row["human_override"] is None
        for phrase in row["delivery_phrases"]:
            assert phrase["basis"] in (psa.BASIS_SOURCE, psa.BASIS_INFERRED)


def test_pronoun_binding_is_marked_lower_confidence_and_wrong_role_is_not_bound():
    rows = [("tom", '"Leave it."'), ("narr", "She whispered gently.")]
    result = psa.analyze_scenes(make_project(), make_segments(rows))
    bound = by_id(result, "u0")
    assert [cue["binding"] for cue in bound["tag_cues"]] == ["adjacent_pronoun"]
    assert bound["emotion"]["confidence"] == "low"

    other = psa.analyze_scenes(make_project(), make_segments([("tom", '"Leave it."'),
                                                              ("narr", "Mara whispered gently."),
                                                              ("mara", '"No."')]))
    assert by_id(other, "u0")["tag_cues"] == []  # Mara's tag belongs to Mara's adjacent line only
    assert [cue["binding"] for cue in by_id(other, "u2")["tag_cues"]] == ["named_role"]


def test_same_role_introductory_pronoun_tag_binds_forward_with_source_evidence():
    result = psa.analyze_scenes(make_project(), make_segments([
        ("mara", '"Do not jinx us."'), ("narr", "She suddenly shouted,"),
        ("mara", '"What was that, anyway!"')]))
    assert by_id(result, "u0")["tag_cues"] == []
    following = by_id(result, "u2")
    assert [c["binding"] for c in following["tag_cues"]] == ["same_role_intro"]
    assert following["delivery_phrases"][0]["basis"] == psa.BASIS_SOURCE
    assert following["delivery_phrases"][0]["phrase"] == "raised voice"
    assert following["tag_cues"][0]["evidence"][0]["utterance_id"] == "u1"


def test_introductory_pronoun_between_different_speakers_stays_ambiguous():
    result = psa.analyze_scenes(make_project(), make_segments([
        ("mara", '"Leave."'), ("narr", "She suddenly shouted,"), ("tom", '"No."')]))
    assert not any(c["binding"] == "same_role_intro" for r in result["directions"] for c in r["tag_cues"])


def test_dead_and_anyway_do_not_imply_grief_or_evasion():
    result = psa.analyze_scenes(make_project(), make_segments([
        ("tom", '"I am not convinced we are not dead."'), ("mara", '"What was that, anyway?"')]))
    assert by_id(result, "u0")["emotion"]["label"] != "sadness"
    assert by_id(result, "u1")["intention"]["value"] != "deflect"
    tagged = psa.analyze_scenes(make_project(), make_segments([
        ("narr", "Tom said sadly."), ("tom", '"She is gone."')]))
    assert by_id(tagged, "u1")["emotion"]["label"] == "sadness"
    assert by_id(tagged, "u1")["emotion"]["basis"] == psa.BASIS_SOURCE


def test_human_direction_outranks_source_and_inference_and_is_validated():
    rows = [("narr", "Tom answered sharply."), ("tom", '"Never mind, it is nothing!"')]
    result = psa.analyze_scenes(
        make_project(), make_segments(rows),
        explicit_directions={"u1": {"intention": "reassure", "emotion_label": "warmth", "delivery": ["gentle", "slow"]},
                             "missing": {"intention": "explain"}, "u0": {"intention": "not-an-intention"}})
    row = by_id(result, "u1")
    assert row["effective"]["intention"] == {"value": "reassure", "basis": psa.BASIS_HUMAN}
    assert row["effective"]["emotion"]["basis"] == psa.BASIS_HUMAN
    assert row["effective"]["delivery_basis"] == psa.BASIS_HUMAN
    assert row["intention"]["value"] == "deflect" and row["intention"]["basis"] == psa.BASIS_INFERRED
    assert row["native_instruction_candidate"]["phrases"] == ["gentle", "slow"]
    assert row["resolution"]["precedence_trace"][1] == "manual_line"
    codes = {(hold["code"], hold.get("utterance_id")) for hold in result["holds"]}
    assert ("input_ineligible", "missing") in codes and ("schema_unknown", "u0") in codes


# ------------------------------------------- pacing, response, emphasis


def test_response_hints_and_boundaries():
    rows = [("mara", '"Where were you last night?"'), ("tom", '"Out."'), ("mara", '"Out where—"'),
            ("tom", '"Nowhere!"'), ("mara", '"...Fine."'), ("tom", '"Um, thanks."')]
    result = psa.analyze_scenes(make_project(), make_segments(rows))
    quick = by_id(result, "u1")
    assert quick["response"] == {**quick["response"], "relation": "answer", "responds_to": "u0",
                                 "timing": "immediate"}
    assert quick["boundary"]["before"]["value"] == "quick"
    assert by_id(result, "u2")["boundary"]["after"]["value"] == "quick"  # cut off by the next speaker
    interrupted = by_id(result, "u3")
    assert interrupted["response"]["relation"] == "interruption" and interrupted["response"]["timing"] == "immediate"
    hesitant = by_id(result, "u4")
    assert hesitant["boundary"]["before"]["value"] == "hesitant" and hesitant["response"]["timing"] == "after_beat"
    assert by_id(result, "u5")["boundary"]["before"]["value"] == "hesitant"
    assert by_id(result, "u0")["response"]["relation"] == "opens_exchange"


def test_trailing_ellipsis_is_withheld_and_rapid_exchange_is_brisk():
    rows = [("mara", '"Well, I thought..."'), ("tom", '"Yes?"'), ("mara", '"No."'), ("tom", '"Right."')]
    result = psa.analyze_scenes(make_project(), make_segments(rows))
    assert by_id(result, "u0")["boundary"]["after"]["value"] == "withheld"
    assert result["scenes"][0]["summary"]["rapid_exchange_utterance_ids"] == ["u0", "u1", "u2", "u3"]
    assert by_id(result, "u2")["tempo"]["value"] == "brisk"


def test_emphasis_comes_from_source_markup_and_capitals_with_offsets():
    text = '"I said *never*, and I meant it, NOT maybe."'
    result = psa.analyze_scenes(make_project(), make_segments([("tom", text)]))
    found = {item["text"]: item for item in by_id(result, "u0")["emphasis"]}
    assert set(found) == {"never", "NOT"}
    assert found["never"]["markup"] is True and found["NOT"]["markup"] is False
    assert all(item["basis"] == psa.BASIS_SOURCE for item in found.values())
    assert all(text[item["start"]:item["end"]] == item["text"] for item in found.values())
    shout = psa.analyze_scenes(make_project(), make_segments([("tom", '"GET OUT OF HERE NOW!"')]))
    assert by_id(shout, "u0")["emphasis"] == []  # an all-caps line is a loud line, not word emphasis
    assert by_id(shout, "u0")["emotion"]["intensity"] == "high"


def test_comic_scene_lands_the_punchline_and_dramatic_scene_holds():
    comic_rows = [("narr", "Tom grinned and told the joke."), ("tom", '"So the horse walks into a bar."'),
                  ("mara", '"Haha, and the bartender says: why the long face!"')]
    comic = psa.analyze_scenes(make_project(), make_segments(comic_rows))
    assert comic["scenes"][0]["summary"]["tone"] == "comic"
    final = by_id(comic, "u2")
    assert final["ending"]["style"] == "punchline_landing" and final["ending"]["scene_final"] is True
    punch = [item for item in final["emphasis"] if item["role"] == "punchline_final_word"]
    assert [item["text"] for item in punch] == ["face"] and punch[0]["basis"] == psa.BASIS_INFERRED
    assert final["boundary"]["after"]["value"] == "beat_after"

    drama = psa.analyze_scenes(make_project(), make_segments(
        [("narr", "Silence filled the room and tears ran down her face."), ("mara", '"He is gone."')]))
    assert drama["scenes"][0]["summary"]["tone"] == "dramatic"
    assert by_id(drama, "u1")["ending"]["style"] == "dramatic_hold"
    assert by_id(drama, "u1")["emphasis"] == []

    mixed = psa.analyze_scenes(make_project(), make_segments(
        [("narr", "He joked and giggled through the funeral and the silence."), ("mara", '"Hush."')]))
    assert mixed["scenes"][0]["summary"]["tone"] == "mixed"


def test_unknown_lines_make_no_invented_claims():
    result = psa.analyze_scenes(make_project(), make_segments([("tom", '"The meeting is at three."')]))
    row = by_id(result, "u0")
    assert row["emotion"]["label"] is None and row["emotion"]["basis"] == psa.BASIS_NONE
    assert row["intention"]["value"] == "unknown"
    assert row["delivery_phrases"] == [] and row["native_instruction_candidate"]["text"] is None
    non_latin = psa.analyze_scenes(make_project(), make_segments([("tom", "「本当に？」")]))
    assert by_id(non_latin, "u0")["ending"]["terminal"] == "question"
    assert by_id(non_latin, "u0")["word_count"] is None


# ----------------------------------------------------------- scene splitting


def test_scenes_split_on_chapter_scene_break_and_size():
    rows = [("narr", "One."), ("mara", '"Two."'), ("narr", "* * *"), ("tom", '"Three."'), ("narr", "Four.")]
    chapter2 = make_segments([("tom", '"Five."')], chapter="c2", start=5)
    result = psa.analyze_scenes(make_project(), make_segments(rows) + chapter2)
    assert [[u["utterance_id"] for u in scene["utterances"]] for scene in result["scenes"]] == [
        ["u0", "u1", "u2"], ["u3", "u4"], ["u5"]]
    assert by_id(result, "u2")["status"] == "scene_break"
    assert len({scene["scene_id"] for scene in result["scenes"]}) == 3

    many = make_segments([("tom", f'"Line {i}."') for i in range(7)])
    capped = psa.analyze_scenes(make_project(), many, max_scene_utterances=3)
    assert [len(scene["utterances"]) for scene in capped["scenes"]] == [3, 3, 1]
    with pytest.raises(psa.SceneAnalysisError):
        psa.analyze_scenes(make_project(), many, max_scene_utterances=0)


def test_scene_id_is_stable_across_prose_edits():
    segments = make_segments(DRAMA)
    edited = copy.deepcopy(segments)
    edited[2].text = '"Elsewhere."'
    first = psa.analyze_scenes(make_project(), segments)["scenes"][0]
    second = psa.analyze_scenes(make_project(), edited)["scenes"][0]
    assert first["scene_id"] == second["scene_id"] and first["analysis_key"] != second["analysis_key"]


# --------------------------------------------- existing Performance Director seam


def test_scene_intention_record_matches_the_existing_seam_and_registry_proofs():
    segments = make_segments(DRAMA)
    project = make_project()
    result = psa.analyze_scenes(project, segments)
    scene = result["scenes"][0]
    assert scene["scene_intention_reason"] is None
    plan = director.SceneIntention.model_validate(scene["scene_intention"])
    assert plan.interpretation.mode == "deterministic" and plan.author.origin == "deterministic"
    assert plan.source_revision_sha256 == SOURCE
    assert {turn.utterance_id for turn in plan.turns} <= {u.id for u in plan.utterances}
    assert all(turn.confidence == "deterministic" for turn in plan.turns)  # never "manual", never "model"

    # The plan is bound to the registry-resolved digest and to the exact prose: the same live state
    # reproduces the same plan identity, an edited line does not.
    # (performance_director.scene_proofs still compares the anchor id itself with a SHA; see node A notes.)
    again = psa.analyze_scenes(project, segments)["scenes"][0]["scene_intention"]
    assert again["cache_key"] == scene["scene_intention"]["cache_key"]
    changed = copy.deepcopy(segments)
    changed[1].text = '"Where were you, really?"'
    edited = psa.analyze_scenes(project, changed)["scenes"][0]["scene_intention"]
    assert edited["cache_key"] != scene["scene_intention"]["cache_key"]
    assert [u["text_sha256"] for u in edited["utterances"]] != [u["text_sha256"] for u in scene["scene_intention"]["utterances"]]


def test_missing_source_revision_still_analyzes_but_reports_no_plan():
    result = psa.analyze_scenes(make_project(), make_segments(DRAMA, source=None))
    scene = result["scenes"][0]
    assert scene["scene_intention"] is None and scene["scene_intention_reason"].startswith("plan_stale")
    assert by_id(result, "u2")["boundary"]["before"]["value"] == "quick"
    assert by_id(result, "u1")["resolution"]["available"] is False


def test_unattributed_line_is_held_without_an_instruction():
    rows = [("tom", '"Fine."'), (None, '"Who said that?"')]
    segments = make_segments(rows)
    segments[1].role_id = None
    result = psa.analyze_scenes(make_project(), segments)
    row = by_id(result, "u1")
    assert row["status"] == "held" and row["kind"] == "unattributed"
    assert any(hold["code"] == "attribution_unresolved" and hold["utterance_id"] == "u1" for hold in result["holds"])
    assert row["native_instruction_candidate"] is None
    assert row["resolution"]["available"] is False


# ------------------------------------------- controls and the native seam


def test_controls_requested_versus_actually_supported_default_runtime():
    result = psa.analyze_scenes(make_project(), make_segments(DRAMA))
    controls = result["controls"]
    assert controls["applied_by_analysis"] == []
    assert set(controls["requested"]) >= {"native_instruction", "emphasis", "pace_hint", "emotion_label"}
    default = controls["supported"]["faster-base-032"]
    if not default["available"]:
        pytest.skip("capability owner unavailable in this checkout")
    assert default["controls"]["emphasis"]["state"] == "unsupported"
    assert default["controls"]["emotion_label"]["state"] == "unsupported"
    assert default["controls"]["native_instruction"]["state"] in ("available_not_forwarded", "unsupported")
    assert default["controls"]["native_instruction"]["state"] != "forwarded_experimental"
    for row in result["directions"]:
        if row["resolution"].get("available"):
            assert row["resolution"]["resolved"]["native_instruction"] is None  # nothing forwards by default
            assert row["resolution"]["resolved"]["seed_salt"] == 0
    assert result["side_effects"] == {"network": False, "gpu": False, "model_inference": False,
                                      "synthesis": False, "project_written": False, "voice_changed": False,
                                      "cache_written": False}


def test_candidate_instruction_is_valid_bounded_and_forwards_only_on_the_experimental_profile(forwarding_runtime):
    rows = [("narr", "Tom answered nervously."), ("tom", '"Well... I suppose you are right!"')]
    result = psa.analyze_scenes(make_project(), make_segments(rows))
    row = by_id(result, "u1")
    candidate = row["native_instruction_candidate"]
    assert candidate["text"] == director.validate_native_instruction(candidate["text"])
    assert candidate["chars"] == len(candidate["text"]) <= director.MAX_INSTRUCTION_CHARS
    assert len(candidate["phrases"]) <= director.MAX_DELIVERY_PRIORITIES
    assert row["what_if"]["faster-base-032"]["native_instruction_state"] == "advisory"
    assert row["what_if"]["faster-base-032"]["would_forward"] is False
    assert row["what_if"][psa.EXPERIMENTAL_RUNTIME_PROFILE]["would_forward"] is True
    # The real resolution (project runtime = default) still forwards nothing.
    assert row["resolution"]["resolved"]["native_instruction"] is None
    experimental = psa.analyze_scenes(make_project(), make_segments(rows), runtime_profile=psa.EXPERIMENTAL_RUNTIME_PROFILE)
    resolved = by_id(experimental, "u1")["resolution"]
    # Even on the forwarding runtime, an inferred candidate is never resolved into a real instruction: only
    # a human correction or an approved profile habit reaches the resolver's real pass.
    assert resolved["resolved"]["native_instruction"] is None
    assert resolved["runtime_profile"] == psa.EXPERIMENTAL_RUNTIME_PROFILE
    assert resolved["precedence_trace"][0] == "source_voice_safeguards"


def test_required_forward_on_an_unsupported_runtime_is_a_hold_not_a_silent_advisory(forwarding_runtime):
    result = psa.analyze_scenes(make_project(), make_segments([("tom", '"Fine."')]),
                                explicit_directions={"u0": {"delivery": ["dry"], "require_forward": True}})
    resolution = by_id(result, "u0")["resolution"]
    assert any(d["state"] == "hold" for d in resolution["dispositions"])
    assert resolution["resolved"]["native_instruction"] is None


def test_dossier_habits_are_evidence_only_spoiler_safe_and_never_approved():
    fact = lambda fid, value, after: CharacterFact(  # noqa: E731
        id=fid, field="cadence", value=value, level="EXPLICIT", confidence=0.9, source="fixture",
        available_after_chapter=after)
    roles = make_roles()
    project = make_project(roles)
    project.character_cast.dossiers["tom"] = CharacterDossier(
        role_id="tom", canonical_name="Tom", revision="r1",
        facts=[fact("f1", "dry and unhurried", 0), fact("f2", "later-reveal rasp", 9)])
    rows = [("narr", "Tom answered sharply."), ("tom", '"You will not touch it."')]
    result = psa.analyze_scenes(project, make_segments(rows))
    tom = next(role for role in result["roles"] if role["role_id"] == "tom")
    assert [habit["value"] for habit in tom["habits"]] == ["dry and unhurried"]  # chapter-2 spoiler excluded
    assert tom["profile_status"] == "compiled_not_approved"
    candidate = by_id(result, "u1")["native_instruction_candidate"]
    assert candidate["habit_included"] is True and "dry and unhurried" in candidate["text"]
    assert "later-reveal rasp" not in json.dumps(result)
    assert all(row["approved"] is False for row in result["directions"])


# ------------------------------------------------- optional code and safety


def test_missing_optional_experimental_code_is_reported_not_raised(monkeypatch):
    def unavailable() -> psa._Seams:
        return psa._Seams(reasons={"performance_director": "ModuleNotFoundError: gone",
                                   "provider_capabilities": "ModuleNotFoundError: gone"})

    monkeypatch.setattr(psa, "_load_seams", unavailable)
    result = psa.analyze_scenes(make_project(), make_segments(DRAMA))
    assert result["integrations"]["performance_director"] == {"available": False, "reason": "ModuleNotFoundError: gone"}
    assert result["investigation"]["available"] is False
    row = by_id(result, "u2")
    assert row["boundary"]["before"]["value"] == "quick"  # text analysis is unaffected
    assert row["resolution"] == {"available": False, "reason": "ModuleNotFoundError: gone"}
    assert result["scenes"][0]["scene_intention"] is None
    assert row["native_instruction_candidate"]["text"] is None


def test_analysis_opens_no_network_and_importing_it_pulls_no_runtime(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("scene analysis must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    psa.analyze_scenes(make_project(), make_segments(DRAMA))

    watchers = ("torch", "dubbing_service.app.providers.tts", "dubbing_service.app.audiobook.renderer")
    code = ("import sys;import dubbing_service.app.audiobook.performance_scene_analysis;"
            f"print([name for name in {watchers!r} if name in sys.modules])")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(REPO_ROOT),
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_investigation_reports_capability_facts_without_claiming_quality():
    result = psa.analyze_scenes(make_project(), make_segments(DRAMA))
    investigation = result["investigation"]
    if not investigation["available"]:
        pytest.skip("capability owner unavailable in this checkout")
    assert investigation["quality_evidence"] is None
    default = investigation["profiles"]["faster-base-032"]
    experimental = investigation["profiles"][psa.EXPERIMENTAL_RUNTIME_PROFILE]
    if default["available"]:
        assert default["native_instruction"]["state"] != "forwarded_experimental"
    if experimental["available"]:
        assert experimental["capability_sha256"] != default["capability_sha256"]
    assert "unmeasured" in investigation["note"]


def test_human_correction_is_the_only_path_to_a_resolved_instruction(forwarding_runtime):
    rows = [("narr", "Tom answered nervously."), ("tom", '"Well... I suppose you are right!"')]
    result = psa.analyze_scenes(make_project(), make_segments(rows), runtime_profile=psa.EXPERIMENTAL_RUNTIME_PROFILE,
                                explicit_directions={"u1": {"delivery": ["dry", "unhurried"]}})
    resolution = by_id(result, "u1")["resolution"]
    assert resolution["resolved"]["native_instruction"] == "dry; unhurried."
    assert [d["state"] for d in resolution["dispositions"] if d["control"] == "native_instruction"] == ["forward_native"]
    assert by_id(result, "u1")["approved"] is False  # a correction is a request, not an approval


# ------------------------------------------------ source revision registry (node A)


def scene_of(result: dict) -> dict:
    return result["scenes"][0]


def test_anchor_id_is_resolved_through_the_registry_and_is_never_taken_for_a_hash():
    result = psa.analyze_scenes(make_project(), make_segments(DRAMA))
    scene = scene_of(result)
    assert scene["scene_intention_reason"] is None
    assert scene["source"]["source_revision_ids"] == [SOURCE_ID]
    assert scene["source"]["source_revision_sha256"] == SOURCE != SOURCE_ID
    assert scene["scene_intention"]["source_revision_sha256"] == SOURCE
    assert result["source"] == {"registry_status": "ok", "registry_detail": "",
                                "revisions": [{"id": SOURCE_ID, "sha256": SOURCE}]}


def test_two_ids_for_identical_source_bytes_still_resolve_to_one_proof():
    registry = make_registry() + make_registry(revision_id="src_alias0002")
    segments = make_segments(DRAMA)
    segments[2].source_anchor = SourceAnchor(source_revision_id="src_alias0002")
    result = psa.analyze_scenes(make_project(source_revisions=registry), segments)
    assert scene_of(result)["scene_intention"]["source_revision_sha256"] == SOURCE


def test_a_hash_shaped_anchor_id_is_not_a_bypass_without_a_registry():
    segments = make_segments(DRAMA, source=SOURCE)  # the pre-repair shape: the anchor "is" the digest
    result = psa.analyze_scenes(make_project(source_revisions=[]), segments)
    scene = scene_of(result)
    assert scene["scene_intention"] is None
    assert scene["scene_intention_reason"].startswith("plan_stale: source_registry_missing")
    assert scene["source"]["source_revision_sha256"] is None
    assert result["source"]["registry_status"] == "source_registry_missing"
    assert {hold["code"] for hold in scene["holds"]} == {"source_registry_missing"}


def test_hash_shaped_anchor_id_must_still_be_in_the_registry():
    segments = make_segments(DRAMA, source=SOURCE)
    result = psa.analyze_scenes(make_project(), segments)  # registry holds SOURCE_ID only
    assert scene_of(result)["scene_intention"] is None
    assert scene_of(result)["scene_intention_reason"].startswith("plan_stale: source_revision_unknown")


def test_unknown_anchor_id_fails_closed_but_the_prose_is_still_analysed():
    result = psa.analyze_scenes(make_project(), make_segments(DRAMA, source="src_other"))
    scene = scene_of(result)
    assert scene["scene_intention"] is None
    assert scene["scene_intention_reason"].startswith("plan_stale: source_revision_unknown")
    assert "src_other" in scene["scene_intention_reason"]
    assert by_id(result, "u2")["boundary"]["before"]["value"] == "quick"
    assert by_id(result, "u1")["resolution"]["available"] is False


@pytest.mark.parametrize("registry, code", [
    (lambda: make_registry() + make_registry(digest="b" * 64), "source_registry_duplicate"),
    (lambda: make_registry() + make_registry(), "source_registry_duplicate"),
    (lambda: make_registry(digest="A" * 64), "source_registry_malformed"),
    (lambda: make_registry(digest="a" * 63), "source_registry_malformed"),
    (lambda: make_registry(digest="g" * 64), "source_registry_malformed"),
    (lambda: make_registry(revision_id=""), "source_registry_malformed"),
    (lambda: [], "source_registry_missing"),
])
def test_registry_defects_fail_closed(registry, code):
    result = psa.analyze_scenes(make_project(source_revisions=registry()), make_segments(DRAMA))
    scene = scene_of(result)
    assert scene["scene_intention"] is None and scene["source"]["source_revision_sha256"] is None
    assert scene["scene_intention_reason"].startswith(f"plan_stale: {code}")
    assert result["source"]["registry_status"] == code and result["source"]["revisions"] == []


def test_a_registry_defect_elsewhere_in_the_registry_still_fails_closed():
    registry = make_registry() + make_registry(digest="not-a-digest", revision_id="src_unused")
    result = psa.analyze_scenes(make_project(source_revisions=registry), make_segments(DRAMA))
    assert scene_of(result)["scene_intention"] is None
    assert result["source"]["registry_status"] == "source_registry_malformed"


def test_missing_mixed_or_partial_anchors_are_typed_not_guessed():
    mixed_registry = make_registry() + make_registry(digest="b" * 64, revision_id="src_second0002")
    segments = make_segments(DRAMA)
    segments[3].source_anchor = SourceAnchor(source_revision_id="src_second0002")
    mixed = psa.analyze_scenes(make_project(source_revisions=mixed_registry), segments)
    assert scene_of(mixed)["scene_intention_reason"].startswith("plan_stale: source_revision_mixed")

    partial = make_segments(DRAMA)
    partial[2].source_anchor = None
    result = psa.analyze_scenes(make_project(), partial)
    assert scene_of(result)["scene_intention_reason"].startswith("plan_stale: source_anchor_missing")
    assert "u2" in scene_of(result)["scene_intention_reason"]


def test_editing_the_registered_digest_invalidates_the_cache_and_leaves_the_old_entry(tmp_path):
    segments = make_segments(DRAMA)
    first = psa.analyze_scenes(make_project(), segments, cache_root=tmp_path)
    old_key = scene_of(first)["analysis_key"]
    old_bytes = (tmp_path / psa.CACHE_DIRECTORY / f"{old_key}.json").read_bytes()

    edited = psa.analyze_scenes(make_project(source_revisions=make_registry(digest="b" * 64)), segments,
                                cache_root=tmp_path)
    assert scene_of(edited)["analysis_key"] != old_key
    assert edited["cache"]["scenes"][0]["status"] == "miss_written"
    assert scene_of(edited)["scene_intention"]["source_revision_sha256"] == "b" * 64
    assert scene_of(edited)["scene_id"] == scene_of(first)["scene_id"]
    assert (tmp_path / psa.CACHE_DIRECTORY / f"{old_key}.json").read_bytes() == old_bytes

    again = psa.analyze_scenes(make_project(), segments, cache_root=tmp_path)
    assert again["cache"]["scenes"][0]["status"] == "hit"


def test_registry_id_is_provenance_while_the_plan_stays_bound_to_the_same_digest(tmp_path):
    first = psa.analyze_scenes(make_project(), make_segments(DRAMA), cache_root=tmp_path)
    renamed = psa.analyze_scenes(make_project(source_revisions=make_registry(revision_id="src_renamed0009")),
                                 make_segments(DRAMA, source="src_renamed0009"), cache_root=tmp_path)
    assert scene_of(renamed)["scene_intention"]["source_revision_sha256"] == SOURCE
    assert scene_of(renamed)["analysis_key"] != scene_of(first)["analysis_key"]


def test_no_source_or_voice_change_means_a_cache_hit_and_identical_interpretation(tmp_path):
    segments = make_segments(DRAMA)
    first = psa.analyze_scenes(make_project(), segments, cache_root=tmp_path)
    second = psa.analyze_scenes(make_project(), segments, cache_root=tmp_path)
    assert second["cache"]["scenes"][0]["status"] == "hit"
    assert second["directions"] == first["directions"]
    assert [(row["summary"], row["turns"]) for row in second["scenes"]] ==         [(row["summary"], row["turns"]) for row in first["scenes"]]


def test_literal_utf8_prose_is_hashed_cached_and_cited_without_escaping(tmp_path):
    rows = [("narr", "Tom didn’t answer… he stared at the 門 for a long moment."),
            ("mara", "“Où étais-tu ? 你去哪儿了？”"),
            ("tom", "“I couldn’t say.”")]
    segments = make_segments(rows)
    result = psa.analyze_scenes(make_project(), segments, cache_root=tmp_path)
    for segment in segments:
        row = by_id(result, segment.id)
        assert row["text_sha256"] == hashlib.sha256(segment.text.encode("utf-8")).hexdigest()
    for item in evidence_items(result["directions"]):
        prose = next(s.text for s in segments if s.id == item["utterance_id"])
        assert prose[item["start"]:item["end"]] == item["text"]
    stored = (tmp_path / psa.CACHE_DIRECTORY / f"{scene_of(result)['analysis_key']}.json").read_bytes()
    assert "didn’t answer…".encode("utf-8") in stored and "你去哪儿了".encode("utf-8") in stored
    assert b"\\u2019" not in stored and b"\\u4f60" not in stored
    assert [u["text"] for u in scene_of(result)["utterances"]] == [s.text for s in segments]


# ------------------------------------------------------- scene carry (node A)


EXCHANGE = [
    ("mara", '"Come closer."'),
    ("narr", "Mara whispered softly."),
    ("tom", '"Why?"'),
    ("mara", '"Because they are listening."'),
    ("tom", '"Who is?"'),
    ("mara", '"Everyone."'),
]


def test_a_source_cue_persists_across_the_whole_exchange_with_its_evidence():
    result = psa.analyze_scenes(make_project(), make_segments(EXCHANGE))
    cue = by_id(result, "u0")["tag_cues"][0]
    assert cue["phrase"] == "soft" and cue["binding"] == "named_role"
    assert by_id(result, "u0")["carried_cues"] == []

    carried = {uid: by_id(result, uid)["carried_cues"] for uid in ("u2", "u3", "u4", "u5")}
    assert [(c["scope"], c["turn_distance"]) for c in carried["u3"]] == [("speaker_continuation", 1)]
    assert [(c["scope"], c["turn_distance"]) for c in carried["u5"]] == [("speaker_continuation", 2)]
    assert [(c["scope"], c["turn_distance"]) for c in carried["u2"]] == [("exchange_register", 1)]
    assert [(c["scope"], c["turn_distance"]) for c in carried["u4"]] == [("exchange_register", 2)]
    for items in carried.values():
        for item in items:
            assert item["basis"] == psa.BASIS_INFERRED and item["confidence"] == "low"
            assert item["from_utterance_id"] == "u0"
            assert [(e["utterance_id"], e["text"]) for e in item["evidence"]] == [("u1", "whispered"),
                                                                                   ("u1", "softly")]
    assert {by_id(result, i)["exchange"]["id"] for i in ("u0", "u2", "u3", "u4", "u5")} == {"x0"}
    phrase = by_id(result, "u3")["delivery_phrases"][0]
    assert phrase["phrase"] == "soft" and phrase["source"] == "carried_cue" and phrase["basis"] == psa.BASIS_INFERRED
    assert by_id(result, "u1")["carried_cues"] == [] and by_id(result, "u1")["exchange"] is None


def test_carry_never_relabels_inference_as_explicit_source():
    result = psa.analyze_scenes(make_project(), make_segments(EXCHANGE))
    for row in result["directions"]:
        for item in row["carried_cues"]:
            assert item["basis"] != psa.BASIS_SOURCE
        for phrase in row["delivery_phrases"]:
            if phrase["source"] == "carried_cue":
                assert phrase["basis"] == psa.BASIS_INFERRED


def test_carried_emotion_is_low_confidence_inferred_and_points_at_the_author_cue():
    rows = [("mara", '"I can\'t."'), ("narr", "Mara stammered."), ("tom", '"Try."'),
            ("mara", '"I really can\'t."')]
    result = psa.analyze_scenes(make_project(), make_segments(rows))
    own = by_id(result, "u0")["emotion"]
    assert own["label"] == "fear" and own["basis"] == psa.BASIS_SOURCE
    later = by_id(result, "u3")["emotion"]
    assert later["label"] == "fear" and later["basis"] == psa.BASIS_INFERRED and later["confidence"] == "low"
    assert later["carried_from"] == "u0" and later["evidence"][0]["text"] == "stammered"
    # The partner is not handed the speaker's emotion.
    assert by_id(result, "u2")["emotion"]["label"] is None


def test_a_new_author_cue_ends_the_carry_and_a_loud_line_clashes_with_a_hush():
    rows = EXCHANGE[:4] + [("narr", "Mara shouted."), ("mara", '"Everyone!"'), ("tom", '"Quiet."'),
                           ("mara", '"Never."')]
    result = psa.analyze_scenes(make_project(), make_segments(rows))
    shouted = by_id(result, "u5")
    assert shouted["carried_cues"] == []
    assert shouted["delivery_phrases"][0]["source"] == "speech_tag"
    assert shouted["delivery_phrases"][0]["phrase"] == "raised voice"

    clash = psa.analyze_scenes(make_project(), make_segments(EXCHANGE[:4] + [("tom", '"Hm."'),
                                                                             ("mara", '"EVERYONE!"')]))
    assert by_id(clash, "u5")["carried_cues"] == []


def test_carry_is_bounded_by_turns_pronoun_guesses_and_exchange_breaks():
    turns = [("mara", '"Closer."'), ("narr", "Mara whispered softly.")]
    for _ in range(5):
        turns += [("tom", '"Go on."'), ("mara", '"Not yet."')]
    result = psa.analyze_scenes(make_project(), make_segments(turns))
    carried = [any(c["scope"] == "speaker_continuation" for c in by_id(result, f"u{i}")["carried_cues"])
               for i in range(2, len(turns)) if turns[i][0] == "mara"]
    assert carried == [True, True, True, False, False]  # psa.MAX_CARRY_TURNS == 3

    pronoun = psa.analyze_scenes(make_project(), make_segments([("mara", '"Closer."'),
                                                                ("narr", "She whispered softly."),
                                                                ("narr", "The lamp guttered."),
                                                                ("tom", '"Why?"'), ("mara", '"Listen."')]))
    assert by_id(pronoun, "u0")["tag_cues"][0]["binding"] == "adjacent_pronoun"
    assert by_id(pronoun, "u4")["carried_cues"] == [] and by_id(pronoun, "u3")["carried_cues"] == []

    gap = [("narr", "The hall emptied."), ("narr", "Lamps burned low."), ("narr", "Rain fell outside.")]
    broken = psa.analyze_scenes(make_project(), make_segments(EXCHANGE[:4] + gap + [("mara", '"Now."')]))
    assert by_id(broken, "u0")["exchange"]["id"] != by_id(broken, "u7")["exchange"]["id"]
    assert by_id(broken, "u7")["carried_cues"] == []


def test_scene_break_ends_an_exchange_and_the_carry():
    rows = EXCHANGE[:4] + [("narr", "* * *"), ("mara", '"Morning."')]
    result = psa.analyze_scenes(make_project(), make_segments(rows))
    assert len(result["scenes"]) == 2
    assert by_id(result, "u5")["carried_cues"] == []


def test_unknown_stays_unknown_when_nothing_is_cued():
    rows = [("mara", '"Pass the salt."'), ("tom", '"Here."'), ("mara", '"Thanks."')]
    result = psa.analyze_scenes(make_project(), make_segments(rows))
    for row in result["directions"]:
        assert row["carried_cues"] == []
    assert by_id(result, "u0")["intention"] == {"value": "unknown", "basis": "none", "confidence": None,
                                                "evidence": []}


def test_character_habit_stays_separate_from_scene_phrases():
    result = psa.analyze_scenes(make_project(), make_segments(EXCHANGE))
    candidate = by_id(result, "u3")["native_instruction_candidate"]
    assert candidate["scene_phrases"][0] == "soft" and "habit_phrase" in candidate
    narrator = by_id(result, "u1")["native_instruction_candidate"]
    assert narrator["text"] is None and narrator["habit_phrase"] is None
