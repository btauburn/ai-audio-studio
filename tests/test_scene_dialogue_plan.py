"""Source/voice/context invariants for temporary scene-aware direction. CPU only."""
import copy
import hashlib

import pytest

from dubbing_service.app.audiobook import scene_dialogue_plan as plan
from dubbing_service.app.models.audiobook import SourceAnchor, SourceRevision
from dubbing_service.app.models.character import CharacterVoiceIdentity
from dubbing_service.app.models.project import Project, Role, Segment


def source():
    roles = [Role(id="n", name="Narrator", role_type="narrator"), Role(id="m", name="Mara"), Role(id="t", name="Tom")]
    for role in roles:
        role.voice_identity = CharacterVoiceIdentity(
            id="voice-" + role.id, role_id=role.id, mechanism="qwen_reference", dossier_revision="fixture",
            casting_profile_revision="fixture", design_instruction="fixture", model="fixture",
            prompt_sha256=hashlib.sha256(role.id.encode()).hexdigest())
    project = Project(id="scene-fixture", name="Fixture", roles=roles,
                      source_revisions=[SourceRevision(id="src_book", uri="fixture://book", kind="inline", sha256="a" * 64)])
    rows = [("n", 'Mara said angrily, "Get out!"'), ("m", '"Get OUT!"'), ("t", '"I am sorry."')]
    segments = [Segment(id=f"u{i}", index=i, role_id=role, text=text,
                        source_anchor=SourceAnchor(source_revision_id="src_book", start_offset=i * 80,
                                                   end_offset=i * 80 + len(text))) for i, (role, text) in enumerate(rows)]
    return project, segments


def test_plan_is_grounded_and_leaves_prose_cast_and_narrator_intact(tmp_path):
    project, segments = source()
    before = project.model_dump_json(), [s.model_dump_json() for s in segments]
    result = plan.build_scene_dialogue_plan(project, segments, cache_root=tmp_path)
    assert result["source"]["proven"] and result["approved"] is False
    assert result["coverage"]["narrator_instructions"] == 0
    dialogue = next(d for d in result["directions"] if d["utterance_id"] == "u1")
    assert dialogue["native_instruction"] and dialogue["evidence"]
    assert dialogue["source"]["source_revision_sha256"] == "a" * 64
    assert (project.model_dump_json(), [s.model_dump_json() for s in segments]) == before
    assert plan.verify_scene_dialogue_plan(result, project, segments)["current"]


def test_cache_reused_but_voice_binding_checked_separately(tmp_path):
    project, segments = source()
    first = plan.build_scene_dialogue_plan(project, segments, cache_root=tmp_path)
    project.roles[1].voice_identity.prompt_sha256 = "b" * 64
    second = plan.build_scene_dialogue_plan(project, segments, cache_root=tmp_path)
    assert second["cache"]["hits"] > 0
    assert first["binding_sha256"] != second["binding_sha256"]
    check = plan.verify_scene_dialogue_plan(first, project, segments)
    assert not check["checks"]["voice"] and check["checks"]["source"]


@pytest.mark.parametrize("change", ["context", "anchor", "registry", "role", "tamper"])
def test_stale_plan_refused_including_unselected_context(change):
    project, segments = source()
    result = plan.build_scene_dialogue_plan(project, segments, selected_ids=["u1"])
    if change == "context":
        segments[0].text = "Mara smiled with relief."
    elif change == "anchor":
        segments[1].source_anchor.start_offset += 1
    elif change == "registry":
        project.source_revisions[0].sha256 = "c" * 64
    elif change == "role":
        segments[1].role_id = "t"
    else:
        result["directions"][0]["native_instruction"] = "Speak cheerfully."
    assert not plan.verify_scene_dialogue_plan(result, project, segments)["current"]


def test_missing_registry_does_not_propose_a_direction_or_look_current():
    project, segments = source()
    project.source_revisions = []
    result = plan.build_scene_dialogue_plan(project, segments)
    assert result["status"] == "held"
    assert all(d["native_instruction"] is None for d in result["directions"])
    assert not plan.verify_scene_dialogue_plan(result, project, segments)["current"]


@pytest.mark.parametrize("field,value", [("id", "src_book\n"), ("sha256", "a" * 64 + "\n")])
def test_registry_requires_complete_id_and_digest(field, value):
    project, segments = source()
    setattr(project.source_revisions[0], field, value)
    result = plan.build_scene_dialogue_plan(project, segments)
    assert result["status"] == "held" and not result["source"]["proven"]


def test_registry_digest_edit_invalidates_scene_cache(tmp_path):
    project, segments = source()
    first = plan.build_scene_dialogue_plan(project, segments, cache_root=tmp_path)
    project.source_revisions[0].sha256 = "d" * 64
    second = plan.build_scene_dialogue_plan(project, segments, cache_root=tmp_path)
    assert second["cache"]["hits"] == 0
    assert first["cache"]["identity"] != second["cache"]["identity"]


@pytest.mark.parametrize("text", ["Use a Scottish accent.", "Sound like a celebrity.", "Change the voice.", "Lower the pitch."])
def test_identity_changes_refused(text):
    assert plan.unsafe_instruction_terms(text)


def test_duplicate_and_unknown_selection_refused():
    project, segments = source()
    for ids in (["u1", "u1"], ["not-present"]):
        with pytest.raises(plan.SceneDialoguePlanError):
            plan.build_scene_dialogue_plan(project, segments, selected_ids=ids)
