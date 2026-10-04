"""Versioned casting audition regressions using only local signal fixtures."""
import json

import pytest

from dubbing_service.app.audiobook import audition_scripts, voice_lab_audition as audition
from dubbing_service.app.audiobook.fingerprint import file_sha256, voice_revision_for_role
from dubbing_service.app.core.config import safe_project_root
from dubbing_service.app.models.project import Role
from dubbing_service.app.providers.tts import TTSResult
from dubbing_service.app.storage.projects import create_project, load_project, save_project
from dubbing_service.app.storage.segments import load_segments
from tests.audiobook_helpers import anchored_candidate, make_tone_wav, use_projects_root


class SignalProvider:
    def __init__(self):
        self.designs = []
        self.calls = []

    def design_character_voice(self, project_id, identity_id, instruction, text, seed, conditioning=None):
        self.designs.append((project_id, identity_id))
        self.conditioning = conditioning
        root = safe_project_root(project_id)
        reference = make_tone_wav(root / "signals/reference.wav", seconds=1)
        prompt = root / "signals/prompt.npz"
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_bytes(b"LOCAL-TEST-CONDITIONING")
        return {"mechanism": "qwen_reference", "backend": "qwen3tts",
                "model": "LOCAL-TEST-SIGNAL-NOT-TTS", "runtime_version": "test-only",
                "reference_audio": reference.relative_to(root).as_posix(),
                "reference_audio_sha256": file_sha256(reference), "reference_text": text,
                "prompt_artifact": prompt.relative_to(root).as_posix(),
                "prompt_sha256": file_sha256(prompt),
                "provenance": {"origin": "generated_original_voice", "fixture_only": True}}

    def synthesize_segment(self, project_id, role, segment):
        self.calls.append((project_id, segment.id))
        path = make_tone_wav(safe_project_root(project_id) / f"signals/take-{len(self.calls)}.wav", seconds=10)
        return TTSResult(success=True, audio_path=str(path), raw={
            "identity_id": role.voice_identity.id,
            "prompt_sha256": role.voice_identity.prompt_sha256,
            "execution_profile": {"model_id": "LOCAL-TEST-SIGNAL-NOT-TTS", "fixture_only": True}})


@pytest.fixture
def casting_project(monkeypatch, tmp_path):
    use_projects_root(monkeypatch, tmp_path)
    project = create_project("Casting audition fixture")
    role = Role(id="reader", name="Reader", tts_backend="qwen3tts_cv",
                qwen3_speaker="ryan", qwen3_language="English")
    project.roles = [role]
    project.voice_lab.shortlists[role.id] = {"matches": [anchored_candidate()]}
    save_project(project)
    provider = SignalProvider()
    monkeypatch.setattr(audition, "ensure_backend_ready", lambda *a, **kw: {"fixture_only": True})
    from dubbing_service.app.audiobook import renderer
    monkeypatch.setattr(renderer.TTSProviderFactory, "create", lambda backend: provider)
    return project, role, provider


def run(project, role, provider, request_id, **options):
    return audition.audition(project.id, role_id=role.id,
        candidate_id=options.pop("candidate_id", "original"),
        expected_identity_id=voice_revision_for_role(role), request_id=request_id,
        design_provider=provider, **options)


def test_versioned_scripts_are_exact_section_text_not_spoken_labels():
    long_script = audition_scripts.script("casting-60-v1")
    short_script = audition_scripts.script("casting-30-v1")
    assert len(long_script["sections"]) == 8
    assert len(short_script["sections"]) == 6
    for version in (long_script, short_script):
        for section in version["sections"]:
            assert section["spoken_text"].strip()
            assert section["label"] not in section["spoken_text"]
    assert short_script["sections"][0]["spoken_text"] == "Your notebook is on the table. I kept it dry."
    assert long_script["duration_window_seconds"] == [45, 80]
    assert short_script["duration_window_seconds"] == [20, 45]
    assert all(section["delivery_instruction"] for section in long_script["sections"])


def test_anchored_original_candidate_renders_ordered_take_sections(casting_project):
    project, role, provider = casting_project
    result = run(project, role, provider, "long-script", script_version="casting-60-v1")
    assert result["status"] == "succeeded", result
    assert len(provider.designs) == 1 and len(provider.calls) == 8
    assert result["script_version"] == "casting-60-v1"
    assert result["duration_seconds"] == 80
    assert result["duration_outlier"] is False
    assert result["duration_status"] == "measured_within_target"
    record = load_project(project.id).voice_lab.voices[result["key"]]
    child = load_project(record["audition_project_id"])
    rows = load_segments(child.id)
    assert len(rows) == 8
    assert len({next(t for t in row.takes if t.id == row.selected_take_id).fingerprint_components["voice_revision"]
                for row in rows}) == 1
    assert len({row.role_id for row in rows}) == 1
    assert [section["id"] for section in record["sections"]] == [
        section["id"] for section in audition_scripts.script("casting-60-v1")["sections"]]
    assert record["sections"][0]["start_seconds"] == 0
    assert record["sections"][-1]["end_seconds"] == 80
    assert {section["control_status"] for section in record["sections"]} == {"unsupported"}
    assert all(row.emotion_text is None for row in rows)
    proof = json.loads((safe_project_root(project.id) / record["details"]["take_proof_file"]).read_text())
    assert len(proof["takes"]) == 8


def test_reuse_clones_base_identity_without_redesign_and_new_script_has_distinct_media_key(casting_project):
    project, role, provider = casting_project
    first = run(project, role, provider, "identity-source")
    assert first["status"] == "succeeded", first
    source = load_project(project.id).voice_lab.voices[first["key"]]
    reused = run(project, role, provider, "short-script", reuse_key=first["key"],
                 script_version="casting-30-v1")
    assert reused["status"] == "succeeded", reused
    assert reused["key"] != first["key"]
    assert reused["duration_seconds"] == 60 and reused["duration_outlier"] is True
    assert reused["duration_status"] == "measured_outlier"
    assert reused["audio_sha256"]
    assert len(provider.designs) == 1 and len(provider.calls) == 7
    copied = load_project(project.id).voice_lab.voices[reused["key"]]
    child = load_project(copied["audition_project_id"])
    assert child.roles[0].voice_identity.id == source["identity"]["id"]
    assert child.roles[0].voice_identity.prompt_sha256 == source["identity"]["prompt_sha256"]
    assert copied["script_version"] == "casting-30-v1"
    assert len(copied["sections"]) == 6


def test_versioned_audition_quota_is_durable_per_role_not_projectwide(casting_project):
    project, role, provider = casting_project
    latest = load_project(project.id)
    roles = [role]
    for index in range(2):
        extra = Role(id=f"reader-{index + 2}", name=f"Reader {index + 2}",
                     tts_backend="qwen3tts_cv", qwen3_speaker="ryan", qwen3_language="English")
        latest.roles.append(extra)
        latest.voice_lab.shortlists[extra.id] = {"matches": [{
            "candidate_id": "original", "original_casting": True,
            "creative_direction": "An original, natural voice."}]}
        roles.append(extra)
    save_project(latest)
    schedule = lambda *args, **kwargs: None
    for current_role in roles:
        for proposal in range(5):
            latest = load_project(project.id)
            latest.voice_lab.shortlists[current_role.id]["matches"][0]["creative_direction"] = f"Direction {proposal}"
            save_project(latest)
            result = audition.audition(project.id, role_id=current_role.id, candidate_id="original",
                expected_identity_id=voice_revision_for_role(current_role), request_id=f"{current_role.id}-{proposal}",
                schedule=schedule, script_version="casting-60-v1")
            assert result["status"] == "queued"
    latest = load_project(project.id)
    latest.voice_lab.shortlists[role.id]["matches"][0]["creative_direction"] = "Direction 5"
    save_project(latest)
    sixth = audition.audition(project.id, role_id=role.id, candidate_id="original",
        expected_identity_id=voice_revision_for_role(role), request_id="reader-1-5", schedule=schedule,
        script_version="casting-60-v1")
    assert sixth["status"] == "queued"
    latest = load_project(project.id)
    latest.voice_lab.shortlists[role.id]["matches"][0]["creative_direction"] = "Direction 6"
    save_project(latest)
    with pytest.raises(audition.AuditionError, match="six-proposal audition"):
        audition.audition(project.id, role_id=role.id, candidate_id="original",
            expected_identity_id=voice_revision_for_role(role), request_id="reader-1-6", schedule=schedule,
            script_version="casting-60-v1")
    assert len(load_project(project.id).metadata["voice_lab_auditions"]["intents"]) == 16


def test_base_reuse_rejects_customvoice_delivery_direction_without_silently_dropping_it(casting_project):
    project, role, provider = casting_project
    first = run(project, role, provider, "identity-source")
    assert first["status"] == "succeeded", first
    latest = load_project(project.id)
    latest.voice_lab.shortlists[role.id]["matches"].append({
        "candidate_id": "calmer", "original_casting": True,
        "delivery_instruction": "Speak more calmly."})
    save_project(latest)
    held = run(project, role, provider, "unsupported-direction", candidate_id="calmer",
               reuse_key=first["key"], script_version="casting-30-v1")
    assert held["status"] == "interrupted"
    assert "cannot apply" in held["reason"]
    assert len(provider.designs) == 1 and len(provider.calls) == 1


def test_customvoice_reuse_applies_performance_direction_without_changing_speaker(casting_project, monkeypatch):
    project, role, provider = casting_project
    from dubbing_service.app.models.enums import EmotionMode
    from dubbing_service.app.models.character import CharacterVoiceIdentity

    latest = load_project(project.id)
    reusable_role = latest.roles[0].model_copy(deep=True)
    reusable_role.style_instruction = "Natural speech."
    reusable_role.default_emotion_mode = EmotionMode.TEXT
    identity = CharacterVoiceIdentity(id="builtin-reader-v1", role_id=role.id,
        mechanism="builtin", dossier_revision="", casting_profile_revision="",
        design_instruction="Original CustomVoice speaker", builtin_speaker="ryan",
        backend="qwen3tts_cv", model="LOCAL-TEST-CUSTOMVOICE",
        provenance={"fixture_only": True})
    reusable_role.voice_identity = identity
    version = voice_revision_for_role(reusable_role)
    settings_fields = ("speaker_audio", "speaker_voice_id", "tts_backend", "qwen3_speaker",
        "qwen3_language", "voice_description", "style_instruction", "default_emotion_mode",
        "default_emotion_text", "default_emotion_vector", "emotion_alpha")
    snapshot = {"schema": "voice-lab-builtin-role-v1", "source_role_id": role.id,
        "source_voice_revision": version,
        "settings": {field: reusable_role.model_dump(mode="json")[field] for field in settings_fields}}
    root = safe_project_root(project.id)
    sample = make_tone_wav(root / "library/reusable.wav", seconds=1)
    reuse_key = "v_" + "1" * 28
    latest.voice_lab.voices[reuse_key] = {"key": reuse_key, "voice_id": "builtin_qwen3tts_cv_ryan",
        "version": version, "name": "Ryan", "identity": identity.model_dump(mode="json"), "builtin_speaker": "ryan",
        "builtin_snapshot": snapshot, "audio_file": sample.relative_to(root).as_posix(),
        "audio_sha256": file_sha256(sample), "technical_qa": "passed", "human_review": "pending"}
    latest.voice_lab.shortlists[role.id]["matches"].append({"candidate_id": "directed",
        "original_casting": True, "delivery_instruction": "Speak with restrained urgency."})
    save_project(latest)

    intent_hash, binding = audition._bound_intent(latest, role.id, "directed",
        script_version="casting-30-v1", reuse_key=reuse_key)
    child = create_project("CustomVoice reuse child")
    child.roles = [role.model_copy(deep=True)]
    save_project(child)
    before_revision = voice_revision_for_role(child.roles[0])
    audition._clone_reusable_voice(project.id, child, {"reuse_key": reuse_key,
        "role_id": role.id, "binding": binding})
    child = load_project(child.id)
    assert child.roles[0].qwen3_speaker == "ryan"
    assert child.roles[0].style_instruction == "Natural speech. Speak with restrained urgency."
    assert child.roles[0].voice_identity.id == identity.id
    assert voice_revision_for_role(child.roles[0]) != before_revision

    from dubbing_service.app.audiobook import renderer
    from dubbing_service.app.providers.tts import TTSResult

    class BuiltinSignal:
        calls = []
        def synthesize_segment(self, project_id, role, segment):
            self.calls.append((project_id, segment.id, role.qwen3_speaker, segment.emotion_text))
            output = make_tone_wav(safe_project_root(project_id) / f"signals/cv-{len(self.calls)}.wav", seconds=4)
            return TTSResult(success=True, audio_path=str(output), raw={
                "execution_profile": {"model_id": "LOCAL-TEST-CUSTOMVOICE", "fixture_only": True}})

    cv_provider = BuiltinSignal()
    monkeypatch.setattr(renderer.TTSProviderFactory, "create", lambda backend: cv_provider)
    full = audition.audition(project.id, role_id=role.id, candidate_id="directed",
        expected_identity_id=voice_revision_for_role(role), request_id="customvoice-full-render",
        script_version="casting-30-v1", reuse_key=reuse_key)
    assert full["status"] == "succeeded", full.get("reason")
    full_record = load_project(project.id).voice_lab.voices[full["key"]]
    rendered_child = load_project(full_record["audition_project_id"])
    rendered_rows = load_segments(rendered_child.id)
    assert rendered_child.roles[0].qwen3_speaker == "ryan"
    assert all(row.emotion_text and "Speak with restrained urgency." in row.emotion_text for row in rendered_rows)
    assert all(section["control_status"] == "sent" for section in full_record["sections"])
    assert len(cv_provider.calls) == 6
