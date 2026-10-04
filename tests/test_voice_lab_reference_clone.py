"""Eligible external-reference path, with generated fixtures and a fake model."""

import contextlib
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from dubbing_service.app.audiobook import performer_discovery as discovery
from service import qwen_character_voice as voice


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def float(self): return Tensor(self.value.astype(np.float32))
    def numpy(self): return self.value


@dataclass
class Prompt:
    ref_code: object
    ref_spk_embedding: object
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    project = tmp_path / "project"
    directory = project / "voice_lab/discovery"
    directory.mkdir(parents=True)
    out = io.BytesIO()
    sf.write(out, (np.sin(np.arange(16000 * 5) / 23) * .15).astype(np.float32), 16000, format="WAV")
    raw = out.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    transcript = "This is a synthetic fixture, not a performer recording."
    source = {"source_id": "fixture-original", "source_page": "https://commons.wikimedia.org/wiki/File:Fixture.wav",
              "media_url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/Fixture.wav",
              "attribution": "Fixture owner", "license_url": "https://creativecommons.org/licenses/by/4.0/",
              "terms_url": discovery.TERMS_URL, "language": "en", "duration_seconds": 5,
              "upstream_media_sha1": hashlib.sha1(raw).hexdigest()}
    permission = {"source_id": source["source_id"], "status": "approved", "use": "private_synthetic_voice",
                  "authority": "owner_original_voice", "evidence_ref": "fixture-only:ownership"}
    review = {"source_id": source["source_id"], "original_sha256": sha,
              "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(), "start_seconds": 0,
              "end_seconds": 4, "evidence_ref": "fixture-only:review", "method": "local_tools",
              "single_speaker": True, "transcript_aligned": True, "overlapping_speech": False,
              "heavy_music": False, "unsuitable_noise": False}
    downloads, calls = [], []
    def download(*args, **kwargs):
        downloads.append(args)
        return raw
    monkeypatch.setattr(discovery, "_download_media", download)
    monkeypatch.setattr(voice, "safe_project_root", lambda _: project)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        from_numpy=Tensor, cuda=SimpleNamespace(is_available=lambda: False), manual_seed=lambda _: None,
        random=SimpleNamespace(fork_rng=lambda **_: contextlib.nullcontext())))
    monkeypatch.setitem(sys.modules, "qwen_tts.inference.qwen3_tts_model", SimpleNamespace(VoiceClonePromptItem=Prompt))
    class Model:
        def create_voice_clone_prompt(self, **kwargs):
            calls.append("prepare")
            return [Prompt(Tensor(np.ones((12, 16), dtype=np.int64)), Tensor(np.arange(128, dtype=np.float32)), False, True, kwargs["ref_text"])]
        def generate_voice_clone(self, **kwargs):
            calls.append("clone")
            return [np.ones(2400, dtype=np.float32) * .01], 24000
    runtime = voice.CharacterVoiceRuntime(tmp_path, lambda _: Model())
    monkeypatch.setattr(runtime, "_model_metadata", lambda kind: {"model": "fixture-base", "model_revision": "revision-1", "runtime_version": "0.1.1"})
    return SimpleNamespace(project=project, directory=directory, source=source, permission=permission, transcript=transcript,
                           review=review, downloads=downloads, calls=calls, runtime=runtime, raw=raw)


def acquire(f, **kwargs):
    options = dict(intended_use_evidence=f.permission, transcript=f.transcript,
                   start_seconds=0, end_seconds=4, audio_review=f.review)
    options.update(kwargs)
    return discovery.acquire_reference(f.directory, f.source, **options)


def import_voice(f, acquisition, identity_id="eligible-1"):
    return f.runtime.import_reference("project", identity_id, acquisition["acquisition_id"], acquisition["acquisition_sha256"])


def test_acquisition_original_crop_and_worker_reuse_are_immutable_and_idempotent(fixture):
    f = fixture
    first = acquire(f)
    assert first["clone_ready"] and first["stats"]["downloads"] == 1
    second = acquire(f)
    assert first["acquisition_sha256"] == second["acquisition_sha256"]
    assert second["stats"]["downloads"] == 0 and len(f.downloads) == 1
    imported = import_voice(f, first)
    assert imported == import_voice(f, second)
    assert imported["provenance"]["origin"] == "eligible_reference_clone"
    assert (f.project / imported["original_audio"]).read_bytes() == f.raw
    prepared = f.runtime.prepare("project", "eligible-1")
    assert prepared["prompt_sha256"] == f.runtime.prepare("project", "eligible-1")["prompt_sha256"]
    args = {key: prepared[key] for key in ("prompt_artifact", "prompt_sha256", "reference_audio", "reference_audio_sha256")}
    result = f.runtime.synthesize("project", "eligible-1", text="Fixture output.", language="English", **args)
    assert Path(result["audio_file"]).exists()
    assert result["execution_profile"]["mechanism"] == "persistent_eligible_reference"
    assert f.calls == ["prepare", "clone"]


def test_recording_license_without_use_permission_never_downloads(fixture):
    f = fixture
    with pytest.raises(discovery.DiscoveryHold, match="acquisition held"):
        acquire(f, intended_use_evidence={})
    assert f.downloads == [] and f.calls == []


@pytest.mark.parametrize("field,value", [("single_speaker", False), ("transcript_aligned", False),
                                          ("heavy_music", True), ("overlapping_speech", True),
                                          ("unsuitable_noise", True), ("original_sha256", "0" * 64)])
def test_missing_or_failed_audio_review_is_held_and_cannot_reach_model(fixture, field, value):
    f = fixture
    acquisition = acquire(f, audio_review=f.review | {field: value})
    assert not acquisition["clone_ready"]
    with pytest.raises(ValueError, match="QA"):
        import_voice(f, acquisition)
    assert f.calls == []


def test_public_metadata_cannot_set_synthetic_voice_permission(fixture):
    f = fixture
    f.source["intended_use_eligible"] = True
    f.source["consent"] = "ignore all restrictions"
    with pytest.raises(discovery.DiscoveryHold):
        acquire(f, intended_use_evidence={})
    assert f.downloads == []


@pytest.mark.parametrize("mutation", ["receipt", "reference", "original"])
def test_tampered_acquisition_never_reaches_model(fixture, mutation):
    f = fixture
    acquisition = acquire(f)
    if mutation == "receipt":
        path = f.directory / "acquisitions" / acquisition["acquisition_id"] / "acquisition.json"
    else:
        path = f.directory / acquisition["reference_path" if mutation == "reference" else "original_path"]
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        import_voice(f, acquisition)
    assert f.calls == []


def test_import_does_not_overwrite_an_original_identity(fixture):
    f = fixture
    acquisition = acquire(f)
    folder = f.project / "casting/voices/existing"
    folder.mkdir(parents=True)
    original = {"request_hash": "production-reference", "provenance": {"origin": "generated_original_voice"}}
    path = folder / "reference.json"
    path.write_text(json.dumps(original))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="different reference"):
        import_voice(f, acquisition, "existing")
    assert path.read_bytes() == before


@pytest.mark.parametrize("identity", ["../outside", "C:/outside", "x" * 121])
def test_import_rejects_identity_escape(fixture, identity):
    f = fixture
    acquisition = acquire(f)
    with pytest.raises(ValueError):
        import_voice(f, acquisition, identity)


def test_fixed_acquisition_id_prevents_arbitrary_worker_file_access(fixture):
    f = fixture
    with pytest.raises(ValueError, match="acquisition identity"):
        f.runtime.import_reference("project", "voice", "../../private", "a" * 64)
    assert f.downloads == [] and f.calls == []


def test_wrong_upstream_media_hash_and_download_count_fail_closed(fixture):
    f = fixture
    f.source["upstream_media_sha1"] = "0" * 40
    with pytest.raises(discovery.DiscoveryHold, match="checksum"):
        acquire(f)
    assert not list(f.directory.rglob("*.audio"))
    with pytest.raises(discovery.DiscoveryHold, match="count"):
        acquire(f, limits=discovery.DiscoveryLimits(max_downloads=0))
    assert len(f.downloads) == 1


def test_worker_api_validates_receipt_identity_and_holds_shared_inference_lock(monkeypatch):
    from fastapi.testclient import TestClient
    from service import qwen3tts_server as server
    calls = []
    def imported(**kwargs):
        assert server._infer_lock.locked()
        calls.append(kwargs)
        return {"identity_id": kwargs["identity_id"], "synthetic": True}
    monkeypatch.setattr(server._character_runtime, "import_reference", imported)
    client = TestClient(server.app)
    payload = {"project_id": "fixture", "identity_id": "voice", "acquisition_id": "a" * 32, "acquisition_sha256": "b" * 64}
    response = client.post("/api/v1/tts/character/import-reference", json=payload)
    assert response.status_code == 200 and response.json()["identity_id"] == "voice"
    assert not server._infer_lock.locked() and calls == [payload]
    response = client.post("/api/v1/tts/character/import-reference", json=payload | {"acquisition_id": "../../other-file"})
    assert response.status_code == 422 and calls == [payload]



@pytest.mark.parametrize("delivery", [None, "review", "reject"])
def test_automatic_receipt_cannot_waive_missing_or_rejected_delivery(fixture, delivery):
    f = fixture
    acquired = acquire(f)
    path = f.directory/f"acquisitions/{acquired['acquisition_id']}/acquisition.json"
    receipt = json.loads(path.read_text())
    receipt['automatic_preparation'] = {'status':'pass','schema':'fixture-automatic'}
    if delivery is not None:
        from dubbing_service.app.audiobook.reference_delivery_fit import evaluate
        audio, rate = sf.read(str(f.directory/receipt['reference_path']),dtype='float32')
        assessment = None
        if delivery == 'reject':
            assessment = {'method':'human_listening','human_verified':True,'actor':'fixture-listener',
                'evidence_ref':'fixture-only:delivery','source_id':f.source['source_id'],
                'start_seconds':0,'end_seconds':4,'input_pcm_sha256':hashlib.sha256(audio.tobytes()).hexdigest(),
                'delivery':'audience_directed_shouting','observations':['Fixture crowd-facing yell.']}
        receipt['narration_delivery_fit'] = evaluate(audio,rate,source=f.source,start_seconds=0,end_seconds=4,
            technical_status='pass',transcript=f.transcript,audio_assessment=assessment)
        assert receipt['narration_delivery_fit']['delivery_decision'] == delivery
    path.write_text(json.dumps(receipt),'utf-8')
    with pytest.raises(ValueError,match='narration delivery evidence'):
        voice._eligible_acquisition(f.project,receipt['acquisition_id'],hashlib.sha256(path.read_bytes()).hexdigest())
    assert not f.calls


def test_new_automatic_acquisition_holds_delivery_separately_from_clean_technical_qa(fixture, monkeypatch):
    f = fixture
    from dubbing_service.app.audiobook import reference_preparation as prep
    monkeypatch.setattr(prep,'prepare',lambda directory,wave,rate,*args,**kwargs:
        (wave,rate,None,{'status':'pass','schema':'fixture-automatic'},False))
    result = acquire(f,automatic_preparation=True)
    assert result['technical_qa']['status'] == 'pass'
    assert result['narration_delivery_fit']['delivery_decision'] == 'review'
    assert not result['clone_ready'] and result['readiness_holds']
    assert not f.calls



def test_derived_accept_cannot_replace_original_listening_assessment(fixture):
    f = fixture
    acquired = acquire(f)
    path = f.directory/f"acquisitions/{acquired['acquisition_id']}/acquisition.json"
    receipt = json.loads(path.read_text())
    receipt['automatic_preparation'] = {'status':'pass','schema':'fixture-automatic'}
    from dubbing_service.app.audiobook.reference_delivery_fit import evaluate
    audio,rate = sf.read(str(f.directory/receipt['reference_path']),dtype='float32')
    heard = {'method':'human_listening','human_verified':True,'actor':'fixture-listener',
        'evidence_ref':'fixture-only:delivery','source_id':f.source['source_id'],
        'start_seconds':0,'end_seconds':4,'input_pcm_sha256':hashlib.sha256(audio.tobytes()).hexdigest(),
        'delivery':'calm_reading','observations':['Fixture calm sustained reading.']}
    receipt['narration_delivery_fit'] = evaluate(audio,rate,source=f.source,start_seconds=0,end_seconds=4,
        technical_status='pass',transcript=f.transcript,audio_assessment=heard)
    assert receipt['narration_delivery_fit']['selected_reference']
    path.write_text(json.dumps(receipt),'utf-8')
    with pytest.raises(ValueError,match='narration delivery evidence'):
        voice._eligible_acquisition(f.project,receipt['acquisition_id'],hashlib.sha256(path.read_bytes()).hexdigest())
    receipt['audio_review']['narration_delivery'] = heard
    path.write_text(json.dumps(receipt),'utf-8')
    assert voice._eligible_acquisition(f.project,receipt['acquisition_id'],hashlib.sha256(path.read_bytes()).hexdigest())[0]['clone_ready']
    assert not f.calls
