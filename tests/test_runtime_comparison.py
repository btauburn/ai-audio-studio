import hashlib,json
from pathlib import Path
import pytest
from dubbing_service.app.audiobook import runtime_comparison as comparison
from dubbing_service.app.core.tts_backends import build_backend_command
from dubbing_service.app.routers.settings import TTSServiceStartPayload

def test_missing_bundle_is_pending_without_voice_mutation(tmp_path,monkeypatch):
    from dubbing_service.app.storage import projects
    calls=[];monkeypatch.setattr(projects,'load_project',lambda p:calls.append(p));monkeypatch.setattr(comparison,'BUNDLE',tmp_path)
    assert comparison.load_comparison('book')['status']=='pending'
    assert calls==['book']

def test_exact_project_allowlist_hash_and_lane(tmp_path,monkeypatch):
    from dubbing_service.app.storage import projects
    monkeypatch.setattr(projects,'load_project',lambda p:None);monkeypatch.setattr(comparison,'BUNDLE',tmp_path)
    data=b'RIFF exact frozen audio';(tmp_path/'sample.wav').write_bytes(data)
    digest=hashlib.sha256(data).hexdigest()
    payload={'review_project_id':'book','pairs':[{'id':'short-narrator','baseline':{'audio_file':'sample.wav','audio_sha256':digest}}]}
    (tmp_path/'comparison.json').write_text(json.dumps(payload))
    assert comparison.load_comparison('other')['status']=='unavailable'
    assert comparison.load_comparison('book')['pairs'][0]['baseline']['audio_url'].endswith('/short-narrator/baseline')
    path,checksum=comparison.comparison_audio('book','short-narrator','baseline');assert path.read_bytes()==data and checksum==digest
    (tmp_path/'sample.wav').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):comparison.comparison_audio('book','short-narrator','baseline')
    for pair,lane in [('../escape','baseline'),('short-narrator','other')]:
        with pytest.raises(ValueError):comparison.comparison_audio('book',pair,lane)
    assert json.loads((tmp_path/'comparison.json').read_text())==payload

def test_artifact_traversal_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(comparison,'load_comparison',lambda p:{'pairs':[{'id':'p','candidate':{'audio_file':'../secret','audio_sha256':'0'*64}}]})
    monkeypatch.setattr(comparison,'BUNDLE',tmp_path)
    with pytest.raises(ValueError,match='artifact path'):comparison.comparison_audio('book','p','candidate')

def test_faster_runtime_keeps_backend_port_identity_and_default(tmp_path):
    for env,script in [('qwen3tts','qwen3tts_server.py'),('faster_qwen_base032','faster_qwen3tts_base_server.py')]:
        py=tmp_path/'environments'/env/'.venv/Scripts/python.exe';py.parent.mkdir(parents=True);py.touch()
        f=tmp_path/'service'/script;f.parent.mkdir(exist_ok=True);f.touch()
    (tmp_path/'models/qwen3_tts/Qwen3-TTS-VoiceDesign').mkdir(parents=True)
    current=build_backend_command(tmp_path,'qwen3tts',tmp_path/'platform',qwen_execution_profile='native-current')
    default=build_backend_command(tmp_path,'qwen3tts',tmp_path/'platform')
    fast=build_backend_command(tmp_path,'qwen3tts',tmp_path/'platform',qwen_execution_profile='faster-base-032')
    assert default == fast
    assert 'qwen3tts_server.py' in current[2] and 'faster_qwen3tts_base_server.py' in fast[2]
    assert current[current.index('--port')+1]==fast[fast.index('--port')+1]=='8003'
    assert current[current.index('--dtype')+1]==fast[fast.index('--dtype')+1]=='bfloat16'
    assert current[current.index('--model_dir')+1]==fast[fast.index('--model_dir')+1]

def test_explicit_start_profile_does_not_change_normal_payload():
    assert TTSServiceStartPayload(backend='qwen3tts').qwen_execution_profile is None
    assert TTSServiceStartPayload(backend='qwen3tts',qwen_execution_profile='faster-base-032',persist_default=False).persist_default is False
    with pytest.raises(ValueError):TTSServiceStartPayload(qwen_execution_profile='unreviewed')
