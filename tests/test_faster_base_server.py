import threading
import importlib.util
import os
import sys
from types import SimpleNamespace
from pathlib import Path
import pytest
from fastapi import HTTPException
from service.faster_qwen3tts_base_adapter import FasterBaseAdapter
server=None

@pytest.fixture(scope='module',autouse=True)
def isolated_server_module():
    """Import the profile against a private native module; never patch shared tests."""
    global server
    import service
    import service.qwen3tts_server as shared
    original_loader=shared._load_qwen3tts_model_if_needed
    original_call=shared._character_call
    original_routes=list(shared.app.router.routes)
    original_parser_actions=list(shared.parser._actions)
    with pytest.MonkeyPatch.context() as private:
        native_name='service._faster_base_test_native'
        spec=importlib.util.spec_from_file_location(native_name,shared.__file__)
        native=importlib.util.module_from_spec(spec)
        private.setitem(sys.modules,native_name,native);spec.loader.exec_module(native)
        with pytest.MonkeyPatch.context() as imports:
            imports.setitem(sys.modules,'service.qwen3tts_server',native)
            imports.setattr(service,'qwen3tts_server',native)
            for flag in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']:imports.setenv(flag,os.environ.get(flag,''))
            name='service._faster_base_test_server'
            spec=importlib.util.spec_from_file_location(name,Path(shared.__file__).with_name('faster_qwen3tts_base_server.py'))
            server=importlib.util.module_from_spec(spec)
            private.setitem(sys.modules,name,server);spec.loader.exec_module(server)
        assert shared._load_qwen3tts_model_if_needed is original_loader
        assert shared._character_call is original_call
        assert shared.app.router.routes==original_routes
        assert shared.parser._actions==original_parser_actions
        yield
        assert shared._load_qwen3tts_model_if_needed is original_loader
        assert shared._character_call is original_call
        assert shared.app.router.routes==original_routes
        assert shared.parser._actions==original_parser_actions
    server=None

def test_profile_import_leaves_shared_native_runtime_unchanged():
    import service.qwen3tts_server as shared
    assert server.native is not shared
    assert shared._character_call is not server.character_call
    assert shared._load_qwen3tts_model_if_needed is not server.load_model
    assert not any(action.dest=='execution_profile' for action in shared.parser._actions)

class Adapter(FasterBaseAdapter):
    def __init__(self):
        super().__init__(None)
        self._metadata={'request':'first'}
    @property
    def last_metadata(self):
        assert server.native._infer_lock.locked(), 'metadata escaped inference lease'
        return self._metadata
    @last_metadata.setter
    def last_metadata(self,value):self._metadata=value

@pytest.fixture
def worker(monkeypatch):
    model=Adapter()
    monkeypatch.setattr(server.native,'qwen3tts_model',model)
    monkeypatch.setattr(server.native,'_infer_lock',threading.Lock())
    monkeypatch.setattr(server.native,'_model_lock',threading.Lock())
    monkeypatch.setattr(server,'_active_cancel_event',None)
    return model

def test_response_metadata_is_attributed_under_infer_lock(worker):
    def synthesize():return {'execution_profile':{},'audio_file':None}
    result=server.character_call(synthesize)
    assert result['runtime_generation']=={'request':'first'}
    worker._metadata={'request':'second'}
    assert result['runtime_generation']=={'request':'first'}
    assert server.request_event() is None

def test_cancel_during_conditioning_survives_model_lookup(worker):
    def synthesize():
        assert server.cancel_character()['success'] is True
        server.load_model('base')
        assert worker.cancel_event.is_set()
        raise RuntimeError('cancelled before decode')
    with pytest.raises(HTTPException,match='cancelled') as exc:server.character_call(synthesize)
    assert exc.value.status_code==500
    assert server.cancel_character()['state']=='idle'
    def synthesize():
        server.load_model('base');assert not worker.cancel_event.is_set()
        return {'execution_profile':{},'audio_file':None}
    assert server.character_call(synthesize)['fallback'] is None

def test_initial_model_load_cancellation_is_acknowledged(worker,monkeypatch):
    monkeypatch.setattr(server.native,'qwen3tts_model',None)
    monkeypatch.setattr(server.native,'config',SimpleNamespace(device='cuda:0',dtype='bfloat16',base_model_dir='models/qwen3_tts/Qwen3-TTS-Base'))
    expected=server.ROOT/'models/qwen3_tts/Qwen3-TTS-Base'
    monkeypatch.setattr(server.native,'require_project_model_dir',lambda *a,**k:expected)
    monkeypatch.setattr(server.native,'_model_verification_cache',SimpleNamespace(verify=lambda *a,**k:None))
    monkeypatch.setattr(server.native,'_unload_qwen3tts_model',lambda **k:None)
    def load(path):
        assert server.cancel_character()['success'] is True
        return worker
    monkeypatch.setattr(server.FasterBaseAdapter,'load',load)
    def synthesize():
        server.load_model('base');assert worker.cancel_event.is_set()
        raise RuntimeError('cancelled after model load')
    with pytest.raises(HTTPException,match='cancelled'):server.character_call(synthesize)

def test_completed_request_refuses_late_cancellation(worker):
    class Profile(dict):
        def update(self,**kwargs):
            assert server.cancel_character()=={'success':False,'state':'idle','runtime_profile':'faster-base-032'}
            return super().update(**kwargs)
    def synthesize():return {'execution_profile':Profile()}
    assert server.character_call(synthesize)['execution_profile']['runtime_profile']=='faster-base-032'

def test_oom_returns_clear_failure_and_releases_request(worker):
    def synthesize():raise RuntimeError('CUDA out of memory')
    with pytest.raises(HTTPException,match='CUDA out of memory') as exc:server.character_call(synthesize)
    assert exc.value.status_code==500 and not server.native._infer_lock.locked()
    assert server.request_event() is None

def test_alternate_checkpoint_is_refused_before_model_unload(worker,monkeypatch):
    monkeypatch.setattr(server.native,'qwen3tts_model',None)
    monkeypatch.setattr(server.native,'config',SimpleNamespace(device='cuda:0',dtype='bfloat16',base_model_dir='models/alternate'))
    monkeypatch.setattr(server.native,'require_project_model_dir',lambda *a,**k:server.ROOT/'models/alternate')
    monkeypatch.setattr(server.native,'_unload_qwen3tts_model',lambda **k:pytest.fail('do not unload approved model before rejecting alternate'))
    with pytest.raises(ValueError,match='exact approved'):server.load_model('base')

def test_health_exposes_experimental_profile(worker):
    import asyncio
    data=asyncio.run(server.experimental_health())
    assert data['backend']=='qwen3tts' and data['experimental'] is True
    assert data['faster_version']=='0.3.2' and data['runtime_profile']=='faster-base-032'
