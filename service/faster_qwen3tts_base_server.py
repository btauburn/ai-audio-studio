"""Experimental Base CUDA graph profile on the existing Qwen backend/API.

Launch only from the isolated Faster 0.3.2 environment. Ordinary VoiceDesign
reference creation retains the existing native path; all prepared character
auditions and production use this same Base adapter. No automatic fallback.
"""
from __future__ import annotations
import os,sys,time,threading
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import service.qwen3tts_server as native
from service.faster_qwen3tts_base_adapter import FasterBaseAdapter

PROFILE='faster-base-032'
native._runtime_profile=PROFILE
native._runtime_sources.update(native.source_snapshot(ROOT, ('service/faster_qwen3tts_base_server.py','service/faster_qwen3tts_base_adapter.py')))
native._runtime_snapshot=native.WorkerIdentitySnapshot(ROOT,native._parse_server_args(),native._runtime_sources,PROFILE)
native.parser.add_argument('--execution-profile',choices=[PROFILE],default=PROFILE)
original_load=native._load_qwen3tts_model_if_needed
_request_state_lock=threading.Lock()
_active_cancel_event=None
def approved_base_path(path: Path) -> Path:
    expected=(ROOT/'models/qwen3_tts/Qwen3-TTS-Base').resolve()
    if path.resolve()!=expected:raise ValueError('Experimental Faster Base must use the exact approved manifest snapshot')
    return expected
def request_event():
    with _request_state_lock:return _active_cancel_event
def load_model(kind='voice_design',attention='auto'):
    if kind!='base':return original_load(kind,attention)
    if isinstance(native.qwen3tts_model,FasterBaseAdapter):
        native.qwen3tts_model.cancel_event=request_event() or threading.Event()
        return
    with native._model_lock:
        args=native.config or native._parse_server_args();native.config=args
        if args.device!='cuda:0' or args.dtype!='bfloat16':raise ValueError('Faster Base requires CUDA:0 and the current BF16 precision')
        path=approved_base_path(native.require_project_model_dir(native.project_root,args.base_model_dir,label='Base model directory',required_files=native._QWEN_REQUIRED_FILES))
        native._model_verification_cache.verify(ROOT/'models/manifests/qwen3_tts_base.json',project_root=ROOT)
        native._unload_qwen3tts_model(release_verification=False)
        native.qwen3tts_model=FasterBaseAdapter.load(path)
        native.qwen3tts_model.cancel_event=request_event() or threading.Event()
        native._active_model_kind='base';native._active_attention='sdpa'
def character_call(method,**kwargs):
    global _active_cancel_event
    # The lease spans preparation, load, generation and metadata attribution.
    # Cancellation belongs to the request, including time before a model exists.
    with native._infer_lock:
        event=threading.Event() if method.__name__=='synthesize' else None
        with _request_state_lock:_active_cancel_event=event
        try:
            expected=kwargs.pop("expected_runtime_sha256",None)
            binding=native.base_runtime_identity() if method.__name__=="synthesize" else None
            if expected is not None and expected!=binding["sha256"]:
                raise native.HTTPException(status_code=409,detail="Worker runtime changed before synthesis; no audio generated")
            result=method(**kwargs)
            if binding is not None:result["worker_runtime_identity"]=binding
            with _request_state_lock:
                cancelled=event.is_set() if event is not None else False
                _active_cancel_event=None
            if method.__name__=='synthesize' and isinstance(native.qwen3tts_model,FasterBaseAdapter):
                if cancelled:
                    # Acknowledged cancellation cannot publish a successful Take.
                    output=result.get('audio_file')
                    if output:
                        candidate=Path(output).resolve()
                        if candidate.parent==(ROOT/'outputs').resolve() and candidate.name.startswith('qwen_identity_'):
                            candidate.unlink(missing_ok=True)
                    raise RuntimeError('character generation cancelled before response')
                result['execution_profile'].update(runtime_profile=PROFILE,faster_version='0.3.2',experimental=True)
                result['runtime_generation']=dict(native.qwen3tts_model.last_metadata)
                result['generation']=dict(native.qwen3tts_model.last_metadata)
                result['fallback']=None
            return result
        except (ValueError,FileNotFoundError) as exc:
            raise native.HTTPException(status_code=400,detail=str(exc)) from exc
        except native.HTTPException:raise
        except Exception as exc:
            native.logger.exception('Experimental character voice operation failed')
            raise native.HTTPException(status_code=500,detail=str(exc)) from exc
        finally:
            with _request_state_lock:_active_cancel_event=None
native._load_qwen3tts_model_if_needed=load_model
native._character_call=character_call
app=native.app
@app.post('/api/v1/tts/character/cancel')
def cancel_character():
    with _request_state_lock:
        if _active_cancel_event is not None:
            _active_cancel_event.set()
            return {'success':True,'state':'cancellation_requested','runtime_profile':PROFILE}
    return {'success':False,'state':'idle','runtime_profile':PROFILE}
# Expose active runtime provenance before spending GPU time on synthesis.
native.app.router.routes[:]=[route for route in native.app.router.routes if getattr(route,'path',None)!='/health']
@app.get('/health')
async def experimental_health():
    data=await native.health_check()
    return {**data.model_dump(),'runtime_profile':PROFILE,'faster_version':'0.3.2','experimental':True}
if __name__=='__main__':
    args=native._parse_server_args()
    import uvicorn
    uvicorn.run(app,host=args.host,port=args.port,reload=False,log_level='info')
