#!/usr/bin/env python3
"""
Qwen3-TTS VoiceDesign FastAPI服务器
提供REST API接口，支持基于文本描述生成音色并合成语音
"""

import os
import sys
import base64
import argparse
from pathlib import Path
from typing import Dict, Optional
import uuid
from datetime import datetime
import time
from contextlib import asynccontextmanager
from threading import Lock
import logging
import traceback
import gc

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from service.model_path_policy import require_project_model_dir  # noqa: E402
from service.qwen_character_voice import CharacterVoiceRuntime  # noqa: E402
from service.pinned_model_verification import PinnedModelVerificationCache  # noqa: E402
from dubbing_service.app.core.audio_path_policy import (  # noqa: E402
    AudioFileNotFoundError,
    InvalidAudioPathError,
    resolve_existing_audio_name,
)

# ==================== 配置 ====================
parser = argparse.ArgumentParser(description="Qwen3-TTS VoiceDesign API Server")
parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器地址")
parser.add_argument("--port", type=int, default=8003, help="服务器端口")
parser.add_argument("--model_dir", type=str, default="models/qwen3_tts/Qwen3-TTS-VoiceDesign", help="模型目录")
parser.add_argument("--base-model-dir", type=str, default="models/qwen3_tts/Qwen3-TTS-Base")
parser.add_argument("--device", type=str, default="cuda:0", help="运行设备")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="数据类型")
parser.add_argument("--outputs-keep-days", type=int, default=30, help="输出文件保留天数，0 表示不清理")
parser.add_argument("--outputs-max-files", type=int, default=5000, help="输出目录最多保留文件数，0 表示不限制")


def _parse_server_args() -> argparse.Namespace:
    """解析服务参数并忽略未知参数，便于被外部启动器/测试框架导入。"""

    args, unknown = parser.parse_known_args()
    if unknown:
        logger.debug("忽略未知启动参数: %s", unknown)
    return args


# 全局变量
qwen3tts_model = None
_active_model_kind = None
_active_attention = None
config = None
from service.qwen_runtime_identity import WorkerIdentitySnapshot, source_snapshot
_runtime_sources = source_snapshot(Path(project_root), ("service/qwen3tts_server.py", "service/qwen_character_voice.py", "service/qwen_runtime_identity.py"))
_runtime_profile = "native-base"
_runtime_snapshot = WorkerIdentitySnapshot(Path(project_root), _parse_server_args(), _runtime_sources, _runtime_profile)
def base_runtime_identity():
    return _runtime_snapshot.read(config or _parse_server_args())

_model_lock: Optional[Lock] = Lock()
_infer_lock: Optional[Lock] = Lock()
_outputs_cleanup_lock: Optional[Lock] = Lock()
_model_verification_cache = PinnedModelVerificationCache()
_last_outputs_cleanup = 0.0
OUTPUTS_CLEANUP_MIN_INTERVAL = 600

_QWEN_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "merges.txt",
    "tokenizer_config.json",
    "vocab.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
)

def _resolve_download_audio(directory: Path, filename: str) -> Path:
    try:
        return resolve_existing_audio_name(directory, filename)
    except InvalidAudioPathError as exc:
        raise HTTPException(status_code=400, detail="非法音频文件名") from exc
    except AudioFileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="文件不存在") from exc


def _validated_model_path(args: argparse.Namespace) -> Path:
    return require_project_model_dir(
        project_root,
        args.model_dir,
        label="Qwen3-TTS VoiceDesign model directory",
        required_files=_QWEN_REQUIRED_FILES,
    )


def _load_qwen3tts_model_if_needed(kind: str = "voice_design", attention: str = "auto") -> None:
    """按需加载全局 Qwen3-TTS VoiceDesign 模型"""
    global qwen3tts_model, config, _active_model_kind, _active_attention

    if qwen3tts_model is not None and _active_model_kind == kind and _active_attention == attention:
        return

    lock = _model_lock
    if lock is not None:
        lock.acquire()
    try:
        if qwen3tts_model is not None and _active_model_kind == kind and _active_attention == attention:
            return

        if config is None:
            args = _parse_server_args()
            config = args

        if kind not in {"voice_design", "base"}:
            raise ValueError("unsupported character voice model kind")
        model_path = (_validated_model_path(config) if kind == "voice_design" else
                      require_project_model_dir(project_root, config.base_model_dir,
                                                label="Qwen3-TTS Base model directory", required_files=_QWEN_REQUIRED_FILES))
        if attention == "sdpa":
            manifest_name = "voicedesign" if kind == "voice_design" else "base"
            verification = _model_verification_cache.verify(
                Path(project_root) / "models" / "manifests" / f"qwen3_tts_{manifest_name}.json",
                project_root=Path(project_root),
            )
            logger.info("Character model integrity: kind=%s cached=%s pinned=%s",
                        kind, verification.cache_hit, verification.pinned)
        # The inference lock held by callers spans the switch and generation.
        # Never keep both large Qwen models resident on the single GPU.
        _unload_qwen3tts_model(release_verification=False)
        logger.info(f"🔧 正在加载项目内本地模型: {model_path}")

        import torch
        from qwen_tts import Qwen3TTSModel

        # 确定数据类型
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(config.dtype, torch.bfloat16)

        # 检查是否支持 flash_attention_2
        attn_impl = "flash_attention_2"
        try:
            __import__("flash_attn")
        except ImportError:
            attn_impl = "eager"
            logger.warning("⚠️ FlashAttention 未安装，使用默认注意力实现")
        if attention == "sdpa":
            attn_impl = "sdpa"

        model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map=config.device,
            dtype=dtype,
            attn_implementation=attn_impl,
            local_files_only=True,
        )
        qwen3tts_model = model
        _active_model_kind = kind
        _active_attention = attention
        logger.info("Qwen3-TTS model loaded: kind=%s attention=%s", kind, attn_impl)
    finally:
        if lock is not None and lock.locked():
            try:
                lock.release()
            except Exception:
                pass


def _unload_qwen3tts_model(*, release_verification: bool = True) -> None:
    """卸载全局 Qwen3-TTS 模型并释放显存"""
    global qwen3tts_model, _active_model_kind, _active_attention

    if release_verification:
        _model_verification_cache.clear()

    qwen3tts_model = None
    _active_model_kind = None
    _active_attention = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("✅ Qwen3-TTS 模型已卸载，显存已释放")


def cleanup_outputs_dir(outputs_dir: Path, keep_days: int, max_files: int) -> None:
    if keep_days <= 0 and max_files <= 0:
        return
    global _last_outputs_cleanup
    now = time.time()
    if now - _last_outputs_cleanup < OUTPUTS_CLEANUP_MIN_INTERVAL:
        return
    lock = _outputs_cleanup_lock
    if lock is not None and not lock.acquire(blocking=False):
        return
    try:
        _last_outputs_cleanup = now
        if not outputs_dir.exists():
            return
        files = [p for p in outputs_dir.glob("*.wav") if p.is_file()]
        if keep_days > 0:
            cutoff = now - keep_days * 86400
            for file_path in files:
                try:
                    if file_path.stat().st_mtime < cutoff:
                        file_path.unlink()
                except Exception:
                    pass
            files = [p for p in outputs_dir.glob("*.wav") if p.is_file()]
        if max_files > 0 and len(files) > max_files:
            files.sort(key=lambda p: p.stat().st_mtime)
            for file_path in files[:-max_files]:
                try:
                    file_path.unlink()
                except Exception:
                    pass
    finally:
        if lock is not None and lock.locked():
            try:
                lock.release()
            except Exception:
                pass


# ==================== 生命周期管理 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global config

    args = _parse_server_args()
    config = args
    _validated_model_path(args)

    for folder in ["temp", "outputs", "prompts"]:
        Path(folder).mkdir(exist_ok=True)

    try:
        cleanup_outputs_dir(Path("outputs"), args.outputs_keep_days, args.outputs_max_files)
    except Exception:
        pass

    logger.info("⚙️ Qwen3-TTS VoiceDesign API 已启动，模型将按需延迟加载")

    try:
        yield
    finally:
        logger.info("🔄 正在清理资源...")
        _unload_qwen3tts_model()
        temp_dir = Path("temp")
        if temp_dir.exists():
            for file in temp_dir.glob("*.wav"):
                try:
                    file.unlink()
                except Exception:
                    pass
        logger.info("✅ 资源清理完成")


# ==================== FastAPI应用 ====================
app = FastAPI(
    title="Qwen3-TTS VoiceDesign API",
    description="基于文本描述生成音色并合成语音 - 本地部署版",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:9000", "http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 数据模型 ====================
class TTSRequest(BaseModel):
    """语音合成请求 - 兼容现有 TTS API 格式"""
    text: str = Field(..., description="要合成的文本内容", min_length=1, max_length=10000)
    
    # VoiceDesign 特有参数
    instruct: str = Field(..., description="音色描述文本，如'温柔的年轻女性，声音甜美清脆'")
    language: str = Field("Chinese", description="语言: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian, Auto")
    
    # 兼容 IndexTTS2/CosyVoice3 的参数（部分忽略）
    speaker_audio: Optional[str] = Field(None, description="参考音频（VoiceDesign 模式下忽略）")
    emotion_mode: int = Field(0, description="情感模式（VoiceDesign 使用 instruct 描述）")
    emotion_text: Optional[str] = Field(None, description="情感描述（会合并到 instruct）")
    
    # 输出选项
    return_base64: bool = Field(False, description="是否返回base64编码的音频")
    save_file: bool = Field(True, description="是否保存音频文件到服务器")


class TTSResponse(BaseModel):
    """语音合成响应 - 与 IndexTTS2/CosyVoice3 API 格式兼容"""
    success: bool
    message: str
    audio_file: Optional[str] = None
    audio_base64: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: int = 24000


class HealthResponse(BaseModel):
    """健康检查响应"""
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_version: str
    device: str
    model_loaded: bool
    backend: str = "qwen3tts"
    instance_token: Optional[str] = None
    active_model_kind: Optional[str] = None
    character_voice_capabilities: list[str] = Field(default_factory=lambda: ["original_reference", "safe_persistent_prompt", "base_reuse", "eligible_reference_import"])


# ==================== 工具函数 ====================
def audio_to_base64(file_path: str) -> str:
    """将音频文件转换为base64"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def cleanup_temp_file(file_path: str):
    """后台清理临时文件"""
    try:
        if os.path.exists(file_path) and "temp" in file_path:
            os.remove(file_path)
    except Exception:
        pass


# ==================== API端点 ====================
@app.get("/", response_model=Dict[str, str])
async def root():
    """API根路径"""
    return {
        "service": "Qwen3-TTS VoiceDesign API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/v1/tts/runtime-identity")
def runtime_identity_endpoint():
    return base_runtime_identity()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if qwen3tts_model else "model_not_loaded",
        model_version="Qwen3-TTS-12Hz-1.7B-Base" if _active_model_kind == "base" else "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device=config.device if config else "cuda:0",
        model_loaded=qwen3tts_model is not None,
        instance_token=os.environ.get("DUBBING_WORKER_INSTANCE_TOKEN"),
        active_model_kind=_active_model_kind,
    )


@app.post("/api/v1/tts/load-model")
def load_model():
    """手动加载模型"""
    try:
        with _infer_lock:
            _load_qwen3tts_model_if_needed()
        return {
            "success": True,
            "message": "模型加载成功",
            "model_loaded": qwen3tts_model is not None
        }
    except Exception as e:
        logger.error(f"加载模型失败: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


@app.post("/api/v1/tts/unload-model")
def unload_model():
    """卸载模型释放显存"""
    with _infer_lock:
        _unload_qwen3tts_model()
    return {"success": True, "message": "模型已卸载"}


@app.post("/api/v1/tts/synthesize", response_model=TTSResponse)
def synthesize_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    VoiceDesign 语音合成
    
    根据文本描述生成音色并合成语音，无需参考音频。
    """
    try:
        # 合并 instruct 和 emotion_text
        instruct = request.instruct or ""
        if request.emotion_text and request.emotion_text.strip():
            if instruct:
                instruct = f"{instruct}，{request.emotion_text}"
            else:
                instruct = request.emotion_text
        
        if not instruct:
            instruct = "用自然、清晰的声音朗读"
        
        # 映射语言参数 (支持10种语言 + Auto)
        language_map = {
            "auto": "Auto", "Auto": "Auto",
            "zh": "Chinese", "Chinese": "Chinese",
            "en": "English", "English": "English",
            "ja": "Japanese", "Japanese": "Japanese",
            "ko": "Korean", "Korean": "Korean",
            "de": "German", "German": "German",
            "fr": "French", "French": "French",
            "ru": "Russian", "Russian": "Russian",
            "pt": "Portuguese", "Portuguese": "Portuguese",
            "es": "Spanish", "Spanish": "Spanish",
            "it": "Italian", "Italian": "Italian",
        }
        language = language_map.get(request.language, request.language)
        
        # 生成输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"qwen3tts_{timestamp}_{uuid.uuid4().hex}.wav"
        output_path = Path("outputs") / output_filename
        
        logger.info(f"🎙️ VoiceDesign 合成: instruct={instruct[:50]}..., text={request.text[:50]}...")
        start_time = time.time()
        
        # 执行推理
        import soundfile as sf
        
        with _infer_lock:
            _load_qwen3tts_model_if_needed()
            wavs, sr = qwen3tts_model.generate_voice_design(
                text=request.text,
                language=language,
                instruct=instruct,
            )
        
        if wavs is None or len(wavs) == 0:
            raise HTTPException(status_code=500, detail="合成失败：没有输出")
        
        # 保存音频文件
        sf.write(str(output_path), wavs[0], sr)
        
        duration = len(wavs[0]) / sr
        elapsed = time.time() - start_time
        logger.info(f"✅ 合成完成: {duration:.2f}s 音频, 耗时 {elapsed:.2f}s")
        
        # 构建响应
        response = TTSResponse(
            success=True,
            message="合成成功",
            audio_file=str(output_path.absolute()) if request.save_file else None,
            duration_seconds=duration,
            sample_rate=sr
        )
        
        if request.return_base64:
            response.audio_base64 = audio_to_base64(str(output_path))
        
        if not request.save_file:
            background_tasks.add_task(cleanup_temp_file, str(output_path))
        elif config:
            background_tasks.add_task(
                cleanup_outputs_dir,
                Path("outputs"),
                config.outputs_keep_days,
                config.outputs_max_files,
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"合成失败: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"合成失败: {str(e)}")


@app.get("/api/v1/tts/download/{filename}")
async def download_audio(filename: str):
    """下载生成的音频文件"""
    file_path = _resolve_download_audio(Path("outputs"), filename)
    return FileResponse(str(file_path), media_type="audio/wav", filename=file_path.name)


def _character_model(kind: str):
    _load_qwen3tts_model_if_needed(kind, attention="sdpa")
    return qwen3tts_model


_character_runtime = CharacterVoiceRuntime(Path(project_root), _character_model)


class CharacterDesignRequest(BaseModel):
    project_id: str
    identity_id: str
    instruction: str = Field(min_length=1, max_length=4000)
    text: str = Field(min_length=1, max_length=1000)
    seed: int = Field(0, ge=0, le=2147483647)
    # Acoustic conditioning derived from the candidate's performer reference. It
    # is a numeric target profile plus its provenance, never a performer name.
    conditioning: Optional[dict] = None


class CharacterPrepareRequest(BaseModel):
    project_id: str
    identity_id: str


class CharacterAcousticsRequest(CharacterPrepareRequest):
    conditioning: Optional[dict] = None


class CharacterImportReferenceRequest(CharacterPrepareRequest):
    acquisition_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    acquisition_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class CharacterRenderRequest(CharacterPrepareRequest):
    prompt_artifact: str
    prompt_sha256: str
    reference_audio: str
    reference_audio_sha256: str
    text: str = Field(min_length=1, max_length=10000)
    language: str = "English"
    performance: dict = Field(default_factory=dict)
    expected_runtime_sha256: Optional[str] = Field(None, pattern=r"^[a-f0-9]{64}$")


def _character_call(method, **kwargs):
    try:
        with _infer_lock:
            expected = kwargs.pop("expected_runtime_sha256", None)
            binding = base_runtime_identity() if method.__name__ == "synthesize" else None
            if expected is not None and expected != binding["sha256"]:
                raise HTTPException(status_code=409, detail="Worker runtime changed before synthesis; no audio generated")
            result = method(**kwargs)
            if binding is not None:
                result["worker_runtime_identity"] = binding
                result["fallback"] = None
            return result
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Character voice operation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/tts/character/design")
def design_character_voice(request: CharacterDesignRequest):
    return _character_call(_character_runtime.design, **request.model_dump())


@app.post("/api/v1/tts/character/acoustics")
def condition_character_reference(request: CharacterAcousticsRequest):
    return _character_call(_character_runtime.acoustics, **request.model_dump())


@app.post("/api/v1/tts/character/prepare")
def prepare_character_voice(request: CharacterPrepareRequest):
    return _character_call(_character_runtime.prepare, **request.model_dump())


@app.post("/api/v1/tts/character/import-reference")
def import_character_reference(request: CharacterImportReferenceRequest):
    return _character_call(_character_runtime.import_reference, **request.model_dump())


@app.post("/api/v1/tts/character/synthesize")
def synthesize_character_voice(request: CharacterRenderRequest):
    return _character_call(_character_runtime.synthesize, **request.model_dump())


# ==================== 主入口 ====================
if __name__ == "__main__":
    args = _parse_server_args()
    logger.info(f"🚀 启动 Qwen3-TTS VoiceDesign API 服务: http://{args.host}:{args.port}")
    uvicorn.run(
        "qwen3tts_server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )
