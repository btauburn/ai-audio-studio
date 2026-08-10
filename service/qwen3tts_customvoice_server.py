#!/usr/bin/env python3
"""
Qwen3-TTS CustomVoice FastAPI服务器
提供REST API接口，支持内置音色 + 文本指令控制情感
"""

import os
import sys
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import uuid
from datetime import datetime
import time
from contextlib import asynccontextmanager
from threading import Lock
import logging
import traceback

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
from dubbing_service.app.core.audio_path_policy import (  # noqa: E402
    AudioFileNotFoundError,
    InvalidAudioPathError,
    resolve_existing_audio_name,
)

# ==================== 配置 ====================
parser = argparse.ArgumentParser(description="Qwen3-TTS CustomVoice API Server")
parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器地址")
parser.add_argument("--port", type=int, default=8004, help="服务器端口")
parser.add_argument("--model_dir", type=str, default="models/qwen3_tts/Qwen3-TTS-CustomVoice", help="模型目录")
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
config = None
_model_lock: Optional[Lock] = Lock()
_infer_lock: Optional[Lock] = Lock()
_outputs_cleanup_lock: Optional[Lock] = Lock()
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

# CustomVoice 内置音色列表 (官方支持: Aiden, Dylan, Eric, Ono_Anna, Ryan, Serena, Sohee, Uncle_Fu, Vivian)
CUSTOM_VOICE_SPEAKERS = [
    {"id": "vivian", "name": "Vivian", "gender": "female", "language": "Chinese", "description": "明亮、略带锐气的年轻女声"},
    {"id": "serena", "name": "Serena", "gender": "female", "language": "Chinese", "description": "温暖柔和的年轻女声"},
    {"id": "uncle_fu", "name": "Uncle Fu", "gender": "male", "language": "Chinese", "description": "音色低沉醇厚的成熟男声"},
    {"id": "dylan", "name": "Dylan", "gender": "male", "language": "Chinese", "description": "清晰自然的北京青年男声（北京方言）"},
    {"id": "eric", "name": "Eric", "gender": "male", "language": "Chinese", "description": "活泼、略带沙哑明亮感的成都男声（四川方言）"},
    {"id": "ryan", "name": "Ryan", "gender": "male", "language": "English", "description": "富有节奏感的动态男声"},
    {"id": "aiden", "name": "Aiden", "gender": "male", "language": "English", "description": "清晰中频的阳光美式男声"},
    {"id": "ono_anna", "name": "Ono Anna", "gender": "female", "language": "Japanese", "description": "轻快灵活的俏皮日文女声"},
    {"id": "sohee", "name": "Sohee", "gender": "female", "language": "Korean", "description": "富含情感的温暖韩文女声"},
]

# API id → 官方模型 speaker name 映射（generate_custom_voice 需要首字母大写 / 下划线格式）
_SPEAKER_ID_TO_MODEL_NAME: Dict[str, str] = {
    "vivian": "Vivian",
    "serena": "Serena",
    "uncle_fu": "Uncle_Fu",
    "dylan": "Dylan",
    "eric": "Eric",
    "ryan": "Ryan",
    "aiden": "Aiden",
    "ono_anna": "Ono_Anna",
    "sohee": "Sohee",
}

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
        label="Qwen3-TTS CustomVoice model directory",
        required_files=_QWEN_REQUIRED_FILES,
    )


def _load_qwen3tts_model_if_needed() -> None:
    """按需加载全局 Qwen3-TTS CustomVoice 模型"""
    global qwen3tts_model, config

    if qwen3tts_model is not None:
        return

    lock = _model_lock
    if lock is not None:
        lock.acquire()
    try:
        if qwen3tts_model is not None:
            return

        if config is None:
            args = _parse_server_args()
            config = args

        model_path = _validated_model_path(config)
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

        model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map=config.device,
            dtype=dtype,
            attn_implementation=attn_impl,
            local_files_only=True,
        )
        qwen3tts_model = model
        logger.info("✅ Qwen3-TTS CustomVoice 模型加载完成")
    finally:
        if lock is not None and lock.locked():
            try:
                lock.release()
            except Exception:
                pass


def _unload_qwen3tts_model() -> None:
    """卸载全局 Qwen3-TTS 模型并释放显存"""
    global qwen3tts_model

    qwen3tts_model = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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

    logger.info("⚙️ Qwen3-TTS CustomVoice API 已启动，模型将按需延迟加载")

    yield

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
    title="Qwen3-TTS CustomVoice API",
    description="内置音色 + 文本指令控制情感 - 本地部署版",
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
    """语音合成请求"""
    text: str = Field(..., description="要合成的文本内容", min_length=1, max_length=10000)
    
    # CustomVoice 特有参数
    speaker: str = Field("vivian", description="内置音色：vivian, uncle_fu, aiden, dylan, eric, ryan, serena, ono_anna, sohee")
    instruct: Optional[str] = Field(None, description="情感指令，如'用愤怒的语气说'")
    language: str = Field("Chinese", description="语言: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian, Auto")
    
    # 兼容参数
    emotion_text: Optional[str] = Field(None, description="情感描述（会作为 instruct）")
    
    # 输出选项
    return_base64: bool = Field(False, description="是否返回base64编码的音频")
    save_file: bool = Field(True, description="是否保存音频文件到服务器")


class TTSResponse(BaseModel):
    """语音合成响应"""
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
    backend: str = "qwen3tts_cv"
    instance_token: Optional[str] = None


class SpeakersResponse(BaseModel):
    """音色列表响应"""
    speakers: List[Dict[str, str]]


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
        "service": "Qwen3-TTS CustomVoice API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if qwen3tts_model else "model_not_loaded",
        model_version="Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device=config.device if config else "cuda:0",
        model_loaded=qwen3tts_model is not None,
        instance_token=os.environ.get("DUBBING_WORKER_INSTANCE_TOKEN"),
    )


@app.get("/api/v1/tts/speakers", response_model=SpeakersResponse)
async def get_speakers():
    """获取支持的音色列表"""
    return SpeakersResponse(speakers=CUSTOM_VOICE_SPEAKERS)


@app.post("/api/v1/tts/load-model")
def load_model():
    """手动加载模型"""
    try:
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
    _unload_qwen3tts_model()
    return {"success": True, "message": "模型已卸载"}


@app.post("/api/v1/tts/synthesize", response_model=TTSResponse)
def synthesize_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    CustomVoice 语音合成
    
    使用内置音色 + 可选情感指令合成语音。
    """
    try:
        _load_qwen3tts_model_if_needed()
        
        if qwen3tts_model is None:
            raise HTTPException(status_code=500, detail="模型加载失败")
        
        # 处理情感指令
        instruct = request.instruct or ""
        if request.emotion_text and request.emotion_text.strip():
            if instruct:
                instruct = f"{instruct}，{request.emotion_text}"
            else:
                instruct = request.emotion_text
        
        # 映射语言参数 (支持10种语言)
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
        
        # 验证音色（大小写不敏感）
        speaker_input = request.speaker.strip()
        speaker_lower = speaker_input.lower()
        valid_ids = {s["id"] for s in CUSTOM_VOICE_SPEAKERS}
        if speaker_lower in valid_ids:
            speaker = speaker_lower
        elif speaker_input in _SPEAKER_ID_TO_MODEL_NAME.values():
            # 用户直接传了官方名称（如 "Vivian"），反查 id
            speaker = next((k for k, v in _SPEAKER_ID_TO_MODEL_NAME.items() if v == speaker_input), "vivian")
        else:
            speaker = "vivian"
        
        # 生成输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"qwen3tts_cv_{timestamp}_{uuid.uuid4().hex}.wav"
        output_path = Path("outputs") / output_filename
        
        logger.info(f"🎙️ CustomVoice 合成: speaker={speaker}, instruct={instruct[:50] if instruct else 'None'}..., text={request.text[:50]}...")
        start_time = time.time()
        
        # 执行推理
        import soundfile as sf
        
        with _infer_lock:
            # CustomVoice 使用 generate_custom_voice 方法
            # 将 API 小写 id 映射为官方模型期望的 speaker 名称
            model_speaker = _SPEAKER_ID_TO_MODEL_NAME.get(speaker, speaker)
            wavs, sr = qwen3tts_model.generate_custom_voice(
                text=request.text,
                language=language,
                speaker=model_speaker,
                instruct=instruct if instruct else None,
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


# ==================== 主入口 ====================
if __name__ == "__main__":
    args = _parse_server_args()
    logger.info(f"🚀 启动 Qwen3-TTS CustomVoice API 服务: http://{args.host}:{args.port}")
    uvicorn.run(
        "qwen3tts_customvoice_server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )
