#!/usr/bin/env python3
"""
CosyVoice3 FastAPI服务器
提供REST API接口，与IndexTTS2 API保持兼容
"""

import os
import sys
import base64
import re
import argparse
from pathlib import Path
from typing import Optional, Dict
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
sys.path.insert(0, os.path.join(project_root, "cosyvoice"))
sys.path.insert(0, os.path.join(project_root, "cosyvoice", "third_party", "Matcha-TTS"))

from service.model_path_policy import (  # noqa: E402
    require_project_model_dir,
    require_project_model_dir_from_env,
)

# ==================== 配置 ====================
parser = argparse.ArgumentParser(description="CosyVoice3 API Server")
parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器地址")
parser.add_argument("--port", type=int, default=8002, help="服务器端口")
parser.add_argument("--model_dir", type=str, default="models/cosyvoice3/Fun-CosyVoice3-0.5B", help="模型目录")
parser.add_argument("--outputs-keep-days", type=int, default=30, help="输出文件保留天数，0 表示不清理")
parser.add_argument("--outputs-max-files", type=int, default=5000, help="输出目录最多保留文件数，0 表示不限制")


def _parse_server_args() -> argparse.Namespace:
    """解析服务参数并忽略未知参数，便于被外部启动器/测试框架导入。"""

    args, unknown = parser.parse_known_args()
    if unknown:
        logger.debug("忽略未知启动参数: %s", unknown)
    return args


# 全局变量
cosyvoice_model = None
config = None
_model_lock: Optional[Lock] = Lock()
_infer_lock: Optional[Lock] = Lock()
_outputs_cleanup_lock: Optional[Lock] = Lock()
_last_outputs_cleanup = 0.0
OUTPUTS_CLEANUP_MIN_INTERVAL = 600

_COSY_REQUIRED_FILES = (
    "cosyvoice3.yaml",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "CosyVoice-BlankEN/config.json",
    "CosyVoice-BlankEN/model.safetensors",
    "CosyVoice-BlankEN/merges.txt",
    "CosyVoice-BlankEN/tokenizer_config.json",
    "CosyVoice-BlankEN/vocab.json",
)
_WETEXT_REQUIRED_FILES = (
    "en/tn/tagger.fst",
    "en/tn/verbalizer.fst",
    "zh/tn/tagger.fst",
    "zh/tn/verbalizer.fst",
)

_SAFE_COMPONENT_RE = re.compile(r"[^0-9A-Za-z_\-.\u4e00-\u9fff]+")


def _sanitize_filename_component(value: str, default: str = "file") -> str:
    if not isinstance(value, str):
        return default
    value = value.strip()
    if not value:
        return default
    value = value.replace("/", "_").replace("\\", "_")
    value = _SAFE_COMPONENT_RE.sub("_", value)
    value = value.strip(" ._-")
    if not value:
        return default
    return value[:64]


def _validated_model_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    model_dir = require_project_model_dir(
        project_root,
        args.model_dir,
        label="CosyVoice3 model directory",
        required_files=_COSY_REQUIRED_FILES,
    )
    wetext_dir = require_project_model_dir_from_env(
        project_root,
        "COSYVOICE_WETEXT_DIR",
        label="CosyVoice3 WeText directory",
        required_files=_WETEXT_REQUIRED_FILES,
    )
    return model_dir, wetext_dir


def _load_cosyvoice_model_if_needed() -> None:
    """按需加载全局 CosyVoice 模型"""
    global cosyvoice_model, config

    if cosyvoice_model is not None:
        return

    lock = _model_lock
    if lock is not None:
        lock.acquire()
    try:
        if cosyvoice_model is not None:
            return

        if config is None:
            args = _parse_server_args()
            config = args

        model_dir, wetext_dir = _validated_model_paths(config)
        os.environ["COSYVOICE_WETEXT_DIR"] = str(wetext_dir)
        logger.info(f"🔧 正在加载 CosyVoice 模型: {model_dir}")

        from cosyvoice.cli.cosyvoice import AutoModel
        model = AutoModel(model_dir=str(model_dir))
        cosyvoice_model = model
        logger.info("✅ CosyVoice 模型加载完成")
        logger.info(f"采样率: {model.sample_rate}")
    finally:
        if lock is not None and lock.locked():
            try:
                lock.release()
            except Exception:
                pass


def _unload_cosyvoice_model() -> None:
    """卸载全局 CosyVoice 模型并释放显存"""
    global cosyvoice_model

    cosyvoice_model = None
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
    _, wetext_dir = _validated_model_paths(args)
    os.environ["COSYVOICE_WETEXT_DIR"] = str(wetext_dir)

    for folder in ["temp", "outputs", "prompts"]:
        Path(folder).mkdir(exist_ok=True)

    try:
        cleanup_outputs_dir(Path("outputs"), args.outputs_keep_days, args.outputs_max_files)
    except Exception:
        pass

    logger.info("⚙️ CosyVoice3 API 已启动，模型将按需延迟加载")

    yield

    logger.info("🔄 正在清理资源...")
    _unload_cosyvoice_model()
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
    title="CosyVoice3 API",
    description="零样本语音合成服务 - 支持音色克隆和自然语言指令控制",
    version="3.0.0",
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
    """语音合成请求 - 兼容IndexTTS2 API格式"""
    text: str = Field(..., description="要合成的文本内容", min_length=1, max_length=10000)
    speaker_audio: str = Field(..., description="参考音频路径或base64编码")
    
    # CosyVoice3 特有参数
    mode: str = Field("instruct", description="推理模式: zero_shot | instruct | cross_lingual，默认使用instruct")
    instruct: Optional[str] = Field(None, description="指令文本，如'请用开心的语气说'、'用广东话表达'")
    prompt_text: Optional[str] = Field(None, description="参考音频的文本内容（zero_shot模式使用）")
    speed: float = Field(1.0, description="语速调节，0.5-2.0，1.0为正常语速", ge=0.5, le=2.0)
    
    # 兼容IndexTTS2的情感参数（会自动转换为CosyVoice3的instruct模式）
    emotion_mode: int = Field(0, description="情感模式: 0=默认, 3=使用emotion_text作为指令", ge=0, le=3)
    emotion_text: Optional[str] = Field(None, description="情感描述文本（emotion_mode=3时转为instruct）")
    
    # 输出选项
    return_base64: bool = Field(False, description="是否返回base64编码的音频")
    save_file: bool = Field(True, description="是否保存音频文件到服务器")


class TTSResponse(BaseModel):
    """语音合成响应 - 与IndexTTS2 API格式兼容"""
    success: bool
    message: str
    audio_file: Optional[str] = None
    audio_base64: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: int = 22050


class HealthResponse(BaseModel):
    """健康检查响应"""
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_version: str
    device: str
    model_loaded: bool
    backend: str = "cosyvoice3"
    instance_token: Optional[str] = None


# ==================== 工具函数 ====================
def save_base64_audio(base64_str: str, output_path: str) -> str:
    """保存base64编码的音频到文件"""
    try:
        if ',' in base64_str and 'base64' in base64_str:
            base64_str = base64_str.split(',')[1]
        audio_data = base64.b64decode(base64_str)
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        return output_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"解析base64音频失败: {str(e)}")


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
        "service": "CosyVoice3 TTS API",
        "version": "3.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    import torch
    return HealthResponse(
        status="healthy",
        model_version="CosyVoice3-0.5B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_loaded=cosyvoice_model is not None,
        instance_token=os.environ.get("DUBBING_WORKER_INSTANCE_TOKEN"),
    )


@app.post("/api/v1/tts/synthesize", response_model=TTSResponse)
def synthesize_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    语音合成 - 兼容IndexTTS2 API格式
    
    支持的模式：
    - zero_shot: 纯音色克隆
    - instruct: 克隆 + 自然语言指令控制（方言/情感/语速）
    - cross_lingual: 跨语言克隆
    """
    try:
        _load_cosyvoice_model_if_needed()
        
        if cosyvoice_model is None:
            raise HTTPException(status_code=500, detail="模型加载失败")
        
        # 处理参考音频
        speaker_audio_path = request.speaker_audio
        temp_audio_path = None
        
        if request.speaker_audio.startswith("data:") or len(request.speaker_audio) > 500:
            # base64编码的音频
            temp_audio_path = f"temp/ref_{uuid.uuid4().hex}.wav"
            speaker_audio_path = save_base64_audio(request.speaker_audio, temp_audio_path)
        elif not os.path.isabs(request.speaker_audio):
            # 相对路径转绝对路径
            speaker_audio_path = os.path.join(project_root, request.speaker_audio)
        
        if not os.path.exists(speaker_audio_path):
            raise HTTPException(status_code=400, detail=f"参考音频不存在: {speaker_audio_path}")
        
        # 确定推理模式和指令
        mode = request.mode
        instruct = request.instruct
        
        # 兼容IndexTTS2的emotion_mode=3（文本情感）
        if request.emotion_mode == 3 and request.emotion_text:
            mode = "instruct"
            # 自动添加 <|endofprompt|> 后缀
            instruct = request.emotion_text
            if not instruct.endswith("<|endofprompt|>"):
                instruct = f"You are a helpful assistant. {instruct}<|endofprompt|>"
        
        # 生成输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_preview = _sanitize_filename_component(request.text[:20])
        output_filename = f"cosyvoice_{timestamp}_{text_preview}_{uuid.uuid4().hex[:8]}.wav"
        output_path = Path("outputs") / output_filename
        
        logger.info(f"🎙️ 开始合成: mode={mode}, text={request.text[:50]}...")
        start_time = time.time()
        
        # 执行推理
        import torch
        import torchaudio
        
        # 获取语速参数
        speed = getattr(request, 'speed', 1.0) or 1.0
        
        with _infer_lock:
            if mode == "zero_shot":
                prompt_text = request.prompt_text or "You are a helpful assistant.<|endofprompt|>"
                results = list(cosyvoice_model.inference_zero_shot(
                    request.text,
                    prompt_text,
                    speaker_audio_path,
                    stream=False,
                    speed=speed
                ))
            elif mode == "instruct":
                # 确保 instruct 包含 <|endofprompt|> 标记
                if not instruct:
                    instruct = "You are a helpful assistant.<|endofprompt|>"
                elif "<|endofprompt|>" not in instruct:
                    instruct = f"You are a helpful assistant. {instruct}<|endofprompt|>"
                results = list(cosyvoice_model.inference_instruct2(
                    request.text,
                    instruct,
                    speaker_audio_path,
                    stream=False,
                    speed=speed
                ))
            elif mode == "cross_lingual":
                results = list(cosyvoice_model.inference_cross_lingual(
                    request.text,
                    speaker_audio_path,
                    stream=False,
                    speed=speed
                ))
            else:
                raise HTTPException(status_code=400, detail=f"不支持的模式: {mode}")
        
        if not results:
            raise HTTPException(status_code=500, detail="合成失败：没有输出")
        
        # 合并所有音频片段
        audio_chunks = [chunk['tts_speech'] for chunk in results]
        audio = torch.cat(audio_chunks, dim=1)
        
        # 保存文件
        torchaudio.save(str(output_path), audio, cosyvoice_model.sample_rate)
        
        duration = audio.shape[1] / cosyvoice_model.sample_rate
        elapsed = time.time() - start_time
        logger.info(f"✅ 合成完成: {duration:.2f}s 音频, 耗时 {elapsed:.2f}s")
        
        # 清理临时文件
        if temp_audio_path:
            background_tasks.add_task(cleanup_temp_file, temp_audio_path)
        
        # 构建响应
        response = TTSResponse(
            success=True,
            message="合成成功",
            audio_file=str(output_path.absolute()) if request.save_file else None,
            duration_seconds=duration,
            sample_rate=cosyvoice_model.sample_rate
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


@app.post("/api/v1/tts/load-model")
def load_model():
    """手动触发加载模型"""
    try:
        _load_cosyvoice_model_if_needed()
        return {
            "success": True,
            "message": "模型加载成功",
            "model_loaded": cosyvoice_model is not None
        }
    except Exception as e:
        logger.error(f"加载模型失败: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


@app.post("/api/v1/tts/unload-model")
def unload_model():
    """卸载模型释放显存"""
    _unload_cosyvoice_model()
    return {"success": True, "message": "模型已卸载"}


@app.get("/api/v1/tts/download/{filename}")
async def download_audio(filename: str):
    """下载生成的音频文件"""
    # 路径穿越防护：仅允许简单文件名
    safe_name = os.path.basename(filename.replace("\\", "/"))
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="非法文件名")
    outputs_dir = Path("outputs").resolve()
    file_path = (outputs_dir / safe_name).resolve()
    if not file_path.is_relative_to(outputs_dir):
        raise HTTPException(status_code=400, detail="非法文件路径")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(file_path), media_type="audio/wav", filename=safe_name)


# ==================== 主入口 ====================
if __name__ == "__main__":
    args = _parse_server_args()
    logger.info(f"🚀 启动 CosyVoice3 API 服务: http://{args.host}:{args.port}")
    uvicorn.run(
        "cosyvoice_server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )
