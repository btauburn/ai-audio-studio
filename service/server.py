#!/usr/bin/env python3
"""
IndexTTS2 FastAPI服务器
提供REST API接口供其他工具调用
"""

import os
import sys
import base64
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
import time
from contextlib import asynccontextmanager
from threading import Lock
import logging
import traceback

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "indextts"))

from indextts.infer_v2 import IndexTTS2  # noqa: E402
from dubbing_service.app.core.audio_path_policy import (  # noqa: E402
    AudioFileNotFoundError,
    InvalidAudioPathError,
    managed_audio_roots,
    resolve_existing_audio_file,
    resolve_existing_audio_name,
)
from service.model_path_policy import require_project_model_dir  # noqa: E402

# ==================== 配置 ====================
parser = argparse.ArgumentParser(description="IndexTTS2 API Server")
parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器地址")
parser.add_argument("--port", type=int, default=8001, help="服务器端口")
parser.add_argument("--model_dir", type=str, default="models/index_tts", help="模型目录（相对于项目根目录）")
parser.add_argument(
    "--aux-model-dir",
    type=str,
    default=None,
    help="IndexTTS2 辅助模型目录；默认使用 <model_dir>/aux_models",
)
parser.add_argument("--fp16", action="store_true", help="使用FP16推理")
parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="禁用FP16推理")
parser.add_argument("--deepspeed", action="store_true", default=False, help="使用DeepSpeed加速")
parser.add_argument("--cuda-kernel", dest="cuda_kernel", action="store_true", help="强制使用CUDA kernel")
parser.add_argument("--no-cuda-kernel", dest="cuda_kernel", action="store_false", help="禁用CUDA kernel")
parser.add_argument("--outputs-keep-days", type=int, default=30, help="输出文件保留天数，0 表示不清理")
parser.add_argument("--outputs-max-files", type=int, default=5000, help="输出目录最多保留文件数，0 表示不限制")
parser.set_defaults(fp16=True, cuda_kernel=None)  # CUDA环境默认启用FP16，cuda_kernel自动检测


def _parse_server_args() -> argparse.Namespace:
    """解析服务参数并忽略未知参数，便于被外部启动器/测试框架导入。"""

    args, unknown = parser.parse_known_args()
    if unknown:
        logger.debug("忽略未知启动参数: %s", unknown)
    return args


# 全局变量
tts_model = None
config = None
_model_lock: Optional[Lock] = Lock()
_infer_lock: Optional[Lock] = Lock()
_outputs_cleanup_lock: Optional[Lock] = Lock()
_last_outputs_cleanup = 0.0
OUTPUTS_CLEANUP_MIN_INTERVAL = 600

_INDEX_REQUIRED_FILES = (
    "config.yaml",
    "gpt.pth",
    "s2mel.pth",
    "bpe.model",
    "feat1.pt",
    "feat2.pt",
    "wav2vec2bert_stats.pt",
    "qwen0.6bemo4-merge/config.json",
    "qwen0.6bemo4-merge/model.safetensors",
    "qwen0.6bemo4-merge/tokenizer.json",
    "qwen0.6bemo4-merge/tokenizer_config.json",
)
_INDEX_AUX_REQUIRED_FILES = (
    "w2v-bert-2.0/config.json",
    "w2v-bert-2.0/model.safetensors",
    "w2v-bert-2.0/preprocessor_config.json",
    "maskgct/semantic_codec/model.safetensors",
    "campplus/campplus_cn_common.bin",
    "bigvgan/config.json",
    "bigvgan/bigvgan_generator.pt",
)

def _safe_resolve_under_dir(base_dir: Path, filename: str) -> Path:
    if not isinstance(filename, str) or not filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    filename = os.path.basename(filename.replace("\\", "/"))
    if filename in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="非法文件名")
    candidate = (base_dir / filename).resolve()
    base_resolved = base_dir.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail="非法文件路径")
    return candidate


def _resolve_reference_audio(value: str) -> Path:
    try:
        return resolve_existing_audio_file(
            value,
            roots=managed_audio_roots(project_root),
            relative_bases=(Path(project_root), Path.cwd()),
            allow_bare_name=True,
        )
    except InvalidAudioPathError as exc:
        raise HTTPException(status_code=400, detail="参考音频路径非法") from exc
    except AudioFileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="参考音频不存在") from exc


def _resolve_download_audio(directory: Path, filename: str) -> Path:
    try:
        return resolve_existing_audio_name(directory, filename)
    except InvalidAudioPathError as exc:
        raise HTTPException(status_code=400, detail="非法音频文件名") from exc
    except AudioFileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="文件不存在") from exc


def _validated_model_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    model_dir = require_project_model_dir(
        project_root,
        args.model_dir,
        label="IndexTTS2 model directory",
        required_files=_INDEX_REQUIRED_FILES,
    )
    configured_aux = args.aux_model_dir or model_dir / "aux_models"
    aux_model_dir = require_project_model_dir(
        project_root,
        configured_aux,
        label="IndexTTS2 auxiliary model directory",
        required_files=_INDEX_AUX_REQUIRED_FILES,
    )
    return model_dir, aux_model_dir

def _load_tts_model_if_needed() -> None:
    """按需加载全局 TTS 模型，避免在应用启动时立即占用显存。"""

    global tts_model, config

    if tts_model is not None:
        return

    if _model_lock is None:
        # 理论上不会发生，仅作防御
        lock = None
    else:
        lock = _model_lock

    if lock is not None:
        lock.acquire()
    try:
        if tts_model is not None:
            return

        if config is None:
            args = _parse_server_args()
            config = args

        model_dir, aux_model_dir = _validated_model_paths(config)
        logger.info("🔧 正在加载 IndexTTS2 模型到显存...")

        model = IndexTTS2(
            model_dir=str(model_dir),
            cfg_path=str(model_dir / "config.yaml"),
            use_fp16=config.fp16,
            use_deepspeed=config.deepspeed,
            use_cuda_kernel=config.cuda_kernel,
            aux_model_dir=str(aux_model_dir),
        )
        tts_model = model
        logger.info("✅ TTS 模型加载完成")
        logger.info(f"设备: {getattr(tts_model, 'device', 'unknown')}")
        logger.info(f"FP16: {config.fp16}, CUDA Kernel: {config.cuda_kernel}")
    finally:
        if lock is not None and lock.locked():
            try:
                lock.release()
            except Exception:
                pass


def _unload_tts_model() -> None:
    """卸载全局 TTS 模型并尽量释放显存。"""

    global tts_model

    tts_model = None
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # 环境可能没有 torch 或 CUDA，忽略释放失败
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
    """应用生命周期管理（启动和关闭）"""
    global config

    # 启动时仅解析参数和创建必要目录，不立即加载模型
    args = _parse_server_args()
    config = args
    _validated_model_paths(args)

    try:
        if getattr(args, "cuda_kernel", None) is None:
            import torch  # type: ignore
            from torch.utils import cpp_extension  # type: ignore

            if torch.cuda.is_available() and cpp_extension.CUDA_HOME is None:
                args.cuda_kernel = False
                logger.info("CUDA kernel disabled because CUDA Toolkit (nvcc) was not found")
    except Exception:
        pass

    for folder in ["temp", "outputs", "prompts"]:
        Path(folder).mkdir(exist_ok=True)

    try:
        cleanup_outputs_dir(Path("outputs"), args.outputs_keep_days, args.outputs_max_files)
    except Exception:
        pass

    logger.info("⚙️ IndexTTS2 API 已启动，TTS 模型将按需延迟加载（首次调用 /tts 时加载）。")

    yield  # 应用运行

    # 关闭时清理
    logger.info("🔄 正在清理资源...")
    # 清理临时文件
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
    title="IndexTTS2 API",
    description="零样本语音合成服务 - 支持情感控制和音色克隆",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:9000", "http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 数据模型 ====================
class TTSRequest(BaseModel):
    """语音合成请求
    
    IndexTTS2支持零样本音色克隆和丰富的情感控制，可以从3-10秒的参考音频克隆任何人的声音。
    """
    text: str = Field(..., description="要合成的文本内容（支持中英文）", min_length=1, max_length=10000, examples=["你好，这是IndexTTS2的测试。"])
    speaker_audio: str = Field(..., description="参考音频：项目/音色库内路径或base64编码字符串，用于克隆音色", examples=["dubbing_projects/example/roles/prompts/voice.wav", "UklGRi4gAABXQVZF..."])
    
    @field_validator('emotion_vector')
    @classmethod
    def validate_emotion_vector(cls, v):
        if v is not None:
            if len(v) != 8:
                raise ValueError('情感向量必须包含8个元素')
            if not all(0 <= x <= 1 for x in v):
                raise ValueError('情感向量的每个值必须在0-1之间')
        return v
    emotion_mode: int = Field(0, description="情感控制模式：0=使用说话人情感, 1=使用情感参考音频（分离音色和情感）, 2=使用8维情感向量（精确控制）, 3=使用文本描述（实验性）", ge=0, le=3, examples=[0, 1, 2])
    emotion_audio: Optional[str] = Field(None, description="情感参考音频（仅emotion_mode=1时使用）：可以用A的音色 + B的情感", examples=["E:/audio/happy_voice.wav"])
    emotion_vector: Optional[List[float]] = Field(None, description="8维情感向量（仅emotion_mode=2时使用）：[快乐, 愤怒, 悲伤, 恐惧, 厌恶, 忧郁, 惊讶, 平静]，每个值0-1，可混合", examples=[[1.0, 0, 0, 0, 0, 0, 0, 0], [0.6, 0, 0, 0, 0, 0, 0.4, 0]])
    emotion_text: Optional[str] = Field(None, description="情感描述文本（仅emotion_mode=3时使用）：如'高兴的'、'悲伤的'等", examples=["高兴的", "平静的"])
    emotion_alpha: float = Field(0.65, description="情感强度：控制情感的影响程度，0.0=无情感，1.0=完全情感", ge=0.0, le=1.0, examples=[0.65, 0.8])
    use_random: bool = Field(False, description="是否使用随机情感采样：启用后每次生成会有轻微变化")
    
    # 高级生成参数（可选，使用默认值即可获得良好效果）
    max_text_tokens_per_segment: int = Field(120, description="每段最大文本token数：控制长文本分段", ge=20, le=600)
    max_mel_tokens: int = Field(1500, description="最大mel token数：影响生成音频的最大长度", ge=50, le=2000)
    interval_silence: int = Field(200, description="分段间隔静音时长(毫秒)：长文本分段间的停顿时间", ge=0, le=2000, examples=[100, 200, 500])
    do_sample: bool = Field(True, description="是否使用采样：启用可获得更自然的语音")
    temperature: float = Field(0.8, description="生成温度：控制随机性，0.7=稳定，0.8=平衡，0.9=多样", ge=0.1, le=2.0, examples=[0.7, 0.8, 0.9])
    top_p: float = Field(0.8, description="核采样参数：控制采样多样性，越高越多样", ge=0.0, le=1.0, examples=[0.8, 0.85, 0.9])
    top_k: int = Field(30, description="Top-K采样：限制候选token数量，越小越保守", ge=0, le=100, examples=[15, 30, 50])
    num_beams: int = Field(3, description="Beam搜索宽度：越大质量越高但速度越慢", ge=1, le=10)
    repetition_penalty: float = Field(10.0, description="重复惩罚：防止重复内容，建议1.5-2.0", ge=0.1, le=20.0, examples=[1.5, 1.8, 2.0])
    length_penalty: float = Field(0.0, description="长度惩罚：正值偏好长句，负值偏好短句", ge=-2.0, le=2.0)
    
    # 输出选项
    return_base64: bool = Field(False, description="是否返回base64编码的音频：适合前端直接播放")
    save_file: bool = Field(True, description="是否保存音频文件到服务器：保存在outputs/目录")


class TTSResponse(BaseModel):
    """语音合成响应"""
    success: bool
    message: str
    audio_file: Optional[str] = None
    audio_base64: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: int = 22050
    # 当使用情感文本模式 (emotion_mode=3) 且在解析情感分值时检测到异常数值时，
    # 返回一条非致命的 warning，提示前端本次合成属于“降级成功”。
    emotion_warning: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_version: str
    device: str
    model_loaded: bool
    backend: str = "indextts"
    instance_token: Optional[str] = None


# ==================== 工具函数 ====================
def save_base64_audio(base64_str: str, output_path: str) -> str:
    """保存base64编码的音频到文件"""
    try:
        if ',' in base64_str and 'base64' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        compact = "".join(base64_str.split())
        audio_data = base64.b64decode(compact, validate=True)
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
@app.get("/")
async def root() -> Dict[str, Any]:
    """根路径 - API信息"""
    return {
        "name": "IndexTTS2 API",
        "version": "2.0.0",
        "description": "零样本语音合成服务",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "synthesize": "/api/v1/tts/synthesize",
            "synthesize_file": "/api/v1/tts/synthesize-file",
            "upload_prompt": "/api/v1/prompts/upload",
            "list_prompts": "/api/v1/prompts/list",
            "download": "/api/v1/outputs/{filename}",
            "upload": "/api/v1/upload"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if tts_model else "model_not_loaded",
        model_version="2.0.0",
        device=str(tts_model.device if tts_model and hasattr(tts_model, 'device') else "unknown"),
        model_loaded=tts_model is not None,
        instance_token=os.environ.get("DUBBING_WORKER_INSTANCE_TOKEN"),
    )


@app.post("/api/v1/tts/load-model")
def api_load_tts_model() -> Dict[str, Any]:
    """手动触发加载 TTS 模型到显存。"""

    try:
        _load_tts_model_if_needed()
    except Exception as e:
        logger.error("手动加载 TTS 模型失败", exc_info=True)
        raise HTTPException(status_code=500, detail=f"加载 TTS 模型失败: {e}")

    return {
        "success": tts_model is not None,
        "model_loaded": tts_model is not None,
        "device": str(tts_model.device if tts_model and hasattr(tts_model, "device") else "unknown"),
    }


@app.post("/api/v1/tts/unload-model")
def api_unload_tts_model() -> Dict[str, Any]:
    """手动卸载 TTS 模型，释放显存。"""

    _unload_tts_model()

    return {
        "success": True,
        "model_loaded": tts_model is not None,
        "device": str(tts_model.device if tts_model and hasattr(tts_model, "device") else "unknown"),
        "message": "已请求卸载 TTS 模型并释放显存",
    }


@app.post("/api/v1/tts/synthesize", response_model=TTSResponse)
def synthesize_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    语音合成 - JSON请求
    
    支持的情感控制模式：
    - 0: 使用说话人音色中的情感（默认）
    - 1: 使用情感参考音频
    - 2: 使用情感向量控制 [喜,怒,哀,惧,厌恶,低落,惊喜,平静]
    - 3: 使用情感描述文本（实验性功能）
    """
    if not tts_model:
        try:
            _load_tts_model_if_needed()
        except Exception as e:
            logger.error("在合成请求中加载 TTS 模型失败", exc_info=True)
            raise HTTPException(status_code=503, detail=f"模型加载失败: {e}")
    if not tts_model:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        start_time = time.time()
        
        # 处理说话人参考音频
        try:
            speaker_audio_path = str(_resolve_reference_audio(request.speaker_audio))
        except HTTPException:
            # 路径未命中托管音频时，仅允许将输入作为严格 base64 解码；
            # 绝不会把未经批准的绝对路径交给文件系统。
            speaker_audio_path = str(
                _safe_resolve_under_dir(
                    Path("temp"),
                    f"speaker_{uuid.uuid4().hex}.wav",
                )
            )
            save_base64_audio(request.speaker_audio, speaker_audio_path)
            background_tasks.add_task(cleanup_temp_file, speaker_audio_path)
        
        # 处理情感参考音频（与WebUI一致）
        emotion_audio_path = None
        if request.emotion_mode == 0:
            # emotion_mode=0: 使用说话人情感，移除外部情感参考
            emotion_audio_path = None
        elif request.emotion_mode == 1 and request.emotion_audio:
            try:
                emotion_audio_path = str(_resolve_reference_audio(request.emotion_audio))
            except HTTPException:
                emotion_audio_path = str(
                    _safe_resolve_under_dir(
                        Path("temp"),
                        f"emotion_{uuid.uuid4().hex}.wav",
                    )
                )
                save_base64_audio(request.emotion_audio, emotion_audio_path)
                background_tasks.add_task(cleanup_temp_file, emotion_audio_path)
        
        # 生成输出文件路径
        output_filename = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
        if request.save_file:
            output_path = _safe_resolve_under_dir(Path("outputs"), output_filename)
        else:
            output_path = _safe_resolve_under_dir(Path("temp"), output_filename)
            background_tasks.add_task(cleanup_temp_file, str(output_path))
        
        # 准备情感向量（与WebUI一致：使用normalize_emo_vec）
        emotion_vector = None
        if request.emotion_mode == 2 and request.emotion_vector:
            if len(request.emotion_vector) != 8:
                raise HTTPException(status_code=400, detail="情感向量必须包含8个浮点数 [喜,怒,哀,惧,厌恶,低落,惊喜,平静]")
            # ✅ 关键修复：调用normalize_emo_vec与WebUI保持一致
            emotion_vector = tts_model.normalize_emo_vec(request.emotion_vector, apply_bias=True)
            logger.info(f"Normalized emotion vector: {emotion_vector}")
        
        # 处理情感文本（与WebUI一致：空字符串转None）
        emotion_text = request.emotion_text
        if emotion_text is not None and emotion_text.strip() == "":
            emotion_text = None
        
        # 调用TTS推理（与WebUI完全一致）
        acquired_infer_lock = False
        if _infer_lock is not None:
            _infer_lock.acquire()
            acquired_infer_lock = True
        try:
            tts_model.infer(
                spk_audio_prompt=speaker_audio_path,
                text=request.text,
                output_path=str(output_path),
                emo_audio_prompt=emotion_audio_path,
                emo_alpha=request.emotion_alpha,
                emo_vector=emotion_vector,  # 已通过normalize_emo_vec处理
                use_emo_text=(request.emotion_mode == 3),
                emo_text=emotion_text,  # 已处理空字符串
                use_random=request.use_random,
                max_text_tokens_per_segment=request.max_text_tokens_per_segment,
                interval_silence=request.interval_silence,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k if request.top_k > 0 else None,  # 与WebUI一致
                num_beams=request.num_beams,
                repetition_penalty=request.repetition_penalty,
                length_penalty=request.length_penalty,
                max_mel_tokens=request.max_mel_tokens,
                verbose=False
            )
        finally:
            if acquired_infer_lock and _infer_lock is not None:
                try:
                    _infer_lock.release()
                except Exception:
                    pass
        
        duration = time.time() - start_time
        
        # 如果在情感文本模式下，QwenEmotion 在 clamp_score 中遇到过异常数值，
        # 则附加一条非致命 warning，提示用户情感描述存在潜在问题。
        emotion_warning: Optional[str] = None
        try:
            qwen_emo = getattr(tts_model, "qwen_emo", None)
            if (
                qwen_emo is not None
                and getattr(qwen_emo, "last_clamp_issues", 0) > 0
                and request.emotion_mode == 3
            ):
                emotion_warning = (
                    "情感描述解析时检测到部分异常数值，已按 0.0 处理后继续合成。"
                    "建议检查或简化情感描述文本，以获得更稳定的情感效果。"
                )
        except Exception:
            # 任何用于生成 warning 的内部错误都不影响本次合成结果
            emotion_warning = None

        # 准备响应
        response_data = {
            "success": True,
            "message": "语音合成成功",
            "audio_file": f"outputs/{output_filename}" if request.save_file else None,
            "duration_seconds": round(duration, 2),
            "sample_rate": 22050,
            "emotion_warning": emotion_warning,
        }
        
        # 如果需要返回base64
        if request.return_base64:
            response_data["audio_base64"] = audio_to_base64(str(output_path))

        if request.save_file and config:
            background_tasks.add_task(
                cleanup_outputs_dir,
                Path("outputs"),
                config.outputs_keep_days,
                config.outputs_max_files,
            )
        
        return TTSResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        # 打印完整 traceback，方便排查语音合成内部错误
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")


@app.post("/api/v1/tts/synthesize-file")
async def synthesize_tts_file(
    background_tasks: BackgroundTasks,
    text: str = Form(..., description="要合成的文本"),
    speaker_audio: UploadFile = File(..., description="说话人参考音频文件"),
    emotion_mode: int = Form(0, description="情感控制模式 0-3"),
    emotion_audio: Optional[UploadFile] = File(None, description="情感参考音频（可选）"),
    emotion_vector: Optional[str] = Form(None, description="情感向量JSON字符串，如'[1,0,0,0,0,0,0,0]'"),
    emotion_alpha: float = Form(0.65, description="情感权重"),
    emotion_text: Optional[str] = Form(None, description="情感描述文本（可选）"),
    use_random: bool = Form(False, description="是否使用随机情感采样"),
    
    # 高级生成参数
    max_text_tokens_per_segment: int = Form(120, description="每段最大文本token数"),
    max_mel_tokens: int = Form(1500, description="最大mel token数"),
    interval_silence: int = Form(200, description="分段间隔静音时长(毫秒)"),
    do_sample: bool = Form(True, description="是否使用采样"),
    temperature: float = Form(0.8, description="温度参数"),
    top_p: float = Form(0.8, description="top_p采样参数"),
    top_k: int = Form(30, description="top_k采样参数"),
    num_beams: int = Form(3, description="beam搜索数量"),
    repetition_penalty: float = Form(10.0, description="重复惩罚"),
    length_penalty: float = Form(0.0, description="长度惩罚")
):
    """
    语音合成 - 文件上传版本
    
    与 /api/v1/tts/synthesize 功能完全相同，唯一区别是：
    - 输入：支持直接上传音频文件（不需要转base64或提供路径）
    - 输出：直接返回音频文件流（不是JSON）
    
    支持所有IndexTTS2的参数和功能。
    适合从n8n、Dify、Postman等工具调用。
    """
    if not tts_model:
        try:
            _load_tts_model_if_needed()
        except Exception as e:
            logger.error("在文件上传合成请求中加载 TTS 模型失败", exc_info=True)
            raise HTTPException(status_code=503, detail=f"模型加载失败: {e}")
    if not tts_model:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 保存上传的说话人音频
        speaker_path = _safe_resolve_under_dir(Path("temp"), f"speaker_{uuid.uuid4().hex}.wav")
        with open(speaker_path, "wb") as f:
            f.write(await speaker_audio.read())
        background_tasks.add_task(cleanup_temp_file, str(speaker_path))
        
        # 处理情感音频（与WebUI一致）
        emotion_path = None
        if emotion_mode == 0:
            # emotion_mode=0: 使用说话人情感，移除外部情感参考
            emotion_path = None
        elif emotion_audio and emotion_mode == 1:
            emotion_path = _safe_resolve_under_dir(Path("temp"), f"emotion_{uuid.uuid4().hex}.wav")
            with open(emotion_path, "wb") as f:
                f.write(await emotion_audio.read())
            background_tasks.add_task(cleanup_temp_file, str(emotion_path))
        
        # 解析并规范化情感向量（与WebUI一致）
        emo_vector = None
        if emotion_mode == 2 and emotion_vector:
            try:
                import json
                raw_vector = json.loads(emotion_vector)
                if len(raw_vector) != 8:
                    raise HTTPException(status_code=400, detail="情感向量必须包含8个浮点数")
                # ✅ 关键修复：调用normalize_emo_vec与WebUI保持一致
                emo_vector = tts_model.normalize_emo_vec(raw_vector, apply_bias=True)
                logger.info(f"Normalized emotion vector: {emo_vector}")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="情感向量格式错误，应为JSON数组")
        
        # 处理情感文本（与WebUI一致：空字符串转None）
        if emotion_text is not None and emotion_text.strip() == "":
            emotion_text = None
        
        # 生成输出
        output_filename = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.wav"
        output_path = _safe_resolve_under_dir(Path("outputs"), output_filename)
        
        # 调用TTS（完整参数）
        acquired_infer_lock = False
        if _infer_lock is not None:
            _infer_lock.acquire()
            acquired_infer_lock = True
        try:
            tts_model.infer(
                spk_audio_prompt=str(speaker_path),
                text=text,
                output_path=str(output_path),
                emo_audio_prompt=str(emotion_path) if emotion_path else None,
                emo_alpha=emotion_alpha,
                emo_vector=emo_vector,
                use_emo_text=(emotion_mode == 3),
                emo_text=emotion_text,
                use_random=use_random,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                interval_silence=interval_silence,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                max_mel_tokens=max_mel_tokens,
                verbose=False
            )
        finally:
            if acquired_infer_lock and _infer_lock is not None:
                try:
                    _infer_lock.release()
                except Exception:
                    pass
        
        if config:
            background_tasks.add_task(
                cleanup_outputs_dir,
                Path("outputs"),
                config.outputs_keep_days,
                config.outputs_max_files,
            )

        # 直接返回文件
        return FileResponse(
            str(output_path),
            media_type="audio/wav",
            filename=output_filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")


@app.post("/api/v1/upload")
async def upload_audio(audio: UploadFile = File(..., description="音频文件")) -> Dict[str, Any]:
    """
    上传音频文件到临时目录
    
    返回服务器上的文件路径，可以在后续的合成请求中使用
    """
    try:
        # 保存到prompts文件夹（永久保存）
        filename = f"{uuid.uuid4().hex}.wav"
        save_path = _safe_resolve_under_dir(Path("prompts"), filename)
        
        with open(save_path, "wb") as f:
            f.write(await audio.read())
        
        return {
            "success": True,
            "message": "音频上传成功",
            "path": f"prompts/{filename}",
            "filename": filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.post("/api/v1/prompts/upload")
async def upload_prompt(
    name: str = Form(..., description="参考音频名称"),
    audio: UploadFile = File(..., description="音频文件")
) -> Dict[str, Any]:
    """上传并保存命名的参考音频"""
    try:
        filename = f"prompt_{uuid.uuid4().hex}.wav"
        save_path = _safe_resolve_under_dir(Path("prompts"), filename)
         
        with open(save_path, "wb") as f:
            f.write(await audio.read())
         
        return {
            "success": True,
            "message": "参考音频上传成功",
            "path": f"prompts/{filename}",
            "name": name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.get("/api/v1/prompts/list")
async def list_prompts() -> Dict[str, Any]:
    """列出所有保存的参考音频"""
    try:
        prompts = []
        if os.path.exists("prompts"):
            for file in os.listdir("prompts"):
                if file.endswith(".wav"):
                    file_path = _safe_resolve_under_dir(Path("prompts"), file)
                    prompts.append({
                        "filename": file,
                        "path": f"prompts/{file}",
                        "size_bytes": os.path.getsize(str(file_path)),
                        "size_mb": round(os.path.getsize(str(file_path)) / 1024 / 1024, 2)
                    })
        return {"prompts": prompts, "count": len(prompts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取列表失败: {str(e)}")


@app.delete("/api/v1/prompts/{filename}")
async def delete_prompt(filename: str) -> Dict[str, Any]:
    """删除指定的参考音频"""
    try:
        file_path = _resolve_download_audio(Path("prompts"), filename)
         
        file_path.unlink()
        return {
            "success": True,
            "message": "参考音频删除成功",
            "filename": filename
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@app.get("/api/v1/outputs/{filename}")
async def download_output(filename: str):
    """下载生成的音频文件"""
    file_path = _resolve_download_audio(Path("outputs"), filename)
    return FileResponse(str(file_path), media_type="audio/wav", filename=file_path.name)


# ==================== 主函数 ====================
if __name__ == "__main__":
    args = _parse_server_args()
    
    # 显示CUDA kernel状态
    cuda_kernel_status = "自动检测" if args.cuda_kernel is None else ("启用" if args.cuda_kernel else "禁用")
    
    print(f"""
╔══════════════════════════════════════════════════╗
║        IndexTTS2 API Server v2.0.0               ║
╚══════════════════════════════════════════════════╝

🌐 服务地址: http://{args.host}:{args.port}
📖 API文档: http://{args.host}:{args.port}/docs
🔧 模型目录: {args.model_dir}
💾 输出目录: service/outputs/
⚙️  加速配置:
   - FP16: {'启用' if args.fp16 else '禁用'}
   - DeepSpeed: {'启用' if args.deepspeed else '禁用'}
   - CUDA Kernel: {cuda_kernel_status}

启动中...
    """)
    
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )
