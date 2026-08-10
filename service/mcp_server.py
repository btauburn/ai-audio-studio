#!/usr/bin/env python3
"""
IndexTTS2 MCP (Model Context Protocol) 服务器
供Claude Desktop和Claude Code调用
"""

import os
import sys
import json
import asyncio
import argparse
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import uuid
from pathlib import Path
import signal
from urllib.parse import urlparse
from urllib.request import url2pathname

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "indextts"))

from indextts.infer_v2 import IndexTTS2  # noqa: E402

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).with_name("mcp_server.log"), encoding="utf-8", delay=True),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)


def _file_uri_to_path(uri: str) -> Optional[Path]:
    if not isinstance(uri, str) or not uri:
        return None
    try:
        parsed = urlparse(uri)
    except Exception:
        return None
    if parsed.scheme != "file":
        return None
    if parsed.path:
        return Path(url2pathname(parsed.path))
    if parsed.netloc:
        return Path(url2pathname(parsed.netloc))
    return None


class MCPServer:
    """MCP协议服务器"""

    def __init__(self, model_dir: str = "models/index_tts", use_fp16: bool = True, use_cuda_kernel: Optional[bool] = None):
        # 使用pathlib处理路径
        self.model_dir = Path(project_root) / model_dir
        logger.info(f"初始化MCP服务器，模型目录: {self.model_dir}")

        # 创建必要的文件夹（使用pathlib）
        service_dir = Path(__file__).parent
        self.output_dir = service_dir / "mcp_outputs"
        self.prompts_dir = service_dir / "mcp_prompts"
        self.temp_dir = service_dir / "mcp_temp"

        for folder in [self.output_dir, self.prompts_dir, self.temp_dir]:
            folder.mkdir(exist_ok=True)

        self.use_fp16 = use_fp16
        self.use_cuda_kernel = use_cuda_kernel

        # 初始化TTS模型
        try:
            logger.info("🔧 正在加载IndexTTS2模型...")
            self.tts = IndexTTS2(
                model_dir=str(self.model_dir),
                cfg_path=str(self.model_dir / "config.yaml"),
                use_fp16=use_fp16,
                use_cuda_kernel=use_cuda_kernel
            )
            logger.info("✅ 模型加载成功")
            logger.info(f"设备: {getattr(self.tts, 'device', 'unknown')}")
            logger.info(f"FP16: {use_fp16}, CUDA Kernel: {use_cuda_kernel}")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理MCP请求"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        logger.info(f"收到请求: {method}")

        try:
            if method == "initialize":
                response = self._initialize(params)
            elif method == "tools/list":
                response = self._list_tools()
            elif method == "tools/call":
                response = await self._call_tool(params)
            elif method == "resources/list":
                response = self._list_resources()
            elif method == "resources/read":
                response = self._read_resource(params)
            else:
                response = {"error": {"code": -32601, "message": f"Unknown method: {method}"}}

            if request_id is not None:
                response["id"] = request_id

            return response

        except Exception as e:
            logger.error(f"处理请求出错: {e}", exc_info=True)
            return {
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }

    def _initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """初始化响应"""
        return {
            "result": {
                "protocolVersion": "1.0.0",
                "serverInfo": {
                    "name": "indextts2",
                    "version": "2.0.0"
                },
                "capabilities": {
                    "tools": {},
                    "resources": {}
                }
            }
        }

    def _list_tools(self) -> Dict[str, Any]:
        """列出可用工具"""
        return {
            "result": {
                "tools": [
                    {
                        "name": "synthesize_speech",
                        "description": "【完整版语音合成】使用IndexTTS2从参考音频克隆音色并生成语音。支持4种情感控制模式+10种高级生成参数（温度、采样、重复惩罚等），可以精确控制语音质量和情感表达。适合需要完全自定义参数的高级场景。参考音频只需3-10秒即可克隆任何人的声音。",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "要合成的文本内容（支持中英文）"
                                },
                                "speaker_audio_path": {
                                    "type": "string",
                                    "description": "说话人参考音频文件路径，用于克隆音色"
                                },
                                "emotion_mode": {
                                    "type": "string",
                                    "enum": ["speaker", "audio", "vector", "text"],
                                    "description": "情感控制模式: speaker=使用说话人情感, audio=使用情感参考音频, vector=使用情感向量, text=使用文本描述",
                                    "default": "speaker"
                                },
                                "emotion_audio_path": {
                                    "type": "string",
                                    "description": "情感参考音频路径（emotion_mode=audio时使用）"
                                },
                                "emotion_text": {
                                    "type": "string",
                                    "description": "情感描述文本，如'高兴的'、'悲伤的'等（emotion_mode=text时使用）"
                                },
                                "emotion_vector": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": "情感向量 [喜,怒,哀,惧,厌恶,低落,惊喜,平静]，8个0-1之间的数值"
                                },
                                "emotion_alpha": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "default": 0.65,
                                    "description": "情感强度：控制情感影响程度，0.65为API优化值（IndexTTS2原始默认1.0），适合大多数场景"
                                },
                                "use_random": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "是否使用随机情感采样"
                                },
                                "max_text_tokens_per_segment": {
                                    "type": "number",
                                    "default": 120,
                                    "minimum": 20,
                                    "maximum": 600,
                                    "description": "每段最大文本token数：控制长文本如何分段处理，较小值分段更细，适合短句；较大值适合长段落"
                                },
                                "max_mel_tokens": {
                                    "type": "number",
                                    "default": 1500,
                                    "minimum": 50,
                                    "maximum": 2000,
                                    "description": "最大mel token数：限制生成音频的最大长度，避免过长导致质量下降"
                                },
                                "do_sample": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "是否使用采样：启用后语音更自然多变，禁用则更稳定但可能机械"
                                },
                                "temperature": {
                                    "type": "number",
                                    "minimum": 0.1,
                                    "maximum": 2.0,
                                    "default": 0.8,
                                    "description": "生成温度：控制随机性，0.7=稳定一致，0.8=平衡（推荐），0.9=多样性高"
                                },
                                "top_p": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "default": 0.8,
                                    "description": "核采样参数(top_p)：控制采样多样性，0.8-0.9为推荐范围，越高越多样"
                                },
                                "top_k": {
                                    "type": "number",
                                    "default": 30,
                                    "minimum": 0,
                                    "maximum": 100,
                                    "description": "Top-K采样：限制候选token数量，15-30为保守值，50+为激进值，0表示不限制"
                                },
                                "num_beams": {
                                    "type": "number",
                                    "default": 3,
                                    "minimum": 1,
                                    "maximum": 10,
                                    "description": "Beam搜索宽度：越大质量越高但速度越慢，3-5为推荐范围"
                                },
                                "repetition_penalty": {
                                    "type": "number",
                                    "default": 10.0,
                                    "minimum": 0.1,
                                    "maximum": 20.0,
                                    "description": "重复惩罚：防止重复生成。默认10.0较高，强烈建议使用 1.5-2.0（如 1.8）以获得更流畅的语音。只有出现重复问题时才提高值"
                                },
                                "length_penalty": {
                                    "type": "number",
                                    "default": 0.0,
                                    "minimum": -2.0,
                                    "maximum": 2.0,
                                    "description": "长度惩罚：正值鼓励生成更长的语音，负值鼓励更短，0为中性（推荐）"
                                },
                                "interval_silence": {
                                    "type": "number",
                                    "default": 200,
                                    "minimum": 0,
                                    "maximum": 2000,
                                    "description": "分段间隔静音时长(毫秒)：控制长文本分段间的停顿时间，影响语音的自然度和节奏感"
                                },
                                "output_filename": {
                                    "type": "string",
                                    "description": "输出文件名（可选，默认自动生成）"
                                }
                            },
                            "required": ["text", "speaker_audio_path"]
                        }
                    },
                    {
                        "name": "clone_voice",
                        "description": "【快速音色克隆】最简单的语音生成方式，自动使用最佳参数。只需提供文本和参考音频，即可克隆音色并生成语音。适合快速生成、日常对话等场景。",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "要说的内容（支持中英文）"
                                },
                                "speaker_audio": {
                                    "type": "string",
                                    "description": "参考音频的绝对路径，用于克隆音色（建议3-10秒清晰音频）"
                                }
                            },
                            "required": ["text", "speaker_audio"]
                        }
                    },
                    {
                        "name": "emotional_speech",
                        "description": "【情感语音生成】生成带有特定情感的语音。支持8种基本情感：快乐、愤怒、悲伤、恐惧、厌恶、忧郁、惊讶、平静。可以控制情感强度。适合需要表达明确情感的场景。",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "要说的内容"
                                },
                                "speaker_audio": {
                                    "type": "string",
                                    "description": "参考音频的绝对路径"
                                },
                                "emotion": {
                                    "type": "string",
                                    "enum": ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"],
                                    "description": "选择一种情感：happy=快乐/高兴, angry=愤怒/生气, sad=悲伤/难过, afraid=恐惧/害怕, disgusted=厌恶/反感, melancholic=忧郁/低落, surprised=惊讶/惊喜, calm=平静/冷静"
                                },
                                "emotion_intensity": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "default": 0.8,
                                    "description": "情感强度：0.0=几乎无情感，0.5=适中，0.8=明显（推荐），1.0=极强"
                                }
                            },
                            "required": ["text", "speaker_audio", "emotion"]
                        }
                    },
                    {
                        "name": "mixed_emotion_speech",
                        "description": "【混合情感语音】同时混合多种情感，创造更细腻的情感表达。例如：60%快乐+40%惊讶，或30%悲伤+70%平静。适合表达复杂情感，如又惊又喜、忧伤但平静等。",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "要说的内容"
                                },
                                "speaker_audio": {
                                    "type": "string",
                                    "description": "参考音频的绝对路径"
                                },
                                "emotion_mix": {
                                    "type": "object",
                                    "description": "情感混合配置：key是情感名称（happy/angry/sad/afraid/disgusted/melancholic/surprised/calm），value是权重（0-1）。建议总和为1.0",
                                    "example": {"happy": 0.6, "surprised": 0.4}
                                }
                            },
                            "required": ["text", "speaker_audio", "emotion_mix"]
                        }
                    },
                    {
                        "name": "batch_synthesize",
                        "description": "【批量语音合成】一次性生成多段语音，所有音频使用相同的音色。非常适合生成有声书章节、播客片段、教学材料等长内容。相比逐个生成更高效。",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text_list": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "要合成的文本列表"
                                },
                                "speaker_audio": {
                                    "type": "string",
                                    "description": "参考音频路径"
                                },
                                "emotion_mode": {
                                    "type": "string",
                                    "default": "speaker",
                                    "description": "情感控制模式"
                                }
                            },
                            "required": ["text_list", "speaker_audio"]
                        }
                    },
                    {
                        "name": "list_saved_prompts",
                        "description": "【列出参考音频】查看所有已保存的参考音频文件。这些文件可以作为speaker_audio参数使用，无需重复上传。返回文件名、路径和大小信息。",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "description": "无需参数"
                        }
                    },
                    {
                        "name": "list_generated_audio",
                        "description": "【列出生成的音频】查看最近生成的音频文件列表。可以用来查看历史生成记录、找回之前的音频、或管理输出文件。按时间倒序排列。",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "limit": {
                                    "type": "number",
                                    "default": 10,
                                    "description": "返回数量限制（默认10个，最多显示最近的N个文件）"
                                }
                            }
                        }
                    }
                ]
            }
        }

    async def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        logger.info(f"调用工具: {tool_name}")

        try:
            if tool_name == "synthesize_speech":
                result = await self._synthesize_speech(arguments)
            elif tool_name == "clone_voice":
                result = await self._clone_voice(arguments)
            elif tool_name == "emotional_speech":
                result = await self._emotional_speech(arguments)
            elif tool_name == "mixed_emotion_speech":
                result = await self._mixed_emotion_speech(arguments)
            elif tool_name == "batch_synthesize":
                result = await self._batch_synthesize(arguments)
            elif tool_name == "list_saved_prompts":
                result = await self._list_saved_prompts(arguments)
            elif tool_name == "list_generated_audio":
                result = await self._list_generated_audio(arguments)
            else:
                return {"error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}

            return {
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, ensure_ascii=False, indent=2)
                        }
                    ]
                }
            }

        except Exception as e:
            logger.error(f"工具调用失败: {e}", exc_info=True)
            return {
                "error": {
                    "code": -32603,
                    "message": f"Tool execution failed: {str(e)}"
                }
            }

    def _validate_text(self, text: str) -> None:
        """验证文本输入"""
        if not text or not text.strip():
            raise ValueError("文本不能为空")
        if len(text) > 10000:
            raise ValueError("文本过长，最多10000字符")

    def _validate_audio_path(self, path: str) -> None:
        """验证音频路径"""
        if not Path(path).exists():
            raise FileNotFoundError(f"音频文件不存在: {path}")
        if not path.endswith(('.wav', '.mp3', '.flac')):
            raise ValueError("音频格式必须是wav/mp3/flac")

    async def _synthesize_speech(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """完整的语音合成"""
        text = args["text"]
        speaker_audio = args["speaker_audio_path"]

        # 验证输入
        self._validate_text(text)
        self._validate_audio_path(speaker_audio)

        emotion_mode = args.get("emotion_mode", "speaker")
        output_filename = args.get("output_filename", f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")

        if not output_filename.endswith('.wav'):
            output_filename += '.wav'

        output_path = str(self.output_dir / output_filename)

        # 映射情感模式
        mode_map = {"speaker": 0, "audio": 1, "vector": 2, "text": 3}
        emo_mode_int = mode_map.get(emotion_mode, 0)

        # 准备参数（完整版）
        kwargs = {
            "spk_audio_prompt": speaker_audio,
            "text": text,
            "output_path": output_path,
            # 基础参数
            "verbose": False,
            # 生成参数
            "use_random": args.get("use_random", False),
            "max_text_tokens_per_segment": args.get("max_text_tokens_per_segment", 120),
            "interval_silence": args.get("interval_silence", 200),
            "do_sample": args.get("do_sample", True),
            "temperature": args.get("temperature", 0.8),
            "top_p": args.get("top_p", 0.8),
            "top_k": args.get("top_k", 30) if args.get("top_k", 30) > 0 else None,  # 与WebUI一致
            "num_beams": args.get("num_beams", 3),
            "repetition_penalty": args.get("repetition_penalty", 10.0),
            "length_penalty": args.get("length_penalty", 0.0),
            "max_mel_tokens": args.get("max_mel_tokens", 1500),
        }

        emotion_alpha = args.get("emotion_alpha", 0.65)

        # 情感控制参数（与WebUI完全一致）
        if emo_mode_int == 0:
            # emotion_mode=0: 使用说话人情感，移除外部情感参考
            kwargs["emo_audio_prompt"] = None
        elif emo_mode_int == 1:
            emo_audio_path = args.get("emotion_audio_path")
            if not isinstance(emo_audio_path, str) or not emo_audio_path:
                raise ValueError("emotion_mode=audio 时必须提供 emotion_audio_path")
            self._validate_audio_path(emo_audio_path)
            kwargs["emo_audio_prompt"] = emo_audio_path
            kwargs["emo_alpha"] = emotion_alpha
        elif emo_mode_int == 2:
            raw_vec = args.get("emotion_vector")
            if not isinstance(raw_vec, list) or len(raw_vec) != 8:
                raise ValueError("emotion_mode=vector 时必须提供 8 维 emotion_vector")
            if not all(isinstance(x, (int, float)) and 0 <= float(x) <= 1 for x in raw_vec):
                raise ValueError("emotion_vector 每个值必须在 0-1 之间")
            # ✅ 关键修复：调用normalize_emo_vec与WebUI保持一致
            normalized_vec = self.tts.normalize_emo_vec([float(x) for x in raw_vec], apply_bias=True)
            logger.info(f"Normalized emotion vector: {normalized_vec}")
            kwargs["emo_vector"] = normalized_vec
            kwargs["emo_alpha"] = emotion_alpha
        elif emo_mode_int == 3:
            emo_text = args.get("emotion_text")
            if isinstance(emo_text, str) and emo_text.strip() == "":
                emo_text = None
            elif not isinstance(emo_text, str):
                emo_text = None
            kwargs["emo_audio_prompt"] = None
            kwargs["use_emo_text"] = True
            kwargs["emo_text"] = emo_text
            kwargs["emo_alpha"] = emotion_alpha

        # 执行合成
        self.tts.infer(**kwargs)

        return {
            "success": True,
            "output_file": output_path,
            "message": "✅ 语音已生成",
            "details": {
                "text": text,
                "emotion_mode": emotion_mode,
                "output_path": output_path
            }
        }

    async def _clone_voice(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """简化的音色克隆"""
        text = args["text"]
        speaker_audio = args["speaker_audio"]

        # 验证输入
        self._validate_text(text)
        self._validate_audio_path(speaker_audio)

        output_path = str(self.output_dir / f"clone_{uuid.uuid4().hex[:8]}.wav")

        self.tts.infer(
            spk_audio_prompt=speaker_audio,
            text=text,
            output_path=output_path,
            verbose=False
        )

        return {
            "success": True,
            "output_file": output_path,
            "message": f"✅ 音色克隆完成: {output_path}",
            "text": text
        }

    async def _emotional_speech(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """情感语音生成"""
        text = args["text"]
        speaker_audio = args["speaker_audio"]
        emotion = args["emotion"]
        intensity = args.get("emotion_intensity", 0.8)

        # 验证输入
        self._validate_text(text)
        self._validate_audio_path(speaker_audio)

        if emotion not in ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]:
            raise ValueError(f"未知的情感类型: {emotion}")

        if not 0 <= intensity <= 1:
            raise ValueError("情感强度必须在0-1之间")

        # 情感向量映射
        emotion_vectors = {
            "happy": [1, 0, 0, 0, 0, 0, 0, 0],
            "angry": [0, 1, 0, 0, 0, 0, 0, 0],
            "sad": [0, 0, 1, 0, 0, 0, 0, 0],
            "afraid": [0, 0, 0, 1, 0, 0, 0, 0],
            "disgusted": [0, 0, 0, 0, 1, 0, 0, 0],
            "melancholic": [0, 0, 0, 0, 0, 1, 0, 0],
            "surprised": [0, 0, 0, 0, 0, 0, 1, 0],
            "calm": [0, 0, 0, 0, 0, 0, 0, 1]
        }

        emo_vec = [v * intensity for v in emotion_vectors[emotion]]
        # ✅ 关键修复：调用normalize_emo_vec与WebUI保持一致
        normalized_vec = self.tts.normalize_emo_vec(emo_vec, apply_bias=True)
        logger.info(f"Emotional speech - normalized vector: {normalized_vec}")

        output_path = str(self.output_dir / f"{emotion}_{uuid.uuid4().hex[:8]}.wav")

        self.tts.infer(
            spk_audio_prompt=speaker_audio,
            text=text,
            output_path=output_path,
            emo_vector=normalized_vec,  # 使用规范化后的向量
            emo_alpha=0.8,
            verbose=False
        )

        return {
            "success": True,
            "output_file": output_path,
            "emotion": emotion,
            "intensity": intensity,
            "message": f"✅ {emotion}情感语音已生成: {output_path}"
        }

    async def _mixed_emotion_speech(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """混合情感语音"""
        text = args["text"]
        speaker_audio = args["speaker_audio"]
        emotion_mix = args["emotion_mix"]

        # 验证输入
        self._validate_text(text)
        self._validate_audio_path(speaker_audio)

        if not isinstance(emotion_mix, dict):
            raise ValueError("emotion_mix必须是字典类型")

        emotion_map = {
            "happy": 0, "angry": 1, "sad": 2, "afraid": 3,
            "disgusted": 4, "melancholic": 5, "surprised": 6, "calm": 7
        }

        # 构建情感向量
        emo_vec = [0.0] * 8
        for emotion_name, weight in emotion_mix.items():
            if emotion_name in emotion_map:
                emo_vec[emotion_map[emotion_name]] = weight

        # ✅ 关键修复：调用normalize_emo_vec与WebUI保持一致
        normalized_vec = self.tts.normalize_emo_vec(emo_vec, apply_bias=True)
        logger.info(f"Mixed emotion speech - normalized vector: {normalized_vec}")

        output_path = str(self.output_dir / f"mixed_{uuid.uuid4().hex[:8]}.wav")

        self.tts.infer(
            spk_audio_prompt=speaker_audio,
            text=text,
            output_path=output_path,
            emo_vector=normalized_vec,  # 使用规范化后的向量
            emo_alpha=0.8,
            verbose=False
        )

        return {
            "success": True,
            "output_file": output_path,
            "emotion_mix": emotion_mix,
            "emotion_vector": emo_vec,
            "message": f"✅ 混合情感语音已生成: {output_path}"
        }

    async def _batch_synthesize(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """批量合成"""
        text_list = args["text_list"]
        speaker_audio = args["speaker_audio"]

        # 验证输入
        if not isinstance(text_list, list):
            raise ValueError("text_list必须是列表类型")
        if len(text_list) == 0:
            raise ValueError("text_list不能为空")
        if len(text_list) > 100:
            raise ValueError("批量合成最多支持100条")

        self._validate_audio_path(speaker_audio)

        results = []
        for i, text in enumerate(text_list):
            output_path = str(self.output_dir / f"batch_{i+1}_{uuid.uuid4().hex[:6]}.wav")

            try:
                self.tts.infer(
                    spk_audio_prompt=speaker_audio,
                    text=text,
                    output_path=output_path,
                    verbose=False
                )
                results.append({
                    "index": i + 1,
                    "success": True,
                    "text": text,
                    "output_file": output_path
                })
            except Exception as e:
                results.append({
                    "index": i + 1,
                    "success": False,
                    "text": text,
                    "error": str(e)
                })

        success_count = sum(1 for r in results if r["success"])

        return {
            "total": len(text_list),
            "success_count": success_count,
            "failed_count": len(text_list) - success_count,
            "results": results,
            "message": f"✅ 批量合成完成: {success_count}/{len(text_list)} 成功"
        }

    async def _list_saved_prompts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """列出保存的参考音频"""
        prompts = []
        if self.prompts_dir.exists():
            for file in self.prompts_dir.glob("*.wav"):
                prompts.append({
                    "filename": file.name,
                    "path": str(file),
                    "size_mb": round(file.stat().st_size / 1024 / 1024, 2)
                })

        return {
            "count": len(prompts),
            "prompts": prompts
        }

    async def _list_generated_audio(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """列出生成的音频"""
        limit = min(args.get("limit", 10), 100)  # 最多100个
        audio_files = []

        if self.output_dir.exists():
            files = sorted(
                self.output_dir.glob("*.wav"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            for file in files[:limit]:
                audio_files.append({
                    "filename": file.name,
                    "path": str(file),
                    "size_mb": round(file.stat().st_size / 1024 / 1024, 2),
                    "created": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })

        return {
            "count": len(audio_files),
            "audio_files": audio_files
        }

    def _list_resources(self) -> Dict[str, Any]:
        """列出可用资源"""
        return {
            "result": {
                "resources": [
                    {
                        "uri": self.output_dir.resolve().as_uri(),
                        "name": "生成的语音文件",
                        "description": "所有通过MCP生成的语音文件",
                        "mimeType": "text/plain"
                    },
                    {
                        "uri": self.prompts_dir.resolve().as_uri(),
                        "name": "参考音频文件",
                        "description": "保存的参考音频",
                        "mimeType": "text/plain"
                    }
                ]
            }
        }

    def _read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """读取资源"""
        uri = params.get("uri", "")

        target_dir: Optional[Path] = None
        if isinstance(uri, str):
            if uri.startswith("file://mcp_outputs"):
                target_dir = self.output_dir
            elif uri.startswith("file://mcp_prompts"):
                target_dir = self.prompts_dir
            else:
                path = _file_uri_to_path(uri)
                if path is not None:
                    try:
                        path_norm = os.path.normcase(str(path.resolve()))
                    except Exception:
                        path_norm = os.path.normcase(str(path))

                    try:
                        output_norm = os.path.normcase(str(self.output_dir.resolve()))
                        prompts_norm = os.path.normcase(str(self.prompts_dir.resolve()))
                    except Exception:
                        output_norm = os.path.normcase(str(self.output_dir))
                        prompts_norm = os.path.normcase(str(self.prompts_dir))

                    if path_norm == output_norm:
                        target_dir = self.output_dir
                    elif path_norm == prompts_norm:
                        target_dir = self.prompts_dir

        if target_dir is None:
            return {"error": {"code": -32602, "message": "Invalid resource URI"}}

        files = self._list_files(str(target_dir))
        return {
            "result": {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "text/plain",
                        "text": json.dumps(files, ensure_ascii=False, indent=2)
                    }
                ]
            }
        }

    def _list_files(self, directory: str) -> List[Dict[str, Any]]:
        """列出目录中的文件"""
        dir_path = Path(directory)
        if not dir_path.exists():
            return []

        files = []
        for file in dir_path.glob("*.wav"):
            files.append({
                "name": file.name,
                "path": str(file),
                "size": file.stat().st_size
            })
        
        return files
    
    def cleanup(self) -> None:
        """清理临时文件"""
        logger.info("🔄 正在清理临时文件...")
        if self.temp_dir.exists():
            for file in self.temp_dir.glob("*.wav"):
                try:
                    file.unlink()
                    logger.debug(f"删除临时文件: {file.name}")
                except Exception as e:
                    logger.warning(f"无法删除文件 {file.name}: {e}")
        logger.info("✅ 临时文件清理完成")


async def main():
    """主函数 - stdio通信"""
    global server_instance
    parser = argparse.ArgumentParser(description="IndexTTS2 MCP Server")
    parser.add_argument("--model_dir", type=str, default="models/index_tts", help="模型目录")
    parser.add_argument("--fp16", action="store_true", help="使用FP16")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="禁用FP16")
    parser.add_argument("--cuda-kernel", dest="cuda_kernel", action="store_true", help="强制使用CUDA kernel")
    parser.add_argument("--no-cuda-kernel", dest="cuda_kernel", action="store_false", help="禁用CUDA kernel")
    parser.set_defaults(fp16=True, cuda_kernel=None)  # 默认FP16=True，cuda_kernel自动检测
    args = parser.parse_args()
    
    # 初始化服务器
    server_instance = MCPServer(
        model_dir=args.model_dir,
        use_fp16=args.fp16,
        use_cuda_kernel=args.cuda_kernel
    )
    server = server_instance
    
    # 注册信号处理
    def signal_handler(signum, frame):
        logger.info(f"收到信号 {signum}，准备关闭...")
        server.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("✅ MCP服务器已启动，等待请求...")
    
    # 标准输入输出通信
    while True:
        try:
            # 从stdin读取JSON-RPC请求
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            
            if not line:
                logger.info("输入流关闭，退出服务器")
                break
            
            line = line.strip()
            if not line:
                continue
            
            # 解析请求
            request = json.loads(line)
            
            # 处理请求
            response = await server.handle_request(request)
            
            # 发送响应到stdout
            print(json.dumps(response, ensure_ascii=False), flush=True)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            error_response = {
                "error": {
                    "code": -32700,
                    "message": "Parse error"
                }
            }
            print(json.dumps(error_response), flush=True)
        
        except Exception as e:
            logger.error(f"未预期的错误: {e}", exc_info=True)
            error_response = {
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
            print(json.dumps(error_response), flush=True)


server_instance = None


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️  服务器被用户中断")
        if server_instance:
            server_instance.cleanup()
    except Exception as e:
        logger.error(f"❌ 服务器错误: {e}", exc_info=True)
        if server_instance:
            server_instance.cleanup()
        sys.exit(1)
    finally:
        if server_instance:
            server_instance.cleanup()
            logger.info("👋 MCP服务器已关闭")
