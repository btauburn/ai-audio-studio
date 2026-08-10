# TTS 工作进程说明（当前仓库集成入口）

`service/` 现在是配音平台的内部推理工作进程层，不是用户启动整个平台的入口。生产用户先运行根目录 `start_platform.bat` 或用根 `.venv` 启动 `dubbing_service.app.main:app`，再由平台配置页按需启动一个 TTS 后端。

当前解释器和端口边界：

| 后端 | 脚本 | 解释器 | 默认端口 |
| --- | --- | --- | ---: |
| IndexTTS2 | `server.py` | 根 `.venv/Scripts/python.exe` | 8001 |
| CosyVoice3 | `cosyvoice_server.py` | `environments/cosyvoice/.venv/Scripts/python.exe` | 8002 |
| Qwen VoiceDesign | `qwen3tts_server.py` | `environments/qwen3tts/.venv/Scripts/python.exe` | 8003 |
| Qwen CustomVoice | `qwen3tts_customvoice_server.py` | `environments/qwen3tts/.venv/Scripts/python.exe` | 8004 |

三个 uv 环境只能通过根 `scripts/bootstrap.ps1` 重建。不要在 `service/` 内运行 requirements/pip 把 CosyVoice 或 Qwen 装进根环境，也不要依赖旧 `.venv_qwen3tts`。

平台维护的精确命令由 `dubbing_service/app/core/tts_backends.py` 生成，并强制使用项目内 `models/`。运行时设置 Hugging Face/Transformers 离线变量；模型缺失时使用 `scripts/model_provision.py` 的 manifest + ModelScope 恢复流程，不允许服务启动时自动下载。

需要隔离故障时，可从仓库根目录手工启动；通常不要同时运行多个 GPU 模型：

```powershell
.\.venv\Scripts\python.exe -u .\service\server.py --host 127.0.0.1 --port 8001 --model_dir .\models\index_tts --fp16 --no-cuda-kernel

.\environments\cosyvoice\.venv\Scripts\python.exe -u .\service\cosyvoice_server.py --host 127.0.0.1 --port 8002 --model_dir .\models\cosyvoice3\Fun-CosyVoice3-0.5B

.\environments\qwen3tts\.venv\Scripts\python.exe -u .\service\qwen3tts_server.py --host 127.0.0.1 --port 8003 --model_dir .\models\qwen3_tts\Qwen3-TTS-VoiceDesign --device cuda:0 --dtype bfloat16

.\environments\qwen3tts\.venv\Scripts\python.exe -u .\service\qwen3tts_customvoice_server.py --host 127.0.0.1 --port 8004 --model_dir .\models\qwen3_tts\Qwen3-TTS-CustomVoice --device cuda:0 --dtype bfloat16
```

每个服务提供 `/health`、模型 load/unload 和 `/api/v1/tts/synthesize`。工作进程只绑定 loopback；对外业务入口是 9000 平台。完整生产操作见 [`../docs/operations.md`](../docs/operations.md)。

> 下方保留的长篇内容主要是 IndexTTS2 单后端 REST/MCP 的历史和高级参考。凡是涉及单一环境、`uv run`、`requirements.txt`、旧模型目录或单独 `start_api.bat` 的内容，都不能覆盖上面的当前多后端运行边界。

---

# IndexTTS2 单后端服务完整指南（高级参考）

> 🎯 **超详细使用指南** - 包含REST API和MCP服务的完整参数说明、使用示例和最佳实践

## 📖 目录

- [项目简介](#项目简介)
- [目录结构](#目录结构)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [核心概念](#核心概念)
- [REST API 详细指南](#rest-api-详细指南)
- [MCP 服务详细指南](#mcp-服务详细指南)
- [参数完全手册](#参数完全手册)
- [不同工具调用示例](#不同工具调用示例)
- [常见问题](#常见问题)
- [性能优化](#性能优化)
- [审查与质量保证](#审查与质量保证)

---

## 项目简介

IndexTTS2 服务提供两种访问方式：

### 🌐 REST API 服务
- **用途**：通用的HTTP API接口
- **适用场景**：Web应用、移动应用、第三方集成
- **端口**：默认 8001
- **文档**：自带 Swagger UI (`/docs`)

### 🤖 MCP 服务 (Model Context Protocol)
- **用途**：AI助手集成协议
- **适用场景**：Claude Desktop、n8n、Dify等AI工具
- **通信**：stdio 标准输入输出
- **特点**：结构化工具调用，更适合AI代理

---

## 目录结构

```
service/
├── server.py                 # REST API 服务器
├── mcp_server.py            # MCP 服务器
├── check_environment.py     # 环境检查工具
├── start_api.bat            # REST API 启动脚本（Windows）
├── start_mcp.bat            # MCP 启动脚本（Windows，仅调试）
├── setup_mcp.py             # Claude Desktop MCP 配置工具（生成/写入配置）
├── validate_config.py       # Claude Desktop MCP 配置验证工具
├── test_mcp.py              # MCP 测试示例
├── outputs/                 # REST API 生成的音频
├── prompts/                 # REST API 参考音频
├── mcp_outputs/             # MCP 生成的音频
├── mcp_prompts/             # MCP 参考音频
├── examples/
│   ├── test_api.py         # REST API 测试示例
│   └── test_api_file_advanced.py  # REST API 文件上传高级测试
└── claude_desktop_config.json  # Claude Desktop 配置示例
```

---

## 环境准备

### 1️⃣ 系统要求

- **Python**: 3.11+
- **GPU**: NVIDIA GPU（推荐，用于加速）
- **内存**: 8GB+ RAM
- **存储**: 5GB+ 可用空间（模型文件）

### 2️⃣ 安装依赖

项目使用 `uv` 进行依赖管理：

```bash
# 进入项目根目录
cd e:\index-tts

# 安装 API/MCP 服务依赖
uv sync --extra api

# 或者安装所有可选依赖
uv sync --all-extras
```

### 3️⃣ 环境检查

运行环境检查工具：

```bash
cd service
uv run check_environment.py
```

**成功输出示例**：
```
✅ Python 版本: 3.11.x
✅ PyTorch: 已安装
✅ CUDA: 可用
✅ FastAPI: 已安装
✅ 模型文件: 完整
✅ 环境检查通过！
```

---

## 快速开始

### 🚀 启动 REST API 服务

```bash
cd service

# 基础启动（CUDA环境自动启用FP16和CUDA Kernel加速）
uv run server.py

# 禁用FP16（不推荐）
uv run server.py --no-fp16

# 禁用CUDA Kernel（不推荐）
uv run server.py --no-cuda-kernel

# 自定义端口
uv run server.py --port 8080 --host 0.0.0.0

# CPU模式（自动检测，无需手动指定）
uv run server.py  # 无CUDA时自动使用CPU
```

Windows 用户也可以直接双击 `service/start_api.bat` 启动；它显式使用根 `.venv`，固定项目内模型/缓存并保持离线，默认 FP16。

如需全精度 FP32（禁用 FP16），可在命令行中运行：
```bash
start_api.bat fp32
```

**验证服务**：
- 浏览器访问：http://localhost:8001/docs
- 健康检查：`curl http://localhost:8001/health`

### 🤖 启动 MCP 服务

MCP服务通过stdio通信，通常由MCP客户端（如Claude Desktop）自动启动。

 **手动测试**：
 ```bash
 cd service
 ..\.venv\Scripts\python.exe test_mcp.py
 ```
 
 **Claude Desktop 配置** (Windows):
 ```json
 {
   "mcpServers": {
     "indextts2": {
       "command": "uv",
       "args": [
         "run",
         "--directory",
         "E:/index-tts/service",
         "E:/index-tts/service/mcp_server.py"
       ],
       "env": {
         "PYTHONPATH": "E:/index-tts"
       }
     }
   }
 }
 ```

推荐使用自动配置工具（会自动备份并写入 Claude Desktop 配置，生成的配置显式使用项目根 `.venv`）：
```bash
.\.venv\Scripts\python.exe .\service\setup_mcp.py
```

验证配置：
```bash
.\.venv\Scripts\python.exe .\service\validate_config.py
```

配置文件路径：`%APPDATA%\Claude\claude_desktop_config.json`

---

## 核心概念

### 🎤 音色克隆 (Voice Cloning)

**原理**：通过3-10秒的参考音频，克隆说话人的音色特征。

**关键参数**：
- `speaker_audio`: 参考音频路径或base64编码
- 音频质量要求：清晰、无背景噪音、单人说话

### 🎭 情感控制 (Emotion Control)

IndexTTS2 支持 **4种情感模式**：

| 模式 | emotion_mode | 说明 | 适用场景 |
|------|--------------|------|----------|
| **说话人情感** | 0 | 从speaker_audio提取情感 | 保持原音色原情感 |
| **情感参考音频** | 1 | 单独指定emotion_audio | 分离音色和情感 |
| **情感向量** | 2 | 使用8维情感向量 | 精确情感控制 |
| **文本描述** | 3 | 文本描述情感（实验性） | 自然语言控制 |

### 📊 8维情感向量

```python
# 格式：[快乐, 愤怒, 悲伤, 恐惧, 厌恶, 忧郁, 惊讶, 平静]
emotion_vector = [1.0, 0, 0, 0, 0, 0, 0, 0]  # 纯粹的快乐

# 混合情感示例
emotion_vector = [0.6, 0, 0, 0, 0, 0, 0.4, 0]  # 60%快乐 + 40%平静
```

**情感索引**：
- 0: Happy (快乐)
- 1: Angry (愤怒)
- 2: Sad (悲伤)
- 3: Afraid (恐惧)
- 4: Disgusted (厌恶)
- 5: Melancholic (忧郁)
- 6: Surprised (惊讶)
- 7: Calm (平静)

### ⚙️ 核心参数快览

以下是最常用的参数及其默认值。更完整的参数说明请参见下文“📋 完整请求参数说明”章节。

| 参数 | 默认值 | 范围 | 说明 | 常用程度 |
|------|--------|------|------|---------|
| `text` | 必需 | - | 要合成的文本 | ⭐⭐⭐⭐⭐ |
| `speaker_audio` | 必需 | - | 参考音频（音色克隆） | ⭐⭐⭐⭐⭐ |
| `temperature` | 0.8 | 0.1-2.0 | 控制生成随机性和多样性 | ⭐⭐⭐⭐ |
| `top_p` | 0.8 | 0.0-1.0 | 核采样，控制多样性 | ⭐⭐⭐ |
| `top_k` | 30 | 0-100 | Top-K采样（不影响语速） | ⭐⭐ |
| `repetition_penalty` | 10.0 | 0.1-20.0 | 重复惩罚（⚠️默认值高，一般建议1.5-2.0） | ⭐⭐⭐ |
| `emotion_alpha` | 0.65 | 0.0-1.0 | 情感强度（IndexTTS2原始默认1.0，API设为0.65更温和） | ⭐⭐ |
| `interval_silence` | 200 | 0-2000 | 分段停顿时间(ms) | ⭐⭐⭐ |

**💡 提示**: 
- 大多数场景使用默认值即可
- `repetition_penalty`默认10.0较高，如语音不流畅可降低到1.5-2.0
- `emotion_alpha`在API中默认0.65（IndexTTS2原始为1.0），已优化为更适合日常使用的值

---

## REST API 详细指南

### 📡 API 端点总览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/` | GET | API 信息 |
| `/api/v1/tts/synthesize` | POST | 语音合成（JSON请求） |
| `/api/v1/tts/synthesize-file` | POST | 语音合成（文件上传） |
| `/api/v1/upload` | POST | 上传音频并返回在服务器上的路径（通用上传接口） |
| `/api/v1/prompts/upload` | POST | 上传并保存命名的参考音频 |
| `/api/v1/prompts/list` | GET | 列出参考音频 |
| `/api/v1/prompts/{filename}` | DELETE | 删除指定参考音频 |
| `/api/v1/outputs/{filename}` | GET | 下载生成的音频 |

**两个合成端点的区别**：

| 特性 | `/synthesize` | `/synthesize-file` |
|------|---------------|-------------------|
| 请求格式 | JSON | multipart/form-data |
| 音频输入 | 文件路径或base64 | 直接上传文件 ✅ |
| 支持参数 | 完整所有参数 ✅ | 完整所有参数 ✅ |
| 返回内容 | JSON元数据 | 音频文件流 |
| 适合场景 | 前端应用、已有文件 | 工具集成、快速测试 |

**💡 提示**：两个端点功能完全相同，只是输入和输出方式不同。选择最适合您使用场景的端点即可。

### 🔥 REST API 使用示例

#### 示例1：基础语音合成（文件路径）

```python
import requests

response = requests.post("http://localhost:8001/api/v1/tts/synthesize", json={
    "text": "你好，这是IndexTTS2的语音合成测试。",
    "speaker_audio": "E:/audio/my_voice.wav",  # 绝对路径
    "save_file": True
})

result = response.json()
print(f"生成成功: {result['audio_file']}")
# 输出：生成成功: outputs/tts_20241113_120530.wav
```

#### 示例2：使用 Base64 编码音频

```python
import requests
import base64

# 读取音频并转为base64
with open("my_voice.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8001/api/v1/tts/synthesize", json={
    "text": "这是使用base64编码的音频参考。",
    "speaker_audio": audio_base64,  # base64字符串
    "save_file": False,             # 不保存到服务器
    "return_base64": True           # 返回base64格式
})

result = response.json()

# 保存返回的音频
if result.get("audio_base64"):
    with open("output.wav", "wb") as f:
        f.write(base64.b64decode(result["audio_base64"]))
    print("音频已保存到 output.wav")
```

#### 示例3：情感向量控制

```python
import requests

# 测试不同情感
emotions = {
    "高兴": [1.0, 0, 0, 0, 0, 0, 0, 0],
    "愤怒": [0, 1.0, 0, 0, 0, 0, 0, 0],
    "悲伤": [0, 0, 1.0, 0, 0, 0, 0, 0],
    "平静": [0, 0, 0, 0, 0, 0, 0, 1.0],
    "混合（惊喜+快乐）": [0.6, 0, 0, 0, 0, 0, 0.4, 0]
}

for emotion_name, emotion_vec in emotions.items():
    response = requests.post("http://localhost:8001/api/v1/tts/synthesize", json={
        "text": f"这是{emotion_name}的情感测试",
        "speaker_audio": "E:/audio/voice.wav",
        "emotion_mode": 2,           # 使用情感向量模式
        "emotion_vector": emotion_vec,
        "emotion_alpha": 0.8,        # 情感强度
        "save_file": True
    })
    
    result = response.json()
    print(f"{emotion_name}: {result['audio_file']}")
```

#### 示例4：分离音色和情感

```python
import requests

response = requests.post("http://localhost:8001/api/v1/tts/synthesize", json={
    "text": "用A的音色，配上B的情感",
    "speaker_audio": "E:/audio/person_A.wav",    # 音色来源
    "emotion_mode": 1,                            # 情感参考音频模式
    "emotion_audio": "E:/audio/person_B_happy.wav",  # 情感来源
    "emotion_alpha": 0.7,                        # 情感强度
    "save_file": True
})

result = response.json()
print(f"生成: {result['audio_file']}")
```

#### 示例5：文件上传方式

```python
import requests

with open("my_voice.wav", "rb") as f:
    files = {"speaker_audio": ("voice.wav", f, "audio/wav")}
    data = {
        "text": "这是文件上传方式的测试",
        "emotion_mode": 0,
        "temperature": 0.85
    }
    
    response = requests.post(
        "http://localhost:8001/api/v1/tts/synthesize-file",
        files=files,
        data=data
    )

# 直接获取音频数据
with open("output.wav", "wb") as f:
    f.write(response.content)
print("音频已保存")
```

#### 示例5b：文件上传 + 完整参数控制

```python
import requests
import json

# 使用文件上传方式，同样支持所有高级参数！
with open("my_voice.wav", "rb") as f:
    files = {"speaker_audio": ("voice.wav", f, "audio/wav")}
    data = {
        "text": "这是带有完整参数控制的文件上传测试",
        "emotion_mode": 2,
        "emotion_vector": json.dumps([0.8, 0, 0, 0, 0, 0, 0.2, 0]),  # 80%快乐 + 20%平静
        "emotion_alpha": 0.75,
        "temperature": 0.7,          # 更稳定
        "top_k": 15,                 # 更保守
        "top_p": 0.85,
        "repetition_penalty": 1.8,   # 更强的重复惩罚
        "max_mel_tokens": 1200,      # 限制长度
    }
    
    response = requests.post(
        "http://localhost:8001/api/v1/tts/synthesize-file",
        files=files,
        data=data
    )

with open("output_advanced.wav", "wb") as f:
    f.write(response.content)
print("✅ 高质量音频已生成！")
```

**💡 重要**：`/synthesize-file` 现在支持所有参数，包括：
- ✅ `emotion_vector`（需要JSON字符串格式）
- ✅ `max_mel_tokens`、`top_k`、`repetition_penalty` 等所有生成参数
- ✅ 与 `/synthesize` 功能完全对等

#### 示例6：上传和管理参考音频

```python
import requests

# 1. 上传参考音频（推荐使用通用上传接口）
with open("my_voice.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8001/api/v1/upload",
        files={"audio": ("my_voice.wav", f, "audio/wav")}
    )
    result = response.json()
    prompt_path = result["path"]
    print(f"上传成功: {prompt_path}")
    # 输出：上传成功: prompts/xxxxxxxx.wav

# 2. 列出所有参考音频
response = requests.get("http://localhost:8001/api/v1/prompts/list")
prompts = response.json()
print(f"找到 {prompts['count']} 个参考音频:")
for p in prompts['prompts']:
    print(f"  - {p['filename']} ({p['size_mb']} MB)")

# 3. 使用已上传的参考音频
response = requests.post("http://localhost:8001/api/v1/tts/synthesize", json={
    "text": "使用已保存的参考音频",
    "speaker_audio": prompt_path,  # 使用之前上传的音频
    "save_file": True
})
```

#### 示例7：长文本合成（调整分段停顿）

```python
import requests

# 长文本会自动分段合成，可以控制分段间的停顿时间
response = requests.post("http://localhost:8001/api/v1/tts/synthesize", json={
    "text": "这是第一段。这是第二段。这是第三段。这是第四段。",
    "speaker_audio": "E:/audio/voice.wav",
    
    # 分段控制
    "max_text_tokens_per_segment": 80,  # 每段最多80个tokens
    "interval_silence": 300,             # ⭐新参数：分段间停顿300ms
    "save_file": True
})
```

#### 示例8：高级参数调优

```python
import requests

response = requests.post("http://localhost:8001/api/v1/tts/synthesize", json={
    "text": "这是参数调优的测试，追求更自然的语音效果。",
    "speaker_audio": "E:/audio/voice.wav",
    
    # 生成参数（完整控制）
    "temperature": 0.75,         # 降低随机性，更稳定
    "top_p": 0.9,               # 提高采样阈值
    "top_k": 20,                # 增加候选数
    "repetition_penalty": 1.8,  # 加强重复惩罚
    "max_mel_tokens": 1200,     # 限制最大长度
    "interval_silence": 200,    # 分段停顿时间
    
    # 情感控制
    "emotion_mode": 2,
    "emotion_vector": [0.3, 0, 0, 0, 0, 0, 0, 0.7],  # 平静为主，略带愉悦
    "emotion_alpha": 0.65,
    
    "save_file": True
})

result = response.json()
print(f"生成: {result['audio_file']}")
```

## 参数完全手册

### 📋 完整请求参数说明（主要参数一览）

---

## 🔹 基础参数（必需）

### `text`
- **类型**: `string`
- **必需**: ✅ 是
- **默认值**: 无
- **作用**: 要合成语音的文本内容
- **支持**: 中文、英文、中英混合
- **常用程度**: ⭐⭐⭐⭐⭐ 必需参数
- **建议修改**: ✅ 每次调用都需要提供

**示例**：
```json
"text": "你好，欢迎使用IndexTTS2语音合成服务。"
```

---

### `speaker_audio`
- **类型**: `string`
- **必需**: ✅ 是
- **默认值**: 无
- **作用**: 参考音频，用于克隆目标说话人的音色
- **格式**: 
  - 文件路径（如 `E:/audio/voice.wav`）
  - base64编码的音频数据
- **要求**: 
  - 格式：WAV、MP3等常见音频格式
  - 时长：建议3-10秒，清晰的语音片段
  - 质量：越清晰效果越好
- **常用程度**: ⭐⭐⭐⭐⭐ 必需参数
- **建议修改**: ✅ 每次调用根据需要的音色选择

**示例**：
```json
"speaker_audio": "E:/audio/my_voice.wav"
```

---

## 🔹 情感控制参数（可选）

### `emotion_mode`
- **类型**: `int`
- **必需**: ❌ 否
- **默认值**: `0`
- **参数范围**: 0-3
- **作用**: 选择情感控制的方式
  - `0`: 使用说话人音频中的原始情感（最简单）
  - `1`: 使用另一段音频的情感（音色+情感分离）
  - `2`: 使用8维情感向量（精确控制）
  - `3`: 使用文本描述情感（实验性）
- **常用程度**: ⭐⭐⭐ 中等
- **建议修改**: 
  - ✅ 需要情感控制时修改为1或2
  - ❌ 一般场景使用默认值0即可

---

### `emotion_audio`
- **类型**: `string`
- **必需**: ❌ 否（仅在`emotion_mode=1`时需要）
- **默认值**: `null`
- **作用**: 情感参考音频，提取情感但不克隆音色
- **使用场景**: 想用A的音色 + B的情感
- **格式**: 文件路径或base64编码
- **常用程度**: ⭐ 较少使用
- **建议修改**: ❌ 仅在特殊需求时使用

**示例**：
```json
{
  "emotion_mode": 1,
  "speaker_audio": "voice_a.wav",  // A的音色
  "emotion_audio": "happy_b.wav"    // B的快乐情感
}
```

---

### `emotion_vector`
- **类型**: `array[float]`
- **必需**: ❌ 否（仅在`emotion_mode=2`时需要）
- **默认值**: `null`
- **参数范围**: 8个浮点数，每个0.0-1.0
- **作用**: 精确控制8种情感的混合比例
- **维度顺序**: `[快乐, 愤怒, 悲伤, 恐惧, 厌恶, 忧郁, 惊讶, 平静]`
- **常用程度**: ⭐⭐ 中等偏低
- **建议修改**: ✅ 需要精确情感控制时使用

**使用建议**：
- 单一情感：如 `[1.0, 0, 0, 0, 0, 0, 0, 0]` = 完全快乐
- 混合情感：如 `[0.6, 0, 0, 0, 0, 0, 0.4, 0]` = 60%快乐 + 40%平静
- 中性：如 `[0, 0, 0, 0, 0, 0, 0, 1.0]` = 完全平静

**示例**：
```json
{
  "emotion_mode": 2,
  "emotion_vector": [0.7, 0, 0, 0, 0, 0, 0.3, 0]  // 开心中带着平静
}
```

---

### `emotion_text`
- **类型**: `string`
- **必需**: ❌ 否（仅在`emotion_mode=3`时需要）
- **默认值**: `null`
- **作用**: 用文本描述期望的情感
- **格式**: 如 "高兴的"、"悲伤的"、"平静的"
- **状态**: ⚠️ 实验性功能，效果不稳定
- **常用程度**: ⭐ 很少使用
- **建议修改**: ❌ 不推荐使用，建议用`emotion_vector`

---

### `emotion_alpha`
- **类型**: `float`
- **必需**: ❌ 否
- **默认值**: `0.65` (API优化值，IndexTTS2原始默认为`1.0`)
- **参数范围**: 0.0-1.0
- **作用**: 控制情感对生成语音的影响程度
  - `0.0`: 完全不使用情感（纯音色克隆）
  - `0.3-0.5`: 轻微情感影响
  - `0.65`: **API默认**，平衡的情感表达
  - `0.8-0.9`: 较强情感影响
  - `1.0`: IndexTTS2原始默认，完全情感
- **常用程度**: ⭐⭐ 中等偏低
- **建议修改**: ❌ API默认值0.65通常效果最好

**使用建议**：
- 新闻播报/客服：`0.3-0.5`（平淡、客观）
- 日常对话：`0.65`（API默认，推荐）
- 情感表达/有声书：`0.8-0.9`（较强烈）
- 完全情感：`1.0`（IndexTTS2原始默认）

**⚠️ 关于默认值的说明**：
- **IndexTTS2原始默认值**: `1.0`（完全情感）
- **API设置为0.65的原因**:
  - 更温和的情感表达，避免过度夸张
  - 适合更广泛的应用场景
  - 用户可根据需要调整为1.0获得完全情感

**技术细节**：
- 仅在使用情感控制（emotion_mode=1,2,3）时生效
- emotion_mode=0时，此参数无效

---

### `use_random`
- **类型**: `bool`
- **必需**: ❌ 否
- **默认值**: `false`
- **作用**: 是否使用随机情感采样，增加每次生成的变化
- **效果**: 启用后同样文本每次生成会略有不同
- **常用程度**: ⭐ 较少使用
- **建议修改**: ❌ 一般保持关闭，保证结果一致性

**使用场景**：
- 需要多样性的场景（如游戏NPC对话）
- 不需要完全一致的重复生成

---

## 🔹 高级生成参数（可选，影响质量和速度）

### `max_text_tokens_per_segment`
- **类型**: `int`
- **必需**: ❌ 否
- **默认值**: `120`
- **参数范围**: 20-600
- **作用**: 控制长文本如何分段处理
  - 较小值：分段更细，每段更短
  - 较大值：分段更粗，每段更长
- **影响**: 
  - 分段过细：可能不够流畅
  - 分段过粗：单段可能过长影响质量
- **常用程度**: ⭐⭐ 中等偏低
- **建议修改**: ❌ 默认值120适合大多数场景

**使用建议**：
- 短句/对话：`80-100`
- 一般文本：`120`（默认）
- 长段落：`150-200`

---

### `max_mel_tokens`
- **类型**: `int`
- **必需**: ❌ 否
- **默认值**: `1500`
- **参数范围**: 50-2000
- **作用**: 限制单段生成音频的最大长度
- **影响**: 
  - 过小：长文本会被强制截断
  - 过大：可能导致质量下降
- **常用程度**: ⭐ 较少使用
- **建议修改**: ❌ 默认值足够使用

**使用建议**：
- 一般保持默认1500
- 仅在生成非常长的单段时考虑调整

---

### `interval_silence` ⭐ 新增
- **类型**: `int`
- **必需**: ❌ 否
- **默认值**: `200`
- **参数范围**: 0-2000（毫秒）
- **作用**: 控制长文本分段间的停顿时间
- **影响**: 
  - `0`: 无停顿，连续播放
  - `100-200`: 自然停顿（推荐）
  - `300-500`: 明显停顿
  - `500+`: 较长停顿
- **常用程度**: ⭐⭐⭐ 中等
- **建议修改**: ✅ 可根据需要调整停顿感

**使用建议**：
- 快速朗读：`100-150ms`
- 正常朗读：`200ms`（默认）
- 新闻播报：`250-300ms`
- 有声书章节间：`400-600ms`

---

### `do_sample`
- **类型**: `bool`
- **必需**: ❌ 否
- **默认值**: `true`
- **作用**: 是否使用采样生成
  - `true`: 使用采样，语音更自然多变
  - `false`: 不使用采样，语音更稳定但可能机械
- **常用程度**: ⭐ 较少使用
- **建议修改**: ❌ 保持默认`true`

**使用建议**：
- 一般场景：保持 `true`（推荐）
- 需要完全一致的输出：设为 `false`

---

### `temperature`
- **类型**: `float`
- **必需**: ❌ 否
- **默认值**: `0.8`
- **参数范围**: 0.1-2.0
- **作用**: 控制生成的随机性/多样性
  - `0.5-0.7`: 非常稳定，较机械
  - `0.7-0.8`: 稳定自然（推荐）
  - `0.8-0.9`: 平衡，自然度高
  - `0.9-1.2`: 多样性高，略不稳定
  - `1.2+`: 非常随机，可能不连贯
- **常用程度**: ⭐⭐⭐⭐ 高
- **建议修改**: ✅ 可以调整以获得不同风格

**使用建议**：
- 新闻播报：`0.7`（稳定）
- 日常对话：`0.8`（默认，推荐）
- 有声书：`0.85-0.9`（自然）
- 创意内容：`1.0`（多样）

**示例**：
```json
{
  "temperature": 0.75,  // 追求稳定
  "text": "..."
}
```

---

### `top_p`
- **类型**: `float`
- **必需**: ❌ 否
- **默认值**: `0.8`
- **参数范围**: 0.0-1.0
- **作用**: 核采样参数，控制采样的多样性
  - `0.5-0.7`: 保守，稳定
  - `0.8`: 平衡（推荐）
  - `0.85-0.9`: 较多样
  - `0.95+`: 非常多样
- **常用程度**: ⭐⭐⭐ 中等
- **建议修改**: ✅ 可与temperature配合调整

**使用建议**：
- 追求稳定：`0.7-0.8`
- 追求自然：`0.8-0.85`（推荐）
- 追求多样：`0.9+`

---

### `top_k`
- **类型**: `int`
- **必需**: ❌ 否
- **默认值**: `30`
- **参数范围**: 0-100（0表示不限制）
- **作用**: Top-K采样，限制每步生成时从概率分布中采样的候选token数量
  - **影响内容多样性**，不影响语速
  - 较小值：只考虑最可能的少数token，生成更稳定、保守
  - 较大值：考虑更多候选token，生成更多样、有创意
  - `0`: 不限制，考虑所有可能的token
- **与其他参数的关系**: 通常与`top_p`配合使用，两者都是控制采样多样性的方法
- **常用程度**: ⭐⭐ 中等偏低
- **建议修改**: ❌ 默认值30已经很好，不建议修改

**使用建议**：
- 追求稳定一致：`15-20`（保守，适合需要可预测输出的场景）
- 一般使用：`30`（默认，推荐）
- 追求多样创意：`50-70`（激进，适合创意内容）
- 不限制：`0`（完全依赖temperature和top_p）

**⚠️ 注意**：
- top_k **不影响语速**，只影响生成内容的多样性和随机性
- 建议优先调整`temperature`和`top_p`，而不是`top_k`
- 过小的值可能导致重复或机械化的输出

---

### `num_beams`
- **类型**: `int`
- **必需**: ❌ 否
- **默认值**: `3`
- **参数范围**: 1-10
- **作用**: Beam搜索的宽度
  - 越大：质量可能越高，但速度越慢
  - 越小：速度快，但质量可能下降
- **影响**: 
  - `1`: 贪婪搜索，最快但质量可能不稳定
  - `3`: 平衡（推荐）
  - `5-7`: 高质量，速度较慢
- **常用程度**: ⭐⭐ 中等偏低
- **建议修改**: ❌ 默认值3已经很好

**使用建议**：
- 快速生成：`1-2`
- 一般使用：`3`（默认，推荐）
- 追求质量：`5`（速度会明显变慢）

---

### `repetition_penalty`
- **类型**: `float`
- **必需**: ❌ 否
- **默认值**: `10.0` ⚠️ **注意：此默认值较高**
- **参数范围**: 0.1-20.0
- **推荐范围**: `1.5-2.0`（大多数场景）
- **作用**: 惩罚已经出现过的token，防止生成重复内容
  - `1.0`: 无惩罚
  - `1.2-1.5`: 轻微惩罚
  - `1.5-2.0`: **推荐范围**，温和惩罚，适合大多数场景
  - `2.0-3.0`: 较强惩罚
  - `10.0`: **IndexTTS2默认**，强惩罚，防止严重重复
  - `>5.0`: 过强惩罚，可能导致不自然的输出
- **常用程度**: ⭐⭐⭐ 中等
- **建议修改**: ✅ **强烈建议**调整到1.5-2.0

**⚠️ 重要说明**：
- **为什么默认值是10.0**：
  - IndexTTS2设置此高值是为了防止严重的重复问题
  - 这是一个保守的默认值，确保不会出现重复
  
- **为什么推荐1.5-2.0**：
  - 默认值10.0过高，可能影响流畅度
  - 大多数LLM和TTS应用使用1.2-2.0
  - 1.5-2.0是经过验证的最佳平衡点

**使用建议**：
- **首次使用**: 尝试`1.8`（推荐起点）
- **无重复问题**: `1.2-1.5`
- **有轻微重复**: `1.5-2.0`
- **有严重重复**: `2.0-3.0`
- **极端重复**: 保持`10.0`或更高

**典型场景配置**：
```json
{
  "repetition_penalty": 1.8,  // 推荐：温和的重复惩罚
  "temperature": 0.8
}
```

**性能影响**：
- 值越高，生成越保守
- 过高的值会牺牲流畅度和自然度
- 建议从1.8开始，根据效果调整

**技术参考**：
- Hugging Face社区推荐：1.5
- 大多数应用默认：1.2-2.0
- IndexTTS2保守默认：10.0

---

### `length_penalty`
- **类型**: `float`
- **必需**: ❌ 否
- **默认值**: `0.0`
- **参数范围**: -2.0-2.0
- **作用**: 影响生成音频的长度倾向
  - 负值（`-1.0 ~ -0.5`）: 鼓励生成更短的语音
  - `0.0`: 中性，不影响长度
  - 正值（`0.5 ~ 1.0`）: 鼓励生成更长的语音
- **常用程度**: ⭐ 很少使用
- **建议修改**: ❌ 一般保持0.0（中性）

**使用建议**：
- 大多数场景：保持 `0.0`（默认）
- 语音过长：尝试 `-0.5`
- 语音过短：尝试 `0.5`

---

## 🔹 输出选项

### `save_file`
- **类型**: `bool`
- **必需**: ❌ 否
- **默认值**: `true`
- **作用**: 是否将生成的音频保存到服务器
- **保存路径**: `outputs/` 目录
- **文件名**: 自动生成，格式 `tts_YYYYMMDD_HHMMSS.wav`
- **常用程度**: ⭐⭐⭐⭐ 高
- **建议修改**: ✅ 根据需要选择

**使用建议**：
- 需要下载音频文件：设为 `true`
- 只需临时使用：设为 `false`

---

### `return_base64`
- **类型**: `bool`
- **必需**: ❌ 否
- **默认值**: `false`
- **作用**: 是否在响应中返回base64编码的音频数据
- **用途**: 适合前端直接播放，无需下载文件
- **注意**: 会增加响应大小
- **常用程度**: ⭐⭐⭐ 中等
- **建议修改**: ✅ 前端应用可以启用

**使用建议**：
- Web前端播放：设为 `true`
- 服务器端处理：设为 `false`（默认）

---

## 📊 参数使用频率总结

| 常用程度 | 参数 | 建议 |
|---------|------|------|
| ⭐⭐⭐⭐⭐ 必需 | `text`, `speaker_audio` | 必须提供 |
| ⭐⭐⭐⭐ 常用 | `temperature`, `save_file` | 可调整优化 |
| ⭐⭐⭐ 中等 | `emotion_mode`, `interval_silence`, `top_p`, `repetition_penalty` | 特定需求时调整 |
| ⭐⭐ 偶尔 | `emotion_vector`, `emotion_alpha`, `top_k`, `num_beams`, `max_text_tokens_per_segment` | 进阶用户使用 |
| ⭐ 很少 | `emotion_audio`, `emotion_text`, `use_random`, `max_mel_tokens`, `do_sample`, `length_penalty` | 特殊场景或调试 |

---

## 💡 快速配置建议

### 场景1：日常使用（推荐，使用优化默认值）
```json
{
  "text": "你的文本",
  "speaker_audio": "voice.wav",
  "repetition_penalty": 1.8  // 推荐：比默认10.0更流畅
}
```

### 场景2：追求稳定性
```json
{
  "text": "你的文本",
  "speaker_audio": "voice.wav",
  "temperature": 0.7,
  "top_p": 0.75,
  "repetition_penalty": 1.8
}
```

### 场景3：追求自然度
```json
{
  "text": "你的文本",
  "speaker_audio": "voice.wav",
  "temperature": 0.85,
  "top_p": 0.85,
  "repetition_penalty": 1.5,
  "interval_silence": 250
}
```

### 场景4：有声书/播客
```json
{
  "text": "你的文本",
  "speaker_audio": "voice.wav",
  "temperature": 0.85,
  "repetition_penalty": 1.6,
  "interval_silence": 300,
  "max_text_tokens_per_segment": 150
}
```

### 场景5：精确情感控制
```json
{
  "text": "你的文本",
  "speaker_audio": "voice.wav",
  "emotion_mode": 2,
  "emotion_vector": [0.7, 0, 0, 0, 0, 0, 0.3, 0],
  "emotion_alpha": 0.75,
  "repetition_penalty": 1.8
}
```

---

## ⚠️ 重要提示

1. **⭐ repetition_penalty建议调整**：
   - 默认值10.0较高，可能影响流畅度
   - **强烈建议**首次使用时设置为 `1.8`
   - 如果语音流畅无重复，可降低到 `1.5`
   - 只有出现重复问题时才考虑提高

2. **emotion_alpha已优化**：
   - API默认0.65已优化（IndexTTS2原始为1.0）
   - 如需完全情感可设置为1.0
   - 一般场景无需调整

3. **不建议同时大幅调整多个参数**：
   - 建议一次只调整1-2个参数，观察效果
   - 从`repetition_penalty`和`temperature`开始

4. **默认值说明**：
   - 大多数参数默认值适合80%场景
   - 但`repetition_penalty`是例外（默认10.0过高）

5. **参数相互影响**：
   - `temperature`和`top_p`会相互影响，建议配合调整
   - `top_k`影响较小，优先调整其他参数

6. **质量 vs 速度权衡**：
   - `num_beams`越大质量越好但速度越慢
   - 默认值3已经是很好的平衡点

---

**💡 所有参数在 `/api/v1/tts/synthesize` 和 `/api/v1/tts/synthesize-file` 两个端点中都完全支持**

**💡 MCP服务的 `synthesize_speech` 工具也支持所有这些情感控制和生成相关参数（不包括 `save_file`、`return_base64` 等仅与HTTP输出格式相关的选项）**

### 🔄 API 响应格式

**成功响应**：
```json
{
  "success": true,
  "message": "✅ 语音合成成功",
  "audio_file": "outputs/tts_20241113_120530.wav",
  "audio_base64": "UklGRi4gAABXQVZF...",
  "duration_seconds": 3.5,
  "sample_rate": 22050
}
```

**错误响应**：
```json
{
  "detail": "参考音频文件不存在: xxx.wav"
}
```

---

## MCP 服务详细指南

### 🤖 MCP 简介

**MCP (Model Context Protocol)** 是由Anthropic开发的开放协议，用于AI助手与外部工具的集成。

**特点**：
- 🔧 **结构化工具调用**：定义明确的工具和参数
- 💬 **自然交互**：AI助手可以理解和调用工具
- 🔄 **双向通信**：通过stdio进行JSON-RPC通信
- 🎯 **适合AI代理**：Claude Desktop、n8n、Dify等

### 🛠️ MCP 工具列表

IndexTTS2 MCP服务提供**7个强大工具**：

| 工具名 | 说明 | 适用场景 |
|--------|------|----------|
| `clone_voice` | 快速音色克隆 | 日常对话、快速生成 |
| `emotional_speech` | 单一情感语音 | 明确情感表达 |
| `mixed_emotion_speech` | 混合情感语音 | 复杂细腻情感 |
| `batch_synthesize` | 批量合成 | 有声书、播客 |
| `synthesize_speech` | 完整参数合成 | 高级自定义需求 |
| `list_saved_prompts` | 列出参考音频 | 管理音频资源 |
| `list_generated_audio` | 列出生成音频 | 查看历史记录 |

### 📝 在 Claude Desktop 中使用

**步骤1：配置**

编辑配置文件（Windows: `%APPDATA%\Claude\claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "indextts2": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "E:/index-tts/service",
        "E:/index-tts/service/mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "E:/index-tts"
      }
    }
  }
}
```

**步骤2：重启 Claude Desktop**

**步骤3：使用示例**

```
用户：帮我用我的声音说"今天天气真好"
Claude：✅ 已生成：mcp_outputs/clone_abc123.wav

用户：用高兴的情感说话
Claude：✅ 已生成：mcp_outputs/happy_xyz789.wav
```

### 🔧 工具参数详解

#### clone_voice
- `text`: 要合成的文本
- `speaker_audio`: 参考音频绝对路径

#### emotional_speech
- `text`: 要合成的文本
- `speaker_audio`: 参考音频路径
- `emotion`: happy|angry|sad|afraid|disgusted|melancholic|surprised|calm
- `emotion_intensity`: 0.0-1.0，默认0.8

#### mixed_emotion_speech
- `text`: 要合成的文本
- `speaker_audio`: 参考音频路径
- `emotion_mix`: 情感混合，如 `{"happy": 0.6, "surprised": 0.4}`

#### batch_synthesize
- `text_list`: 文本数组
- `speaker_audio`: 参考音频路径

#### synthesize_speech（完整版）

⭐ **支持与 REST API 相同的情感控制与生成参数**，与 IndexTTS2 核心推理行为完全对等（不包含 REST 专用的 `return_base64`、`save_file` 等输出选项）。

**基础参数**：
- `text`: 要合成的文本
- `speaker_audio_path`: 参考音频路径

**情感控制参数**（同REST API）：
- `emotion_mode`: speaker|audio|vector|text
- `emotion_audio_path`: 情感参考音频
- `emotion_vector`: 8维情感向量数组
- `emotion_text`: 情感描述
- `emotion_alpha`: 情感强度
- `use_random`: 随机采样

**高级生成参数**（同REST API）：
- `max_text_tokens_per_segment`: 每段token数（默认120）
- `max_mel_tokens`: 最大mel tokens（默认1500）
- `interval_silence`: ⭐分段停顿时长ms（默认200）
- `do_sample`: 是否采样（默认true）
- `temperature`: 温度（默认0.8）
- `top_p`: 核采样（默认0.8）
- `top_k`: Top-K（默认30）
- `num_beams`: Beam搜索（默认3）
- `repetition_penalty`: 重复惩罚（默认10.0）
- `length_penalty`: 长度惩罚（默认0.0）

**输出选项**：
- `output_filename`: 输出文件名（可选）

**💡 使用示例**：
```json
{
  "name": "synthesize_speech",
  "arguments": {
    "text": "这是完整参数控制",
    "speaker_audio_path": "E:/audio/voice.wav",
    "emotion_mode": "vector",
    "emotion_vector": [0.8, 0, 0, 0, 0, 0, 0.2, 0],
    "temperature": 0.7,
    "top_k": 15,
    "repetition_penalty": 1.8,
    "interval_silence": 300
  }
}
```

---

## 不同工具调用示例

### 🌐 使用 curl (命令行)

#### 基础合成
```bash
curl -X POST http://localhost:8001/api/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是测试",
    "speaker_audio": "E:/audio/voice.wav",
    "save_file": true
  }'
```

#### 文件上传
```bash
curl -X POST http://localhost:8001/api/v1/tts/synthesize-file \
  -F "speaker_audio=@voice.wav" \
  -F "text=这是文件上传测试" \
  -F "emotion_mode=0" \
  --output output.wav
```

#### 列出参考音频
```bash
curl http://localhost:8001/api/v1/prompts/list
```

### 💻 使用 JavaScript/Node.js

```javascript
const axios = require('axios');
const fs = require('fs');

// 基础合成
async function synthesize() {
  const response = await axios.post('http://localhost:8001/api/v1/tts/synthesize', {
    text: '你好，这是JavaScript调用测试',
    speaker_audio: 'E:/audio/voice.wav',
    emotion_mode: 2,
    emotion_vector: [0.8, 0, 0, 0, 0, 0, 0, 0.2],
    save_file: true
  });
  
  console.log('生成成功:', response.data.audio_file);
}

// Base64方式
async function synthesizeBase64() {
  const audioBuffer = fs.readFileSync('voice.wav');
  const audioBase64 = audioBuffer.toString('base64');
  
  const response = await axios.post('http://localhost:8001/api/v1/tts/synthesize', {
    text: '这是Base64测试',
    speaker_audio: audioBase64,
    return_base64: true,
    save_file: false
  });
  
  // 保存返回的音频
  if (response.data.audio_base64) {
    const buffer = Buffer.from(response.data.audio_base64, 'base64');
    fs.writeFileSync('output.wav', buffer);
    console.log('音频已保存');
  }
}

synthesize();
```

### 🐍 使用 Python

```python
import requests
import base64

class IndexTTS2Client:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
    
    def synthesize(self, text, speaker_audio, **kwargs):
        """基础合成"""
        data = {
            "text": text,
            "speaker_audio": speaker_audio,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/api/v1/tts/synthesize", json=data)
        return response.json()
    
    def synthesize_with_emotion(self, text, speaker_audio, emotion_vector, emotion_alpha=0.8):
        """情感合成"""
        return self.synthesize(
            text=text,
            speaker_audio=speaker_audio,
            emotion_mode=2,
            emotion_vector=emotion_vector,
            emotion_alpha=emotion_alpha,
            save_file=True
        )
    
    def upload_prompt(self, file_path):
        """上传参考音频并获取在服务器上的路径"""
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/api/v1/upload",
                files={"audio": (file_path, f, "audio/wav")}
            )
        return response.json()

# 使用示例
client = IndexTTS2Client()

# 上传参考音频
result = client.upload_prompt("my_voice.wav")
prompt_path = result["path"]

# 合成语音
result = client.synthesize_with_emotion(
    text="这是Python客户端测试",
    speaker_audio=prompt_path,
    emotion_vector=[0.7, 0, 0, 0, 0, 0, 0, 0.3]
)
print(f"生成: {result['audio_file']}")
```

### 🔧 在 n8n 中使用

**步骤1**：添加HTTP Request节点

**步骤2**：配置节点
- Method: POST
- URL: `http://localhost:8001/api/v1/tts/synthesize`
- Body Content Type: JSON
- Body:
```json
{
  "text": "{{ $json.text }}",
  "speaker_audio": "{{ $json.speaker_audio }}",
  "emotion_mode": 2,
  "emotion_vector": [1, 0, 0, 0, 0, 0, 0, 0],
  "save_file": true
}
```

**步骤3**：处理响应
- 使用Code节点提取 `audio_file` 字段
- 可选：使用Move Binary Data节点下载音频文件

### 🎨 在 Postman 中测试

**1. 创建新请求**
- Method: POST
- URL: `http://localhost:8001/api/v1/tts/synthesize`

**2. 设置Headers**
```
Content-Type: application/json
```

**3. 设置Body (raw JSON)**
```json
{
  "text": "这是Postman测试",
  "speaker_audio": "E:/audio/voice.wav",
  "emotion_mode": 2,
  "emotion_vector": [0.8, 0, 0, 0, 0, 0, 0.2, 0],
  "temperature": 0.85,
  "save_file": true
}
```

**4. 发送请求并查看响应**

---

## 常见问题

### ❓ 为什么我的参考音频无法识别？

**原因**：路径问题
- ✅ 使用绝对路径：`E:/audio/voice.wav`
- ❌ 避免相对路径：`../audio/voice.wav`（除非确认服务器能访问）

**解决方案**：
1. 使用绝对路径
2. 或上传到服务器：使用`/api/v1/upload`获取在服务器上的有效路径
3. 或使用base64编码

### ❓ 生成的语音质量不好怎么办？

**参考音频质量**：
- 使用清晰、无噪音的音频
- 时长：3-10秒最佳
- 格式：WAV格式，16kHz或24kHz采样率

**参数调优**：
```python
{
  "temperature": 0.7,          # 降低随机性
  "repetition_penalty": 1.8,   # 加强重复惩罚
  "top_p": 0.9                 # 提高采样质量
}
```

### ❓ 如何控制语速？

**说明**：当前版本不支持直接的语速控制。

**间接方法**：
- 在文本中添加标点符号和停顿
- 调整`temperature`参数可能会略微影响节奏

### ❓ MCP服务无法连接Claude Desktop

**检查清单**：
1. ✅ 配置文件路径正确
2. ✅ uv 命令可用（推荐使用 `uv run setup_mcp.py` 自动写入配置）
3. ✅ `PYTHONPATH`环境变量设置正确
4. ✅ 已重启Claude Desktop
5. ✅ 检查`mcp_server.log`日志文件

**Windows路径注意**：
- 推荐使用正斜杠：`E:/index-tts/service/mcp_server.py`
- 或使用双反斜杠：`E:\\index-tts\\service\\mcp_server.py`

### ❓ API返回400错误

**常见原因**：
1. 缺少必需参数（`text`或`speaker_audio`）
2. `emotion_vector`格式错误（必须是8个数字的数组）
3. `emotion_mode`值不在0-3范围内
4. 参考音频文件不存在

**调试方法**：
```python
response = requests.post(...)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
```

### ❓ 如何批量处理大量文本？

**方案1：使用batch_synthesize (MCP)**
```python
# 通过MCP客户端
{
  "name": "batch_synthesize",
  "arguments": {
    "text_list": ["文本1", "文本2", "文本3"],
    "speaker_audio": "path/to/voice.wav"
  }
}
```

**方案2：使用Python循环 (REST API)**
```python
texts = ["文本1", "文本2", "文本3"]
for i, text in enumerate(texts):
    result = client.synthesize(
        text=text,
        speaker_audio="voice.wav",
        save_file=True
    )
    print(f"完成 {i+1}/{len(texts)}: {result['audio_file']}")
```

---

## 性能优化

### ⚡ 服务器端优化

**默认加速选项**（v2.0+）：
```bash
# 基础启动（CUDA环境自动启用FP16和CUDA Kernel加速）
uv run server.py

# 如需禁用加速（调试时）
uv run server.py --no-fp16 --no-cuda-kernel
```

**性能对比**：
- 纯CPU：~5-10秒/句
- GPU (FP32, --no-fp16)：~2-3秒/句
- GPU (FP16 + CUDA Kernel, 默认)：~1-2秒/句 ⭐推荐

### 💾 模型缓存

**HuggingFace缓存**：
```bash
# 设置全局缓存目录
export HF_HUB_CACHE=F:/HF_CACHE  # Linux/Mac
set HF_HUB_CACHE=F:\HF_CACHE     # Windows CMD
$env:HF_HUB_CACHE="F:\HF_CACHE"  # Windows PowerShell
```

### 🔄 并发处理

**使用异步请求** (Python):
```python
import asyncio
import aiohttp

async def synthesize_async(session, text, speaker_audio):
    async with session.post(
        'http://localhost:8001/api/v1/tts/synthesize',
        json={"text": text, "speaker_audio": speaker_audio, "save_file": True}
    ) as response:
        return await response.json()

async def batch_process(texts, speaker_audio):
    async with aiohttp.ClientSession() as session:
        tasks = [synthesize_async(session, text, speaker_audio) for text in texts]
        results = await asyncio.gather(*tasks)
        return results

# 运行
texts = ["文本1", "文本2", "文本3"]
results = asyncio.run(batch_process(texts, "voice.wav"))
```

### 📊 参数建议

**快速生成（牺牲质量）**：
```json
{
  "temperature": 0.9,
  "top_k": 10,
  "repetition_penalty": 1.3
}
```

**高质量生成（较慢）**：
```json
{
  "temperature": 0.7,
  "top_k": 25,
  "top_p": 0.95,
  "repetition_penalty": 1.8
}
```

**平衡模式（推荐）**：
```json
{
  "temperature": 0.8,
  "top_k": 15,
  "top_p": 0.85,
  "repetition_penalty": 1.5
}
```

---

## 审查与质量保证

为了确保本服务可直接用于生产，我们已经对以下文件进行了多轮严格的代码比对和审查：

- `indextts/infer_v2.py`（核心推理逻辑）
- `service/server.py`（REST API）
- `service/mcp_server.py`（MCP 服务）
- `webui.py`（官方 WebUI 调用逻辑）

**1. 功能与参数完整性**

- REST API 的 `/api/v1/tts/synthesize` 与 `/api/v1/tts/synthesize-file`，以及 MCP 的 `synthesize_speech` 工具，都已完整暴露 IndexTTS2 的主要功能和关键参数（文本、4种情感控制模式、所有生成参数、分段控制等）。
- 默认值与底层 `infer_v2.py` 和 WebUI 保持一致，只有以下两点是**有意优化**：
  - `emotion_alpha`：IndexTTS2 原始默认为 `1.0`，API/MCP 默认为 `0.65`（更温和的情感强度，也已在文档中说明）。
  - `repetition_penalty`：默认 `10.0`（较高，原始配置），文档和 MCP Schema 中都**强烈推荐**在大多数场景使用 `1.5-2.0`（如 `1.8`）。

**2. 与 WebUI 行为的一致性**

- 情感向量模式（`emotion_mode=2`）在 REST 和 MCP 中都会调用 `normalize_emo_vec(..., apply_bias=True)`：
  - 应用情感偏置（削弱易产生怪异效果的情感）
  - 将情感向量总和限制在 `≤ 0.8`
- 以下边界行为已与 WebUI 完全对齐：
  - `emotion_mode=0`：强制清除外部情感参考音频，仅使用说话人原始情感。
  - 空字符串 `emotion_text` 自动转为 `None`（由底层逻辑决定是否使用主文本推情感）。
  - `top_k=0` 时统一转换为 `None`，表示“不要额外限制 Top-K”。
- 这意味着：**同一组参数配置，在 WebUI、REST API、MCP 下的合成效果是一致的。**

**3. 生产环境健壮性**

- REST API：
  - 使用 `lifespan` 管理模型生命周期，关闭时清理 `temp/` 下临时文件。
  - 基于 Pydantic 的输入验证（长度/范围/类型），结合自定义校验器（如 8 维情感向量）。
  - 统一的错误处理、设备与加速配置日志（FP16/CUDA kernel）、CORS 支持等。
- MCP 服务：
  - 使用 `pathlib` 和统一的 `_validate_text` / `_validate_audio_path` 等校验方法。
  - 对批量调用、情感参数、路径等做了严格检查，并在关闭/信号时清理临时文件。

综合多份审查（API/MCP 100% 报告、设计审查、参数审查、WebUI 一致性报告）的结论：

- 🟢 功能覆盖：REST + MCP 已完整覆盖 IndexTTS2 的主要能力和关键参数。
- 🟢 行为一致：与 WebUI 的情感处理和采样逻辑完全一致。
- 🟢 生产就绪：错误处理、日志、资源管理、验证逻辑完善，可直接用于生产部署。

**4. 未暴露的高级特性（未来可选增强）**

- IndexTTS2 底层还存在少量高级参数目前**没有作为 REST/MCP 参数暴露**：
  - `stream_return`: 流式返回，支持边生成边推送音频（适合实时/低延迟场景）。
  - `more_segment_before`: 更细粒度的分段控制，高级用法。
- 这些特性目前不会影响普通使用场景，如有需要，可以在未来版本中按需增加对应的 API/MCP 接口。

**5. 文档来源说明**

- 本 README 已整合各类审查/修复报告中的**有效结论**和**最终设计决策**，是当前唯一需要关注的服务说明文档。
- 目录中的其他 `*_REPORT.md`、`*_SUMMARY.md` 等历史审查文档，只是中间过程的记录，可以视为内部备忘，不再建议单独阅读。

---

## 📚 更多资源

- **项目主页**: https://github.com/index-tts/index-tts
- **Swagger文档**: http://localhost:8001/docs (启动服务后访问)
- **测试示例**: `examples/test_api.py`、`examples/test_api_file_advanced.py`、`test_mcp.py`
- **环境检查**: `check_environment.py`

---

## 📝 总结

本文档涵盖了IndexTTS2服务的完整使用方法：

✅ **REST API**: 适合Web应用、移动应用、第三方集成  
✅ **MCP服务**: 适合Claude Desktop、n8n、Dify等AI工具  
✅ **丰富示例**: Python、JavaScript、curl、Postman等  
✅ **详细参数**: 完整的参数说明和调优建议  
✅ **故障排查**: 常见问题和解决方案  
✅ **性能优化**: 加速技巧和并发处理  

祝您使用愉快！🎉
