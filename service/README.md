# TTS 工作进程层

`service/` 是配音平台内部的推理工作进程，不是用户启动整个平台的入口。正常使用请从根目录运行 `start_platform.bat`，再在“平台配置”中选择并启动一个后端。

| 后端 | 脚本 | 解释器 | 端口 |
| --- | --- | --- | ---: |
| IndexTTS2 | `server.py` | 根 `.venv` | 8001 |
| CosyVoice3 | `cosyvoice_server.py` | `environments/cosyvoice/.venv` | 8002 |
| Qwen VoiceDesign | `qwen3tts_server.py` | `environments/qwen3tts/.venv` | 8003 |
| Qwen CustomVoice | `qwen3tts_customvoice_server.py` | `environments/qwen3tts/.venv` | 8004 |

平台维护的权威命令位于 `dubbing_service/app/core/tts_backends.py`。所有工作进程只绑定 loopback；对外业务入口是 9000 平台。平台同一时刻只托管一个 GPU 模型。

## 隔离诊断

从项目根目录先设置严格离线和项目内模型路径：

```powershell
$env:HF_HOME=(Resolve-Path .\models\.runtime_cache)
$env:HF_HUB_CACHE="$env:HF_HOME\hub"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$env:COSYVOICE_WETEXT_DIR=(Resolve-Path .\models\cosyvoice3\wetext)
```

然后只启动一个需要诊断的后端：

```powershell
.\.venv\Scripts\python.exe -u .\service\server.py --host 127.0.0.1 --port 8001 --model_dir .\models\index_tts --fp16 --no-cuda-kernel

.\environments\cosyvoice\.venv\Scripts\python.exe -u .\service\cosyvoice_server.py --host 127.0.0.1 --port 8002 --model_dir .\models\cosyvoice3\Fun-CosyVoice3-0.5B

.\environments\qwen3tts\.venv\Scripts\python.exe -u .\service\qwen3tts_server.py --host 127.0.0.1 --port 8003 --model_dir .\models\qwen3_tts\Qwen3-TTS-VoiceDesign --device cuda:0 --dtype bfloat16

.\environments\qwen3tts\.venv\Scripts\python.exe -u .\service\qwen3tts_customvoice_server.py --host 127.0.0.1 --port 8004 --model_dir .\models\qwen3_tts\Qwen3-TTS-CustomVoice --device cuda:0 --dtype bfloat16
```

每个服务提供 `/health`、模型 load/unload 和 `/api/v1/tts/synthesize`。诊断结束必须正常停止进程，再回到平台使用。

## 兼容和旧入口

- `start_api.bat`：受支持的 IndexTTS2 单后端诊断入口，显式使用根 `.venv`、项目内模型和离线缓存；不要与 9000 平台并行。
- `check_environment.py`：兼容入口，仅转发到权威 `scripts/platform_doctor.py`。
- `start_mcp.bat`、`mcp_server.py`、`setup_mcp.py`：保留的旧 MCP 兼容工具。`start_mcp.bat --check` 只验证入口；实际运行会直接加载 IndexTTS2，绕过平台 GPU 租约，且尚未完成现代 Claude Desktop 客户端验收，只能隔离调试。
- 根目录 `webui.py`：上游 IndexTTS2 Gradio demo，不是平台 WebUI，不能与平台并行。

旧单后端 REST/MCP 的长篇参考说明已归档到 [README_SERVICE_LEGACY.md](../archive/README_SERVICE_LEGACY.md)。当前生产操作以 [operations.md](../docs/operations.md) 为准。
