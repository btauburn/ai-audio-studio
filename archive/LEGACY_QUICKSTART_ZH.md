# IndexTTS 配音平台开箱即用说明

适用场景：Windows 重装、项目目录迁移或新机器首次部署。所有命令都在仓库根目录执行。

## 1. 当前目录结构

```text
index-tts/
├── .venv/                                # 平台 + IndexTTS2，Python 3.12.13
├── environments/
│   ├── qwen3tts/
│   │   ├── pyproject.toml / uv.lock
│   │   └── .venv/                        # Qwen VD/CV，Python 3.12.13
│   └── cosyvoice/
│       ├── pyproject.toml / uv.lock
│       └── .venv/                        # CosyVoice3，Python 3.10.19
├── models/
│   ├── manifests/                        # revision、大小、SHA256
│   ├── recovery_blobs/                    # 可选的已验证本地恢复缓存（不进入 Git）
│   ├── index_tts/                        # IndexTTS2 主模型
│   ├── qwen3_tts/
│   │   ├── Qwen3-TTS-VoiceDesign/
│   │   └── Qwen3-TTS-CustomVoice/
│   └── cosyvoice3/
│       ├── Fun-CosyVoice3-0.5B/
│       └── wetext/
├── dubbing_service/                      # 9000 平台业务/API/WebUI
├── service/                              # 8001～8004 TTS 工作进程
├── dubbing_projects/                     # 默认用户项目数据（私密）
└── scripts/                              # 重建、预检、恢复和备份
```

以下旧路径不再是生产事实：

- `checkpoints/`：不是当前模型目录；
- `.venv_qwen3tts`：旧环境已删除，注册表不会回退使用；
- `models/hf_cache`：损坏/重复缓存已删除，不是恢复来源；
- CosyVoice 复用 `.venv`：已废弃，必须使用独立 Python 3.10 环境。

## 2. 系统准备

需要：

- Windows 10/11；
- NVIDIA GPU 与兼容驱动；
- `uv` 在 PATH；
- `ffmpeg` 和 `ffprobe` 在 PATH；
- 足够的模型、项目和备份空间。

检查：

```powershell
uv --version
ffmpeg -version
ffprobe -version
nvidia-smi
```

不要求全局 `python`。不要用全局 pip 修环境。

## 3. 重建三个 uv 环境

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Runtime All
```

该脚本会：

1. 通过 uv 安装固定 Python 3.12.13 与 3.10.19；
2. 用根 `uv.lock` 重建平台/IndexTTS2；
3. 用 Qwen 自己的 lockfile 重建 Qwen 环境；
4. 用 CosyVoice 自己的 lockfile 重建 Cosy 环境；
5. 对三个环境分别执行 `uv pip check`。

不要把三份 `pyproject.toml` 合并，也不要手工复制旧 `.venv`。

## 4. 恢复本机配置

第一次部署可复制无密钥模板：

```powershell
Copy-Item .\dubbing_service\config.example.json .\dubbing_service\config.json
```

默认服务地址：

| 服务 | 地址 |
| --- | --- |
| 平台 | `http://127.0.0.1:9000` |
| IndexTTS2 | `http://127.0.0.1:8001` |
| CosyVoice3 | `http://127.0.0.1:8002` |
| Qwen VoiceDesign | `http://127.0.0.1:8003` |
| Qwen CustomVoice | `http://127.0.0.1:8004` |
| DeepSeek | `https://api.deepseek.com` |

DeepSeek 默认使用 `deepseek-v4-flash`，必要时单次回退 `deepseek-v4-pro`。把 API Key 写入环境变量或平台的本机 Provider 配置；不要把真实密钥写入示例、文档或 Git。

## 5. 校验模型

模型已经完整复制时先做快速预检：

```powershell
.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --startup
```

doctor 要求 9000 和 8001～8004 端口空闲，因此应在平台/TTS 全部停止时运行。

发布前做完整 SHA256 校验：

```powershell
.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --full
```

缺文件时不要随意复制缓存。对对应 manifest 先预演，再显式允许 ModelScope：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\index_tts2.json --dry-run --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\index_tts2.json --allow-modelscope-download
```

其他 manifest 位于同一目录。供应脚本不会访问 Hugging Face；所有下载都必须匹配固定 ModelScope revision、大小和 SHA256，坏文件会进入 quarantine。

## 6. 启动

### 仅本机使用（默认）

```bat
start_platform.bat
```

一键启动器默认监听 `127.0.0.1:9000`。

### 受信任局域网 / Mac 访问

```bat
start_platform.bat --lan
```

若只想验证启动入口、三环境和模型是否完整而暂不打开平台，可运行：

```bat
start_platform.bat --check
```

只有 `--lan` 模式监听 `0.0.0.0:9000`。平台没有内置账户认证：必须使用 Windows 防火墙限制可信网段，严禁直接暴露公网。

### 仅本机使用

```powershell
.\.venv\Scripts\python.exe -u -m uvicorn dubbing_service.app.main:app --host 127.0.0.1 --port 9000
```

打开：

```text
http://127.0.0.1:9000/web/
```

页面不会自动加载 TTS。在“平台配置”选择 IndexTTS2、CosyVoice3、Qwen VoiceDesign 或 Qwen CustomVoice，然后明确点击启动/加载模型。平台同一时刻只托管一个 GPU TTS 进程。

## 7. 首次验收

依次完成：

1. `GET /api/v1/ready` 返回 `status=ok`；
2. WebUI 正常打开，浏览器控制台无错误；
3. 平台配置能保存四个独立 TTS 地址；
4. 四后端逐个 start → `/health` → 合成样音 → stop，确认端口释放；
5. DeepSeek Flash 连接测试，必要时显式测试 Pro；
6. 用测试项目跑创建 → 分段 → 角色配置 → 合成 → 编辑 → 导出 → 字幕；
7. 运行项目数据审计；
8. 做一次冷备份并在隔离目录恢复演练。

详细命令见 [docs/operations.md](../docs/operations.md)。

## 8. 备份

停止平台和 TTS 后，先只读验证默认备份清单和不可重新下载的恢复材料：

```powershell
.\scripts\backup_project.ps1 -ValidateOnly
```

通过后再创建备份：

```powershell
.\scripts\backup_project.ps1
```

默认备份排除 API Key、本机 Provider/配置和普通模型权重；保留全部 manifest，并对当前存在且被 manifest 引用的可选 `models/recovery_blobs/` 做 SHA256 校验。普通模型可按固定 ModelScope 来源重新恢复。备份仍可能包含作品文本、prompt 和参考声音，必须私密加密保存。生产项目若通过 `DUBBING_PROJECTS_ROOT` 放在仓库外，需要另做数据盘备份。

只导出无模型/无密钥/无用户数据的源码：

```powershell
.\scripts\export_source.ps1 -ValidateOnly
.\scripts\export_source.ps1 -DestinationRoot E:\source-exports\index-tts
```

## 9. 常见问题

### doctor 报端口已占用

它是启动前预检。先用 `Get-NetTCPConnection` 和进程命令行确认是否已有平台/TTS 正常运行；记录健康结果后停止，再重跑 doctor。

### 模型目录存在但启动失败

运行对应 manifest 的完整 `model_preflight.py`。目录存在不能证明大文件完整，禁止跳过大小/SHA256 校验。

### 后端保存后仍调用旧模型

配置后端不等于运行后端。查看 `/api/v1/settings/tts-service/status` 的 `config_backend` 与 `runtime_backend`，停止旧进程后启动目标后端。

### Qwen 或 CosyVoice import 冲突

先确认进程解释器。Qwen 必须来自 `environments/qwen3tts/.venv`，CosyVoice 必须来自 `environments/cosyvoice/.venv`；不要向根 `.venv` 补装它们的包。

### DeepSeek 不自动回退 Pro

401/403/400 是凭据、权限或参数错误，设计上不回退。修复 Base URL/API Key/模型权限；只有可重试故障或输出校验失败才进行一次 Pro 回退。

## 10. 生产前提醒

技术恢复不代表商业授权已经解决。IndexTTS2 自定义许可、MaskGCT 的非商业条款、参考声音授权以及 CosyVoice/Qwen/third-party notice 都必须审查。平台默认无登录鉴权；局域网模式也必须限制网络边界。
