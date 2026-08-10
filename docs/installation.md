# GitHub 新机安装与模型配置

本文面向从 GitHub 全新克隆仓库的 Windows 用户。所有命令都在仓库根目录执行，不依赖全局 Python，也不要用全局 `pip` 修补模型环境。

> [!IMPORTANT]
> 先选择安装方式。**完整安装**恢复四个 TTS 后端和全部字幕功能，可以直接使用 `start_platform.bat`；**部分开发安装**只恢复正在开发的后端，不能通过启动器的完整 doctor，必须手工启动平台。

## 1. 选择安装方式

| 方式 | 环境与模型 | 启动方式 | 适合场景 |
| --- | --- | --- | --- |
| 完整安装（推荐） | 三个 uv 环境、4 个 TTS 后端、4 套字幕/对齐模型 | `start_platform.bat` | 日常制作、完整验收 |
| 部分开发安装 | 平台环境 + 选定后端环境与模型 | 手工运行 `uvicorn` | 后端开发、节省临时磁盘 |

完整安装的模型最终占用约 31.85 GiB：

| 模型组 | 约占用 |
| --- | ---: |
| IndexTTS2 | 8.26 GiB |
| CosyVoice3 + WeText | 8.18 GiB |
| 两个 Qwen3-TTS 模型 | 8.42 GiB |
| 全部字幕/对齐模型 | 6.99 GiB |

下载缓存、原子复制临时文件、三个虚拟环境和项目输出还会占用额外空间。完整安装建议至少预留 **80 GiB**；长期制作还应为项目音频和备份单独留空间。

## 2. 支持范围与硬件基线

当前验收范围：

- Windows 10/11；
- NVIDIA GPU；
- PowerShell 5.1 或 PowerShell 7；
- Git + Git LFS、uv、FFmpeg/ffprobe；
- 单机使用，或将 Windows 主机作为受信任局域网服务端。

本项目当前实机验证使用 **RTX 4070 Ti SUPER 16 GiB**。三个环境分别锁定 CUDA 12.9、13.0 和 12.4 系 PyTorch wheel；显卡驱动必须能支持其中最新的 wheel。更低显存、较旧驱动、CPU-only、AMD GPU、macOS 推理和 Linux 部署尚未完成本版本验收，不能视为已支持配置。

Mac 可以作为局域网浏览器客户端，但 TTS 工作进程仍运行在 Windows/NVIDIA 主机上。

## 3. 安装系统工具

可使用 Windows Package Manager：

```powershell
winget install --id Git.Git -e
winget install --id GitHub.GitLFS -e
winget install --id astral-sh.uv -e
winget install --id Gyan.FFmpeg -e
```

安装或更新 NVIDIA 驱动后重新打开 PowerShell，检查：

```powershell
git --version
git lfs version
uv --version
ffmpeg -version
ffprobe -version
nvidia-smi
```

任一命令不可用时，先修复 PATH 或驱动，不要继续创建虚拟环境。

## 4. 克隆仓库并创建本机配置

```powershell
git lfs install
git clone https://github.com/tadashijlu/ai-audio-studio.git
Set-Location .\ai-audio-studio
git lfs pull

Copy-Item .\dubbing_service\config.example.json .\dubbing_service\config.json
```

仓库只包含源码、锁文件、模型 manifest 和说明文档，不包含模型、API Key、作品、参考声音或生成结果。`dubbing_service/config.json` 已被 Git 忽略；真实密钥不要写入示例、README、Issue 或提交记录。

默认地址：

| 服务 | 地址 |
| --- | --- |
| 平台 | `http://127.0.0.1:9000` |
| IndexTTS2 | `http://127.0.0.1:8001` |
| CosyVoice3 | `http://127.0.0.1:8002` |
| Qwen VoiceDesign | `http://127.0.0.1:8003` |
| Qwen CustomVoice | `http://127.0.0.1:8004` |
| DeepSeek API | `https://api.deepseek.com` |

## 5. 重建 uv 环境

### 5.1 完整安装

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\bootstrap.ps1 -Runtime All
```

脚本会安装固定的 Python 3.12.13 与 3.10.19，创建三个隔离环境，并分别运行 `uv pip check`：

```text
.venv/                              平台 + IndexTTS2
environments/qwen3tts/.venv/        Qwen VoiceDesign + CustomVoice
environments/cosyvoice/.venv/       CosyVoice3
```

看到 `Environment bootstrap completed` 且命令退出码为 0，才继续下载模型。三个环境的 PyTorch、Transformers 和音频依赖存在冲突，不能合并。

### 5.2 部分开发安装

部分安装始终需要平台环境；再按目标后端补环境：

```powershell
# 所有部分安装都需要
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Runtime Platform

# 开发任一 Qwen 后端时再执行
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Runtime Qwen

# 开发 CosyVoice3 时再执行
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Runtime CosyVoice
```

只开发 IndexTTS2 时不需要 Qwen 或 CosyVoice 环境。

## 6. 从 ModelScope 下载模型

`models/manifests/*.json` 固定了 ModelScope 仓库、revision、目标路径、文件大小和 SHA256。供应器不会在运行时隐式联网；只有显式传入 `--allow-modelscope-download` 才会访问 ModelScope，校验通过后才会把文件原子提升到运行目录。

> [!WARNING]
> `--allow-modelscope-download` 表示操作者明确允许本次联网下载，**不代表取得额外许可权**。下载前应阅读 manifest、ModelScope 模型页、[多许可证边界](../LICENSES/README.md)和[第三方说明](../THIRD_PARTY_NOTICES.md)。IndexTTS2 的辅助集合包含标记为 `CC-BY-NC-4.0` 的 MaskGCT 权重；用于商业作品前必须替换、取得单独许可或获得适当法律结论。

### 6.1 完整安装：八份 manifest

统一脚本会先预演全部来源，再下载四个 TTS 后端和四套字幕/对齐模型。预演不会产生模型文件：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_models.ps1 `
  -Profile Full `
  -DryRun
```

阅读相关条款并确认后执行实际下载：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_models.ps1 `
  -Profile Full `
  -AcceptThirdPartyLicenses
```

`-AcceptThirdPartyLicenses` 只记录使用者已阅读条款，不授予任何额外权利。下载中断后可以重新执行同一命令；已通过大小和 SHA256 校验的目标不会重复恢复。损坏的旧目标会进入 `models/.quarantine/`，不要手工把未知缓存覆盖到运行目录。

### 6.2 部分开发安装：选择 manifest

统一脚本提供三个部分配置：

```powershell
# 四个 TTS 后端（包含 IndexTTS2 与 MaskGCT 非商业边界）
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_models.ps1 `
  -Profile Tts `
  -AcceptThirdPartyLicenses

# Qwen CustomVoice + VoiceDesign + CosyVoice3，不含 IndexTTS2
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_models.ps1 `
  -Profile Core `
  -AcceptThirdPartyLicenses

# 四套字幕/对齐模型
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_models.ps1 `
  -Profile Subtitles `
  -AcceptThirdPartyLicenses
```

只开发单个后端时可直接使用平台环境中的 `model_provision.py`：

```powershell
# Qwen CustomVoice：内置音色，不需要参考声音
.\.venv\Scripts\python.exe .\scripts\model_provision.py `
  --manifest .\models\manifests\qwen3_tts_customvoice.json `
  --allow-modelscope-download

# Qwen VoiceDesign：用文本描述设计音色
.\.venv\Scripts\python.exe .\scripts\model_provision.py `
  --manifest .\models\manifests\qwen3_tts_voicedesign.json `
  --allow-modelscope-download

# CosyVoice3，同时恢复项目内离线 WeText
.\.venv\Scripts\python.exe .\scripts\model_provision.py `
  --manifest .\models\manifests\cosyvoice3.json `
  --allow-modelscope-download

# IndexTTS2，同时恢复 BigVGAN、MaskGCT、CampPlus、W2V-BERT 等辅助文件
.\.venv\Scripts\python.exe .\scripts\model_provision.py `
  --manifest .\models\manifests\index_tts2.json `
  --allow-modelscope-download
```

中文或中英混合字幕可另外安装 Turbo 与中文对齐模型：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py `
  --manifest .\models\manifests\whisper_large_v3_turbo.json `
  --allow-modelscope-download

.\.venv\Scripts\python.exe .\scripts\model_provision.py `
  --manifest .\models\manifests\whisperx_zh_alignment.json `
  --allow-modelscope-download
```

`large-v3` 适合质量优先，`distil-large-v3` 只建议用于纯英文。字幕不是 TTS 合成的前置条件。

## 7. 验证安装

### 7.1 完整安装

确保平台及 8001～8004 工作进程均已停止，然后运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --startup
```

成功标准是最后一行显示：

```text
Doctor summary: ... passed, 0 failed
```

发布前或怀疑文件损坏时执行完整 SHA256 校验：

```powershell
.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --full
```

doctor 会检查三个环境、八份 manifest、本机配置、FFmpeg 和五个端口。端口已被正在运行的平台占用时会失败，这是启动前检查的预期行为。

### 7.2 部分开发安装

完整 doctor 会报告未安装的环境和模型，因此部分安装只验证实际选择的边界。例如 Qwen CustomVoice：

```powershell
uv pip check --python .\.venv\Scripts\python.exe
uv pip check --python .\environments\qwen3tts\.venv\Scripts\python.exe

.\.venv\Scripts\python.exe .\scripts\model_preflight.py `
  --manifest .\models\manifests\qwen3_tts_customvoice.json

ffmpeg -version
ffprobe -version
```

将 manifest 和环境路径换成实际安装的后端。每条命令都必须以退出码 0 结束。

## 8. 启动、健康检查与首次样音

### 8.1 完整安装

默认只允许本机访问：

```bat
start_platform.bat
```

只有确实需要受信任局域网客户端时才使用：

```bat
start_platform.bat --lan
```

`--lan` 监听 `0.0.0.0:9000`。平台没有账户认证和 TLS，必须用 Windows 防火墙限制可信来源，不能做路由器端口映射或直接暴露公网。

### 8.2 部分开发安装

部分安装不要运行 `start_platform.bat`，直接启动平台：

```powershell
.\.venv\Scripts\python.exe -u -m uvicorn `
  dubbing_service.app.main:app `
  --host 127.0.0.1 `
  --port 9000
```

浏览器打开 `http://127.0.0.1:9000/web/`。平台打开时不会自动加载 TTS。

### 8.3 平台与工作进程健康检查

平台启动后，在另一个 PowerShell 窗口运行：

```powershell
Invoke-RestMethod http://127.0.0.1:9000/api/v1/ready
Invoke-RestMethod http://127.0.0.1:9000/api/v1/health
Invoke-RestMethod http://127.0.0.1:9000/api/v1/settings/tts-service/status
```

`ready` 应返回 `status = ok`。尚未启动 TTS 时，health 中的 `tts_service_running` 或 `tts_model_loaded` 为 false/null 属于正常状态。

随后进入 WebUI 的“平台配置”：

1. 选择已经安装的 TTS 后端并保存；
2. 点击启动/加载模型；
3. 等待状态显示运行中、模型已加载；
4. 再检查对应工作进程：

```powershell
# 根据当前后端只检查一个端口
Invoke-RestMethod http://127.0.0.1:8001/health  # IndexTTS2
Invoke-RestMethod http://127.0.0.1:8002/health  # CosyVoice3
Invoke-RestMethod http://127.0.0.1:8003/health  # Qwen VoiceDesign
Invoke-RestMethod http://127.0.0.1:8004/health  # Qwen CustomVoice
```

响应应包含正确的后端身份，并显示 `model_loaded = true` 和预期 CUDA 设备。切换模型前先在 UI 停止当前工作进程。

首次样音建议使用 Qwen CustomVoice，或为 IndexTTS2/CosyVoice3 准备一段已授权的 3～10 秒干净参考声音：

1. 新建测试项目和一个角色；
2. 输入一到两句无敏感内容的短文本；
3. 生成一段音频并完整试听；
4. 确认没有报错、空音频、削波或明显异常速度。

## 9. 可选：配置 DeepSeek

DeepSeek 只用于自动分段；没有 API Key 时仍可手工分段或导入 JSON。推荐在平台已经启动后，通过“平台配置 → LLM 提供商”新建：

```text
名称       DeepSeek
Base URL   https://api.deepseek.com
API Key    你的 DeepSeek API Key
```

保存后，密钥进入操作系统凭据库，本地 JSON 只保存引用。内置模型预设为：

```text
deepseek-v4-flash    默认分段模型
deepseek-v4-pro      可重试错误或输出校验失败时最多回退一次
```

配置后可执行真实连接检查；命令会产生外部请求并可能计费：

```powershell
.\.venv\Scripts\python.exe .\scripts\test_llm_connection.py --include-pro
```

也可以只向当前 PowerShell 会话注入环境变量，并在**同一窗口**启动：

```powershell
$env:DUBBING_LLM_BASE_URL = "https://api.deepseek.com"
$env:DUBBING_LLM_API_KEY = "在本机填写真实密钥"
.\start_platform.bat
```

关闭该 PowerShell 后变量失效。DeepSeek 会收到用于分段的原始文本；敏感、保密或受限制内容应使用手工分段或本地生成的 JSON。

## 10. 首次完整验收

不要直接使用长篇商业项目。先以 2～5 分钟脚本完成：

1. 新建测试项目、章节和角色；
2. 上传合法参考声音，或使用 Qwen CustomVoice 内置音色；
3. 手工分段，或在测试项目中验证 DeepSeek 分段；
4. 至少合成 A/B 两个 Take 并分别试听；
5. 保存一次非破坏编辑，并渲染为新 Take；
6. 采用最终 Take，逐段完成审听；
7. 导入一段已授权 BGM 或 SFX；
8. 导出预览、运行质量检查并生成正式 Master；
9. 生成并解压发布 ZIP，核对音频、字幕、元数据、授权清单和校验和；
10. 用耳机与普通扬声器完整听一遍。

具体操作见 [有声作品制作使用指南](user_guide.md)，故障处理见 [生产运维手册](operations.md)。

## 11. 更新与备份

更新代码、重建环境或恢复模型前，先停止平台和 TTS，再验证并创建冷备份：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\backup_project.ps1 -ValidateOnly
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\backup_project.ps1
```

默认项目数据位于 `dubbing_projects/`。如果通过 `DUBBING_PROJECTS_ROOT` 使用仓库外数据盘，备份脚本不会自动纳入该外部目录，必须另做卷备份或冷备份。

备份可能包含作品文本、参考声音和授权素材，应加密保管。API Key 和普通模型权重不在默认备份中；换机后需重新注入密钥，并按 manifest 恢复模型。
