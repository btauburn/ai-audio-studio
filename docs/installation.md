# GitHub 新机安装与模型配置

本文面向从 GitHub 全新克隆仓库的 Windows 用户。命令均在仓库根目录执行，不依赖全局 Python，也不要使用全局 `pip` 修补模型环境。

## 1. 支持范围与磁盘预算

当前经过验证的运行路径：

- Windows 10/11；
- NVIDIA GPU 与可用的驱动；
- PowerShell 5.1 或 PowerShell 7；
- Git + Git LFS；
- uv；
- FFmpeg 和 ffprobe。

CPU-only、AMD GPU、macOS 模型推理和 Linux 部署尚未作为本版本验收范围。Mac 可以作为受信任局域网中的浏览器客户端，但 TTS 工作进程仍在 Windows 主机运行。

模型最终占用约为：

| 模型组 | 约占用 |
| --- | ---: |
| IndexTTS2 | 8.26 GiB |
| CosyVoice3 + WeText | 8.18 GiB |
| 两个 Qwen3-TTS 模型 | 8.42 GiB |
| 全部字幕/对齐模型 | 6.99 GiB |
| 全部最终模型 | 31.85 GiB |

下载缓存、原子复制临时文件、三个虚拟环境和项目输出还会占用额外空间。完整安装建议预留至少 80 GiB；只安装当前需要的后端可明显减少占用。

## 2. 安装系统工具

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

## 3. 克隆仓库

```powershell
git clone https://github.com/tadashijlu/ai-audio-studio.git
Set-Location .\ai-audio-studio
git lfs install
git lfs pull
```

如果要使用尚未合并到默认分支的发布 PR，请按该 PR 页面给出的分支名检出；正常发布后直接使用默认分支即可。

仓库只包含源码、锁文件、模型 manifest 和说明文档。模型、API Key、作品、声音和生成结果不会随 Git 下载。

## 4. 重建三个 uv 环境

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Runtime All
```

该脚本会固定安装 Python 3.12.13 和 3.10.19，并分别创建：

```text
.venv/                              平台 + IndexTTS2
environments/qwen3tts/.venv/        Qwen VoiceDesign + CustomVoice
environments/cosyvoice/.venv/       CosyVoice3
```

随后分别执行依赖一致性检查。三个环境的 PyTorch、Transformers 和音频依赖存在版本冲突，不能合并。

只修复某个环境时：

```powershell
.\scripts\bootstrap.ps1 -Runtime Platform
.\scripts\bootstrap.ps1 -Runtime Qwen
.\scripts\bootstrap.ps1 -Runtime CosyVoice
```

## 5. 创建本机配置

```powershell
Copy-Item .\dubbing_service\config.example.json .\dubbing_service\config.json
```

`config.json` 已被 Git 忽略。不要在 `config.example.json`、README、Issue 或提交记录里填写真实密钥。

默认端口：

| 服务 | 地址 |
| --- | --- |
| 平台 | `http://127.0.0.1:9000` |
| IndexTTS2 | `http://127.0.0.1:8001` |
| CosyVoice3 | `http://127.0.0.1:8002` |
| Qwen VoiceDesign | `http://127.0.0.1:8003` |
| Qwen CustomVoice | `http://127.0.0.1:8004` |
| DeepSeek API | `https://api.deepseek.com` |

## 6. 下载模型

### 6.1 供应规则

`models/manifests/*.json` 固定了目标路径、ModelScope 模型、revision、文件大小和 SHA256。`model_provision.py` 只会在显式传入 `--allow-modelscope-download` 时联网；文件校验通过后才会进入运行目录，损坏的旧文件会移到 `models/.quarantine/`。

先预演不会下载：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py `
  --manifest .\models\manifests\qwen3_tts_customvoice.json `
  --dry-run --allow-modelscope-download
```

### 6.2 两个 Qwen3-TTS 模型

Qwen CustomVoice 是内置音色与风格指令后端；VoiceDesign 用文字设计音色。分别下载：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\qwen3_tts_customvoice.json --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\qwen3_tts_voicedesign.json --allow-modelscope-download
```

### 6.3 CosyVoice3

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\cosyvoice3.json --allow-modelscope-download
```

该 manifest 同时恢复 CosyVoice3 和项目内离线 WeText 文件。

### 6.4 IndexTTS2

IndexTTS2 主模型及 BigVGAN、MaskGCT、CampPlus、W2V-BERT 等辅助文件均由 manifest 固定的 ModelScope 来源恢复：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\index_tts2.json --allow-modelscope-download
```

BigVGAN 使用 `nv-community/bigvgan_v2_22khz_80band_256x` 的固定 ModelScope 文件 revision；供应器会核对其大小和 SHA256，不接受运行时隐式下载。

### 6.5 字幕模型

中文或中英混合内容建议先安装 Turbo 与中文对齐模型：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisper_large_v3_turbo.json --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisperx_zh_alignment.json --allow-modelscope-download
```

需要最高质量或纯英文快速识别时再安装：

```powershell
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisper_large_v3.json --allow-modelscope-download
.\.venv\Scripts\python.exe .\scripts\model_provision.py --manifest .\models\manifests\whisper_distil_large_v3.json --allow-modelscope-download
```

`distil-large-v3` 仅建议用于纯英文。字幕不是 TTS 合成的前置条件，不需要时可以暂不安装。

### 6.6 完整安装八份 manifest

一键启动的严格 doctor 会检查四个 TTS 后端和全部字幕选项。要恢复完整功能，可统一预演并下载八份 manifest：

```powershell
$manifests = Get-ChildItem .\models\manifests\*.json

foreach ($manifest in $manifests) {
    & .\.venv\Scripts\python.exe .\scripts\model_provision.py `
        --manifest $manifest.FullName `
        --allow-modelscope-download `
        --dry-run
    if ($LASTEXITCODE -ne 0) { throw "模型预演失败：$($manifest.Name)" }
}

foreach ($manifest in $manifests) {
    & .\.venv\Scripts\python.exe .\scripts\model_provision.py `
        --manifest $manifest.FullName `
        --allow-modelscope-download
    if ($LASTEXITCODE -ne 0) { throw "模型下载失败：$($manifest.Name)" }
}
```

只安装部分模型时可以手工启动平台进行开发，但 `start_platform.bat` 的完整准入检查会报告缺失项。正式使用建议完成全部八份 manifest。

## 7. 配置 DeepSeek

推荐先启动平台，再在“平台配置 → LLM 提供商”中新建提供商：

```text
名称       DeepSeek
Base URL   https://api.deepseek.com
API Key    你的 DeepSeek API Key
```

保存后密钥进入操作系统凭据库，本地 JSON 只保存引用。然后确认或新建两个 LLM 配置：

```text
deepseek-v4-flash    默认，优先用于分段
deepseek-v4-pro      可重试错误或输出校验失败时最多回退一次
```

也可以只在当前 PowerShell 会话注入：

```powershell
$env:DUBBING_LLM_BASE_URL = "https://api.deepseek.com"
$env:DUBBING_LLM_API_KEY = "你的密钥"
```

不要使用 `setx` 把密钥写进命令历史或共享脚本。LLM 功能是可选的；没有 Key 时仍可手工分段或导入 JSON。

## 8. 启动前检查

确保平台和所有 TTS 进程已经停止：

```powershell
.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --startup
```

需要读取所有模型并校验 SHA256 时：

```powershell
.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --full
```

doctor 会检查环境、模型、配置、路径和端口。若报告端口占用，先判断是否已有健康的平台实例；不要为了通过检查而强制结束不明进程。

## 9. 启动平台

### 仅本机使用（默认、推荐）

```bat
start_platform.bat
```

也可以直接运行：

```powershell
.\.venv\Scripts\python.exe -u -m uvicorn dubbing_service.app.main:app --host 127.0.0.1 --port 9000
```

打开 `http://127.0.0.1:9000/web/`。

### 受信任局域网使用

```bat
start_platform.bat --lan
```

`--lan` 会显式监听 `0.0.0.0:9000`，便于同一局域网中的其他设备访问。平台没有登录认证和 TLS，必须用 Windows 防火墙限制可信地址，不能做路由器端口映射，也不能直接暴露公网。

页面打开后进入“平台配置”：

1. 选择一个 TTS 后端；
2. 保存配置；
3. 点击启动/加载模型；
4. 等待状态显示健康且模型已加载；
5. 切换模型前先停止当前工作进程。

## 10. 首次验收

不要直接用长篇商业项目作为第一次运行。先用 2～5 分钟脚本完成：

1. 新建测试项目和一个角色；
2. 上传合法且清晰的参考声音，或使用 Qwen CustomVoice 内置音色；
3. 手工分段或测试 DeepSeek 分段；
4. 至少合成两个 A/B Take 并分别试听；
5. 做一次非破坏编辑并渲染为新 Take；
6. 将最终版本标记为采用并完成审听；
7. 导入一段已授权 BGM 或 SFX；
8. 生成预览、运行质量检查、生成正式 Master；
9. 生成并解压发布 ZIP，核对 WAV/MP3、字幕、元数据、授权清单与校验和；
10. 用耳机和普通扬声器完整听一遍。

功能操作详见 [有声作品制作使用指南](user_guide.md)，故障处理详见 [生产运维手册](operations.md)。

## 11. 更新与备份

更新代码前停止平台并备份。默认项目数据在 `dubbing_projects/`，建议通过 `DUBBING_PROJECTS_ROOT` 指向独立数据盘。

```powershell
.\scripts\backup_project.ps1 -ValidateOnly
.\scripts\backup_project.ps1
```

备份仍可能包含作品文本、参考声音和授权素材，应加密保管。API Key 不在备份内；换机后需要重新注入。
