<p align="center">
  <img src="dubbing_service/webui/brand/wordmark.svg" width="560" alt="AI 有声工作室 / AI Audio Studio" />
</p>

<h1 align="center">AI 有声工作室</h1>

<p align="center">
  面向故事、资讯、知识内容与播客的 Windows 本地 AI 有声制作工作台
</p>

<p align="center">
  <a href="https://github.com/tadashijlu/ai-audio-studio/actions/workflows/windows-no-model-ci.yml"><img alt="Windows no-model CI" src="https://github.com/tadashijlu/ai-audio-studio/actions/workflows/windows-no-model-ci.yml/badge.svg" /></a>
  <img alt="Platform layer license: Apache-2.0" src="https://img.shields.io/badge/platform%20layer-Apache--2.0-22c28f" />
  <img alt="Windows 10 and 11" src="https://img.shields.io/badge/Windows-10%20%7C%2011-2563eb" />
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776ab" />
</p>

<p align="center">
  <a href="README_EN.md">English</a> ·
  <a href="docs/installation.md">安装</a> ·
  <a href="docs/user_guide.md">使用指南</a> ·
  <a href="docs/architecture.md">架构</a> ·
  <a href="CONTRIBUTING.md">参与贡献</a>
</p>

> [!IMPORTANT]
> 当前版本为 `0.2.0-beta.1`，面向 Windows + NVIDIA GPU，适合个人本地制作与技术预览。平台自研层由 **TadashiJLU** 以 Apache-2.0 开源；仓库中的 IndexTTS2、CosyVoice、Matcha-TTS、浏览器依赖及另行下载的模型继续适用各自许可证。它们并未被统一重新许可，商业使用前请先阅读[许可证边界](LICENSES/README.md)。

AI 有声工作室把零散的 TTS 调用整理成一条可追溯的制作链：

**脚本 → 音色 → A/B Take → 非破坏编辑 → 声音设计 → Master → 发布包**

它不是“输入文字、下载一个音频”的演示页，而是为反复审听、修改、混音和正式交付设计的本地工作室。

![AI 有声工作室配音审听界面](docs/assets/screenshots/studio-overview.png)

## 为什么做这个项目

制作可发布的有声作品，真正耗时的通常不是单次合成，而是脚本分段、角色与音色一致性、多个版本比较、局部返工、声音设计、响度质检和交付整理。本项目将这些环节保存在同一个工程中，优先保证：

- **可回退**：A/B 槽位相互独立，每次生成形成不可变 Take，除非手动清理；
- **非破坏**：裁剪、静音、增益、变速和淡化先记录为编辑配方，原始 Take 不被覆盖；
- **可审听**：每段有待审、需修改、通过、问题备注和最终 Take 选择；
- **可交付**：Dialogue / BGM / Ambience / SFX 四轨汇入 Master，并输出字幕、元数据、授权清单与校验和；
- **本地优先**：作品、声音、密钥、模型和生成结果默认只留在本机，不进入 Git。

## 核心能力

| 工作区 | 已实现能力 |
| --- | --- |
| 脚本与角色 | 手工分段、JSON 导入、可选 DeepSeek V4 Flash / Pro 分段、章节/场景、角色 Voice Bible、发音词典 |
| 音色 | 参考音频、全局音色库、角色级后端、Qwen CustomVoice 内置音色、Qwen VoiceDesign 文本音色设计 |
| 配音审听 | 每段独立 A/B、不可变 Take、历史恢复/固定、最终 Take 选择、人工审听状态、持久任务队列 |
| 非破坏编辑 | 裁剪、删除区间、静音、音量、变速、淡入淡出、头尾留白；需要固化时渲染为新 Take |
| 声音设计 | Dialogue / BGM / Ambience / SFX 四轨、Stable Audio 3 句子级音效规划/生成、素材授权记录、循环、增益、淡化、对白交叉淡化、BGM ducking |
| 导出发布 | Preview、响度/峰值质检、不可变 Master、WAV/MP3、SRT/VTT、章节、封面、元数据、授权清单、ZIP 校验和 |

<details>
<summary><strong>查看更多界面</strong></summary>

### 声音设计

![声音设计工作区](docs/assets/screenshots/sound-design.png)

### Master 与发布

![Master 与发布工作区](docs/assets/screenshots/publish-workflow.png)

</details>

## TTS 后端

| 后端 | 适合场景 | 音色方式 | 许可提醒 |
| --- | --- | --- | --- |
| IndexTTS2 | 主播旁白、参考声音复刻、细粒度情感 | 参考音频 + 8 维情感/文本/音频控制 | 上游自定义许可；当前辅助清单含 `CC-BY-NC-4.0` 的 MaskGCT，不能默认用于商业生产 |
| Qwen3-TTS CustomVoice | 快速角色分配、无需参考音频 | 内置音色 + 情感指令 | 权重不进仓库；使用前核对模型页条款 |
| Qwen3-TTS VoiceDesign | 文本描述设计新音色 | 自然语言音色描述 | 权重不进仓库；使用前核对模型页条款 |
| CosyVoice3 | 自然语言控制语气和风格 | 参考声音/指令控制 | vendored 代码为 Apache-2.0；模型权重仍按模型页条款 |

模型权重、API Key、参考声音和作品数据均不随仓库发布。模型供应脚本只从 manifest 固定的 ModelScope 来源下载，并在写入项目目录前校验文件大小和 SHA256。

## 快速开始：完整功能安装

### 1. 准备环境

- Windows 10/11、NVIDIA GPU 和可用驱动；
- 已验证基线：RTX 4070 Ti SUPER 16 GiB；更低显存尚未形成正式支持矩阵；
- 建议至少 **80 GiB** 可用磁盘空间，用于约 31.86 GiB 模型、下载缓存和三个 uv 环境；
- 首次安装需要联网，日常推理按离线模型目录运行。

在 PowerShell 中安装系统工具：

```powershell
winget install --id Git.Git -e
winget install --id GitHub.GitLFS -e
winget install --id astral-sh.uv -e
winget install --id Gyan.FFmpeg -e
```

安装后重新打开 PowerShell。

### 2. 克隆、环境与配置

```powershell
git lfs install
git clone https://github.com/tadashijlu/ai-audio-studio.git
Set-Location .\ai-audio-studio
git lfs pull

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Runtime All
Copy-Item .\dubbing_service\config.example.json .\dubbing_service\config.json
```

项目刻意保留四个隔离环境，不能把依赖合并到一个 Python 环境：

- `.venv`：平台、IndexTTS2、字幕；
- `environments/qwen3tts/.venv`：两个 Qwen3-TTS 后端；
- `environments/cosyvoice/.venv`：CosyVoice3。
- `environments/stableaudio/.venv`：Stable Audio 3 Small-SFX。

### 3. 从固定来源安装全部模型

先阅读 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和各 manifest 的许可字段。预演不会修改模型文件：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_models.ps1 -Profile Full -DryRun
```

`Full` 还包含官方门控的 Stable Audio 3 Small-SFX。先在模型页接受 Stability AI 与 Gemma 条款并完成适用注册，再将 read token 仅放入当前 PowerShell 进程；若暂不启用音效生成，请选择现有的 `Tts` / `Core` / `Subtitles` profile：

```powershell
$env:HF_TOKEN = '<Hugging Face read token>'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_models.ps1 `
  -Profile Full `
  -AcceptThirdPartyLicenses `
  -StableAudioLicenseTier community
Remove-Item Env:HF_TOKEN
```

`-AcceptThirdPartyLicenses` 只表示你已阅读相关条款，不会额外授予商业或再分发权利。下载中断后可安全重跑，已有正确文件不会重复恢复。

### 4. 检查并启动

平台和 TTS 工作进程均未运行时执行：

```powershell
.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --startup
.\start_platform.bat
```

doctor 应以 `0 failed` 结束。启动后打开 `http://127.0.0.1:9000/web/`，进入“平台配置”，选择一个 TTS 后端并点击启动/加载模型。

> [!WARNING]
> 默认启动仅绑定 `127.0.0.1`。平台目前没有账户、TLS 或权限系统；不要直接暴露到公网。只有受信任局域网确有需要时才使用 `start_platform.bat --lan`，并自行配置防火墙。

完整命令、部分后端开发模式、健康检查和首次样音见[安装指南](docs/installation.md)。

## 可选：DeepSeek 自动分段

平台不依赖 LLM 也能手工分段或导入 JSON。需要自动分段时，在平台启动后进入“平台配置 → LLM 提供商”，新建 DeepSeek：

- Base URL：`https://api.deepseek.com`
- 首选模型：`deepseek-v4-flash`
- 备选模型：`deepseek-v4-pro`
- API Key：保存到操作系统凭据库，不写入 Git 配置

这两个模型 ID 与 Base URL 已按 [DeepSeek 官方 API 文档](https://api-docs.deepseek.com/quick_start/pricing/)核对。原始文本会发送给 DeepSeek；敏感或受限制内容请使用手工分段或本地 JSON。

## 运行架构

平台只管理一个 GPU TTS 工作进程，切换后端时会释放旧进程，避免多个模型同时争用显存。

| 服务 | 端口 | 解释器 | 项目内模型目录 |
| --- | ---: | --- | --- |
| Web/API 平台 | 9000 | `.venv` | — |
| IndexTTS2 | 8001 | `.venv` | `models/index_tts` |
| CosyVoice3 | 8002 | `environments/cosyvoice/.venv` | `models/cosyvoice3/Fun-CosyVoice3-0.5B` |
| Qwen VoiceDesign | 8003 | `environments/qwen3tts/.venv` | `models/qwen3_tts/Qwen3-TTS-VoiceDesign` |
| Qwen CustomVoice | 8004 | `environments/qwen3tts/.venv` | `models/qwen3_tts/Qwen3-TTS-CustomVoice` |
| Stable Audio 3 Small-SFX | 8005 | `environments/stableaudio/.venv` | `models/stable_audio_3/small-sfx` |

更多模块边界、持久化结构和并发约束见[架构说明](docs/architecture.md)。

## 数据与隐私边界

以下内容被 `.gitignore` 排除，禁止使用 `git add -f` 强行提交：

- `.env`、`dubbing_service/config.json`、`dubbing_service/llm_providers.json`；
- `models/` 下的权重、下载缓存、运行缓存和隔离区；
- `.venv*` 以及三个隔离运行环境中的 `.venv`；
- `dubbing_projects/`、音色库、参考声音、BGM/SFX 和其他作品素材；
- `outputs/`、`temp/`、日志、字幕中间文件和生成音频。

发布源码前运行白名单检查：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\export_source.ps1 -ValidateOnly
```

## 文档

### 开始使用

- [新机安装、模型下载与首次验收](docs/installation.md)
- [从脚本到发布包的完整使用指南](docs/user_guide.md)
- [模型目录与供应链规则](models/README.md)

### 架构与运维

- [系统架构](docs/architecture.md)
- [生产运维与故障排查](docs/operations.md)
- [启动与巡检清单](docs/production_startup_and_inspection_checklist.md)
- [安全策略](SECURITY.md)
- [依赖安全与兼容风险](docs/dependency_security.md)

### 许可证与维护

- [多许可证路径边界](LICENSES/README.md)
- [第三方源码、浏览器依赖与模型说明](THIRD_PARTY_NOTICES.md)
- [固定的第三方源码基线](docs/third_party_sources.md)
- [参与贡献](CONTRIBUTING.md) · [支持渠道](SUPPORT.md) · [版本记录](CHANGELOG.md)

上游 IndexTTS2 的原始介绍保存在 `archive/`。根目录 `webui.py` 是上游单模型 Gradio 演示，不是本平台入口。

## 当前边界

- 当前重点是单人本地生产，不建设多用户、云同步、平台发布 API 或完整 DAW；
- BGM 目前仍是素材导入；SFX 可选用 Stable Audio 3 Small-SFX 本地生成，但必须先通过当前许可回执和真实 12 类/30 次质量门；尚未部署音乐生成模型；
- GitHub CI 不下载模型、不占用 GPU；真实 TTS、字幕和发布包 E2E 需在本地 GPU 机器验收；
- 只支持清单固定、SHA256 校验并放在项目目录内的模型；不要用来加载来源不明的
  checkpoint、Transformers 仓库或训练配置；
- IndexTTS2 当前辅助依赖中的 MaskGCT 具有非商业限制，替换或获得单独许可前不能把该链路视为商业放行。
- Stable Audio 3 权重适用 Stability AI Community/Enterprise License，bundled T5Gemma 另受 Gemma 条款约束；平台 Apache-2.0 许可不覆盖这些权重。

## 参与贡献

欢迎提交可复现的 Bug、文档改进和聚焦有声制作流程的功能建议。开始前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 与 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。安全问题请按 [SECURITY.md](SECURITY.md) 私下报告，不要公开粘贴密钥、声音或作品数据。

## 许可证与归属

Copyright 2026 **TadashiJLU**.

平台自研层采用 [Apache License 2.0](LICENSE)。本仓库是多许可证分发：上游 IndexTTS2 及其派生修改、vendored CosyVoice/Matcha-TTS、浏览器依赖、模型权重和示例媒体仍受各自条款约束。以 [LICENSES/README.md](LICENSES/README.md)、相邻许可证和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 为准。

本项目基于 [index-tts/index-tts](https://github.com/index-tts/index-tts) 的固定上游提交构建；本项目的修改不受上游原权利人背书、保证或担保。
