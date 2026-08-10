# AI 有声工作室（IndexTTS 多后端配音平台）

这是一个面向 Windows 本机制作的 AI 有声内容工作室，用于把故事、资讯、知识内容和播客脚本制作成可审听、可混音、可打包发布的有声作品。

项目以 [IndexTTS2](https://github.com/index-tts/index-tts) 为上游基础，增加了项目管理、DeepSeek 分段、四种 TTS 后端、不可变 Take、独立 A/B 版本、非破坏编辑、审听、BGM/Ambience/SFX、Master、字幕和发布包等完整工作流。

> 当前支持边界：Windows 10/11、NVIDIA GPU、单机或受信任局域网。平台没有账户认证，严禁直接暴露到公网。商业发布前还必须分别确认模型、声音、音乐、音效和原始文本的授权。

## 核心能力

- TTS：IndexTTS2、Qwen3-TTS CustomVoice、Qwen3-TTS VoiceDesign、CosyVoice3。
- 脚本：手工分段、JSON 导入，或使用 DeepSeek Flash/Pro 自动分段。
- 音色：参考音频、全局音色库、角色级后端、CustomVoice 内置音色和 VoiceDesign 音色设计。
- 配音审听：每段独立 A/B、不可变 Take、历史恢复/固定、最终 Take 选择和审听状态。
- 非破坏编辑：裁剪、静音、音量、变速、淡入淡出、头尾留白；需要固化时渲染为新的 Take。
- 声音设计：Dialogue/BGM/Ambience/SFX 四轨、素材授权记录、增益、循环、淡入淡出、对白交叉淡化和 BGM 自动闪避。
- 交付：响度与峰值质检、不可变 Master、WAV/MP3、SRT/VTT、章节、封面、元数据、授权清单和校验和 ZIP。

## 先读这个

从 GitHub 新下载项目时，不包含以下本机内容：模型权重、API Key、虚拟环境、作品项目、参考声音、BGM/SFX、缓存、日志和生成音频。

完整部署顺序是：

1. 安装 Git、Git LFS、uv、FFmpeg 和 NVIDIA 驱动；
2. 克隆仓库并重建三个隔离的 uv 环境；
3. 从 ModelScope 下载所需模型；
4. 在本机配置 DeepSeek API Key；
5. 运行 doctor；
6. 启动平台并完成首个短项目验收。

详细命令见 [GitHub 新机安装与模型配置](docs/installation.md)。实际制作方法见 [有声作品制作使用指南](docs/user_guide.md)。

## 最短启动路径

```powershell
git clone https://github.com/tadashijlu/ai-audio-studio.git
Set-Location .\ai-audio-studio
git lfs install
git lfs pull

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1 -Runtime All
Copy-Item .\dubbing_service\config.example.json .\dubbing_service\config.json
```

模型下载完成后，在平台和 TTS 均未运行时检查：

```powershell
.\.venv\Scripts\python.exe .\scripts\platform_doctor.py --startup
```

默认仅本机启动：

```bat
start_platform.bat
```

只有确实需要受信任局域网访问时才使用：

```bat
start_platform.bat --lan
```

浏览器访问：`http://127.0.0.1:9000/web/`。

`start_platform.bat --check` 只做启动预检；如果 9000 或 8001～8004 已被正在运行的平台占用，端口检查失败属于预期结果，不代表代码损坏。

## 运行架构

平台与三个模型栈必须保持隔离，不能把依赖合并到同一个环境。

| 服务 | 端口 | 解释器 | 项目内模型目录 |
| --- | ---: | --- | --- |
| 平台 + IndexTTS2 | 9000 / 8001 | `.venv` | `models/index_tts` |
| CosyVoice3 | 8002 | `environments/cosyvoice/.venv` | `models/cosyvoice3/Fun-CosyVoice3-0.5B` |
| Qwen VoiceDesign | 8003 | `environments/qwen3tts/.venv` | `models/qwen3_tts/Qwen3-TTS-VoiceDesign` |
| Qwen CustomVoice | 8004 | `environments/qwen3tts/.venv` | `models/qwen3_tts/Qwen3-TTS-CustomVoice` |

平台同一时间只托管一个 GPU TTS 工作进程。WebUI 不会在打开时自动加载模型；请在“平台配置”中选择后端并明确启动。

## DeepSeek 配置

推荐在 WebUI 的“平台配置 → LLM 提供商”中新建 DeepSeek：

- Base URL：`https://api.deepseek.com`
- API Key：填写自己的 Key；保存后进入操作系统凭据库，不进入 Git 配置文件
- 首选模型：`deepseek-v4-flash`
- 备选模型：`deepseek-v4-pro`

也可以只对当前 PowerShell 会话设置环境变量：

```powershell
$env:DUBBING_LLM_BASE_URL = "https://api.deepseek.com"
$env:DUBBING_LLM_API_KEY = "请在本机填写，不要写入仓库"
```

DeepSeek 会收到用于自动分段的原始文本。涉及敏感、保密或受限制内容时，请改用手工分段或导入本地生成的 JSON。

## 本地数据与 Git 边界

以下内容由 `.gitignore` 排除，不应通过 `git add -f` 强行提交：

- `.env`、`dubbing_service/config.json`、`dubbing_service/llm_providers.json`
- `models/` 下的权重、下载缓存、运行缓存和隔离区
- `.venv*` 及三个运行环境中的 `.venv`
- `dubbing_projects/`、音色库、参考声音、BGM/SFX 和其他作品素材
- `outputs/`、`temp/`、日志、字幕中间文件和生成音频

发布前可运行无模型、无密钥、无用户数据的白名单检查：

```powershell
.\scripts\export_source.ps1 -ValidateOnly
```

## 文档导航

- [新机安装、模型下载与首次验收](docs/installation.md)
- [从脚本到发布包的完整使用指南](docs/user_guide.md)
- [第三方来源、归属与许可边界](NOTICE.md)
- [架构说明](docs/architecture.md)
- [生产运维与故障排查](docs/operations.md)
- [启动与巡检清单](docs/production_startup_and_inspection_checklist.md)
- [模型目录与供应链规则](models/README.md)
- [第三方源码与版本基线](docs/third_party_sources.md)
- [本机恢复及审查报告](docs/recovery_report.md)
- [AI 有声导演工作室生产化方案](docs/ai_audio_studio_production_plan.md)
- [安全策略](SECURITY.md)

上游 IndexTTS2 的原始介绍与示例保存在 [archive/README_ROOT_WITH_UPSTREAM_REFERENCE.md](archive/README_ROOT_WITH_UPSTREAM_REFERENCE.md)。根目录 `webui.py` 是上游单模型 Gradio 演示，不是本平台入口。

## 安全与许可证

- 一键启动默认只绑定 `127.0.0.1`。平台没有登录、权限管理或 TLS；显式 `--lan` 模式必须配置防火墙。
- 不要在 Issue、日志、截图或提交记录中粘贴 API Key、私人脚本、参考声音或作品文件。
- 当前 GitHub 仓库按私人项目保存，并不表示已经公开开源。平台新增代码尚未授予独立的公开再分发许可；IndexTTS2、MaskGCT、CosyVoice、Qwen、WebUI vendor 及其他第三方组件继续适用各自许可证或使用条款。
- 当前 IndexTTS2 辅助模型清单包含具有非商业限制的 MaskGCT 权重。技术可运行不等于已经获准商业生产或再分发。
- 克隆、合成或发布任何人的声音前，必须取得明确授权，并遵守适用的隐私、人格权、版权和 AI 内容标识规则。
