# 支持指南

Copyright 2026 TadashiJLU.

## 支持范围

当前项目优先支持文档规定的 Windows 桌面安装、IndexTTS2、Qwen3-TTS、CosyVoice3，以及文本到可发布有声作品的核心流程。它面向本地个人工作站，不承诺公网 SaaS、多用户、视频制作或第三方平台运营支持。

使用前请先阅读 `docs/installation.md`、`docs/user_guide.md` 和 `SECURITY.md`，再运行：

```powershell
.\start_platform.bat --check
```

## 在哪里提问

- 可复现的软件缺陷：使用 Bug Report Issue 表单。
- 明确改善核心音频生产流程的建议：使用 Feature Request 表单。
- 安全漏洞：遵循 `SECURITY.md`，不要公开敏感细节。
- 第三方模型、驱动、CUDA、DeepSeek 或发布平台问题：优先查阅对应服务的官方文档；维护者只能尽力协助项目集成部分。

提交问题时请提供分支或提交号、Windows 版本、GPU 与显存、Python/uv 版本、TTS 后端、最小复现步骤和脱敏日志。不要上传 API Key、私人音频、作品文本、模型权重、项目数据或商业素材。

## 不构成的承诺

维护者不保证响应时间，也不提供法律、版权、医疗、财务或平台合规意见。模型输出质量和商业发布权利必须由发布者结合人工审听、素材凭证、适用法律与目标平台规则自行确认。
