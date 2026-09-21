# 贡献指南

Copyright 2026 TadashiJLU.

感谢你帮助改进 AI Audio Studio。项目当前聚焦 Windows 桌面环境中的高质量有声作品制作：文本与分段、音色与 TTS、Take 审听、非破坏编辑、声音设计、混音质检和发布包。

## 开始之前

较大的设计变更、新模型接入和工作流重构，请先提交 Feature Request，说明实际制作场景、运行成本、模型或素材来源及许可边界。缺陷修复可直接提交小而集中的 Pull Request，但建议先确认没有重复 Issue。

贡献者必须遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) 和 [SECURITY.md](SECURITY.md)。不要在 Issue、提交、日志或测试夹具中放入密钥、私人音频、作品原稿、模型权重、生成内容或无权再分发的素材。

## 开发环境

本项目使用 Python 3.12、uv 和多个相互隔离的 TTS 环境。完整的本机安装以 `docs/installation.md` 为准。根环境含 CUDA 版 PyTorch 和模型运行依赖，体积较大；GitHub CI 使用 `.github/requirements-ci.txt` 运行不加载模型的测试子集，不能代替 GPU 与真实模型验收。

修改依赖后，应更新并检查对应 lockfile：

```powershell
uv lock
uv lock --project environments/cosyvoice
uv lock --project environments/qwen3tts

uv lock --check
uv lock --check --project environments/cosyvoice
uv lock --check --project environments/qwen3tts
```

WebUI 静态资源依赖位于 `dubbing_service/webui/package-lock.json`，更新时使用 `npm ci` 验证可复现安装。

## 分支与提交

- 从最新默认分支创建短期分支，建议使用 `fix/`、`feature/`、`docs/` 或 `chore/` 前缀。
- 一个 Pull Request 只解决一个清晰问题；避免混入格式化、模型文件或无关重构。
- 提交信息使用祈使语气，说明结果，例如 `Fix take selection persistence`。
- 保留用户数据的向后兼容性；若必须迁移数据，应提供可恢复路径和迁移测试。

## 提交前验证

至少运行与改动相关的测试，并执行：

```powershell
.\scripts\export_source.ps1 -ValidateOnly
.\scripts\ci_source_guard.ps1
uv lock --check
```

在按 `docs/installation.md` 引导完成的根环境中，运行完整测试套件（不要加 `--ignore`）：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

`.github/requirements-ci.txt` 只覆盖无模型子集（不含 `torch` 与 `uvicorn`），所以 CI 仍会跳过 `tests/test_indextts_audio_safety.py` 与 `tests/test_service_server_api.py`；本地根环境具备 CUDA/PyTorch 依赖，应运行全部测试文件。涉及 IndexTTS2 推理或完整服务入口时，还应用项目目录内、许可合规的模型完成真实音频冒烟测试。涉及 CosyVoice3 或 Qwen3-TTS 时，必须在各自 uv 环境中单独验收，不能把依赖合并进根环境。

Pull Request 应写明已运行和未运行的检查。通过测试不表示生成内容已达到发布质量；听感、响度、爆音、截断、字幕同步和商业授权仍需人工验收。

## 许可与来源

提交贡献即表示你有权提供该内容，并同意按仓库中适用于相应文件的许可条款分发。仓库存在按组件划分的许可边界；修改前请阅读根许可文件、`LICENSES/README.md`、`NOTICE.md` 和 `THIRD_PARTY_NOTICES.md`。新增第三方代码、模型、音色、数据、音乐或音效时，必须记录原始来源、版本或 revision、许可、修改情况和再分发限制。
