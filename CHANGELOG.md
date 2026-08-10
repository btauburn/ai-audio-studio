# Changelog

Copyright 2026 TadashiJLU.

本文件记录面向使用者和贡献者的重要变化。项目处于 beta 阶段，正式版本发布前可以调整内部接口；涉及项目数据的变更仍应提供兼容或迁移路径。

格式参考 Keep a Changelog，版本号遵循语义化版本。

## [Unreleased]

## [0.1.0-beta.1] - 2026-08-10

### Added

- Windows-first AI Audio Studio workflow covering script segmentation, roles,
  four TTS backends, immutable Takes, independent A/B slots, non-destructive
  edits, review, four-track sound design, Masters, subtitles, and release
  packages.
- Reproducible three-environment uv bootstrap and manifest-pinned ModelScope
  model provisioning with size/SHA256 verification.
- Original project logo, launcher icon, WebUI favicon, screenshots, bilingual
  repository landing pages, and a 1280 × 640 social preview.
- Apache-2.0 licensing for the TadashiJLU-authored platform layer, with an
  explicit path-level map for upstream and third-party licenses.
- Windows 无模型 CI：检查三套 uv lockfile、源码发布边界、潜在密钥、大文件、JavaScript 语法、致命 Python 静态错误和离线测试。
- Dependabot、Issue 表单、Pull Request 模板、贡献指南、支持指南、行为准则和代码所有者元数据。

### Fixed

- 保留每段 A/B 独立版本和不可变 Take，避免切换或重新合成时静默丢失另一版本。
- 将音频编辑改为编辑配方和“渲染为新 Take”，不再覆盖原始 Take。
- 兼容旧工程的无时区时间戳与新工程 UTC 时间戳，避免项目列表刷新后返回 500。
- 修复 Windows 启动器编码/路径问题，并统一运行时、模型目录和离线缓存策略。

### Security

- 在自动化检查中阻止本机 Provider 配置、项目数据、项目目录内模型权重和高置信度凭据进入源码提交。

<!-- 发布版本时，把 Unreleased 内容移动到带日期的版本标题，并补充比较链接。 -->

[Unreleased]: https://github.com/tadashijlu/ai-audio-studio/compare/v0.1.0-beta.1...HEAD
[0.1.0-beta.1]: https://github.com/tadashijlu/ai-audio-studio/releases/tag/v0.1.0-beta.1
