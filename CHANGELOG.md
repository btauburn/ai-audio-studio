# Changelog

Copyright 2026 TadashiJLU.

本文件记录面向使用者和贡献者的重要变化。项目处于 beta 阶段，正式版本发布前可以调整内部接口；涉及项目数据的变更仍应提供兼容或迁移路径。

格式参考 Keep a Changelog，版本号遵循语义化版本。

## [Unreleased]

## [0.2.0-beta.1] - 2026-08-12

### Added

- Stable Audio 3 Small-SFX 本地音效与环境声工作流：LLM 编导计划、人工批准、
  A/B 候选、不可变生成结果、采用/解除采用和四轨 Preview。
- 句段 `start|end + offset` 语义锚点；更换 Take、变速或对白交叉淡化后，
  SFX/Ambience 会按最新对白时间轴重新定位，文本变化则标记为 stale。
- 独立 Python 3.11 / CUDA 运行时、固定模型与上游代码 revision 的门控供应、
  Community/Enterprise 许可回执、生成 provenance 和真实 CUDA 验收脚本。
- 持久化 SoundPlan 与生成任务，支持断线继续、取消、崩溃恢复、候选清理和
  幂等采用；旧项目与旧绝对时间片段无需迁移。

### Changed

- Preview、Master 与 Release 统一使用相同的语义锚点解析和四轨混音入口；
  Ambience 循环默认使用 1 秒交叉淡化。
- TTS GPU lease 泛化为生成音频互斥通道；Stable Audio 批次与 TTS/Whisper
  严格互斥，运行时离线加载，异常释放失败时终止 worker 回收 CUDA 上下文。
- Master 和发布包冻结声音设计、素材授权与生成 provenance；正式制作会对
  缺失、过期或不一致的声音/素材/模型许可 fail closed。

### Fixed

- 修复多段 `full-with-edits` 在第二段访问未定义 `dialogue_crossfade` 的问题，
  并补充交叉淡化、变速、换 Take 和语义重定位回归测试。
- 加固生成任务双写、取消、恢复、候选提升与采用/解除采用的补偿事务，避免
  孤儿 WAV、永久排队状态和 SoundPlan/SoundDesign 漂移。
- 拒绝持久化 ID、符号链接和 Junction 导致的任务、计划、候选或正式素材
  越界读写；GET 接口不再隐式写入计划状态。

### Security

- Stable Audio worker 仅监听 loopback，health/load/unload/generate 全部要求
  每实例随机令牌；令牌只经进程环境传递，不写入项目、日志或 provenance。
- 模型文件按固定 revision、大小和 SHA256 原子供应，运行时同时校验隔离环境
  中 Stable Audio 上游 commit，拒绝联网回退或被污染的运行时。
- Preview 对过期语义/许可给出警告；Master 与 Release 对实际使用和冻结快照
  中的声音、上传素材及生成素材许可缺失或过期一律阻断。

### Validation

- Windows / RTX 4070 Ti SUPER 上完成 12/12 模型文件 SHA256 校验、24 个质量
  候选生成、30/30 技术运行、5/5 次 TTS→SFX→TTS 显存交接和 warm p95 门。
- 人工语义与制作可用性评分尚未完成；`quality_gate.json` 因此仍缺失，产品会
  按设计保持 Stable Audio 生成功能关闭，不能把技术通过视为质量通过。

## [0.1.0-beta.1] - 2026-08-11

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
- 将 CTranslate2 升至 4.8.1，兼容已移除 `pkg_resources` 的新版
  Setuptools，同时保持 Faster-Whisper / WhisperX 字幕链路可用。
- 修复 LLM Provider 对 DeepSeek 域名的子串判断，拒绝伪装域名、URL
  内嵌凭据、非法端口和控制字符。
- 将两处可被超长文本放大的正则替换为线性字符串处理。

### Security

- 在自动化检查中阻止本机 Provider 配置、项目数据、项目目录内模型权重和高置信度凭据进入源码提交。
- TTS Worker 的参考音频、下载和删除操作统一限制在管理员配置的音频根目录，
  并校验扩展名、存在性及符号链接/Junction 最终边界。
- Dependabot 只为可独立解析的根环境、Qwen3-TTS、CI 与 Actions 生成补丁升级，
  排除 vendored、历史和旧式重复清单；Windows CUDA 专用的 CosyVoice3 与预编译
  WebUI 资产采用人工升级。真实运行时告警继续公开，并记录可达性、缓解措施和退出条件。
- CosyVoice3 升级到 PyTorch 2.6.0 + CUDA 12.4，修复旧版
  `torch.load` 关键漏洞；生产环境移除 Lightning、Diffusers、ONNX 导出、
  PyArrow 等训练/工具依赖，并将 ModelScope/Protobuf 固定到已修复版本。
- Click、Gradio、aiohttp、cryptography、PostCSS 与 Nano ID 更新至补丁版本；
  模型继续采用固定 revision、SHA256 校验和离线加载。

<!-- 发布版本时，把 Unreleased 内容移动到带日期的版本标题，并补充比较链接。 -->

[Unreleased]: https://github.com/tadashijlu/ai-audio-studio/compare/v0.2.0-beta.1...HEAD
[0.2.0-beta.1]: https://github.com/tadashijlu/ai-audio-studio/compare/v0.1.0-beta.1...v0.2.0-beta.1
[0.1.0-beta.1]: https://github.com/tadashijlu/ai-audio-studio/releases/tag/v0.1.0-beta.1
