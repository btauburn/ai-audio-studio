# 第三方源码基线

本项目把经过 Windows/离线适配的 CosyVoice3 源码作为 vendored source 随项目维护，而不是把它伪装成一个未登记的根级 Git 子模块。这样，根仓库和白名单源码导出都能完整收录实际运行代码。

## CosyVoice

- 上游：`https://github.com/FunAudioLLM/CosyVoice.git`
- 本轮整理前的上游提交：`c93d3dda01ae69fde8a4b2372f0e4260135599c3`
- 已知本地适配文件：`cosyvoice/cosyvoice/cli/frontend.py`、`cosyvoice/cosyvoice/cli/cosyvoice.py`、`cosyvoice/cosyvoice/dataset/processor.py`、`cosyvoice/vllm_example.py`、`cosyvoice/webui.py`、`cosyvoice/cosyvoice/utils/wetext_utils.py`、`cosyvoice/third_party/Matcha-TTS/matcha/models/components/flow_matching.py`、`cosyvoice/third_party/Matcha-TTS/matcha/utils/__init__.py`、`cosyvoice/third_party/Matcha-TTS/matcha/hifigan/xutils.py`。这些文件保留各自的 Apache-2.0 或 MIT 许可，并带有 TadashiJLU 修改声明；适配内容包括 Windows/WeText、离线模型策略和推理/训练依赖解耦。
- 2026-07-11 已移除 `cosyvoice/.git/` 嵌套元数据；源文件和本地改动保持不变，由项目根仓库统一管理。

可逆历史备份位于：

- 私密恢复备份中的 `git-history/cosyvoice-c93d3dda.bundle`
- 私密恢复备份中的 `git-history/cosyvoice-local-changes.patch`
- 私密恢复备份中的 `git-history/cosyvoice-untracked-wetext_utils.py`

## Matcha-TTS

- 上游：`https://github.com/shivammehta25/Matcha-TTS.git`
- CosyVoice 工作树使用的提交：`dd9105b34bf2be2230f4aa1e4769fb586a3c824e`
- 2026-07-11 已移除 `cosyvoice/third_party/Matcha-TTS/.git` 控制文件，使实际工作树与 CosyVoice 一起作为 vendored source 管理。
- 完整历史备份：私密恢复备份中的 `git-history/matcha-tts-dd9105b3.bundle`

## 升级规则

1. 在项目外临时 clone 对应上游并固定目标提交，不要直接在生产工作树内 `git pull`。
2. 重新应用并审查本地 Windows、WeText 和严格离线适配。
3. 运行三环境依赖检查、doctor、全量测试和 CosyVoice3 真实 E2E。
4. 替换 vendored source 后更新本文的提交号、许可证/NOTICE 审计和恢复报告。

vendoring 只解决源码可复现性，不改变任何上游许可证或 NOTICE 义务。
