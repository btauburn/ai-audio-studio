# 依赖安全与兼容风险

本文记录受支持运行环境中暂时保留的依赖安全告警、实际可达性、缓解措施和退出条件。它不是“忽略告警”清单；真实告警继续保留在 GitHub Dependabot 中，直到完成兼容升级或移除相关依赖。

## 当前快照

截至 2026-08-12，受支持清单共有 62 条开放 Dependabot 记录：0 Critical、13 High、31 Medium、18 Low。相同漏洞会同时出现在 `pyproject.toml` 与 `uv.lock`，按“包 + GHSA”去重后为 20 项。

受支持的安装清单只有：

- 根 `pyproject.toml` 与 `uv.lock`：平台、IndexTTS2 和字幕；
- `environments/cosyvoice/pyproject.toml` 与 `uv.lock`：CosyVoice3；
- `environments/qwen3tts/pyproject.toml` 与 `uv.lock`：两个 Qwen3-TTS Worker；
- `environments/stableaudio/pyproject.toml` 与 `uv.lock`：Stable Audio 3 Small-SFX Worker；
- `dubbing_service/webui/package.json` 与 `package-lock.json`：WebUI 构建工具；
- `.github/requirements-ci.txt`：无模型 CI 的最小依赖镜像。

`cosyvoice/requirements.txt`、训练示例、历史归档和 `service/requirements.txt` 不是本项目的安装入口。来自这些未使用清单的告警会以 `not_used` 处置，并保留具体理由；不会用这种理由关闭受支持清单中的真实告警。

## 暂时保留的风险

| 依赖 | 当前兼容边界 | 告警与实际可达性 | 当前缓解措施 | 退出条件 |
| --- | --- | --- | --- | --- |
| Protobuf `3.19.6`（根环境） | `descript-audiotools 0.7.2` 要求 `<3.20`，IndexTTS2 的 DAC 路径需要该依赖 | 2 项 High，涉及不受信任 Protobuf JSON 的递归/拒绝服务；平台 HTTP API 不接收 Protobuf JSON | 只处理平台内部固定数据和已校验模型，不把 Protobuf 解析器暴露为网络输入 | 升级或替换 DAC 依赖后，完成 IndexTTS2 合成与音频回读回归 |
| PyTorch Lightning `2.6.0`（根环境传递依赖） | 由 `WhisperX → pyannote-audio → lightning` 引入，上游暂无修复版本 | 1 项 High，涉及 `load_from_checkpoint` 反序列化；受支持字幕流程不接受用户指定 checkpoint | 仅使用固定的本地字幕/对齐模型；禁止加载任意 checkpoint、训练配置或序列化对象 | 上游提供修复，或从字幕推理链移除该加载器后完成字幕 E2E |
| Transformers `4.52.4` / `4.51.3` / `4.57.3` / `5.15.0` | IndexTTS2、CosyVoice3、Qwen3-TTS 与 Stable Audio 分别依赖不兼容 API；部分 RCE 修复要求新版 | 告警集中在 Trainer、LightGlue、Donut、Marian、AdamWeightDecay 等未使用路径，以及恶意模型初始化 | 只允许 manifest 固定 revision、大小和 SHA256 的项目内模型；Worker 使用本地路径和离线模式；不接受远程仓库、`trust_remote_code` 或任意模型目录 | 为每个后端分别升级并通过模型加载、全部合成/生成模式、卸载重载和音频回读矩阵 |
| Torch `2.8.x` / `2.6.0` / `2.10.0` / `2.7.1` | 四个环境的 CUDA/模型算子兼容要求不同；CosyVoice3 已从存在 Critical 的 2.3.1 升至实机验证的 2.6.0 | 剩余告警涉及 packed sequence、LSTM cell、量化 Sigmoid、profiler/JIT 等；受支持 Worker 推理不调用这些路径，`torch.jit.script` 只见于离线导出工具 | 不暴露训练/导出工具；固定本地模型；按环境隔离 Torch/CUDA；CosyVoice3 三模式与 Stable Audio 技术门均已完成真实 GPU 验证 | 按环境单独升级，不跨后端打包；通过 CUDA、ONNX、模型加载、合成/生成和反复卸载回归 |

## 操作约束

- 默认只绑定 `127.0.0.1`，不要把平台直接暴露到互联网；
- 不支持任意模型目录、远程 Transformers 仓库、训练配置或 checkpoint；
- 模型只能通过 `scripts/install_models.ps1` 从 manifest 固定、显式授权的 ModelScope 或门控 Hugging Face revision 恢复，并逐文件校验大小和 SHA256；
- 依赖变更必须先通过 lock、源码边界、离线测试，再对受影响后端执行真实 GPU E2E；
- 不为了让扫描数字归零而合并跨环境的大批升级，也不把真实运行时告警错误标记为“未使用”。

Dependabot 数量会随上游公告变化，以仓库 Security 页面为实时来源。每次预发布都应重新审查 High/Critical 告警，并更新本文件中的兼容边界与退出条件。
