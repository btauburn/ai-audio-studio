# 安全策略

## 支持边界

当前版本面向个人 Windows 工作站和受信任局域网，不是公网 SaaS。平台没有账户认证、租户隔离、TLS 终止或公网防护。

`start_platform.bat` 默认仅监听 `127.0.0.1`。也可手工运行：

```powershell
.\.venv\Scripts\python.exe -u -m uvicorn dubbing_service.app.main:app --host 127.0.0.1 --port 9000
```

只有显式运行 `start_platform.bat --lan` 才会进入局域网模式；此时必须通过 Windows 防火墙限制来源。不要配置公网端口映射。

## 不应提交或公开的内容

- API Key、访问令牌、`.env` 和本机 Provider 配置；
- 作品原稿、项目 JSON、参考声音、录音和角色音色；
- 模型权重、模型缓存和虚拟环境；
- BGM、Ambience、SFX、封面及其商业授权凭证；
- 生成音频、字幕中间文件、日志和备份。

提交前运行：

```powershell
.\scripts\export_source.ps1 -ValidateOnly
git status --short
```

不要使用 `git add -f` 绕过忽略规则。若密钥曾进入提交，即使随后删除，也应立即在服务商后台撤销并重新生成，然后清理 Git 历史。

## 模型与依赖供应链

- 只使用 `models/manifests/` 中固定 revision、大小与 SHA256 的模型，并通过
  `scripts/install_models.ps1` 恢复；
- TTS Worker 只接受项目内已批准的模型目录和托管音频根目录，运行期按离线模式
  加载；
- 不要加载来源不明的 checkpoint、Transformers 仓库、训练配置或序列化对象，
  即使底层框架本身提供相应入口；
- `cosyvoice/requirements.txt` 和训练示例清单属于 vendored 上游参考，不是本项目的
  安装入口；受支持环境以根 `uv.lock` 和 `environments/*/uv.lock` 为准；
- 模型栈的依赖升级必须经过真实 GPU 模型加载、合成和音频回读验收，不能仅为消除
  扫描提示而跳过兼容性验证。

当前受支持环境中暂时保留的 Torch、Transformers、Protobuf 和 PyTorch Lightning
告警，以及对应的可达性、缓解措施和退出条件，记录在
[依赖安全与兼容风险](docs/dependency_security.md)。这些真实告警继续保留在 GitHub
Dependabot 中，不以“未使用”为由隐藏。

## 报告漏洞

请使用 GitHub 的[私密漏洞报告](https://github.com/tadashijlu/ai-audio-studio/security/advisories/new)，提供受影响版本、风险、最小复现和建议修复方式。不要先创建公开 Issue，也不要附带真实密钥、私人音频或作品内容。

如果该入口暂时不可用，可以创建一个**不包含漏洞细节**的普通 Issue，仅请求维护者建立私密沟通渠道。维护者确认安全渠道后再发送脱敏证据。

当前 beta 版本尽力为仍受支持的最新版本提供安全修复；已经淘汰的实验分支或历史归档文档不单独维护。

## 内容与声音安全

只有在获得说话人和素材权利人授权后，才能克隆、生成、编辑或发布相关声音。发布者自行负责适用的版权、人格权、隐私、AI 合成内容标识和平台规则合规。
