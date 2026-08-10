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

## 报告问题

安全问题不要附带真实密钥、私人音频或作品内容。公开仓库可以只提供最小复现、受影响版本和脱敏日志；需要传递敏感证据时，应先与维护者建立私密沟通渠道。

## 内容与声音安全

只有在获得说话人和素材权利人授权后，才能克隆、生成、编辑或发布相关声音。发布者自行负责适用的版权、人格权、隐私、AI 合成内容标识和平台规则合规。
