"""Provider abstractions for external services.

当前包含：
- TTS Provider：封装对 IndexTTS2 REST 服务的调用
- LLM Provider：封装对兼容 OpenAI API 协议的大模型的调用

后续所有与 TTS/LLM 交互的逻辑都应通过本目录中的 Provider 完成，
避免在业务路由中直接拼接 HTTP 请求。
"""
