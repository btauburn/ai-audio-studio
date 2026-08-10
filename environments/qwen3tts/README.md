# Qwen3-TTS runtime

This isolated uv project serves both local models:

- `models/qwen3_tts/Qwen3-TTS-CustomVoice`
- `models/qwen3_tts/Qwen3-TTS-VoiceDesign`

Create or repair it only through the repository bootstrap script. Normal platform
startup calls this environment's interpreter directly and never performs an
implicit dependency sync.
