# CosyVoice3 runtime

This uv project isolates CosyVoice3 from the IndexTTS2/platform environment.
It deliberately uses the Windows-compatible `wetext` path; the bundled ttsfrd
wheels are Linux-only and are not runtime dependencies on Windows.

The engine source remains pinned under `cosyvoice/`, and the model lives at
`models/cosyvoice3/Fun-CosyVoice3-0.5B`.
