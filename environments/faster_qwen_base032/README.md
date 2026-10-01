# Experimental Faster Base environment

The contained `.venv` has faster-qwen3-tts 0.3.2 installed from the verified local registry wheel with `--no-deps --no-index`. Its `.pth` reuses existing Qwen dependencies read-only. Do not install qwen-tts-hf into either environment. Keep qwen-tts 0.1.1 / Transformers 4.57.3 / Torch 2.10.0+cu130 pinned. Adapter startup verifies exact versions and Faster source SHA256.

No `setup_windows.bat`, checkpoint downloads, optional FlashAttention/Triton/compile dependencies or live-environment updates were used. See `docs/faster-qwen-base-experiment.md` for operation, limits and rollback. The profile remains experimental and launch-only.
