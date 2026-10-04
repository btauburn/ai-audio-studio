# Experimental Faster Base environment

> Current decision, 2026-10-04: **USER-APPROVED DEFAULT** is the implemented
> `faster-base-032` Base cloning/character profile. This supersedes the earlier
> native-current/auto preference and blind-listening promotion gate. CustomVoice
> remains a separate lane. Explicit runtime choices and immutable job/Take bindings
> are preserved; mismatched historical resume must fail before synthesis. User
> approval does not assert universal audio parity or that every sample was heard.
> The October 1-2 experimental/default/pending-approval statements below describe
> historical state. Frozen A/B audio, ASR flags, word endings, chunk boundaries,
> voice identity and longform listening remain useful quality evidence.
> See `docs/audiobook-current-checkpoint.md` for the current implementation receipt.


The contained `.venv` has faster-qwen3-tts 0.3.2 installed from the verified local registry wheel with `--no-deps --no-index`. Its `.pth` reuses existing Qwen dependencies read-only. Do not install qwen-tts-hf into either environment. Keep qwen-tts 0.1.1 / Transformers 4.57.3 / Torch 2.10.0+cu130 pinned. Adapter startup verifies exact versions and Faster source SHA256.

No `setup_windows.bat`, checkpoint downloads, optional FlashAttention/Triton/compile dependencies or live-environment updates were used. See `docs/faster-qwen-base-experiment.md` for operation, limits and rollback. The profile remains experimental and launch-only.
