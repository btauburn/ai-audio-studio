# Evaluate Faster Qwen Base without changing saved voices

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


## Why
The owner requests substantially faster generation for the same saved Qwen voices and the same inference path in auditions and production. Saved voice identity and preferences must survive independently of runtime selection.

## What Changes
- Add an isolated, explicitly selected Faster 0.3.2 Base CUDA graph profile to the existing Qwen character API, pinned to current checkpoint/dependencies/settings.
- Freeze matched workloads and preserve baseline/candidate raw evidence, safety tests and diagnostics.
- Add a collapsed read-only comparison panel and authorized ranged audio to the existing private Voice Lab projection.
- Keep native production default and require human identity/naturalness/expressiveness/completeness review before promotion.

## Impact and discovery
The existing Studio backend remains owner; no new queue or canonical voice store. No checkpoint, reference, cast, rating/favorite, credential or hosting migration. The parent completed GitHub-first HomeOS discovery: latest immutable projection `homeos-efa7c1c0727f-af753cabe44f` contained no Audio/VoiceLab/AssetForge surface and owner searches found none. Local implementation is the verified ownership source. This proposal records the projection mismatch; no architecture mirror was pushed.
