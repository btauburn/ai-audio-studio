# Evaluate Faster Qwen Base without changing saved voices

## Why
The owner requests substantially faster generation for the same saved Qwen voices and the same inference path in auditions and production. Saved voice identity and preferences must survive independently of runtime selection.

## What Changes
- Add an isolated, explicitly selected Faster 0.3.2 Base CUDA graph profile to the existing Qwen character API, pinned to current checkpoint/dependencies/settings.
- Freeze matched workloads and preserve baseline/candidate raw evidence, safety tests and diagnostics.
- Add a collapsed read-only comparison panel and authorized ranged audio to the existing private Voice Lab projection.
- Keep native production default and require human identity/naturalness/expressiveness/completeness review before promotion.

## Impact and discovery
The existing Studio backend remains owner; no new queue or canonical voice store. No checkpoint, reference, cast, rating/favorite, credential or hosting migration. The parent completed GitHub-first HomeOS discovery: latest immutable projection `homeos-efa7c1c0727f-af753cabe44f` contained no Audio/VoiceLab/AssetForge surface and owner searches found none. Local implementation is the verified ownership source. This proposal records the projection mismatch; no architecture mirror was pushed.
