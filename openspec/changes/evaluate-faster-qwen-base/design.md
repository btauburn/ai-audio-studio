# Design

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


`qwen3tts` remains the backend. An explicit per-launch `faster-base-032` profile selects an isolated environment and server wrapper. Prepared auditions and production use the same `CharacterVoiceRuntime.synthesize` Base adapter; VoiceDesign reference creation retains its current implementation. Runtime provenance belongs in generated Take metadata, not saved voice identity. The native launch remains the default.

The adapter verifies exact dependency versions, source hash, CUDA/BF16 requirements and the approved absolute snapshot. Model manifest verification precedes any unload/load. Conditioning uses immutable content hashes and precomputed prompts, with no extra reference silence. Predictor sampling is pinned before graph capture and each request reseeds after warmup. A single inference lease spans preparation, model load, output generation and metadata attribution. Cancellation events are request-owned and atomic with completion. Unsupported controls, sequence-cap/budget exhaustion, OOM and interruption cannot produce silent fallback or successful partial audio.

Static sequence size 2048 is not a longform guarantee. Prefill plus the production token budget must fit; emitted EOS is necessary but cannot prove transcript coverage. Segmented generation follows frozen production chunking. Performance captures wall/audio seconds, RTF and inverse, cold/load/warmup, CUDA memory, GPU utilization, process/system RAM, raw hashes, errors/retries and explicit fallback. Speaker cosine and signal diagnostics flag listening work, never substitute for it.

The comparison bundle is a fixed local output location bound to one authorized review project. Each audio request validates pair/lane, path containment and SHA256 before existing ranged delivery. The private projection adds only bounded GET/HEAD routes and the external script asset. The advanced panel reads pre-generated audio; it does not change canonical voices or trigger GPU work. A single comparison player coordinates native controls, ordinary playback, shortcuts and polling.

No independent service, database, download daemon or external architecture publication is introduced. Scoped source snapshots preserve pre-existing dirty changes. All GPU experiments restore the native worker. Production promotion remains a separate explicit decision after human review and outstanding completeness/longform gates pass.


## Runtime-bound jobs and cached takes

The worker exposes `/api/v1/tts/runtime-identity` without loading weights. Its process-frozen digest covers the approved absolute Base checkpoint/manifest, package versions, implementation sources, dtype/attention, sampling and seed policy, token budget and disabled fallback. Changes to those inputs on disk require restart. Requests send the expected worker digest; a mismatch returns 409 under the inference lease before loading or synthesis. Success metadata must return that same digest and no fallback.

The renderer adds its implementation/chunk policy to that contract. Jobs persist per-backend runtime bindings before dispatch across their complete planned scope; each chunk checks the worker again. Take fingerprints include actual runtime provenance. Resume checks references, role aliases, selected takes and settings before using cached output. Missing worker endpoints, mismatched provenance, and legacy completed Qwen audio without a binding fail clearly. An operator must explicitly start a new whole rendition to change runtime; completed historical audio is preserved. Cancellation, OOM or an uncertain bound-worker failure does not trigger automatic reassembly/retry or runtime switching. CustomVoice remains separate; its checkpoint is present and CPU contracts were checked, but no new CustomVoice GPU sample was generated.

This closes the recoverable resume gap: a job cannot silently append Faster chunks to a native rendition or the reverse. Actual wrong-digest HTTP returned 409 without model load/audio. Existing completed jobs keep their recorded status without requiring a live worker lookup. Legacy audio is retained but cannot be silently reused as runtime-bound output.
