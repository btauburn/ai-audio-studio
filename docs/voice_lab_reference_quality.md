# Active local reference quality checks — 2026-10-02

Voice Lab now offers **Check reference locally** beside recorded references. It runs installed CPU tools, returns pass/review/fail for supported technical checks, caches exact byte/transcript/pipeline-bound evidence, and leaves voice selection and approval separate.

The phone displays the Attenborough source transcript as user-confirmed supplied text, not ASR or audition acceptance. The 20.88-second synthetic audition remains Ready. Current narrator, other characters, ratings and project bytes are unchanged.

## Supported checks

| Check | Rule |
|---|---|
| Decode | Finite mono 3–30-second crop, 8–192kHz, bounded 24MB input. Invalid audio fails before inference. |
| Signal | RMS below .005 or more than 65% near-silent samples fails. |
| Clipping | Peak above 1 or more than .1% clipped samples fails. |
| Speech | CPU Silero VAD: at least 1.5 seconds and 25% speech. VAD does not count speakers. |
| Background | Speech/non-speech RMS ratio below 8dB needs review when at least .3 seconds of non-speech exists. This is a heuristic, not measured SNR/music detection. |
| ASR | Existing English faster-whisper base.en, CPU int8/two threads. Mean word confidence below .85, more than 10% words below .5, low segment confidence or high no-speech probability needs review. |
| Transcript | Normalized WER ≤.1 passes, >.1 and <.5 needs review, ≥.5 fails. Unchanged machine drafts cannot supply independent alignment evidence. Unsaved/later-generated replacements remain unverified; saved hash-bound corrections are separate targets, not verified truth. |
| Crop coverage | Bounds-checked ASR word times and VAD coverage; uncovered edge speech above .65s or word/VAD overlap below .6 needs review. Estimated timestamps are not forced alignment or proof that every phoneme is complete. |

**Music and overlap detection are unavailable:** no installed validated music classifier, overlap detector or diarizer exists here. Those checks are never marked pass. Identity, consent, licensing and endorsement remain unassessed. A technical pass covers only supported checks; it cannot unlock cloning or approve a synthetic voice. Existing source eligibility is unchanged.

## Binding and UI

The API accepts a fixed acquisition ID/receipt SHA and verifies the current reference bytes. Quality artifacts are separate from immutable receipts/audio and never change cast or ratings. Cache keys bind exact WAV bytes, stripped expected text and its provenance, language, model hashes, settings, wrapper source and package versions. GET rechecks current audio before showing cached evidence. Tampering fails; changed text/pipeline becomes stale. Frozen loaded source/dependency guards require restart after disk changes, including SciPy.

The UI uses the existing authenticated private request/origin guard. Only the check button is disabled during CPU work; text/audio remain usable. Concurrent edits discard stale results; ASR replacement invalidates a result tied to old text. Correction saves distinguish their submitted snapshot from newer unsaved edits.

No packages/models were installed, no audio uploaded and no TTS generated for these checks. The installed ASR model is English only; unsupported language/tool failures report review/unavailable.

## Evidence and limits

The frozen BBC crop SHA256 is 19764be5f553466f8aa2ce7d0aacfdeb65bda3651cd275203c42bff779497f82. ASR matched all 17 supplied words (WER 0). Calibration returned pass; the active phone check returned review because two words were below .5 confidence. This boundary is preserved rather than tuned to force a favorable grade. Source confirmation and the audition remain unchanged.

Deliberately wrong words, silence, hard clipping and added white noise failed. Two-tone contamination needed review; an offset same-speaker mixture failed on transcript/confidence evidence. These surrogates do not validate music or overlap detection. Thresholds have not been population-calibrated; phoneme completeness and source isolation are not certified.

Final verification: **32 focused Python tests, 159 UI tests and Ruff F/E9 passed**. Before the final narrow provenance/dependency guards, **244 applicable Python regressions** passed. Independent review closed with no remaining blocking finding. Actual 390×844 browser checking and 9.3-second reference/20.88-second audition playback passed. Browser POST testing supplied the configured legitimate forwarded phone origin only for the tested POST; application guards were unchanged.

Activation checked canonical queues were idle and verified current process ownership before restarting only Studio9100, private8779 and managed Qwen8003. Unrelated9000/8778 remained untouched. Faster Base profile/worker contract stayed unchanged; no default promotion, generation, rating, feedback or notification was dispatched.

Rollback is a scoped removal/revert of the checker module/API/UI and its separate caches, followed by an idle-queue restart of the same owners. Preserve acquisition receipts, corrections, audio, prompts, identities and the existing renderer. No broad reset, mirror push, publication or unrelated edit cleanup occurred. Integration shares files with pre-existing uncommitted work; no broad commit was made.

Primary tooling references: [faster-whisper VAD](https://github.com/SYSTRAN/faster-whisper#vad-filter) and [Silero VAD](https://github.com/snakers4/silero-vad).
