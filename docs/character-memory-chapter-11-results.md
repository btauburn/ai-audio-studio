# Character memory and chapter 11 — verified review edition

The next improvement carries source-grounded character delivery habits between
chapters. Earlier V2 requests contained the local scene without this book-scoped
memory. The new optional memory is separate from voice identity and temporary
emotion: scene intention and explicit author cues override a habit. Narration
receives no character direction. **Acting improvement is not established; human
listening and production approval remain pending.**

The implementation was committed and pushed **before synthesis** in
[`31d73ae`](https://github.com/btauburn/ai-audio-studio/commit/31d73ae3a8388da5de1a2e32b53dddc262aead31),
on `feature/casting-performance-director-20261011` in Brian's Studio fork. No
upstream merge or default-model change was made.

## Implemented owners and actual delegation

| Responsibility | Implementation / contribution |
| --- | --- |
| Character memory | Actual Sonnet 5.5 engineering session implemented `dubbing_service/app/audiobook/character_performance_memory.py`, optional `coached_chapter_batch.py` enrichment, tests and documentation. Pure capture/validation/enrichment; no inference inside the module. |
| Fresh chapter review | A second actual Sonnet 5.5 engineering session implemented `fresh_coached_chapter.py`, `scripts/render_fresh_coached_chapter.py`, tests and documentation. A serial follow-up integrated original source/provider contexts and 60 additional fixture cases. |
| Integration and delivery | Codex root prepared the exact chapter and returning cast, captured and validated model data, ran independent tests/review, committed, executed one GPU worker, verified artifacts/retry, and delivered private review audio. |

The two original engineering workers had disjoint write scopes and terminal
success receipts. Sonnet 5.5 also performed the source attribution, one book-memory
capture over prior chapters4–10, and one whole-chapter scene-coaching call for
chapter11. Haiku was not needed. There are no per-sentence model calls.

See [memory contracts](character-performance-memory.md) and
[fresh review source/provider contracts](fresh-coached-chapters.md). Existing
batch rendering still requires and reuses selected neutral Takes; the new fresh
path synthesizes every speech line for review without manufacturing a Take.

## Actually executed and measured

| Check | Actual evidence |
| --- | --- |
| Source | Exact registered EPUB chapter11; 303 ordered source rows, 302 speech segments, one canonical separator and zero exclusions. Complete nonblank prose and source anchors verified. |
| Cast | Six returning characters and one narrator retain five recorded clone identities. Original source-use permission and draft selection are rechecked in two original provider contexts. No target approval, selection, human event, reference promotion or Take is created. |
| Memory | 22 source-grounded, unapproved profiles captured from prior chapter requests; book/role/evidence/provenance hashes validated. All six chapter11 dialogue roles covered. |
| Direction | 103 dialogue requests prove native instruction forwarding; 199 narrator clips are fresh neutral synthesis with zero instructions. Same approved Base checkpoint through the private `faster-base-032-direction-x1` wrapper; no fallback or global model switch. |
| Audio | One complete private WAV/MP3 chapter: 1197.770s (19.96min); integrated loudness -25.60LUFS. All blocking sample/chapter QA and exact audio hashes passed. |
| Timing | 301 measured boundaries; 121.050s additional padding. 11 boundaries need no added silence. Existing leading/trailing silence is credited, the source separator has a1.25s target, and speech is never trimmed. |
| Generation | Accepted sample receipts sum to 580.039s request latency for 1076.720s generated audio. Sampled total-device VRAM peaks at 8350MiB on the RTX4070. These are receipt totals, not wall time or a controlled performance comparison. |
| Completed retry | Actual CLI retry: zero requests, zero workers, 1347 unchanged render artifacts. Source reproof records are excluded from the immutable count. |
| Preservation | All16 unrelated baseline files and the global voice bank unchanged. Original review runtime restored with its exact prior identity; production9141 and earlier private listening routes preserved. |
| Delivery | Brian-only paused player; Range206, hash-matching MP3 download, absent/wrong identity403. One authorized Pushover submission accepted; phone display unconfirmed. |

Supporting offline CPU Whisper Base.en diagnostics cover **all302** clips with
zero model downloads: 243 edits over 3272 source words
(7.43% ASR disagreement), 5 possible omissions and
74 tail-token disagreements. Fantasy names, spelling, punctuation and
interjections limit this measurement. These are review flags, not confirmed
audible omissions, clipped endings or acting judgments. No paired chapter11
baseline was rendered, so no comparative quality gain is claimed. The earlier
20-line baseline/v1/v2 pilot remains available separately.

Seven exact evidence offsets in the scene response were corrected to unique
source substrings; one delivery instruction was rephrased to avoid the existing
conservative identity-word guard. One book-memory evidence pointer was corrected
using a unique exact own-role quote. Raw responses and all normalizations are
retained privately; no identity guard or source prose was changed.

## Tests and review

Codex root independently ran **791 checks**:199 memory/batch/coaching,
142 fresh-chapter cases (82 original plus60 provider-context cases), and450
scene/timing/evaluation/export regression cases. All passed. Fixtures use invented
prose, synthetic tone audio, fake voices/workers and explicit fixture labels;
they cannot approve the real audio. Default structured Codex `gpt-5.6-sol`/`max`
review covered two complete bundle passes and returned no actionable findings
at its configured P0 threshold; changed-content TruffleHog was clean.

## Launch and verify on Brian's machine

The prepared private source, memory, coaching, raw samples, receipts, WAV/MP3 and
source-to-audio map are under:

`C:\Users\brian\omni-audiobook\runtime_evidence\2026-10-12-performance-memory-chapter-11`

From `C:\Users\brian\omni-audiobook\ai-audio-studio`:

```powershell
$Chapter11Root = 'C:\Users\brian\omni-audiobook\runtime_evidence\2026-10-12-performance-memory-chapter-11'
.\.venv\Scripts\python.exe scripts/render_fresh_coached_chapter.py `
  --epub C:\Users\brian\Downloads\ArKendrithyst.epub `
  --source-root "$Chapter11Root\source-projects" --project 65c8b759 --chapter 11 `
  --output-dir "$Chapter11Root\render" `
  --performance-memory "$Chapter11Root\character-performance-memory.json" `
  --coaching-bundle "$Chapter11Root\chapter11-coaching.json" --generate

.\.venv\Scripts\python.exe -m pytest `
  tests/test_character_performance_memory.py tests/test_coached_chapter_batch.py `
  tests/test_scene_coaching.py tests/test_fresh_coached_chapter.py -q
```

The completed first command verifies and reuses all existing audio with zero
inference requests. For a new chapter/edition use a new output directory, root
prepared exact source/attribution, current source permissions, retained cast
bindings and explicit private review authorization; do not mutate this edition.
Memory must be supplied again on every invocation. New generation requires an
idle, root-managed serial GPU lane; the CLI does not park other services.

The local listening route is port9145 and the private review URL is recorded in
`mobile-page-verification.json`. Tailscale access as the enabled Brian recipient
is required. Audio, private prose, model packets, clone assets and credentials
are excluded from Git.

## Limits and next verification

Four turns have uncertain speaker attribution (quote indices23,88,89,90), with
their original model confidence preserved. The explicit shared Bacci/Savral
“Uhhh” uses Bacci as a single representative; two subsequent untagged turns use
an inferred alternating order. These are transparent review choices, not source
certainty or human-approved casting. A simultaneous ensemble renderer is absent.

The memory is captured interpretation rather than an approved speaking-style
profile. Native instruction forwarding is verified, but effect size, recognizable
habits and character consistency require listening. Narration is newly synthesized
neutral review audio, without selected or approved Takes. Automatic model calls
inside the live Studio UI, voice-specific instruction calibration, independent
emotion/duration control and directed production promotion remain future work.
No additional model download or casting approval was needed for this private
chapter11 edition; human listening remains the next quality decision.
