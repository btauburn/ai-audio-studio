# Coached full-chapter renders (V2) — October 12, 2026

`scripts/render_coached_chapters.py` (implemented in
`dubbing_service/app/audiobook/coached_chapter_batch.py`) extends the
[scene coaching pilot](scene-coaching-v2.md) from a 20-line selection to complete
chapters. It is a resumable, review-only batch over existing approved cloned
chapter projects. It chooses no voice, selects no Take, approves nothing and does
not change the production default. Listening quality of the result is pending;
nothing here claims improved audio.

The work is split so that model coaching and GPU synthesis stay separate, owned
steps. Existing owners are reused, not duplicated:

| Concern | Owner reused |
|---|---|
| Canonical chapter, source proof, selected neutral Takes | `voice_lab_chapter_export._source_snapshot` / `_take_snapshot` |
| Proof that Takes used the approved Faster Base route | `runtime_binding.take_runtime_profiles` |
| Numbered EPUB chapter selection | `epub_chapter_selection.select_epub_chapters` |
| Scenes and base recommendations | `scene_dialogue_plan.build_scene_dialogue_plan` / `verify_scene_dialogue_plan` |
| Model coaching validation and overlay | `scene_coaching.validate_bundle` / `apply_coaching` |
| Clone table and files, isolated project copy, direction guards, receipts, source contract | `scripts/scene_dialogue_pilot.py` |
| Serial directed worker and render dispatch | `scripts/directors_vertical_slice.py` (`Worker(directed=True)`, `render`) |
| Boundary measurement/classification, WAV/MP3, loudness, QA | `performance_boundaries`, `voice_lab_render`, `qa` |

## Stages

All three stages use one private output directory outside the canonical project
store, and one frozen `state.json`.

1. **Prepare** (`prepare_batch`, CPU only). Reads the exact EPUB bytes, selects the
   numbered chapters, and for each chapter project proves: the EPUB SHA-256 is a
   registered source revision; the project's chapter title is the EPUB title; every
   nonblank EPUB character appears, in order, in the live segments (only whitespace
   is ignored, so headings and scene-break markers are not lost); the chapter's
   source, ordered coverage, break manifest and exclusions are intact; every speech
   role has a locked eligible cloned identity on the Qwen Base route with verified
   reference/prompt files; and every selected neutral Take passes release QA and
   proves the `faster-base-032` route. It then freezes every speech line, copies
   the neutral narrator audio (hash verified) to `neutral/`, builds the base scene
   plan for the chapter's speech ids, and writes **one** request per chapter:
   `requests/request-{project_id}.json` with keys `instruction`, `constraints`,
   `response_contract`, `scenes`. Each scene carries its full context (neutral
   narration included, `narrator: true`) and `selected_dialogue_ids` for non-narrator
   speech only. There are no per-sentence calls and no narrator directions are
   requested. The status is `awaiting_coaching`. No instruction is invented.
   Cloned references and approved audition evidence are copied to `projects/` with
   the pilot's helper so the worker never reads the canonical store.
2. **Apply** (`apply_bundles`, CPU only). For each `coaching-{project_id}.json` in
   `--coaching-dir`: the bundle's request must equal the frozen request exactly, its
   hashes and captured model provenance must validate, and `apply_coaching` must
   accept it against the live source and the base plan. It cannot clear a source,
   approval or safety hold. The result must cover every dialogue line, keep narration
   neutral, bind each direction to the frozen text hash and neutral Take voice
   revision, and pass the pilot's direction guards (native instruction at most 220
   characters, no accent/pitch/gender/performer/identity wording). The current
   canonical clone files, the isolated copies and the project's semantic source
   contract are re-verified. Only then are `coaching/bundle-*.json`,
   `plans/coached-*.json` and `applied/*.json` published, exclusively. Missing
   bundles leave `awaiting_coaching`; a refused bundle makes the status `held` with
   the reason and publishes nothing for that project. All seven applied means
   `ready`.
3. **Generate** (`generate_batch`, explicit root-owned GPU lane). Re-derives every
   direction from the immutable published files (not from `state.json`), re-checks
   the source contract (the pilot's `source_contract`, which ignores only the
   delivery notification counters), neutral copies, canonical neutral Takes and clone
   files, and then renders only the **new coached dialogue**, one request at a time
   through a **single** `Worker(directed=True)` for the whole batch. Each result is
   checked with `pilot.verify_sample` (exact text, voice, request hash, runtime pin,
   forwarded instruction digest and policy, no fallback) and technical QA. Narration
   is never synthesized and never carries an instruction; the audit proves it. Each
   chapter is then assembled in source order from the reused neutral narration and
   the new dialogue, with the existing semantic boundary classifier, measured
   leading/trailing silence, no trimming, and the recorded 1.25 s target for source
   scene separators. WAV and MP3 are produced with the shared `voice_lab_render`
   tools and an exclusive chapter receipt is written.

## Output layout

```
state.json                         frozen section + mutable status/coaching ledger
requests/request-{project}.json    one whole-scene request per chapter (hash-bound)
plans/base-{project}.json          base scene plan (unapproved recommendation)
plans/coached-{project}.json       coached plan (published once, immutable)
coaching/bundle-{project}.json     the captured model bundle that was applied
applied/{project}.json             hashes, model provenance, direction digest
neutral/{project}/{utterance}.wav  verified copies of the existing neutral Takes
projects/{project}/                isolated worker copy of metadata + clone files
samples/{project}/{utterance}.wav  new dialogue audio and the render owner's raw receipts
receipts/{project}/{utterance}.json verified sample receipts (coaching, QA, neutral Take reference)
runtime-pin.json                   worker runtime pinned on first dispatch
forwarding-audit.json              exact forwarding/narration audit
chapters/{project}/chapter-NNN.{wav,mp3,json}   review audio and the exclusive receipt
batch-result.json
```

The chapter receipt keeps `human_review: pending`, `production_ready: false`,
`takes_selected: false`, `approvals_created: false`, and a full `source_to_audio`
map: EPUB-verified source anchor, role, clone binding, audio hash and timing for
every speech line, `scene_break`/`excluded` rows, and for dialogue the raw render
receipt, instruction hash and coaching reading. Reused narration is marked
`existing_neutral_take_reused_unchanged`; the existing neutral Take reference
(Take id, audio hash, fingerprint, voice version) is also retained for dialogue and
is never replaced or selected. The scene reading (objective, emotion, cadence,
emphasis, ending, confidence) is stored separately from the native instruction
that conditions the voice. Spoken text is never rewritten.

## Retry and resume

* Prepare, apply and generate are idempotent. Identical inputs republish identical
  bytes (a no-op); different bytes for any published artifact are a conflict and
  stop the batch. A changed bundle after application is reported as `held` and the
  published plan is kept.
* A completed render is verified byte-for-byte on retry; if nothing is missing, no
  worker is created. A partial run resumes only missing dialogue with one worker.
* Any unresolved pending dispatch record without a receipt is treated as ambiguous.
  The batch holds **before** a worker exists and never resends. The pilot's single
  proven pre-inference approval-file refusal is resolved the same way it is there.
* A different runtime than the pinned one, a profile other than
  `faster-base-032-direction-x1`, unconfirmed forwarding, a worker fallback or a
  different voice identity is a refusal. No preset is substituted for a clone.

## Running it

```powershell
$run = 'C:/Users/brian/omni-audiobook/runtime_evidence/2026-10-12-coached-chapters-4-10-recovery-1'
.\.venv\Scripts\python.exe scripts/render_coached_chapters.py `
  --epub C:\Users\brian\Downloads\ArKendrithyst.epub `
  --source-root C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-28-autocast-elf-acceptance\projects `
  --projects a0889ff6 f0e957c4 e0eb19c3 03b34b2c fc01e3bd 4ab87955 acc9c1fe `
  --chapters 4 5 6 7 8 9 10 `
  --output-dir $run
```

Chapter 7 uses `03b34b2c` (the EOS repaired paragraph rendition). Supply the
captured responses as `coaching-{project_id}.json` in a directory and add
`--coaching-dir DIR`; add `--generate` only when the isolated GPU lane is idle and
port 8143 is free. `--projects` and `--chapters` are ordered pairs; chapters must be
ascending and may be space or comma separated. Stdout ends with one ASCII JSON
summary line (the render owner prints progress lines before it). Exit code `0`
means prepared/awaiting, ready or generated; `2` means held or refused.

Private books, voices and audio stay outside Git.

## Verification

```powershell
.venv/Scripts/python.exe -m pytest tests/test_coached_chapter_batch.py -q
```

The tests use only labeled synthetic fixtures: invented prose, tone-signal Takes,
fake clone files, a synthetic bundle standing in for a model response and a fake
worker. No real book, voice, model, GPU, network or live project is touched. They
cover source completeness (character-level and whitespace-only), cast/clone
refusal, bundle hash/evidence/coverage/safety, narrator never dispatched, safe
retry and conflict, ambiguous pending dispatch, source/cast/neutral/clone drift,
notification-counter tolerance, resume with a single worker, exclusions and scene
separators, and fixture labeling of every receipt.


## Executed chapter batch — October 12, 2026

The renderer was committed and pushed **before chapter synthesis** in
[`a6133e1`](https://github.com/btauburn/ai-audio-studio/commit/a6133e1dd6a6f922ae0d1209a69f9ea4715cceb8). All seven
numbered EPUB chapters were then rendered to private WAV/MP3 exports and verified
against their complete source maps. Every chapter remains `review_only`, with
`human_review: pending`, `quality_improvement: not_established` and
`production_ready: false`.

| Chapter | MP3 duration | Coached dialogue | Reused neutral narration | Added boundary padding (s) | Measured MP3 LUFS |
|---|---:|---:|---:|---:|---:|
| 4 | 38:31 | 118 | 123 | 65.740 | -26.0 |
| 5 | 25:18 | 90 | 85 | 49.100 | -26.2 |
| 6 | 27:42 | 132 | 108 | 72.530 | -25.3 |
| 7 | 19:47 | 73 | 58 | 36.540 | -24.9 |
| 8 | 14:42 | 92 | 64 | 39.390 | -25.8 |
| 9 | 17:04 | 93 | 70 | 43.360 | -25.6 |
| 10 | 18:03 | 59 | 57 | 32.230 | -26.2 |

Total listening audio: **2.685 hours**. The batch uses
1,222 speech segments, with **657 instruction-conditioned dialogue samples** and
**565 unchanged neutral narrator Takes**, retaining all five existing cloned
identities. The exact EPUB, canonical source/Take contracts, clone references,
prompt files, global voice registry and all 18 pre-existing local-file baselines
were rechecked. No approval, Take selection, reference replacement, prose rewrite
or global runtime switch was created.

Measured boundary assembly added 338.889s across
1215 speech boundaries. 152
already contained enough measured silence and needed zero added padding. Source
separators retain their recorded 1.25s target. No synthesized speech was trimmed.
Master durations equal the sum of source clip durations and recorded padding;
decoded MP3 duration agrees within 0.1s. LUFS is a measurement of each delivered
file, not a claim of final mastering or a listening preference.

The sum of the **final accepted dialogue samples' original request latencies** is
2049.959s for
3329.842s of generated dialogue. Peak sampled
total-device VRAM was 8696MiB, including
desktop/background allocations. These receipts span the original and recovery
workers; the sum is not batch wall time, excludes discarded outputs/diagnostic
probes, and is not a speed comparison. GPU requests were serial. The chat-owned
idle review tree was parked and restored with the identical original runtime
identity; the other review/production services were preserved.

**Actual delegation and interpretation:** one Claude CLI `claude-sonnet-5-5`
engineering worker implemented the batch module, CLI, 58 fixture checks and initial
documentation. The root owned source reconciliation, the existing separator fix
adoption, integration, additional checks, commit/push, GPU lane, recovery and
private delivery. Seven further successful Sonnet 5.5 calls interpreted whole
chapter-scene packets, with no tools or per-sentence calls. Raw provider responses
and hashes remain private. Bounded editorial normalizations retained raw outputs,
repaired source-evidence offsets and removed identity wording; evidence added by
the root is marked as editorial. Haiku 6 was not needed. The eight captured Claude
CLI receipts report $7.6430054 at list cost; this is not actual account billing or
the cost of the complete root session.

**Recovery evidence:** the original edition stopped at an overdriven single-word
laugh that failed the blocking peak check. A separate recovery edition retained
both original outputs and changed only two unapproved temporary directions: the
laugh and a warning ending flagged by ASR. It verified and reused 358 raw samples,
then generated 299 missing/revised samples; derived receipts were rebuilt against
the current plan. The revised laugh passed QA. The warning's former eight-word
tail deletion is absent from the revised ASR result, which still has two word
substitutions. No prose or voice identity changed, and ASR does not prove acting
quality.

One subsequent HTTP400 reference-isolation replay refusal returned no audio
before synthesis. The immutable acquisition and its exact full CPU replay passed
fresh validation; plain, threaded, Base-loaded and post-synthesis diagnostic
probes reproduced no recipe difference. The precise cause is unconfirmed. The
original refusal/reservation was retained with a root-owned one-retry resolution;
the fresh worker completed without a provenance or QA waiver. This is not an
automatic retry of an unknown dispatch outcome.

Offline CPU Whisper Base.en checked all 657 dialogue samples, with no downloads:
516 edits over 10192 source words
(5.06% ASR disagreement), 58 possible
omissions and 97 tail-token mismatches. These include
fantasy names, spelling and short interjections; they are supporting diagnostics,
not confirmed errors. Existing neutral narration was hash-reused, not re-synthesized.
The complete chapter receipts and all sample-level blocking technical QA passed.
Human judgment of intelligibility, acting, voice consistency and transitions is
still required.

Verification: **58 batch checks and 284 selected regression checks passed**.
Structured Codex review reported no accepted findings at its configured P0
threshold; changed-content TruffleHog was clean. All seven private Brian-only
players/downloads were checked with matching MP3 hashes and Range206; missing or
wrong identity receives403. Earlier private Serve routes were preserved. Private
EPUB prose, reference recordings, prompts, model packets and audio remain outside
Git. The existing 20-line baseline/v1/v2 listening comparison remains available;
the full chapters are new review exports, not blind comparative ratings or
production approval.

The completed recovery edition is:
`C:/Users/brian/omni-audiobook/runtime_evidence/2026-10-12-coached-chapters-4-10-recovery-1`.
Use the command above with this output directory and its `coaching/` directory.
An identical completed `--generate` retry verifies existing artifacts and launches
zero inference workers. Listening files are
`chapters/<project-id>/chapter-004.mp3` through `chapter-010.mp3`, each alongside its
WAV master and source-to-audio receipt. Run evidence is under the existing private
Task Observer run `2026-10-12-coached-chapters-4-10`.

Next evaluation remains **proposed**: calibrate instruction response for each
approved clone using human preferences before promoting scene coaching into
production. Native instruction forwarding is verified; a substantial acting
improvement has not been established.

## Known limits

* The measured run above exercises the real worker, ffmpeg, source and clone
  contracts. Scene analysis still caps individual scenes at 40 utterances, and
  successful technical checks do not establish better acting.
* Preparation refuses rather than guesses when the EPUB hash is not a registered
  source revision of the chosen project (including the repaired chapter 7
  rendition) or a Take does not prove `faster-base-032`.
* Technical QA and boundary measurements are not acting quality; human listening
  decides.
