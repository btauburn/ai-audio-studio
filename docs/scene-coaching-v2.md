# Stronger scene coaching — October 12, 2026

Brian listened to the first scene pilot and reported slight improvement with
insufficient contrast. This follow-up applies whole-scene model coaching to all
14 dialogue lines in the same 20-line selection. The six narrator lines, cast
identities, original prose, generation settings and semantic boundary policy
remain fixed. Listening quality and production approval of v2 remain pending.

## Implementation

`dubbing_service/app/audiobook/scene_coaching.py` accepts a captured coaching
request/response bundle, validates exact dialogue coverage, source evidence and
emphasis offsets, and overlays temporary native instructions on an existing
source-proven plan. Emotion, intention/objective, cadence, emphasis, endings and
confidence remain separate structured fields. Model interpretation is labeled
`model`, never `explicit_human` or author-written evidence. The API makes no
network or inference call. It refuses narrator instructions, identity changes,
changed live context, unresolved source/voice/safety holds, hash mismatches and
conflicting cache entries. It creates no approval or Take.

Interpretation caches bind the individual scene's entire supplied context,
model, policy and directions. They do not bind the voice, so an approved voice
revision can reuse the interpretation while the existing plan's voice/source
binding changes. Prose or attribution changes invalidate reuse. Instructions
are bounded to 220 characters and use the existing native instruction guard.

`performance_scene_analysis.py` version `pd-scene-analysis-3` stops inferring
grief from the words “dead”/“died” alone and evasion from “anyway” alone. An
introductory pronoun tag ending in comma/colon, between two lines already
attributed to the same role, binds to the following line. Different-speaker
ambiguity is preserved. Author-written cues still outrank keyword guesses.

`scripts/scene_dialogue_pilot.py` adds `--coaching-file` and `--selection-from`.
The latter preserves a verified previous pilot's exact scene/line/cast selection
and reuses its neutral baseline clips, including neutral narration for the new
directed arm. Original raw receipts and generation measurements are retained;
the reuse ledger records where they came from. Only new dialogue is synthesized.
Previous audio is read only. Strict comparative QA validates old clips before
reuse and verifies the new paired samples afterward. Unknown dispatch outcomes
continue to hold rather than trigger automatic resends.

The exported page shows baseline, previous v1 and stronger v2 players in each
of four scenes. The comparative manifest and blind ballots remain two-arm
baseline versus v2. V1 is a labeled historical reference, not a third blind arm.
The player lists the proposed directions, objectives and confidence. This is an
isolated extension of the existing director runner; the normal casting UI and
production engine do not automatically invoke a new coaching service.

## Model and editorial provenance

The installed Claude routes were probed: Sonnet 5.5 stopped at the account spend
limit; Haiku 6 was rejected as an unrecognized model. Neither authored this
coaching run. One actual Codex CLI request using `gpt-5.6-sol` interpreted all
four bounded scenes together, without tool calls. Its captured usage was 20,064
input and 7,932 output tokens (including 4,905 reasoning tokens). The raw response
and CLI events remain private alongside their hashes.

Two normalizations are recorded explicitly: one exact evidence quotation had
its offset repaired by a unique source-substring lookup; one delivery sentence
was rephrased to avoid the conservative identity-word guard's “sound” match.
The raw model result is retained. Neither normalization changes source prose or
claims a human decision. No additional model download was needed.

## Launch and retry

Use a new output directory outside the source store. Supply a captured bundle
with schema `scene-coaching-bundle-v1`, its original request, validated response,
request/response SHA-256 and actual model-event receipt provenance. The private
run contains an example; book text and cloned references must not enter Git.

```powershell
.venv/Scripts/python.exe scripts/scene_dialogue_pilot.py `
  --source-root C:/Users/brian/omni-audiobook/runtime_evidence/2026-09-28-autocast-elf-acceptance/projects `
  --projects 9c7257c4 b3b6b320 c59f2ad6 `
  --output-dir C:/Users/brian/omni-audiobook/runtime_evidence/2026-10-12-scene-coaching-v2 `
  --selection-from C:/Users/brian/omni-audiobook/runtime_evidence/2026-10-12-scene-dialogue-pilot `
  --coaching-file C:/Users/brian/.codex/task-observer/runs/2026-10-12-scene-coaching-v2/coaching-bundle.json `
  --min-directed-dialogue 14 --min-directed-scenes 4
```

Preparation is CPU-only. Add `--generate` only with the isolated GPU lane idle
and port 8143 free. Do not launch beside another GPU inference worker. This
run's private helper checks job queues and exact ownership, parks only the
chat-owned review worker, and restores the same runtime identity afterward.
Completed retries reuse verified samples; a changed coaching plan needs a new
directory. The approved `faster-base-032` default is untouched.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_scene_coaching.py tests/test_performance_scene_analysis.py tests/test_scene_dialogue_pilot.py -q
```

## Evaluation

Actual synthesis completed: 14 new dialogue requests, 26 reused neutral
samples, 40 samples in the paired evaluator and 12 labeled scene players.
Forwarding receipts prove all 14 new instructions reached the experimental Base
route; baseline and narration carry zero instructions. Clip totals are 60.240s
baseline and 61.120s v2. New requests took 84.388s in total, including 48.71s for
the first request's startup/conditioning work. Largest sampled allocation was
8,023 MiB of total device VRAM, including desktop/background use. These are
observations from one run, not a throughput improvement claim.

There were no acoustic flags in the directed arm. The reused baseline retains
four isolated full-scale samples below the harness's likely-clipping threshold.
Offline CPU Whisper Base.en reported 10 disagreements out of 184 words in both
arms (5.43%); v1 had 12. Each arm has one possible omission, in shared neutral
narration. Tail disagreements are three baseline versus two v2. The v1 dinosaur
word disagreement is absent in v2, although laugh-token differences remain.
ASR is supporting evidence and contains spelling/name/tokenization errors;
these counts do not prove intelligibility or acting quality.

Private review: `https://brian-desktop.tail3e7427.ts.net:8443/scene-coaching-v2/`.
All 12 audio URLs returned Range HTTP206; unauthenticated and wrong-recipient
backend requests returned 403. Previous Tailscale Serve routes were preserved.
Pushover accepted one explicitly requested notification (HTTP200); phone display
is not confirmed. Mixed scenes alone are exposed through the existing Brian-only
boundary; source/reference assets and the sealed ballot key stay private.

Verification: 116 focused checks passed before synthesis. The broader selected
casting, performance, comparative-QA and Take suite passed 1,281 tests plus 30
subtests; its 30 warnings were Pydantic enum-serialization warnings. After a
diagnostic-only amendment preserving the previous interpretation alongside model
coaching, all 18 coaching checks passed again. Structured pre-commit review is
required before publication; its receipt remains in the private run directory.

Human listening is decisive; stronger instructions alone do not establish an
improvement. Full-chapter directed production, automatic model orchestration
inside Studio, performance promotion and IndexTTS comparison remain proposed.
