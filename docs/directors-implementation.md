# Casting and Performance Directors — October 11 implementation

Implemented in `feature/casting-performance-director-20261011` on the existing
Asset Forge / Voice Lab repository. The lead was **Codex**, with five actual
**Claude Sonnet 5.5** CLI workers. This was not an Opus runtime. Their structured
returns, prompts, actual model usage, preflight graph and preservation captures
are in `C:/Users/brian/.codex/task-observer/runs/2026-10-11-casting-performance-director/`.

## Implemented owners

| Owner | Implementation | Actual contribution |
|---|---|---|
| Sonnet A | `app/audiobook/casting_director_discovery.py` | Approved-library-first discovery, immutable generic references, trusted external-source metadata with separate recording/use/consent holds, original design directions for gaps, protection and deduplication |
| Sonnet B | `app/audiobook/casting_director_matching.py` | Book-supported traits, role importance and relationships, explainable ranking, cast and shortlist similarity penalties; unknown quality stays unmeasured |
| Sonnet C | `app/audiobook/performance_scene_analysis.py` | Cached scene interpretation, evidence spans, emotion/intention/pacing/emphasis/response/boundary recommendations, narrator preservation and optional existing direction resolver integration |
| Sonnet D | `app/audiobook/director_evaluation.py`, `scripts/compare_directors.py` | Four-arm acoustic reader, identity/prose/runtime checks, reproducible blind ballots, receipt-bound latency/VRAM and human review gate |
| Sonnet E | `webui/casting-director.{js,css}`, `webui/voice-lab.html` | Additive Voice Lab panel; existing portraits, playback, ratings, accept/reject and library workflows remain the owners |
| Codex lead | `app/audiobook/casting_director.py`, `app/routers/directors.py`, `app/main.py`, small Casting Review/private proxy hooks | Shared contracts, current evidence projection, protected casting, role-specific rejection filters, immutable analysis, fresh common-script auditions, retries, final integration and live verification |
| Codex lead | `scripts/directors_vertical_slice.py`, `scripts/transcribe_director_comparison.py` | Isolated real-audio slice, semantic padding that accounts for existing silence, GPU telemetry and offline CPU ASR diagnostics |

`app/` and `webui/` paths above are relative to `dubbing_service/`.
Tests are `tests/test_casting_director_*.py`, `test_performance_scene_analysis.py`,
`test_director_evaluation.py`, `test_directors_delivery.py`,
`test_director_asr_diagnostics.py`, and `casting_director_ui.test.cjs`.

## Architecture and behavior

The existing project, library, source policies, Casting Review rounds, audition
jobs, Qwen provider, GPU lease, Take commit and QA owners remain authoritative.
The new report is a recommendation, never an approval or new project database.

An explicit analysis projects current book evidence and rebuilt profiles on a
deep copy. It does not save a changed dossier, role, identity or Take. Approved
library versions are searched first. Weak unapproved presets and 1–3-star
candidates are held. Protected library voices remain reusable; a rejection for
this particular role still excludes that exact reference from its new shortlist.
Roles without an adequate prepared reference can receive up to five original
model-design directions; these are labeled directions until an explicit audition
generates audio. Unsupported accents are not added to these directions.

Scores combine character fit, cast distinction, human rating and separately
reported fidelity/clarity/naturalness/stability. Missing measurements contribute
no invented evidence. Similarity currently uses metadata traits, not speaker
embeddings. Character identity, habitual speaking style, scene intention and
boundary timing remain separate. Every inferred fact and cue is labeled.

Reports and scene interpretations are exclusively published under
`<project>/voice_lab/directors/`. Cache identity includes source, attribution,
roles, approvals, preferences, policy, library versions, runtime and algorithm
revision. Interpretations have a separate voice-independent cache. GET, opening,
navigation and playing audio never generate or start an LLM call.

An audition request freezes the report and exact role revision. Imports use the
existing library owner and transfer no approval. Reused references receive new
audio on the same common audition script through the normal Faster Base provider
and immutable Take/QA pipeline. The prior library recording remains historical
evidence. Accepted/rejected voices, stars and library publication still use the
existing explicit controls. Approved characters cannot be recast through this
new endpoint. Retrying an operation returns its original receipt.

Endpoints, beneath `/api/v1/projects/ID/voice-lab/directors`:

- `GET ""`: read the current cached report, or `status=needs_analysis`.
- `POST /analyze`, header `X-Review-Request: 1`: extract/cache recommendations.
- `POST /auditions`, same header, JSON
  `{request_id, role_id, report_sha256}`: queue ordinary Casting Review auditions.

The private delivery projection forwards these three operations and only
forwards the explicit review header for director mutations. Existing Tailscale,
origin, project allowlist and feedback-authority checks remain in force.

## Real vertical slice and measured results

Runtime artifacts:
`C:/Users/brian/omni-audiobook/runtime_evidence/2026-10-11-directors-vertical-slice-v2/`.
Source project `80b99ad8` was copied into this isolated store. Its actual approved
narrator, Harry and Ron retain the same references in all arms. An extra unassigned
review role compares the existing synthetic Dumbledore reference with a ranked
library reference. This does not replace Dumbledore or any source-book assignment.
The 12 evaluation lines are exact sentence slices from existing original audition
text, rearranged as an explicitly labeled evaluation script. No book prose was
rewritten, and this is not a full-book performance test.

The review project is `a5d03cea`. Five distinct immutable reference identities
produced **five real QA-passed, common-script auditions**, lasting approximately
17–28 seconds. Actual portraits, play controls, 1–5-star controls and explicit
selection/rejection are displayed in the existing casting inbox. Human decisions
remain pending; API regressions exercise acceptance/rejection without inventing
a real human approval.

The unchanged default worker was first exercised independently: 4.4 seconds of
narrator audio, 37.95 seconds cold wall time. Its waveform was byte-identical to
the neutral narrator control from the experimental wrapper.

All four comparison arms used the same pinned Base checkpoint, Faster 0.3.2,
Torch 2.10.0+cu130, BF16, SDPA, full ICL references and generation settings.
The neutral baseline in that comparison runs the parent's unchanged synthesis
path inside `faster-base-032-direction-x1`, keeping the runtime contract identical
across arms. The global `faster-base-032` default was not changed.

| Arm | Clips | Generated speech, s | Median request wall time, s | Forwarded dialogue instructions | Added boundary padding, ms | ASR WER |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 12 | 48.24 | 1.62 | 0 | 3850 | 0.78% |
| Casting | 12 | 49.44 | 1.61 | 0 | 3850 | 1.55% |
| Performance | 12 | 46.08 | 1.51 | 9 | 3590 | 0.78% |
| Combined | 12 | 47.52 | 1.62 | 9 | 3190 | 0.78% |

The dataset has 129 reference words per arm. ASR found zero deleted words and one
extra casting-only disagreement at a sentence ending. These are recognition
diagnostics, not proof of audible correctness. Local
`models/whisper/faster-whisper-base.en` ran on CPU/int8; its exact file hashes,
word timestamps, texts and edit alignments are retained. No model was downloaded.
Acoustic measurements found zero clipped or non-finite samples. Sampling through
`nvidia-smi` observed a maximum **6476 MiB** of total-device memory, including the
desktop/background allocation. This is a sampled device peak, not an allocator
peak. Requests include cold conditioning/model loads; these are visible in the
receipt stage timings rather than presented as steady-state throughput.

The scene analyzer conservatively proposed no specific native instruction for
these neutral excerpts. The emotion experiment consequently uses **explicitly
labeled engineering comparator hypotheses**, stored in `direction-hypotheses.json`:
concern/urgency, restrained dry amusement, and gentle reassurance. They are not
claimed as book facts or human-approved directions. Eighteen instruction receipts
verify actual forwarding; **zero narrator instructions** were sent. Whether they
sound better remains unestablished until human listening.

Semantic timing uses five narrative/dialogue boundaries, five replies and one
same-speaker boundary. It measures strict leading/trailing silence before adding
padding and never trims speech. Existing silence totaled 1050–2090 ms depending
on the arm. Targets are experimental hypotheses, not measurements of commercial
audiobooks. The 350-ms comparison baseline is a controlled listening condition,
not a measurement of existing book-export pauses. ASR and acoustics do not certify
naturalness, role distinction or long-form voice consistency.

Playable mixes: `baseline-semantic-v2-scene.wav`, `casting-semantic-v2-scene.wav`,
`performance-semantic-v2-scene.wav`, `combined-semantic-v2-scene.wav`.
`comparison-manifest-directed.json` binds all 48 clips, voices, source text,
runtime hashes and telemetry. `evaluation/ballot.json` is the reproducible blind
ballot; keep `ballot_key.json` from blind reviewers. `asr/asr-diagnostics.json`
contains the supporting recognition results. No improved-quality claim or
production promotion is made.

## Verification and preservation

- Final feature/source integration and ASR checks: **149 passed**. The final
  evidence/rejection/engine guard, ASR and private delivery checks also passed 44
  focused tests. A source-only dependency snapshot passed 82 with 11 explicit
  local-environment/private-capture skips; the installed environment passed the
  corresponding runtime/fingerprint composition checks.
- JavaScript feature plus existing portrait, casting, transcript, comparison and
  startup checks: 161 passed, including the final runtime-disposition regression.
- Existing and directed Faster adapter/server unittest checks: 45 passed in the
  isolated Faster environment.
- Broad CPU run: **3003 passed, 2 failed, 2 skipped, 162 subtests** in 1090.64 s.
  One failure was the private proxy forwarding the review header on unrelated
  library mutations; the fix limits it to director operations. The other was
  `test_reference_quality::test_api_uses_fixed_hashes_and_preserves_immutable_receipt_and_clone_hold`;
  it passes in isolation. Focused private-proxy/reference checks then passed 38.
  A completely green full-suite rerun has not been established; do not substitute
  the focused result for it. The two established CPU exclusions remain
  `test_indextts_audio_safety.py` and `test_service_server_api.py`.
- Initial dirty state: 61 files captured separately. The related direction
  subsystem is adopted in its own commit. The delivery owner was extended; two
  dependency tests gained explicit missing-environment/capture skips and one
  test's trailing space was removed. Unrelated working changes retain their bytes,
  including the independently owned source-separator changes in the mixed export
  file, which remain unstaged.
- Selected source project `80b99ad8`, main book segment metadata, global
  `voices.json` and library asset bytes are unchanged. The 698-file audit observed
  13 unrelated live project JSON revisions advancing while existing background
  owners continued running. Their newer metadata was retained; no reset or
  restoration was attempted. They are not evidence of a byte-identical entire
  runtime store.

## Launch and reproduce

The current review is available locally at `http://127.0.0.1:9140/`; its source is
an isolated review launcher in this task's workspace. The ordinary product page
is `http://127.0.0.1:9000/web/voice-lab.html?project=ID` after the Studio process
has been started with this feature's code. Existing running production services
were not restarted. Expand **Review recommendations**, then choose **Analyze
characters and scenes**. For an unapproved role, choose **Request fresh auditions**
and listen/rate/accept/reject in the existing inbox. The library remains the normal
reusable-voice workflow.

From the repository, prepare and explicitly generate another isolated slice:

```powershell
.\.venv\Scripts\python.exe scripts/directors_vertical_slice.py `
  --source-root C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-28-autocast-elf-acceptance\projects `
  --source-project 80b99ad8 `
  --output-dir C:\Users\brian\omni-audiobook\runtime_evidence\my-director-review --generate
.\.venv\Scripts\python.exe scripts/compare_directors.py `
  C:\Users\brian\omni-audiobook\runtime_evidence\my-director-review\comparison-manifest-directed.json `
  --output-dir C:\Users\brian\omni-audiobook\runtime_evidence\my-director-review\evaluation
.\.venv\Scripts\python.exe scripts/transcribe_director_comparison.py `
  C:\Users\brian\omni-audiobook\runtime_evidence\my-director-review\comparison-manifest-directed.json `
  --model-dir models\whisper\faster-whisper-base.en `
  --output-dir C:\Users\brian\omni-audiobook\runtime_evidence\my-director-review\asr
```

The demo is explicitly tailored to the existing source project's narrator/Harry/
Ron/Dumbledore audition roles. Its output must be outside the source root. GPU
work is sequential on isolated port 8143; an occupied port or insufficient free
memory refuses generation. Completed receipt/hash pairs are reused. Ambiguous
dispatches are held, and known validation refusals retain their earlier evidence
when a corrected request receives a new dispatch identity.

## Limits, optional models and approvals

The existing M1–M5 direction subsystem, including `performance_director.py`,
`performance_boundaries.py`, `performance_provider.py` and the two direction
worker files, was already present as uncommitted work. Its related source was
adopted in a separate dependency commit after independent review and regression
verification, with attribution as pre-existing work. Unrelated reference-policy,
source-separator and Reaper changes remain unstaged. The live comparator still
requires the separately installed pinned Faster environment and approved private
voice assets. Tests requiring that environment or private historical captures
skip explicitly in a source-only checkout. No global instruction switch or
automatic production enablement was introduced.

External discovery currently accepts trusted source metadata and the existing
owner's rights decision; it does not invent consent, scrape/download recordings
automatically or bypass reference-quality/portability holds. Some preserved
recorded voices retain historical rights/portability cautions. Human approval,
external speaker authorization, long-form listening and any per-book performance
enablement remain separate decisions. Full speaker-embedding measurement and
automatic model-based scene reasoning remain future optional work.

The official [IndexTTS repository](https://github.com/index-tts/index-tts) now
describes 2.5 emotion/timbre separation and `duration_factor` controls. It is a
possible independent comparator, not the installed default. This run did not
download 2.5, validate its license bundle, run it on the RTX 4070, or promote it.
Those checks remain required before adoption. Installed Faster code confirms its
native instruction seam; current [Faster documentation](https://github.com/andimarafioti/faster-qwen3-tts)
also favors full ICL over experimental x-vector-only instruction conditioning.
