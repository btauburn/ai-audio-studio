# Scene-aware dialogue pilot — October 12, 2026

Implemented and rendered: a source-bound, review-only performance pilot over 20
unchanged lines from Ar'Kendrithyst chapters 1–3. The existing recorded clones
remain assigned. Narration receives no native instruction. This is an audition
of temporary delivery, not a voice replacement or production promotion.

## Architecture and contracts

`dubbing_service/app/audiobook/performance_scene_analysis.py` now resolves
`source_anchor.source_revision_id` through `project.source_revisions`. An ID
is not a SHA-256. Missing, duplicate, malformed and unknown registry entries
produce holds. Analyzer version `pd-scene-analysis-2` invalidates the previous
interpretation cache. Author-written delivery cues can carry through bounded
conversation turns, with explicit evidence spans and lower confidence; a
contradiction, narrative gap or scene boundary ends that carry.

`scene_dialogue_plan.py:build_scene_dialogue_plan` reads complete scene context
and returns the selected lines' short delivery proposals. Interpretation is
cached independently of voice identity. Source, exact text, attribution, aliases,
scene context, source anchors and current voice revisions are checked separately.
Character habits remain separate from scene intention. Instructions naming a
different speaker, accent, pitch or performer are refused. Unknown cues stay
unknown. These are deterministic source-cue proposals, not model interpretations
or human decisions; every proposal remains unapproved and review pending.

`casting_director.py` integration version v8 includes the source registry in
its report binding and exposes `scene_dialogue_plan` beside the existing scene
analysis. `webui/casting-director.js` displays those proposals in the existing
casting panel. Reading, navigation and playback never request synthesis.

`scripts/scene_dialogue_pilot.py` prepares an isolated copy and freezes complete
contiguous mini-scenes. Selection preserves feasible line/narration combinations
before ranking lexical cues. It copies existing clone assets, acquisitions and
the exact audition audio needed to verify existing human selections. It creates
no approval. Both arms use the pinned `faster-base-032-direction-x1` experimental
wrapper, full reference conditioning, identical text/voice/settings and the same
semantic boundary policy. The approved default remains `faster-base-032`.

Generation is explicit (`--generate`), sequential, and refused if port 8143 is
occupied or less than 8,500 MiB is free. Completed hash-bound samples resume;
ambiguous dispatches hold. A recorded pre-inference HTTP400 approval-evidence
refusal can be retried only after the existing human selection and exact sample
are verified; its original reservation/failure are retained. Source changes hold
before dispatch. Only independent completion-notification delivery bookkeeping
and its revision counters are excluded from project comparison; cast, approval,
source and segment changes are never excluded.

`scene_dialogue_evaluation.py:evaluate_scene_pilot` validates the two-arm
`scene-dialogue-comparison-v1` manifest, exact audio hashes, worker identities,
clone/prompt/reference parity, settings parity and instruction receipts. The
wrapper's unchanged parent path emits no direction receipt for neutral requests;
the evaluator checks its empty controls and actual seed and labels that proof
explicitly, without fabricating a receipt. Directed dialogue must have the full
native forwarding receipt. Per-scene blind ballots and a separate answer key
support human review. Synthetic fixtures remain labeled and excluded from
production readiness. The human gate recommends; it never approves or changes
defaults. Latency, VRAM and acoustic measures are supporting evidence only.

`scripts/transcribe_director_comparison.py` accepts this paired manifest as well
as the earlier four-arm export. It uses the already-installed Whisper Base.en
on CPU, checks audio hashes, and labels ASR disagreements as diagnostics.

## Actual run

Private evidence root:
`C:\Users\brian\omni-audiobook\runtime_evidence\2026-10-12-scene-dialogue-pilot`.
Source chapter projects are `9c7257c4`, `b3b6b320`, `c59f2ad6`. The 20-line set
contains six narrator and fourteen dialogue lines in four mini-scenes selected
around reassurance, humor, fear and worry. It covers the narrator, Jane, Erick
and the Darkness, using the existing Gyles Brandreth, Emma Watson, p254 and p226
clones. The registrar's clone remains retained but is outside this small set.

Forty actual samples were rendered. Seven directed dialogue requests carry
verified native instruction receipts; seven other dialogue lines remain neutral
because the deterministic planner found no supported direction. Baseline and
narration carry zero instructions. Both arms passed the strict comparative
evaluation, with four blind ballots covering all twenty lines. No human review
or improved-quality verdict has been recorded.

| Measurement | Baseline | Directed |
|---|---:|---:|
| Individual clip duration total | 60.240 s | 60.080 s |
| Mean silence fraction | 21.93% | 22.15% |
| Generation latency total | 87.263 s | 43.522 s |
| Largest sampled device allocation | 8,083 MiB | 8,069 MiB |
| ASR word disagreement, 184 source words | 10 edits / 5.43% | 12 edits / 6.52% |
| ASR possible omissions / tail disagreements | 1 / 3 | 1 / 3 |
| Acoustic likely-clipping / active-tail flags | 0 / 0 | 0 / 0 |

The latency totals include warmup, reference conditioning and graph initialization
on the first baseline requests. Baseline-first alternating order biases latency;
this is not evidence that direction makes inference faster. VRAM is sampled
total-device allocation, including desktop/background use, not tensor-only
allocation. A few baseline samples contain isolated full-scale frames below the
likely-clipping threshold; that does not establish audible clipping or a gain.
ASR disagreement increased slightly on the directed arm. Human listening must
decide character fit, naturalness, role separation, consistency and preference.

Generation temporarily parked only this chat's idle review worker tree. The
baseline runtime identity was restored byte-for-byte, and the 9140 review and
9141 chapter player were restored. The baseline is available for normal lazy
model loading. No production service restart or global model switch occurred.
Original segment files, clone references/prompts, approved Take audio and the
global voice bank remain unchanged. Background completion-notification retries
advanced project metadata counters; their live changes were preserved.

The local listening page is
`http://127.0.0.1:9141/scene-pilot/`. Its eight mixed-scene players returned HTTP206
Range responses. The three existing chapter MP3 players also retained HTTP206
playback. Browser inspection confirmed all eight players are initially paused.
Only mixed review exports are exposed by that static mount, not clone references
or the blind answer key. Private book/audio assets are not committed to Git.

## Reproduction

From the repository in PowerShell, prepare without GPU work:

```powershell
.\.venv\Scripts\python.exe scripts/scene_dialogue_pilot.py `
  --source-root C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-28-autocast-elf-acceptance\projects `
  --projects 9c7257c4 b3b6b320 c59f2ad6 `
  --output-dir C:\Users\brian\omni-audiobook\runtime_evidence\my-scene-pilot
```

Use an output directory outside the source store. A ready plan is required;
add `--generate` to render and evaluate through the isolated GPU lane when the
normal worker is idle and safely parked by its owner. The CLI never terminates
another worker. Output includes `state.json`, frozen plans, raw sample/worker
receipts, `forwarding-audit.json`, `scene-comparison.json`, mixed scenes with
timing receipts, `evaluation/`, and a portable `review/index.html` with audio.
Serve only the review directory for labeled listening:

```powershell
.\.venv\Scripts\python.exe -m http.server 9142 --bind 127.0.0.1 `
  --directory C:\Users\brian\omni-audiobook\runtime_evidence\my-scene-pilot\review
```

Optional diagnostics use the installed local model, without downloads:

```powershell
.\.venv\Scripts\python.exe scripts/transcribe_director_comparison.py `
  C:\Users\brian\omni-audiobook\runtime_evidence\my-scene-pilot\scene-comparison.json `
  --model-dir models/whisper/faster-whisper-base.en `
  --output-dir C:\Users\brian\omni-audiobook\runtime_evidence\my-scene-pilot\asr
```

Focused CPU verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_scene_dialogue_plan.py `
  tests/test_performance_scene_analysis.py tests/test_scene_dialogue_pilot.py `
  tests/test_scene_dialogue_evaluation.py tests/test_casting_director_integration.py `
  tests/test_director_asr_diagnostics.py -q -p no:cacheprovider
node --test tests/casting_director_ui.test.cjs
```

The broader relevant regression selection passed 1,011 Python checks and 30
subtests. After the final CLI preparation-retry adjustment, all 36 runner checks
passed separately. All 20 casting-panel UI checks passed. A real completed-run
`--generate` retry reused the forty saved samples and the identical evaluation,
without starting another synthesis worker. These are selected regression suites,
not a claim that every test in the repository was run.

## Models, contributions and remaining work

Three actual Claude CLI workers used canonical `claude-sonnet-5-5`: A implemented
source-registry repair, bounded cue carry and the scene-plan module; B implemented
the isolated runner and CPU tests; C implemented comparative QA, blind ballots,
human gates and CPU tests. All three hit the account's spend limit before final
handoff. Codex completed integration, corrected selection/receipt seams, added
scene-plan and preservation regressions, rendered/evaluated the actual pilot and
published the results. The CLI's list-cost receipts total approximately $5.13;
this does not measure the lead agent's cost or establish delegation savings.

The exact `claude-haiku-6` probe returned `unrecognized_model`; Haiku 6 was not
used, and no older model was substituted under that name. Sonnet interpreted
engineering requirements, not individual book scenes at runtime. Runtime
scene reasoning remains deterministic, cached, and limited by cue coverage.

Next work should follow listening feedback: improve source-grounded intention
interpretation, especially reassurance versus fast punctuation-derived delivery,
with one validated, cached model interpretation per scene when the requested
model is available. Preserve neutral narration, exact prose and stable identities.
Full-chapter directed production, speaker embeddings, long-form consistency and
IndexTTS comparison remain unvalidated. No additional model download was needed.
