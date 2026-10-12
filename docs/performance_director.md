# Performance Director — direction storage and experimental runtime (M4–M5)

This document covers the part of the Performance Director that M4 implements:
project-owned direction storage, the direction resolver, and the single
fingerprint hook. It is written for the person operating this repository, not as
a specification.

M4 changes **no behaviour for any existing project**. Every project today has no
`performance_director` key in its `project.json`, and with no index the code
reads no performance storage at all and computes byte-identical synthesis
requirements. The switch that turns any of this on is a reviewed decision, not
something the code does by itself.

## What M4 adds

| Piece | Where | State |
|---|---|---|
| Immutable direction stores | `performance/profiles/`, `performance/scenes/`, `performance/directions/`, `performance/policies/` under a project root | Written only by the functions in `performance_director.py`; every write is content-addressed and exclusive |
| Index writer with an expected revision | `performance_director.commit_index` | Compares against a freshly loaded index; never trusts the caller's number |
| Pure resolver | `performance_director.resolve_directions` | Reads no file and calls no model |
| Active-set loader | `performance_director._active_direction_set` | Loads once per (project, set hash) with a bounded process cache, and still notices a changed file |
| The one fingerprint hook | `pipeline.requirement_for` → `performance_director.requirement_direction` | Adds the `performance_direction` component only when something is really forwarded |

Not in M4: the experimental worker adapter and server, profile registration,
production exclusion (M5), any live pilot, any delivery or publication, and any
change to `renderer.py`, `providers/tts.py` or `service/`.

## The off switch, and what "off" means

```json
{"performance_director": {"schema_version": "performance-director-index-v1",
                          "enabled": false, "revision": 0, "pilot": null,
                          "active": {"profiles": {}, "scene_plans": {},
                                     "direction_set": null, "boundary_policy": null},
                          "history": []}}
```

- **No key** or **`enabled: false`** — the hot path returns immediately. No
  directory is created, no file is read, and `requirement_for` produces exactly
  the value it produced before M4. `tests/test_performance_fingerprint_compat.py`
  proves this against the preserved original `fingerprint.py` bytes.
- **`enabled: true` with a `direction_set` pointer** — the pointer is resolved,
  the set is read, and the line's direction decides whether a component is added.
- **Anything wrong** — a missing file, unreadable JSON, a hash that does not
  match its name, a set belonging to another project, a held line, a stale text,
  role or voice binding, or a runtime that differs from the one that resolved the
  direction — is a typed refusal. There is no fallback to "no direction", because
  that would silently compute a requirement the operator did not ask for.

## Storage layout and identity

```
<project root>/performance/profiles/<role_id>/<sha256>.json
<project root>/performance/scenes/<scene_id>/<sha256>.json
<project root>/performance/directions/<sha256>.json
<project root>/performance/policies/<sha256>.json
```

The file name **is** the content hash, computed with the accepted M1 rule
(`performance_director.content_sha256`, which excludes `created_at` and the
record's own hash). Because identity excludes timestamps, storing the same
content twice is verified reuse rather than a byte conflict; storing *different*
content under an existing hash is refused, never overwritten. A failed publish
leaves its staging file (`.<name>.<uuid>.stage`) on disk as evidence.

`role_id` and `scene_id` become path components, so they are validated: no
separators, no control characters, no `.`/`..`, no leading or trailing padding,
and no trailing dot. Every resolved path is checked to stay inside the canonical
project root, including the case where `performance/` is a symlink pointing
outside it.

An index may only point at an artifact that exists, parses, matches its own
schema and matches its own content hash. `commit_index` verifies every pointer
before it saves anything, so an index can never name a nonexistent artifact.

## Writing the index

```python
performance_director.commit_index(project_id, expected_revision, change,
                                  actor="who", event_id="evt-…")
```

- Holds the projects-root writer lock and the project metadata fence, so two
  writers cannot both win.
- Re-reads the index and compares `expected_revision` with the revision on disk.
  A mismatch is `plan_stale`.
- Applies `change` to a copy. The callback may not rewrite the schema version,
  the revision or the history; doing so is refused and nothing is saved.
- Increments the revision exactly once, appends a bounded history entry
  (`revision`, `actor`, `event_id`, `at`, `previous`, `active`, `enabled`,
  `pilot`, `payload_sha256`), and saves atomically through `save_project`.
- Repeating an `event_id` whose resulting payload is identical returns the
  verified prior result without a second increment. The same `event_id` with a
  different payload is `direction_conflict`.
- An `actor` string is audit attribution. It is not human approval; the review
  gate for a live pilot is a separate owner.

Rollback is clearing the pointer. Fingerprints return to their previous values,
baseline Takes become current again, and directed Takes remain as history.

## Resolution precedence

`resolve_directions(profile, plan, capability, *, policy)` is pure. It returns one
`UtteranceDirection` per scene utterance, and it never blends or truncates.

Highest precedence first:

1. **Source and voice safeguards** — a performer reference, a voice change or a
   numeric rate promise is refused with `direction_conflict`. Narrator
   preservation is enforced here and cannot be widened by policy
   (`DirectionPolicy.narrator_forwarding` is pinned to `false`).
2. **A human line correction** — `policy.line_overrides[<utterance_id>]`.
3. **A human scene correction** — a plan turn whose `confidence` is `manual`.
4. **The approved profile** — `DeliveryProfile.habits`, one per dimension.
5. **An inferred intention** — a `deterministic` or `model` plan turn. It is
   recorded in the precedence trace and never becomes explicit or approved.
6. **Baseline** — no instruction at all.

Each level that contributed is named in the row's `precedence_trace`, and the row
records the profile, scene-plan and capability hashes plus the resolver version
in `sources`.

Exactly one disposition (`forward_native`, `assembly_timing`, `advisory`,
`unsupported`, `hold`) is returned per requested control, and each control is
named at most once.

### The policy

```json
{"schema_version": "direction-policy-v1",
 "max_instruction_chars": 240, "max_priorities": 3,
 "forbidden_terms": [], "narrator_forwarding": false,
 "allow_forward_native": true,
 "line_overrides": {"<utterance_id>": {"delivery": ["measured"],
                                       "intention": "deflect", "pace_hint": "quick",
                                       "seed_salt": 0, "require_forward": false}}}
```

`max_instruction_chars` may be lower than 240 but never higher, and
`max_priorities` may be lower than three but never higher. A policy cannot make
the resolver more permissive than the M1 contracts.

### Holds instead of guesses

| Situation | Result |
|---|---|
| The composed instruction exceeds the policy limit | `hold`; it is never truncated |
| More priorities than the policy allows | `hold` |
| Two contradictory priorities (a bounded antonym pair) | `hold`; never blended |
| A control marked `require_forward` that this runtime cannot forward | `hold` with a `capability_unsupported` reason |
| A control this runtime does not support and that was not required | `unsupported` |

A held line is refused at the hook, so nothing is rendered from it.

## Which controls reach the runtime

The capability record comes from the M1 owner
(`provider_capabilities.performance_capabilities`) for the project's own runtime
(`project.metadata["requested_runtime"]`, default `faster-base-032`).

- On the default profile the native instruction is `available_not_forwarded`, so
  every native control resolves to `advisory`, no instruction is set, and
  **nothing enters a fingerprint**.
- Only the experimental profile with full supported reference conditioning
  reports `forwarded_experimental` and can resolve `forward_native`. M4 neither
  registers nor launches it.
- The M1 probe and default profiles keep seed control unsupported. M5's appended
  `experimental_direction_capabilities` reports explicit salt support only for
  the hashed direction adapter with full ICL conditioning. The effective seed is
  `(identity_text_seed + explicit_salt) % 2**32`. Narrators never forward salts.
- Legacy `emotion_text`, `style_instruction`, intensity, emphasis, role delivery
  and automatic labels are never converted into a native instruction. They are
  recorded as requests with an `unsupported` disposition, and the resolver reads
  neither `Segment.emotion_text` nor `Role.style_instruction`.

## The fingerprint component

`fingerprint.SynthesisRequirement.performance_direction` is an optional component
holding `direction_component(resolved)` — the SHA-256 of the canonical
`{control_policy, native_instruction, seed_salt}`.

It is serialized **only when present**, following the existing `volume`
precedent, so adding it cannot stale a Take that no instruction reached:

- instruction `null` and salt `0` → the component is absent and the requirement
  is unchanged;
- an instruction or a non-zero salt → the component appears and that line's
  fingerprint changes.

`pipeline.requirement_for` is the single funnel. `casting_review` (frozen
manifests), `voice_lab_chapter` (chapter forks), `book_creator` (bindings) and
the M3 pilot assembler all reach the component automatically because they all
call it.

### The one exception to the metadata rule

`Project.metadata` is documented as never participating in a synthesis
fingerprint. The active direction pointer lives in that dictionary and
`requirement_for` reads it, so this one key is a deliberate exception. The model
docstring says so, and so does the hook's docstring. If you add another piece of
metadata that changes audio, that is a different decision and needs its own note.

## Known limits

- No live grant, reviewer or delivery bridge. Nothing is published, notified or
  approved by this code.
- No model, worker, GPU or network call anywhere in M4, including in its tests.
- The baseline strict-casting and strict-runtime gaps are unchanged and visible.
- The synthetic fixtures in `tests/test_performance_*.py` are not
  production-ready inputs.
- `boundary_policy` pointers are validated as content-addressed artifacts under
  `performance/policies/`; no M4 code compiles or consumes a boundary policy.

## M5 experimental runtime

The registered `faster-base-032-direction-x1` profile uses the existing Faster
0.3.2 environment and the two new worker files. Registration does not launch a
worker or change the default. `project_profile` permits it only on an isolated
review-only Performance Director pilot with the existing pilot index and ledger.

An injected `DirectedQwen3TTSProvider` uses the canonical M4 active direction by
default, preserves the parent's project/rendition binding, checks all seven
worker source hashes, and verifies every receipt field. No direction follows
the parent path; held, stale or unsupported directions fail before dispatch.
Instruction tokens enter the parent's unchanged capacity check. The preparation
wrapper exists only while the parent serial lock is owned and is restored before
that lock is released. A rejected competing request cannot erase owner metadata.

`casting_assembly`, `autocast_book` and `book_creator` refuse experimental or
unknown runtime Takes, including nested renderer metadata. Voice Lab chapter
review keeps all existing cast/source/QA checks and labels experimental receipts
with `experimental_runtime: true`; the label participates in review identity.

The active-set cache checks bytes even when size/mtime are unchanged and returns
defensive copies. Caller `performance_direction` overrides are refused, including
null overrides. Canonical scene/alias/profile/voice/runtime proofs remain checked.

CPU tests execute the actual baseline wrapper against fake model/HTTP boundaries.
They establish contracts, not speech quality or GPU equivalence. Live Stage B,
instruction approval, launch window, delivery, listening verdicts and M7 remain
separate decisions. M6 can add advisory measurement tools before real B exists;
it cannot manufacture that listening evidence.

## M6 delivery outcome measurement and review tools

M6 is CPU-only, additive and advisory. It measures one already-rendered Take,
compares two measured Takes of the same line, and reports what the accepted M3
gate already says about a pilot's pairs. It synthesizes nothing, loads no model
or ASR, uses no GPU, writes no project, records no review and produces no
verdict. A real Stage B listening review has not happened, so every review status
it can report today is `review_pending`.

| Piece | Function | What it does |
|---|---|---|
| Per-Take outcome | `performance_boundaries.measure_delivery_take` | Extends a copy of the M2 `edge-measurement-v1` record with a `pd-delivery-outcome-1` delivery block |
| Same-line comparison | `performance_boundaries.compare_delivery_outcomes` | Reports deltas, flags and unknowns for a control/directed pair |
| Read-only review | `performance_boundaries.delivery_review_summary` | Returns the M3 `improvement_gate` result for every pair, plus an aggregate |

### The delivery record

```python
measure_delivery_take(audio_path, expected_sha256, *, text, scratch_root,
                      lexical_receipt=None, generation=None)
```

The result is the M2 record plus `delivery`, and the M2 record is never
rewritten: the detector's own identity stays verbatim under
`delivery.edge_identity.content_sha256`, while the record's top-level
`content_sha256` is recomputed through the same canonical owner over the
enriched content. A caller that only calls `measure_take` sees exactly what it
saw before M6.

The delivery block carries the exact decoded duration, sample rate, frame count
and channel count; the counted-text-word rate over total elapsed duration; an
explicitly named articulation-rate **signal proxy** from the acoustic owner; the
pause count, rate, mean length and fraction; median pitch, range and the p10/p90
statistics; the M2 onset step and peak; RMS/peak/dynamic-range loudness; the
acoustic owner's schema, method, parameters, package versions, notes and analysed
scope; the lexical outcome; and the generation guard evidence. Every number is a
block with `value`, `units`, `method`, `sample_count`, `scope` and, when
unmeasurable, a `reason`. `delivery.unknowns` lists every metric that stayed
null.

Two rates are deliberately not interchangeable:

- `elapsed_text_rate.words_per_minute` — the caller's counted expected-text
  tokens divided by the Take's whole decoded duration.
- `articulation_rate_proxy.syllables_per_minute` — the accepted
  `service.voice_acoustics` signal proxy, named as such and never presented as a
  lexical measurement.

The owner's 30-second analysis limit, its 16 kHz analysis rate and its mean
downmix of multi-channel input are recorded in `delivery.acoustics` and in every
metric's `scope`. A Take longer than 30 seconds is measured in full for frames
and duration, and analysed only over the first 30 seconds, with both numbers
reported.

### The scratch snapshot

`scratch_root` must be an **existing directory the caller owns**; there is no
default, no hidden temporary and no projects root fallback. The exact approved
bytes are read once, verified against `expected_sha256`, and captured
exclusively (`mode 'xb'`) as
`<scratch_root>/pd-delivery-snapshots/<audio_sha256>.wav`. Both the M2 detector
and the acoustic owner read that snapshot, and its bytes are re-verified after
measurement; a change is a refusal, not a quietly attributed result. An existing
snapshot named by the same digest is reused only after its bytes are verified
identical, and nothing is ever overwritten, moved or deleted.

### Lexical receipts

`lexical_receipt` is an optional, bounded, caller-supplied record from the
**existing** `reference-asr-v1` owner (`reference_transcripts.py`). Supplying it
grants no permission to run that owner, which M6 never does.

A receipt is refused rather than applied when it is malformed or mismatched:
wrong schema, an unrecognised status, `machine_generated` not `true`,
`human_verified` not `false`, an `audio_sha256` that does not bind to this Take's
exact bytes, a `text_sha256` that disagrees with its own text, a duration that
disagrees with the decoded Take, more than 200 segments or 1000 words,
out-of-order or out-of-bounds timestamps, a probability above 1, or any
non-finite number anywhere.

A usable draft receipt adds a deterministic token-normalised word error rate, the
expected final word's presence, position and probability, the lowest word
probability and the alignment's timing proxies. A missing, `unavailable` or
`failed` alignment leaves all of those null with the owner's own reason. ASR
confidence is machine evidence and is never reviewer authentication.

### The comparison and its threshold

```python
compare_delivery_outcomes(control, directed, *, wer_threshold=0.1)
```

Both records must be valid enriched records of **different audio** measured
against the **same expected text** by the same module, detector, acoustic schema,
method and parameters. Anything else is refused rather than blended.

`0.1` is the local initial advisory threshold convention: P12 names a threshold
without fixing its value, so the value lives here, is reported in every
comparison as `flags.wer_threshold` and `flags.threshold_basis`, and is **not**
Brian's listening criterion. A directed Take whose word error rate is above it
(so exactly `0.1` is not flagged) or that is missing its final expected word sets
`flags.for_listening`. Missing measurements stay unknown: their delta is `null`
with a reason, never an optimistic zero, and unknown evidence sets
`flags.insufficient_evidence` and still flags for listening. The comparison
reports `rejects_take: false`, `triggers_retry: false`, `preferred_arm: null` and
`grants_scope: false` on every result.

### The read-only review summary

```python
delivery_review_summary(pilot_project_id, pair_ids)
```

Every pair is resolved by the accepted M3 `performance_pilot.improvement_gate`
and returned **verbatim** — this function reproduces no part of that acceptance
predicate, so the exact-edition, artifact-hash, full-heard-coverage,
authenticated-preference, stated-benefit and no-new-defect requirements stay
entirely with their owner. The aggregate groups pairs into `supported`,
`review_pending`, `inconclusive`, `blocked` and `not_improved` and reports one
`status`. No pair, no review, no resolved edition, an unrecognised status, or a
pair the owner refuses all stay `review_pending`; only a pair the owner marks
`supported` and `supports_improvement` counts as support, so the aggregate can
never be vacuously supported. `requires_separate_stage_c_approval` is always
`true`. The summary reads no blind key, writes no review, rating, favourite,
approval or ledger entry, and reports `blind_key_read: false`,
`records_written: false` and `grants_scope: false`.

### Known limits

- No live Stage B audio, instruction approval, launch window or listening verdict
  exists. Everything above is a tool; the human gate is still closed.
- The synthetic fixtures in `tests/test_performance_delivery_outcomes.py` are
  generated PCM bursts, not speech, and its lexical receipts are fake records
  shaped like the owner's output: no ASR model runs and nothing there is evidence
  about a real transcript or about how a Take sounds.
- Numbers are advisory. Whether a directed line is better is a listening verdict,
  and M6 cannot produce one.
- No model, worker, GPU, network or service call exists anywhere in M6.

### Independent review clarifications

Source, parameter and package identities are refreshed for each measurement.
Draft ASR receipts require recorded model, pipeline, settings, dependency and
duration provenance, exact text/segment/word agreement, and a mono 3–30 second
crop. A draft without word timestamps leaves WER and final-word outcomes unknown.
Final-word presence is a conservative proxy: the terminal aligned token must match
and the alignment must contain every expected occurrence of that token. It can
flag an extra trailing recognition token for listening; it does not establish
audible truncation.

Acoustic sample counts describe the owner's 16 kHz analysed signal. Peak and RMS
describe mean-downmixed, DC-corrected, resampled audio. Each delta carries units,
method, both sample counts and both scopes; whole-Take versus truncated-prefix
acoustic coverage produces an unknown delta. Comparison validates the original
edge digest, all audio/text bindings, acoustic source and package identities.
Scratch snapshot paths must remain inside the explicit scratch root.

The summary delegates to the M3 gate. That owner reads its private pair binding
and may return a resolved arm in its gate; the wrapper neither directly reads nor
exports the private key and returns owner gates verbatim. No synthetic authenticated
test-double result is a live listening verdict.

