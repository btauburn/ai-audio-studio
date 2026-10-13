# Fresh scene-coached review chapter — October 12, 2026

`scripts/render_fresh_coached_chapter.py` (implemented in
`dubbing_service/app/audiobook/fresh_coached_chapter.py`) is the review pipeline for a
chapter that has **approved cloned voices but no selected Takes**. It is deliberately
separate from [coached chapters](coached-chapters.md): `coached_chapter_batch` requires
existing selected, QA-passed neutral Takes and reuses them for narration. That contract is
unchanged and nothing here fakes or weakens it. In the fresh pipeline **every** speech
line — narration included — is newly synthesized for listening review.

It is review only. It selects no Take, creates no Take, records no human event or
approval, changes no production default and never writes the canonical project.
Listening quality is pending; nothing here claims improved audio
(`quality_improvement: not_established`).

## What Root provides (not done here)

Root prepares a canonical private derived chapter project with the existing import / fork
/ source helpers: `metadata.chapter_rerun` (authorized, recorded source revisions,
utterances, breaks, exclusions), the exact EPUB as a registered source revision, a resolved
attribution on every spoken line, and every speech role's approved clone files under
`casting/voices/<id>/`. This pipeline never imports, drafts or infers a voice, never
infers attribution and never edits that project.

## Contracts it enforces at prepare time

| Concern | Owner reused / rule |
|---|---|
| Exact EPUB chapter (SHA, title, member) | `epub_chapter_selection.select_epub_chapters` |
| Derived chapter / source / coverage / break / exclusion proof | `voice_lab_chapter_export._source_snapshot` (no `_take_snapshot`; no selected-Take requirement) |
| Whitespace-only completeness | every nonblank EPUB character must appear, in order, in the live segments |
| Resolved attribution, natural-speed unedited lines, Qwen Base route | each speech segment's attribution must be known and equal its role; `qwen3tts` |
| Cloned-only identities | `pilot.role_table` + `cloned_voice_policy.role_holds(mode="cloned_only")`; a malformed project voice policy fails closed; no provider presets, generated originals or synthetic references |
| Clone files | `pilot.verify_identity_files` on the canonical store *and* the isolated worker copy (also re-proves recorded-draft human selection) |
| Approval is evidence, not a status string | the role's casting assignment must be approved and unrejected, name the same heard-sample record and identity/conditioning hashes, match the voice revision (`voice_revision_for_role`), be backed by a positive human `use` event for that role/key/voice/revision/sample, and the heard sample's bytes must verify |
| Exact clone source asset | the acquisition receipt the clone's provenance names (`voice_lab/discovery/acquisitions/<id>/acquisition.json`) must exist and match its SHA-256 |
| Scenes, request | `scene_dialogue_plan.build_scene_dialogue_plan` / `verify_scene_dialogue_plan`; one whole-scene request that carries narration as context and selects every dialogue id exactly once |
| Frozen | every line (text, SHA, source anchor, role, voice, voice revision), role clone/approval/source-asset binding, the semantic source contract (`pilot.source_contract`), the request, base plan, isolated tree and boundary policy |

An absent attribution, absent/unlocked/non-clone voice, missing approval evidence, a
source or safety hold in the base plan, or a source that does not cover the EPUB prose
refuses preparation with the reason; nothing is published for a refused chapter.

## API

```python
from dubbing_service.app.audiobook import fresh_coached_chapter as fresh

state = fresh.prepare_chapter(args, *, plan_builder=None)   # CPU; status awaiting_coaching | held
state = fresh.apply_bundle(args, state)                     # CPU; status ready | held | awaiting_coaching
result = fresh.generate_chapter(args, state, worker_class=None, render_fn=None)   # explicit GPU lane
state = fresh.load_state(output_dir)                        # proves the frozen section and published inputs
```

`args` needs `epub`, `source_root`, `project`, `chapter` (int), `output_dir`, and optionally
`coaching_bundle`, `performance_memory`. All three functions raise
`FreshChapterError` for a fail-closed refusal (the render owner's `RuntimeError` for an
ambiguous dispatch is also reported as held by the CLI). `plan_builder`, `worker_class`
and `render_fn` are injection points for tests; the defaults are the existing plan owner
and `directors_vertical_slice.Worker` / `render`.

## CLI

```powershell
.\.venv\Scripts\python.exe scripts/render_fresh_coached_chapter.py `
  --epub C:\path\to\book.epub --source-root C:\path\to\projects `
  --project DERIVED_PROJECT_ID --chapter 11 --output-dir C:\path\to\private\out `
  [--performance-memory MEMORY.json] [--coaching-bundle BUNDLE.json] [--generate]
```

1. **Prepare** (no flags): writes `requests/request.json` and prints a one-line ASCII JSON
   summary. Give the request to the model reader outside this tool.
2. **Apply**: add `--coaching-bundle` with the captured response. It must equal the frozen
   request exactly, validate under `scene_coaching.validate_bundle` / `apply_coaching`
   (captured model provenance, exact evidence spans, one direction per dialogue id, none for
   narration), pass the pilot's direction guards (≤ 220 characters, no accent / pitch /
   gender / performer / identity wording, no numeric rate) and bind each direction to the
   frozen text and voice revision. It cannot clear a source, approval or safety hold.
3. **Generate**: add `--generate` (explicit, root-owned serial lane; idle GPU, port 8143
   free). Exit code `0` means prepared / ready / generated; `2` means held or refused.

### Performance memory

`--performance-memory FILE` is a captured `character_performance_memory` bundle. At
prepare time it is validated (`validate_bundle`), must be for this exact EPUB, must cover
exactly this chapter's non-narrator roles by `role_id` (`require_role_coverage`), and a
fixture memory is refused for a non-fixture chapter. The request is then enriched with
`enrich_coaching_request(request, bundle, book_sha256=<EPUB sha>)`; the result may add
context only — each scene's source fields, the response contract and the base constraints
must remain exactly as frozen — and must carry the memory hash. The exact bytes are
published once to `memory/performance-memory.json` and bound by hash. A bound chapter
needs the same file on every later invocation (it is never silently reused); a chapter
prepared without memory can never gain one. Memory is an unapproved model interpretation
and never touches narration.

## Retained existing cast (private review)

A fresh chapter may reuse the cloned cast already used for the same book without a per-role
human `use` approval on the new chapter project. This is **private review generation, not
promotion**: source-use permission for a recording is distinct from voice-quality or
role-fit approval, and nothing here creates, copies or implies a target approval, selection
or human event. The current approved-role mode above is unchanged and is used for every
role that has no declaration; the two modes may be mixed per role.

Root supplies two explicit private metadata blocks on the target project before
preparation (both are frozen, and both are covered by the semantic source contract):

* `metadata.existing_book_cast_sources`, keyed by canonical `role_id`, each
  `{project_id, role_id, book_sha256, voice_id, reference_audio_sha256, prompt_sha256,
  project_sha256, origin_project_sha256, origin_source_root, target_approval_created: false}`.
  `project_id` is the **provider**: the immutable original book-role project that holds the
  role's recorded clone.
* `metadata.owner_review_authorization`: `schema: fresh-chapter-owner-review-v1`,
  `authorized: true`, `scope: private_chapter_review`, `book_sha256`, `chapter_number`,
  `role_voice_bindings_sha256`, `author: user_request`, a non-empty `instruction` (the owner's
  request, recorded verbatim) and no quality-approval claim. The binding hash is
  `pilot.value_sha({role_id: {provider_project_id, voice_id, reference_audio_sha256,
  prompt_sha256}, ...})` over every declared role, so a provider, voice or hash cannot be
  switched without a new authorization.

For each declared role, prepare accepts only the same book SHA, the same canonical role id
and name, and a clone identical to the provider's (all identity fields, provenance and
files). It verifies, read only:

* the staged `source_root/<provider>/project.json` equals `project_sha256`, and the origin
  `origin_source_root/<provider>/project.json` is still that project (byte-identical, or an
  identical `pilot.source_contract` when only delivery-notification bookkeeping moved; both
  facts are recorded);
* the provider is bound to this book's source revision, passes `role_table` /
  `cloned_voice_policy` and the Qwen Base route;
* `pilot.verify_identity_files` on the **provider** context (staged, isolated and origin),
  which for a recorded draft clone requires the provider context's own current positive
  human selection — a withdrawn selection holds. Ordinary non-draft eligible references need
  no human event, matching the synthesis owner;
* the hash-bound acquisition receipt proves eligible recorded source-use
  (`performer_discovery.reference_eligibility`), `clone_ready` and passing technical QA, and
  the target's source asset and `reference.json` equal the provider's;
* the target's clone bytes as plain assets (no draft approval is consulted on the target).

The synthesis owner is unchanged: it still replays `_eligible_acquisition` and enforces
`recorded_audition_policy` against the project it is asked about. Each exact provider
context is therefore copied byte for byte to `output/projects/<provider>` with
`pilot.prepare_isolated_project` (its file tree is hashed into the state), beside the
retained target project in `output/projects/<target>`.

Rendering separates source from provider. Requests are made with `provider_project_id`, and
`pilot.verify_sample` is given a render-line copy with that project id so the expected
request SHA is the real one. The source line (project id, anchor, role, text hash, scene,
memory bindings) stays the new chapter's. Every sample receipt, source-to-audio row, audit,
result and chapter receipt records `source_project_id`, `provider_project_id`,
`cast_mode`, the owner authorization, `target_approvals_created: 0`,
`role_fit_approval: not_established`, `human_listening: pending`, and the original
source-use lineage (origin hashes, acquisition, eligible source-use, original draft
selection when one exists). Staged, isolated and origin drift, a changed declaration or
authorization and an unavailable origin are re-checked before apply, before any worker and
on every retry, including a completed one.

## Output layout

```
state.json                         frozen section + mutable status/coaching ledger
requests/request.json              the whole-scene coaching request (hash bound)
memory/performance-memory.json     optional exact memory capture
plans/base.json                    base scene plan (unapproved recommendation)
plans/coached.json                 coached plan (published once, immutable)
coaching/bundle.json               the captured model bundle that was applied
applied.json                       hashes, model provenance, memory binding, direction digest
projects/<project>/                isolated worker copy of metadata + clone files (+ approval evidence)
projects/<provider>/               exact original book-role contexts (retained-cast mode only)
samples/<utterance>.wav            every newly synthesized line + the render owner's raw receipts
receipts/<utterance>.json          verified sample receipts (fresh schema, clone binding, QA, coaching)
runtime-pin.json                   worker runtime pinned on first dispatch
forwarding-audit.json              exact forwarding / narration audit
source-checks/*.json               live source verification records
chapter/chapter-NNN.{wav,mp3,json} review audio and the exclusive chapter receipt
result.json
```

Schemas are `fresh-coached-chapter-*-v1`; they are not the selected-Take schemas. Every
receipt, the audit, the result and the state carry `review_only: true`,
`human_review: pending`, `production_ready: false`, `default_changed: false`,
`approvals_created: false`, `takes_created: false`, `selections_created: false`,
`human_events_created: false` and `quality_improvement: not_established`; fixtures also
carry a visible fixture label and are excluded from readiness claims.

## Generation

One serial `Worker(directed=True)` renders **all** speech in source order:

* **Narration** is dispatched with `native_instruction: None` on the same approved Base
  clone route (the same directed wrapper, no instruction). It is *newly generated neutral
  review audio*, not a reused, selected or approved Take. Every sample receipt, every
  `source_to_audio` row, the audit and the result record this explicitly
  (`audio_provenance: fresh_neutral_review_synthesis`, `narration.reused_approved_take:
  false`, `selected_take_used: false`).
* **Dialogue** forwards its captured instruction. `pilot.verify_sample` checks exact text,
  voice, request hash, runtime pin, the forwarded instruction digest and policy, and that
  there was no fallback. A worker that reports another identity or prompt is refused.
* Every clip passes blocking technical QA (`qa.qa_take_audio`) and its raw render receipt
  hash is recorded.

Assembly follows the measured boundary policy (`performance_boundaries`): the leading and
trailing silence already inside the clips is measured and deducted from each boundary's
target (`added = max(0, target − existing)`); recorded source separators raise a boundary
to 1.25 s; nothing is trimmed and no prose is rewritten. WAV, MP3, loudness and QA use the
shared `voice_lab_render` / `qa` owners. The chapter receipt keeps the complete
`source_to_audio` map (source anchor, text, role, cloned reference with source asset and
approval binding, audio hash, timing, boundary arithmetic, raw receipt, coaching reading)
plus `scene_break` / `excluded` rows, and an `accounting` block whose speech seconds plus
added padding equal the master duration.

## Retry and resume

* Prepare, apply and generate are idempotent. Identical inputs republish identical bytes;
  a different bundle after application is reported as `held` and the published plan is
  kept; a different byte for any published artifact is a conflict.
* Before any worker starts, generation re-derives directions from the immutable published
  files and re-verifies the canonical and isolated clone files, the approval evidence
  (assignment, human event, heard sample), the source asset, the semantic source contract
  (only delivery notification counters may differ), the derived-chapter proof and the
  memory binding. Drift refuses the run.
* A completed render is verified byte-for-byte; if nothing is missing no worker is created.
  A partial run resumes only the **missing** samples with one worker; nothing already
  recorded is resent. A completed chapter is verified, never re-assembled: master / MP3
  bytes and decoded durations, clone bindings, every line's audio and raw receipt.
* Any unresolved pending dispatch record without a receipt is ambiguous: the run holds
  **before** a worker exists and never resends. The pilot's single proven pre-inference
  approval-file refusal is resolved exactly as in the pilot.
* A runtime differing from the pin, a profile other than `faster-base-032-direction-x1`,
  unconfirmed forwarding, a fallback or a different voice is a refusal. No preset is ever
  substituted for a clone. A clip that fails technical QA holds the run; the saved clip is
  verified (not regenerated) on retry.

## Limitations

* Narration is fresh synthesis; it will differ from any earlier approved narrator Take and
  carries no approval. This pipeline does not decide which narration to keep.
* It does not import or fork a chapter, draft voices, attribute speakers or infer
  approvals; Root's prepared project is the only input it trusts, and only after the
  checks above.
* The approval check follows the recorded-draft owner's current conventions (assignment,
  positive human `use` event, revision, heard-sample bytes). If those conventions change,
  update `_casting_evidence`; do not weaken it to a status string. The retained-cast mode never
  substitutes for it: it applies only to roles with an explicit owner-authorized declaration.
* Retained-cast mode proves permission to use the recorded source and the original context's
  own selection for draft clones. It says nothing about voice quality or role fit; those
  stay pending listening review.
* No audio-quality, listening or approval claim is made. The tests use synthetic tone audio
  and a fake worker only.
* Private books, voices and audio stay outside Git.

## Verification

```powershell
.venv/Scripts/python.exe -m pytest tests/test_fresh_coached_chapter.py -q
```

The tests use only labeled synthetic fixtures: invented prose, tone-signal audio, fake
clone files, fake approval and acquisition records, a synthetic bundle standing in for a
model response, a synthetic performance-memory bundle and a fake worker. No real book,
voice, model, GPU, network or live project is touched. They cover exact source and EPUB
completeness, cloned-only and approval-evidence rejection, absent attribution, no Take
creation, narration actually dispatched without an instruction, dialogue forwarding,
role / source / approval / clone drift, immutable outputs, ambiguous dispatch, failed QA,
resume with no extra worker and no reuse, source separators and accounting, memory
binding, and the CLI.
