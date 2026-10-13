# Character performance memory — October 12, 2026

Experimental, review-only, and isolated from the production default. Nothing here is evidence that
any audio sounds better, that any voice is calibrated, or that a human approved anything.

## Rationale

Whole-scene coaching (`scene_coaching.py`) reads one scene at a time. A character who recurs across a
book can then be directed with no memory of how the same character was delivered in earlier scenes and
chapters. This seam lets a model read the existing coaching requests of several chapter projects of the
same book once, propose a few bounded delivery habits per non-narrator role with exact source evidence,
and hands those notes to the unchanged scene-coaching step as context.

The memory is an **unapproved model interpretation**. It is not an approved `DeliveryProfile`, not an
author or human instruction, creates no Take, approves nothing, and never outranks a scene's own intent or
author-written cues. It is book-scoped and voice-independent: roles are matched only by the stable
`role_id` shared by the book's chapter projects (recorded in the root's architecture probe, which this module does not read). Names
are consistency checks, never a mapping, and no voice identifier is a persona. Narrators never receive a
profile or an instruction.

## API (`dubbing_service/app/audiobook/character_performance_memory.py`)

The module is pure: no model call, network call or file access.

| Function | Purpose |
| --- | --- |
| `build_memory_request(chapter_requests, *, book_sha256)` | One canonical `character-performance-memory-request-v1` from existing scene-coaching requests (a list, or `{project_id: request}`). Rows are copied by known field only. Refuses ambiguous roles, names or narration. |
| `capture_bundle(request, response, provenance)` | Binds a supplied `character-performance-memory-proposals-v1` response and its model provenance to the request by hash and validates it before a `character-performance-memory-bundle-v1` exists. |
| `validate_bundle(bundle)` | Re-proves schema, exact markers, request canonicality, request/response/memory hashes, model provenance, every evidence span against the embedded source, and exact role coverage. Returns an independent summary. |
| `enrich_coaching_request(request, memory_bundle, *, book_sha256)` | Deep copy of an existing four-field coaching request. Each scene gains `character_performance_memory` (memory hash plus notes for the profiled non-narrator roles present) and `constraints` gains one entry carrying the memory hash. `instruction` and `response_contract` are untouched, so response fields do not change. |
| `require_role_coverage(summary, roles)` | Early role check (used by the batch before it publishes anything for a chapter). |
| `enriched_memory_sha256(request)` | The single memory hash an enriched request carries, `None` for a plain one; inconsistent carriage is refused. |

Response profile (exactly the requested non-narrator roles, one each): `role_id`, `role_name`,
`delivery_habits` (1–3 strings, ≤120 chars), `contrast_notes` (≤240 chars), `confidence`
(`low`/`moderate`/`high`) and `evidence` (1–8 spans `{project_id, utterance_id, start, end, text}`).
At least one span must be spoken by the profiled role; narration may be additional evidence.

Delivery text reuses `scene_dialogue_plan.unsafe_instruction_terms` and
`performance_director.validate_native_instruction` (no speaker/identity/accent/pitch/age terms, no
numeric rate, no borrowed voice) and additionally refuses narration, persona, identity and voice-id words
and any character name. `role_name` is metadata and is not checked as delivery text. Duplicate roles,
habits and evidence spans, and identical habit sets shared by two roles, are refused. Provenance must be
`origin: model` with `model_requested` and `receipt_sha256`.

### Why the existing cache binds the memory

`scene_coaching.apply_coaching` keys each scene on `_sha(scene)` and on the hash of
`instruction`/`constraints`/`response_contract`. Enrichment changes both: the scene gains the notes and
the memory hash, and the constraints gain a line containing the memory hash. A different source, model,
provenance, policy or role set gives a different memory hash and therefore different cache keys; no
`scene_coaching` change is needed. The added constraint tells the coaching model to let scene intent and
author cues override a habit, to use at most one supported habit beside the scene intent, not to push
every line into one mood or a shout, to keep articulation and endings clear, and not to use numeric rates
or change accent, pitch, age, texture or identity.

## Batch integration (`coached_chapter_batch.py --performance-memory FILE`)

Optional. Without the option nothing changes: requests, `FROZEN_KEYS`, frozen hashes, completed retries
and receipts are byte-identical to before (asserted by test). With it:

* The bundle is fully validated, and checked against the EPUB's book hash, before any output directory
  exists. Role coverage (every non-narrator speech role needs a profile with the same name; narrators
  none) is checked per chapter before anything for that chapter is published. A fixture memory refuses
  non-fixture chapters.
* The exact supplied bytes are published once, privately, as `memory/performance-memory.json`
  (`pilot.publish`: identical retry is a no-op, a different one is a conflict).
* Each frozen request is the enriched request; the request index row also records
  `performance_memory_sha256`, so the existing frozen hash binds it without adding a frozen key. State
  gains a non-frozen `performance_memory` binding; applied receipts and chapter receipts record the
  memory hash and `approved: false`.
* `load_state` re-verifies the private copy, the binding and that every request carries the same hash. A
  removed binding, edited copy or edited request is refused.
* Resume (`prepare_batch`, `apply_bundles`, `generate_batch`) **holds** (`BatchError`, exit 2) when a bound
  batch gets no memory, a missing file, or a memory with a different hash; and when a memory-less
  directory is given one. Coaching captured for an earlier memory is refused by the existing
  "not captured for the frozen request" check. Use a new output directory for a changed memory.

The memory-bound flow therefore needs two directories: prepare without memory to obtain the requests,
build and capture the memory over them (plus any earlier chapters' requests), then prepare with
`--performance-memory` into a fresh output directory. Existing sources, Takes and cloned profiles are
still required by the batch; fresh chapters are handled by a separate module that calls this seam.

## Limitations

* No inference happens here. A real memory needs one model call by the root owner; its capture stays private.
* Evidence validation proves exact quotation, not that a habit is true or well supported.
* The identity-term guard is conservative (for example it refuses "sound…"); phrase habits around it.
* Names in delivery text are matched by whole-word, case-insensitive full role name only.
* Nothing is established about audio quality, acting improvement or calibrated voice behavior until a
  human listens; every fixture in the tests is synthetic and labeled.

## Verification

```
.venv/Scripts/python.exe -m pytest tests/test_character_performance_memory.py -q
.venv/Scripts/python.exe -m pytest tests/test_coached_chapter_batch.py tests/test_scene_coaching.py -q
```

Result when written: 107 passed for the first command; 199 passed for
`tests/test_character_performance_memory.py tests/test_coached_chapter_batch.py tests/test_scene_coaching.py`. All tests use labeled CPU synthetic fixtures; no GPU, model,
network, real book or live project is touched.
