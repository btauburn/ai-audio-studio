> **October 4 follow-up repair:** exact fingerprinted reference-source bytes
> were restored after a byte-only whitespace cleanup invalidated frozen QC
> evidence. See [source fingerprint compatibility](reference-source-fingerprint-compatibility.md).
> Default selection, immutable voices, receipts and source-policy guards remain unchanged.

# Audiobook implementation checkpoint — 2026-10-04

This is the current source architecture for Alfred's review. It supersedes the
September 25 v1.2 snapshot and the former native-current/auto production preference.
The checkpoint starts from `e1f18cb8b313d1748a78ca90b26b4f5ddc156df7` on `main`.
Existing implementation, dependencies, tests, schemas, UI and operating scripts
are captured together; private runtime databases, audio, model weights and secrets
remain external to Git under the existing ignore policy.

## Evidence labels

- **IMPLEMENTED:** source present in this checkpoint, including the default below.
- **TESTED:** explicitly identified automated checks or dated runtime receipts.
- **USER-APPROVED DEFAULT:** Brian's October 4 instruction promotes Faster Qwen.
  This authorizes promotion; it does not establish that all audio was listened to.
- **PROPOSED:** future work without corresponding source or verification. No
  proposed feature is required to reproduce the current implementation.

## Canonical Qwen Base default

The exact implemented default is **`faster-base-032`**, backend **`qwen3tts`**.
It launches `service/faster_qwen3tts_base_server.py` from the existing
`environments/faster_qwen_base032/.venv`, on port 8003, CUDA:0, BF16, SDPA.
The Base adapter retains its version checks: Faster 0.3.2, qwen-tts 0.1.1,
Transformers 4.57.3 and Torch 2.10.0+cu130. Native VoiceDesign reference generation
remains available inside this wrapper. No dependency/model upgrade is part of the
promotion. Consult that environment's README, adapter and model manifests for
installation and hash verification.

The source of the default is `DEFAULT_QWEN_BASE_PROFILE` in
`dubbing_service/app/core/tts_backends.py`. It is consumed by:

| Surface | Default owner |
| --- | --- |
| Project rendering, cloning and readiness | `audiobook/project_qwen_runtime.py:project_profile` |
| Worker launch with an omitted profile | `core/tts_backends.py:build_backend_command` |
| Settings/service launch with an omitted Base profile | `routers/settings.py:_start_tts_service_exclusive` |
| New Base audition intents, frozen before their identity hash | `audiobook/voice_lab_audition.py:_bound_intent` |

Paths in the table are relative to `dubbing_service/app/`. Explicit
`metadata.requested_runtime = "native-current"` remains supported. Unknown or
incompatible worker identity is refused; no native/Faster fallback is permitted.
The existing CLI requires an explicitly named backend for engine rendering and
uses the same project policy; its safe fixture default remains a fixture.

**CustomVoice is separate.** `qwen-cudagraph`, backend `qwen3tts_cv`, port 8004,
has its own worker/environment. The existing editor setting
`tts_params.qwen3_execution_profile` controls CustomVoice and retains its prior
semantics. It must not be populated with the Base-only profile.

## Responsibility and persistence map

| Responsibility | Implementation and durable contract |
| --- | --- |
| Book intake and attribution | `adapter.py`, `autocast_import.py`, `book_creator.py`, `character_research.py`; preserved source bytes/revision, ordered chapters and attributed segments |
| Character evidence and casting direction | `casting_profiles.py`, `performer_catalog.py`, `performer_reference.py`, `service/voice_acoustics.py`, public pinned catalogs; explicit/supported/inferred/unknown facts remain distinct |
| Human casting | `casting_review.py`, `casting_inference.py`, `casting_anchors.py`; durable run/round/replacement intents and revision-bound approval events |
| Clone/design identity | `voice_lab.py`, `character_identity.py`, `reference_quality.py`, `reference_portability.py`; immutable reference/prompt/sample hashes and provenance |
| Library and listening | `voice_lab_library.py`, `voice_lab_audition.py`, Voice Lab/library routes and `webui/voice-lab.*`; versioned identities, independent favorites/ratings/assignments |
| Worker correctness | `runtime_binding.py`, `project_qwen_runtime.py`, `voice_lab_runtime.py`, provider TTS and worker lifecycle registry; actual profile/model/runtime checks |
| Render/QA/Take commit | `renderer.py`, `pipeline.py`, `qa.py`, `persistence.py`, Take services; immutable Takes, fail-closed QA and optimistic concurrency |
| Recovery and metadata | `storage/atomic.py`, `storage/segmentation_commit.py`, metadata locks and recovery markers; bounded Windows replacement retry and rollback |
| Review delivery | `audiobook/delivery/`, verified export receipts and private review UI; projection of existing verified audio, separate feedback, no synthesis on browsing |

Bare module names in this table are under `dubbing_service/app/audiobook/`.
`storage/`, `routers/`, `providers/`, and `models/` are under `dubbing_service/app/`;
`webui/` is under `dubbing_service/`, and `service/` is repository-relative.
Schemas live in `dubbing_service/app/models/{project,character,voice_lab}.py`; the main app registers
the new routes and bounded maintenance/recovery owners. These untracked modules
were real dependencies of the previously modified tracked files and belong in
the same implementation checkpoint.

## Casting changes captured since the previous checkpoint

AutoCast import retains the pinned isolated speaker analyzer, source evidence and
strict unresolved-attribution holds. A durable source-revision/pipeline run key
starts eligible casting once; opening, refreshing or playing Voice Lab is read-only.
Pipeline revisions retain superseded rounds/history and never overturn approved
assignments. Existing identities and completed auditions are reused.

Casting now provides character rounds, shared versioned comparison scripts,
bounded auditions, measured sections, raw human directions and revision-bound
feedback. Stars are explicit. Rejection creates one durable replacement intent;
unrated rounds wait. Ratings save serially per character so later edits are not
dropped. Approval/assignment, library rating, favorite and heard-sample feedback
remain separate. Historical feedback from stale tabs is retained without applying
it as current approval. Duplicate requests and polling do not synthesize again.

Recorded-reference acquisition, source-quality checks, immutable processing
recipes, reference-clone prompt reuse, original VoiceDesign, keeper-first library
reuse, source-backed performer evidence and delivery/browser persistence are
included. Source eligibility, automated acoustics and explicit human casting
selection retain separate meanings. These features supersede the earlier
original-VoiceDesign-only/no-casting-GUI description.

## Initial generic voice-library baseline

The accepted current cast is six exact existing versions. See
[`generic-voice-baseline.json`](generic-voice-baseline.json) for immutable identity,
version, actual model/runtime, selection evidence, rating state, provenance,
asset paths and SHA-256 values. Snape is **p254_021_mic1**, replacing the earlier
unselected p232 draft. Dumbledore is an original synthetic VoiceDesign reference
reused through Base, not a clone of a recorded Richard Harris voice.

| Current role | Preserved source identity | Existing stars |
| --- | --- | --- |
| Narrator | David Attenborough recorded reference | 5 |
| Harry | VCTK p226_023_mic1 recorded reference | null |
| Hermione | Emma Watson processed recorded reference | null |
| Ron | VCTK p273_023_mic1 recorded reference | null |
| Dumbledore | Rowan, original synthetic reference | null |
| Snape | VCTK p254_021_mic1 recorded reference | null |

All six have explicit Brian selection events; null stars were preserved. Historical
names containing “review pending,” acquisition `production_approved=false`, and
source-fit review flags were preserved rather than rewritten. Hermione's source
processing caveat and the VCTK draft policies remain visible.

The immutable publication API registered those exact versions in the existing
global library: one idempotent existing narrator plus five appended versions.
The shelf now has 42 rows / 43 versions. Its pre-existing 29 rated/promoted
keepers remain separately identified: 28 synthetic versions and Attenborough.
No new rating, favorite, preferred-version promotion or audio generation occurred.
Original source JSON and selected assets, all prior global rows, versions and
preferences were checked unchanged.

Runtime stores are intentionally outside source control:

- Global metadata/assets: `dubbing_service/voices.json`, `dubbing_service/voices/`.
- Source accepted cast: `../runtime_evidence/2026-09-28-autocast-elf-acceptance/projects/80b99ad8/`.
- Checksummed preservation bundle: `../runtime_evidence/2026-10-04-generic-voice-baseline/`.
  It includes before/after global metadata, selected source metadata, original
  acquisition assets, immutable library assets and a byte-verified backup manifest.

Attenborough and Hermione portable recorded-reference bundles passed preparation.
Harry, Ron and Snape remain held by the existing exact narration-delivery replay
guard; their source-store/preservation-bundle dependencies are recorded. No guard
was relaxed and no reference was regenerated. Dumbledore requires no recorded-source
portability bundle. These are local preserved assets, not an off-machine backup or
a self-contained binary Git distribution.

## Compatibility and safety

Existing schema defaults permit older projects to load without replacing data.
Normal transaction recovery remains bounded and explicitly governed. The checkpoint
does not bulk-migrate private projects, rewrite Take history, alter existing
runtime fingerprints or rehash stored audition intents.

New Base requests freeze Faster before their intent digest. Already-stored requests
reuse their original receipt and binding. A native-bound render cannot append Faster
Takes to the same bound job: select native explicitly to resume that rendition,
or intentionally create a separate Faster rendition. Source hashes are part of
renderer/cache identity; changed bytes or checkout line endings can invalidate an
old resume binding. Preserve old receipts and select a deliberate new rendition
instead of rewriting the binding to evade that check.

QA failure, unknown remote outcomes and incompatible/missing identity remain holds.
Concurrent Take commits retain the latest unrelated edits and reject conflicting
state. Import now uses the existing bounded replacement helper for promotion and
rollback: a brief Windows reader can be retried without weakening permanent-error
failure or transaction cleanup/recovery. Actual Windows reader-lock regressions
cover both project and segment metadata.

## Verification and remaining evidence

Current automated verification receipts are recorded in the Alfred handoff. The
focused default/casting/persistence/render checks passed 143 tests; the import,
metadata retry and voice consistency checks passed 44. The JavaScript suite passed
203; the existing isolated Faster environment ran all nine adapter unittest checks.
The source guard and offline lock consistency check passed. No package/environment
upgrade or model generation was needed. The final available CPU suite passed: **2273 passed, 2 skipped, 3 warnings, 105 subtests passed in 855.81s (0:14:15)**.

The unrestricted main-environment pytest collection is blocked in
`tests/test_indextts_audio_safety.py` and `tests/test_service_server_api.py`
by missing `torch` / `librosa`; those modules are explicitly excluded from
the CPU suite. Reproduce the available CPU and isolated adapter checks:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/test_indextts_audio_safety.py --ignore=tests/test_service_server_api.py -p no:cacheprovider
.\environments\faster_qwen_base032\.venv\Scripts\python.exe -m unittest tests.test_faster_base_adapter -v
```

Installed-version inspection also verified the pinned versions, generate.py
SHA-256 `972d167ff41cb10593fe5dcfa9a4fd07d7a0ec680e86d86b9fd394bf2280c35f`,
CUDA 13.0 availability and NVIDIA GeForce RTX 4070. No model was loaded.
 The isolated Faster environment has no pytest, so its adapter suite
uses its existing unittest entry point. Ruff is absent. These are test-environment
limits, not evidence of successful GPU/model inference in this session.

Brian's explicit Faster default approval supersedes the old blind-listening
promotion gate. Matched native/Faster samples, actual heard-sample feedback,
accent/timbre/identity comparison, final words/chunk boundaries, ASR tail/repetition
diagnostics, clipping/finite/EOS checks, speed/VRAM and longform stability remain
useful quality evidence. No full-book listening or universal parity is claimed.

The original `1de4bf49` chapters 256–258 remain **631/634** selected and QA-passed;
all 631 selected audio hashes were verified. Missing selections are
`seg_86957b72b63f60ee` (256/index234), `seg_465784cde8353227` (257/index2), and
`seg_66ec076c3511fe36` (258/index76). Chapter review state remains draft. This
checkpoint does not generate the remaining three or mark listening complete.


## Guarded cross-project chapter review (2026-10-04)

The subsequent bounded ELF preflight identified import ordering for exact source-selected recorded drafts and loss of governed narration exclusions in chapter forks. The scoped repair stages an unassigned sample for the existing target selection owner; immutable reference assessments, worker authorization, QA and Takes remain intact. See [selected-recorded-voice-review.md](selected-recorded-voice-review.md) for the implemented guards and Brian-authorized exact role mapping. Runtime audio and listening evidence remain separate from this source checkpoint.
