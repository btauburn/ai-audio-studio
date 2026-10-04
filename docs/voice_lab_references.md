# Voice Lab discovery, evidence, and eligible references

Verified 2026-09-26. This is an extension of Studio's existing project and worker
ownership. No application, project database, queue, background daemon, or approval
service was imported. No upstream implementation code was copied for this work.

## Python owner API

`dubbing_service.app.audiobook.performer_discovery` exports:

```python
discover(directory, description="", *, traits=None, preferences=(), limit=5,
         online=True, limits=None, role_id=None) -> dict
rank_candidates(description="", *, traits=None, preferences=(), limit=5,
                role_id=None) -> list[dict]
reference_eligibility(source, *, intended_use_evidence=None) -> dict
probe_openverse(directory, *, limits=None) -> dict
acquire_reference(directory, source, *, intended_use_evidence, transcript=None,
                  start_seconds, end_seconds, audio_review=None, limits=None, review_only=False) -> dict
```

`directory` is `<existing project>/voice_lab/discovery`. `discover` returns
`matches`, `sources`, `status`, `errors`, `stats`, `limits`, and a durable
`cursor` with completed/pending candidate IDs. Each match supplies `id` and
`candidate_id`, performer, specific performance, verified credit facts, source
URLs, tags, `creative_direction`/`generation_direction`/`design_instruction`,
numeric `score_factors`, and an explanation. All current matches are explicitly
`metadata_only_exploratory`; none claims measured celebrity acoustics.

Ranking works locally, including sparse requests such as `young male`. Its ten
curated performance references include all four requested seed performers plus
Tom Holland, Timothée Chalamet, Daniel Radcliffe, Saoirse Ronan, Laura Dern and
LeVar Burton. Performance vocal-age/delivery/accent directions are creative
interpretations, separate from verified casting credits. There is no use of
performers' current biographical age as a ranking proxy. Appearance fields in a
structured dossier do not affect ranking. Unknown traits stay unconstrained.
The initial catalog is English; other-language matches remain exploratory with
an explicit compatibility penalty, not a claimed multilingual recording match.

Preference rows must include `voice_id`, `version` (positive integer or immutable
revision string), and the exact 64-character `audio_sha256`. They can additionally
include `candidate_id`, `inspiration={performer,performance}`, `tags`, `favorite`,
`rating`, and role-specific `suitability={role_id: rating}` (or numeric suitability
with `role_id`). Ratings are 1–5; missing ratings are neutral. Exact performance
feedback has the largest affinity, related performer and shared-trait feedback
less. Contributions are displayed, capped, and combined with a small deterministic
diversity penalty. Newly proposed candidates always start unfavorited and unrated.
No ranking function changes an established cast or starts synthesis.

Auto Cast uses the same catalogue, but as a *default* rather than an opt-in
search: `casting_anchors.anchored_plan` turns one character's dossier-derived
profile into one distinct performer and performance per candidate, and only a
reference that shares a supported vocal trait counts as an anchor. The proposal
stores the performer, performance, matched traits, rationale and catalogue
version, and the design instruction names that reference while still requiring
an original synthetic voice. A candidate that shares no trait is recorded as
unanchored with the reason it has none. The ranking itself sends nothing and
performs no network call.

## Bounded official metadata discovery

Only curated performer names, entity IDs and official Commons filenames leave the
device. Character descriptions and chapter/source text are not uploaded. Requests
use fixed HTTPS endpoints, a descriptive User-Agent, no redirects, and no supplied
URLs. Wikidata `wbsearchentities` resolves names against up to eight results;
performer occupation metadata distinguishes namesakes, then `P31=Q5` verifies a
human entity. Remaining ambiguity stays held. `wbgetentities` reads P990 and P407
qualifiers. `imageinfo` retrieves Commons extended license/source metadata.

[Wikidata P990](https://www.wikidata.org/wiki/Property:P990) describes spoken-voice
recordings, not necessarily the creative inspiration performance.
[Wikidata data access](https://www.wikidata.org/wiki/Wikidata:Data_access) documents
the APIs. [CommonsMetadata](https://www.mediawiki.org/wiki/Extension:CommonsMetadata)
documents `imageinfo`/`extmetadata`, including artist, license, credit, permission
and restriction fields. Downloaded fields remain untrusted data; HTML is removed
before display and never interpreted as instructions or permission decisions.

Default per-call bounds: 12 HTTP attempts including retries, 8 candidate matches,
2 recordings per performer, 2 MB per metadata response, 24 MB total cache storage,
one retry with bounded exponential backoff, 8-second HTTP timeout, 7-day successful
cache TTL and 5-minute durable failure cooldown. Bounds themselves are validated.
Metadata snapshots are content-addressed; cached payloads are hash-checked. A
nonblocking OS file lock prevents duplicate concurrent discovery within a project.
Completed requests survive interruptions and are reused on resume. Cache capacity
exhaustion preserves evidence and reports a hold rather than deleting anything.
The owner scheduler decides when to call this function and owns pause/status.

`probe_openverse` makes one optional bounded metadata search. The actual probe
returned three results but established no relevant performer identities or
synthetic-use eligibility, so Openverse was not adopted as a recording provider.
See [Openverse's documentation](https://docs.openverse.org/).

## Eligibility and acquisition

Recording download and private synthetic-voice use are separate decisions.
Recognized CC recording licenses, attribution, source/terms, permitted media URL
and absence of restrictions can establish download eligibility. Only a trusted
local, source-bound owner decision can supply intended-use evidence:

```json
{
  "source_id": "the-exact-recording-source-id",
  "status": "approved",
  "use": "private_synthetic_voice",
  "authority": "performer_consent",
  "evidence_ref": "local-reference-to-the-supporting-evidence"
}
```

The authority may instead be `voice_license` or `owner_original_voice` when that
is the documented basis. Neither public availability nor a fetched page's
`Permission` field populates this object. [CC BY 4.0's notice](https://creativecommons.org/licenses/by/4.0/)
explicitly distinguishes copyright licensing from other necessary permissions.
Missing metadata or evidence stays in the review collection without prompting
on each discovery run. Original VoiceDesign continues independently.

Eligible acquisition allows only `https://upload.wikimedia.org/wikipedia/commons/`
with redirects disabled. Defaults are at most two stored original downloads and
8 MB per original; hard maxima remain enforced by `DiscoveryLimits`. The original
is preserved unchanged, SHA-256 recorded, and Commons SHA-1 checked when supplied.
Replay/resume reuses completed original bytes. The source, terms, license and
eligibility accompany the immutable acquisition receipt.

Local `soundfile` decoding selects one contiguous 3–30 second mono segment from
an original at most ten minutes long (8–96 kHz). It checks finite samples, decoded
boundaries, clipping, silence and RMS. Preprocessing is crop plus lossless float
WAV encoding only; no pitch, gain, denoising or resampling changes identity.

Transcript alignment, speaker overlap, music and noise are not inferred from
amplitude. Acquisition additionally requires `audio_review` from a local tool or
human review, bound to `source_id`, `original_sha256`, `transcript_sha256`,
`start_seconds`, and `end_seconds`. It must identify `method` (`local_tools` or
`human_review`), an `evidence_ref`, `single_speaker=true`,
`transcript_aligned=true`, `overlapping_speech=false`, `heavy_music=false`, and
`unsuitable_noise=false`. Missing evidence creates a held receipt. Missing transcripts launch installed offline CPU ASR against the exact mono crop.
Machine transcripts include confidence/timestamps and remain unreviewed; ASR does
not establish speaker count, overlap, music or noise acceptance. Missing evidence
stays held. No diarization or music separation is automatically launched.

## Qwen worker integration

The existing Qwen worker exposes the additive method:

```python
provider._character_request("import-reference", {
    "project_id": project_id,
    "identity_id": new_identity_id,
    "acquisition_id": acquisition["acquisition_id"],
    "acquisition_sha256": acquisition["acquisition_sha256"],
})
provider._character_request("prepare", {
    "project_id": project_id, "identity_id": new_identity_id,
})
```

`POST /api/v1/tts/character/import-reference` accepts those four fields, not an
arbitrary path, URL or browser upload. It resolves only
`voice_lab/discovery/acquisitions/<id>/acquisition.json` inside the existing
project. Before import, prepare and synthesis, it rechecks eligibility, hash-bound
review, source bytes and transcript, actual crop equality and PCM QA. Identity
files preserve the original and the conditioning segment. Changed requests must
use a new identity ID. Interrupted imports resume without new download or voice
generation. All worker methods use the existing `_infer_lock` and provider/lease
boundary; there is no additional GPU authority.

Original generated voices keep `provenance.origin=generated_original_voice`.
Acquired clones use `eligible_reference_clone` and retain acquisition/source
evidence. Both use the existing numeric NPZ prompt representation, with pickle
disabled and model/runtime hashes checked. The clone path is Base, never
CustomVoice. The [Qwen README](https://github.com/QwenLM/Qwen3-TTS#voice-design-then-clone)
documents `create_voice_clone_prompt` and subsequent `generate_voice_clone` reuse.
This path was exercised with fake model fixtures only; no real celebrity clone
or GPU model switch was performed in the blocked-source assignment.

## Upstream revision and license review

The GitHub API supplied these exact revisions on 2026-09-26. They are reference
revisions, not a claim that the local dirty checkout equals remote HEAD.

| Reference | Revision | Observed license / use |
| --- | --- | --- |
| [AI Audio Studio](https://github.com/tadashijlu/ai-audio-studio) | `d6e0edd14a635d2582a86c6b37cb3298cf7b72c3` | Apache-2.0 platform; existing repository retains its documented multi-license boundaries. |
| [Audiobooker](https://github.com/mcp-tool-shop-org/audiobooker) | `651511a388bc27568f2985b3960d09e7e7ef8d15` | MIT; existing analysis/compiler provider remains isolated. |
| [Castwright](https://github.com/dudarenok-maker/Castwright) | `e0e1ebebd38ff5ec4bb83507f19f01cb585aa5a1` | FSL-1.1-ALv2, verified from LICENSE content (GitHub labels it NOASSERTION). No code copied. |
| [audiobook-studio](https://github.com/builtbybasit/audiobook-studio) | `6b831b8fd29ae9225fd31b22b4390ac17cf8dfa9` | No LICENSE returned; no code copied. |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | `022e286b98fbec7e1e916cb940cdf532cd9f488e` | Apache-2.0 code; use existing pinned model manifests and runtime. |

[Castwright's cast-review documentation](https://github.com/dudarenok-maker/Castwright/wiki/Reviewing-Cast-and-Assigning-Voices)
describes character rows, reusable voices, duplicate/alias review and A/B
comparison. These were workflow references only. The candidate-review repository
describes itself as a Vue 3 prototype with a permanent seeded demo and partial
backend; its README states real-provider integration is future work. Package
version is `0.0.1`. Its tree includes backend/job/cast code and tests; it was not
treated as proof of a production-ready voice pipeline.

## Actual exercise and limits of the evidence

Evidence root:
`C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-26-voice-lab\discovery`.

- `network-exercise-resolved.json`: successful structured entity discovery for
  nine selected performers across young-male, narrator and young-female requests.
- `network-final-metadata.json`: Attenborough P990 recording duration 36.38979 s
  and Fry P990 recording duration 8.91646 s, as reported by Commons metadata.
  Both recording licenses permit download, while synthetic-use evidence remains
  absent. Neither recording was downloaded or acoustically analyzed.
- `network-exercise.json`: initial probe, including Openverse; retained to show
  the namesake ambiguity found and subsequently corrected using occupation data.
- Narrator resume made zero HTTP requests and reused eight cached responses.
- `upstream-revisions.json` and individual upstream JSON files preserve revision,
  license response hashes and the observed license text. Candidate-review README,
  package and tree evidence are also retained.

Verification command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_performer_discovery.py tests/test_voice_lab_reference_clone.py tests/test_character_qwen.py -q
```

The 71 checks passed at the final discovery/worker checkpoint, including exact-version ranking feedback,
source/permission holds, cache integrity and resume, request/storage bounds,
namesake resolution, immutable original/crop import, rejected stale/corrupt
receipts, Base prompt reuse and worker HTTP input/lock checks. A regression also
checks Studio's actual CastingProfile presentation, perceived vocal age, texture
and warmth dimensions; unknown decisions and appearance stay excluded. Existing generated
original-reference tests remain passing. Test waveforms and fake synthesis exist
only in isolated pytest temporary directories; they are not listening-test audio.
Real discovery generation totals: **0 downloaded audio files, 0 reference audio
seconds generated, 0 candidate audio seconds generated, 0 listening-test files**.


## Qualitative named-reference feedback and automatic transcripts (2026-10-02)

The phone page saves nonempty qualitative direction without requiring stars.
The saved event remains bound to the heard candidate. Explicit named performer
requests and retained named direction select an exact eligible imported recording
or a clearly held reference option, never an unrelated VoiceDesign substitution.
Later performance wording does not erase the named request. Approval still needs
an explicit human four/five-star decision; source review never becomes a rating.

Stereo originals are preserved unchanged. Preparation records the exact mean of
the two decoded channels, contiguous crop and lossless float WAV. The worker
reconstructs and checks that exact operation; missing conversion provenance fails.

Automatic reference transcription uses materialized project-local
`models/whisper/faster-whisper-base.en`, installed faster-whisper 1.2.1, CPU int8,
two threads, beam five, word timestamps, no VAD, no previous-text conditioning,
temperature zero, and local-files-only. No ASR GPU or model download occurs.
Audio SHA-256, exact model-file hashes, language and settings bind the cache.
Missing models/packages, empty output, invalid timestamps, nonfinite values and
corrupt cache are explicit states. English is currently supported. Other
languages remain unavailable rather than being silently transcribed as English.

`POST references/{acquisition_id}/transcript` verifies receipt/reference hashes
and returns the draft. `POST .../transcript/correction` saves a separate immutable
correction with optimistic original-ASR and current-correction hashes. Listing
and reload retrieve that correction; original ASR, acquisition and QA are unchanged.
Corrections are explicitly unaligned/unreviewed. To use corrected words, create
a new acquisition with the corrected text and fresh hash-bound audio review.
`GET references` and the bounded reference audio route power the phone panel.
Playback and correction never synthesize, assign, favorite, rate or notify.

`review_only=True` permits inspection of a download-eligible licensed source
without intended-use approval, but unconditionally holds cloning. Normal import
still revalidates source-use, exact audio review, conditioning and provenance.
First auditions of eligible imported recordings may reuse prepared conditioning
before synthetic audio exists; ordinary unheard VoiceDesign reuse is still denied.
The isolated audition child receives exact immutable receipt/original/reference
assets so production rendering uses the same verified acquisition.

The BBC Attenborough excerpt is CC BY 3.0, with recording permission documented
on Commons (VRT ticket 2013111310011767). The owner explicitly requested a private
local reference-cloning experiment. The source-bound owner decision records that
precise licensed-recording/private-request basis using the existing `voice_license`
authority; it does not claim performer consent, endorsement, general legal
clearance, public-use permission, or production voice acceptance. Five source
audio review observations remain required and are not inferred from copyright
metadata, waveform amplitude or ASR confidence.


## WebM/Opus references (2026-10-02)

An explicitly bounded Commons WebM/Opus original can now be prepared with the
existing installed FFmpeg and FFprobe. Discovery defaults remain8MB per source
and24MB storage; owner-controlled limits may raise a WebM source to32MB within
the existing total storage bound. Other decoder-supported originals retain16MB
maximum. At most ten minutes, one Opus audio stream, mono/stereo,8–96kHz and
an exact3–30second sample crop are accepted. Unsupported media, incomplete
crops, changed decoder binaries/version, missing executables and timeout fail
clearly. No automatic install, network decoder protocols or format substitution.

The immutable original is retained unchanged. A source_decode receipt records
FFmpeg version and both executable SHA256 hashes. Acquisition and worker use
the same sample-index decode, stereo arithmetic mean and float WAV path; the
worker independently reconstructs and compares every sample. The sample path
for existing soundfile originals and receipts remains unchanged. Existing
FFmpeg/FFprobe on PATH or Windows WinGet Links are used; optional trusted local
configuration uses VOICE_LAB_FFMPEG and VOICE_LAB_FFPROBE. Changed executable
provenance requires a fresh reviewed acquisition, never silent reuse.

Emma Watson's ONU Brasil recording is CC BY3.0 on Commons. The owner's explicit
2026-10-02 private actor-reference request and exact licensed recording use the
same existing source-bound voice_license decision as the Attenborough experiment.
This is not a performer-consent, endorsement, blanket-rights or production
acceptance claim. Review staging is separate from live projects and never
marks source suitability passed from waveform or ASR metrics. Music and
overlapping/single-speaker evidence remain unavailable in the installed QC;
the exact clip requires actual listening evidence before an audition.

Verification includes real installed-codec disposable Opus fixtures, crop replay,
changed-decoder rejection, review-only rejection before any model, original/crop
integrity and unchanged generated/original-reference workflows. Synthetic
fixtures are never presented as actor or listening-test audio.


### Explicit experimental vocals isolation (2026-10-02)

performer_discovery.acquire_reference(..., isolate_speech=True) is an opt-in
owner-side preparation step. It requires the already installed Torch/Torchaudio
runtime and a locally provisioned, exact official HDemucs weight file; it never
downloads a package or model. Studio's lightweight environment does not need
Torch when this option is off. Run preparation in the existing Qwen/Faster
environment, whose dependency versions must match the importing worker.

The immutable original recording and exact contiguous source crop remain the
provenance base. reference_separation.py records resampling to44.1kHz,
mixture normalization, the vocals stem and its arithmetic stereo mean. Model
SHA256, loaded runtime versions, loaded pipeline hash, input/output PCM hashes
and output length form source_processing. Owner audio review must bind that
exact recipe. Worker import, preparation and synthesis re-decode the original
and replay the processing, requiring bit-identical PCM before using a prompt.
Edited source or dependency drift requires a restart. CPU RNG and thread
settings are preserved. Default unprocessed receipts retain their exact format.

Weight URL: https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt
Expected SHA256: a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816
Fixed local path: models/reference_preparation/hdemucs_high_trained.pt.
The recorded hash is a local acquisition digest, not a claimed published digest.

Isolation can change timbre or introduce artifacts. Passing classifier/ASR
checks does not approve speaker identity, naturalness or production use.
The resulting audition stays experimental and unrated until actual review.
An isolation failure stops import/render; no substitute voice or runtime is
silently selected. Rollback is the original renderer and unmodified references;
do not reinterpret a processed reference as an original raw recording.

For the Emma Watson experiment, the original CC BY3.0 ONU Brasil recording is
preserved, as is the unprocessed7.35–17.8second crop and residual. Exact
processed PCM passes fixed Music<0.2 and overlap/singleton-speaker gates,
supported ASR/VAD/clipping/coverage checks and cross-process replay. This
licensed-recording plus explicit owner private-use request basis follows the
existing Attenborough voice_license path. It does not assert performer consent,
endorsement, public publication rights or production voice acceptance.


Automatic local source preparation (2026-10-02)
----------------------------------------------

Unreviewed acquisitions now use the existing CPU source checks and one bounded
vocals-isolation fallback when music or noise is detected. Explicit existing
reviews, manually processed receipts, approved identities, ratings, favorites and
assignments retain their original behavior. The owner tool is POST
/api/v1/projects/{project_id}/voice-lab/references/{acquisition_id}/prepare with
acquisition_sha256 and optional expected_text. It reads the fixed project-owned
original and produces a separate immutable acquisition, never a selection or a
new audition. It never downloads a missing original.

AST AudioSet (sigmoid multilabel probabilities) and pyannote local segmentation
run offline with the already installed pinned weights in the existing Faster
worker Python, on CPU. Music/Noise >=0.2 permits one cleanup pass; Speech must
exceed0.5, overlap argmax must be absent and overlap probability peak below0.1.
Local speaker classes cannot establish identity. Crops spanning multiple10s
windows remain held for speaker review because no global speaker-consistency
model is installed. This conservative limit applies only to automatic
preparation; existing reviewed longer references remain valid.

Raw source hashes, exact crop, model/code/dependency fingerprints, pre/post
classification, ASR/VAD/alignment checks and the unchanged v1 separation recipe
are recorded. Machine ASR draft text remains a draft even when submitted unchanged
as expected_text. Cleanup may produce a processed review-only reference while
transcript alignment is held. Failed/unavailable checks never declare a clean or
clone-ready source. No retry loop, model download, TTS call or implicit fallback.

The versioned cache binds request and whole-result hashes, exact input/output PCM
and rates, and rederives the final decision from required evidence. Paired
metadata/audio eviction honors both a16MB cache cap and the caller's total
discovery storage budget, reserving2MB for acquisition outputs. The prepare tool
has a bounded48MB total budget for its existing32MB maximum original. A cached
failed isolation/post-check is a hold, and is reused without a second separation.
A changed pipeline/model/dependency fingerprint invalidates cache reuse; changed
loaded source requires restarting Studio. Corrupt cache artifacts hold rather
than regenerate silently.

Processed portraits display 'Processed reference · may affect timbre' only when
the exact key/audio hash has recorded isolation provenance. Machine source checks
and the processed label do not constitute human voice identity, naturalness,
expressiveness or production acceptance.

Rollback: disable automatic_preparation explicitly at the owner acquisition
call (False), and stop using the prepare endpoint. Existing explicit-review
acquisition and original renderer remain available. Do not edit the preserved
reference_separation.py v1 file: existing receipt replay binds its exact source
hash. Changing that recipe requires a separately versioned implementation and
new reviewed acquisitions.
