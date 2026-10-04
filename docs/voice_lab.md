# Automatic Voice Lab and Auto Match

Implemented in the existing Studio project/Takes, voice library, Qwen worker,
GPU-operation, durable-job and private delivery boundaries. No additional
database, queue daemon, delivery application or public host was introduced.

## Open and review

Studio: `http://127.0.0.1:9000/web/voice-lab.html` (choose a project).
The existing private listening room links to `/voice-lab`; it proxies Studio
for project state and streams verified audio, preserving Range/HEAD requests.
Its configured recipient, host, origin, project and mutation controls apply.

On Brian's phone, keep Tailscale connected and open the private HTTPS Voice Lab
link in Safari. The desktop must be awake with Studio and private delivery
running. Tap **Listen to chapter**; use **Feedback** beside the playback
controls to reach the timestamped note form. Cast hearts and ratings save to
the same exact voice versions shown on the desktop. Reload an already-open page
after a player update. Playback starts within the tap handler; saving the prior
position runs independently so a slow network save cannot delay the play call.

Play, favorite, rate, then Next. Feedback stores exact voice version, Take and
audio hash in `Project.voice_lab`. Role suitability and timestamped technical
or performance notes are separate. Favorites never assign a role; imports and
variants never inherit ratings. A missing rating contributes no dislike.
Saved preferences immediately rerank the project's shortlists and influence
subsequent Auto Match calls. These preference records are currently project
scoped; importing a global library version starts unrated in the destination.

The existing global library stores immutable production samples and conditioning
assets as additive `voice_lab_versions`. Browse/import works in an empty project.
One media element and guarded asynchronous selection prevent overlapping/stale
playback. Space, F, 1–5 and arrows leave typing and focused controls alone.

## Discovery and generation

Auto Match accepts a sparse description or existing CastingProfile. A bounded
official-source catalog ranks specific performances, including the four seeded
families and additional performers. Wikidata/Commons entity resolution and P990
metadata augment it. Creative directions, verified metadata and actual audio
observations remain distinct; metadata-only matches are exploratory. This is
bounded discovery, not exhaustive casting or acoustic analysis.

Download eligibility and synthetic-use eligibility are separate. Missing
permission or hash-bound local audio-review evidence holds a reference rather
than triggering a download/clone. `performer_discovery.acquire_reference`
supports preserved originals, bounded clean crops and immutable receipts.
Local transcript/single-speaker/music/noise review is required; no automatic
ASR/diarization adapter was added. See `voice_lab_references.md` for evidence,
limits, provenance and exact upstream revisions/licenses. No upstream code was
copied for this feature.

Original design uses the existing Qwen provider to create VoiceDesign reference
audio and a reusable Base prompt. Conditioning alone cannot be favorited/rated:
the record gains review audio only from the actual production Take path.
The additive eligible-reference import endpoint validates the acquisition receipt
again inside the existing worker. CustomVoice remains a built-in-voice backend.

Explicit assignment validates the current cast revision and preserved
conditioning. Legacy built-in reuse requires recoverable exact settings; an
incomplete historical recording stays playable but cannot silently recast a role.
Role-bound legacy revisions differ when assigned to another character and are
reported as such. Existing Takes never change.

More like this preserves the original and prepares two unrated related design
directions. Audio replenishment remains idle while this listening test is
pending. Pausable discovery uses the existing durable job owner, bounded cycles
and startup recovery. It never dispatches GPU work from playback or feedback.

## Source-bound listening operation

All routes below extend `/api/v1/projects/{id}/voice-lab`:

| Operation | Method/path |
| --- | --- |
| View, verified audio | `GET /`, `GET /audio/{key}` |
| Review | `POST /preferences`, `/position`, `/issues` |
| Discovery | `POST /match`, `/more-like`, `/automation` |
| Explicit design/casting | `POST /design`, `/import-reference`, `/assign` |
| Existing global library | `GET /library`, `POST /library/publish`, `/library/import` |
| Isolated listening workflow | `POST /source-preflight`, `/excerpt-manifest`, `/prepare-cast`, `/render-excerpt` |
| Explicit alternate chapter | `POST /fork-chapter`, `/render-chapter`, `/export-chapter` |

Inspect the whole correct local chapter, use `select_contiguous_excerpt` to
select complete canonical utterances, and persist its manifest before synthesis.
The manifest binds source bytes, deterministic EPUB chapter-body extraction or
actual plain-text chapter heading, full chapter hash, offsets, IDs, attribution,
aliases, role turn counts and timing evidence. Imported index 1 is insufficient.

The isolated operation requires four real speaking identities plus narration,
with distinct final conditioning identities proven from actual provider-call
metadata. It shares production GPU priority and renders through the existing
immutable Take pipeline and blocking QA. Natural decoded duration must be
285–315 seconds. At most two timing adjustments can trim complete boundaries or
add up to eight adjacent utterances / 60 estimated seconds per round, retaining
all required speakers. Updated manifests precede added work; valid Takes are
reused. There is no padding, speed change, paraphrase or interior omission.

An accepted export retains clean PCM master, MP3, exact source-to-audio map,
hash-bound QA and measured loudness. A/B accepts the identical ordered passage,
attenuates playback to matched measured loudness, and preserves original files.
No extra full comparison edition is automatically rendered. Human review stays
pending while a technically valid review copy is playable.

### Explicit alternate chapter renditions

A separate user request may rerun a complete available chapter. `fork-chapter`
copies its exact canonical chapter ID, source snapshot, stable utterance IDs and
spoken-text history into a new Studio project. It clears the new project's old
cast identities, Takes and review state. The source project remains unchanged.
Source-supported attribution corrections must bind the source and utterance
hashes and retain the previous attribution as derivation evidence.

`render-chapter` requires reusable locked voices for every actual speaking role
and resolved attribution before claiming the existing GPU lane. It returns an
existing durable synthesis job. The same request resumes valid Takes; source or
cast changes cannot replace newer work. `export-chapter` assembles complete,
natural-speed, QA-passed source audio and explicit source scene pauses into a
clean master and MP3, and publishes it to the same Voice Lab player with human
review pending. Chapter renditions do not complete or loosen the separate
Chapter 1 excerpt's source, timing or speaker-count requirements.

## Current acceptance boundary

The September 26 assignment's actual Chapter 1 test is source-blocked. Local
Super Supportive EPUBs/canonical snapshots contain chapters 256–260. Additional
local PDFs contain chapter 187 and a chapter 203 recap. No literal Chapter 1 was
found; no source passage or character dossiers/cast were fabricated.

The initial feature verification used copied pre-existing audio in a clearly
labeled regression project. The Chapter 1 listening-test project still contains
the source blocker and exploratory shortlist.

The separately authorized complete Chapter 258 rerun now exercises actual Qwen
VoiceDesign conditioning, saved Base prompts and production Takes in derived
project `4814e40f`. Its 102 spoken utterances use four original synthetic voices
(narrator, Alden, Stuart and library interface); the five source scene breaks
remain pauses. The decoded WAV and MP3 are 622.650167 seconds at natural speed.
All selected Takes pass hash-bound technical QA. A separate offline transcript
audit has 100 passes and two numeral-format review flags, with exact-audio-bound
equivalence notes; raw results remain unchanged. Human performance review is pending.

The complete EPUB body and immutable selected audio were independently checked,
and all 1,813 previously selected audio hashes and original selection pointers
remain unchanged. Continuous HTTPS browser playback and character jumps work.
Physical iPhone playback and the separate Chapter 1 five-role test remain
unverified. No celebrity recording was cloned, and no additional complete
comparison edition was generated.

Evidence and detailed report:
`C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-26-voice-lab\implementation-report.md`.

Chapter 258 evidence and listening entry point:
`C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-26-voice-lab\chapter-258-rerun\chapter-258-report.md`.
