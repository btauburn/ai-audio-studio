# Voice Lab v1 integration contract

Studio owns projects, roles, immutable Takes, voice profiles and feedback. This
feature extends `Project.voice_lab`; it does not create a project store or queue.
Existing production identities are read-only unless explicitly assigned with an
expected-current-identity token. An isolated listening-test project may auto-cast.

## HTTP / browser contract

Desktop page `/static/voice-lab.html?project=ID` uses
`/api/v1/projects/ID/voice-lab`. The private delivery page `/voice-lab?project=ID`
uses `/api/voice-lab/ID`. Both expose the same JSON and call the same Studio owner.
The private server retains Brian-only Tailscale auth and an explicit project
allowlist. Mutations send `X-Review-Request: 1`; browser code never calls synthesis
on load, play, favorite, rating or navigation.

- `GET {base}`: `{project_id, project_name, revision, scope, voices, roles,
  shortlists, review_position, automation, excerpt, issues}`.
- Each `voice`: `{key, voice_id, version, name, kind, synthetic, role_id,
  role_name, tags, inspiration, explanation, details, favorite, rating,
  suitability, audio_url, audio_sha256, duration_seconds, technical_qa,
  human_review, lines}`. `inspiration` is `{performer, performance, source_urls}`
  or null. `lines` has `{utterance_id, start_seconds, end_seconds}`. `audio_url`
  is null until actual production-path audio is available. No design-reference
  waveform is presented as a production audition.
- `POST {base}/preferences`: `{key, audio_sha256, favorite?, rating?,
  suitability?, role_id?}`; null clears rating/suitability. Returns saved voice
  preference and updated ranking. A version without production audio cannot be
  favorited/rated. General preference, role suitability and defects are separate.
- `POST {base}/position`: `{key?, audio_sha256?, seconds, filter?}` saves position.
- `POST {base}/match`: `{description?, role_id?, limit?}` performs bounded cached
  discovery and returns `{matches, ...}`. Shortlists have metadata-only exploratory
  labels. No synthesis; source text is never sent to remote discovery.
- `POST {base}/more-like`: `{key, request_id}` records at most two related design
  directions, no rating inheritance; speculative synthesis remains paused.
- `POST {base}/assign`: `{key, role_id, expected_identity_id}` explicitly assigns
  this version for future Takes using the existing owner. Does not rewrite Takes.
- `POST {base}/automation`: `{paused}` changes bounded discovery state; status and
  durable request cursor included in GET. Audio replenishment stays idle while
  first listening test is pending.
- `POST {base}/issues`: `{audio_sha256, seconds, text, category}` stores exact
  audio-version feedback, never starts repairs. `category`: technical/performance.
- `GET {base}/audio/{key}` streams verified bytes and supports Range.

`excerpt` is null or `{status, reason, manifest_id, chapter_title, duration_seconds,
audio_url, audio_sha256, technical_qa, human_review, map, comparisons}`. `map`
contains `{utterance_id, role_id, role_name, start_seconds, end_seconds}`. Existing
audio comparisons must have identical passage/utterance identities and each
supplies measured loudness (gain only during playback; source bytes preserved).
No comparison edition is synthesized for this assignment.

## Python seams

`performer_discovery.py` owns bounded structured discovery, immutable source
evidence and ranking. Expose a documented simple Python API callable by
`voice_lab.py`, accepting compact traits/description and stored preference rows.
Cache files live under the existing project's `voice_lab/discovery` directory.
Cloning requires separate download and intended-use eligibility. Missing evidence
is a hold; open recording license alone never asserts personality/voice rights.

`listening_excerpt.py` owns literal chapter identity and contiguous source
manifest validation, with no network/GPU work. It consumes existing canonical
source revisions/utterances/role aliases. Missing Chapter 1 produces a durable
blocked-source report, never a substitute demo. Every numeric duration and role
boundary has passing and failing regression cases.

Production synthesis stays behind existing Qwen provider, Take commit and TTS
lease. Stable references already use VoiceDesign plus Base. No new model, global
backend switch or silent fallback is required. Preserve every existing project
selection/hash in baseline and final audits.


## Portrait review selection (2026-10-02)

Heart/keeper and character assignment remain independent. Heart calls the existing
`POST {base}/preferences` with exact `key`, `audio_sha256` and `favorite`; it never
assigns, rates or starts synthesis. The check mark calls `POST {base}/casting/feedback`
with `event_id`, `role_id`, `candidate_id`, exact `key`/`audio_sha256`,
`round_revision`, `assignment_revision`, `action="use"` and
`selection_confirmed=true`. Stars may be omitted. This is an explicit human
selection of the heard audition, not an invented four/five-star rating. Existing
ratings remain untouched and legacy explicitly rated selection stays supported.
The same identity, audio integrity, fixture exclusion, uniqueness, mutation lease,
optimistic revision and idempotency checks apply. Conflicting low stars supplied
with selection are rejected. A favorite alone never constitutes selection.

Casting role rows expose `selected_candidate_id` only for a valid current human
decision. Candidate `favorite` is bound to its exact audio hash. Each role can also
expose `current_audio` with key, name, kind, synthetic/provenance/inspiration,
duration, audio hash, verified availability/error, URL and favorite. This is the
current voice's existing sample, independently of review-round candidates; it does
not create a round, rating or approval. A stale assignment key is excluded when
its saved voice revision differs from the current role. Missing/corrupt bytes have
no playable URL. Original synthetic samples must be labelled as such, never as
actor-reference clones. Character selection does not promote an inference runtime
or replace its separate validation gates.
