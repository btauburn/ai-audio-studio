## ADDED Requirements

### Requirement: Literal local chapter identity gates GPU work
Studio SHALL require literal Chapter 1 of Super Supportive, established from local
source metadata, TOC, heading, immutable bytes and complete extracted text before
design or rendering of the isolated listening test.

#### Scenario: Imported subset starts at Chapter 256
- **WHEN** import index 1 refers to Chapter 256 and no literal Chapter 1 is local
- **THEN** the listening test is blocked with the available chapter range and no
  GPU generation or substitute audio occurs

### Requirement: Contiguous five-minute source coverage
The persisted excerpt SHALL contain complete adjacent source utterances with
interleaved narration and at least four alias-resolved actual speaking characters.
Both estimated and decoded final duration SHALL fall within 285–315 seconds at 1x.

#### Scenario: Mentioned character or alias would inflate the count
- **WHEN** a candidate has four labels but fewer than four resolved dialogue people
- **THEN** validation rejects the excerpt and never treats a mention as speech

#### Scenario: Source or selection changes during render
- **WHEN** the source, manifest, cast, utterance requirement or selected Take changes
- **THEN** the stale result cannot replace the current listening selection

### Requirement: Evidence-separated performance matching
Auto Match SHALL accept sparse descriptions, rank specific verified performances,
retain source URLs and uncertainty, separate creative traits from acoustic
measurements, and use exact-version preferences without changing existing casts.

#### Scenario: A voice is rated
- **WHEN** an exact production voice version receives a favorite or 1–5 rating
- **THEN** the saved feedback affects explainable matching and unrated variants
  remain neutral with independent ratings

### Requirement: Eligible reference acquisition
Discovery SHALL bound requests, bytes, storage and retries; separate download and
intended-use eligibility; preserve source/license/terms evidence and original bytes;
and hold ambiguous sources without bypassing restrictions.

#### Scenario: Open recording license has no synthetic-use evidence
- **WHEN** a performer recording is freely downloadable but intended-use rights
  are unresolved
- **THEN** it remains held and original voice design is available instead

### Requirement: Stable reusable production voices
Original voice design SHALL use saved VoiceDesign references and reusable Base
conditioning behind the existing worker. Eligible reference clones and built-in
provider voices SHALL remain distinguishable. Versions SHALL be immutable and
only production-path audio SHALL be favoritable.

#### Scenario: Reference design finishes before production speech
- **WHEN** a voice has conditioning artifacts but no real production Take
- **THEN** it cannot be rated as a heard voice until an exact production recording
  is available, after which its first reviewed binding persists

### Requirement: Integrated private review
Studio and its existing private listening page SHALL support Play, Favorite, Rate,
Next, seeking, role line jumps, timestamped issues, explicit assignment, and
same-passage loudness-matched A/B while preventing overlapping playback.

#### Scenario: Private feedback is saved
- **WHEN** Brian submits a rating through the verified private route
- **THEN** the canonical Studio owner persists the exact voice/audio version,
  reports failures, retains it after restart and does not trigger synthesis

### Requirement: Bounded discovery preserves production priority
Automatic discovery SHALL use the existing durable job owner with pause/status,
bounded resume and production priority. Speculative audio SHALL remain idle while
the first listening test is pending.

#### Scenario: A production synthesis job is queued or active
- **WHEN** a discovery or voice-design operation is requested
- **THEN** it yields to production and does not change the running backend, voice,
  selected Take or GPU execution policy

### Requirement: Explicit complete-chapter alternate rendition
A separately authorized chapter rerun SHALL create a derived Studio project
with exact canonical source coverage, fresh stable cast identities and preserved
original project audio and selections. It SHALL use the existing synthesis job,
GPU lease, immutable Take and blocking QA owners. Human listening approval SHALL
remain pending while a technically valid continuous review copy is playable.

#### Scenario: Chapter 258 is separately authorized while Chapter 1 is unavailable
- **WHEN** the user explicitly requests Chapter 258 with a new cast
- **THEN** the application renders the exact available Chapter 258 in a separate
  rendition, preserves all its actual speakers and source utterances, and does
  not claim it completes the blocked Chapter 1 listening test

#### Scenario: Derived source attribution needs correction
- **WHEN** a reviewed source audit resolves a previously unknown speaker or alias
- **THEN** the correction binds exact source and utterance hashes and evidence,
  preserves the prior attribution in derivation metadata, and only changes the
  new chapter rendition

#### Scenario: Interrupted chapter render resumes
- **WHEN** the same request resumes with unchanged source and cast
- **THEN** valid selected Takes are reused without duplicate synthesis, queued
  production retains GPU priority, and stale results cannot supersede newer work
