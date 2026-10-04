# Human-directed casting

The Voice Lab casting inbox separates a library favorite, a library rating,
a heard audition, and a character-fit approval. Existing unknown rating
provenance is retained and is never converted into Brian's casting approval.

## Workflow

Choose **Character cast (AutoCast)** when importing local EPUB/TXT to retain
Audiobooker speaker analysis. **Narrator reading** remains the explicit default
for the existing single-narrator creator. AutoCast uses the installed pinned
isolated analyzer and preserves the exact source bytes. Unresolved dialogue is
held for attribution review; it cannot silently become Narrator. Analyzer
availability failures do not fall back to a narration-only analysis.

**Prepare cast** prepares up to three audition candidates for unresolved actual
speakers in the requested utterance scope. Aliases resolve to canonical roles;
non-speaking names do not become approval requirements. Existing approved
assignments remain in force. A source-grounded casting profile supplies design
direction where available. Performer matching remains optional inspiration.

**Auto Cast starts by itself.** A successful AutoCast import records a durable
run intent — bound to the project, the source revision/content hash and the
casting pipeline version — and the existing maintenance owner starts the rounds
from it. No **Prepare cast** click is required for a new project, and Voice Lab
never triggers casting by being opened, refreshed or polled. The run key makes a
duplicate completion event, a page refresh and a server restart no-ops: one
source revision and pipeline version produce exactly one effective run, and an
unchanged source preserves human ratings and accepted voices. A genuinely new
source revision mints a new key, so new or changed characters are processed
while every existing approval stays in force.

**A pipeline bump re-casts, and must be asked to.** The run key alone only
*relabels* a run: `prepare()` skips a character that already has a round, so
bumping the version would have produced a completed run still holding the old
candidates. The version is therefore recorded on the round
(`round.pipeline_version`) and *compared* rather than merely hashed into
identifiers, and a round produced by an earlier version is stale — no longer what
this pipeline would propose. Re-casting a stale round discards nothing: the
superseded round, with every rating and direction on it, moves to history, and the
request receipt names the characters it re-cast (`recast_role_ids`). An approved
character is skipped before the staleness check, so a version bump can never
overturn the human's decision, and re-running at the current version is
idempotent — a refresh, a restart or a duplicate completion event never disturbs a
round under review.

Initial comparisons use `casting-60-v1`; replacements use `casting-30-v1`.
The spoken scripts are independent of the book and versioned together for the
comparison. Each candidate retains one conditioning identity across sections.
Audio is rendered as ordinary immutable Takes and assembled into one playable
audition with section positions. Actual decoded duration and outliers are
recorded; there is no artificial padding, duration-driven stretching or clipping.

Save submits stars and the raw direction together. No stars are inferred from
text. A 1–3-star decision creates one durable replacement operation, with unrated
new candidates. Rejecting the current assignment revokes that character's
future strict production approval. An unrelated approved role is unchanged.
An unrated round waits without generating new proposals.

One rating is saved at a time per character, and a rating given while another is still
being saved is never dropped: the click says it is waiting, goes out as soon as the
previous one is confirmed, and is sent against the revisions that save refreshed, so
the server does not file it as historical. Saving and reloading the casting view are
kept cheap on purpose (the book is read once per view, project files are written
compactly), because the page polls while auditions are being made.

The list says what is left: "11 of 13 voices chosen · 2 left: Jorildyn, Faylar", and "All 13 voices
chosen" when the last one is done. A character counts as chosen while a valid human approval stands
(the same test the book uses to count voices), even if a later round was reopened for it, so it
leaves the list; only characters the book needs are counted. "Use this voice" removes the character
from the list as soon as the server confirms it, without waiting for the list to be fetched again.

The page stays quiet while it is open. Saved preview receipts already carry their current status when
the page loads, and one that is only queued waits for someone to press Resume, so only a receipt that is
running is followed (every three seconds, and not while the tab is hidden); a preview started on the page is
followed to its end. Polling every saved receipt used to keep Studio at full load for as long as any page
was open once a few dozen stale ones had piled up, which is what made ratings slow and unreliable.

"Open an existing book" (Studio and private review alike) asks Studio for `GET /api/v1/projects/summaries`: the
id and name of every project, remembered until a project file changes, in milliseconds. It used to read the
full project list (tens of megabytes) with a ten-second limit, which came back empty while Studio was busy.
The private service only lists books it shares: its own project directory, `voice_lab_projects` in its
config, and books recorded as private uploads.

**Save feedback & use this voice** requires an explicit 4–5-star decision on
the exact playable candidate. Old tabs receive a conflict and their feedback
is retained as history. A duplicate event cannot create another replacement.
Identical effective identities cannot be approved for distinct characters.
The legacy direct-assignment controls are disabled in strict casting projects.
Qwen reference assignment verifies and copies the heard conditioning bytes into
the provider's identity directory. The reviewed recording and its metadata keep
their original immutable paths; only the assigned role uses the production paths.

Directions retain the human's raw words and exact comparison baselines. Accent,
depth and timbre requests require voice selection/design and remain acoustically
unverified until listening. Native Qwen Base cloning has no natural-language
delivery control. Unsupported performance-only requests retain the voice and
do not churn candidates. Supported CustomVoice delivery previews retain the
preset identity. A request such as “British” does not imply a regional accent.

**Render approved scope** creates a durable production request. While required
current approvals are missing it owns no GPU lease. Existing application
maintenance revalidates the source and approvals, freezes a manifest, and
dispatches the same request once. Changed source requires explicit resume.
Failures are bounded or held for attention. A rejection while rendering keeps
the finished Take as history but prevents it from being selected as current.
An explicit resume of a stopped or parked request grants a fresh bounded retry
budget while preserving total attempts, failure details and manifest history.
Resume cannot reset a queued or running request's state or counters.
The current strict worker produces Takes; chapter/full-project export remains
a separate existing action, guarded by current approval and matching selected
Take fingerprints, bytes and QA before assembly and publication.

## Authority and persistence

`project.voice_lab.casting` is an additive v1 state in canonical Studio JSON:
roles, immutable feedback events, request deduplication, durable work intents,
production receipts and manifest history. Existing project locks and atomic
saves commit the event, assignment and replacement intent together. Existing
SynthesisJob storage and the application maintenance owner perform work; no
second production store, queue or notification daemon is introduced.

`project.character_cast.autocast` is the additive v1 receipt for the automatic
run: its key, source revision, content hash, pipeline version, status, attempt
count and the roles it queued. It is the only durable record of "this source has
been cast", so the trigger is recoverable rather than remembered.

### One identity unit

Every speaking segment resolves through speaker attribution to exactly one
canonical `characterId`. `Segment.role_id` and `Attribution.speaker_id` both
hold a canonical `Role.id`; `Attribution.label` keeps the analyzer's raw speaker
label as evidence only. `character_identity` is the single resolver: it follows
the project alias map, and maps a legacy raw label onto a role **only when
exactly one role can mean it**. A representational difference — an older project
that stored the raw name, or a held utterance whose label now resolves — is
reconciled automatically at the authoritative write and on the next cast
refresh. A genuine ambiguity fails closed *for its own segment*: it is reported
as Needs Attention with both stored values as evidence, and every unaffected
character continues into Auto Cast. One ambiguous speaker can never blank the
whole cast.

Voice Lab reports a real workflow stage (extracting, segmenting, resolving
characters, casting, awaiting review, ready for audio, needs attention) instead
of collapsing every pre-audition state into "0 characters".

A local live process retains the projects-root lock even when its heartbeat
is old. Dead-owner recovery must establish that the recorded owner is gone;
an ambiguous process query cannot authorize a second writer. Test application
lifespans use disposable project roots so recovery tests cannot scan live
casting work.

The fingerprint separates effective synthesis from approval provenance. The
new strict workflow pins local checkpoint/adapter/runtime inputs; favorites,
notification state and unrelated role edits are excluded. Existing historical
Takes are not rewritten or eagerly regenerated. Incompatible old Takes fail
reuse checks in the new strict scope; compatible unaffected Takes remain reusable.

## Provider evidence and limits

The local capability record distinguishes documented controls from acoustic
verification. Qwen CustomVoice exposes nine validated presets. Missing/unknown
speakers are rejected by the client and native/accelerated workers. Requests
are serialized one utterance at a time, with exactly one output required;
mixed batch broadcasting is not exposed. The accelerated path retains its
existing graph, EOS and cancellation protections.

The installed native Qwen package was 0.1.1 at discovery; Base, VoiceDesign and
CustomVoice 1.7B manifests/assets were present. No 0.6B checkpoint or optional
speaker-embedding evaluator was installed for this change. Checkpoint hashes
in the capability descriptor are manifest-pinned expectations, not a claim that
every local weight byte was rehashed during each request. Live model acoustics,
voice distinctness and accent compliance require the separate runtime/listening
exercise. Upstream references: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
and [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts).

## Private phone review

The existing private Tailscale service remains Brian-only. A new AutoCast
project uploaded through that authenticated service receives a canonical
Brian-only review grant. Other projects require the existing explicit sharing
configuration. Paths are containment-checked, MK is unchanged, credentials stay
server-side, and no public tunnel is added.
The inbox defaults to characters needing action, retains local drafts until
acknowledgement, and shares one normal-speed player with immutable version URLs.
The existing notification ledger delivers four durable Auto Cast review events:
`CHARACTER_REVIEW_READY`, `REPLACEMENT_REVIEW_READY`, `CASTING_COMPLETE` and
`CASTING_NEEDS_ATTENTION`. A review event exists only once a round reaches
`awaiting_review` with playable, verified candidates, so queued work never
notifies. Several characters finishing inside the coalescing window become one
message; each event key is bound to its project, character and round revision,
so reprocessing the same state, a browser refresh or a server restart can never
produce a second effective push, while a genuinely new replacement round can.
Delivery is at-most-once per event through the existing ledger and never gates
casting: a transport failure leaves the audition, rating and approval intact.
A character waiting for feedback is never left unannounced:

- **Every round is its own notification.** The ledger's artifact identity includes
  the round, so a re-cast that happens to reproduce identical audio is still
  announced instead of being reported as "already notified".
- **A provider outage does not end the retries.** The first attempts back off at
  one, two and three minutes; after that the interval holds at 30 minutes
  (`EXHAUSTED_RETRY_SECONDS`) until the provider accepts the push. An ambiguous
  `delivery_unknown` is still never blindly resent, because it may already have
  reached the phone.
- **One slow voice cannot hold back the rest.** A round still producing a sibling
  is announced once its playable auditions have waited ten minutes
  (`PARTIAL_REVIEW_SECONDS`).

Each push deep-links to
`/voice-lab?project=<id>&character=<roleId>` (or `&view=casting` for an
aggregate or final message) and refuses to send a loopback review URL. Provider
accepted, delivery unknown, page opened, playback and feedback are separate
facts. A successful API call does not prove phone receipt or listening.

### Where the link lands: "Where is my book?"

The casting page always answers one question first, from `GET
/voice-lab/casting/book` (`autocast_book.book_status`). It shows exactly one of:

- **Voices waiting for you**: characters with a round awaiting review, each a
  card to audition and rate.
- **Your book is being recorded**: a whole-book progress bar (lines recorded of
  lines in scope, from the production job's completed segments), a second bar for
  the chapter being recorded with the time until it finishes, and the chapter list.
  Each chapter becomes playable the moment its last line is recorded.
- **Your book is ready**: every chapter, playable and downloadable.
- **Your book needs a decision** (blocked): the reason, in words, with the choices
  below. A blocked book is never shown as recording or finished.

A book is only reported finished when every chapter in its frozen plan has a
verified recording; a succeeded production receipt with a chapter missing is
reported as incomplete.

### Time until the chapter is finished

The time left is measured, never assumed. The worker records how long each line it
really synthesizes took, beside the line's length, in
`casting/production_pace.json` (a small file next to the project, because rewriting
`project.json` per line costs more than a line of speech). The estimate applies that
seconds-per-character over the last 40 lines to the characters still to record in the
chapter (and, separately, in the book), and subtracts the time the line in flight has
already taken. Until three lines have been measured in the current run the page says
"Estimating time left…" instead of showing a number. A reused Take is instant and
says nothing about synthesis speed, so it is never counted, and a new run (for
example after a restart) measures its own pace. The time is repainted in place on
every poll, so a countdown never rebuilds the panel or moves keyboard focus.

### Why validating a running book is cheap

A frozen manifest proves the audio being recorded still belongs to the cast a person
approved. Re-deriving it from scratch (parse the project and every segment, resolve
every speaker, rebuild every requirement) costs about seven seconds on the acceptance
book. It used to run three times before every line, one of them while holding the
project lock: about twenty seconds of checking per line. Validation is now in two
tiers:

- **Per line (`manifest_still_current`, about 13 ms).** It works on state
  `render_utterances` has already read for that line: every frozen approval and
  assignment, one digest over the whole book's source (so an edit to any line stops
  the run), and this line's own synthesis requirement. Inside the project lock only
  the approvals are re-read (`cast_current`), because that is what can be revoked
  mid-render; the merge judges the Take itself against current state.
- **Full (`manifest_current`).** It runs where audio is published: at the start of a
  run, as each chapter is assembled, and at the final receipt. Nothing is announced
  or called finished without it.

The worker itself reads the project and the book once per run, not once per line.

**Walking lines that are already recorded.** A restart makes the worker walk every line it
recorded before. That used to hand each one to `render_utterances`, which re-reads the
project and the book per line (about 1.4 s of CPU on the acceptance book, so an hour of
walking with the page unusable). A line that already has a selected Take is now
recognised on one snapshot, read-only (`_RecordedLines` in `casting_review.py`): the same
Take check as resume (audio hash, synthesis requirement, QA identity, with no re-audit and
no writes) and the same per-line manifest guard. The snapshot is built the first time it is
needed, so a run that is recording new lines never pays for it, and it is read again as soon
as `project.json` or the segment file changes on disk. A line that is anything other than
cleanly valid goes through `render_utterances` exactly as before, and each chapter is still
re-proved from its Takes when it is published.

### The listening file

Chapter MP3s are encoded for audiobook platforms (ACX/Audible): 44.1 kHz, 192 kbps constant
bit rate, mono, peaks at or under -3 dBFS. The encoder (`_listening_mp3` in
`autocast_book.py`) resamples with soxr and puts a lookahead limiter at about -3.5 dBFS in
front of the encoder, so the MP3's own overshoot cannot push the decoded peak past -3 dBFS.
The limiter does nothing to speech below that level, so loudness is unchanged (the
acceptance chapters sit at about -19 to -20 LUFS, inside ACX's -23 to -18 dB RMS window).
Only the delivered copy is processed: `clean-master.wav` stays the lossless join of the
approved Takes, and no Take, voice or manifest is touched. Chapters published before this
change keep their original files.

### One push per finished chapter

As each chapter's last utterance is recorded, the production worker assembles it
from the approved takes (`autocast_book.export_chapter`: scene breaks become
recorded pauses, never speech; QA, loudness and duration are checked against the
current frozen manifest) and publishes it as an immutable recording. It then
raises a `CHAPTER_READY` event and the same durable ledger sends
"🎧 New chapter ready" naming the book's own chapter title, its length and how
many chapters are ready, linking to `&view=casting&chapter=<id>` so the page opens
straight onto that chapter. Each chapter is announced once (its event key is the
recording's digest), a provider outage is retried like any other push, and a
notification failure never undoes a recorded chapter. A chapter that fails to
assemble is retried once; if it still cannot be built the book is marked failed
naming that chapter instead of being called finished.

### Lines nobody has attributed

Auto Cast never lets the narrator stand in for a character on its own, so lines
the analyzer could not place hold the book back. The page names how many there are
and offers explicit, recorded, human-authoritative choices
(`POST /voice-lab/casting/production/resolve`, idempotent per request id):

- **Accept the suggested speakers and skip the rest** (`auto`, the primary choice)
  applies every real proposed character and leaves whatever is still unresolved out
  of the recording. A placeholder label ("unknown", "unattributed", "speaker", …) or
  a placeholder role with such a name is never treated as a suggestion, even when a
  role of that name exists in the project.
- **Read the rest with the narrator** attributes every remaining line to the
  project's single narrator character.
- `suggested` (accept only) and `skip` (skip only) are the two halves of `auto`.

A choice is recorded as the person's own decision, and the blocked production is
**re-planned in the same step**: its frozen scope and source binding described the
book before the decision, and a blocked production is otherwise never looked at
again, so without this a decision changed nothing until the service was restarted.

### Skipped lines

A line can be left out of the recording on purpose, and that is never silent: each
skip is stored durably (`autocast_import.skipped`) with a reason and who made it, and
the page shows how many lines were skipped, which ones, and why.

| reason | meaning |
| --- | --- |
| `unknown_speaker` | no speaker could be found for the line |
| `unspeakable` | the voice model refused the line |
| `web_address` | the line is a link, or the voice refused a line because of the address in it |
| `failed_qa` | the recording kept failing its quality checks |
| `human` | you skipped it |

Skipped lines are outside the spoken scope everywhere it is decided (scope,
manifest, chapter assembly), so nothing waits for their audio, and a chapter whose
every line is skipped disappears from the book. A skipped line can be brought back
(`POST .../production/restore`, per reason or per line), which re-opens even a
finished book for just those lines. A line skipped for having no speaker is
remembered as one you want to decide yourself, so automatic handling does not skip
it again. A line that is already recorded cannot be skipped, and skipping or
restoring waits while the book is being recorded.

**Web addresses are never read aloud** (`web_addresses.py`). Web serials carry links
("Table of Contents URL: https://...", "Join the Discord: https://..."), and a voice model
cannot read one: it runs out of its generation budget and refuses the line, which used
to stop the book. An address is found even after the import spelled its numbers out
(`/fiction/one hundred four thousand.../`), and then:

- **Cleanup, at import.** A line that is only a link (an address with at most a short
  label for it) is skipped; an address inside a sentence is taken out of that sentence's
  spoken text, with the word that pointed at it ("at", "see", ...). The source text stays
  exactly as imported.
- **Generation.** A link line still in a book (one imported before this existed) is left
  out before the voice is asked; a sentence with an address that the voice refuses is
  left out too. Both happen whatever automatic handling says, and neither counts against
  the guard that stops a run when the voice service itself seems to be failing.

A web-address line you bring back is yours to decide about: it is tried, and if the voice
refuses it the book waits for you as with any other refused line.

### Attributing speakers from the story

When lines have no speaker (or the analyzer's guess is wrong), the speakers can be worked
out from a reading of the scene instead of skipped or read by the narrator. Studio
offers this to its own tooling as `POST /voice-lab/casting/production/attribute`; the
phone-facing service does not expose it. A request carries `assignments` (line to
character), optional `new_roles` (characters the analyzer never found) and `aliases`
(a duplicate merged into the character it really is, such as "The Librarian" into
"Filaurel"), and is idempotent per request id. It:

- records each attribution as the person's own decision, replacing the analyzer's
  guess, and never hands narration to a character or a placeholder ("unknown",
  "narrator", "speaker") to anyone;
- creates each new character the way the analyzer would, refreshes the story-derived
  dossiers and casting profiles, and **restarts the cast call**: a casting round is
  opened for each new character and the casting run is re-opened to include them, so the
  progress counts, the "voices ready" pushes and the completion event describe the
  whole cast;
- applies aliases after that refresh (which recomputes them), so one character has one
  voice;
- re-plans the production, so the book waits for the new voices to be chosen and then
  records every line by itself. Nothing is recorded with a stand-in voice.

The earlier "Accept suggested speakers" guesses are not trusted blindly: in the
acceptance book about half of those labels were wrong once read against the scene
(speaker tags smeared across alternating lines), and were corrected this way.

### Automatic handling

Automatic handling (`autocast_import.attribution_policy`, on by default for a new
book, switchable from the page or with `POST .../production/policy`) means the book
does not stop for these lines:

- Before the maintenance pass reads the production, real suggested speakers are
  accepted and the remaining unresolved lines are skipped, so a book never reaches
  "needs attention" (and never sends that push) for lines you already said how to
  handle.
- While recording, a line the voice model refuses (its generation-budget refusal) or
  whose recording keeps failing quality checks is tried twice and then skipped. The
  skip and the re-frozen manifest for what is left are committed together, so the
  rest of the run is always validated against exactly what is being recorded. With
  automatic handling off, the run stops at that line instead and names it.
- **It never skips past a service that is failing.** Anything that says nothing about
  the line itself (the voice service unreachable, a cast that changed under the run)
  stops the run instead. A run also stops after five refused lines in a row, or after
  skipping more than 5% of the book (at least ten lines), and remembers the line it
  stopped at, so the page can offer "Skip this line" or "Try again".

## Performer anchoring, deferred replacement and preferred voices

Three extensions sit on top of the existing inbox, dossier and recast work.
None of them introduces a second store, a second assignment path or a second
audition pipeline.

**Anchoring.** Every newly proposed candidate is grounded in one *specific*
performer and performance from the performer pool, chosen from the character's
dossier-derived profile through `casting_anchors`. The proposal carries the
performer, the performance, the traits that matched, the rationale and the
catalogue version, so the review surface can say *why* a voice was proposed.
Anchors are deterministic for a given profile, so a refresh or a resumed round
cannot reshuffle its own evidence, and the design instruction both names the
reference and requires an original voice that does not reproduce the recording.

**The performer pool.** The pool is a hundred validated references, not a handful
of names: ten hand-written seeds in `performer_discovery` plus ninety documented
credits in `performer_catalog_sources.json`, spanning masculine and feminine
presentations, young to mature ages, and British, American, Irish, Scottish,
Welsh, Australian, Canadian, African and South-Asian accent families across
audiobook narration, documentary, broadcasting, animation and stage work. Every
row is validated by `performer_catalog.normalise`, which refuses a reference with
no specific performance, no retrievable credit URL, no stated fact or no
recognised casting tag — a famous name without evidence is the generic "AI voice"
problem wearing a name, and it is rejected rather than ranked. The pool can also
grow at runtime: `performer_catalog.adopt` adds a performer whose credit evidence
has been retrieved, into a separate `performer_catalog_adopted.json` so an upgrade
never overwrites discovered work, and refuses a second adoption of the same
identity rather than silently editing the evidence a cast was built on. A row
refused at load time is reported in `CATALOG_POOL["refused"]`, so a shrinking pool
is visible instead of being cast from quietly.

**One book, different voices.** The pool alone is not enough: the highest-scoring
performer for a trait set would otherwise be proposed to every similar character,
so a ten-character book could end up with the same three names repeated.
`casting_review._cast_performers` collects the performers already cast or proposed
to *other* characters in the project, and the round planner ranks them last. This
is a preference, never a hard exclusion, because a cast large enough to exhaust the
pool still needs anchored candidates — an unanchored voice is the worse outcome.
Within one round the exclusion *is* hard, so the three options offered for a
character never collide with each other.

**Casting inference, not generic fallback.** A dossier that establishes only one
or two dimensions ("young male", or nothing at all) no longer produces a
generically described synthetic voice. `casting_inference` supplies the *missing*
dimensions for each of several differently emphatic cast concepts — warmth and
vulnerability, confidence and wit, depth and gravity, youth and energy, age and
texture — and each concept resolves its own performer, so the options differ by
casting dimension rather than by name alone. Two rules keep this honest: a value
from the book and a casting interpretation are never mixed, and the interpolated
dimensions carry `level = CASTING_INFERENCE` with the basis for the
interpolation and no evidence ids. The review surface shows *Book-supported
traits* and *Casting inference (not in the book)* as separate groups. With
inference in place, a candidate that genuinely cannot be anchored is a **blocked
fallback** rather than a normal proposal: it states why (no fitting performance,
or every fitting performer already used by another option in the round) and
`voice_lab.design` refuses to generate it.

**Acoustic characterization: the reference changes the sound.** A performer name
in a prompt is prose. Before synthesis, each candidate's reference is converted
into a numeric acoustic target and the generated original reference is *shaped
onto it*:

1. `performer_reference.resolve` turns the anchor and its concept into a
   reference-evidence record (performer, performance, source URLs, source type,
   retrieval state and timestamp, rationale) and a numeric `conditioning`
   payload: target median pitch and range, speaking rate, pause behaviour, level,
   dynamic range, spectral brightness, breathiness, rasp, articulation,
   sentence-ending slope, resonance, vocal weight and age impression.
2. Charactersation is labelled. Where permitted reference audio exists for the
   anchor (an eligible `performer_discovery` acquisition in the project), the
   profile is **measured** by `service/voice_acoustics`. Where no permitted audio
   exists, every value is a documented prior under `inferred` and the reason is
   recorded — the layer never silently downgrades to name-only prompting.
3. The worker (`CharacterVoiceRuntime.design` → `acoustics` → `prepare`) designs
   the original reference, measures it, derives a bounded shaping plan against
   the target, applies it to the audio, re-measures the result, and only then
   extracts the Base prompt — so the prompt encodes the *conditioned* reference.
   A conditioned identity cannot be prepared without its conditioning pass.
4. The receipt states what was achieved: the applied semitones and rate, the
   pre- and post-conditioning profiles, and a three-valued reachability verdict.
   The correction is bounded (±3 semitones, ×0.85–1.18 rate) because larger
   changes cost more in artefacts than they recover in accuracy, so a target
   outside the bound is reported as a shortfall rather than presented as reached.
   Reachability is split into `reached`, `shortfall` and `out_of_scope`: a
   documented prior carries keys no pitch/rate transform can move (peak level,
   voiced fraction, sentence-ending energy drop), and counting those as either
   reached or missed would make the actuator's scope into a verdict about the run.
   A comparison that touched no controllable key reports `could_not_measure`,
   never a pass. Rate metrics are derived from the pre-stretch measurement and the
   applied factor rather than re-detected, because a time-stretch's own segment
   duplication is counted as new syllables by envelope-peak detection; both the
   raw and derived values and the substitution are recorded.

Candidates whose recognized dimensions are already answered by a previously
approved voice are ordered after more complementary concepts, so a reused voice
competes with exploration instead of crowding it out.

The layer is exercised against the real acceptance project by
`tools/run_acoustic_acceptance.py` (live GPU worker) and reported by
`tools/build_acoustic_acceptance_report.py`; the report for `ba9786ea` is at
`runtime_evidence/2026-09-28-autocast-elf-acceptance/acoustic-conditioning/acceptance-report.md`.

**Deferred replacement.** A 1-3 star rating records the rating, the qualitative
feedback and an explicit replacement request, and then *leaves the round
exactly as it was*. The reviewed batch stays stable while the human scores the
remaining options: same round, same candidates, same audio, same controls. A
single durable replacement request per character accumulates the retained
direction, and the maintenance owner promotes it into its own later round only
once every playable candidate in the reviewed round has been rated. An explicit
`Recast`/`Preview` without a low rating, and `Try new voices`, remain immediate.
An approved character releases its queued request as `superseded`.

**Automatic production.** Approving the last required character is the last
action a human has to take. The same maintenance owner that starts casting also
calls `casting_review._automatic_production`, which creates the production receipt
keyed by `AUTOMATIC_PRODUCTION_ID` once — and only once — every required character
in scope is approved. There is no button to press and no view to open: the
`CASTING_COMPLETE` push now says generation is starting, and the finished render
emits its own `PRODUCTION_READY` event, which reports success with the manifest, or
reports the bounded failure with its reason and asks for an explicit resume, or
reports `blocked` when the scope cannot be frozen at all. `blocked` is the honest
outcome for a **partially attributed book**: the importer parks dialogue it cannot
resolve, and rendering only the attributed lines would ship an audiobook silently
missing the rest. The refusal names the population rather than the lookup that
failed — how many of how many utterances have no resolved speaker, split by how
many carry a proposal a human can confirm and how many have no speaker evidence
and must be attributed from the text. Because a block is terminal for the
promotion loop, the reason carries a `reason_version` so a later, better diagnosis
can reach the human once, and a block clears itself as soon as the scope becomes
freezable, without a manual resume.

Guarded deliberately: a partly-reviewed cast is never rendered (unapproved
characters would be produced with placeholder voices), a receipt that already
exists is never duplicated, and a terminally failed render is not retried in a loop
— resuming is the human's decision.

**Preferred voices.** A rating of 4 or 5 stars publishes the immutable voiceversion into the existing global voice library
(`storage/voices.voice_lab_preferred`), together with its traits, its
inspiration, the characters and projects it has served, and the ratings that
produced the preference. A 1-3 star rating is recorded on the same version but
never promoted, so history is complete and the pool stays clean. When a new
character dossier arrives, matching prior voices are proposed *alongside*
freshly generated anchored candidates, with their prior rating, fit percentage,
matched traits and previous use stated. A prior voice is only ever a candidate:
it is never assigned, and its exact audio enters the project through the
existing library import when the option is actually auditioned. Failures there
block that one option with its reason rather than substituting a different
voice. The stable identity of a reusable voice is its library id plus version,
so `voice V -> 5 stars in project A -> candidate for a different character in
project C` never conflates the two characters.

## Earlier listening-test decision

This mission authorizes bounded casting auditions independently of the old
five-minute Chapter 1 acceptance test. The old source blocker remains valid.
The earlier `more-like` discovery/replenishment path keeps its historical gate.
The new inbox has its own explicit Prepare/Recast/Preview actions; GET/poll/play
never dispatches audio. See `openspec/changes/harden-human-casting` for the new
contract and `add-automatic-voice-lab` for the separate historical requirement.
