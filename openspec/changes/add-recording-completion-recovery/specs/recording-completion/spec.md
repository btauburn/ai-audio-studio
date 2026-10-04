## ADDED Requirements

### Requirement: Visible recording library
Studio SHALL show all existing published book, Voice Lab and private delivery
recordings in a searchable home library with Play, Download and Review links.
The private projection SHALL retain existing project sharing restrictions.

#### Scenario: Open the completed Snow recording
- **WHEN** Brian opens the completion link
- **THEN** the exact saved recording is selected without synthesis or autoplay
- **AND** individual chapter and combined outputs remain available

### Requirement: Bounded durable recovery
New book creator requests SHALL enable two automatic retries after eligible
interruptions, with 30 and 120 second backoffs, through existing job ownership.
Cancelled, paused, stale or ambiguously active work SHALL NOT be automatically
redispatched. Finished verified Takes SHALL be reused without recasting.

#### Scenario: Confirmed failure interrupts an unfinished utterance
- **WHEN** the worker has returned a confirmed failure and production is idle
- **THEN** the same request resumes after its backoff with at most two retries
- **AND** a third automatic retry is refused

#### Scenario: Owner restarts during an uncertain external operation
- **WHEN** a durable in-flight marker has no matching completed Take
- **THEN** recovery holds and explains the uncertainty instead of duplicating work

### Requirement: Artifact-bound completion pushes
New book creator requests SHALL notify only Brian using the existing private
transport after all requested exports succeed and pass actual file validation.
Notifications SHALL link the exact saved output, retain durable deduplication,
and distinguish provider acceptance from phone receipt. Historical requests
SHALL require an explicit notification request.

#### Scenario: Repeated completion check
- **WHEN** a provider already accepted a push for these artifacts
- **THEN** subsequent checks do not resend it

#### Scenario: Unknown transport outcome
- **WHEN** the transport outcome cannot establish whether a push was accepted
- **THEN** automatic resend is held and the status remains visible

### Requirement: Long-running progress
Book progress SHALL remain recoverable after refresh, temporary network errors
and renders longer than 30 minutes, with automatic recovery and push status
visible separately from technical QA and human review.

#### Scenario: Overnight completion
- **WHEN** a three-hour request finishes while the page is hidden or closed
- **THEN** reopening displays its durable completed status and recordings
