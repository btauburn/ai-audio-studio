## ADDED Requirements

### Requirement: Delivery preserves canonical selected audio
The delivery operation SHALL include only complete, currently selected, reviewable audio with verified output hashes and existing blocking QA enforced, without synthesis or Take mutation.

#### Scenario: Selected or QA state changed
- **WHEN** selection, canonical revision, coverage, QA binding or output bytes differ from the export receipt
- **THEN** the artifact SHALL NOT be admitted as current reviewable audio

### Requirement: Listening is private and seekable
The listening service SHALL authorize Brian through the private Tailscale identity boundary and serve only explicitly manifested files, with streaming, byte Range and individual downloads.

#### Scenario: Phone listens privately
- **WHEN** Brian opens a batch URL through Tailscale
- **THEN** he sees a mobile player for the exact chapters and may play, seek and download

#### Scenario: Unauthorized or traversal request
- **WHEN** a request lacks the allowed identity or requests an unmanifested path
- **THEN** access SHALL be refused

### Requirement: Feedback belongs to the heard version
Feedback SHALL persist project, chapter, exact artifact hash, revision, creation time and any playback timestamp without changing production Takes.

#### Scenario: A newer audio revision is delivered
- **WHEN** feedback exists for an earlier batch
- **THEN** it SHALL remain bound to the earlier artifact and SHALL NOT appear as feedback on the newer revision

### Requirement: Notifications are explicit and idempotent
Delivery SHALL use the existing Pushover transport and record durable intent before sending. Accepted, pending and unknown artifact deliveries SHALL block automatic duplicate sends. An explicit force operation MAY intentionally resend.

#### Scenario: Ambiguous provider outcome
- **WHEN** a request times out, returns a server error, or has an ambiguous success body
- **THEN** its outcome SHALL remain unknown and SHALL NOT trigger an automatic retry

### Requirement: Brian is the sole V1 recipient
Brian SHALL be the only enabled recipient; MK SHALL remain disabled. Provider acceptance, direct-file transfer and physical phone observation SHALL be reported separately.

#### Scenario: Disabled recipient requested
- **WHEN** delivery is requested for MK
- **THEN** no notification or file transfer SHALL be attempted
