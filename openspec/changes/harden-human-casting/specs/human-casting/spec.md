## ADDED Requirements

### Requirement: Scoped genuine approval
Studio SHALL park strict production until each actual canonical speaking role
in the requested source scope has a current genuine human character-fit rating
of at least four stars and an explicit assignment decision on the heard version.
Library favorites, legacy ratings and technical QA SHALL NOT create approval.

#### Scenario: Rejection and replayed feedback
- **WHEN** a human rates the current assignment at most three stars
- **THEN** one durable replacement round is created and that assignment is rejected
- **AND** replaying the same event creates no second operation
- **AND** the replacement has no inherited stars and unrelated roles remain approved

#### Scenario: Old tab after approval
- **WHEN** feedback names an obsolete round or assignment revision
- **THEN** it is saved as history with a conflict and cannot replace the current cast

### Requirement: Exact voice dispatch and immutable auditions
Studio SHALL use one stable conditioning identity for each candidate's versioned
audition and SHALL reject missing/unknown provider voices and wrong output cardinality.
Requested unsupported controls SHALL remain visible without repeated synthesis.

#### Scenario: Standard scripts
- **WHEN** an initial or replacement round is requested
- **THEN** candidates share the respective 60-second or 30-second versioned spoken script
- **AND** actual decoded durations, outliers, section positions and Take hashes are retained

### Requirement: Durable guarded resume and export
Strict jobs SHALL release synthesis resources while awaiting human review,
revalidate source and cast before resuming, and retain obsolete in-flight Takes
without publishing them as current. Assembly SHALL verify the current frozen
manifest and selected Takes before assembly and final publication.

### Requirement: Private review projection
The existing private Voice Lab SHALL present action-needed characters, atomic
feedback controls and immutable audio. Notifications SHALL use the existing
Brian-only ledger and distinguish provider acceptance from device/listening evidence.
