## ADDED Requirements

### Requirement: Persistent desktop access
Voice Library SHALL offer desktop and Windows sign-in shortcuts that use the existing Studio and private review services, preserve other processes and configurations, and perform no synthesis on launch.

#### Scenario: Double launch after sign-in
- **WHEN** the owner opens the desktop shortcut while the matching Studio is running
- **THEN** it reuses the same process and opens the library without starting a GPU job

### Requirement: Local book narration
The creator SHALL accept bounded eligible local EPUB/TXT files, preserve reading order and source provenance, create isolated canonical projects, and generate selected chapter MP3s using an exact saved narrator version and the existing production pipeline.

#### Scenario: Interrupted MP3 job is resumed
- **WHEN** the same request is resumed after interruption
- **THEN** completed verified Takes are reused and existing production selections are not overwritten

### Requirement: Short production auditions
Each role-specific match SHALL offer explicit approximately ten-second natural-pace production previews, cached playback and separate exact-version assignment. Preview generation SHALL NOT change a book's cast or inherit preferences.

#### Scenario: Owner replays or reopens a preview
- **WHEN** a verified preview exists for the same role, profile and candidate direction
- **THEN** playback uses its saved audio without generation, and Use assigns only the version actually heard

### Requirement: Shared owner and private boundary
Private access SHALL retain the existing Brian-only boundary, generated URL translation, bounded uploads and hash-verified audio ranges. Worker startup SHALL preserve active production, provider defaults and existing output files.

#### Scenario: Another worker starts during a preview readiness check
- **WHEN** a managed worker appears before the startup operation acquires ownership
- **THEN** the preview refuses replacement under the existing process lock and remains resumable
