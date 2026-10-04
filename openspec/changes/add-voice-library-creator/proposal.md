# Voice Library access, book MP3 creation, and short casting previews

## Why
The existing library requires manually started services, and performance suggestions do not provide an obvious short listen-and-select flow. The owner requested reboot access, local book drag/drop, and about ten-second character previews.

## What Changes
- A desktop shortcut opens the existing Voice Lab and reuses or starts Studio. A sign-in shortcut starts its services quietly, without GPU generation.
- Voice Lab accepts local EPUB/TXT books and creates chapter MP3s with a chosen immutable saved narrator version through canonical Studio jobs and Takes.
- Each role-specific suggestion offers an explicit short preview, cached playback, and a separate exact-version assignment action.
- The existing Brian-only private route exposes the same controls, bounded uploads and hash-bound ranged audio.

## Impact
Extends existing UI, provider startup, project/source storage, durable jobs and delivery proxy. No new queue, daemon, production database, model installation or external hosting. Existing books, cast selections, voice preferences and audio remain intact. Chapter 1's earlier source blocker is unchanged.
