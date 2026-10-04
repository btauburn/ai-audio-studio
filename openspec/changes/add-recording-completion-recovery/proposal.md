# Completed recordings, recovery and push notifications

Snow I and II completed overnight, but the home page hid their MP3s behind an
imported-book chooser and stopped polling long jobs after 30 minutes. Surface
published recordings on the existing home page, keep durable progress visible,
resume eligible interrupted book jobs with bounded retries, and send Brian an
existing private Pushover notification linking to the exact completed recording.

This extends Studio's canonical requests, jobs, selected Takes and private
delivery transport. It adds no database, independent queue, daemon or hosting.
Historical requests remain opt-out unless explicitly requested. The user's
September 27 instruction authorizes completion pushes and automatic recovery
for new book creator requests, replacing the former manual-only recovery rule.

## Impact

- Book request recovery and notification metadata; existing owner lifecycle.
- A read-only catalog of published creator, Voice Lab and delivery recordings.
- Private proxy controls, responsive home library and long-running progress.
- Exact-artifact notification deduplication, safe handling of uncertain inference
  outcomes, bounded failure tests and preservation evidence.
