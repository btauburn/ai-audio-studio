# Design

## Authority and revision boundary

Studio project.json, segments.json, selected Takes and their hash-bound executed QA remain the authority. Configured export receipts are candidate evidence. The projection verifies canonical snapshot hashes, ordered coverage, selected IDs, source WAV hashes, QA identity, explicit scene breaks, final MP3 hashes and export fingerprint. Unselected, stale, incomplete, blocked or changed outputs fail closed. Receipts from previous canonical snapshots remain excluded from current discovery.

The delivery manifest contains no source text. Immutable batch identity includes project, recipient, chapter, audio hash and revision. Old review URLs remain tied to old audio; changed bytes are refused. Feedback observations bind project, batch, chapter, revision, artifact hash, export receipt hash, timestamp and creation time. Overall notes bind every artifact in the batch. No Take is modified.

## Privacy

Standalone FastAPI binds 127.0.0.1 only, with proxy headers disabled. Tailscale Serve HTTPS authenticates devices and supplies its overwritten Tailscale-User-Login header. The service checks loopback peer, Host and Brian's configured login. MK is structurally configured but cannot be enabled in V1. Neither Funnel nor router ports are used. Production service and renderer processes are untouched.

Only manifest keys resolve audio; files must remain inside operator-configured export roots. Each response copies bytes into a bounded-memory private spool, hashes it, then streams that exact handle with single byte Range, suffix Range, HEAD and If-Range support. A concurrent source replacement cannot substitute unverified bytes after the hash check. Invalid/unsatisfiable and multipart ranges return 416.

Feedback accepts JSON only, requires a custom request header, validates Origin, limits body to 16KiB and text to 4000 characters, and accepts finite timestamps from zero through chapter duration. Exclusive atomic JSON publication stores immutable observations in the existing Studio project directory. There is no user-controlled output path.

## Notification and delivery state

Import the existing Pushover publisher through a configured local checkout; do not execute travel decisions or reuse its trip ledger. Wrap only its response classification to conservatively preserve server errors and ambiguous successes as unknown, avoiding ambiguous retries. Credentials remain in the existing private credential store. Provider responses are allowlisted into safe receipts.

Studio's writer lock primitive is scoped to the delivery notification ledger, with a thread lock around process-reentrant locking. Durable claims precede external requests. Missing/uncertain outcomes and overlapping accepted artifacts block duplicate sends; explicit force is recorded. File-transfer receipts are separate and preserve uncertainty. Provider acceptance never sets phone-observed status.

## Operations

Private configuration selects project, export glob(s), allowed audio roots, transport checkout and identity. Run prepare to admit existing audio, start to launch the loopback service, and deliver --recipient brian --ready for endpoint verification and explicit send. Operator-installed user logon startup restores the service; Tailscale Serve configuration persists. No notification is sent at startup. Future completed revisions can be included through the configured export paths, and automatic triggering remains a later governed decision.
