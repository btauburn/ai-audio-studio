# Private audiobook review delivery

Delivery reads Studio's selected Takes and verified listening-export receipts. It never renders audio, chooses a Take, updates canonical metadata or approves a release Master. Feedback is an observation attached to the exact delivered revision.

Run from the Studio repository with its existing Python environment:

```powershell
.\.venv\Scripts\python.exe -m dubbing_service.app.audiobook.delivery prepare --recipient brian
.\.venv\Scripts\python.exe -m dubbing_service.app.audiobook.delivery start
.\.venv\Scripts\python.exe -m dubbing_service.app.audiobook.delivery deliver --recipient brian --ready --taildrop
.\.venv\Scripts\python.exe -m dubbing_service.app.audiobook.delivery feedback
```

`deliver` rereads canonical state, verifies selected source audio and finished MP3 bytes, publishes the immutable batch, verifies loopback and private HTTPS manifests and Range responses, rechecks current selection, then sends the notification. `--taildrop` adds optional direct files to the configured, verified same-account iPhone. Repeating the same batch suppresses duplicate notifications and file transfers. Use `--force` (alias `--resend`) only for a deliberate resend. A new revision has a different identity. When any requested artifact overlaps a previous accepted/pending/unknown notification, the entire requested batch is suppressed; form a batch of only newly reviewable chapters to notify just those.

Private configuration is `%USERPROFILE%\.config\audiobook-delivery\config.json`, outside source control. `--config PATH` before the subcommand selects another private file, and `AUDIOBOOK_DELIVERY_CONFIG` selects it for processes that do not take that flag (including a Studio started against a different projects root). Fields: `project_dir`, `book_title`, `export_manifests` (glob list), `allowed_audio_roots`, `notification_transport_root`, `listening_base_url`, `port`, optional `studio_base_url` (the loopback HTTP origin of the Studio this projection reads; default `http://127.0.0.1:9000`), and `recipients`. Brian's recipient has `enabled`, `tailscale_login`, `notification_transport: pushover`, `allow_download`, `allow_feedback`, and optional `taildrop_dns`. `mk.enabled` must be false. V1 deliberately rejects enabling other recipients; the config separates notification/access choices for later implementation.

The notification transport comes from the existing configured trip-research checkout. Its credential loader uses existing `PUSHOVER_APP_TOKEN`, `PUSHOVER_USER_KEY` and optional `PUSHOVER_DEVICE`. No keys are copied into Studio. `validate-notifications` uses Pushover's non-message validation endpoint to distinguish configuration presence from valid provider/device credentials. `accepted` means provider success, never proof that the phone displayed the notification.

Current receipt format is `verified_listening_export`, schema 1, with ordered selected Take/QA bindings, canonical `rendition_state_hashes`, `chapter_exports` and source fingerprint. The demonstrated `three-chapter-listening-export-v3` receipts satisfy it. Future exports must emit this verified receipt contract; merely putting an MP3 in the configured folder cannot make it reviewable. Old revision directories remain available for prior reviews; discovery admits receipts matching the current canonical bytes only.

`start` launches a hidden, detached loopback service on the configured port, default 8778. `serve` runs it in the foreground. `scripts/start_audiobook_delivery.ps1` is suitable for current-user logon startup. Startup does not send notifications. Tailscale Serve must route a private HTTPS origin to this loopback service (the installed instance uses a separate HTTPS port to preserve existing routes). No Funnel or router port is required. Brian's phone must stay connected to Tailscale while streaming; downloaded files can play offline. Windows must be awake to stream or save feedback.

Review records stay under the Studio project:

- `delivery/manifests/<batch_id>.json`: immutable chapter/audio/revision projection, no source text
- `delivery/notifications/{intents,outcomes}/`: safe provider receipts and duplicate claims
- `delivery/taildrop/`: separate direct transfer evidence
- `delivery/feedback/<feedback_id>.json`: immutable timestamped or overall review observation
- `delivery/events/`: projection, feedback and delivery operation observations

## Auto Cast casting review links

Auto Cast review notifications reuse this service and its ledger. The `voice_lab` page honours two deep-link parameters, so a notification opens the exact review instead of a landing page:

- `/voice-lab?project=<projectId>&character=<roleId>`: opens the casting inbox, scrolls the character into view, marks its newest playable audition as current, and leaves Play, stars and direction controls immediately reachable.
- `/voice-lab?project=<projectId>&view=casting`: opens the casting overview. Aggregate, `CASTING_COMPLETE` and `CASTING_NEEDS_ATTENTION` notifications use this form.

Opening or refreshing either link reads canonical state only; it never writes casting state and never causes a notification. The notification layer refuses to send a loopback review host, because such a link is schema-valid and still unusable on a phone.

The same mobile page shows saved feedback on desktop. The `feedback` command returns full artifact-bound records for governed correction work. No correction or regeneration is automatically started. Review inputs are limited to 4000 characters and 16KiB requests, and timestamps must be finite and within the chapter. Streaming uses a verified private spool so a source replacement cannot swap bytes between validation and delivery.

Verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_audiobook_delivery_projection.py tests/test_audiobook_delivery_notify.py tests/test_audiobook_delivery_server.py
.\.venv\Scripts\python.exe -m pytest tests
```

Provider semantics: [Pushover API](https://pushover.net/api). Private proxy identity: [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve).
