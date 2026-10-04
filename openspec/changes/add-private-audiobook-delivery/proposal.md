# Private audiobook phone delivery

Brian needs to review completed Studio chapter exports on his iPhone. Existing output files and Pushover credentials exist, but there is no reusable private chapter player or durable feedback binding to those audio versions.

Add a downstream delivery projection of verified listening exports. Recheck canonical selected Takes, existing output-bound QA, chapter coverage, export fingerprints and actual bytes. Publish an immutable review batch under the Studio project, expose only manifested audio through a loopback service behind Tailscale Serve, and reuse the existing trip-research Pushover transport. Capture chapter/timestamp and overall observations without mutating Takes.

Explicit operations prepare, start, deliver --recipient brian --ready, status, and feedback support repeat use. Record notification intent before transmission; accepted, pending and uncertain outcomes suppress repeated sends. A deliberate --force/--resend permits intentional notification replay. MK remains disabled in V1. Optional Taildrop uses only the configured, freshly verified same-account iPhone.

Non-goals: synthesis, canonical state migration, QA changes, public hosting, a second audiobook database, background automatic notifications, or enrolment of another recipient.

Authorization: the user explicitly requests implementation and real delivery, including private endpoint setup and Brian-only Pushover/Taildrop. No existing Studio OpenSpec process/directory was found during read-only discovery; this scoped change records the implemented contract locally without altering the separate trip-research specifications.
