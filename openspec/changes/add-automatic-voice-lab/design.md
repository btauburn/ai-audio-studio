# Design

The implementation extends existing Studio ownership. `Project.voice_lab` holds
immutable reviewed voice snapshots, exact version/audio feedback, shortlist state,
the excerpt and discovery cursor. `Voice.voice_lab_versions` makes reusable voices
available through the existing global library. A favorite never assigns a Role.
Imported library versions start unrated and do not change a cast.

Only existing production Takes can supply review audio. A generated VoiceDesign
reference is conditioning, not a production audition. The first real Base Take
binds a previously empty version; the first feedback save freezes that binding.
Profile edits create new fingerprints. Explicit role assignment checks the current
identity and uses existing project mutation reservations.

Discovery ranks verified performance-credit references locally. Only public
performer names are sent to official fixed metadata endpoints. Raw character
descriptions, dossiers and source prose stay local. Cache/request/storage/retry
bounds and durable held records prevent uncontrolled acquisition. Recording
download rights and synthetic-voice intended-use rights are independently checked.
Eligible acquisition preserves original bytes, bounded crop/transcript/review
evidence and hashes. The existing Qwen worker rechecks these before Base prompts.

Chapter identity is proven against immutable original source bytes, metadata,
TOC, heading and a deterministic complete chapter extraction. Selection requires
contiguous exact offsets, stable utterances, alias-resolved speakers, actual
dialogue and interleaved narration. The persisted manifest precedes all GPU work.
Rendering reuses the existing pipeline, global GPU lease, durable job owner,
blocking QA and revision-aware Take commits. Assembly changes neither text,
speed nor interior content. Decoded output must be 285–315 seconds. Actual Take
provider metadata must prove the distinct conditioning identities used.

The review UI uses one audio element, exact audio hashes, local playback gain for
verified same-passage A/B and keyboard/touch controls. The existing Tailscale
delivery service checks the same recipient boundary and an explicit project
allowlist, then proxies to the canonical Studio process; it never writes project
metadata itself. Startup can resume bounded opted-in discovery, with production
jobs retaining priority. Speculative audio stays idle pending the first test.

No public hosting, notifications, subscriptions, new daemon, application database,
GPU fallback or existing production recast is introduced.
