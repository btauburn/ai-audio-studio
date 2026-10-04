# Bounded public model research in AutoCast

Status: implemented locally, tested with offline providers. No live OpenRouter calls, keys, new credentials, audio generation, grants or service restarts are authorized by this implementation checkpoint. Faster Qwen remains experimental and the global renderer default remains unchanged.

## Existing owner and authority

The source import records `metadata.autocast_model_research.requested=true` on future imports. The existing `casting_review.start_pending_autocast` owner runs research before `prepare` establishes new casting rounds. Missing model configuration enters the existing `needs_attention` state with an actionable reason. Existing projects without that requested flag keep their current workflow. Opening a page never requests research.

The same read-only `_keeper_proposals` decision is used before model access and when constructing rounds. Compatible approved historical samples are offered first with zero model/tool calls. Existing approved assignments remain authoritative. Explicit recasting does not manufacture a rating or acceptance. Keeper availability is rechecked immediately before creating rounds.

There is no independent casting queue. An in-process start mutex prevents concurrent starts, while the canonical durable run/request remains the authority. Slow research runs outside the global dispatcher lock and project metadata locks. Concurrent scans skip a run whose scope/research request is incomplete. Restart recovery re-enters the existing owner and its journal; an uncertain model completion becomes a needs-attention hold, never an automatic repeat charge.

## Exact provider setup gate

Studio reads the declarative `dubbing_service/llm_models.json` and `llm_providers.json` registries without migrating them or resolving unrelated credentials. The selected configuration is the project `autocast_model_research.llm_config_id`, or one explicit registry default. It must name the **exact** OpenRouter `deepseek/...` model ID and link to a Studio provider with base URL `https://openrouter.ai/api/v1` and a secure `keyring:` or `env:` credential reference. Direct DeepSeek aliases are never substituted. Only after those checks can the selected reference be resolved through the existing secret store.

The verified editor configuration registers OpenRouter but does not establish a selected OpenRouter DeepSeek ID or a callable Studio bridge. The existing Studio default is a direct DeepSeek alias with no linked provider; it fails this gate. No OpenRouter model ID has been guessed.

User setup, if separately authorized: in VS Code use **Chat: Manage Language Models**, choose OpenRouter, and supply only the exact DeepSeek model ID. Studio cannot automatically consume an editor registration. A direct Studio route requires the user to authorize separate registration and enter their existing key themselves in Studio Settings' LLM service/provider password field, using the OpenRouter base URL; then link a model configuration with the exact ID. Never paste the key in chat. This checkpoint does not register that provider or make a live call.

Official references: [VS Code model management](https://code.visualstudio.com/docs/agent-customization/language-models), [editor language-model API](https://code.visualstudio.com/api/extension-guides/ai/language-model), [OpenRouter tool calling](https://openrouter.ai/docs/guides/features/tool-calling).

## Public inputs and advisory outputs

Only a locally registered public work title/author, catalog performers/performances, and allowlisted abstract vocal traits enter the model prompt. Unknown work titles, role names, arbitrary IDs, aliases, dossiers, prose, feedback, audio and credentials stay local. Outbound role labels are positional `role_1` tokens; returned recommendations are mapped back locally. Queries must consist entirely of sanctioned alphabetic public tokens and limited punctuation. Numeric, non-Latin, URL, secret-like or unknown text is held conservatively.

Only `search_public_metadata` and `fetch_public_text` may execute. Fetch URLs must have appeared in a successful captured public result and still pass the adapter's host/DNS/peer/framing restrictions. Evidence is untrusted data. Model recommendations must use provided catalog IDs, exact captured citation URLs and a bounded JSON schema. They can reorder the existing catalog proposal list; local matched-trait, performer exclusion and acquisition gates remain in force. This stage cannot discover arbitrary new performers beyond that provided catalog.

No source fact, audio identity, reference eligibility, cloning permission, human rating, production assignment or voice approval is added. Metadata citations do not prove acoustic similarity or naturalness. Research does not start TTS. Changed source, profiles, assignments, preferences or casting events hold stale results, including changes between research commit and round creation.

## Budgets and receipts

The non-streaming model protocol supports typed assistant tool calls and tool-role messages. Per bound research request: at most 4 model turns, 8 tool dispatches, 8 roles and 3 recommendations per role; 2,048 requested output tokens per turn; 64 KiB model input and response bounds; 90 seconds cumulative journaled execution; 15 seconds maximum model request wait. There are no implicit model/provider fallbacks or automatic model retries.

Model response bodies reject compression and exceedance before JSON parsing. A total-deadline caller watchdog and at most two outstanding daemon requests prevent stalled transports from tying up the caller indefinitely. Closing a client cannot guarantee a remote completion or charge was cancelled; the journal retains an uncertain completion hold. Cancellation is cooperative at model/tool boundaries, not a promise of instant remote cancellation. The public adapter's existing 8-second search/5-second fetch bounds and hard 256 KiB wire cap remain authoritative; native adapter timeouts are clamped to remaining research time, and a bounded leaf-tool watchdog limits the owner wait. Late public GET results are discarded without changing owner state; two outstanding tool threads and the adapter DNS slots keep abandonment finite.

Receipts are local SHA256-enveloped `voice_lab/discovery/model-research-*.json` journals bound to source/settings/context, exact route, limits and tool-schema hashes. They record actual request IDs, finish reasons, wall time and reported token usage, plus tool argument/result hashes and captured URLs. No inferred price, account quota, private credentials or fabricated metrics are recorded. The existing discovery owner provides atomic durable writes and a 24 MB storage limit.

A durable intent pointer freezes the initial binding before any model access. Changing inputs or route under the same request ID holds rather than creating a fresh charge budget. Completed cache reuse resolves no credential. An in-flight marker is saved **before** a model call and cleared only when its complete tool protocol or final report is durably committed. Completed cache reuse makes no additional model/tool calls. A process interruption during a public GET may replay that GET, consuming the cumulative tool budget before another model turn. An uncertain completion, held failure or corrupt receipt stays held; it is never silently retried. Restart is not an automatic retry for a held journal. A separately authorized new research intent must be reviewed before changing bindings to initiate another potentially charged call. Do not delete receipts to clear a hold.

The standalone adapter integration preserves owner retries: one retry for typed HTTP429/5xx, typed API maxlag, bounded DNS/network/deadline errors; 0.25-second backoff and a durable 300-second cooldown on exhaustion. Policy, peer, framing, parse and cap failures do not retry. Media acquisition remains a separate existing owner path.

## Verification and activation

Offline suites exercise typed forwarding, malformed/unrequested/duplicate calls, strict JSON final schema, private IDs/text filtering, exact model/provider gates, zero vault access when configuration is missing, captured citations, cache corruption, crash windows, cumulative budgets, cancellation/stale source, keeper-first behavior, canonical future-import trigger, missing-config needs-attention and cross-project dispatcher responsiveness. Public transport tests use simulated wire sockets, including typed retries and oversized/framing/peer holds; no external network is contacted.

Activation remains separate: resolve the exact model/authorized Studio route, review this implementation and its limits, and obtain approval for a bounded live model call before restarting an owned service to use the new source. Offline fixture results are not a live model or voice-quality acceptance. No new auditions, full-cast notification or production promotion is implied. Existing character-source holds and human A/B review remain in force.

Rollback: do not reset the repository or remove unrelated edits. Keep per-file pre-model-loop backups and revert only this reviewed hunk set/new module and future-import requested flag, retaining all receipts, reference assets, ratings and voice identities. The original Qwen renderer remains available through its existing rollback profile.

## Architecture governance

GitHub-first HomeOS discovery found no Voice Lab/audio/AssetForge surface in the authoritative projection (`homeos-efa7c1c0727f-af753cabe44f`). Local code is the implementation evidence. Proposed ownership: AutoCast remains the orchestration owner; public metadata transport is a leaf capability; the model protocol is a bounded stage within the owner; secure provider registration belongs to Studio Settings. This document is a local proposal/handoff, not a mirror update or published architecture decision. No mirror push is authorized.
