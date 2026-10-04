# Ownership and failure boundaries

The existing Studio process owns a periodic maintenance pass after acquiring
its projects-root lock. Recovery dispatches the same request through the same
render function and existing GPU leases. The pass uses durable receipt state;
its in-memory task set only tracks shutdown. It is not another queue authority.

New book requests enable completion pushes and two automatic recovery attempts,
with 30 and 120 second backoffs. Explicit pauses, cancellations, changed source
or narrator and ambiguous in-flight external synthesis hold automatic dispatch.
Elapsed time alone never proves a remote GPU operation has stopped. Completed
valid Takes are reused, while production priority and QA gates stay intact.

Only succeeded requests with every expected MP3/master and matching hashes can
notify. Existing delivery intent/outcome claims own deduplication. A transport
acceptance is distinct from a confirmed phone notification. Unknown acceptance
is held instead of resent. Historical requests are never bulk-enabled.

The catalog derives from existing published exports and delivery manifests.
It publishes metadata and checked media URLs, not filesystem paths. The private
service filters non-creator projects through its current share allowlist. Media
continues to be served as a hash-verified byte snapshot with Range support.

The home page separates finished recordings, book creation and voice samples.
Links open an exact saved recording paused. Viewing, feedback and playback do
not synthesize. Live job polling survives long renders and transient read errors.
