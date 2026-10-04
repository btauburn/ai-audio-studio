> **Current checkpoint (2026-10-04):** Base cloning/rendering defaults to `faster-base-032`; the six existing accepted identities are preserved as the initial generic voice-library baseline. See [current implementation and evidence](audiobook-current-checkpoint.md).

# Voice Library, book MP3s and quick voice testing

Open **Voice Library** on the Windows desktop. Its sign-in shortcut quietly starts
Studio and the existing private review service after reboot. The desktop must be
awake and signed in for phone access. Opening the library creates no new render
request. Studio can resume previously authorized work with automatic recovery
enabled. Repeated launches reuse the running services.

Desktop library: <http://127.0.0.1:9000/web/voice-lab.html>

Brian's existing private library:
<https://brian-desktop.tail3e7427.ts.net:8443/voice-lab?home=1>

## Create an MP3

1. Drop one local EPUB or UTF-8 TXT onto the book area, or use its file picker.
2. Import it, select chapters, and choose a saved narrator version.
3. Select **Create MP3**. Play or download the finished chapter recordings or
   the combined selected-chapter recording.

This initial creator mode is **Narrator reading**: the selected voice reads all
source text, including dialogue. It does not infer a new character cast. The
source stays local. EPUB reading order is taken from its package spine; chapter
numbers are not inferred from an imported subset's position.

The **Finished recordings** section shows published chapter MP3s, combined books,
Voice Lab recordings and existing private delivery editions immediately. Search
by book/chapter, then Play, Download or Open & review. A completion link selects
the exact saved recording paused. Saved voice samples remain in their own area.

Closing the page retains the request. New book requests enable automatic recovery
and a private completion push to Brian. Eligible interruptions get at most two
retries after 30 and 120 seconds, reusing valid completed Takes. Pause automatic
recovery also lets the current utterance finish before stopping. Cancelled jobs,
changed source/cast/selections and uncertain remote inference outcomes are held
with a visible reason. A timeout alone does not prove the GPU worker stopped;
those cases stay held until a matching completed Take can be verified or the
worker outcome can be reconciled. Manual Resume cannot bypass that hold. Older requests
remain unchanged unless explicitly enabled. Existing production has GPU priority.
A reading retains its chosen narrator; import a
new reading for a different narrator edition instead of changing earlier Takes.

The progress display keeps checking long jobs while visible, backs off on network
errors, and refreshes on return. Completion pushes use the existing Pushover
configuration only after all requested exports pass file/hash checks. Accepted
or uncertain submissions are not automatically resent. Provider acceptance does
not confirm display on the phone. No speculative voice audition batch is started.

Uploads are bounded to 32 MiB (slightly less through the private multipart proxy),
200 chapters and two million source characters. A render is bounded to 2,000
utterances or 250,000 characters; larger books can be processed by chapter batches.
Other file formats and encrypted EPUB resources are not supported by this flow.

## Test and choose a character voice

Open a project's voices and listening page. Each Auto Match card names its role
and has **Preview · ~10 seconds**. That explicit action creates an original
synthetic voice and a short production-path sample. Natural duration varies;
the actual measured duration is displayed. The output is not a recording of the
referenced performer.

Use **Play preview**, then **Use for this character** when ready. Playback and
selection are separate; the preview itself never changes the book's cast. The
exact heard voice version is used, and stale selections are rejected. Existing
recorded chapters retain their audio until an explicit new rendering operation.
Favorites/ratings remain independent of assignment and do not carry to variants.

Saved previews are reused when reopening or replaying. Queued/interrupted work
can be resumed explicitly. No generation is triggered by browsing, playing or
feedback. Budgets are 12 explicit preview intents per project, 100 request
receipts and three attempts per intent. Failed jobs show their reason.

## Operations and evidence

Shortcut installer: `scripts/install_voice_library_shortcut.ps1`.
Launcher: `scripts/voice_library_launcher.py` (`--no-browser` for sign-in).
Logs: `%LOCALAPPDATA%\AI Audio Studio\voice-library`.
The launcher refuses a different service occupying port 9000 rather than
stopping it. Automatic worker starts preserve output files and provider defaults,
and refuse replacement of another managed worker.

Verification evidence for this follow-up is stored under
`C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-26-voice-lab\desktop-creator`.
The original Chapter 1 source blocker remains separate. These tiny original-text
test outputs are not the requested Super Supportive Chapter 1 listening test.

Snow I/II verification, recording library and completion recovery evidence:
`C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-27-snow-recovery`.
