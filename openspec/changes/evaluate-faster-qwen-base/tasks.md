# Tasks

> Current decision, 2026-10-04: **USER-APPROVED DEFAULT** is the implemented
> `faster-base-032` Base cloning/character profile. This supersedes the earlier
> native-current/auto preference and blind-listening promotion gate. CustomVoice
> remains a separate lane. Explicit runtime choices and immutable job/Take bindings
> are preserved; mismatched historical resume must fail before synthesis. User
> approval does not assert universal audio parity or that every sample was heard.
> The October 1-2 experimental/default/pending-approval statements below describe
> historical state. Frozen A/B audio, ASR flags, word endings, chunk boundaries,
> voice identity and longform listening remain useful quality evidence.
> See `docs/audiobook-current-checkpoint.md` for the current implementation receipt.


- [x] Inspect local ownership, checkpoint and existing voice/casting state.
- [x] Research and verify pinned lowest-dependency-change candidate.
- [x] Freeze identities, prompts, references, text, settings, seeds and chunking.
- [x] Preserve native baseline and build isolated guarded Base adapter/profile.
- [x] Add read-only private paired playback and safety/regression tests.
- [x] Obtain independent implementation review and fix reported races/access/playback defects.
- [x] Finalize longform metrics, actual worker checks and preservation/rollback evidence (see final report).
- [x] Bind actual worker/runtime provenance to jobs, per-chunk requests, Take fingerprints and resume/cache checks; reject mismatch/legacy reuse before synthesis.
- [x] Exercise actual VoiceDesign-to-Base shared audition/production path and 124-chunk real chapter stress.
- [x] Add offline ASR diagnostics with versioned edit/tail/repetition flags; preserve raw recordings.
- [x] Deliver explicitly requested six provisional cast samples and labeled sampler via Library/private phone playback; one completion notification accepted.
- [ ] Human identity, naturalness, expressiveness, completeness and longform A/B review.
- [ ] Explicit production promotion only after all gates pass; native remains default until then.
