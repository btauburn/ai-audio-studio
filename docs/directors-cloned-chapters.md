# Directors architecture follow-up — 12 October 2026

Implemented on `feature/casting-performance-director-20261011`, continuing
[the director implementation](directors-implementation.md). Code belongs to the
existing Studio/Voice Lab owners. Private books, voices, models, audio and runtime
databases remain outside Git. The GitHub fork is `btauburn/ai-audio-studio` because
the authenticated account has no push permission on the upstream repository.

## Cloned-only contract

Set `project.metadata.director_voice_policy = {"mode": "cloned_only"}` for a new
book. Missing policy retains the existing behavior. Malformed policy fails
closed. `app/audiobook/cloned_voice_policy.py` is a pure eligibility check;
source/rights, approval, asset verification and model readiness remain separate.
Here, a clone means an actual recorded-reference Qwen Base identity with
`eligible_reference_clone` provenance and `actor_audio_used=true`. Reference
conditioning alone does not qualify a synthetic original voice as a clone.

The integrated owners enforce this at several boundaries:

- `casting_director.py` binds policy into the immutable analysis identity,
  filters recommendations and rechecks candidates before audition preparation.
  Current algorithm is `casting-director-integration-v7`.
- A protected synthetic voice keeps its saved approval but is reported as
  `blocked_by_voice_policy`. It cannot satisfy cloned-only coverage or render.
- `casting_review.py` refuses the ordinary synthetic proposal fallback for an
  opted-in book; explicit ranked reference auditions remain available.
- `voice_lab_audition.py` requires an exact reusable recorded clone, freezes
  policy into the intent hash, and carries it into the child project.
- `pipeline.py` rejects missing/non-clone roles or a different renderer before
  provider initialization, automatic voice design or inference.
- `webui/casting-director.js` shows the policy and any saved-cast conflict.

No candidate is promoted, rated or approved by this policy. Existing 4–5-star
approvals are preserved, including synthetic voices excluded from a cloned-only
book. Rights/consent are never inferred from accessibility, ratings or this flag.

## Numbered source selection and review exports

`app/audiobook/epub_chapter_selection.py` selects explicit numbered chapter
titles through the existing bounded EPUB parser. It excludes positional guesses
such as cover/information documents, rejects missing/ambiguous labels, and
retains exact text, archive member, source SHA-256 and chapter text SHA-256.
Spelled-out chapter numbers are currently unsupported.

`scripts/render_cloned_chapters.py` builds new review exports from approved,
selected, immutable Takes. It validates the exact supplied EPUB, complete ordered
chapter prose, canonical source/cast/release/QA evidence, clone reference and
prompt bytes, and the actual Faster Base profile recorded on every Take. It
never starts synthesis, changes a voice, selects a Take, edits source metadata,
or changes defaults. Retry validates the frozen inputs and published hashes.

Timing uses the existing semantic classifier and policy. Strict leading/trailing
silence is measured first; only the remaining target pause is added. Speech is
never trimmed, accelerated or rewritten. Existing source separator targets stay
accounted for. Unknown classifications remain visible in the source-to-audio map.
These pause targets are experimental hypotheses requiring human listening.

Character analysis and cached scene recommendations run on a private project
projection, with caches in the review destination. Original project/segment
metadata is verified unchanged before publication. Native emotion instructions
were **not** applied to these full chapters; the existing instruction comparator
remains a separately guarded pilot, not a global production switch.

## Actual first-three-chapter result

The uploaded Ar'Kendrithyst EPUB has Information frontmatter before chapters
001–003. Its SHA-256 is
`5abdf0ba046ec8891c559945ccba81a60f72a6ef10bb1dc3a241bcc0e657d868`.
An existing completed, authorized clone-only run was found for this same source.
Its exact final selections were retained rather than replaced by new casting.

The retained cast is Gyles Brandreth as narrator, VCTK p254 as Erick, Emma Watson
as Jane, p226 as the Darkness, and p273 as the registrar. These are existing
private/local reference clones; this run grants no additional source or public
use rights. Original chapter source and all approved assets remain intact.

| Chapter | Reused Takes | Timed duration | Added measured padding |
|---|---:|---:|---:|
| 001 | 130 | 906.43 s (15:06) | 35.71 s |
| 002 | 160 | 1039.68 s (17:20) | 40.00 s |
| 003 | 190 | 1085.72 s (18:06) | 48.36 s |

All 480 selected Take identities/hashes are identical to the prior chapter export
selections. New synthesis count is zero. Human listening remains pending and
improved quality is **not established**. Acoustic metrics and successful decode
do not establish acting, role separation or listener preference.

Local artifacts:
`C:/Users/brian/omni-audiobook/runtime_evidence/2026-10-12-arkendrithyst-cloned-chapters/`.
`exports/chapter-001..003.{wav,mp3,json}` contain playable audio, immutable export
receipts and complete source-to-audio timing maps. Chapter director reports and
scene cache are separate. The task's local A/B player is
`http://127.0.0.1:9141/`; audio Range requests returned HTTP 206 for all chapters.
Baseline files are retained in the earlier private run.

## Verification and reproducibility

The regression run passed 549 Python checks, including the former reference-
quality regression, and 178 JavaScript checks. Final focused casting/audition
checks cover the subsequent policy-bypass closure. This is a relevant subset;
the prior full-suite order-dependent failure has not received a green full-suite
rerun. Exact test receipts and actual Sonnet metadata are in
`C:/Users/brian/.codex/task-observer/runs/2026-10-12-cloned-chapters/`.

From the Studio repository, reproduce the review against the installed private
approved source store:

```powershell
.\.venv\Scripts\python.exe scripts/render_cloned_chapters.py `
  --epub C:\Users\brian\Downloads\ArKendrithyst.epub `
  --reuse-approved-root C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-28-autocast-elf-acceptance\projects `
  --reuse-projects 9c7257c4 b3b6b320 c59f2ad6 `
  --output-dir C:\Users\brian\omni-audiobook\runtime_evidence\2026-10-12-arkendrithyst-cloned-chapters

.\.venv\Scripts\python.exe -m pytest tests/test_cloned_voice_policy.py `
  tests/test_epub_chapter_selection.py tests/test_cloned_director_integration.py `
  tests/test_cloned_chapter_render.py tests/test_casting_auditions.py `
  tests/test_voice_lab_audition.py tests/test_strict_casting_pipeline.py -q
node --test tests/casting_director_ui.test.cjs
```

Private approved assets and canonical chapter rendition state are required for
the actual EPUB export. CPU unit fixtures require neither these files nor a GPU.
Unrelated local source-policy/source-separator/Reaper work stays unstaged and is
not claimed as part of this follow-up.

## Sonnet contributions and remaining limits

Sonnet A implemented pure clone eligibility and negative cases. Sonnet B
implemented numbered chapter selection and archive/source regressions. Sonnet C
read all three chapters, produced grounded character evidence and audited the
existing library/Grok records. Sonnet D independently mapped quoted dialogue
and recorded attribution holds. All four actually ran Claude Sonnet 5.5 and
finished; their output/evidence is preserved outside Git.

Codex integrated policy into canonical casting/audition/render owners, kept
approved source attribution authoritative, validated/reassembled the real
chapters, tested delivery, and published the architecture. No worker selected
or approved a new voice.

The offline analyzer still leaves many unknown speakers and mistakes spell
names for roles. New attribution proposals contain uncertainty and mixed/open
quotation cases; they were not substituted for this book's existing approved
source attribution. Grok's available research handoffs contain no completed
starter-library voice search; no public source was cloned based on that claim.
New-source cloning, automatic model reasoning, speaker-embedding measurements,
long-form human validation and IndexTTS 2.5 remain unvalidated or held. No model
or reference download was needed for these three chapters.
