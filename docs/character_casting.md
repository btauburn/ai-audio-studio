> **Historical original-design scope:** This document describes the initial research/VoiceDesign casting subsystem. Recorded-reference cloning, acquisition, the casting inbox and revision-bound approvals are now implemented; see [human casting](human_casting.md), [reference cloning](voice_lab_references.md) and the [2026-10-04 current checkpoint](audiobook-current-checkpoint.md). The original limits below apply to that earlier subsystem, not the whole current app.

# Character research and original voice casting

Studio now researches imported character text before designing an original Qwen voice. The audiobook adapter, project store, role IDs, segment attribution, Take provenance, render fingerprints, QA, and optimistic persistence remain the production authority. Audiobooker is an isolated analysis provider.

## Data flow and ownership

1. The existing ingest adapter preserves the source revision and attributed segments.
2. Incremental, anchored English extraction appends provenance-bearing character facts. Explicit aliases consolidate casting without rewriting human attribution.
3. A separate casting profile resolves source priority and story scope, preserves conflicts, and explains acoustic interpretations. Physical size never determines pitch. Compatible voice descriptions accumulate.
4. Optional editorial cast palettes match performance traits. Actor and character names remain metadata; only acoustic traits enter VoiceDesign instructions.
5. A bounded audition job designs original references, extracts reusable Qwen Base prompts, renders short neutral/elevated/quiet texts, and evaluates audio and cast contrast.
6. The selected identity is stored on the existing Role. Subsequent dialogue reuses its prompt. Scene direction is stored separately in each Take.

`Project.character_cast` has schema version 1. Older project JSON loads with empty defaults; existing fields and files are retained. Facts include source/revision, excerpt, location, confidence, timestamp, chapter availability and story scope. Evidence levels are EXPLICIT, SUPPORTED, INFERRED and UNKNOWN. New evidence does not mutate locked identities: changed requirements create proposed revisions and retain profile history.

The subsystem writes through existing metadata locks and atomic project persistence. GPU operations run outside metadata locks under the existing TTS operation lease. Each completed reference, prompt and sample is checkpointed, so interrupted casting resumes. Voice artifacts are project-relative and hash checked. Generated samples, source text and weights are ignored runtime data.

## Qwen mechanism actually used

The installed `qwen-tts` 0.1.1 exposes `generate_voice_design`, `create_voice_clone_prompt`, and `generate_voice_clone`. The original 1.7B VoiceDesign reference and exact transcript feed 1.7B Base with `x_vector_only_mode=False`. Numeric speaker embeddings and reference codes are serialized in safe NPZ plus JSON metadata; loading never enables pickle. This is reuse of an original generated voice, with no actor recording or outside reference accepted at the new design endpoint.

The existing `qwen3tts` worker switches VoiceDesign/Base models under its lock, retaining one active GPU model. Official pinned manifests and runtime versions are recorded. Reference design intent, model provenance, artifact hashes, prompt metadata and generation settings survive reload. Missing, modified or incompatible prompts fail explicitly rather than designing a replacement voice.

Base uses sampled decoding with a stable identity/text seed and bounded duration/token checks. Voice stability comes from the persistent prompt. Greedy decoding was rejected during real acceptance because it produced excessive repetition.

Base does **not** expose natural-language scene emotion control. Emotion/intensity/delivery are preserved as separate intent and unsupported fields are reported. Audiobook rendering and auditions apply pace and gain with pitch-preserving FFmpeg processing and stamp baked values on Takes. Elevated audition text is not proof of controlled emotional acting.

## Use

Run from the repository root with the normal Qwen worker available for audio:

```powershell
.venv\Scripts\python.exe -m dubbing_service.app.audiobook.cli ingest --file book.epub
.venv\Scripts\python.exe -m dubbing_service.app.audiobook.cli cast --project-id PROJECT --no-audition --report cast.json
.venv\Scripts\python.exe -m dubbing_service.app.audiobook.cli cast --project-id PROJECT --max-characters 2 --max-candidates 2 --report cast.json
.venv\Scripts\python.exe -m dubbing_service.app.audiobook.cli cast --project-id PROJECT --palette harry-potter-film-cast --max-characters 2 --report inspired.json
```

Ingest performs research/profile preparation without GPU work. The normal audiobook Qwen rendering path automatically prepares missing character identities before rendering. `cast` pre-casts a bounded group explicitly. Narration and unresolved speaker labels retain the existing narration path. Existing custom built-in or user reference assignments are preserved.

`--max-chapter` is an imported **zero-based chapter index**, including any frontmatter entries. The boundary filters source evidence, aliases, external results and casting context. `--allow-spoilers` is explicit opt-in. `--adaptation` permits direct character-to-adaptation mapping; otherwise matching uses performance archetypes. Changing a palette on a locked project proposes new casting requirements; use a separate project for an alternate cast.

Reports include every dossier, evidence/rationale, inspiration match and confidence caveat, identity status, candidates, measured collisions, exceptions and proposed revisions. JSON retains complete history; it is an audit artifact and can contain previously ingested later-chapter facts. Casting context always applies the configured boundary.

Default policy allows up to 8 characters per casting batch and up to 3 candidates (configurable ceiling 4). Important uncertain characters receive more candidates. Rendering batches only cast needed roles. Locked identities and completed auditions are reused. Research caches chunks and external failures, and explicit aliases are resolved before generation.

## Audition and ensemble verification

Tests cover silence/clipping, signal duration, pitch/register mismatch, gross identity instability and intentionally colliding voices. Similarity combines lightweight pitch/timbre proxies. Frequent conversation neighbors receive stronger contrast pressure. Major/recurring collisions block automatic selection and appear in review. These measurements do not prove perceived age, acting quality, intelligibility or biometric identity.

The built-in Harry Potter film palette contains eight editorial performance archetypes with official cast-association links. Acoustic traits are editorial interpretations, not claims of measured actor acoustics. Matching protects explicit source requirements and flags weak evidence. Custom palettes are supported by the Python API; arbitrary new named franchise research is not implemented.

## Implemented limits

- Automatic research recognizes conservative anchored English constructions. Complex coreference, multilingual literary interpretation and semantic whole-scene analysis remain partial. Unknown traits stay unknown; major-character uncertainty is reported.
- An external research provider protocol, cached failure path, source precedence and spoiler gate are implemented and tested. No live external research provider is configured by default. `--external-research` therefore reports unavailable research instead of inventing it.
- No new Character/Cast GUI, semantic LLM extraction service, actor-audio acquisition, celebrity cloning, learned speaker-embedding judge or automatic perceptual age estimator is implemented.
- The end-to-end automatic audio path is the audiobook Qwen renderer/CLI. Existing manual WebUI render flows can reuse stored identities but have no new automatic casting screen.
- Proposed revisions are retained for review; a dedicated recast/approval UI is not implemented. Existing rendered work remains intact.

## Verification entry points

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts/character_casting_acceptance.py --file PATH_TO_EPUB --output-root PRIVATE_REPORT_DIRECTORY --audio-characters 2
```

The acceptance script runs default and inspired casts in separate projects, saves evidence/collision reports, checks persisted identity equality, and renders one short source line per selected character through the normal pipeline. It never requests complete chapter renders. If the supplied source is unavailable, it explicitly records substitution with an original synthetic fixture.

New tests are in `test_character_research.py`, `test_character_casting.py`, `test_character_casting_integration.py`, and `test_character_qwen.py`. Signal doubles are explicitly labeled fixtures. Real model acceptance is separate from unit tests.
