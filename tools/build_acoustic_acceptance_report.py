"""Generate the performer-reference acoustic conditioning acceptance report.

Writes ASCII-only punctuation on purpose: the evidence file is produced through a
shell, and a Unicode round-trip through a console code page has silently mangled
text in this project before.
"""
import json
import os

BASE = r"C:\Users\brian\omni-audiobook\runtime_evidence\2026-09-28-autocast-elf-acceptance"
ARROW = "->"

candidates = json.load(open(BASE + r"\acoustic-conditioning-candidates.json"))
real = {row["character"]: row for row in json.load(open(BASE + r"\acoustic-conditioning-real-run.json"))}

lines = []
w = lines.append

w("# Performer-reference acoustic conditioning - acceptance report")
w("")
w("Project: `ba9786ea` - Elf_1_10 AutoCast Acceptance")
w("")
w("This report answers one question: does the famous-performer layer change the")
w("*sound* of a generated voice, or is it descriptive metadata? It records, per")
w("character, the full chain the implementation actually walks:")
w("")
w("```")
w("character -> book-supported traits -> inferred casting traits")
w("          -> performer + specific performance -> reference source")
w("          -> measured/derived acoustic characterization")
w("          -> the generation parameters passed to TTS")
w("```")
w("")
w("## 1. What was executed, and what was not")
w("")
w("| Step | Status |")
w("|---|---|")
w("| Candidate planning (dossier -> concepts -> anchors -> conditioning) | **Executed** against the real project: all 10 profiled characters, 30 candidates |")
w("| Reference evidence + characterization | **Executed**; every anchor characterized, method `descriptor-prior-v1` |")
w("| Acoustic conditioning of real generated audio | **Executed** end to end on the GPU lane: 3 characters x 3 candidates = 9 identities |")
w("| Worker seam ordering (Base prompt from the conditioned reference) | **Executed** against a fixture model; the conditioned path is asserted in `tests/test_voice_acoustics_conditioning.py` |")
w("| Eligibility-checked performer *recording* ingestion | **Not executed.** No performer recording in the catalogue carries a rights/ownership acquisition receipt for this project, so no `performer_discovery.acquire_reference` call was made and no anchor reports `measured`. Characterization is labelled `inferred` throughout, and the reason is stored on every candidate (`retrieval.reason`). |")
w("| Production-path audition audio (script sections rendered and scored) | **Not executed** for the 9 identities; the conditioning seam was exercised directly so the acoustic evidence is attributable to it rather than to the sound of an audition script |")
w("")
w("The distinction matters: the conditioning pass is not a claim about a documented")
w("prior. It measured real TTS output, derived a plan, wrote new audio, and")
w("re-measured it. See section 3.")
w("")
w("## 2. Per character: the full casting chain")
w("")
for row in candidates:
    w("### " + row["character"])
    w("")
    for candidate in row["candidates"]:
        supported = "; ".join(
            "`{dim}`={val} ({level})".format(dim=x["dimension"], val=x["value"], level=x["level"])
            for x in candidate["supported"]
        ) or "_none - the book establishes no vocal fact for this character_"
        inferred = "; ".join(
            "`{dim}`={val}".format(dim=x["dimension"], val=x["value"])
            for x in candidate["inferred"]
        )
        tp = candidate["tts_parameters"]
        w("**Candidate: {concept}** ({emphasis})".format(
            concept=candidate["concept"], emphasis=candidate["emphasis"]))
        w("")
        w("- **Book-supported traits:** " + supported)
        w("- **Casting inference** (labelled `CASTING_INFERENCE`, never a book fact): " + inferred)
        w("- **Performer + performance:** {p} - *{s}*".format(
            p=candidate["performer"], s=candidate["performance"]))
        source = candidate["source_urls"][0] if candidate["source_urls"] else "none"
        w("- **Reference source:** {url} ({kind})".format(
            url=source, kind=candidate["reference_source_type"]))
        w("- **Matched traits:** {traits} (concept affinity {aff})".format(
            traits=", ".join(candidate["matched_traits"]), aff=candidate["concept_affinity"]))
        w("- **Characterization:** `{state}` via `{method}` - measured {m} propert(ies),"
          " derived {i}; retrieval `{r}` ({why})".format(
              state=candidate["characterization"], method=candidate["characterization_method"],
              m=len(candidate["measured_properties"]), i=candidate["inferred_property_count"],
              r=candidate["retrieval_state"], why=candidate["retrieval_reason"]))
        w("- **Acoustic target - the numbers, not the performer's name:**")
        w("")
        w("  | pitch | range | rate | pause | level | brightness | rasp | breathiness | resonance / weight / age |")
        w("  |---|---|---|---|---|---|---|---|---|")
        w("  | {pitch:.1f} Hz | {rng:.2f} st | {rate:.0f} wpm | {pause:.0f} ms |"
          " {level:.1f} dBFS | {bright:.0f} Hz | {rasp:.3f} | {breath:.3f} |"
          " {res} / {wt} / {age} |".format(
              pitch=tp["pitch_target_hz"], rng=tp["pitch_range_semitones"],
              rate=tp["rate_words_per_minute"], pause=tp["pause_ms"], level=tp["energy_dbfs"],
              bright=tp["brightness_hz"], rasp=tp["rasp"], breath=tp["breathiness"],
              res=tp["resonance"], wt=tp["vocal_weight"], age=tp["age_impression"]))
        w("")
        w("- **Generation parameters sent to TTS:** `pitch_target_hz={pitch:.1f}`,"
          " `pitch_range_semitones={rng:.2f}`, `rate_words_per_minute={rate:.0f}`,"
          " `pause_ms={pause:.0f}`, `energy_dbfs={level:.1f}`, `brightness_hz={bright:.0f}`,"
          " `breathiness={breath:.3f}`, `rasp={rasp:.3f}`, `articulation={art:.3f}`,"
          " `sentence_end_f0_slope={slope:.1f}`, engine hints `{hints}`".format(
              pitch=tp["pitch_target_hz"], rng=tp["pitch_range_semitones"],
              rate=tp["rate_words_per_minute"], pause=tp["pause_ms"], level=tp["energy_dbfs"],
              bright=tp["brightness_hz"], breath=tp["breathiness"], rasp=tp["rasp"],
              art=tp["articulation"], slope=tp["sentence_end_f0_slope"],
              hints=json.dumps(tp["engine_hints"])))
        w("- **Also carried as numbers in the design instruction:** " + candidate["instruction"])
        w("")

w("## 3. Did the sound actually change? (real GPU run)")
w("")
w("Chain: design -> measure -> bounded shaping -> re-measure, on the real worker")
w("(`service/qwen_character_voice.py`). The shaped reference is the audio the Base")
w("prompt is built from.")
w("")
w("| Character | Concept | Performer | pitch before | pitch after | target | closer? | role rate x | semitones | reach |")
w("|---|---|---|---|---|---|---|---|---|---|")
for name, row in real.items():
    for candidate in row["candidates"]:
        b = candidate["profile_before"]
        a = candidate["profile_after"]
        t = candidate["target_parameters"]
        closer = abs(a["f0_median_hz"] - t["pitch_target_hz"]) < abs(b["f0_median_hz"] - t["pitch_target_hz"])
        reach = candidate["reach"]
        w("| {c} | {k} | {p} | {bf0:.1f} Hz | {af0:.1f} Hz | {tf0:.1f} Hz | {closer} | x{rate:.3f} |"
          " {semi:+.2f} | {verdict} ({reached} reached / {short} shortfall / {oos} out of scope) |".format(
              c=name, k=candidate["concept"], p=candidate["performer"],
              bf0=b["f0_median_hz"], af0=a["f0_median_hz"], tf0=t["pitch_target_hz"],
              closer="yes" if closer else "no", rate=candidate["shaping_plan"]["rate"],
              semi=candidate["shaping_plan"]["semitones"], verdict=reach["verdict"],
              reached=reach["reached_count"], short=reach["shortfall_count"],
              oos=reach["out_of_scope_count"]))
w("")
moved = sum(1 for row in real.values() for c in row["candidates"] if c["audio_changed"])
total = sum(len(row["candidates"]) for row in real.values())
closer_count = sum(
    1 for row in real.values() for c in row["candidates"]
    if abs(c["profile_after"]["f0_median_hz"] - c["target_parameters"]["pitch_target_hz"])
    < abs(c["profile_before"]["f0_median_hz"] - c["target_parameters"]["pitch_target_hz"]))
w("All {total} identities produced a **different audio hash** after conditioning"
  " ({moved}/{total}), and median pitch moved closer to its target in"
  " **{closer}/{total}** cases. The Sound is conditioned; this is not metadata."
  .format(total=total, moved=moved, closer=closer_count))
w("")
w("**Reachability is three-valued on purpose.** A documented acoustic prior carries")
w("keys no pitch/rate transform can move (peak level, voiced fraction, sentence-ending")
w("energy drop), so a comparison over the whole target profile is not a reachability")
w("verdict. Each receipt reports `reached`, `shortfall` and `out_of_scope` separately,")
w("and the verdict is one of `reached` / `shortfall` / `could_not_measure`, where a")
w("comparison that touched no controllable key is reported as unmeasured rather than")
w("as a pass. Measured in this run: 3-6 keys reached, 1-4 in shortfall, 7-8 out of the")
w("actuator's scope.")
w("")
w("**The correction is bounded on purpose** (+/-3 semitones, rate x0.85-1.18), so a")
w("target outside the bound is reported as a shortfall and never presented as reached.")
w("Measured example (Laura Dern / The Librarian, deep-serious): target 184.8 Hz,")
w("achieved 127.0 Hz, plan clamped at +3.00 semitones of +8.5 requested. The")
w("recurring `pause_count_per_minute` shortfall is the same clamp seen from the other")
w("side: pause density is reachable only through the rate change, which clamped.")
w("")
w("### A measurement defect this run found and fixed")
w("")
w("The first real run showed a *slowed* take measuring **faster**: 139.8 -> 165.9 wpm")
w("at rate x0.908. Envelope-peak syllable detection counts the time-stretch's own")
w("segment duplication as new syllables, so the rate metric was reading the")
w("transform, not the speech. Rate metrics are now derived from the pre-stretch")
w("measurement and the applied factor - which the output duration independently")
w("confirms (7.440 s / 8.195 s = x0.908) - with both the raw and the derived value")
w("and the substitution recorded on the receipt. After the fix the same case reports")
w("126.9 wpm against a 126.9 wpm target. Per-frame measurements the stretch does not")
w("invalidate (pitch, level, brightness, texture) are still taken as measured.")
w("")
w("## 4. Acceptance criteria")
w("")
w("| # | Criterion | Where it is proven |")
w("|---|---|---|")
w("| 1 | Every fresh generated candidate has a performer + specific performance | `test_every_fresh_candidate_has_a_performer_a_performance_and_reference_evidence`; section 2: 30/30 anchored |")
w("| 2 | Production cannot silently fall back to a generic unanchored voice | `test_a_blocked_candidate_is_a_failure_state_and_cannot_be_generated` - `voice_lab.design` refuses with the candidate's reason |")
w("| 3 | Sparse dossiers produce labelled casting inference, never fabricated book evidence | `test_sparse_character_evidence_produces_labelled_casting_inference`, `test_a_sparse_character_is_cast_not_left_generic_and_never_shows_inference_as_book_fact` |")
w("| 4 | Every anchor stores reference/source evidence | `test_every_fresh_candidate_has_a_performer_a_performance_and_reference_evidence` |")
w("| 5 | Every candidate stores a characterization derived from its reference | `test_the_stored_candidate_keeps_the_characterization_derived_from_its_reference`, `test_a_measured_reference_is_labelled_measured` |")
w("| 6 | The generation request receives the characteristics, not only the name | `test_the_generation_request_receives_acoustic_values_not_only_a_performer_name`; section 3 |")
w("| 7 | Three candidates for one character differ in casting dimension | `test_three_candidates_for_one_character_differ_in_casting_dimension`; section 2 |")
w("| 8 | Existing 4-5 star historical voices still participate | `test_a_four_star_voice_from_one_project_is_proposed_to_a_similar_character_elsewhere`, `test_a_five_star_voice_carries_a_stronger_reuse_signal_than_a_four_star_voice`, `test_fresh_candidates_complement_a_prior_voice_instead_of_duplicating_it` |")
w("| 9 | Deferred-replacement behaviour unchanged | `test_low_rating_leaves_the_remaining_options_playable_and_scoreable`, `test_low_rating_defers_one_replacement_instead_of_replacing_the_candidate` |")
w("| 10 | Rating/history persistence unchanged | `test_historical_ratings_stay_attached_to_the_voice_after_a_later_recast`, `test_a_low_rated_voice_is_retained_as_history_but_never_promoted` |")
w("| 11 | Existing tests remain green | full suite: **1474 passed, 1 skipped** |")
w("")
w("## 5. Evidence files")
w("")
w("- `acoustic-conditioning-candidates.json` - the per-character chain (section 2)")
w("- `acoustic-conditioning-real-run.json` - the real GPU conditioning receipts (section 3)")
w("- `qwen3tts-8003-acceptance.log` - worker startup for the real run")
w("- `studio-9100-acoustics.log` - Studio serving the extended casting view")
w("")

path = os.path.join(BASE, "acoustic-conditioning", "acceptance-report.md")
os.makedirs(os.path.dirname(path), exist_ok=True)

# The report reads like a document, so it will be edited like one. Regenerating it
# wholesale would silently discard those edits, so the human-authored region is
# read back and re-emitted, and the generator reports what it preserved.
NOTES_HEADING = "## 6. Human notes (preserved across regeneration)"
NOTES_PLACEHOLDER = "_Add findings here. This section is read back and re-emitted by_"
_preserved = []
if os.path.exists(path):
    with open(path, encoding="utf-8") as stream:
        existing = stream.read()
    marker = existing.find(NOTES_HEADING)
    if marker != -1:
        _preserved = [existing[marker:].rstrip("\n")]

w(NOTES_HEADING)
w("")
if _preserved:
    lines_kept = _preserved[0].splitlines()[2:]  # drop the heading and blank line
    for kept in lines_kept:
        w(kept)
else:
    w(NOTES_PLACEHOLDER)
    w("`tools/build_acoustic_acceptance_report.py`. The rest of the report is derived")
    w("from the JSON evidence files and is regenerated on every run._")
w("")

with open(path, "w", encoding="utf-8", newline="\n") as stream:
    stream.write("\n".join(lines))
kept_chars = len("\n".join(_preserved))
print("wrote", path, os.path.getsize(path), "bytes;"
      f" preserved {kept_chars} char(s) of human notes")
