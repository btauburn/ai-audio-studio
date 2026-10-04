# Fingerprinted reference-source compatibility

Frozen acquisition and draft-screening receipts hash source bytes as part of their
provenance. End-of-file whitespace and line-ending conversion therefore affect
compatibility even when Python's syntax tree and behavior are unchanged.

The October 4 checkpoint's whitespace cleanup removed one terminal newline from
each of `reference_source_qc.py` and `reference_preparation.py`. The former broke
all eight checked-in draft-catalog classifier receipts; the latter changed the
pipeline identity referenced by existing automatic-preparation receipts/cache.

The repair restores the exact original validated bytes, retaining these SHA-256s:

| Source under `dubbing_service/app/audiobook/` | Preserved SHA-256 |
| --- | --- |
| `reference_source_qc.py` | `07b90df2984732b357561ef49df92b6dd53ef5f497110477ef780ea77794d8c5` |
| `reference_preparation.py` | `fc70a953e873ddc8a180b2f5a91af78c67ef277fe8eec0fafa34fbc0caf10908` |

No receipt, model, audio, conditioning artifact, rating, approval, version or source
flag was rewritten. `.gitattributes` preserves LF and exempts those deliberately
pinned terminal newlines from the blank-at-EOF diagnostic. Do not normalize other
fingerprinted source files. A real source change requires genuine revalidation;
updating a stored hash merely to suppress a failure is invalid.

`tests/test_recorded_draft_source_pins.py` exercises the actual checked-in catalog
against the current source policy and verifies source mismatch still fails closed.
Before repair it produced eight failures and one negative-test pass. After repair,
the reference/identity/strict-casting regression set passed 133 tests. Fresh CPU
replay of three preserved recorded-reference acquisitions also passed in their
existing guarded draft mode; unrestricted portability remained held.

The portability message `reference narration delivery evidence failed exact audio
replay` currently covers two cases: a real replay mismatch, and an exact replay
whose delivery assessment still requires review. The reviewed crops, transcripts
and conditioning replay exactly in the latter case. This repair preserves the
existing strict portability policy and project-bound draft authorization; it does
not infer new narration judgments or transfer human selection across projects.
