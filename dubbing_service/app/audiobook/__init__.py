"""Audiobook architecture v1.2 additions to the canonical studio.

This package does **not** introduce a second store.  It adds the pieces the
audiobook pipeline needs on top of the existing project/segment/take storage:

* ``providers``    — source revisions, spoken-text revisions, attribution
* ``identity``     — utterance identity, split/merge lineage, reorder
* ``fingerprint``  — the synthesis requirement fingerprint (input identity)
* ``engines``      — engine capabilities and license scopes
* ``renderer``     — renderer seam, chunking, one-utterance-one-Take guarantee
* ``takes``        — Take commit, staleness and safe promotion
* ``qa``           — Tier-1/Tier-2 quality checks
* ``regeneration`` — dependency-aware selective regeneration
* ``cues``         — utterance-anchored audio cues
* ``adapter``      — optional audiobooker ingest boundary
* ``pipeline``     — orchestration + CLI entry point
"""

__all__: list[str] = []
