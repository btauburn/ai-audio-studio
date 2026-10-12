"""Resumable full-chapter V2 scene-coached review renders over existing cloned chapter projects.

Prepare (CPU only, no model, no GPU): freezes the exact EPUB chapters, canonical source, approved
cloned cast and existing neutral Takes, then writes one whole-scene coaching request per chapter:

    python scripts/render_coached_chapters.py --epub BOOK.epub --source-root PROJECTS \\
        --projects a0889ff6 f0e957c4 --chapters 4 5 --output-dir OUT

Supply captured model responses as OUT-independent ``coaching-{project_id}.json`` bundles and add
``--coaching-dir DIR`` to validate and publish immutable coached plans. Add ``--generate`` (explicit,
root-owned serial GPU lane) to render only the coached dialogue and assemble the chapters; neutral
narration reuses the existing immutable Takes and is never synthesized. Outputs are review only:
no Take is selected, nothing is approved and no default changes. See docs/coached-chapters.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dubbing_service.app.audiobook.coached_chapter_batch import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
