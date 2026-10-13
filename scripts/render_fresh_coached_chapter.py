"""Fresh scene-coached review chapter over a root-prepared derived project (approved clones, no selected Takes).

Prepare (CPU only, no model, no GPU) freezes the exact EPUB chapter, derived source, approved cloned cast and a
whole-scene coaching request:

    python scripts/render_fresh_coached_chapter.py --epub BOOK.epub --source-root PROJECTS \\
        --project PROJECT_ID --chapter 11 --output-dir OUT [--performance-memory MEMORY.json]

Add ``--coaching-bundle FILE`` (a captured model response) to validate and publish the immutable coached plan.
Add ``--generate`` (explicit, root-owned serial GPU lane) to synthesize every speech line - narration neutrally,
dialogue with its captured instruction - and assemble the chapter. Outputs are review only: no Take or selection
is created, nothing is approved and no default changes. See docs/fresh-coached-chapters.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dubbing_service.app.audiobook.fresh_coached_chapter import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
