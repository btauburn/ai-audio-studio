"""Versioned, source-independent scripts for Voice Lab casting auditions.

Section labels are metadata for navigation. Only ``spoken_text`` reaches TTS.
The shorter replacement derives verbatim excerpts from the initial script.
"""
from __future__ import annotations

CASTING_60_VERSION = "casting-60-v1"
CASTING_30_VERSION = "casting-30-v1"

_SCRIPTS = {
    CASTING_60_VERSION: {
        "target_seconds": 60,
        "duration_window_seconds": [45, 80],
        "sections": [
            ("neutral", "NEUTRAL", "Your notebook is on the table. I kept it dry, but the map wasn't so lucky.",
             "Conversational and even, with clear natural articulation."),
            ("quiet", "QUIET", "Don't move. Someone is outside, and I don't think they know we're here.",
             "Quiet and cautious, as if someone might overhear."),
            ("excited", "EXCITED", "You found it! After all that searching, it was here the whole time. Come on!",
             "Excited and energized, while keeping the words clear."),
            ("controlled_intensity", "CONTROLLED INTENSITY", "Put it down. You don't get to decide for everyone, and you don't get to blame me.",
             "Firm and controlled, with restrained intensity."),
            ("vulnerable", "VULNERABLE", "I said I wasn't afraid. That wasn't true. I just didn't want to be left behind.",
             "Vulnerable and restrained; avoid melodrama."),
            ("dry_humor", "DRY HUMOR", "Excellent plan. Just one small problem: the locked door, the missing key, and that goat.",
             "Dry humor, delivered with understatement."),
            ("question_emphasis", "QUESTION / EMPHASIS", "Wait—Tuesday or Thursday? I asked for the blue one, not two.",
             "Ask the question naturally and distinguish blue from two."),
            ("pronunciation", "PRONUNCIATION", "Rowan, bring three fresh batteries. We'll meet by the railway bridge at a quarter past three.",
             "Articulate the name, number words, and place name clearly."),
        ],
    },
    CASTING_30_VERSION: {
        "target_seconds": 30,
        "duration_window_seconds": [20, 45],
        "sections": [
            ("neutral", "NEUTRAL", "Your notebook is on the table. I kept it dry.",
             "Conversational and even, with clear natural articulation."),
            ("quiet", "QUIET", "Don't move. Someone is outside.",
             "Quiet and cautious, as if someone might overhear."),
            ("excited", "EXCITED", "You found it! It was here the whole time.",
             "Excited and energized, while keeping the words clear."),
            ("controlled_intensity", "CONTROLLED INTENSITY", "Put it down. You don't get to decide for everyone.",
             "Firm and controlled, with restrained intensity."),
            ("vulnerable", "VULNERABLE", "I said I wasn't afraid. That wasn't true.",
             "Vulnerable and restrained; avoid melodrama."),
            ("pronunciation", "PRONUNCIATION", "Rowan, bring three fresh batteries. We'll meet by the railway bridge.",
             "Articulate the name, number words, and place name clearly."),
        ],
    },
}


def script(script_version: str | None) -> dict:
    """Return an immutable-by-copy script descriptor; ``None`` is legacy 10s."""
    if script_version is None:
        return {"script_version": None, "target_seconds": 10,
                "duration_window_seconds": None,
                "sections": [{"id": "preview", "label": "PREVIEW", "spoken_text": None}]}
    try:
        template = _SCRIPTS[script_version]
    except (KeyError, TypeError):
        raise ValueError("Unknown audition script version") from None
    return {
        "script_version": script_version,
        "target_seconds": template["target_seconds"],
        "duration_window_seconds": list(template["duration_window_seconds"]),
        "sections": [{"id": section_id, "label": label, "spoken_text": text,
                      "delivery_instruction": instruction}
                     for section_id, label, text, instruction in template["sections"]],
    }


def versions() -> tuple[str, ...]:
    return tuple(_SCRIPTS)
