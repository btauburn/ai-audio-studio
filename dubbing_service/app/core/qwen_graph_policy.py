"""Fail-closed checks for the pinned faster-qwen3-tts CUDA graph loop."""

from __future__ import annotations


class GraphGenerationTruncated(RuntimeError):
    """The graph loop stopped at a token or static-cache boundary, not EOS."""


def classify_graph_termination(
    *,
    generated_steps: int,
    max_new_tokens: int,
    prefill_tokens: int,
    graph_max_seq_len: int,
) -> str:
    """Accept only an observed early EOS exit from faster-qwen3-tts 0.5.0.

    That pinned release returns a token count but silently breaks at the fixed
    talker-cache boundary. Reaching either configured limit is therefore
    treated as truncation; the request must not be decoded or saved as a Take.
    """

    values = (generated_steps, max_new_tokens, prefill_tokens, graph_max_seq_len)
    if any(not isinstance(value, int) for value in values):
        raise TypeError("graph generation limits must be integers")
    if generated_steps < 0 or max_new_tokens <= 0 or prefill_tokens < 0:
        raise ValueError("graph generation returned invalid token counts")
    if prefill_tokens >= graph_max_seq_len:
        raise GraphGenerationTruncated(
            f"graph prefill ({prefill_tokens}) reached capacity ({graph_max_seq_len})"
        )

    capacity_steps = graph_max_seq_len - prefill_tokens
    if generated_steps >= max_new_tokens:
        raise GraphGenerationTruncated(
            f"generation reached max_new_tokens={max_new_tokens} without confirmed EOS"
        )
    if generated_steps >= capacity_steps:
        raise GraphGenerationTruncated(
            "generation reached the fixed CUDA graph buffer capacity "
            f"({capacity_steps} decode steps) without confirmed EOS"
        )
    return "eos"
