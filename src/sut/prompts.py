"""
prompts.py — Prompt construction for the RAG answer pipeline (T5 / F-03/F-04/F-05).

Three public names:
  SYSTEM_PROMPT        — the behaviour contract sent as the system message.
  build_context_block  — render retrieved chunks as a numbered passage block.
  build_user_prompt    — assemble the final user message from question + context.

Usage::

    from src.sut.prompts import SYSTEM_PROMPT, build_context_block, build_user_prompt
    from src.sut.store import RetrievedChunk

    block, marker_map = build_context_block(chunks)
    user_msg = build_user_prompt(question, block)
    answer_text = provider.generate(user_msg, system=SYSTEM_PROMPT)
"""

from __future__ import annotations

from src.sut.store import RetrievedChunk


# ---------------------------------------------------------------------------
# Behaviour contract — sent as the system message on every call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are a financial research assistant that answers questions about SEC filings.

RULES — follow all of them on every response:

1. ANSWER ONLY FROM THE PROVIDED CONTEXT PASSAGES.
   Do not use any outside knowledge, prior training data, or information not
   present in the numbered passages below. If the context does not contain
   enough information to answer the question, say so explicitly.

2. CITE EVERY CLAIM WITH INLINE MARKERS.
   After each factual statement, append the marker of the passage that supports
   it, in the form [c1], [c2], [c3], etc., referring to the numbered context
   passages supplied in the user message. Use only marker numbers that
   correspond to passages actually provided.

3. REFUSE WHEN THE ANSWER IS NOT IN THE SOURCES.
   If the answer cannot be found in the provided context passages, respond
   with exactly: "The answer is not in the provided sources." Do NOT guess,
   infer, or extrapolate beyond what the passages say. Do not fabricate
   citations for information you cannot find.

4. NEVER GIVE INVESTMENT ADVICE, RECOMMENDATIONS, OR PRICE TARGETS.
   You are an information-only assistant operating in a regulated domain.
   Do not recommend buying, selling, or holding any security. Do not provide
   price targets or forecasts. Do not suggest investment strategies. Inform
   only; defer all investment decisions to a qualified financial adviser.
"""


# ---------------------------------------------------------------------------
# Context block builder
# ---------------------------------------------------------------------------


def build_context_block(
    chunks: list[RetrievedChunk],
) -> tuple[str, dict[str, str]]:
    """Render retrieved chunks as a numbered passage block.

    Labels each chunk ``[c1]``, ``[c2]``, … in the order supplied and returns
    both the rendered text and a mapping from each marker key to the
    corresponding ``chunk_id``.

    Args:
        chunks: Retrieved chunks, typically from :func:`~src.sut.retrieve.retrieve`.

    Returns:
        A 2-tuple ``(block_text, marker_map)`` where:
        - ``block_text`` is the formatted context block ready to embed in the
          user prompt.
        - ``marker_map`` maps marker keys (``"c1"``, ``"c2"``, …) to the
          ``chunk_id`` of the corresponding chunk.
    """
    if not chunks:
        return "", {}

    lines: list[str] = []
    marker_map: dict[str, str] = {}

    for i, chunk in enumerate(chunks, start=1):
        marker = f"c{i}"
        marker_map[marker] = chunk.chunk_id
        lines.append(f"[{marker}] {chunk.text}")

    block = "\n\n".join(lines)
    return block, marker_map


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------


def build_user_prompt(question: str, context_block: str) -> str:
    """Assemble the user message from the question and pre-rendered context block.

    Args:
        question:      The user's natural-language question.
        context_block: The rendered context block from :func:`build_context_block`.

    Returns:
        A formatted string ready to pass as ``prompt`` to ``provider.generate``.
    """
    return (
        f"CONTEXT PASSAGES:\n\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer the question using only the context passages above. "
        "Cite every claim with the appropriate [cN] marker."
    )
