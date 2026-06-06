"""
generate.py — Answer generation with citations and boundary enforcement (T5 / F-03/F-04/F-05).

Public surface:
  Answer             — dataclass holding the answer text, citation map, and retrieved chunks.
  answer_question()  — orchestrates retrieve → prompt → generate → parse_citations.

Usage::

    from src.sut.generate import answer_question
    from src.sut.store import VectorStore

    store = VectorStore(persist_path="datasets/chroma")
    answer = answer_question("What was Apple's revenue in FY2024?", store)
    print(answer.answer)
    print(answer.citations)   # {"c1": "acc-0#Item 7#0", ...}
    print(answer.retrieved)   # list[RetrievedChunk]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import RETRIEVAL_K
from src.sut.prompts import SYSTEM_PROMPT, build_context_block, build_user_prompt
from src.sut.providers import Provider, get_provider
from src.sut.retrieve import retrieve
from src.sut.store import RetrievedChunk, VectorStore

# Regex pattern for inline citation markers: [c1], [c2], [c10], etc.
# Defined once here so both the parser and any prompt-building logic share the same format.
CITATION_MARKER_PATTERN: str = r"\[c(\d+)\]"

_CITATION_RE = re.compile(CITATION_MARKER_PATTERN)


# ---------------------------------------------------------------------------
# Answer dataclass
# ---------------------------------------------------------------------------


@dataclass
class Answer:
    """Result of a single answer_question() call.

    Attributes:
        answer:              The model's answer text, verbatim from provider.generate().
                             May be a factual response with [cN] markers, or a refusal
                             phrase if the context did not support an answer.
        citations:           Mapping from marker key to chunk_id for every [cN] marker
                             that appears in ``answer`` AND maps to a retrieved passage.
                             Empty when the model issues a refusal or cites nothing.
        retrieved:           The list of RetrievedChunk objects returned by retrieve(),
                             in retrieval order (closest first). These are the raw
                             passages the model was given as context.
        unmatched_citations: Markers that appear in the answer text but do NOT map to
                             any retrieved context passage (i.e. the model cited a
                             passage that wasn't provided — over-citation / hallucinated
                             citation). Deduplicated, in order of first appearance.
                             Non-empty values allow T10 to flag over-citation instead of
                             having it laundered away silently.
    """

    answer: str
    citations: dict[str, str] = field(default_factory=dict)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    unmatched_citations: list[str] = field(default_factory=list)
    """Markers that appear in the answer text but do NOT map to any retrieved context passage.

    A non-empty list indicates the model cited a passage that was not provided —
    i.e. over-citation / hallucinated citation.  Captured here so the downstream
    T10 citation-validity metric can flag over-citation rather than having it
    silently discarded.
    """


# ---------------------------------------------------------------------------
# answer_question
# ---------------------------------------------------------------------------


def answer_question(
    question: str,
    store: VectorStore,
    provider: Provider | None = None,
    k: int = RETRIEVAL_K,
) -> Answer:
    """Retrieve context, generate an answer, and parse citations.

    Steps:
    1. Retrieve the top-k chunks from ``store`` via :func:`~src.sut.retrieve.retrieve`.
    2. Build a numbered context block and marker→chunk_id map.
    3. Build the user prompt combining context block and question.
    4. Call ``provider.generate(user_prompt, system=SYSTEM_PROMPT)``.
    5. Parse ``[cN]`` markers from the answer text; map each to its chunk_id
       via the marker map. Only markers that appear in the answer AND exist in
       the marker map are included in ``citations``.
    6. Return an :class:`Answer` with the verbatim answer, citations, and
       retrieved chunks.

    The refusal/advice *behaviour* is the model's responsibility — this
    function constructs the prompt correctly and surfaces whatever the model
    returns without post-filtering.

    Args:
        question: Natural-language question to answer.
        store:    Populated :class:`~src.sut.store.VectorStore` to search.
        provider: A :class:`~src.sut.providers.Provider`-conforming object.
                  When ``None``, falls back to :func:`~src.sut.providers.get_provider`
                  (reads ``EVAL_MODE`` env var; defaults to ``OfflineProvider``).
        k:        Number of chunks to retrieve. Defaults to :data:`~src.config.RETRIEVAL_K`.

    Returns:
        An :class:`Answer` instance.
    """
    # --- Step 1: Retrieve ---
    chunks = retrieve(question, store, k=k)

    # --- Step 2: Build context block ---
    context_block, marker_map = build_context_block(chunks)

    # --- Step 3: Build user prompt ---
    user_prompt = build_user_prompt(question, context_block)

    # --- Step 4: Generate ---
    resolved_provider = provider if provider is not None else get_provider()
    answer_text: str = resolved_provider.generate(
        user_prompt,
        system=SYSTEM_PROMPT,
    )

    # --- Step 5: Parse citations ---
    # Find all [cN] markers that appear in the answer text, in order of appearance.
    # Split into matched (marker exists in marker_map) and unmatched (hallucinated/out-of-range).
    # Deduplicate while preserving first-appearance order.
    citations: dict[str, str] = {}
    unmatched_citations: list[str] = []
    seen_unmatched: set[str] = set()

    for num in _CITATION_RE.findall(answer_text):  # returns ["1", "2", ...]
        marker_key = f"c{num}"
        if marker_key in marker_map:
            citations[marker_key] = marker_map[marker_key]
        elif marker_key not in seen_unmatched:
            unmatched_citations.append(marker_key)
            seen_unmatched.add(marker_key)

    # --- Step 6: Return ---
    return Answer(
        answer=answer_text,
        citations=citations,
        retrieved=chunks,
        unmatched_citations=unmatched_citations,
    )
