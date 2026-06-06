"""
retrieve.py — Thin retrieval seam over VectorStore (T4 / F-02).

This module is the single entry point for retrieval in both the runner (T9)
and the answer generation pipeline (T5). Callers always go through
``retrieve()`` rather than calling ``store.query()`` directly, keeping the
two stages decoupled and testable in isolation.

Usage::

    from src.sut.retrieve import retrieve
    from src.sut.store import VectorStore

    store = VectorStore()
    chunks = retrieve("What was Apple's revenue?", store, k=5)
"""

from __future__ import annotations

from src.config import RETRIEVAL_K
from src.sut.store import RetrievedChunk, VectorStore


def retrieve(
    question: str,
    store: VectorStore,
    k: int = RETRIEVAL_K,
) -> list[RetrievedChunk]:
    """Return the top-k most similar chunks for the given question.

    This is a thin wrapper over :meth:`VectorStore.query` that gives the
    retrieval step a stable, importable name and a single place to add
    pre/post-processing later (e.g. re-ranking, score thresholding) without
    touching callers.

    Args:
        question: The user's natural-language question.
        store:    An initialised VectorStore to search.
        k:        Number of chunks to return. Defaults to :data:`RETRIEVAL_K`.

    Returns:
        List of :class:`~src.sut.store.RetrievedChunk` ordered by similarity
        (closest / most relevant first). Empty when the store is empty or
        no results meet the minimum threshold.
    """
    return store.query(question, k=k)
