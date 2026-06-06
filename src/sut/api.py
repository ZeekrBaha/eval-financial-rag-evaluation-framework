"""
api.py — FastAPI query endpoint for the Financial RAG SUT (T6 / F-02/F-03).

Public surface:
  app        — FastAPI application instance.
  get_store  — Dependency function returning the module-level VectorStore.
               Override in tests via ``app.dependency_overrides[get_store]``.

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /query   → QueryResponse (answer, citations, unmatched_citations,
                                 retrieved, latency_ms)

The module-level store is an empty VectorStore on startup.  In production,
ingest real filings before serving requests (the endpoint works with an empty
store — it returns a refusal-style answer, which is correct behaviour).

Usage (dev server)::

    uv run uvicorn src.sut.api:app --reload

Usage (tests)::

    from fastapi.testclient import TestClient
    from src.sut.api import app, get_store
    from src.sut.store import VectorStore

    store = VectorStore()
    store.add(chunks)
    app.dependency_overrides[get_store] = lambda: store
    client = TestClient(app)
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.config import RETRIEVAL_K
from src.eval.runner import RetrievedRef
from src.sut.generate import answer_question
from src.sut.store import VectorStore

# ---------------------------------------------------------------------------
# Module-level store — empty by default; ingest before real use
# ---------------------------------------------------------------------------

_store: VectorStore = VectorStore()


def get_store() -> VectorStore:
    """Return the module-level VectorStore.

    Override in tests via ``app.dependency_overrides[get_store]``.
    """
    return _store


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, description="Non-empty question string.")
    k: int | None = Field(default=None, description="Number of chunks to retrieve.")


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: dict[str, str]
    unmatched_citations: list[str]
    retrieved: list[RetrievedRef]
    latency_ms: int = Field(ge=0)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Financial RAG SUT",
    description="Query endpoint for the Financial RAG evaluation framework.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    store: Annotated[VectorStore, Depends(get_store)],
) -> QueryResponse:
    """Answer a financial question using the RAG pipeline.

    - Retrieves ``k`` (or ``RETRIEVAL_K``) chunks from the store.
    - Generates an answer with inline [cN] citation markers (offline provider
      by default; no API key required).
    - Returns the answer, citation map, unmatched citations, retrieved chunks,
      and wall-clock latency in milliseconds.

    Returns a 422 if ``question`` is empty.
    """
    effective_k = request.k if request.k is not None else RETRIEVAL_K

    t0 = time.monotonic()
    result = answer_question(request.question, store, k=effective_k)
    latency_ms = int((time.monotonic() - t0) * 1000)

    retrieved_refs = [
        RetrievedRef(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            similarity=chunk.similarity,
            issuer=chunk.issuer,
            form=chunk.form,
            filing_date=chunk.filing_date,
            accession=chunk.accession,
            section=chunk.section,
            source_url=chunk.source_url,
        )
        for chunk in result.retrieved
    ]

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        unmatched_citations=result.unmatched_citations,
        retrieved=retrieved_refs,
        latency_ms=latency_ms,
    )
