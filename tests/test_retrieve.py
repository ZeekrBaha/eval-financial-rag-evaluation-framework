"""
Tests for src/sut/retrieve.py — T4: Retrieve (F-02).

TDD: tests written before implementation.
All tests run offline — no network, no API key required.
"""

from __future__ import annotations

from src.sut.ingest import Chunk
from src.sut.retrieve import retrieve
from src.sut.store import RetrievedChunk, VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    text: str = "Apple reported net sales of 383 billion dollars.",
    idx: int = 0,
) -> Chunk:
    return Chunk(
        text=text,
        issuer="AAPL",
        form="10-K",
        filing_date="2024-09-28",
        accession=f"acc-{idx}",
        section="Item 7",
        source_url="https://sec.gov/",
        chunk_id=f"acc-{idx}#Item 7#{idx}",
    )


def _make_store_with_chunks(n: int = 3) -> VectorStore:
    store = VectorStore()
    chunks = [
        _make_chunk(text=f"Financial data passage number {i}", idx=i)
        for i in range(n)
    ]
    store.add(chunks)
    return store


# ---------------------------------------------------------------------------
# retrieve — basic contract
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_retrieve_returns_list_of_retrieved_chunks(self) -> None:
        store = _make_store_with_chunks(3)
        results = retrieve("revenue", store)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_retrieve_respects_k(self) -> None:
        store = _make_store_with_chunks(10)
        results = retrieve("revenue", store, k=3)
        assert len(results) <= 3

    def test_retrieve_default_k_from_config(self) -> None:
        from src.config import RETRIEVAL_K

        store = _make_store_with_chunks(10)
        results = retrieve("revenue", store)
        assert len(results) <= RETRIEVAL_K

    def test_retrieve_results_have_score(self) -> None:
        store = _make_store_with_chunks(2)
        results = retrieve("revenue", store, k=2)
        for r in results:
            assert isinstance(r.score, float)

    def test_retrieve_results_have_full_metadata(self) -> None:
        store = VectorStore()
        chunk = _make_chunk(text="unique passage about earnings", idx=42)
        store.add([chunk])

        results = retrieve("earnings", store, k=1)
        assert len(results) == 1
        r = results[0]
        assert r.issuer == chunk.issuer
        assert r.form == chunk.form
        assert r.filing_date == chunk.filing_date
        assert r.accession == chunk.accession
        assert r.section == chunk.section
        assert r.source_url == chunk.source_url
        assert r.chunk_id == chunk.chunk_id

    def test_retrieve_empty_store_returns_empty(self) -> None:
        store = VectorStore()
        results = retrieve("revenue", store, k=5)
        assert results == []

    def test_retrieve_k_larger_than_store_returns_all(self) -> None:
        store = _make_store_with_chunks(2)
        results = retrieve("revenue", store, k=100)
        assert len(results) == 2

    def test_retrieve_is_thin_wrapper_over_store_query(self) -> None:
        """retrieve() and store.query() should return the same results."""
        store = _make_store_with_chunks(5)
        direct = store.query("revenue", k=3)
        via_retrieve = retrieve("revenue", store, k=3)
        # Chunk IDs should match in the same order
        assert [r.chunk_id for r in via_retrieve] == [r.chunk_id for r in direct]
