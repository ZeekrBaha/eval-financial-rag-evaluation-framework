"""
Tests for src/sut/store.py — T3: VectorStore (Chroma wrapper).

TDD order: tests written BEFORE implementation.
All tests are fully offline — ephemeral Chroma, OfflineProvider for embeddings.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from src.sut.ingest import Chunk
from src.sut.store import VectorStore, RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    text: str = "Apple reported revenue of $391 billion.",
    issuer: str = "AAPL",
    form: str = "10-K",
    filing_date: str = "2024-09-28",
    accession: str = "0000320193-24-000123",
    section: str = "Item 7",
    source_url: str = "https://sec.gov/test",
    idx: int = 0,
) -> Chunk:
    return Chunk(
        text=text,
        issuer=issuer,
        form=form,
        filing_date=filing_date,
        accession=accession,
        section=section,
        source_url=source_url,
        chunk_id=f"{accession}#{section}#{idx}",
    )


# ---------------------------------------------------------------------------
# VectorStore construction
# ---------------------------------------------------------------------------


class TestVectorStoreInit:
    def test_default_construction_is_ephemeral(self) -> None:
        """VectorStore() with no args creates an in-memory store (no disk I/O)."""
        store = VectorStore()
        assert store is not None

    def test_construction_with_collection_name(self) -> None:
        store = VectorStore(collection_name="my_test_collection")
        assert store is not None


# ---------------------------------------------------------------------------
# add — single and batch
# ---------------------------------------------------------------------------


class TestVectorStoreAdd:
    def test_add_single_chunk(self) -> None:
        store = VectorStore()
        chunk = _make_chunk()
        store.add([chunk])  # should not raise

    def test_add_multiple_chunks(self) -> None:
        store = VectorStore()
        chunks = [_make_chunk(text=f"chunk {i}", idx=i) for i in range(5)]
        store.add(chunks)  # should not raise

    def test_add_returns_none(self) -> None:
        store = VectorStore()
        result = store.add([_make_chunk()])
        assert result is None

    def test_add_empty_list_is_noop(self) -> None:
        store = VectorStore()
        store.add([])  # must not raise


# ---------------------------------------------------------------------------
# query — basic roundtrip
# ---------------------------------------------------------------------------


class TestVectorStoreQuery:
    def test_query_returns_list_of_retrieved_chunks(self) -> None:
        store = VectorStore()
        store.add([_make_chunk()])
        results = store.query("revenue", k=1)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)

    def test_query_result_has_score(self) -> None:
        store = VectorStore()
        store.add([_make_chunk()])
        results = store.query("revenue", k=1)
        assert isinstance(results[0].score, float)

    def test_query_result_has_text(self) -> None:
        store = VectorStore()
        store.add([_make_chunk(text="unique filing text")])
        results = store.query("unique filing text", k=1)
        assert results[0].text == "unique filing text"

    def test_query_result_has_all_metadata_fields(self) -> None:
        store = VectorStore()
        chunk = _make_chunk()
        store.add([chunk])
        results = store.query("revenue", k=1)
        r = results[0]
        assert r.issuer == chunk.issuer
        assert r.form == chunk.form
        assert r.filing_date == chunk.filing_date
        assert r.accession == chunk.accession
        assert r.section == chunk.section
        assert r.source_url == chunk.source_url
        assert r.chunk_id == chunk.chunk_id

    def test_query_respects_k_limit(self) -> None:
        store = VectorStore()
        chunks = [_make_chunk(text=f"document {i}", idx=i) for i in range(10)]
        store.add(chunks)
        results = store.query("document", k=3)
        assert len(results) <= 3

    def test_query_default_k_returns_up_to_retrieval_k(self) -> None:
        from src.config import RETRIEVAL_K
        store = VectorStore()
        chunks = [_make_chunk(text=f"text {i}", idx=i) for i in range(10)]
        store.add(chunks)
        results = store.query("text")  # k defaults to RETRIEVAL_K
        assert len(results) <= RETRIEVAL_K

    def test_query_empty_store_returns_empty_list(self) -> None:
        store = VectorStore()
        results = store.query("anything", k=5)
        assert results == []

    def test_query_k_larger_than_store_returns_all(self) -> None:
        store = VectorStore()
        chunks = [_make_chunk(text=f"item {i}", idx=i) for i in range(3)]
        store.add(chunks)
        results = store.query("item", k=10)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# RetrievedChunk model
# ---------------------------------------------------------------------------


class TestRetrievedChunk:
    def test_retrieved_chunk_has_score_field(self) -> None:
        rc = RetrievedChunk(
            text="hello",
            issuer="AAPL",
            form="10-K",
            filing_date="2024-09-28",
            accession="acc-1",
            section="Item 1",
            source_url="https://sec.gov/",
            chunk_id="acc-1#Item 1#0",
            score=0.95,
        )
        assert rc.score == 0.95

    def test_retrieved_chunk_score_is_float(self) -> None:
        rc = RetrievedChunk(
            text="hello",
            issuer="AAPL",
            form="10-K",
            filing_date="2024-09-28",
            accession="acc-1",
            section="Item 1",
            source_url="https://sec.gov/",
            chunk_id="acc-1#Item 1#0",
            score=0.5,
        )
        assert isinstance(rc.score, float)


# ---------------------------------------------------------------------------
# similarity property
# ---------------------------------------------------------------------------


class TestRetrievedChunkSimilarity:
    def test_similarity_is_one_minus_score(self) -> None:
        rc = RetrievedChunk(
            text="hello",
            issuer="AAPL",
            form="10-K",
            filing_date="2024-09-28",
            accession="acc-1",
            section="Item 1",
            source_url="https://sec.gov/",
            chunk_id="acc-1#Item 1#0",
            score=0.25,
        )
        assert rc.similarity == pytest.approx(0.75)

    def test_similarity_clamped_to_zero_for_high_distance(self) -> None:
        rc = RetrievedChunk(
            text="hello",
            issuer="AAPL",
            form="10-K",
            filing_date="2024-09-28",
            accession="acc-1",
            section="Item 1",
            source_url="https://sec.gov/",
            chunk_id="acc-1#Item 1#0",
            score=1.5,  # distance > 1.0 should clamp to 0
        )
        assert rc.similarity == 0.0

    def test_similarity_clamped_to_one_for_negative_distance(self) -> None:
        rc = RetrievedChunk(
            text="hello",
            issuer="AAPL",
            form="10-K",
            filing_date="2024-09-28",
            accession="acc-1",
            section="Item 1",
            source_url="https://sec.gov/",
            chunk_id="acc-1#Item 1#0",
            score=-0.1,  # should clamp to 1.0
        )
        assert rc.similarity == 1.0

    def test_score_field_still_present(self) -> None:
        """Adding similarity must not remove score."""
        rc = RetrievedChunk(
            text="hello",
            issuer="AAPL",
            form="10-K",
            filing_date="2024-09-28",
            accession="acc-1",
            section="Item 1",
            source_url="https://sec.gov/",
            chunk_id="acc-1#Item 1#0",
            score=0.3,
        )
        assert rc.score == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Query ordering: closest-first (ascending distance / descending similarity)
# ---------------------------------------------------------------------------


class TestQueryOrdering:
    def test_results_ordered_ascending_by_score(self) -> None:
        """store.query() must return results ordered closest-first (ascending distance)."""
        store = VectorStore()
        # Ingest distinct chunks; OfflineProvider produces deterministic embeddings.
        chunks = [_make_chunk(text=f"document about topic number {i}", idx=i) for i in range(5)]
        store.add(chunks)
        results = store.query("document about topic", k=5)
        assert len(results) >= 2, "need at least 2 results to check ordering"
        scores = [r.score for r in results]
        assert scores == sorted(scores), (
            f"results not ordered by ascending score (distance): {scores}"
        )

    def test_results_ordered_descending_by_similarity(self) -> None:
        """similarity values must be non-increasing (descending) across results."""
        store = VectorStore()
        chunks = [_make_chunk(text=f"financial report section {i}", idx=i) for i in range(5)]
        store.add(chunks)
        results = store.query("financial report", k=5)
        assert len(results) >= 2
        similarities = [r.similarity for r in results]
        assert similarities == sorted(similarities, reverse=True), (
            f"similarity not descending: {similarities}"
        )


# ---------------------------------------------------------------------------
# persist path (smoke test — creates a PersistentClient if path given)
# ---------------------------------------------------------------------------


class TestVectorStorePersistPath:
    def test_persistent_path_construction(self, tmp_path: "Path") -> None:
        store = VectorStore(persist_path=str(tmp_path / "chroma_db"))
        store.add([_make_chunk()])
        results = store.query("revenue", k=1)
        assert len(results) == 1
