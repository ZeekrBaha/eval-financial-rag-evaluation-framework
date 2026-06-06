"""
Tests for src/sut/ingest.py — T3: Ingest + chunk (offline, no network, no API key).

TDD order: tests written BEFORE implementation.
All tests run fully offline — requests/httpx are poisoned in sys.modules
to prove no network call is made.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from src.sut.ingest import Chunk, parse_and_chunk, ingest_fixture, ingest_filing
from src.sut.store import VectorStore


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

REQUIRED_META_FIELDS = {
    "issuer",
    "form",
    "filing_date",
    "accession",
    "section",
    "source_url",
    "chunk_id",
}

_SAMPLE_META = {
    "issuer": "AAPL",
    "form": "10-K",
    "filing_date": "2024-09-28",
    "accession": "0000320193-24-000123",
    "section": "Item 1",
    "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
}

_SHORT_TEXT = "Apple Inc. reported revenue of $391 billion in fiscal 2024."

# ~1800-word text to exercise multi-chunk path (800-token chunks)
_LONG_TEXT = " ".join(["The company reported strong financial results."] * 200)


@pytest.fixture()
def poison_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace requests and httpx with stub modules that raise on any attribute access."""

    class _Poison:
        """Raises RuntimeError on any attribute access to prove no network call occurred."""

        _name: str

        def __init__(self, name: str) -> None:
            object.__setattr__(self, "_name", name)

        def __getattr__(self, item: str) -> None:
            raise RuntimeError(
                f"Network is poisoned — test must not import or use {self._name}.{item}"
            )

    for mod_name in ("requests", "httpx"):
        stub = types.ModuleType(mod_name)
        stub.__class__ = type(
            mod_name,
            (types.ModuleType,),
            {"__getattr__": lambda self, item: _Poison(mod_name).__getattr__(item)},
        )
        monkeypatch.setitem(sys.modules, mod_name, stub)


# ---------------------------------------------------------------------------
# Chunk dataclass / schema
# ---------------------------------------------------------------------------


class TestChunkSchema:
    def test_chunk_has_all_required_fields(self) -> None:
        chunk = Chunk(
            text="hello",
            issuer="AAPL",
            form="10-K",
            filing_date="2024-09-28",
            accession="0000320193-24-000123",
            section="Item 1",
            source_url="https://sec.gov/...",
            chunk_id="0000320193-24-000123#Item 1#0",
        )
        for field in REQUIRED_META_FIELDS:
            assert hasattr(chunk, field), f"Chunk missing field: {field}"

    def test_chunk_text_accessible(self) -> None:
        chunk = Chunk(
            text="revenue data",
            issuer="MSFT",
            form="10-Q",
            filing_date="2024-03-31",
            accession="0000789019-24-000001",
            section="Item 2",
            source_url="https://sec.gov/...",
            chunk_id="0000789019-24-000001#Item 2#0",
        )
        assert chunk.text == "revenue data"


# ---------------------------------------------------------------------------
# parse_and_chunk behaviour
# ---------------------------------------------------------------------------


class TestParseAndChunk:
    def test_short_text_yields_one_chunk(self) -> None:
        chunks = parse_and_chunk(_SHORT_TEXT, _SAMPLE_META)
        assert len(chunks) == 1

    def test_long_text_yields_multiple_chunks(self) -> None:
        chunks = parse_and_chunk(_LONG_TEXT, _SAMPLE_META)
        assert len(chunks) > 1

    def test_all_chunks_have_required_meta_fields(self) -> None:
        chunks = parse_and_chunk(_LONG_TEXT, _SAMPLE_META)
        for chunk in chunks:
            for field in REQUIRED_META_FIELDS:
                assert hasattr(chunk, field), f"chunk missing field: {field}"
                value = getattr(chunk, field)
                assert value is not None and value != "", (
                    f"chunk.{field} must be non-empty"
                )

    def test_chunk_ids_are_unique_within_filing(self) -> None:
        chunks = parse_and_chunk(_LONG_TEXT, _SAMPLE_META)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "chunk_ids must be unique within a filing"

    def test_chunk_id_contains_accession_and_section(self) -> None:
        chunks = parse_and_chunk(_SHORT_TEXT, _SAMPLE_META)
        cid = chunks[0].chunk_id
        assert _SAMPLE_META["accession"] in cid
        assert _SAMPLE_META["section"] in cid

    def test_chunks_have_overlap(self) -> None:
        """Adjacent chunks should share some words (overlap > 0)."""
        chunks = parse_and_chunk(_LONG_TEXT, _SAMPLE_META)
        if len(chunks) < 2:
            pytest.skip("need at least 2 chunks to test overlap")
        words_a = set(chunks[0].text.split())
        words_b = set(chunks[1].text.split())
        assert len(words_a & words_b) > 0, "adjacent chunks should have word overlap"

    def test_metadata_propagated_to_every_chunk(self) -> None:
        chunks = parse_and_chunk(_LONG_TEXT, _SAMPLE_META)
        for chunk in chunks:
            assert chunk.issuer == _SAMPLE_META["issuer"]
            assert chunk.form == _SAMPLE_META["form"]
            assert chunk.filing_date == _SAMPLE_META["filing_date"]
            assert chunk.accession == _SAMPLE_META["accession"]
            assert chunk.section == _SAMPLE_META["section"]
            assert chunk.source_url == _SAMPLE_META["source_url"]

    def test_chunk_text_is_nonempty(self) -> None:
        chunks = parse_and_chunk(_SHORT_TEXT, _SAMPLE_META)
        for chunk in chunks:
            assert chunk.text.strip(), "chunk text must not be blank"


# ---------------------------------------------------------------------------
# ingest_filing (uses a supplied store, verifies count)
# ---------------------------------------------------------------------------


class TestIngestFiling:
    def test_ingest_filing_returns_chunk_count(self) -> None:
        store = VectorStore()
        count = ingest_filing(_SHORT_TEXT, _SAMPLE_META, store)
        assert count == 1

    def test_ingest_filing_long_text_returns_multiple(self) -> None:
        store = VectorStore()
        count = ingest_filing(_LONG_TEXT, _SAMPLE_META, store)
        assert count > 1


# ---------------------------------------------------------------------------
# ingest_fixture — offline end-to-end (no network)
# ---------------------------------------------------------------------------


class TestIngestFixture:
    def test_fixture_file_ingested_successfully(
        self, tmp_path: Path, poison_network: None
    ) -> None:
        """Write a fixture file, ingest it offline, query, assert results."""
        fixture = tmp_path / "test_filing.txt"
        fixture.write_text(
            "Item 7. Management's Discussion and Analysis of Financial Condition "
            "and Results of Operations.\n\n"
            "Apple Inc. reported total net sales of $391.035 billion for fiscal year 2024, "
            "compared to $383.285 billion in fiscal year 2023. "
            "The increase was primarily driven by growth in Services revenue.\n\n"
            + "Operating expenses were well-managed throughout the period. " * 40,
            encoding="utf-8",
        )
        meta = {
            "issuer": "AAPL",
            "form": "10-K",
            "filing_date": "2024-09-28",
            "accession": "0000320193-24-000456",
            "section": "Item 7",
            "source_url": "https://www.sec.gov/Archives/edgar/data/320193/test.htm",
        }
        store = VectorStore()
        count = ingest_fixture(str(fixture), store, meta=meta)
        assert count >= 1

    def test_fixture_query_returns_chunks_with_metadata(
        self, tmp_path: Path, poison_network: None
    ) -> None:
        """After ingest_fixture, store.query returns RetrievedChunk with all metadata."""
        fixture = tmp_path / "filing.txt"
        fixture.write_text(
            "Item 1A. Risk Factors.\n\n"
            "The company faces significant competition in all markets in which it operates. "
            "Competitors include large well-resourced technology companies. "
            "Market conditions change rapidly.\n\n"
            + "Additional risk factors relate to supply chain disruptions. " * 30,
            encoding="utf-8",
        )
        meta = {
            "issuer": "MSFT",
            "form": "10-K",
            "filing_date": "2024-06-30",
            "accession": "0000789019-24-000099",
            "section": "Item 1A",
            "source_url": "https://www.sec.gov/Archives/edgar/data/789019/test.htm",
        }
        store = VectorStore()
        ingest_fixture(str(fixture), store, meta=meta)

        results = store.query("competition risks", k=3)
        assert len(results) >= 1

        # Every result must carry all metadata fields
        for r in results:
            for field in REQUIRED_META_FIELDS:
                assert hasattr(r, field), f"RetrievedChunk missing field: {field}"
                value = getattr(r, field)
                assert value is not None and value != "", (
                    f"RetrievedChunk.{field} must be non-empty"
                )
            # Score must be present and numeric
            assert isinstance(r.score, float), "score must be a float"

    def test_fixture_no_meta_arg_uses_defaults(
        self, tmp_path: Path, poison_network: None
    ) -> None:
        """ingest_fixture without explicit meta uses sensible defaults (no crash)."""
        fixture = tmp_path / "filing2.txt"
        fixture.write_text("Revenue was $100 billion.", encoding="utf-8")
        store = VectorStore()
        count = ingest_fixture(str(fixture), store)
        assert count >= 1


# ---------------------------------------------------------------------------
# fetch_filing — existence check only (not executed in tests)
# ---------------------------------------------------------------------------


class TestFetchFilingExists:
    def test_fetch_filing_is_importable(self) -> None:
        from src.sut.ingest import fetch_filing
        assert callable(fetch_filing)
