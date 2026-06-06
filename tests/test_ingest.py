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
# Section-aware chunking (new tests — T3 spec fix)
# ---------------------------------------------------------------------------

# A filing text with two recognisable ITEM headers plus body content.
_SECTIONED_TEXT = (
    "Some preamble before any item.\n\n"
    "Item 1. Business.\n\n"
    + "The company operates in many markets. " * 5 + "\n\n"
    "Item 7. Management's Discussion and Analysis.\n\n"
    + "Revenue grew significantly year over year. " * 5
)

_SECTIONED_META = {
    "issuer": "AAPL",
    "form": "10-K",
    "filing_date": "2023-09-30",
    "accession": "AAPL-10K-2023",
    "section": "full",
    "source_url": "https://sec.gov/test",
}


class TestSectionAwareChunking:
    def test_sections_detected_from_item_headers(self) -> None:
        """Chunks from Item 1 and Item 7 sections carry correct section slugs."""
        chunks = parse_and_chunk(_SECTIONED_TEXT, _SECTIONED_META)
        sections_found = {c.section for c in chunks}
        assert "item1" in sections_found, "expected item1 slug"
        assert "item7" in sections_found, "expected item7 slug"

    def test_chunk_ids_include_section_slug(self) -> None:
        """chunk_id must embed the section slug (e.g. AAPL-10K-2023#item7#0)."""
        chunks = parse_and_chunk(_SECTIONED_TEXT, _SECTIONED_META)
        item7_chunks = [c for c in chunks if c.section == "item7"]
        assert item7_chunks, "no item7 chunks found"
        for c in item7_chunks:
            assert "item7" in c.chunk_id, f"item7 not in chunk_id: {c.chunk_id}"
            assert _SECTIONED_META["accession"] in c.chunk_id

    def test_section_idx_restarts_per_section(self) -> None:
        """The first chunk of each section has idx 0 (chunk_id ends with #0)."""
        chunks = parse_and_chunk(_SECTIONED_TEXT, _SECTIONED_META)
        for slug in ("item1", "item7"):
            section_chunks = [c for c in chunks if c.section == slug]
            if section_chunks:
                assert section_chunks[0].chunk_id.endswith("#0"), (
                    f"first chunk of {slug} should have idx 0, "
                    f"got {section_chunks[0].chunk_id}"
                )

    def test_no_headers_falls_back_to_provided_section(self) -> None:
        """Text with no ITEM headers uses the meta-provided section slug."""
        plain_text = "Revenue was $100 billion. Costs were $80 billion."
        meta = dict(_SAMPLE_META)  # section = "Item 1"
        chunks = parse_and_chunk(plain_text, meta)
        assert len(chunks) == 1
        assert chunks[0].section == meta["section"]

    def test_no_headers_falls_back_to_full_when_section_missing(self) -> None:
        """Text with no headers and no section in meta falls back to 'full'."""
        plain_text = "Revenue was $100 billion."
        meta = {k: v for k, v in _SAMPLE_META.items() if k != "section"}
        chunks = parse_and_chunk(plain_text, meta)
        assert chunks[0].section == "full"


class TestCrossFilingChunkIdUniqueness:
    def test_two_different_fixture_files_no_collision(
        self, tmp_path: Path, poison_network: None
    ) -> None:
        """Ingesting two fixture files without explicit accession must not raise."""
        filing_a = tmp_path / "filing_a.txt"
        filing_b = tmp_path / "filing_b.txt"
        filing_a.write_text("Apple revenue was $391 billion in 2024.", encoding="utf-8")
        filing_b.write_text("Microsoft revenue was $245 billion in 2024.", encoding="utf-8")

        store = VectorStore()
        # Neither call supplies meta, so accession is derived from filename stem.
        count_a = ingest_fixture(str(filing_a), store)
        # Must not raise due to duplicate chunk_ids.
        count_b = ingest_fixture(str(filing_b), store)

        assert count_a >= 1
        assert count_b >= 1
        assert store._collection.count() == count_a + count_b


# ---------------------------------------------------------------------------
# TOC-rejection tests (T3 correctness fix)
# ---------------------------------------------------------------------------


# A 10-K whose text includes both a TOC block AND real body headers for item1a.
# The TOC entry has a dot-leader run; the body header is a plain line.
_TOC_FILING_TEXT = (
    # Table of contents block
    "PART I\n\n"
    "   Item 1. Business .......... 3\n"
    "   Item 1A. Risk Factors .......... 12\n"
    "   Item 7. MD&A .......... 45\n\n"
    # Real body content
    "Item 1. Business.\n\n"
    "The company was incorporated in 1976 and operates globally.\n\n"
    "Item 1A.\n\n"
    "Risk Factors body content here. The company faces substantial competition "
    "and regulatory risks in all markets.\n\n"
    "Item 7. Management Discussion.\n\n"
    "Revenue grew 8% year over year driven by Services segment expansion.\n"
)

_TOC_FILING_META = {
    "issuer": "AAPL",
    "form": "10-K",
    "filing_date": "2024-09-28",
    "accession": "0000320193-24-TOC-TEST",
    "section": "full",
    "source_url": "https://sec.gov/test",
}


class TestTOCRejection:
    def test_toc_entry_not_treated_as_header(self) -> None:
        """TOC lines with dot-leaders must not create extra sections."""
        chunks = parse_and_chunk(_TOC_FILING_TEXT, _TOC_FILING_META)
        # Only one item1a section should exist (the body one, not the TOC entry).
        item1a_chunks = [c for c in chunks if c.section == "item1a"]
        assert item1a_chunks, "expected at least one item1a chunk from body"
        # Check body content is present (not clobbered by TOC entry).
        all_text = " ".join(c.text for c in item1a_chunks)
        assert "Risk Factors body content here" in all_text, (
            "body content of item1a was lost; TOC entry may have overwritten it"
        )

    def test_chunk_ids_unique_with_toc_present(self) -> None:
        """Even with a TOC block present, all chunk_ids must be unique."""
        chunks = parse_and_chunk(_TOC_FILING_TEXT, _TOC_FILING_META)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), (
            f"duplicate chunk_ids: {[cid for cid in ids if ids.count(cid) > 1]}"
        )

    def test_only_one_item1a_section_slug(self) -> None:
        """After section detection + dedup, item1a appears exactly once as a slug."""
        from src.sut.ingest import _split_into_sections
        sections = _split_into_sections(_TOC_FILING_TEXT, "full")
        slugs = [slug for slug, _ in sections]
        item1a_count = slugs.count("item1a")
        assert item1a_count == 1, (
            f"expected 1 item1a section, got {item1a_count}; all slugs: {slugs}"
        )

    def test_plural_items_phrase_not_treated_as_header(self) -> None:
        """'Items 1 through 4' (plural) must NOT be detected as a section header."""
        text = "Items 1 through 4 are incorporated by reference from the annual report."
        chunks = parse_and_chunk(text, _SECTIONED_META)
        # Should fall back to the meta-provided section slug, not produce item1.
        assert all(c.section == _SECTIONED_META["section"] for c in chunks), (
            "plural 'Items' phrase was incorrectly parsed as a section header"
        )

    def test_page_number_only_trailer_excluded(self) -> None:
        """A line like 'Item 2. Properties   15' (trailing page number) is a TOC line."""
        text = (
            "Item 2. Properties   15\n\n"  # TOC line — trailing page number
            "Item 2. Properties.\n\n"       # real body header
            "The company owns its headquarters building.\n"
        )
        meta = dict(_TOC_FILING_META)
        from src.sut.ingest import _split_into_sections
        sections = _split_into_sections(text, "full")
        slugs = [slug for slug, _ in sections]
        assert slugs.count("item2") == 1, (
            f"item2 appeared {slugs.count('item2')} times; expected 1. Slugs: {slugs}"
        )


# ---------------------------------------------------------------------------
# fetch_filing — existence check only (not executed in tests)
# ---------------------------------------------------------------------------


class TestFetchFilingExists:
    def test_fetch_filing_is_importable(self) -> None:
        from src.sut.ingest import fetch_filing
        assert callable(fetch_filing)
