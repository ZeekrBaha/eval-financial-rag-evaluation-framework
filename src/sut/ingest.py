"""
ingest.py — Fetch, parse, chunk, and ingest SEC filings into the vector store.

Single responsibility: turn raw filing text into Chunk objects and push them
into a VectorStore.  The live fetch path (fetch_filing) requires network access
and is NOT exercised in tests.  The offline path (ingest_fixture) reads a local
file and runs the same pipeline — no network, no API key.

Chunk shape
-----------
Each chunk carries all 7 metadata fields:
    issuer, form, filing_date, accession, section, source_url, chunk_id

chunk_id is stable across runs:
    f"{accession}#{section}#{idx}"
where idx restarts from 0 within each detected section.

Chunking strategy
-----------------
Section-aware: detects SEC ITEM headers (e.g. "Item 1A.", "ITEM 7") and splits
the filing into sections first.  Within each section the existing ~800-word /
~100-word-overlap window is applied.  Chunks before the first header (or when
no headers exist) use the meta-provided section (or "full").

~800 tokens, ~100 token overlap.  We approximate tokens as whitespace-split
words (1 token ≈ 1 word for English prose, good enough for a ~800-word window).
No tokenizer dependency is required.

Usage (offline)::

    from src.sut.ingest import ingest_fixture
    from src.sut.store import VectorStore

    store = VectorStore()
    count = ingest_fixture("path/to/filing.txt", store, meta={...})
    results = store.query("revenue", k=5)

Usage (live)::

    from src.sut.ingest import fetch_filing
    from src.sut.store import VectorStore

    store = VectorStore(persist_path="datasets/chroma")
    fetch_filing(cik="0000320193", accession="0000320193-24-000123", store=store)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.sut.store import VectorStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Word-based approximation: 1 word ≈ 1 token for English prose.
_CHUNK_SIZE_WORDS = 800
_CHUNK_OVERLAP_WORDS = 100

# Raw EDGAR download cache root.
_RAW_CACHE_DIR = Path(__file__).parent.parent.parent / "datasets" / "raw"

# SEC EDGAR user-agent — required by SEC fair-access policy.
_SEC_USER_AGENT = "eval-financial-rag adatub08@gmail.com"

# Default metadata when none is supplied to ingest_fixture.
_DEFAULT_META: dict[str, str] = {
    "issuer": "unknown",
    "form": "unknown",
    "filing_date": "unknown",
    "accession": "unknown",
    "section": "unknown",
    "source_url": "unknown",
}


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single text chunk from a parsed SEC filing.

    All 7 metadata fields are required.  chunk_id must be unique within a filing.
    """

    text: str
    issuer: str
    form: str
    filing_date: str
    accession: str
    section: str
    source_url: str
    chunk_id: str


# ---------------------------------------------------------------------------
# SEC section-header detection helpers
# ---------------------------------------------------------------------------

# Matches lines like "Item 1.", "ITEM 1A.", "Item 7", "ITEM 1A" etc.
_SEC_ITEM_HEADER_RE = re.compile(
    r"^\s*ITEM\s+(\d+[A-Z]?)\.?",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_section_slug(item_label: str) -> str:
    """Turn a raw ITEM label (e.g. '1A', '7') into a lowercase slug ('item1a', 'item7')."""
    return "item" + item_label.lower()


def _split_into_sections(text: str, fallback_section: str) -> list[tuple[str, str]]:
    """Detect SEC ITEM headers in *text* and return (section_slug, section_text) pairs.

    If no headers are found, returns a single tuple (fallback_section, text).
    The text before the first header (if any) is grouped under fallback_section.
    """
    matches = list(_SEC_ITEM_HEADER_RE.finditer(text))

    if not matches:
        return [(fallback_section, text)]

    sections: list[tuple[str, str]] = []

    # Text before the first header → fallback_section
    preamble = text[: matches[0].start()]
    if preamble.strip():
        sections.append((fallback_section, preamble))

    for i, match in enumerate(matches):
        slug = _normalize_section_slug(match.group(1))
        section_start = match.start()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[section_start:section_end]
        sections.append((slug, section_text))

    return sections


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------


def _chunk_words(
    words: list[str],
    section: str,
    accession: str,
    issuer: str,
    form: str,
    filing_date: str,
    source_url: str,
) -> list[Chunk]:
    """Apply the ~800-word / ~100-word-overlap window over *words* for a single section."""
    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(words):
        end = min(start + _CHUNK_SIZE_WORDS, len(words))
        chunk_text = " ".join(words[start:end])
        chunk_id = f"{accession}#{section}#{idx}"

        chunks.append(
            Chunk(
                text=chunk_text,
                issuer=issuer,
                form=form,
                filing_date=filing_date,
                accession=accession,
                section=section,
                source_url=source_url,
                chunk_id=chunk_id,
            )
        )

        step = _CHUNK_SIZE_WORDS - _CHUNK_OVERLAP_WORDS
        start += step
        idx += 1

    return chunks


def parse_and_chunk(text: str, meta: dict[str, str]) -> list[Chunk]:
    """Split filing text into overlapping word-window chunks with full metadata.

    Detects SEC ITEM headers and splits the filing into sections first.  Within
    each section the existing ~800-word / ~100-word-overlap window is applied.
    chunk_id restarts (idx=0) per section.

    Args:
        text: Raw filing text.
        meta: Dict with keys: issuer, form, filing_date, accession,
              section, source_url.  Missing keys fall back to empty string.

    Returns:
        List of Chunk objects, each carrying all 7 metadata fields.
    """
    issuer = meta.get("issuer", "")
    form = meta.get("form", "")
    filing_date = meta.get("filing_date", "")
    accession = meta.get("accession", "")
    fallback_section = meta.get("section", "") or "full"
    source_url = meta.get("source_url", "")

    if not text.split():
        return []

    sections = _split_into_sections(text, fallback_section)

    chunks: list[Chunk] = []
    for section_slug, section_text in sections:
        words = section_text.split()
        if not words:
            continue
        chunks.extend(
            _chunk_words(
                words=words,
                section=section_slug,
                accession=accession,
                issuer=issuer,
                form=form,
                filing_date=filing_date,
                source_url=source_url,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# ingest_filing — parse, chunk, and add to a store
# ---------------------------------------------------------------------------


def ingest_filing(text: str, meta: dict[str, str], store: "VectorStore") -> int:
    """Parse and chunk a filing text, embed via store's provider, and persist.

    Args:
        text:  Raw filing text.
        meta:  Metadata dict (see parse_and_chunk).
        store: A VectorStore instance to add chunks to.

    Returns:
        Number of chunks ingested.
    """
    chunks = parse_and_chunk(text, meta)
    if chunks:
        store.add(chunks)
    return len(chunks)


# ---------------------------------------------------------------------------
# ingest_fixture — offline path (reads a local file)
# ---------------------------------------------------------------------------


def ingest_fixture(
    path: str,
    store: "VectorStore",
    meta: dict[str, str] | None = None,
) -> int:
    """Read a local fixture filing text file and run the full ingest pipeline.

    No network access.  Used in tests and for reproducible offline demos.

    Args:
        path:  Absolute or relative path to a plain-text filing file.
        store: VectorStore instance to ingest into.
        meta:  Optional metadata dict.  Falls back to _DEFAULT_META for
               any missing key, so callers don't have to supply all fields.

    Returns:
        Number of chunks ingested.
    """
    filing_path = Path(path)
    text = filing_path.read_text(encoding="utf-8")

    effective_meta = dict(_DEFAULT_META)
    # Derive accession from filename stem so fixtures without explicit meta
    # don't collide in the vector store when multiple fixture files are ingested.
    effective_meta["accession"] = filing_path.stem
    if meta:
        effective_meta.update(meta)

    return ingest_filing(text, effective_meta, store)


# ---------------------------------------------------------------------------
# fetch_filing — LIVE path (network required); NOT called in tests
# ---------------------------------------------------------------------------


def fetch_filing(
    cik: str,
    accession: str,
    store: "VectorStore",
    form: str = "10-K",
    issuer: str = "",
    filing_date: str = "",
) -> int:
    """Download a filing from SEC EDGAR, cache it locally, and ingest it.

    This function requires network access.  It is NOT called in offline tests.
    Raw responses are cached under datasets/raw/<accession>.txt so re-runs
    skip the download.

    Args:
        cik:         SEC Central Index Key (zero-padded, e.g. "0000320193").
        accession:   Accession number (e.g. "0000320193-24-000123").
        store:       VectorStore to ingest into.
        form:        Form type label ("10-K" or "10-Q").
        issuer:      Ticker or name (used as metadata).
        filing_date: Filing date string (e.g. "2024-09-28").

    Returns:
        Number of chunks ingested.
    """
    # Lazy imports so offline mode never needs requests/httpx.
    import time  # noqa: PLC0415

    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "requests is not installed. Install with: uv add requests"
        ) from exc

    # Build cache path
    _RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _RAW_CACHE_DIR / f"{accession}.txt"

    if cache_file.exists():
        text = cache_file.read_text(encoding="utf-8")
    else:
        # Construct EDGAR filing index URL.
        # Accession number format: XXXXXXXXXX-YY-ZZZZZZ → XXXXXXXXXXYYYZZZZZZ (no dashes)
        accession_nodash = accession.replace("-", "")
        index_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik}&type={form}"
            f"&dateb=&owner=include&count=10&search_text="
        )
        # Direct document URL for the filing index
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/"
            f"{accession_nodash}/{accession_nodash}.txt"
        )

        headers = {"User-Agent": _SEC_USER_AGENT}

        # Polite rate limiting — SEC requests 10 req/s max
        time.sleep(0.1)

        response = requests.get(filing_url, headers=headers, timeout=30)
        response.raise_for_status()
        text = response.text

        # Cache the raw download
        cache_file.write_text(text, encoding="utf-8")

    source_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/"
        f"{accession.replace('-', '')}/{accession.replace('-', '')}.txt"
    )

    meta = {
        "issuer": issuer,
        "form": form,
        "filing_date": filing_date,
        "accession": accession,
        "section": "full",
        "source_url": source_url,
    }

    return ingest_filing(text, meta, store)
