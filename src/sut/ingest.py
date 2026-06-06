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

Chunking strategy
-----------------
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
# Chunking logic
# ---------------------------------------------------------------------------


def parse_and_chunk(text: str, meta: dict[str, str]) -> list[Chunk]:
    """Split filing text into overlapping word-window chunks with full metadata.

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
    section = meta.get("section", "")
    source_url = meta.get("source_url", "")

    words = text.split()

    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(words):
        end = min(start + _CHUNK_SIZE_WORDS, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

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

        # Advance by (chunk_size - overlap) so the next chunk re-reads the tail.
        step = _CHUNK_SIZE_WORDS - _CHUNK_OVERLAP_WORDS
        start += step
        idx += 1

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
