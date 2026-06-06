"""
store.py — Chroma vector store wrapper.

Single responsibility: persist chunks as embeddings and retrieve by semantic
similarity. Embeddings are produced via get_provider() (OfflineProvider by
default), so no network or API key is needed in offline mode.

Usage:
    from src.sut.store import VectorStore

    store = VectorStore()                          # ephemeral (tests)
    store = VectorStore(persist_path="/tmp/chroma")  # on-disk (live runs)

    store.add(chunks)
    results = store.query("What was Apple's revenue?", k=5)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chromadb

from src.config import RETRIEVAL_K
from src.sut.providers import get_provider

if TYPE_CHECKING:
    from src.sut.ingest import Chunk


# ---------------------------------------------------------------------------
# RetrievedChunk — query result with score attached
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A chunk returned from a similarity query, augmented with a distance score.

    Attributes:
        score:      Raw Chroma cosine distance. LOWER means MORE similar.
                    Results from VectorStore.query() are ordered ascending by
                    score (i.e. closest / most similar first).
        similarity: Derived convenience property: ``1.0 - score``, clamped to
                    [0.0, 1.0].  HIGHER means MORE similar.  Do not use ``score``
                    directly when you want a "higher is better" ranking — use
                    ``similarity`` instead.
    """

    text: str
    issuer: str
    form: str
    filing_date: str
    accession: str
    section: str
    source_url: str
    chunk_id: str
    score: float  # raw Chroma distance — lower = more similar

    @property
    def similarity(self) -> float:
        """Return ``1.0 - distance``, clamped to [0.0, 1.0].

        Higher similarity means more relevant.  Useful for threshold-based
        filtering and display, where "higher is better" is more intuitive
        than the raw Chroma distance ("lower is better").
        """
        return max(0.0, min(1.0, 1.0 - self.score))


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

_COLLECTION_NAME_DEFAULT = "financial_filings"


class VectorStore:
    """Chroma-backed vector store.

    Args:
        collection_name: Name of the Chroma collection to use.
                         Defaults to a unique name so each ephemeral instance
                         is fully isolated from others sharing the same
                         in-memory backend (chromadb EphemeralClient is a
                         process-level singleton in v1.x).
                         Pass an explicit name for persistent / named stores.
        persist_path:    If given, use a PersistentClient at this path.
                         If None (default), use an EphemeralClient (in-memory).
        provider_mode:   Passed to get_provider(); "offline" by default.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_path: str | None = None,
        provider_mode: str = "offline",
    ) -> None:
        if persist_path is not None:
            self._client = chromadb.PersistentClient(path=persist_path)
            # Named persistent stores use the given name or the project default.
            effective_name = collection_name or _COLLECTION_NAME_DEFAULT
        else:
            self._client = chromadb.EphemeralClient()
            # Ephemeral stores get a UUID-based name so concurrent test
            # instances don't share state even though EphemeralClient is a
            # process-level singleton in chromadb ≥ 1.x.
            effective_name = collection_name or f"eph_{uuid.uuid4().hex}"

        self._collection = self._client.get_or_create_collection(
            name=effective_name,
            # Use cosine distance so scores are comparable across dimensions.
            metadata={"hnsw:space": "cosine"},
        )
        self._provider = get_provider(provider_mode)

    # ------------------------------------------------------------------
    # add
    # ------------------------------------------------------------------

    def add(self, chunks: list["Chunk"]) -> None:
        """Embed and persist chunks.

        If the chunk list is empty this is a no-op. Embeddings are produced
        via the configured provider so callers don't call the model directly.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self._provider.embed(texts)

        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "issuer": c.issuer,
                "form": c.form,
                "filing_date": c.filing_date,
                "accession": c.accession,
                "section": c.section,
                "source_url": c.source_url,
                "chunk_id": c.chunk_id,
            }
            for c in chunks
        ]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def query(self, text: str, k: int = RETRIEVAL_K) -> list[RetrievedChunk]:
        """Return the top-k most similar chunks for the query text.

        Args:
            text: Query string.
            k:    Number of results to return.  Defaults to RETRIEVAL_K from config.

        Returns:
            List of RetrievedChunk ordered by similarity (closest first).
            Returns an empty list if the collection is empty.
        """
        # Guard: Chroma raises if n_results > collection size.
        count = self._collection.count()
        if count == 0:
            return []

        n = min(k, count)
        query_embedding = self._provider.embed([text])[0]

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        # result is shaped as {key: [[val, val, ...]]} — the outer list is per
        # query; we only ever send one query, so index 0.
        ids: list[str] = result["ids"][0]
        documents: list[str] = result["documents"][0]  # type: ignore[index]
        metadatas: list[dict] = result["metadatas"][0]  # type: ignore[index]
        distances: list[float] = result["distances"][0]  # type: ignore[index]

        retrieved: list[RetrievedChunk] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            retrieved.append(
                RetrievedChunk(
                    text=doc,
                    issuer=meta.get("issuer", ""),
                    form=meta.get("form", ""),
                    filing_date=meta.get("filing_date", ""),
                    accession=meta.get("accession", ""),
                    section=meta.get("section", ""),
                    source_url=meta.get("source_url", ""),
                    chunk_id=meta.get("chunk_id", ""),
                    score=float(dist),
                )
            )
        return retrieved
