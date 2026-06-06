"""
Tests for src/sut/api.py — T6: FastAPI query endpoint (F-02/F-03).

TDD: tests written before implementation.
All tests run offline — no network, no API key required.
The ephemeral VectorStore is populated via direct Chunk.add() and injected via
FastAPI dependency_overrides so real ingest or live providers are never used.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.sut.api import app, get_store
from src.sut.ingest import Chunk
from src.sut.store import VectorStore

# ---------------------------------------------------------------------------
# Fixture: ephemeral store with a small corpus
# ---------------------------------------------------------------------------

_FIXTURE_CHUNKS = [
    Chunk(
        text="Northwind Financials reported total revenue of $42.3 billion for FY2024.",
        issuer="Northwind Financials",
        form="10-K",
        filing_date="2024-12-31",
        accession="nwf-10k-2024",
        section="Item 7",
        source_url="https://sec.gov/nwf/10k/2024",
        chunk_id="nwf-10k-2024#Item 7#0",
    ),
    Chunk(
        text="Net income for FY2024 was $6.1 billion, or $4.87 per diluted share.",
        issuer="Northwind Financials",
        form="10-K",
        filing_date="2024-12-31",
        accession="nwf-10k-2024",
        section="Item 7",
        source_url="https://sec.gov/nwf/10k/2024",
        chunk_id="nwf-10k-2024#Item 7#1",
    ),
    Chunk(
        text="Operating income increased to $8.2 billion in FY2024 with a 19.4% margin.",
        issuer="Northwind Financials",
        form="10-K",
        filing_date="2024-12-31",
        accession="nwf-10k-2024",
        section="Item 7",
        source_url="https://sec.gov/nwf/10k/2024",
        chunk_id="nwf-10k-2024#Item 7#2",
    ),
]


@pytest.fixture(scope="module")
def client() -> TestClient:
    """TestClient with dependency override pointing to an ephemeral ingested store."""
    store = VectorStore()
    store.add(_FIXTURE_CHUNKS)

    app.dependency_overrides[get_store] = lambda: store
    tc = TestClient(app)
    yield tc  # type: ignore[misc]
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /query — happy path
# ---------------------------------------------------------------------------


class TestQueryHappyPath:
    def test_returns_200(self, client: TestClient) -> None:
        resp = client.post("/query", json={"question": "What was total revenue for FY2024?"})
        assert resp.status_code == 200

    def test_response_has_answer(self, client: TestClient) -> None:
        resp = client.post("/query", json={"question": "What was total revenue for FY2024?"})
        data = resp.json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_response_has_citations(self, client: TestClient) -> None:
        resp = client.post("/query", json={"question": "What was total revenue for FY2024?"})
        data = resp.json()
        assert isinstance(data["citations"], dict)

    def test_response_has_unmatched_citations(self, client: TestClient) -> None:
        resp = client.post("/query", json={"question": "What was total revenue for FY2024?"})
        data = resp.json()
        assert isinstance(data["unmatched_citations"], list)

    def test_response_has_retrieved_list(self, client: TestClient) -> None:
        resp = client.post("/query", json={"question": "What was total revenue for FY2024?"})
        data = resp.json()
        assert isinstance(data["retrieved"], list)
        assert len(data["retrieved"]) > 0

    def test_retrieved_items_have_all_9_fields(self, client: TestClient) -> None:
        """Each retrieved item must expose the full run-contract shape."""
        resp = client.post("/query", json={"question": "What was total revenue for FY2024?"})
        data = resp.json()
        required_fields = {
            "chunk_id", "text", "similarity", "issuer", "form",
            "filing_date", "accession", "section", "source_url",
        }
        for item in data["retrieved"]:
            assert required_fields <= set(item.keys()), (
                f"Retrieved item missing fields: {required_fields - set(item.keys())}"
            )

    def test_response_has_latency_ms(self, client: TestClient) -> None:
        resp = client.post("/query", json={"question": "What was total revenue for FY2024?"})
        data = resp.json()
        assert isinstance(data["latency_ms"], int)
        assert data["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# POST /query — validation errors
# ---------------------------------------------------------------------------


class TestQueryValidation:
    def test_empty_question_returns_422(self, client: TestClient) -> None:
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_missing_question_field_returns_422(self, client: TestClient) -> None:
        resp = client.post("/query", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /query — k parameter
# ---------------------------------------------------------------------------


class TestQueryKParam:
    def test_k_limits_retrieved_length(self, client: TestClient) -> None:
        resp = client.post(
            "/query",
            json={"question": "What was total revenue for FY2024?", "k": 2},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["retrieved"]) <= 2

    def test_k_1_returns_at_most_1_chunk(self, client: TestClient) -> None:
        resp = client.post(
            "/query",
            json={"question": "What was net income?", "k": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["retrieved"]) <= 1
