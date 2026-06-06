"""
Tests for src/sut/generate.py and src/sut/prompts.py — T5: Generate (F-03/F-04/F-05).

TDD: tests written before implementation.
All tests run offline — no network, no API key required.
Stub providers are injected wherever deterministic output is needed.
"""

from __future__ import annotations



from src.sut.ingest import Chunk
from src.sut.generate import Answer, answer_question
from src.sut.prompts import SYSTEM_PROMPT, build_context_block, build_user_prompt
from src.sut.store import RetrievedChunk, VectorStore


# ---------------------------------------------------------------------------
# Stub provider — implements Provider protocol; caller controls generate output
# ---------------------------------------------------------------------------


class StubProvider:
    """Minimal stub satisfying the Provider protocol for deterministic tests."""

    def __init__(self, generate_response: str = "Stub response [c1].") -> None:
        self._response = generate_response

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Return a trivial unit vector (all same dim as OfflineProvider).
        dim = 384
        val = 1.0 / (dim ** 0.5)
        return [[val] * dim for _ in texts]

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        return self._response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    text: str = "Net sales were $383 billion.",
    chunk_id: str = "acc-0#Item 7#0",
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
        chunk_id=chunk_id,
    )


def _make_store(*texts_and_ids: tuple[str, str]) -> VectorStore:
    """Build an ephemeral VectorStore with the given (text, chunk_id) pairs."""
    store = VectorStore()
    chunks = [
        _make_chunk(text=text, chunk_id=cid, idx=i)
        for i, (text, cid) in enumerate(texts_and_ids)
    ]
    store.add(chunks)
    return store


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT content — boundary rules (F-03 / F-04 / F-05)
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_system_prompt_is_nonempty_string(self) -> None:
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT.strip()) > 0

    def test_answer_only_from_context_rule(self) -> None:
        """Prompt must instruct the model not to use outside knowledge."""
        lower = SYSTEM_PROMPT.lower()
        # At least one of these phrases must appear
        phrases = ["only from", "provided context", "outside knowledge", "only the"]
        assert any(p in lower for p in phrases), (
            f"SYSTEM_PROMPT missing answer-from-context instruction. Got:\n{SYSTEM_PROMPT}"
        )

    def test_citation_instruction(self) -> None:
        """Prompt must instruct the model to cite claims with [cN] markers."""
        assert "[c" in SYSTEM_PROMPT, (
            "SYSTEM_PROMPT missing citation marker instruction ([c1], [c2], …)"
        )

    def test_refusal_instruction(self) -> None:
        """Prompt must instruct the model to refuse when answer is not in context."""
        lower = SYSTEM_PROMPT.lower()
        refusal_phrases = ["not in", "not found", "cannot", "refuse", "not supported", "not present"]
        assert any(p in lower for p in refusal_phrases), (
            f"SYSTEM_PROMPT missing refusal instruction. Got:\n{SYSTEM_PROMPT}"
        )

    def test_no_investment_advice_instruction(self) -> None:
        """Prompt must forbid investment advice / recommendations / price targets."""
        lower = SYSTEM_PROMPT.lower()
        advice_phrases = [
            "investment advice",
            "recommend",
            "price target",
            "do not give advice",
            "no advice",
            "not give",
            "inform only",
            "not provide advice",
        ]
        assert any(p in lower for p in advice_phrases), (
            f"SYSTEM_PROMPT missing no-advice instruction. Got:\n{SYSTEM_PROMPT}"
        )


# ---------------------------------------------------------------------------
# build_context_block
# ---------------------------------------------------------------------------


class TestBuildContextBlock:
    def test_returns_tuple_of_str_and_dict(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Revenue was $383B.",
                issuer="AAPL", form="10-K", filing_date="2024-09-28",
                accession="acc-0", section="Item 7", source_url="https://sec.gov/",
                chunk_id="acc-0#Item 7#0", score=0.1,
            )
        ]
        result = build_context_block(chunks)
        assert isinstance(result, tuple)
        assert len(result) == 2
        block, marker_map = result
        assert isinstance(block, str)
        assert isinstance(marker_map, dict)

    def test_single_chunk_labeled_c1(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Revenue was $383B.",
                issuer="AAPL", form="10-K", filing_date="2024-09-28",
                accession="acc-0", section="Item 7", source_url="https://sec.gov/",
                chunk_id="acc-0#Item 7#0", score=0.1,
            )
        ]
        block, marker_map = build_context_block(chunks)
        assert "[c1]" in block
        assert "c1" in marker_map
        assert marker_map["c1"] == "acc-0#Item 7#0"

    def test_multiple_chunks_labeled_sequentially(self) -> None:
        chunks = [
            RetrievedChunk(
                text=f"Passage {i}.",
                issuer="AAPL", form="10-K", filing_date="2024-09-28",
                accession=f"acc-{i}", section="Item 7", source_url="https://sec.gov/",
                chunk_id=f"acc-{i}#Item 7#0", score=0.1,
            )
            for i in range(3)
        ]
        block, marker_map = build_context_block(chunks)
        assert "[c1]" in block
        assert "[c2]" in block
        assert "[c3]" in block
        assert marker_map["c1"] == "acc-0#Item 7#0"
        assert marker_map["c2"] == "acc-1#Item 7#0"
        assert marker_map["c3"] == "acc-2#Item 7#0"

    def test_marker_map_covers_all_chunks(self) -> None:
        n = 5
        chunks = [
            RetrievedChunk(
                text=f"Passage {i}.",
                issuer="AAPL", form="10-K", filing_date="2024-09-28",
                accession=f"acc-{i}", section="Item 7", source_url="https://sec.gov/",
                chunk_id=f"acc-{i}#Item 7#0", score=0.1,
            )
            for i in range(n)
        ]
        _, marker_map = build_context_block(chunks)
        assert len(marker_map) == n
        for j in range(1, n + 1):
            assert f"c{j}" in marker_map

    def test_block_contains_chunk_text(self) -> None:
        chunks = [
            RetrievedChunk(
                text="Unique revenue figure here.",
                issuer="AAPL", form="10-K", filing_date="2024-09-28",
                accession="acc-0", section="Item 7", source_url="https://sec.gov/",
                chunk_id="acc-0#Item 7#0", score=0.1,
            )
        ]
        block, _ = build_context_block(chunks)
        assert "Unique revenue figure here." in block

    def test_empty_chunks_returns_empty(self) -> None:
        block, marker_map = build_context_block([])
        assert block == "" or block.strip() == ""
        assert marker_map == {}


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_returns_string_containing_question_and_context(self) -> None:
        question = "What was Apple's revenue in FY2024?"
        context_block = "[c1] Revenue was $383B."
        prompt = build_user_prompt(question, context_block)
        assert isinstance(prompt, str)
        assert question in prompt
        assert context_block in prompt


# ---------------------------------------------------------------------------
# answer_question — citation parsing
# ---------------------------------------------------------------------------


class TestAnswerQuestion:
    def _make_store_with_passage(self) -> tuple[VectorStore, str]:
        """Returns (store, chunk_id) for a single chunk."""
        chunk_id = "acc-0#Item 7#0"
        store = _make_store(("Net sales were $383 billion.", chunk_id))
        return store, chunk_id

    def test_returns_answer_instance(self) -> None:
        store, _ = self._make_store_with_passage()
        result = answer_question("What were net sales?", store, provider=StubProvider())
        assert isinstance(result, Answer)

    def test_answer_has_answer_field(self) -> None:
        store, _ = self._make_store_with_passage()
        stub = StubProvider("Net sales were $383B [c1].")
        result = answer_question("What were net sales?", store, provider=stub, k=1)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_answer_has_citations_field(self) -> None:
        store, _ = self._make_store_with_passage()
        stub = StubProvider("Net sales were $383B [c1].")
        result = answer_question("What were net sales?", store, provider=stub, k=1)
        assert isinstance(result.citations, dict)

    def test_answer_has_retrieved_field(self) -> None:
        store, _ = self._make_store_with_passage()
        stub = StubProvider("Net sales were $383B [c1].")
        result = answer_question("What were net sales?", store, provider=stub, k=1)
        assert isinstance(result.retrieved, list)
        assert len(result.retrieved) == 1
        assert isinstance(result.retrieved[0], RetrievedChunk)

    def test_citations_parsed_from_answer_text(self) -> None:
        """[c1] appears in the answer → citations["c1"] maps to the right chunk_id."""
        chunk_id = "acc-0#Item 7#0"
        store = _make_store(("Net sales were $383 billion.", chunk_id))
        stub = StubProvider("Net sales were $383B [c1].")
        result = answer_question("What were net sales?", store, provider=stub, k=1)
        assert "c1" in result.citations
        assert result.citations["c1"] == chunk_id

    def test_unused_markers_excluded_from_citations(self) -> None:
        """Markers not present in the answer text are NOT in citations."""
        chunk_id_0 = "acc-0#Item 7#0"
        chunk_id_1 = "acc-1#Item 7#0"
        store = _make_store(
            ("Net sales were $383 billion.", chunk_id_0),
            ("Operating income was $123 billion.", chunk_id_1),
        )
        # Stub only references [c1], not [c2]
        stub = StubProvider("Net sales were $383B [c1].")
        result = answer_question("What were net sales?", store, provider=stub, k=2)
        assert "c1" in result.citations
        assert "c2" not in result.citations

    def test_multiple_citations_parsed(self) -> None:
        """Multiple markers in the answer → both chunk_ids appear in citations.

        Note: retrieval order is determined by embedding similarity, not insertion
        order. The stub returns identical vectors for all texts, so Chroma's
        ordering is non-deterministic. We only assert that both chunk_ids appear
        in the citation values and that both markers are present.
        """
        chunk_id_0 = "acc-0#Item 7#0"
        chunk_id_1 = "acc-1#Item 7#0"
        store = _make_store(
            ("Net sales were $383 billion.", chunk_id_0),
            ("Operating income was $123 billion.", chunk_id_1),
        )
        stub = StubProvider("Revenue [c1] and income [c2] both rose.")
        result = answer_question("What were the financials?", store, provider=stub, k=2)
        # Both markers must be parsed from the answer
        assert "c1" in result.citations
        assert "c2" in result.citations
        # Both chunk_ids must appear somewhere in the citation values
        cited_ids = set(result.citations.values())
        assert chunk_id_0 in cited_ids
        assert chunk_id_1 in cited_ids

    def test_refusal_passthrough(self) -> None:
        """Refusal phrase from provider is passed through; citations empty, no crash."""
        store, _ = self._make_store_with_passage()
        refusal_text = "The answer is not in the provided sources."
        stub = StubProvider(refusal_text)
        result = answer_question("What is the stock price?", store, provider=stub, k=1)
        assert result.answer == refusal_text
        assert result.citations == {}

    def test_answer_text_matches_provider_output(self) -> None:
        """answer.answer must be exactly the string returned by provider.generate."""
        store, _ = self._make_store_with_passage()
        expected = "The revenue was $383B [c1]."
        stub = StubProvider(expected)
        result = answer_question("Revenue?", store, provider=stub, k=1)
        assert result.answer == expected

    def test_no_crash_on_empty_answer(self) -> None:
        """If provider returns empty string, Answer is still returned without crash."""
        store, _ = self._make_store_with_passage()
        stub = StubProvider("")
        result = answer_question("Revenue?", store, provider=stub, k=1)
        assert isinstance(result, Answer)
        assert result.answer == ""
        assert result.citations == {}

    # ------------------------------------------------------------------
    # unmatched_citations — hallucinated / out-of-range markers (T10 support)
    # ------------------------------------------------------------------

    def test_out_of_range_marker_goes_to_unmatched(self) -> None:
        """Model cites [c9] when only c1..c3 exist → c9 in unmatched_citations, not citations."""
        store = _make_store(
            ("Passage A.", "acc-0#Item 7#0"),
            ("Passage B.", "acc-1#Item 7#0"),
            ("Passage C.", "acc-2#Item 7#0"),
        )
        # Stub cites [c9] which is outside the 3-chunk marker_map
        stub = StubProvider("Revenue was high [c9].")
        result = answer_question("Revenue?", store, provider=stub, k=3)
        assert "c9" not in result.citations
        assert result.unmatched_citations == ["c9"]

    def test_mixed_valid_and_hallucinated_markers(self) -> None:
        """Model cites [c1] (valid) and [c7] (invalid) → citations has c1, unmatched has c7."""
        store = _make_store(
            ("Net sales were $383 billion.", "acc-0#Item 7#0"),
        )
        stub = StubProvider("Net sales [c1] grew significantly [c7].")
        result = answer_question("Revenue?", store, provider=stub, k=1)
        assert "c1" in result.citations
        assert "c7" not in result.citations
        assert result.unmatched_citations == ["c7"]

    def test_no_unmatched_when_all_markers_valid(self) -> None:
        """All cited markers map to retrieved chunks → unmatched_citations is empty."""
        store = _make_store(
            ("Net sales were $383 billion.", "acc-0#Item 7#0"),
        )
        stub = StubProvider("Net sales were $383B [c1].")
        result = answer_question("Revenue?", store, provider=stub, k=1)
        assert "c1" in result.citations
        assert result.unmatched_citations == []

    def test_unmatched_citations_deduplicated_in_order(self) -> None:
        """Repeated hallucinated marker appears only once in unmatched_citations."""
        store = _make_store(
            ("Passage A.", "acc-0#Item 7#0"),
        )
        # [c9] appears twice in the answer — should appear once in unmatched_citations
        stub = StubProvider("Claim one [c9]. Claim two [c9].")
        result = answer_question("Revenue?", store, provider=stub, k=1)
        assert result.unmatched_citations == ["c9"]


# ---------------------------------------------------------------------------
# End-to-end offline — real OfflineProvider (no key, no network)
# ---------------------------------------------------------------------------


class TestAnswerQuestionOfflineE2E:
    def test_end_to_end_returns_answer_without_key(self) -> None:
        """answer_question with the real offline provider completes without error."""
        store = VectorStore()
        chunk = _make_chunk(
            text="Apple reported net sales of 383 billion dollars in fiscal year 2024.",
            chunk_id="acc-e2e#Item 7#0",
        )
        store.add([chunk])

        # No provider argument → get_provider() → OfflineProvider (no key needed)
        result = answer_question("What were Apple's net sales?", store, k=1)
        assert isinstance(result, Answer)
        assert isinstance(result.answer, str)
        assert isinstance(result.citations, dict)
        assert isinstance(result.retrieved, list)
