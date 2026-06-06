"""Tests for T10 — programmatic metrics (E-06).

TDD: tests were written before the implementation.

Coverage:
  - Unit tests for each of the 7 metrics (pass + fail cases).
  - Integration tests against run_pass.jsonl and run_fail.jsonl fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.golden import Bucket, GoldenItem
from src.eval.runner import RetrievedRef, RunRecord
from src.eval.metrics.programmatic import (
    MetricResult,
    aggregate_metric,
    citation_validity,
    context_precision,
    context_recall,
    entity_disambiguation,
    metric_rate,
    negative_rejection,
    numerical_exactness,
    score_programmatic,
    temporal_correctness,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATASETS = Path(__file__).parent.parent / "datasets"
GOLDEN_SET = DATASETS / "golden_set.jsonl"
RUN_PASS = DATASETS / "fixtures" / "run_pass.jsonl"
RUN_FAIL = DATASETS / "fixtures" / "run_fail.jsonl"


def _make_ref(
    chunk_id: str,
    text: str,
    issuer: str = "Acme Corp",
    filing_date: str = "2024-01-01",
    accession: str = "ACM-10K-2024",
    section: str = "item7",
) -> RetrievedRef:
    return RetrievedRef(
        chunk_id=chunk_id,
        text=text,
        similarity=0.9,
        issuer=issuer,
        form="10-K",
        filing_date=filing_date,
        accession=accession,
        section=section,
        source_url="https://example.invalid/test",
    )


def _make_record(
    id: str,
    bucket: str,
    answer: str,
    retrieved: list[RetrievedRef],
    citations: dict[str, str],
    unmatched_citations: list[str] | None = None,
    question: str = "What was revenue?",
) -> RunRecord:
    return RunRecord(
        id=id,
        bucket=bucket,
        question=question,
        answer=answer,
        retrieved=retrieved,
        citations=citations,
        unmatched_citations=unmatched_citations or [],
        latency_ms=0,
        mode="replay",
    )


def _make_golden(
    id: str,
    bucket: Bucket,
    expected_sources: list[str] | None = None,
    numeric_answers: list[str] | None = None,
    must_refuse: bool = False,
    reference_answer: str = "Reference answer.",
    question: str = "What was revenue?",
) -> GoldenItem:
    return GoldenItem(
        id=id,
        bucket=bucket,
        question=question,
        reference_answer=reference_answer,
        expected_sources=expected_sources or [],
        numeric_answers=numeric_answers or [],
        must_refuse=must_refuse,
        injection=None,
        advice_boundary=False,
    )


# ---------------------------------------------------------------------------
# 1. numerical_exactness
# ---------------------------------------------------------------------------


class TestNumericalExactness:
    def test_pass_exact_match(self) -> None:
        ref = _make_ref("chunk#0", "Revenue was $58,420 million.")
        rec = _make_record(
            "x",
            "factual_lookup",
            "Total revenue was $58,420 million for fiscal year 2024 [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.factual_lookup, numeric_answers=["58420"])
        result = numerical_exactness(rec, g)
        assert result.applicable is True
        assert result.score == 1.0
        assert result.passed is True

    def test_pass_dollar_sign_and_commas(self) -> None:
        """'$383,285 million' in answer should match numeric_answer '383285'."""
        ref = _make_ref("chunk#0", "Revenue was $383,285 million.")
        rec = _make_record(
            "x",
            "factual_lookup",
            "Total revenue was $383,285 million.",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.factual_lookup, numeric_answers=["383285"])
        result = numerical_exactness(rec, g)
        assert result.applicable is True
        assert result.score == 1.0
        assert result.passed is True

    def test_fail_wrong_number(self) -> None:
        ref = _make_ref("chunk#0", "Revenue was $99,999 million.")
        rec = _make_record(
            "x",
            "factual_lookup",
            "Total revenue was $99,999 million.",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.factual_lookup, numeric_answers=["58420"])
        result = numerical_exactness(rec, g)
        assert result.applicable is True
        assert result.score == 0.0
        assert result.passed is False

    def test_partial_score(self) -> None:
        ref = _make_ref("chunk#0", "Revenue was $58,420 million, profit $12,150 million.")
        rec = _make_record(
            "x",
            "multi_source",
            "Revenue $58,420 million. Profit unknown.",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.multi_source, numeric_answers=["58420", "12150"])
        result = numerical_exactness(rec, g)
        assert result.applicable is True
        assert result.score == pytest.approx(0.5)
        assert result.passed is False

    def test_not_applicable_when_no_numeric_answers(self) -> None:
        ref = _make_ref("chunk#0", "The company is listed in New York.")
        rec = _make_record(
            "x",
            "entity",
            "The company is listed in New York.",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.entity, numeric_answers=[])
        result = numerical_exactness(rec, g)
        assert result.applicable is False

    def test_decimal_numeric_answer(self) -> None:
        ref = _make_ref("chunk#0", "Debt-to-equity ratio was 1.8.")
        rec = _make_record(
            "x",
            "multi_source",
            "The debt-to-equity ratio was 1.8.",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.multi_source, numeric_answers=["1.8"])
        result = numerical_exactness(rec, g)
        assert result.applicable is True
        assert result.score == 1.0
        assert result.passed is True

    def test_verbal_scale_billion_matches_million_golden(self) -> None:
        """Golden '1200' (expressed in millions) matches answer '$1.2 billion'."""
        ref = _make_ref("chunk#0", "Revenue was $1.2 billion.")
        rec = _make_record(
            "x",
            "factual_lookup",
            "Total revenue was $1.2 billion [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.factual_lookup, numeric_answers=["1200"])
        result = numerical_exactness(rec, g)
        assert result.applicable is True
        assert result.score == 1.0
        assert result.passed is True

    def test_percentage_value_below_floor_matches(self) -> None:
        """Golden '7.8' matches answer '7.8%' (no >= 100 floor for target matching)."""
        ref = _make_ref("chunk#0", "Revenue grew approximately 7.8%.")
        rec = _make_record(
            "x",
            "factual_lookup",
            "Revenue grew approximately 7.8% year over year [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.factual_lookup, numeric_answers=["7.8"])
        result = numerical_exactness(rec, g)
        assert result.applicable is True
        assert result.score == 1.0
        assert result.passed is True


# ---------------------------------------------------------------------------
# 2. citation_validity
# ---------------------------------------------------------------------------


class TestCitationValidity:
    def test_pass_number_overlap(self) -> None:
        """Citation supports claim when the cited chunk contains the answer's financial number."""
        chunk_text = "Revenue for FY2024 was $58,420 million, up from last year."
        ref = _make_ref("chunk#0", chunk_text)
        rec = _make_record(
            "x",
            "factual_lookup",
            "Revenue was $58,420 million [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.factual_lookup)
        result = citation_validity(rec, g)
        assert result.applicable is True
        assert result.score == 1.0
        assert result.passed is True

    def test_fail_number_mismatch(self) -> None:
        """Citation fails when chunk has different financial numbers (not matching answer's)."""
        chunk_text = "Capital expenditures for FY2024 were $3,200 million for equipment upgrades."
        ref = _make_ref("chunk#0", chunk_text)
        rec = _make_record(
            "x",
            "factual_lookup",
            "Gross profit was $12,150 million [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.factual_lookup)
        result = citation_validity(rec, g)
        assert result.applicable is True
        assert result.score == 0.0
        assert result.passed is False

    def test_fail_no_number_in_chunk_low_jaccard(self) -> None:
        """Citation fails when chunk has no financial numbers and low text overlap."""
        chunk_text = "The company sells passenger vehicles and commercial vans globally."
        ref = _make_ref("chunk#0", chunk_text)
        rec = _make_record(
            "x",
            "factual_lookup",
            "Revenue was $58,420 million [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.factual_lookup)
        result = citation_validity(rec, g)
        assert result.applicable is True
        assert result.passed is False

    def test_pass_no_numbers_jaccard_overlap(self) -> None:
        """When answer has no financial numbers, jaccard ≥ 0.12 is sufficient."""
        chunk_text = "Cascade Semiconductor designs advanced logic chips for data center applications and AI workloads."
        ref = _make_ref("chunk#0", chunk_text)
        rec = _make_record(
            "x",
            "entity",
            "Cascade Semiconductor designs advanced logic chips for data center applications.",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.entity)
        result = citation_validity(rec, g)
        assert result.applicable is True
        assert result.passed is True

    def test_fail_unmatched_citation_counts_as_invalid(self) -> None:
        """Any marker in unmatched_citations is an over-citation and makes score < 1.0."""
        chunk_text = "Revenue for FY2024 was $58,420 million."
        ref = _make_ref("chunk#0", chunk_text)
        rec = _make_record(
            "x",
            "factual_lookup",
            "Revenue was $58,420 million [c1] [c2].",
            [ref],
            {"c1": "chunk#0"},
            unmatched_citations=["c2"],
        )
        g = _make_golden("x", Bucket.factual_lookup)
        result = citation_validity(rec, g)
        assert result.applicable is True
        # c1 valid (1/2), c2 unmatched (0/2) → score = 0.5
        assert result.score == pytest.approx(0.5)
        assert result.passed is False

    def test_fail_over_citation_bypass_numeric_sentence(self) -> None:
        """Production over-citation bypass: a numeric sentence cited to a number-free
        but topically similar chunk must be INVALID (no Jaccard escape)."""
        chunk_text = (
            "Northwind Motors vehicle sales grew on strong demand; "
            "revenue momentum continuing."
        )
        ref = _make_ref("chunk#0", chunk_text)
        rec = _make_record(
            "x",
            "factual_lookup",
            "Revenue was $58,420 million driven by strong vehicle sales at Northwind Motors [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.factual_lookup)
        result = citation_validity(rec, g)
        assert result.applicable is True
        assert result.score < 1.0
        assert result.passed is False

    def test_not_applicable_when_no_citations(self) -> None:
        """When citations dict is empty (and no unmatched), metric is N/A."""
        rec = _make_record(
            "x",
            "negative",
            "The sources do not contain this information.",
            [],
            {},
        )
        g = _make_golden("x", Bucket.negative)
        result = citation_validity(rec, g)
        assert result.applicable is False

    def test_pass_contextual_chunk_no_numbers_high_jaccard(self) -> None:
        """Chunk with no financial numbers passes via jaccard when answer has numbers but chunk provides context."""
        chunk_text = (
            "Risk factors include supply chain concentration, customer concentration, "
            "semiconductor cyclicality, and geopolitical risks faced by the company."
        )
        ref = _make_ref("chunk#0", chunk_text)
        rec = _make_record(
            "x",
            "long_context",
            (
                "The company faces supply chain concentration, customer concentration, "
                "and semiconductor cyclicality risks [c1]. Revenue was $21,300 million [c2]."
            ),
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("x", Bucket.long_context)
        result = citation_validity(rec, g)
        assert result.applicable is True
        assert result.passed is True


# ---------------------------------------------------------------------------
# 3. negative_rejection
# ---------------------------------------------------------------------------


class TestNegativeRejection:
    def test_pass_has_cue_and_no_citations(self) -> None:
        rec = _make_record(
            "neg-x",
            "negative",
            "The provided sources do not contain any information about that topic.",
            [],
            {},
        )
        g = _make_golden("neg-x", Bucket.negative, must_refuse=True)
        result = negative_rejection(rec, g)
        assert result.applicable is True
        assert result.score == 1.0
        assert result.passed is True

    def test_fail_has_citation(self) -> None:
        """Even with a refusal cue, if citations exist it should fail."""
        ref = _make_ref("chunk#0", "Revenue text.")
        rec = _make_record(
            "neg-x",
            "negative",
            "The sources do not contain this but here is what I found [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("neg-x", Bucket.negative, must_refuse=True)
        result = negative_rejection(rec, g)
        assert result.applicable is True
        assert result.score == 0.0
        assert result.passed is False

    def test_fail_no_cue_with_fabrication(self) -> None:
        """Answer that fabricates information with no refusal cue must fail."""
        ref = _make_ref("chunk#0", "Revenue $58,420 million.")
        rec = _make_record(
            "neg-x",
            "negative",
            "Submarine manufacturing revenue was $1,250 million [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("neg-x", Bucket.negative, must_refuse=True)
        result = negative_rejection(rec, g)
        assert result.applicable is True
        assert result.score == 0.0
        assert result.passed is False

    def test_not_applicable_when_must_refuse_false(self) -> None:
        ref = _make_ref("chunk#0", "Revenue $58,420 million.")
        rec = _make_record(
            "fact-x",
            "factual_lookup",
            "Revenue was $58,420 million [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("fact-x", Bucket.factual_lookup, must_refuse=False)
        result = negative_rejection(rec, g)
        assert result.applicable is False

    def test_pass_cannot_find_cue(self) -> None:
        rec = _make_record(
            "neg-x",
            "negative",
            "I cannot find any reference to this in the provided documents.",
            [],
            {},
        )
        g = _make_golden("neg-x", Bucket.negative, must_refuse=True)
        result = negative_rejection(rec, g)
        assert result.passed is True

    def test_pass_unable_to_find_cue(self) -> None:
        rec = _make_record(
            "neg-x",
            "negative",
            "Unable to find this information in the provided sources.",
            [],
            {},
        )
        g = _make_golden("neg-x", Bucket.negative, must_refuse=True)
        result = negative_rejection(rec, g)
        assert result.passed is True

    def test_pass_do_not_disclose_cue(self) -> None:
        """Broadened cue set: 'do not disclose' with empty citations passes."""
        rec = _make_record(
            "neg-x",
            "negative",
            "The provided filings do not disclose this.",
            [],
            {},
        )
        g = _make_golden("neg-x", Bucket.negative, must_refuse=True)
        result = negative_rejection(rec, g)
        assert result.applicable is True
        assert result.passed is True


# ---------------------------------------------------------------------------
# 4. temporal_correctness
# ---------------------------------------------------------------------------


class TestTemporalCorrectness:
    def test_pass_cites_latest_filing(self) -> None:
        ref_old = _make_ref("acc-old#item7#0", "Old revenue $50,000M.", filing_date="2023-01-01", accession="acc-old")
        ref_new = _make_ref("acc-new#item7#0", "New revenue $60,000M.", filing_date="2024-01-01", accession="acc-new")
        rec = _make_record(
            "temp-x",
            "temporal",
            "Revenue was $60,000M [c1].",
            [ref_new, ref_old],
            {"c1": "acc-new#item7#0"},
        )
        g = _make_golden("temp-x", Bucket.temporal)
        result = temporal_correctness(rec, g)
        assert result.applicable is True
        assert result.score == 1.0
        assert result.passed is True

    def test_fail_cites_superseded_filing(self) -> None:
        ref_old = _make_ref("acc-old#item7#0", "Old revenue $50,000M.", filing_date="2023-01-01", accession="acc-old")
        ref_new = _make_ref("acc-new#item7#0", "New revenue $60,000M.", filing_date="2024-01-01", accession="acc-new")
        rec = _make_record(
            "temp-x",
            "temporal",
            "Revenue was $50,000M [c1].",
            [ref_new, ref_old],  # both retrieved; new is latest
            {"c1": "acc-old#item7#0"},  # but cites OLD
        )
        g = _make_golden("temp-x", Bucket.temporal)
        result = temporal_correctness(rec, g)
        assert result.applicable is True
        assert result.score == 0.0
        assert result.passed is False

    def test_not_applicable_for_non_temporal_bucket(self) -> None:
        ref = _make_ref("chunk#0", "Revenue $58,420M.", filing_date="2024-01-01")
        rec = _make_record(
            "fact-x",
            "factual_lookup",
            "Revenue $58,420M [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("fact-x", Bucket.factual_lookup)
        result = temporal_correctness(rec, g)
        assert result.applicable is False

    def test_pass_only_one_filing_in_retrieved(self) -> None:
        """When only one filing is retrieved, citing it is trivially correct."""
        ref = _make_ref("chunk#0", "Revenue $58,420M.", filing_date="2024-01-01")
        rec = _make_record(
            "temp-x",
            "temporal",
            "Revenue $58,420M [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("temp-x", Bucket.temporal)
        result = temporal_correctness(rec, g)
        assert result.applicable is True
        assert result.passed is True


# ---------------------------------------------------------------------------
# 5. entity_disambiguation
# ---------------------------------------------------------------------------


class TestEntityDisambiguation:
    def test_pass_correct_issuer_in_answer_and_citations(self) -> None:
        ref = _make_ref(
            "NAF-10K-2024#item7#0",
            "Northwind Auto Finance LLC reported $3,210M revenue.",
            issuer="Northwind Auto Finance LLC",
            accession="NAF-10K-2024",
        )
        rec = _make_record(
            "ent-x",
            "entity",
            "Northwind Auto Finance LLC reported revenue of $3,210 million [c1].",
            [ref],
            {"c1": "NAF-10K-2024#item7#0"},
        )
        g = _make_golden(
            "ent-x",
            Bucket.entity,
            expected_sources=["NAF-10K-2024#item7"],
        )
        result = entity_disambiguation(rec, g)
        assert result.applicable is True
        assert result.score == 1.0
        assert result.passed is True

    def test_fail_wrong_issuer_cited(self) -> None:
        """If cited chunks belong to a different issuer, disambiguation fails."""
        ref_correct = _make_ref(
            "NAF-10K-2024#item7#0",
            "Northwind Auto Finance LLC reported $3,210M revenue.",
            issuer="Northwind Auto Finance LLC",
            accession="NAF-10K-2024",
        )
        ref_wrong = _make_ref(
            "NWM-10K-2024#item7#0",
            "Northwind Motors Inc. reported $58,420M revenue.",
            issuer="Northwind Motors Inc.",
            accession="NWM-10K-2024",
        )
        rec = _make_record(
            "ent-x",
            "entity",
            "Northwind Auto Finance LLC reported revenue [c1].",
            [ref_correct, ref_wrong],
            {"c1": "NWM-10K-2024#item7#0"},  # cites wrong issuer
        )
        g = _make_golden(
            "ent-x",
            Bucket.entity,
            expected_sources=["NAF-10K-2024#item7"],
        )
        result = entity_disambiguation(rec, g)
        assert result.applicable is True
        assert result.score == 0.0
        assert result.passed is False

    def test_fail_issuer_not_mentioned_in_answer(self) -> None:
        """Correct issuer must appear in the answer text."""
        ref = _make_ref(
            "NAF-10K-2024#item7#0",
            "Revenue was $3,210M.",
            issuer="Northwind Auto Finance LLC",
            accession="NAF-10K-2024",
        )
        rec = _make_record(
            "ent-x",
            "entity",
            "Revenue was $3,210 million [c1].",  # no issuer name
            [ref],
            {"c1": "NAF-10K-2024#item7#0"},
        )
        g = _make_golden(
            "ent-x",
            Bucket.entity,
            expected_sources=["NAF-10K-2024#item7"],
        )
        result = entity_disambiguation(rec, g)
        assert result.applicable is True
        assert result.passed is False

    def test_not_applicable_for_non_entity_bucket(self) -> None:
        ref = _make_ref("chunk#0", "Revenue $58,420M.")
        rec = _make_record(
            "fact-x",
            "factual_lookup",
            "Revenue $58,420M [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("fact-x", Bucket.factual_lookup, expected_sources=["chunk"])
        result = entity_disambiguation(rec, g)
        assert result.applicable is False


# ---------------------------------------------------------------------------
# 6. context_recall
# ---------------------------------------------------------------------------


class TestContextRecall:
    def test_pass_all_expected_sources_retrieved(self) -> None:
        ref = _make_ref("NWM-10K-2024#item7#0", "Revenue text.")
        rec = _make_record(
            "fact-x",
            "factual_lookup",
            "Revenue was $58,420M [c1].",
            [ref],
            {"c1": "NWM-10K-2024#item7#0"},
        )
        g = _make_golden(
            "fact-x",
            Bucket.factual_lookup,
            expected_sources=["NWM-10K-2024#item7"],
        )
        result = context_recall(rec, g)
        assert result.applicable is True
        assert result.score == pytest.approx(1.0)
        assert result.passed is True

    def test_pass_partial_recall_above_threshold(self) -> None:
        """9/10 expected sources retrieved → recall = 0.9 → passes gate."""
        refs = [_make_ref(f"src-{i}#item7#0", "Text.") for i in range(9)]
        rec = _make_record(
            "long-x",
            "long_context",
            "Summary [c1].",
            refs,
            {"c1": "src-0#item7#0"},
        )
        g = _make_golden(
            "long-x",
            Bucket.long_context,
            expected_sources=[f"src-{i}#item7" for i in range(10)],
        )
        result = context_recall(rec, g)
        assert result.applicable is True
        assert result.score == pytest.approx(0.9)
        assert result.passed is True

    def test_fail_low_recall(self) -> None:
        refs = [_make_ref("src-0#item7#0", "Text.")]
        rec = _make_record(
            "long-x",
            "long_context",
            "Summary [c1].",
            refs,
            {"c1": "src-0#item7#0"},
        )
        g = _make_golden(
            "long-x",
            Bucket.long_context,
            expected_sources=["src-0#item7", "src-1#item7", "src-2#item7"],
        )
        result = context_recall(rec, g)
        assert result.applicable is True
        assert result.score == pytest.approx(1 / 3)
        assert result.passed is False

    def test_not_applicable_when_no_expected_sources(self) -> None:
        rec = _make_record(
            "neg-x",
            "negative",
            "Not in sources.",
            [],
            {},
        )
        g = _make_golden("neg-x", Bucket.negative, expected_sources=[])
        result = context_recall(rec, g)
        assert result.applicable is False

    def test_chunk_id_prefix_matching(self) -> None:
        """expected_source 'NWM-10K-2024#item7' matches chunk_id 'NWM-10K-2024#item7#0'."""
        ref = _make_ref("NWM-10K-2024#item7#0", "Revenue text.")
        rec = _make_record(
            "fact-x",
            "factual_lookup",
            "Revenue [c1].",
            [ref],
            {"c1": "NWM-10K-2024#item7#0"},
        )
        g = _make_golden(
            "fact-x",
            Bucket.factual_lookup,
            expected_sources=["NWM-10K-2024#item7"],
        )
        result = context_recall(rec, g)
        assert result.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. context_precision
# ---------------------------------------------------------------------------


class TestContextPrecision:
    def test_pass_all_retrieved_relevant(self) -> None:
        ref = _make_ref("NWM-10K-2024#item7#0", "Revenue text.")
        rec = _make_record(
            "fact-x",
            "factual_lookup",
            "Revenue [c1].",
            [ref],
            {"c1": "NWM-10K-2024#item7#0"},
        )
        g = _make_golden(
            "fact-x",
            Bucket.factual_lookup,
            expected_sources=["NWM-10K-2024#item7"],
        )
        result = context_precision(rec, g)
        assert result.applicable is True
        assert result.score == pytest.approx(1.0)
        assert result.passed is True

    def test_pass_mostly_relevant_above_threshold(self) -> None:
        refs = [_make_ref(f"NWM-10K-2024#item7#{i}", "Revenue text.") for i in range(9)]
        refs.append(_make_ref("NOISE#item1#0", "Irrelevant text."))
        rec = _make_record(
            "fact-x",
            "factual_lookup",
            "Revenue [c1].",
            refs,
            {"c1": "NWM-10K-2024#item7#0"},
        )
        g = _make_golden(
            "fact-x",
            Bucket.factual_lookup,
            expected_sources=["NWM-10K-2024#item7"],
        )
        result = context_precision(rec, g)
        assert result.applicable is True
        assert result.score == pytest.approx(0.9)
        assert result.passed is True

    def test_fail_low_precision(self) -> None:
        refs = [_make_ref("NWM-10K-2024#item7#0", "Revenue text.")]
        refs += [_make_ref(f"NOISE#{i}#0", "Irrelevant.") for i in range(9)]
        rec = _make_record(
            "fact-x",
            "factual_lookup",
            "Revenue [c1].",
            refs,
            {"c1": "NWM-10K-2024#item7#0"},
        )
        g = _make_golden(
            "fact-x",
            Bucket.factual_lookup,
            expected_sources=["NWM-10K-2024#item7"],
        )
        result = context_precision(rec, g)
        assert result.applicable is True
        assert result.score == pytest.approx(0.1)
        assert result.passed is False

    def test_not_applicable_when_no_expected_sources(self) -> None:
        rec = _make_record(
            "neg-x",
            "negative",
            "Not in sources.",
            [],
            {},
        )
        g = _make_golden("neg-x", Bucket.negative, expected_sources=[])
        result = context_precision(rec, g)
        assert result.applicable is False


# ---------------------------------------------------------------------------
# score_programmatic + aggregation helpers
# ---------------------------------------------------------------------------


class TestScoreProgrammatic:
    def test_raises_if_record_has_no_golden(self) -> None:
        ref = _make_ref("chunk#0", "Revenue text.")
        rec = _make_record(
            "orphan-id",
            "factual_lookup",
            "Revenue [c1].",
            [ref],
            {"c1": "chunk#0"},
        )
        g = _make_golden("different-id", Bucket.factual_lookup)
        with pytest.raises(KeyError):
            score_programmatic([rec], [g])

    def test_returns_metric_result_instances(self) -> None:
        ref = _make_ref("NWM-10K-2024#item7#0", "Revenue was $58,420 million.")
        rec = _make_record(
            "fact-x",
            "factual_lookup",
            "Revenue $58,420 million [c1].",
            [ref],
            {"c1": "NWM-10K-2024#item7#0"},
        )
        g = _make_golden(
            "fact-x",
            Bucket.factual_lookup,
            expected_sources=["NWM-10K-2024#item7"],
            numeric_answers=["58420"],
        )
        results = score_programmatic([rec], [g])
        assert all(isinstance(r, MetricResult) for r in results)
        assert len(results) == 7  # one per metric

    def test_aggregate_metric_mean_of_applicable(self) -> None:
        results = [
            MetricResult("m", "a", applicable=True, score=0.8, passed=False, detail=""),
            MetricResult("m", "b", applicable=True, score=1.0, passed=True, detail=""),
            MetricResult("m", "c", applicable=False, score=0.0, passed=False, detail=""),
        ]
        avg = aggregate_metric(results, "m")
        assert avg == pytest.approx(0.9)

    def test_aggregate_metric_none_when_none_applicable(self) -> None:
        results = [
            MetricResult("m", "a", applicable=False, score=0.0, passed=False, detail=""),
        ]
        assert aggregate_metric(results, "m") is None

    def test_metric_rate(self) -> None:
        results = [
            MetricResult("m", "a", applicable=True, score=1.0, passed=True, detail=""),
            MetricResult("m", "b", applicable=True, score=0.0, passed=False, detail=""),
            MetricResult("m", "c", applicable=False, score=0.0, passed=False, detail=""),
        ]
        rate = metric_rate(results, "m")
        assert rate == pytest.approx(0.5)

    def test_metric_rate_none_when_none_applicable(self) -> None:
        results = [
            MetricResult("m", "a", applicable=False, score=0.0, passed=False, detail=""),
        ]
        assert metric_rate(results, "m") is None


# ---------------------------------------------------------------------------
# INTEGRATION: run_pass fixture — all programmatic gates must clear
# ---------------------------------------------------------------------------


def _load_goldens(path: Path) -> dict[str, GoldenItem]:
    from src.eval.golden import load_goldens
    return {g.id: g for g in load_goldens(path)}


def _load_records(path: Path) -> list[RunRecord]:
    from src.eval.runner import load_replay
    return load_replay(path)


class TestIntegrationRunPass:
    """run_pass.jsonl is designed to PASS all programmatic gates."""

    @pytest.fixture(scope="class")
    def results(self) -> list[MetricResult]:
        goldens = list(_load_goldens(GOLDEN_SET).values())
        records = _load_records(RUN_PASS)
        return score_programmatic(records, goldens)

    def test_citation_validity_rate_is_1(self, results: list[MetricResult]) -> None:
        rate = metric_rate(results, "citation_validity")
        assert rate is not None
        assert rate == pytest.approx(1.0), f"citation_validity rate={rate}"

    def test_negative_rejection_rate_is_1(self, results: list[MetricResult]) -> None:
        rate = metric_rate(results, "negative_rejection")
        assert rate is not None
        assert rate == pytest.approx(1.0), f"negative_rejection rate={rate}"

    def test_numerical_exactness_rate_is_1(self, results: list[MetricResult]) -> None:
        rate = metric_rate(results, "numerical_exactness")
        assert rate is not None
        assert rate == pytest.approx(1.0), f"numerical_exactness rate={rate}"

    def test_temporal_correctness_rate_is_1(self, results: list[MetricResult]) -> None:
        rate = metric_rate(results, "temporal_correctness")
        assert rate is not None
        assert rate == pytest.approx(1.0), f"temporal_correctness rate={rate}"

    def test_entity_disambiguation_rate_is_1(self, results: list[MetricResult]) -> None:
        rate = metric_rate(results, "entity_disambiguation")
        assert rate is not None
        assert rate == pytest.approx(1.0), f"entity_disambiguation rate={rate}"

    def test_context_recall_mean_ge_090(self, results: list[MetricResult]) -> None:
        agg = aggregate_metric(results, "context_recall")
        assert agg is not None
        assert agg >= 0.90, f"context_recall mean={agg}"

    def test_context_precision_mean_ge_085(self, results: list[MetricResult]) -> None:
        agg = aggregate_metric(results, "context_precision")
        assert agg is not None
        assert agg >= 0.85, f"context_precision mean={agg}"


# ---------------------------------------------------------------------------
# INTEGRATION: run_fail fixture — planted defects must trip the gates
# ---------------------------------------------------------------------------


class TestIntegrationRunFail:
    """run_fail.jsonl has 3 broken citations, 1 non-refusing negative, 1 superseded temporal."""

    @pytest.fixture(scope="class")
    def results(self) -> list[MetricResult]:
        goldens = list(_load_goldens(GOLDEN_SET).values())
        records = _load_records(RUN_FAIL)
        return score_programmatic(records, goldens)

    def test_citation_validity_aggregate_below_095(self, results: list[MetricResult]) -> None:
        agg = aggregate_metric(results, "citation_validity")
        assert agg is not None
        assert agg < 0.95, f"Expected citation_validity agg < 0.95, got {agg}"

    def test_negative_rejection_rate_below_095(self, results: list[MetricResult]) -> None:
        rate = metric_rate(results, "negative_rejection")
        assert rate is not None
        assert rate < 0.95, f"Expected negative_rejection rate < 0.95, got {rate}"

    def test_temporal_correctness_rate_below_098(self, results: list[MetricResult]) -> None:
        rate = metric_rate(results, "temporal_correctness")
        assert rate is not None
        assert rate < 0.98, f"Expected temporal_correctness rate < 0.98, got {rate}"
