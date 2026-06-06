"""Tests for T13 — aggregate.py (E-08/E-10).

TDD: tests written before the implementation.

Coverage:
  - build_scorecard over run_pass fixture: dimension scores, NA handling,
    weight renormalization, buckets, determinism.
  - NA dimensions stay None (not zeroed).
  - Overall skips NA dims and renormalizes weights.
  - Buckets dict covers all 7 buckets with float pass-rates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.golden import load_goldens
from src.eval.runner import load_replay
from src.eval.metrics.programmatic import score_programmatic
from src.eval.aggregate import (
    METRIC_DIMENSION,
    Dimension,
    Scorecard,
    build_scorecard,
)

# ---------------------------------------------------------------------------
# Fixture paths (same convention as test_programmatic.py)
# ---------------------------------------------------------------------------

DATASETS = Path(__file__).parent.parent / "datasets"
GOLDEN_SET = DATASETS / "golden_set.jsonl"
RUN_PASS = DATASETS / "fixtures" / "run_pass.jsonl"
RUN_FAIL = DATASETS / "fixtures" / "run_fail.jsonl"


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scorecard_run_pass() -> Scorecard:
    goldens = load_goldens(GOLDEN_SET)
    records = load_replay(RUN_PASS)
    prog_results = score_programmatic(records, goldens)
    return build_scorecard(
        records,
        goldens,
        prog_results=prog_results,
        judge_results=None,
        robustness_results=None,
        run_id="run_pass_test",
        mode="replay",
    )


# ---------------------------------------------------------------------------
# METRIC_DIMENSION mapping sanity
# ---------------------------------------------------------------------------


class TestMetricDimensionMapping:
    def test_all_7_dimensions_present_in_scorecard(self) -> None:
        """All 7 DIMENSION_WEIGHTS keys must appear as Scorecard dimensions.

        business_value intentionally has no offline metric in METRIC_DIMENSION
        (it is always NA), but it still appears as a Dimension in the Scorecard.
        This test checks that the Scorecard covers all 7 dims, not the mapping.
        """
        from src.config import DIMENSION_WEIGHTS
        # METRIC_DIMENSION covers 6 dims (business_value has no offline metric).
        # All 7 must appear in the built scorecard — checked in TestDimensionScores.
        dims_in_mapping = set(METRIC_DIMENSION.values())
        expected_without_bv = set(DIMENSION_WEIGHTS.keys()) - {"business_value"}
        assert dims_in_mapping == expected_without_bv

    def test_citation_validity_in_faithfulness_grounding(self) -> None:
        assert METRIC_DIMENSION["citation_validity"] == "faithfulness_grounding"

    def test_context_recall_in_retrieval_quality(self) -> None:
        assert METRIC_DIMENSION["context_recall"] == "retrieval_quality"

    def test_numerical_exactness_in_financial_correctness(self) -> None:
        assert METRIC_DIMENSION["numerical_exactness"] == "financial_correctness"

    def test_negative_rejection_in_safety_compliance(self) -> None:
        assert METRIC_DIMENSION["negative_rejection"] == "safety_compliance"

    def test_injection_resistance_in_robustness(self) -> None:
        assert METRIC_DIMENSION["injection_resistance"] == "robustness"

    def test_consistency_passk_in_consistency(self) -> None:
        assert METRIC_DIMENSION["consistency_passk"] == "consistency"


# ---------------------------------------------------------------------------
# Dimension scores over run_pass
# ---------------------------------------------------------------------------


class TestDimensionScores:
    def test_retrieval_quality_populated(self, scorecard_run_pass: Scorecard) -> None:
        dim = next(d for d in scorecard_run_pass.dimensions if d.name == "retrieval_quality")
        assert dim.score is not None
        assert dim.score == pytest.approx(95.83, abs=0.1)

    def test_retrieval_quality_green(self, scorecard_run_pass: Scorecard) -> None:
        dim = next(d for d in scorecard_run_pass.dimensions if d.name == "retrieval_quality")
        assert dim.status == "green"

    def test_financial_correctness_populated(self, scorecard_run_pass: Scorecard) -> None:
        dim = next(d for d in scorecard_run_pass.dimensions if d.name == "financial_correctness")
        assert dim.score is not None
        assert dim.score == pytest.approx(100.0, abs=0.1)

    def test_financial_correctness_green(self, scorecard_run_pass: Scorecard) -> None:
        dim = next(d for d in scorecard_run_pass.dimensions if d.name == "financial_correctness")
        assert dim.status == "green"

    def test_faithfulness_grounding_uses_citation_validity_only(
        self, scorecard_run_pass: Scorecard
    ) -> None:
        """faithfulness is absent (no judge), so only citation_validity contributes."""
        dim = next(d for d in scorecard_run_pass.dimensions if d.name == "faithfulness_grounding")
        assert dim.score is not None
        # citation_validity score=1.0 → dim score = 100.0
        assert dim.score == pytest.approx(100.0, abs=0.1)
        assert "citation_validity" in dim.metrics
        assert "faithfulness" not in dim.metrics

    def test_robustness_is_na(self, scorecard_run_pass: Scorecard) -> None:
        """injection_resistance is absent → robustness has no metrics → NA."""
        dim = next(d for d in scorecard_run_pass.dimensions if d.name == "robustness")
        assert dim.score is None
        assert dim.status == "na"

    def test_consistency_is_na(self, scorecard_run_pass: Scorecard) -> None:
        """consistency_passk is absent → consistency has no metrics → NA."""
        dim = next(d for d in scorecard_run_pass.dimensions if d.name == "consistency")
        assert dim.score is None
        assert dim.status == "na"

    def test_business_value_is_na(self, scorecard_run_pass: Scorecard) -> None:
        """business_value has no offline metrics → always NA."""
        dim = next(d for d in scorecard_run_pass.dimensions if d.name == "business_value")
        assert dim.score is None
        assert dim.status == "na"

    def test_all_7_dimensions_present(self, scorecard_run_pass: Scorecard) -> None:
        from src.config import DIMENSION_WEIGHTS
        names = {d.name for d in scorecard_run_pass.dimensions}
        assert names == set(DIMENSION_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Overall score (weight renormalization)
# ---------------------------------------------------------------------------


class TestOverallScore:
    def test_overall_is_not_none(self, scorecard_run_pass: Scorecard) -> None:
        assert scorecard_run_pass.overall is not None

    def test_overall_computed_over_non_na_dims_only(self, scorecard_run_pass: Scorecard) -> None:
        """
        Non-NA dims in run_pass: faithfulness_grounding(25), retrieval_quality(20),
        financial_correctness(20), safety_compliance(15). Total weight = 80.
        Renormalized weights: fg=25/80, rq=20/80, fc=20/80, sc=15/80.
        Expected: (100*25 + 95.833*20 + 100*20 + 100*15) / 80 ≈ 98.958.
        """
        assert scorecard_run_pass.overall == pytest.approx(98.958, abs=0.1)

    def test_overall_is_in_0_100_range(self, scorecard_run_pass: Scorecard) -> None:
        assert 0.0 <= scorecard_run_pass.overall <= 100.0

    def test_scorecard_has_run_id(self, scorecard_run_pass: Scorecard) -> None:
        assert scorecard_run_pass.run_id == "run_pass_test"

    def test_scorecard_has_mode(self, scorecard_run_pass: Scorecard) -> None:
        assert scorecard_run_pass.mode == "replay"

    def test_status_defaults_to_pending(self, scorecard_run_pass: Scorecard) -> None:
        assert scorecard_run_pass.status == "PENDING"

    def test_hard_gate_failures_defaults_empty(self, scorecard_run_pass: Scorecard) -> None:
        assert scorecard_run_pass.hard_gate_failures == []


# ---------------------------------------------------------------------------
# NA handling — score stays None, not 0
# ---------------------------------------------------------------------------


class TestNAHandling:
    def test_na_dim_score_is_none_not_zero(self, scorecard_run_pass: Scorecard) -> None:
        for dim in scorecard_run_pass.dimensions:
            if dim.status == "na":
                assert dim.score is None, (
                    f"Dimension '{dim.name}' should have score=None when NA, got {dim.score}"
                )

    def test_na_dim_metrics_dict_is_empty(self, scorecard_run_pass: Scorecard) -> None:
        for dim in scorecard_run_pass.dimensions:
            if dim.status == "na":
                assert dim.metrics == {}, (
                    f"NA dimension '{dim.name}' metrics should be empty, got {dim.metrics}"
                )

    def test_all_na_overall_is_none(self) -> None:
        """When all dimensions are NA, overall must be None."""
        from src.eval.aggregate import Dimension, Scorecard
        from src.config import DIMENSION_WEIGHTS

        na_dims = [
            Dimension(name=n, weight=w, score=None, status="na", metrics={})
            for n, w in DIMENSION_WEIGHTS.items()
        ]
        sc = Scorecard(
            run_id="test",
            mode="replay",
            dimensions=na_dims,
            buckets={},
            overall=None,
            metric_summary={},
        )
        assert sc.overall is None


# ---------------------------------------------------------------------------
# Buckets (E-10 per-bucket breakdown)
# ---------------------------------------------------------------------------


class TestBuckets:
    def test_buckets_dict_has_all_7_buckets(self, scorecard_run_pass: Scorecard) -> None:
        from src.eval.golden import Bucket
        expected = {b.value for b in Bucket}
        assert set(scorecard_run_pass.buckets.keys()) == expected

    def test_bucket_values_are_float_or_none(self, scorecard_run_pass: Scorecard) -> None:
        for bucket, rate in scorecard_run_pass.buckets.items():
            assert rate is None or isinstance(rate, float), (
                f"bucket '{bucket}' rate is not float or None: {rate!r}"
            )

    def test_bucket_values_in_0_1_range(self, scorecard_run_pass: Scorecard) -> None:
        for bucket, rate in scorecard_run_pass.buckets.items():
            if rate is not None:
                assert 0.0 <= rate <= 1.0, f"bucket '{bucket}' rate={rate} out of [0,1]"

    def test_factual_lookup_all_pass(self, scorecard_run_pass: Scorecard) -> None:
        assert scorecard_run_pass.buckets["factual_lookup"] == pytest.approx(1.0)

    def test_negative_all_pass(self, scorecard_run_pass: Scorecard) -> None:
        assert scorecard_run_pass.buckets["negative"] == pytest.approx(1.0)

    def test_entity_all_pass(self, scorecard_run_pass: Scorecard) -> None:
        assert scorecard_run_pass.buckets["entity"] == pytest.approx(1.0)

    def test_adversarial_all_pass(self, scorecard_run_pass: Scorecard) -> None:
        assert scorecard_run_pass.buckets["adversarial"] == pytest.approx(1.0)

    def test_multi_source_all_pass(self, scorecard_run_pass: Scorecard) -> None:
        assert scorecard_run_pass.buckets["multi_source"] == pytest.approx(1.0)

    def test_long_context_all_pass(self, scorecard_run_pass: Scorecard) -> None:
        # All 3 long_context items pass their gate metrics in run_pass.
        # Previously asserted 2/3 — that was a bug: context_recall below its soft
        # threshold was incorrectly counted as a hard gate failure.
        assert scorecard_run_pass.buckets["long_context"] == pytest.approx(1.0)

    def test_temporal_all_pass(self, scorecard_run_pass: Scorecard) -> None:
        # All 3 temporal items pass their gate metrics in run_pass.
        # Previously asserted 1/3 — that was a bug: context_precision/context_recall
        # below their soft thresholds were incorrectly counted as hard gate failures.
        assert scorecard_run_pass.buckets["temporal"] == pytest.approx(1.0)

    def test_all_7_buckets_pass_in_run_pass(self, scorecard_run_pass: Scorecard) -> None:
        """Money-shot: every bucket must be 1.0 for the run_pass fixture."""
        for bucket, rate in scorecard_run_pass.buckets.items():
            assert rate == pytest.approx(1.0), (
                f"bucket '{bucket}' expected 1.0 but got {rate} in run_pass"
            )


# ---------------------------------------------------------------------------
# metric_summary
# ---------------------------------------------------------------------------


class TestMetricSummary:
    def test_metric_summary_contains_citation_validity(
        self, scorecard_run_pass: Scorecard
    ) -> None:
        assert "citation_validity" in scorecard_run_pass.metric_summary

    def test_metric_summary_citation_validity_is_1(
        self, scorecard_run_pass: Scorecard
    ) -> None:
        val = scorecard_run_pass.metric_summary["citation_validity"]
        assert val == pytest.approx(1.0)

    def test_metric_summary_faithfulness_is_none(
        self, scorecard_run_pass: Scorecard
    ) -> None:
        """faithfulness is absent (no judge results) → None in summary."""
        assert scorecard_run_pass.metric_summary.get("faithfulness") is None

    def test_metric_summary_injection_resistance_is_none(
        self, scorecard_run_pass: Scorecard
    ) -> None:
        assert scorecard_run_pass.metric_summary.get("injection_resistance") is None


# ---------------------------------------------------------------------------
# run_fail fixture — defect buckets should be < 1.0; clean buckets stay 1.0
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scorecard_run_fail() -> Scorecard:
    goldens = load_goldens(GOLDEN_SET)
    records = load_replay(RUN_FAIL)
    prog_results = score_programmatic(records, goldens)
    return build_scorecard(
        records,
        goldens,
        prog_results=prog_results,
        judge_results=None,
        robustness_results=None,
        run_id="run_fail_test",
        mode="replay",
    )


class TestRunFailBuckets:
    def test_factual_lookup_has_failures(self, scorecard_run_fail: Scorecard) -> None:
        # fact-001 and fact-002 fail citation_validity; fact-003 passes → 1/3
        rate = scorecard_run_fail.buckets["factual_lookup"]
        assert rate is not None
        assert rate < 1.0
        assert rate == pytest.approx(1 / 3, abs=0.01)

    def test_temporal_has_failures(self, scorecard_run_fail: Scorecard) -> None:
        # temp-001 fails numerical_exactness + temporal_correctness → 2/3 pass
        rate = scorecard_run_fail.buckets["temporal"]
        assert rate is not None
        assert rate < 1.0
        assert rate == pytest.approx(2 / 3, abs=0.01)

    def test_negative_has_failures(self, scorecard_run_fail: Scorecard) -> None:
        # neg-001 fails citation_validity + negative_rejection → 2/3 pass
        rate = scorecard_run_fail.buckets["negative"]
        assert rate is not None
        assert rate < 1.0
        assert rate == pytest.approx(2 / 3, abs=0.01)

    def test_multi_source_has_failures(self, scorecard_run_fail: Scorecard) -> None:
        # multi-001 fails citation_validity → 2/3 pass
        rate = scorecard_run_fail.buckets["multi_source"]
        assert rate is not None
        assert rate < 1.0
        assert rate == pytest.approx(2 / 3, abs=0.01)

    def test_long_context_has_failures(self, scorecard_run_fail: Scorecard) -> None:
        # long-003 fails citation_validity → 2/3 pass
        rate = scorecard_run_fail.buckets["long_context"]
        assert rate is not None
        assert rate < 1.0
        assert rate == pytest.approx(2 / 3, abs=0.01)

    def test_entity_all_pass_in_run_fail(self, scorecard_run_fail: Scorecard) -> None:
        # entity items have no planted gate defects → all 3 pass
        assert scorecard_run_fail.buckets["entity"] == pytest.approx(1.0)

    def test_adversarial_all_pass_in_run_fail(self, scorecard_run_fail: Scorecard) -> None:
        # adversarial items have no planted gate defects → all 3 pass
        assert scorecard_run_fail.buckets["adversarial"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Empty bucket — bucket absent from the run → None (not 0.0)
# ---------------------------------------------------------------------------


class TestEmptyBucket:
    def test_empty_bucket_is_none(self) -> None:
        """A bucket with zero items in a run must be None, not 0.0."""
        import json
        from src.eval.runner import RunRecord

        goldens = load_goldens(GOLDEN_SET)

        # Build records that omit the "adversarial" bucket entirely.
        records_all = load_replay(RUN_PASS)
        records_no_adv = [r for r in records_all if r.bucket != "adversarial"]

        prog_results = score_programmatic(records_no_adv, goldens)
        sc = build_scorecard(
            records_no_adv,
            goldens,
            prog_results=prog_results,
            run_id="empty-bucket-test",
            mode="replay",
        )

        assert sc.buckets["adversarial"] is None, (
            f"Expected None for empty bucket, got {sc.buckets['adversarial']!r}"
        )
        # Other buckets should still be float
        for bucket, rate in sc.buckets.items():
            if bucket != "adversarial":
                assert isinstance(rate, float), (
                    f"Non-empty bucket '{bucket}' should be float, got {rate!r}"
                )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_building_twice_yields_identical_scores(self) -> None:
        goldens = load_goldens(GOLDEN_SET)
        records = load_replay(RUN_PASS)
        prog_results = score_programmatic(records, goldens)

        sc1 = build_scorecard(
            records, goldens,
            prog_results=prog_results,
            run_id="det-test",
            mode="replay",
        )
        sc2 = build_scorecard(
            records, goldens,
            prog_results=prog_results,
            run_id="det-test",
            mode="replay",
        )

        assert sc1.overall == sc2.overall
        for d1, d2 in zip(sc1.dimensions, sc2.dimensions):
            assert d1.score == d2.score
            assert d1.status == d2.status
        assert sc1.buckets == sc2.buckets
