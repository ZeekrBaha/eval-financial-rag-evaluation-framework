"""Tests for src/eval/report.py — REPORT.md generation.

TDD approach: tests validate the render_report() and write_report() functions.

Coverage:
  - render_report with a passing run produces correct markdown sections.
  - render_report with a failing run lists failing items in Findings.
  - write_report writes the file to the given path.
"""

from __future__ import annotations

from pathlib import Path

from src.eval.aggregate import Dimension, Scorecard
from src.eval.gates import GateOutcome, GateResult
from src.eval.golden import Bucket, GoldenItem
from src.eval.metrics.programmatic import MetricResult


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_golden(item_id: str, bucket: Bucket, must_refuse: bool = False) -> GoldenItem:
    return GoldenItem(
        id=item_id,
        bucket=bucket,
        question="Test question?",
        reference_answer="Test answer.",
        expected_sources=[],
        numeric_answers=[],
        must_refuse=must_refuse,
        injection=None,
        advice_boundary=False,
    )


def _make_passing_scorecard(run_id: str = "run-pass-001") -> Scorecard:
    return Scorecard(
        run_id=run_id,
        mode="replay",
        dimensions=[
            Dimension(
                name="faithfulness_grounding",
                weight=25,
                score=98.0,
                status="green",
                metrics={"citation_validity": 0.98},
            ),
            Dimension(
                name="retrieval_quality",
                weight=20,
                score=91.0,
                status="green",
                metrics={"context_recall": 0.91, "context_precision": 0.91},
            ),
            Dimension(
                name="financial_correctness",
                weight=20,
                score=100.0,
                status="green",
                metrics={"numerical_exactness": 1.0, "temporal_correctness": 1.0},
            ),
            Dimension(
                name="safety_compliance",
                weight=15,
                score=100.0,
                status="green",
                metrics={"negative_rejection": 1.0},
            ),
            Dimension(name="robustness", weight=10, score=None, status="na", metrics={}),
            Dimension(name="consistency", weight=5, score=None, status="na", metrics={}),
            Dimension(name="business_value", weight=5, score=None, status="na", metrics={}),
        ],
        buckets={
            "factual_lookup": 1.0,
            "multi_source": 1.0,
            "temporal": None,
            "negative": 1.0,
            "entity": None,
            "adversarial": None,
            "long_context": None,
        },
        overall=97.25,
        metric_summary={
            "citation_validity": 0.98,
            "context_recall": 0.91,
            "context_precision": 0.91,
            "numerical_exactness": 1.0,
            "temporal_correctness": 1.0,
            "negative_rejection": 1.0,
            "faithfulness": None,
            "hallucination_rate": None,
            "advice_boundary": None,
            "answer_relevance": None,
            "entity_disambiguation": None,
            "injection_resistance": None,
            "consistency_passk": None,
        },
        status="PASS",
        hard_gate_failures=[],
    )


def _make_passing_outcome() -> GateOutcome:
    return GateOutcome(
        status="PASS",
        exit_code=0,
        hard_results=[
            GateResult(
                name="negative_rejection",
                kind="hard",
                threshold=0.95,
                op=">=",
                value=1.0,
                evaluated=True,
                passed=True,
                message="negative_rejection 1.0 >= 0.95 OK",
            ),
        ],
        soft_results=[],
        blocking_failures=[],
        unevaluated_hard=[],
        summary_lines=["RELEASE OK"],
    )


def _make_failing_scorecard(run_id: str = "run-fail-001") -> Scorecard:
    return Scorecard(
        run_id=run_id,
        mode="replay",
        dimensions=[
            Dimension(
                name="faithfulness_grounding",
                weight=25,
                score=0.0,
                status="red",
                metrics={"faithfulness": 0.0, "citation_validity": 0.0},
            ),
            Dimension(
                name="safety_compliance",
                weight=15,
                score=0.0,
                status="red",
                metrics={"negative_rejection": 0.0},
            ),
            Dimension(name="retrieval_quality", weight=20, score=None, status="na", metrics={}),
            Dimension(name="financial_correctness", weight=20, score=None, status="na", metrics={}),
            Dimension(name="robustness", weight=10, score=None, status="na", metrics={}),
            Dimension(name="consistency", weight=5, score=None, status="na", metrics={}),
            Dimension(name="business_value", weight=5, score=None, status="na", metrics={}),
        ],
        buckets={
            "factual_lookup": 0.0,
            "multi_source": None,
            "temporal": None,
            "negative": 0.0,
            "entity": None,
            "adversarial": None,
            "long_context": None,
        },
        overall=0.0,
        metric_summary={
            "citation_validity": 0.5,
            "faithfulness": 0.0,
            "hallucination_rate": 1.0,
            "negative_rejection": 0.0,
            "context_recall": None,
            "context_precision": None,
            "numerical_exactness": None,
            "temporal_correctness": None,
            "entity_disambiguation": None,
            "advice_boundary": None,
            "answer_relevance": None,
            "injection_resistance": None,
            "consistency_passk": None,
        },
        status="BLOCKED",
        hard_gate_failures=["faithfulness", "negative_rejection", "hallucination_rate"],
    )


def _make_failing_outcome() -> GateOutcome:
    return GateOutcome(
        status="BLOCKED",
        exit_code=1,
        hard_results=[
            GateResult(
                name="faithfulness",
                kind="hard",
                threshold=0.95,
                op=">=",
                value=0.0,
                evaluated=True,
                passed=False,
                message="faithfulness 0.0 >= 0.95 FAIL",
            ),
            GateResult(
                name="negative_rejection",
                kind="hard",
                threshold=0.95,
                op=">=",
                value=0.0,
                evaluated=True,
                passed=False,
                message="negative_rejection 0.0 >= 0.95 FAIL",
            ),
            GateResult(
                name="hallucination_rate",
                kind="hard",
                threshold=0.01,
                op="<=",
                value=1.0,
                evaluated=True,
                passed=False,
                message="hallucination_rate 1.0 <= 0.01 FAIL",
            ),
        ],
        soft_results=[],
        blocking_failures=[
            GateResult(
                name="faithfulness",
                kind="hard",
                threshold=0.95,
                op=">=",
                value=0.0,
                evaluated=True,
                passed=False,
                message="faithfulness 0.0 >= 0.95 FAIL",
            ),
        ],
        unevaluated_hard=[],
        summary_lines=[
            "RELEASE BLOCKED",
            "  - faithfulness: 0.0 fails >= 0.95 (hard gate)",
            "  - negative_rejection: 0.0 fails >= 0.95 (hard gate)",
            "  - hallucination_rate: 1.0 fails <= 0.01 (hard gate)",
        ],
    )


def _make_passing_metric_results() -> list[MetricResult]:
    """All items pass — no failures expected in Findings."""
    return [
        MetricResult(
            metric="citation_validity",
            item_id="fact-001",
            applicable=True,
            score=1.0,
            passed=True,
            detail="all citations valid",
        ),
        MetricResult(
            metric="negative_rejection",
            item_id="neg-001",
            applicable=True,
            score=1.0,
            passed=True,
            detail="correctly refused",
        ),
    ]


def _make_failing_metric_results() -> list[MetricResult]:
    """neg-001 fails negative_rejection, fact-001 fails citation_validity."""
    return [
        MetricResult(
            metric="negative_rejection",
            item_id="neg-001",
            applicable=True,
            score=0.0,
            passed=False,
            detail="did not refuse when must_refuse=True",
        ),
        MetricResult(
            metric="citation_validity",
            item_id="fact-001",
            applicable=True,
            score=0.0,
            passed=False,
            detail="citation c1 does not support the claim",
        ),
        MetricResult(
            metric="faithfulness",
            item_id="fact-001",
            applicable=True,
            score=0.0,
            passed=False,
            detail="faithfulness score below threshold",
        ),
    ]


def _make_goldens_for_passing() -> list[GoldenItem]:
    return [
        _make_golden("fact-001", Bucket.factual_lookup),
        _make_golden("neg-001", Bucket.negative, must_refuse=True),
    ]


def _make_goldens_for_failing() -> list[GoldenItem]:
    return [
        _make_golden("fact-001", Bucket.factual_lookup),
        _make_golden("neg-001", Bucket.negative, must_refuse=True),
    ]


# ---------------------------------------------------------------------------
# Tests: render_report passing run
# ---------------------------------------------------------------------------

class TestRenderReportPassing:
    def setup_method(self) -> None:
        from src.eval.report import render_report
        sc = _make_passing_scorecard()
        outcome = _make_passing_outcome()
        goldens = _make_goldens_for_passing()
        metrics = _make_passing_metric_results()
        self.md = render_report(sc, outcome, goldens, metrics)

    def test_contains_title(self) -> None:
        assert "# Evaluation Report" in self.md

    def test_contains_run_id(self) -> None:
        assert "run-pass-001" in self.md

    def test_contains_release_ok(self) -> None:
        assert "RELEASE OK" in self.md

    def test_contains_dimension_name(self) -> None:
        assert "faithfulness_grounding" in self.md

    def test_contains_dimension_table_header(self) -> None:
        assert "| Dimension |" in self.md

    def test_contains_bucket(self) -> None:
        assert "factual_lookup" in self.md

    def test_contains_metric_summary_section(self) -> None:
        assert "## Metric summary" in self.md

    def test_contains_metric_summary_row(self) -> None:
        # citation_validity should appear with a numeric value
        assert "citation_validity" in self.md

    def test_findings_all_passed(self) -> None:
        assert "All evaluated items passed their gate metrics." in self.md

    def test_contains_release_decision_section(self) -> None:
        assert "## Release decision" in self.md

    def test_contains_notes_section(self) -> None:
        assert "## Notes" in self.md

    def test_exit_code_in_header(self) -> None:
        assert "(exit 0)" in self.md

    def test_contains_overall_row(self) -> None:
        assert "**Overall**" in self.md

    def test_contains_per_bucket_section(self) -> None:
        assert "## Per-bucket pass rate" in self.md

    def test_contains_dimensions_section(self) -> None:
        assert "## Dimensions" in self.md


# ---------------------------------------------------------------------------
# Tests: render_report failing run
# ---------------------------------------------------------------------------

class TestRenderReportFailing:
    def setup_method(self) -> None:
        from src.eval.report import render_report
        sc = _make_failing_scorecard()
        outcome = _make_failing_outcome()
        goldens = _make_goldens_for_failing()
        metrics = _make_failing_metric_results()
        self.md = render_report(sc, outcome, goldens, metrics)

    def test_contains_release_blocked(self) -> None:
        assert "RELEASE BLOCKED" in self.md

    def test_exit_code_one(self) -> None:
        assert "(exit 1)" in self.md

    def test_findings_contains_neg001(self) -> None:
        assert "neg-001" in self.md

    def test_findings_contains_fact001(self) -> None:
        assert "fact-001" in self.md

    def test_findings_contains_negative_rejection(self) -> None:
        assert "negative_rejection" in self.md

    def test_findings_contains_citation_validity(self) -> None:
        assert "citation_validity" in self.md

    def test_findings_contains_detail(self) -> None:
        assert "did not refuse when must_refuse=True" in self.md

    def test_findings_contains_bucket(self) -> None:
        # neg-001 is in "negative" bucket
        assert "(negative)" in self.md or "negative" in self.md

    def test_not_all_passed_message(self) -> None:
        assert "All evaluated items passed" not in self.md

    def test_findings_section_header(self) -> None:
        assert "## Findings" in self.md

    def test_metric_summary_fail_annotation(self) -> None:
        # faithfulness 0.0 should show Fail (fails >= 0.95)
        assert "Fail" in self.md

    def test_faithfulness_in_metric_summary(self) -> None:
        assert "faithfulness" in self.md

    def test_notes_section_present(self) -> None:
        assert "## Notes" in self.md

    def test_proposed_gates_note(self) -> None:
        assert "Proposed gates" in self.md

    def test_uncalibrated_note(self) -> None:
        assert "UNCALIBRATED" in self.md

    def test_fictional_data_note(self) -> None:
        assert "Fictional data" in self.md

    def test_incomplete_meaning_note(self) -> None:
        assert "INCOMPLETE meaning" in self.md


# ---------------------------------------------------------------------------
# Tests: soft retrieval metrics excluded from per-item findings
# ---------------------------------------------------------------------------

class TestFindingsExcludeSoftRetrievalMetrics:
    def test_context_precision_not_in_findings(self) -> None:
        """context_precision is a soft retrieval metric, excluded from findings."""
        from src.eval.report import render_report

        sc = _make_passing_scorecard()
        outcome = _make_passing_outcome()
        goldens = _make_goldens_for_passing()

        # Add a "failing" context_precision result — should NOT appear in findings
        metrics = _make_passing_metric_results() + [
            MetricResult(
                metric="context_precision",
                item_id="fact-001",
                applicable=True,
                score=0.0,
                passed=False,
                detail="low precision",
            )
        ]
        md = render_report(sc, outcome, goldens, metrics)
        # The soft retrieval metric should not produce a findings bullet
        assert "context_precision failed" not in md
        # And since no *gate* metric failed, we get the "all passed" message
        assert "All evaluated items passed their gate metrics." in md


# ---------------------------------------------------------------------------
# Tests: write_report
# ---------------------------------------------------------------------------

class TestWriteReport:
    def test_writes_file(self, tmp_path: Path) -> None:
        from src.eval.report import write_report
        sc = _make_passing_scorecard()
        outcome = _make_passing_outcome()
        goldens = _make_goldens_for_passing()
        metrics = _make_passing_metric_results()

        out_path = tmp_path / "REPORT.md"
        write_report(sc, outcome, goldens, metrics, out_path)
        assert out_path.exists()

    def test_file_is_non_empty(self, tmp_path: Path) -> None:
        from src.eval.report import write_report
        sc = _make_passing_scorecard()
        outcome = _make_passing_outcome()
        goldens = _make_goldens_for_passing()
        metrics = _make_passing_metric_results()

        out_path = tmp_path / "REPORT.md"
        write_report(sc, outcome, goldens, metrics, out_path)
        assert out_path.stat().st_size > 0

    def test_file_contains_expected_content(self, tmp_path: Path) -> None:
        from src.eval.report import write_report
        sc = _make_passing_scorecard("t-write-001")
        outcome = _make_passing_outcome()
        goldens = _make_goldens_for_passing()
        metrics = _make_passing_metric_results()

        out_path = tmp_path / "REPORT.md"
        write_report(sc, outcome, goldens, metrics, out_path)
        content = out_path.read_text()
        assert "# Evaluation Report" in content
        assert "t-write-001" in content
        assert "RELEASE OK" in content

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        from src.eval.report import write_report
        sc = _make_passing_scorecard()
        outcome = _make_passing_outcome()
        goldens = _make_goldens_for_passing()
        metrics = _make_passing_metric_results()

        out_path = str(tmp_path / "REPORT.md")
        write_report(sc, outcome, goldens, metrics, out_path)
        assert Path(out_path).exists()
