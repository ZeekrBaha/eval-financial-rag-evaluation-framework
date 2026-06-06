"""Tests for T14 — gates.py (E-09, hard-gate enforcement).

TDD: tests were written before the implementation.

Coverage:
  - run_pass fixture: negative_rejection=1.0 passes; faithfulness/hallucination_rate/
    advice_boundary unevaluated → PASS, exit_code 0.
  - run_fail fixture: negative_rejection≈0.667 < 0.95 → BLOCKED, exit_code 1;
    citation_validity≈0.80 → soft warning.
  - Synthetic: faithfulness=0.90 (below 0.95 threshold) → BLOCKED on faithfulness.
  - Synthetic: hallucination_rate=0.05 (above 0.01 max, op "<=") → BLOCKED.
  - Synthetic: all hard gates pass → PASS exit 0.
  - Determinism: calling enforce twice yields identical summary_lines.
  - Unevaluated hard gates are surfaced but do NOT block.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.aggregate import Dimension, Scorecard, build_scorecard
from src.eval.gates import GateOutcome, GateResult, enforce, evaluate_gates
from src.eval.golden import load_goldens
from src.eval.metrics.programmatic import score_programmatic
from src.eval.runner import load_replay

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

DATASETS = Path(__file__).parent.parent / "datasets"
GOLDEN_SET = DATASETS / "golden_set.jsonl"
RUN_PASS = DATASETS / "fixtures" / "run_pass.jsonl"
RUN_FAIL = DATASETS / "fixtures" / "run_fail.jsonl"


# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------


def _build_scorecard_from_fixture(run_path: Path) -> Scorecard:
    goldens = load_goldens(GOLDEN_SET)
    records = load_replay(run_path)
    prog_results = score_programmatic(records, goldens)
    return build_scorecard(
        records,
        goldens,
        prog_results=prog_results,
        run_id="gate_test",
        mode="replay",
    )


def _make_minimal_scorecard(metric_summary: dict[str, float | None]) -> Scorecard:
    """Build a minimal Scorecard with just enough to drive gate enforcement."""
    from src.config import DIMENSION_WEIGHTS

    dims = [
        Dimension(name=n, weight=w, score=None, status="na", metrics={})
        for n, w in DIMENSION_WEIGHTS.items()
    ]
    return Scorecard(
        run_id="synthetic",
        mode="replay",
        dimensions=dims,
        buckets={},
        overall=None,
        metric_summary=metric_summary,
    )


# ---------------------------------------------------------------------------
# T14-01  run_pass fixture → PASS
# ---------------------------------------------------------------------------


class TestRunPass:
    @pytest.fixture(scope="class")
    def outcome(self) -> GateOutcome:
        sc = _build_scorecard_from_fixture(RUN_PASS)
        return enforce(sc)

    @pytest.fixture(scope="class")
    def scorecard(self) -> Scorecard:
        sc = _build_scorecard_from_fixture(RUN_PASS)
        enforce(sc)
        return sc

    def test_status_is_pass(self, outcome: GateOutcome) -> None:
        assert outcome.status == "PASS"

    def test_exit_code_is_0(self, outcome: GateOutcome) -> None:
        assert outcome.exit_code == 0

    def test_no_blocking_failures(self, outcome: GateOutcome) -> None:
        assert outcome.blocking_failures == []

    def test_scorecard_status_stamped_pass(self, scorecard: Scorecard) -> None:
        assert scorecard.status == "PASS"

    def test_scorecard_hard_gate_failures_empty(self, scorecard: Scorecard) -> None:
        assert scorecard.hard_gate_failures == []

    def test_summary_first_line_release_ok(self, outcome: GateOutcome) -> None:
        assert outcome.summary_lines[0] == "RELEASE OK"

    def test_unevaluated_hard_contains_faithfulness(self, outcome: GateOutcome) -> None:
        unevaluated_names = [r.name for r in outcome.unevaluated_hard]
        assert "faithfulness" in unevaluated_names

    def test_unevaluated_hard_contains_hallucination_rate(self, outcome: GateOutcome) -> None:
        unevaluated_names = [r.name for r in outcome.unevaluated_hard]
        assert "hallucination_rate" in unevaluated_names

    def test_unevaluated_hard_contains_advice_boundary(self, outcome: GateOutcome) -> None:
        unevaluated_names = [r.name for r in outcome.unevaluated_hard]
        assert "advice_boundary" in unevaluated_names

    def test_negative_rejection_evaluated_and_passed(self, outcome: GateOutcome) -> None:
        neg_results = [r for r in outcome.hard_results if r.name == "negative_rejection"]
        assert len(neg_results) == 1
        nr = neg_results[0]
        assert nr.evaluated is True
        assert nr.passed is True


# ---------------------------------------------------------------------------
# T14-02  run_fail fixture → BLOCKED
# ---------------------------------------------------------------------------


class TestRunFail:
    @pytest.fixture(scope="class")
    def outcome(self) -> GateOutcome:
        sc = _build_scorecard_from_fixture(RUN_FAIL)
        return enforce(sc)

    @pytest.fixture(scope="class")
    def scorecard(self) -> Scorecard:
        sc = _build_scorecard_from_fixture(RUN_FAIL)
        enforce(sc)
        return sc

    def test_status_is_blocked(self, outcome: GateOutcome) -> None:
        assert outcome.status == "BLOCKED"

    def test_exit_code_is_1(self, outcome: GateOutcome) -> None:
        assert outcome.exit_code == 1

    def test_summary_first_line_release_blocked(self, outcome: GateOutcome) -> None:
        assert outcome.summary_lines[0] == "RELEASE BLOCKED"

    def test_blocking_failures_non_empty(self, outcome: GateOutcome) -> None:
        assert len(outcome.blocking_failures) > 0

    def test_negative_rejection_in_blocking_failures(self, outcome: GateOutcome) -> None:
        names = [r.name for r in outcome.blocking_failures]
        assert "negative_rejection" in names

    def test_scorecard_status_stamped_blocked(self, scorecard: Scorecard) -> None:
        assert scorecard.status == "BLOCKED"

    def test_scorecard_hard_gate_failures_non_empty(self, scorecard: Scorecard) -> None:
        assert len(scorecard.hard_gate_failures) > 0

    def test_soft_warning_includes_citation_validity(self, outcome: GateOutcome) -> None:
        """citation_validity≈0.80 in run_fail is below the soft gate threshold 0.95."""
        failed_soft_names = [r.name for r in outcome.soft_results if r.passed is False]
        assert "citation_validity" in failed_soft_names

    def test_warnings_section_in_summary(self, outcome: GateOutcome) -> None:
        summary_text = "\n".join(outcome.summary_lines)
        assert "WARNINGS:" in summary_text

    def test_blocking_line_names_negative_rejection(self, outcome: GateOutcome) -> None:
        """The summary must have a blocking line mentioning negative_rejection."""
        summary_text = "\n".join(outcome.summary_lines)
        assert "negative_rejection" in summary_text


# ---------------------------------------------------------------------------
# T14-03  Synthetic: faithfulness=0.90 (hard gate, >= 0.95) → BLOCKED
# ---------------------------------------------------------------------------


class TestSyntheticFaithfulnessBlocked:
    @pytest.fixture(scope="class")
    def outcome(self) -> GateOutcome:
        # faithfulness below threshold; other hard gates unevaluated
        sc = _make_minimal_scorecard({"faithfulness": 0.90})
        return enforce(sc)

    def test_status_blocked(self, outcome: GateOutcome) -> None:
        assert outcome.status == "BLOCKED"

    def test_exit_code_1(self, outcome: GateOutcome) -> None:
        assert outcome.exit_code == 1

    def test_faithfulness_in_blocking(self, outcome: GateOutcome) -> None:
        names = [r.name for r in outcome.blocking_failures]
        assert "faithfulness" in names

    def test_summary_first_line_release_blocked(self, outcome: GateOutcome) -> None:
        assert outcome.summary_lines[0] == "RELEASE BLOCKED"

    def test_message_contains_fail(self, outcome: GateOutcome) -> None:
        fail_results = [r for r in outcome.hard_results if r.name == "faithfulness"]
        assert len(fail_results) == 1
        assert "FAIL" in fail_results[0].message


# ---------------------------------------------------------------------------
# T14-04  Synthetic: hallucination_rate=0.05 (hard gate, <= 0.01) → BLOCKED
# ---------------------------------------------------------------------------


class TestSyntheticHallucinationRateBlocked:
    @pytest.fixture(scope="class")
    def outcome(self) -> GateOutcome:
        sc = _make_minimal_scorecard({"hallucination_rate": 0.05})
        return enforce(sc)

    def test_status_blocked(self, outcome: GateOutcome) -> None:
        assert outcome.status == "BLOCKED"

    def test_exit_code_1(self, outcome: GateOutcome) -> None:
        assert outcome.exit_code == 1

    def test_hallucination_rate_in_blocking(self, outcome: GateOutcome) -> None:
        names = [r.name for r in outcome.blocking_failures]
        assert "hallucination_rate" in names

    def test_op_direction_le(self, outcome: GateOutcome) -> None:
        """Verify the <= direction: 0.05 > 0.01, so it should FAIL."""
        hr = [r for r in outcome.hard_results if r.name == "hallucination_rate"][0]
        assert hr.op == "<="
        assert hr.passed is False
        assert hr.value == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# T14-05  Synthetic: all hard gates evaluated and pass → PASS
# ---------------------------------------------------------------------------


class TestSyntheticAllHardPass:
    @pytest.fixture(scope="class")
    def outcome(self) -> GateOutcome:
        # All hard gates exactly at or above thresholds
        sc = _make_minimal_scorecard(
            {
                "faithfulness": 0.95,          # >= 0.95 ✓
                "negative_rejection": 1.0,      # >= 0.95 ✓
                "hallucination_rate": 0.00,     # <= 0.01 ✓
                "advice_boundary": 1.0,         # >= 1.0  ✓
            }
        )
        return enforce(sc)

    def test_status_pass(self, outcome: GateOutcome) -> None:
        assert outcome.status == "PASS"

    def test_exit_code_0(self, outcome: GateOutcome) -> None:
        assert outcome.exit_code == 0

    def test_no_blocking_failures(self, outcome: GateOutcome) -> None:
        assert outcome.blocking_failures == []

    def test_unevaluated_hard_empty(self, outcome: GateOutcome) -> None:
        assert outcome.unevaluated_hard == []

    def test_all_hard_results_passed(self, outcome: GateOutcome) -> None:
        for r in outcome.hard_results:
            assert r.passed is True, f"{r.name} should have passed"


# ---------------------------------------------------------------------------
# T14-06  Determinism: summary_lines identical on two calls
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_summary_lines_identical(self) -> None:
        sc1 = _build_scorecard_from_fixture(RUN_PASS)
        sc2 = _build_scorecard_from_fixture(RUN_PASS)
        outcome1 = enforce(sc1)
        outcome2 = enforce(sc2)
        assert outcome1.summary_lines == outcome2.summary_lines

    def test_summary_lines_identical_fail(self) -> None:
        sc1 = _build_scorecard_from_fixture(RUN_FAIL)
        sc2 = _build_scorecard_from_fixture(RUN_FAIL)
        outcome1 = enforce(sc1)
        outcome2 = enforce(sc2)
        assert outcome1.summary_lines == outcome2.summary_lines


# ---------------------------------------------------------------------------
# T14-07  GateResult structural checks via evaluate_gates
# ---------------------------------------------------------------------------


class TestEvaluateGates:
    def test_unevaluated_gate_has_passed_none(self) -> None:
        hard, _ = evaluate_gates({})
        for r in hard:
            assert r.evaluated is False
            assert r.passed is None

    def test_unevaluated_gate_message_contains_not_evaluated(self) -> None:
        hard, _ = evaluate_gates({})
        for r in hard:
            assert "not evaluated" in r.message

    def test_evaluated_pass_message_contains_ok(self) -> None:
        hard, _ = evaluate_gates({"negative_rejection": 1.0})
        nr = [r for r in hard if r.name == "negative_rejection"][0]
        assert "OK" in nr.message
        assert nr.evaluated is True
        assert nr.passed is True

    def test_evaluated_fail_message_contains_fail(self) -> None:
        hard, _ = evaluate_gates({"negative_rejection": 0.50})
        nr = [r for r in hard if r.name == "negative_rejection"][0]
        assert "FAIL" in nr.message
        assert nr.evaluated is True
        assert nr.passed is False

    def test_gate_result_kind_hard(self) -> None:
        hard, _ = evaluate_gates({})
        for r in hard:
            assert r.kind == "hard"

    def test_gate_result_kind_soft(self) -> None:
        _, soft = evaluate_gates({})
        for r in soft:
            assert r.kind == "soft"

    def test_value_rounded_to_3dp_in_message(self) -> None:
        hard, _ = evaluate_gates({"faithfulness": 0.12345})
        fr = [r for r in hard if r.name == "faithfulness"][0]
        # message should contain the 3dp-rounded value, not full precision
        assert "0.123" in fr.message
        assert "0.12345" not in fr.message

    def test_unevaluated_hard_does_not_block(self) -> None:
        """A hard gate that is not in metric_summary must NOT cause BLOCKED."""
        # Only provide soft metrics so all hard gates are unevaluated
        sc = _make_minimal_scorecard({})
        outcome = enforce(sc)
        assert outcome.status == "PASS"
        assert outcome.exit_code == 0
        assert len(outcome.unevaluated_hard) == 4  # all 4 hard gates unevaluated

    def test_unevaluated_section_in_summary(self) -> None:
        sc = _make_minimal_scorecard({})
        outcome = enforce(sc)
        summary_text = "\n".join(outcome.summary_lines)
        assert "UNEVALUATED" in summary_text

    def test_ge_boundary_exact_threshold_passes(self) -> None:
        """Edge case: value == threshold with >= should pass."""
        hard, _ = evaluate_gates({"faithfulness": 0.95})
        fr = [r for r in hard if r.name == "faithfulness"][0]
        assert fr.passed is True

    def test_le_boundary_exact_threshold_passes(self) -> None:
        """Edge case: value == threshold with <= should pass."""
        hard, _ = evaluate_gates({"hallucination_rate": 0.01})
        hr = [r for r in hard if r.name == "hallucination_rate"][0]
        assert hr.passed is True

    def test_ge_just_below_threshold_fails(self) -> None:
        hard, _ = evaluate_gates({"faithfulness": 0.9499})
        fr = [r for r in hard if r.name == "faithfulness"][0]
        assert fr.passed is False

    def test_le_just_above_threshold_fails(self) -> None:
        hard, _ = evaluate_gates({"hallucination_rate": 0.0101})
        hr = [r for r in hard if r.name == "hallucination_rate"][0]
        assert hr.passed is False
