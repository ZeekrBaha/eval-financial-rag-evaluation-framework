"""Tests for T14 — gates.py (E-09, hard-gate enforcement).

TDD: tests were written before the implementation.

Coverage:
  - run_pass fixture: negative_rejection=1.0 passes; faithfulness/hallucination_rate/
    advice_boundary unevaluated → INCOMPLETE, exit_code 2.
    (Programmatic-only replay cannot evaluate the judge/robustness hard gates;
    a full run with judge+robustness will produce PASS.)
  - run_fail fixture: negative_rejection≈0.667 < 0.95 → BLOCKED, exit_code 1;
    citation_validity≈0.80 → soft warning.
  - Synthetic: faithfulness=0.90 (below 0.95 threshold) → BLOCKED on faithfulness.
  - Synthetic: hallucination_rate=0.05 (above 0.01 max, op "<=") → BLOCKED.
  - Synthetic: all hard gates pass → PASS exit 0.
  - Synthetic: all hard gates None → INCOMPLETE exit 2.
  - Synthetic: one hard gate failing + others unevaluated → BLOCKED (precedence).
  - Determinism: calling enforce twice yields identical summary_lines.
  - Unevaluated hard gates are surfaced; blocking takes precedence over incompleteness.
  - Fail messages use the word "fails" (not misleading operator-as-assertion form).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.aggregate import Dimension, Scorecard, build_scorecard
from src.eval.gates import GateOutcome, enforce, evaluate_gates
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
# T14-01  run_pass fixture → INCOMPLETE
# ---------------------------------------------------------------------------
# run_pass uses programmatic-only replay: negative_rejection=1.0 is evaluated
# and passes, but faithfulness, hallucination_rate, and advice_boundary require
# a live LLM judge / robustness run and are therefore unevaluated (None).
# Because some hard gates are not evaluated, the status is INCOMPLETE (exit 2),
# NOT PASS. A full live run that supplies judge+robustness results will PASS.


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

    def test_status_is_incomplete(self, outcome: GateOutcome) -> None:
        # Programmatic-only replay cannot evaluate judge/robustness hard gates.
        assert outcome.status == "INCOMPLETE"

    def test_exit_code_is_2(self, outcome: GateOutcome) -> None:
        # exit 2 = INCOMPLETE (some hard gates not measured)
        assert outcome.exit_code == 2

    def test_no_blocking_failures(self, outcome: GateOutcome) -> None:
        assert outcome.blocking_failures == []

    def test_scorecard_status_stamped_incomplete(self, scorecard: Scorecard) -> None:
        assert scorecard.status == "INCOMPLETE"

    def test_scorecard_hard_gate_failures_empty(self, scorecard: Scorecard) -> None:
        assert scorecard.hard_gate_failures == []

    def test_summary_first_line_release_incomplete(self, outcome: GateOutcome) -> None:
        assert outcome.summary_lines[0] == "RELEASE INCOMPLETE"

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

    def test_summary_contains_unevaluated_count(self, outcome: GateOutcome) -> None:
        """INCOMPLETE summary must mention the count of unevaluated hard gates."""
        summary_text = "\n".join(outcome.summary_lines)
        assert "hard gate(s) not evaluated" in summary_text


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
# T14-06  Synthetic: all hard gates None → INCOMPLETE exit 2
# ---------------------------------------------------------------------------


class TestSyntheticAllNoneIncomplete:
    @pytest.fixture(scope="class")
    def outcome(self) -> GateOutcome:
        sc = _make_minimal_scorecard({})  # no metrics at all
        return enforce(sc)

    def test_status_incomplete(self, outcome: GateOutcome) -> None:
        assert outcome.status == "INCOMPLETE"

    def test_exit_code_2(self, outcome: GateOutcome) -> None:
        assert outcome.exit_code == 2

    def test_no_blocking_failures(self, outcome: GateOutcome) -> None:
        assert outcome.blocking_failures == []

    def test_summary_first_line_release_incomplete(self, outcome: GateOutcome) -> None:
        assert outcome.summary_lines[0] == "RELEASE INCOMPLETE"

    def test_summary_mentions_unevaluated_count(self, outcome: GateOutcome) -> None:
        summary_text = "\n".join(outcome.summary_lines)
        assert "hard gate(s) not evaluated" in summary_text


# ---------------------------------------------------------------------------
# T14-07  Synthetic: one hard gate failing + others unevaluated → BLOCKED
#         (BLOCKED takes precedence over INCOMPLETE)
# ---------------------------------------------------------------------------


class TestSyntheticBlockedPrecedenceOverIncomplete:
    @pytest.fixture(scope="class")
    def outcome(self) -> GateOutcome:
        # negative_rejection evaluated and failing; all others unevaluated
        sc = _make_minimal_scorecard({"negative_rejection": 0.50})
        return enforce(sc)

    def test_status_blocked(self, outcome: GateOutcome) -> None:
        assert outcome.status == "BLOCKED"

    def test_exit_code_1(self, outcome: GateOutcome) -> None:
        # Blocking takes precedence over incompleteness → exit 1, not 2
        assert outcome.exit_code == 1

    def test_negative_rejection_in_blocking(self, outcome: GateOutcome) -> None:
        names = [r.name for r in outcome.blocking_failures]
        assert "negative_rejection" in names

    def test_unevaluated_hard_still_populated(self, outcome: GateOutcome) -> None:
        # Other gates are still unevaluated, surfaced for visibility
        unevaluated_names = [r.name for r in outcome.unevaluated_hard]
        assert "faithfulness" in unevaluated_names


# ---------------------------------------------------------------------------
# T14-08  Fail message phrasing: "fails" keyword, not misleading operator form
# ---------------------------------------------------------------------------


class TestFailMessagePhrasing:
    def test_blocking_hard_line_uses_fails(self) -> None:
        """Blocking summary lines must say 'fails', not bare operator (ambiguous)."""
        sc = _make_minimal_scorecard({"negative_rejection": 0.50})
        outcome = enforce(sc)
        blocking_lines = [
            line for line in outcome.summary_lines
            if "negative_rejection" in line
        ]
        assert blocking_lines, "Expected a summary line for negative_rejection"
        assert "fails" in blocking_lines[0], (
            f"Expected 'fails' in blocking line; got: {blocking_lines[0]!r}"
        )

    def test_soft_warning_line_uses_fails(self) -> None:
        """Soft-gate warning lines must say 'fails'."""
        from src.config import SOFT_GATES

        # Find a soft gate metric and drive it below its threshold
        gate = SOFT_GATES[0]
        metric = gate["metric"]
        op = gate["op"]
        # Use a value that definitely fails: 0.0 for >= gates, 1.0 for <= gates
        failing_value = 0.0 if op == ">=" else 1.0
        sc = _make_minimal_scorecard({metric: failing_value})
        outcome = enforce(sc)
        warning_lines = [
            line for line in outcome.summary_lines
            if metric in line
        ]
        assert warning_lines, f"Expected a warning line for {metric}"
        assert "fails" in warning_lines[0], (
            f"Expected 'fails' in warning line; got: {warning_lines[0]!r}"
        )


# ---------------------------------------------------------------------------
# T14-09  Determinism: summary_lines identical on two calls
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
# T14-10  GateResult structural checks via evaluate_gates
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

    def test_unevaluated_hard_yields_incomplete(self) -> None:
        """All hard gates unevaluated → INCOMPLETE (not PASS, not BLOCKED)."""
        # Only provide soft metrics so all hard gates are unevaluated
        sc = _make_minimal_scorecard({})
        outcome = enforce(sc)
        assert outcome.status == "INCOMPLETE"
        assert outcome.exit_code == 2
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
