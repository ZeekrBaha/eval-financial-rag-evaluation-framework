"""gates.py — Hard-gate enforcement for T14 / E-09.

Public surface:
  GateResult   — dataclass: outcome for a single gate check.
  GateOutcome  — dataclass: full release decision with summary.
  evaluate_gates(metric_summary) -> (hard_results, soft_results)
  decide_release(hard_results)   -> (status, exit_code, blocking_failures)
  enforce(scorecard)             -> GateOutcome

Design:
  - BLOCKED if any hard gate is evaluated AND failed (takes precedence).
  - INCOMPLETE if all evaluated hard gates passed but some hard gates were not
    evaluated (metric value is None). Unevaluated gates are surfaced in
    GateOutcome.unevaluated_hard; they never block but they do prevent PASS.
  - PASS only when every hard gate was evaluated and passed.
  - Soft gate failures are warnings only — never block.
  - Values are rounded to 3 decimal places in messages for deterministic output.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Literal

from src.config import HARD_GATES, SOFT_GATES


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Outcome for a single gate evaluation."""

    name: str
    kind: Literal["hard", "soft"]
    threshold: float
    op: str                      # ">=" or "<="
    value: float | None
    evaluated: bool
    passed: bool | None          # None when not evaluated
    message: str


@dataclass
class GateOutcome:
    """Full release decision produced by enforce()."""

    status: Literal["PASS", "BLOCKED", "INCOMPLETE"]
    exit_code: int               # 0 = PASS, 1 = BLOCKED, 2 = INCOMPLETE
    hard_results: list[GateResult]
    soft_results: list[GateResult]
    blocking_failures: list[GateResult]
    unevaluated_hard: list[GateResult]
    summary_lines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
}


def _make_message(name: str, value: float | None, op: str, threshold: float) -> str:
    """Build a deterministic gate message (values rounded to 3 dp)."""
    if value is None:
        return f"{name} not evaluated (requires judge/robustness)"
    rounded = round(value, 3)
    rounded_thresh = round(threshold, 3)
    verdict = "OK" if _OPS[op](value, threshold) else "FAIL"
    return f"{name} {rounded} {op} {rounded_thresh} {verdict}"


def _build_gate_result(
    metric: str,
    kind: Literal["hard", "soft"],
    threshold: float,
    op: str,
    metric_summary: dict[str, float | None],
) -> GateResult:
    value = metric_summary.get(metric)  # None if key absent OR explicitly None
    if value is None:
        return GateResult(
            name=metric,
            kind=kind,
            threshold=threshold,
            op=op,
            value=None,
            evaluated=False,
            passed=None,
            message=_make_message(metric, None, op, threshold),
        )
    compare = _OPS[op]
    passed = compare(value, threshold)
    return GateResult(
        name=metric,
        kind=kind,
        threshold=threshold,
        op=op,
        value=value,
        evaluated=True,
        passed=passed,
        message=_make_message(metric, value, op, threshold),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_gates(
    metric_summary: dict[str, float | None],
) -> tuple[list[GateResult], list[GateResult]]:
    """Evaluate all hard and soft gates against the metric_summary.

    Args:
        metric_summary: Mapping of metric name to aggregate value (or None).

    Returns:
        (hard_results, soft_results) — one GateResult per gate, in config order.
    """
    hard_results: list[GateResult] = [
        _build_gate_result(g["metric"], "hard", g["threshold"], g["op"], metric_summary)
        for g in HARD_GATES
    ]
    soft_results: list[GateResult] = [
        _build_gate_result(g["metric"], "soft", g["threshold"], g["op"], metric_summary)
        for g in SOFT_GATES
    ]
    return hard_results, soft_results


def decide_release(
    hard_results: list[GateResult],
) -> tuple[Literal["PASS", "BLOCKED", "INCOMPLETE"], int, list[GateResult]]:
    """Determine the release decision from hard gate results.

    Priority order:
      1. BLOCKED (exit 1)  — any hard gate evaluated AND failed.
      2. INCOMPLETE (exit 2) — no failures but some hard gates were not evaluated.
      3. PASS (exit 0)     — all hard gates evaluated and passed.

    Unevaluated gates (value=None) never block, but they prevent PASS.

    Args:
        hard_results: List of GateResult with kind=="hard".

    Returns:
        (status, exit_code, blocking_failures)
    """
    blocking = [r for r in hard_results if r.evaluated and r.passed is False]
    if blocking:
        return "BLOCKED", 1, blocking
    if any(r for r in hard_results if not r.evaluated):
        return "INCOMPLETE", 2, []
    return "PASS", 0, []


def enforce(scorecard: "Scorecard") -> GateOutcome:  # type: ignore[name-defined]
    """Run gate enforcement and stamp the scorecard.

    Evaluates all gates, decides release status, stamps scorecard.status and
    scorecard.hard_gate_failures, and builds human-readable summary_lines.

    Args:
        scorecard: A Scorecard (status must be "PENDING" before this call;
                   it will be overwritten).

    Returns:
        GateOutcome with full results and summary.
    """
    from src.eval.aggregate import Scorecard  # local import to avoid circular

    hard_results, soft_results = evaluate_gates(scorecard.metric_summary)
    status, exit_code, blocking_failures = decide_release(hard_results)
    unevaluated_hard = [r for r in hard_results if not r.evaluated]

    # --- Stamp scorecard ---
    scorecard.status = status
    scorecard.hard_gate_failures = [r.name for r in blocking_failures]

    # --- Build summary_lines ---
    lines: list[str] = []

    if status == "BLOCKED":
        lines.append("RELEASE BLOCKED")
        for r in blocking_failures:
            rounded_val = round(r.value, 3) if r.value is not None else r.value
            rounded_thresh = round(r.threshold, 3)
            lines.append(
                f"  - {r.name}: {rounded_val} fails {r.op} {rounded_thresh} (hard gate)"
            )
    elif status == "INCOMPLETE":
        lines.append("RELEASE INCOMPLETE")
        n = len(unevaluated_hard)
        metric_names = ", ".join(r.name for r in unevaluated_hard)
        lines.append(
            f"  - {n} hard gate(s) not evaluated"
            f" (need live judge/robustness): {metric_names}"
        )
    else:
        lines.append("RELEASE OK")

    # Warnings: failed soft gates
    failed_soft = [r for r in soft_results if r.passed is False]
    if failed_soft:
        lines.append("WARNINGS:")
        for r in failed_soft:
            rounded_val = round(r.value, 3) if r.value is not None else r.value
            rounded_thresh = round(r.threshold, 3)
            lines.append(
                f"  - {r.name}: {rounded_val} fails {r.op} {rounded_thresh} (soft gate)"
            )

    # Unevaluated hard gates
    if unevaluated_hard:
        lines.append("UNEVALUATED (need live judge/robustness):")
        for r in unevaluated_hard:
            lines.append(f"  - {r.name}")

    return GateOutcome(
        status=status,
        exit_code=exit_code,
        hard_results=hard_results,
        soft_results=soft_results,
        blocking_failures=blocking_failures,
        unevaluated_hard=unevaluated_hard,
        summary_lines=lines,
    )
