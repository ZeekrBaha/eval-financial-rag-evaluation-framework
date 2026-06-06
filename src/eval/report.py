"""report.py — Auto-generate REPORT.md from a completed evaluation run.

Public surface:
  render_report(scorecard, outcome, goldens, metric_results) -> str
  write_report(scorecard, outcome, goldens, metric_results, path) -> None
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.config import HARD_GATES, SOFT_GATES

if TYPE_CHECKING:
    from src.eval.aggregate import Scorecard
    from src.eval.gates import GateOutcome
    from src.eval.golden import GoldenItem
    from src.eval.metrics.programmatic import MetricResult


# Status label mapped to exit code for human clarity.
_STATUS_LABEL: dict[str, str] = {
    "PASS": "RELEASE OK",
    "BLOCKED": "RELEASE BLOCKED",
    "INCOMPLETE": "RELEASE INCOMPLETE",
}

# Soft retrieval metrics excluded from per-item gate check (mirrors aggregate.py logic).
_SOFT_RETRIEVAL_METRICS: frozenset[str] = frozenset({
    "context_precision",
    "context_recall",
})


def _gate_info(metric: str) -> tuple[str, float] | None:
    """Return (op, threshold) for a metric from HARD_GATES + SOFT_GATES, or None."""
    for g in HARD_GATES:
        if g["metric"] == metric:
            return g["op"], g["threshold"]
    for g in SOFT_GATES:
        if g["metric"] == metric:
            return g["op"], g["threshold"]
    return None


def _gate_kind(metric: str) -> str:
    """Return 'Hard', 'Soft', or '—' for a metric."""
    for g in HARD_GATES:
        if g["metric"] == metric:
            return "Hard"
    for g in SOFT_GATES:
        if g["metric"] == metric:
            return "Soft"
    return "—"


def render_report(
    scorecard: "Scorecard",
    outcome: "GateOutcome",
    goldens: list["GoldenItem"],
    metric_results: list["MetricResult"],
) -> str:
    """Build a Markdown evaluation report string.

    Sections:
      1. Title + mode + Status
      2. Release decision (summary_lines as bullets)
      3. Dimensions table
      4. Per-bucket pass rate table
      5. Metric summary table (with gate info from config)
      6. Findings — failing items
      7. Notes / honesty footer

    Args:
        scorecard:      Completed Scorecard (status already stamped by enforce()).
        outcome:        GateOutcome from enforce().
        goldens:        List of GoldenItems for this run.
        metric_results: Combined list of all MetricResults (prog + rob + judge).

    Returns:
        Markdown string.
    """
    lines: list[str] = []

    status_label = _STATUS_LABEL.get(scorecard.status, scorecard.status)
    exit_code = outcome.exit_code

    # -------------------------------------------------------------------------
    # 1. Title + Status
    # -------------------------------------------------------------------------
    lines.append(f"# Evaluation Report — {scorecard.run_id}")
    lines.append("")
    lines.append(f"**Mode:** {scorecard.mode}")
    lines.append(f"**Status:** {status_label} (exit {exit_code})")
    lines.append("")

    # -------------------------------------------------------------------------
    # 2. Release decision
    # -------------------------------------------------------------------------
    lines.append("## Release decision")
    lines.append("")
    lines.append("```")
    for sl in outcome.summary_lines:
        lines.append(sl)
    lines.append("```")
    lines.append("")

    # -------------------------------------------------------------------------
    # 3. Dimensions table
    # -------------------------------------------------------------------------
    lines.append("## Dimensions")
    lines.append("")
    lines.append("| Dimension | Weight | Score | Status |")
    lines.append("| --- | --- | --- | --- |")
    for dim in scorecard.dimensions:
        score_str = f"{dim.score:.1f}" if dim.score is not None else "—"
        lines.append(f"| {dim.name} | {dim.weight} | {score_str} | {dim.status} |")
    overall_str = f"{scorecard.overall:.1f}" if scorecard.overall is not None else "—"
    lines.append(f"| **Overall** | — | **{overall_str}** | **{scorecard.status}** |")
    lines.append("")

    # -------------------------------------------------------------------------
    # 4. Per-bucket pass rate
    # -------------------------------------------------------------------------
    lines.append("## Per-bucket pass rate")
    lines.append("")
    lines.append("| Bucket | Pass rate |")
    lines.append("| --- | --- |")
    for bucket, rate in scorecard.buckets.items():
        rate_str = f"{rate:.0%}" if rate is not None else "—"
        lines.append(f"| {bucket} | {rate_str} |")
    lines.append("")

    # -------------------------------------------------------------------------
    # 5. Metric summary
    # -------------------------------------------------------------------------
    lines.append("## Metric summary")
    lines.append("")
    lines.append("| Metric | Value | Gate | Hard/Soft | Pass/Fail/NA |")
    lines.append("| --- | --- | --- | --- | --- |")
    for metric, value in sorted(scorecard.metric_summary.items()):
        gate_detail = _gate_info(metric)
        kind = _gate_kind(metric)
        if gate_detail is not None:
            op, threshold = gate_detail
            gate_str = f"{op} {threshold}"
        else:
            gate_str = "—"
            kind = "—"

        if value is None:
            value_str = "—"
            pass_str = "NA"
        else:
            value_str = f"{value:.4f}"
            if gate_detail is not None:
                import operator as _op_mod
                _ops = {">=": _op_mod.ge, "<=": _op_mod.le}
                op_str, threshold_val = gate_detail
                passed = _ops[op_str](value, threshold_val)
                pass_str = "Pass" if passed else "Fail"
            else:
                pass_str = "—"

        lines.append(f"| {metric} | {value_str} | {gate_str} | {kind} | {pass_str} |")
    lines.append("")

    # -------------------------------------------------------------------------
    # 6. Findings — items failing at least one applicable gate metric
    # -------------------------------------------------------------------------
    lines.append("## Findings")
    lines.append("")

    # Build a lookup: item_id -> bucket from goldens
    item_bucket: dict[str, str] = {g.id: g.bucket.value for g in goldens}

    # Build gate metric set (hard + soft, excluding soft retrieval per-item)
    gate_metrics: set[str] = (
        {g["metric"] for g in HARD_GATES} | {g["metric"] for g in SOFT_GATES}
    ) - _SOFT_RETRIEVAL_METRICS

    # Collect failures: {item_id -> list of (metric, detail)}
    item_failures: dict[str, list[tuple[str, str]]] = {}
    for mr in metric_results:
        if not mr.applicable:
            continue
        if mr.metric not in gate_metrics:
            continue
        if not mr.passed:
            item_failures.setdefault(mr.item_id, []).append((mr.metric, mr.detail))

    if not item_failures:
        lines.append("All evaluated items passed their gate metrics.")
    else:
        for item_id in sorted(item_failures.keys()):
            bucket = item_bucket.get(item_id, "unknown")
            for metric, detail in item_failures[item_id]:
                lines.append(f"- {item_id} ({bucket}): {metric} failed — {detail}")
    lines.append("")

    # -------------------------------------------------------------------------
    # 7. Notes
    # -------------------------------------------------------------------------
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **Proposed gates**: All gate thresholds are proposed values from `config.py`"
        " and have not been calibrated against an analyst baseline."
    )
    lines.append(
        "- **Judge UNCALIBRATED**: LLM-judge metrics (faithfulness, hallucination_rate,"
        " answer_relevance, advice_boundary) have not been validated against human labels."
    )
    lines.append(
        "- **Fictional data**: Golden-set questions reference synthetic filings"
        " (Northwind Motors, Cascade Semiconductor, etc.) — not real SEC filings."
    )
    lines.append(
        "- **INCOMPLETE meaning**: Status INCOMPLETE means one or more hard gates were"
        " not evaluated (judge/robustness results unavailable). It does NOT mean the"
        " system performed poorly — it means the decision cannot yet be made."
    )
    lines.append("")

    return "\n".join(lines)


def write_report(
    scorecard: "Scorecard",
    outcome: "GateOutcome",
    goldens: list["GoldenItem"],
    metric_results: list["MetricResult"],
    path: Path | str,
) -> None:
    """Render the Markdown report and write it to *path*.

    Args:
        scorecard:      Completed Scorecard.
        outcome:        GateOutcome from enforce().
        goldens:        List of GoldenItems.
        metric_results: Combined MetricResults (prog + rob + judge).
        path:           Destination path for REPORT.md.
    """
    content = render_report(scorecard, outcome, goldens, metric_results)
    Path(path).write_text(content, encoding="utf-8")
