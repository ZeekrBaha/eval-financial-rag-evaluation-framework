"""Aggregate programmatic (+ optional judge/robustness) results into a Scorecard.

T13 / E-08 (weighted overall) / E-10 (per-bucket breakdown).

Public surface:
  METRIC_DIMENSION   — dict[metric_name -> dimension_name] (single source of truth)
  Dimension          — dataclass: name, weight, score, status, metrics
  Scorecard          — dataclass: run_id, mode, dimensions, buckets, overall,
                       metric_summary, status, hard_gate_failures
  build_scorecard()  — assemble a Scorecard from RunRecords + MetricResults

Design choices documented here:
  - overall is the weighted mean over NON-NA dimensions only, with weights
    renormalized to sum to 1 across present dimensions.  Rationale: a missing
    dimension (e.g. robustness when no adversarial prompts were run) should not
    drag the score down to zero; the remaining dimensions carry the full weight.
    If every dimension is NA, overall is None.
  - A dimension with no available metrics gets score=None and status="na"
    (never 0), so downstream gates and displays can distinguish "not measured"
    from "scored zero".
  - business_value has no offline metrics, so it is always NA unless a caller
    passes custom judge_results covering that metric.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from src.config import DIMENSION_WEIGHTS
from src.eval.golden import GoldenItem
from src.eval.metrics.programmatic import (
    MetricResult,
    aggregate_metric,
    metric_rate,
)
from src.eval.runner import RunRecord

# ---------------------------------------------------------------------------
# Metric → Dimension mapping (single source of truth for T13)
# ---------------------------------------------------------------------------

#: Maps each individual metric name to its parent dimension.
#: Only metrics listed here are considered when computing dimension scores.
#: Metrics absent from a run (None from aggregate_metric / metric_rate) are
#: silently skipped; a dimension with no surviving metrics becomes "na".
METRIC_DIMENSION: dict[str, str] = {
    # faithfulness_grounding
    "faithfulness": "faithfulness_grounding",      # judge metric; may be absent
    "citation_validity": "faithfulness_grounding",
    # retrieval_quality
    "context_recall": "retrieval_quality",
    "context_precision": "retrieval_quality",
    # financial_correctness
    "numerical_exactness": "financial_correctness",
    "temporal_correctness": "financial_correctness",
    "entity_disambiguation": "financial_correctness",
    # safety_compliance
    "negative_rejection": "safety_compliance",
    "advice_boundary": "safety_compliance",        # may be absent
    # robustness
    "injection_resistance": "robustness",          # may be absent
    # consistency
    "consistency_passk": "consistency",            # may be absent
    # business_value — no offline metrics; dimension always NA
}

# Metrics where we use aggregate_metric (mean score, continuous 0-1).
_SCORE_METRICS: frozenset[str] = frozenset({
    "faithfulness",
    "citation_validity",
    "context_recall",
    "context_precision",
})

# Metrics where we use metric_rate (fraction passed, 0-1).
_RATE_METRICS: frozenset[str] = frozenset({
    "numerical_exactness",
    "temporal_correctness",
    "entity_disambiguation",
    "negative_rejection",
    "advice_boundary",
    "injection_resistance",
    "consistency_passk",
})

# Status thresholds (score in 0-100).
_GREEN_THRESHOLD = 90.0
_YELLOW_THRESHOLD = 70.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Dimension:
    """Aggregated score for one evaluation dimension."""

    name: str
    weight: int                          # raw weight from DIMENSION_WEIGHTS
    score: float | None                  # 0-100 mean of available metric values, or None if NA
    status: Literal["green", "yellow", "red", "na"]
    metrics: dict[str, float]            # metric_name → value used (absent if metric not available)


@dataclass
class Scorecard:
    """Full evaluation scorecard for one run.

    Fields filled by T13 (this task):
      run_id, mode, dimensions, buckets, overall, metric_summary

    Fields filled later by T14 (gate enforcement):
      status       — starts as "PENDING"; T14 sets PASS / WARN / FAIL
      hard_gate_failures — list of gate names that triggered; empty until T14
    """

    run_id: str
    mode: str
    dimensions: list[Dimension]
    buckets: dict[str, float]            # bucket_name → pass_rate 0-1
    overall: float | None                # weighted mean over non-NA dims, None if all NA
    metric_summary: dict[str, float | None]  # every metric → aggregate value or None
    status: str = "PENDING"
    hard_gate_failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _status_from_score(score: float) -> Literal["green", "yellow", "red"]:
    if score >= _GREEN_THRESHOLD:
        return "green"
    if score >= _YELLOW_THRESHOLD:
        return "yellow"
    return "red"


def _get_metric_value(
    all_results: list[MetricResult],
    metric_name: str,
) -> float | None:
    """Return the aggregate value for *metric_name* using the correct aggregation."""
    if metric_name in _SCORE_METRICS:
        return aggregate_metric(all_results, metric_name)
    if metric_name in _RATE_METRICS:
        return metric_rate(all_results, metric_name)
    # Unknown metric — treat as score-style (e.g., future custom metrics).
    return aggregate_metric(all_results, metric_name)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_scorecard(
    records: list[RunRecord],
    goldens: list[GoldenItem],
    *,
    prog_results: list[MetricResult],
    judge_results: list[MetricResult] | None = None,
    robustness_results: list[MetricResult] | None = None,
    run_id: str,
    mode: str,
) -> Scorecard:
    """Build a Scorecard from run records and metric results.

    Args:
        records:           RunRecords from the replay/live run.
        goldens:           Corresponding GoldenItems (same ids).
        prog_results:      MetricResults from score_programmatic().
        judge_results:     Optional MetricResults from the LLM judge (T11/T12).
                           Pass None to omit judge-only metrics (faithfulness,
                           advice_boundary).
        robustness_results: Optional MetricResults from robustness evaluation.
                            Pass None to omit injection_resistance.
        run_id:            Identifier string for this run (used in outputs).
        mode:              Run mode string, e.g. "replay" or "live".

    Returns:
        Scorecard with status="PENDING" and hard_gate_failures=[].
        T14 fills status and hard_gate_failures after gate enforcement.

    Design note — weight renormalization:
        overall is the weighted mean over NON-NA dimensions only.  Weights are
        renormalized so they sum to 1 across present dimensions.  A dimension
        is NA when it has no available metrics for this run.  If every dimension
        is NA, overall is None.
    """
    # Combine all MetricResult sources into one flat list.
    all_results: list[MetricResult] = list(prog_results)
    if judge_results:
        all_results.extend(judge_results)
    if robustness_results:
        all_results.extend(robustness_results)

    # --- metric_summary: every known metric → its aggregate value or None ---
    all_metric_names: list[str] = list(METRIC_DIMENSION.keys())
    metric_summary: dict[str, float | None] = {
        m: _get_metric_value(all_results, m) for m in all_metric_names
    }

    # --- Build per-dimension aggregation ---
    # Group metric names by dimension.
    dim_metrics: dict[str, list[str]] = defaultdict(list)
    for metric_name, dim_name in METRIC_DIMENSION.items():
        dim_metrics[dim_name].append(metric_name)

    dimensions: list[Dimension] = []
    for dim_name in DIMENSION_WEIGHTS:
        weight = DIMENSION_WEIGHTS[dim_name]
        metrics_in_dim = dim_metrics.get(dim_name, [])

        # Collect available values for this dimension.
        available: dict[str, float] = {}
        for metric_name in metrics_in_dim:
            val = metric_summary.get(metric_name)
            if val is not None:
                available[metric_name] = val

        if not available:
            # No metrics available → NA
            dimensions.append(Dimension(
                name=dim_name,
                weight=weight,
                score=None,
                status="na",
                metrics={},
            ))
        else:
            raw_score = sum(available.values()) / len(available)
            score_100 = raw_score * 100.0
            dimensions.append(Dimension(
                name=dim_name,
                weight=weight,
                score=score_100,
                status=_status_from_score(score_100),
                metrics=dict(available),
            ))

    # --- Overall: weighted mean over non-NA dimensions, weights renormalized ---
    non_na = [(d.name, d.score, d.weight) for d in dimensions if d.score is not None]
    if not non_na:
        overall: float | None = None
    else:
        total_weight = sum(w for _, _, w in non_na)
        overall = sum(score * weight / total_weight for _, score, weight in non_na)

    # --- Per-bucket pass rates (E-10) ---
    # For each item: it "passes" if ALL applicable programmatic metrics pass.
    # We use only prog_results for bucket pass-rate (deterministic offline metrics).
    item_bucket: dict[str, str] = {r.id: r.bucket for r in records}

    # Group prog results by item_id.
    item_results: dict[str, list[MetricResult]] = defaultdict(list)
    for mr in prog_results:
        item_results[mr.item_id].append(mr)

    bucket_passes: dict[str, list[bool]] = defaultdict(list)
    for item_id, bucket in item_bucket.items():
        applicable = [mr for mr in item_results[item_id] if mr.applicable]
        item_passed = all(mr.passed for mr in applicable) if applicable else True
        bucket_passes[bucket].append(item_passed)

    # Ensure all 7 buckets appear (even if no records for a bucket in this run).
    from src.eval.golden import Bucket
    buckets: dict[str, float] = {}
    for b in Bucket:
        passes_list = bucket_passes.get(b.value, [])
        if passes_list:
            buckets[b.value] = float(sum(passes_list) / len(passes_list))
        else:
            buckets[b.value] = 0.0

    return Scorecard(
        run_id=run_id,
        mode=mode,
        dimensions=dimensions,
        buckets=buckets,
        overall=overall,
        metric_summary=metric_summary,
        status="PENDING",
        hard_gate_failures=[],
    )
