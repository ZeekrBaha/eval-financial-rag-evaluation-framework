"""Tests for T11 — LLM-as-judge metrics (E-07).

TDD: tests written before the implementation.

Coverage:
  - Offline score_judge on run_pass + judge_pass.json → 3 metrics per item;
    aggregate faithfulness == 1.0 (≥0.95), hallucination_rate == 0.0 (≤0.01),
    answer_relevance ≥0.90.
  - Offline on run_fail + judge_fail.json → aggregate faithfulness < 0.95 AND
    hallucination_rate > 0.01 (judge corroborates the block).
  - Missing verdict for an id → ValueError naming the id.
  - Unit: verdict {faithfulness:0.9} → faithfulness MetricResult passed=False;
    hallucination 1 → hallucination_rate MetricResult score=1.0 passed=False.
  - Coverage test: verdict fixtures cover ALL ids in their run fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.golden import GoldenItem, Bucket
from src.eval.runner import RunRecord, RetrievedRef, load_replay
from src.eval.metrics.judge import score_judge
from src.eval.metrics.programmatic import (
    aggregate_metric,
)

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

DATASETS = Path(__file__).parent.parent / "datasets"
GOLDEN_SET = DATASETS / "golden_set.jsonl"
RUN_PASS = DATASETS / "fixtures" / "run_pass.jsonl"
RUN_FAIL = DATASETS / "fixtures" / "run_fail.jsonl"
JUDGE_PASS = DATASETS / "fixtures" / "judge_pass.json"
JUDGE_FAIL = DATASETS / "fixtures" / "judge_fail.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ref(chunk_id: str = "chunk#0", text: str = "Some retrieved text.") -> RetrievedRef:
    return RetrievedRef(
        chunk_id=chunk_id,
        text=text,
        similarity=0.9,
        issuer="Acme Corp",
        form="10-K",
        filing_date="2024-01-01",
        accession="ACM-10K-2024",
        section="item7",
        source_url="https://example.invalid/test",
    )


def _make_record(
    id: str,
    answer: str = "The revenue was $100 million.",
    question: str = "What was revenue?",
    bucket: str = "factual_lookup",
) -> RunRecord:
    ref = _make_ref()
    return RunRecord(
        id=id,
        bucket=bucket,
        question=question,
        answer=answer,
        retrieved=[ref],
        citations={"c1": "chunk#0"},
        unmatched_citations=[],
        latency_ms=0,
        mode="replay",
    )


def _make_golden(id: str, bucket: str = "factual_lookup") -> GoldenItem:
    return GoldenItem(
        id=id,
        bucket=Bucket(bucket),
        question="What was revenue?",
        reference_answer="The revenue was $100 million.",
        numeric_answers=[],
        expected_sources=[],
        must_refuse=False,
        injection=None,
        advice_boundary=False,
    )


def _load_goldens(path: Path) -> list[GoldenItem]:
    from src.eval.golden import load_goldens
    return load_goldens(str(path))


# ---------------------------------------------------------------------------
# Coverage test: verdict fixtures must cover ALL ids in their run fixtures
# ---------------------------------------------------------------------------


def test_judge_pass_covers_all_run_pass_ids() -> None:
    """judge_pass.json must have an entry for every id in run_pass.jsonl."""
    records = load_replay(RUN_PASS)
    verdicts = json.loads(JUDGE_PASS.read_text())
    run_ids = {r.id for r in records}
    verdict_ids = set(verdicts.keys())
    missing = run_ids - verdict_ids
    assert not missing, f"judge_pass.json missing ids: {sorted(missing)}"


def test_judge_fail_covers_all_run_fail_ids() -> None:
    """judge_fail.json must have an entry for every id in run_fail.jsonl."""
    records = load_replay(RUN_FAIL)
    verdicts = json.loads(JUDGE_FAIL.read_text())
    run_ids = {r.id for r in records}
    verdict_ids = set(verdicts.keys())
    missing = run_ids - verdict_ids
    assert not missing, f"judge_fail.json missing ids: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Unit tests: individual verdict → MetricResult behaviour
# ---------------------------------------------------------------------------


def test_unit_faithfulness_below_threshold_is_not_passed(tmp_path: Path) -> None:
    """faithfulness score 0.9 is below the 0.95 threshold → passed=False."""
    verdicts = {"unit-001": {"faithfulness": 0.9, "answer_relevance": 1.0, "hallucination": 0}}
    vfile = tmp_path / "v.json"
    vfile.write_text(json.dumps(verdicts))

    record = _make_record("unit-001")
    golden = _make_golden("unit-001")

    results = score_judge([record], [golden], mode="offline", verdicts_path=vfile)
    faith_result = next(r for r in results if r.metric == "faithfulness")
    assert faith_result.score == pytest.approx(0.9)
    assert faith_result.passed is False


def test_unit_faithfulness_below_threshold(tmp_path: Path) -> None:
    """faithfulness score 0.9 → passed=False (gate is ≥0.95)."""
    verdicts = {"unit-low-faith": {"faithfulness": 0.9, "answer_relevance": 1.0, "hallucination": 0}}
    vfile = tmp_path / "v.json"
    vfile.write_text(json.dumps(verdicts))

    record = _make_record("unit-low-faith")
    golden = _make_golden("unit-low-faith")

    results = score_judge([record], [golden], mode="offline", verdicts_path=vfile)
    faith_result = next(r for r in results if r.metric == "faithfulness")
    assert faith_result.score == pytest.approx(0.9)
    assert faith_result.passed is False
    assert faith_result.applicable is True


def test_unit_hallucination_flag_is_score_1_and_fails(tmp_path: Path) -> None:
    """hallucination=1 → hallucination_rate MetricResult score=1.0, passed=False."""
    verdicts = {"unit-halluc": {"faithfulness": 1.0, "answer_relevance": 1.0, "hallucination": 1}}
    vfile = tmp_path / "v.json"
    vfile.write_text(json.dumps(verdicts))

    record = _make_record("unit-halluc")
    golden = _make_golden("unit-halluc")

    results = score_judge([record], [golden], mode="offline", verdicts_path=vfile)
    halluc_result = next(r for r in results if r.metric == "hallucination_rate")
    assert halluc_result.score == pytest.approx(1.0)
    assert halluc_result.passed is False
    assert halluc_result.applicable is True


def test_unit_clean_verdict_passes_all(tmp_path: Path) -> None:
    """A clean verdict (faithfulness=1.0, answer_relevance=1.0, hallucination=0) → all passed."""
    verdicts = {"unit-clean": {"faithfulness": 1.0, "answer_relevance": 1.0, "hallucination": 0}}
    vfile = tmp_path / "v.json"
    vfile.write_text(json.dumps(verdicts))

    record = _make_record("unit-clean")
    golden = _make_golden("unit-clean")

    results = score_judge([record], [golden], mode="offline", verdicts_path=vfile)
    assert len(results) == 3
    assert all(r.passed for r in results)
    assert all(r.applicable for r in results)


def test_unit_answer_relevance_at_threshold(tmp_path: Path) -> None:
    """answer_relevance=0.90 is exactly at gate → passed=True."""
    verdicts = {"unit-ar": {"faithfulness": 1.0, "answer_relevance": 0.90, "hallucination": 0}}
    vfile = tmp_path / "v.json"
    vfile.write_text(json.dumps(verdicts))

    record = _make_record("unit-ar")
    golden = _make_golden("unit-ar")

    results = score_judge([record], [golden], mode="offline", verdicts_path=vfile)
    ar_result = next(r for r in results if r.metric == "answer_relevance")
    assert ar_result.score == pytest.approx(0.90)
    assert ar_result.passed is True


def test_unit_answer_relevance_below_threshold(tmp_path: Path) -> None:
    """answer_relevance=0.89 is below gate → passed=False."""
    verdicts = {"unit-ar-low": {"faithfulness": 1.0, "answer_relevance": 0.89, "hallucination": 0}}
    vfile = tmp_path / "v.json"
    vfile.write_text(json.dumps(verdicts))

    record = _make_record("unit-ar-low")
    golden = _make_golden("unit-ar-low")

    results = score_judge([record], [golden], mode="offline", verdicts_path=vfile)
    ar_result = next(r for r in results if r.metric == "answer_relevance")
    assert ar_result.passed is False


# ---------------------------------------------------------------------------
# Missing verdict → ValueError naming the id
# ---------------------------------------------------------------------------


def test_missing_verdict_raises_value_error_naming_id(tmp_path: Path) -> None:
    """If a record id is absent from the verdict fixture, raise ValueError naming the id."""
    verdicts = {"other-id": {"faithfulness": 1.0, "answer_relevance": 1.0, "hallucination": 0}}
    vfile = tmp_path / "v.json"
    vfile.write_text(json.dumps(verdicts))

    record = _make_record("missing-id")
    golden = _make_golden("missing-id")

    with pytest.raises(ValueError, match="missing-id"):
        score_judge([record], [golden], mode="offline", verdicts_path=vfile)


def test_missing_verdict_does_not_silently_skip(tmp_path: Path) -> None:
    """Two records: one present, one missing → raises on missing, not partial results."""
    verdicts = {"present-id": {"faithfulness": 1.0, "answer_relevance": 1.0, "hallucination": 0}}
    vfile = tmp_path / "v.json"
    vfile.write_text(json.dumps(verdicts))

    records = [_make_record("present-id"), _make_record("absent-id")]
    goldens = [_make_golden("present-id"), _make_golden("absent-id")]

    with pytest.raises(ValueError, match="absent-id"):
        score_judge(records, goldens, mode="offline", verdicts_path=vfile)


# ---------------------------------------------------------------------------
# Integration: run_pass + judge_pass → gates pass
# ---------------------------------------------------------------------------


def test_score_judge_pass_returns_three_results_per_item() -> None:
    """score_judge emits exactly 3 MetricResults per record."""
    records = load_replay(RUN_PASS)
    goldens = _load_goldens(GOLDEN_SET)

    results = score_judge(records, goldens, mode="offline", verdicts_path=JUDGE_PASS)
    assert len(results) == len(records) * 3


def test_score_judge_pass_faithfulness_gate() -> None:
    """Aggregate faithfulness over run_pass is ≥0.95 (hard gate)."""
    records = load_replay(RUN_PASS)
    goldens = _load_goldens(GOLDEN_SET)

    results = score_judge(records, goldens, mode="offline", verdicts_path=JUDGE_PASS)
    agg = aggregate_metric(results, "faithfulness")
    assert agg is not None
    assert agg >= 0.95, f"Expected faithfulness ≥0.95, got {agg}"


def test_score_judge_pass_hallucination_rate_gate() -> None:
    """Aggregate hallucination_rate over run_pass is ≤0.01 (hard gate)."""
    records = load_replay(RUN_PASS)
    goldens = _load_goldens(GOLDEN_SET)

    results = score_judge(records, goldens, mode="offline", verdicts_path=JUDGE_PASS)
    agg = aggregate_metric(results, "hallucination_rate")
    assert agg is not None
    assert agg <= 0.01, f"Expected hallucination_rate ≤0.01, got {agg}"


def test_score_judge_pass_answer_relevance_gate() -> None:
    """Aggregate answer_relevance over run_pass is ≥0.90 (soft gate)."""
    records = load_replay(RUN_PASS)
    goldens = _load_goldens(GOLDEN_SET)

    results = score_judge(records, goldens, mode="offline", verdicts_path=JUDGE_PASS)
    agg = aggregate_metric(results, "answer_relevance")
    assert agg is not None
    assert agg >= 0.90, f"Expected answer_relevance ≥0.90, got {agg}"


# ---------------------------------------------------------------------------
# Integration: run_fail + judge_fail → judge corroborates failure
# ---------------------------------------------------------------------------


def test_score_judge_fail_faithfulness_below_gate() -> None:
    """Aggregate faithfulness over run_fail is <0.95 (judge corroborates block)."""
    records = load_replay(RUN_FAIL)
    goldens = _load_goldens(GOLDEN_SET)

    results = score_judge(records, goldens, mode="offline", verdicts_path=JUDGE_FAIL)
    agg = aggregate_metric(results, "faithfulness")
    assert agg is not None
    assert agg < 0.95, f"Expected faithfulness <0.95, got {agg}"


def test_score_judge_fail_hallucination_rate_above_gate() -> None:
    """Aggregate hallucination_rate over run_fail is >0.01 (judge corroborates block)."""
    records = load_replay(RUN_FAIL)
    goldens = _load_goldens(GOLDEN_SET)

    results = score_judge(records, goldens, mode="offline", verdicts_path=JUDGE_FAIL)
    agg = aggregate_metric(results, "hallucination_rate")
    assert agg is not None
    assert agg > 0.01, f"Expected hallucination_rate >0.01, got {agg}"


def test_score_judge_fail_returns_three_results_per_item() -> None:
    """score_judge on run_fail emits exactly 3 MetricResults per record."""
    records = load_replay(RUN_FAIL)
    goldens = _load_goldens(GOLDEN_SET)

    results = score_judge(records, goldens, mode="offline", verdicts_path=JUDGE_FAIL)
    assert len(results) == len(records) * 3
