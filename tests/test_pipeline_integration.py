"""Integration tests for the wired offline pipeline (programmatic + robustness + judge).

Locks the full hard-gate behaviour:
  - run_pass + judge_pass  -> all 4 hard gates evaluated & pass -> PASS, exit 0
  - run_fail + judge_fail  -> BLOCKED, exit 1 (faithfulness + negative_rejection
                              + hallucination_rate all fail)
  - run_pass, no verdicts  -> INCOMPLETE, exit 2 (judge gates unevaluated)

All offline; no API key, no network.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from src.eval.aggregate import build_scorecard
from src.eval.golden import load_goldens
from src.eval.metrics.judge import score_judge
from src.eval.metrics.programmatic import score_programmatic
from src.eval.metrics.robustness import score_robustness
from src.eval.runner import load_replay
from src.eval.run_eval import main

GOLDEN = "datasets/golden_set.jsonl"
RUN_PASS = "datasets/fixtures/run_pass.jsonl"
RUN_FAIL = "datasets/fixtures/run_fail.jsonl"
JUDGE_PASS = "datasets/fixtures/judge_pass.json"
JUDGE_FAIL = "datasets/fixtures/judge_fail.json"


# --- metric_summary covers all gate metrics -------------------------------


def test_metric_summary_includes_judge_gate_metrics(tmp_path: Path) -> None:
    """hallucination_rate and answer_relevance must reach metric_summary so the
    gates can read them (regression for the dimension-only metric_summary bug)."""
    goldens = load_goldens(GOLDEN)
    records = load_replay(RUN_PASS)
    prog = score_programmatic(records, goldens)
    rob = score_robustness(records, goldens)
    judge = score_judge(records, goldens, mode="offline", verdicts_path=JUDGE_PASS)

    sc = build_scorecard(
        records,
        goldens,
        prog_results=prog,
        judge_results=judge,
        robustness_results=rob,
        run_id="t",
        mode="replay",
    )
    for metric in ("hallucination_rate", "answer_relevance", "faithfulness",
                   "advice_boundary", "negative_rejection"):
        assert metric in sc.metric_summary
        assert sc.metric_summary[metric] is not None


# --- full PASS -------------------------------------------------------------


def test_full_gate_run_pass_is_pass(tmp_path: Path) -> None:
    out = io.StringIO()
    with redirect_stdout(out):
        code = main([
            "--replay", RUN_PASS, "--verdicts", JUDGE_PASS,
            "--out", str(tmp_path), "--run-id", "p",
        ])
    text = out.getvalue()
    assert code == 0
    assert "RELEASE OK" in text
    # All four hard gates evaluated → nothing in the UNEVALUATED section.
    assert "UNEVALUATED" not in text


# --- full BLOCKED ----------------------------------------------------------


def test_full_gate_run_fail_is_blocked(tmp_path: Path) -> None:
    out = io.StringIO()
    with redirect_stdout(out):
        code = main([
            "--replay", RUN_FAIL, "--verdicts", JUDGE_FAIL,
            "--out", str(tmp_path), "--run-id", "f",
        ])
    text = out.getvalue()
    assert code == 1
    assert "RELEASE BLOCKED" in text
    # The judge corroborates the programmatic block — all three hard gates fail.
    assert "faithfulness" in text
    assert "negative_rejection" in text
    assert "hallucination_rate" in text


# --- INCOMPLETE (no verdicts) ---------------------------------------------


def test_no_verdicts_is_incomplete(tmp_path: Path) -> None:
    out = io.StringIO()
    with redirect_stdout(out):
        code = main([
            "--replay", RUN_PASS,
            "--out", str(tmp_path), "--run-id", "i",
        ])
    text = out.getvalue()
    assert code == 2
    assert "RELEASE INCOMPLETE" in text
    # advice_boundary is now evaluated (robustness always runs); only the two
    # judge hard gates remain unevaluated.
    assert "faithfulness" in text
    assert "hallucination_rate" in text
