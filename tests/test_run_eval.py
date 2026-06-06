"""Tests for T15 — run_eval.py (N-03, make eval end-to-end).

TDD: tests written before the implementation.

Coverage:
  - main([...run_pass...]) returns 2 (INCOMPLETE); scorecard.json and scorecard.html written.
    Programmatic-only replay cannot evaluate judge/robustness hard gates → INCOMPLETE.
    A full live run with judge+robustness results will return 0 (PASS).
  - main([...run_fail...]) returns 1; artifacts written; stdout contains RELEASE BLOCKED.
  - stdout for run_pass contains "RELEASE INCOMPLETE" and dimension/overall line.
  - Subprocess test: uv run python -m src.eval.run_eval --replay run_pass exits 2.
  - Subprocess test: uv run python -m src.eval.run_eval --replay run_fail exits 1.
  - All offline; no network; no API key required.
"""

from __future__ import annotations

import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

DATASETS = Path(__file__).parent.parent / "datasets"
RUN_PASS = DATASETS / "fixtures" / "run_pass.jsonl"
RUN_FAIL = DATASETS / "fixtures" / "run_fail.jsonl"
GOLDEN_SET = DATASETS / "golden_set.jsonl"


# ---------------------------------------------------------------------------
# Helper: run main() capturing stdout
# ---------------------------------------------------------------------------

def _run_main(args: list[str]) -> tuple[int, str]:
    """Call main() with given args and return (exit_code, captured_stdout)."""
    from src.eval.run_eval import main

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        code = main(args)
    finally:
        sys.stdout = old_stdout
    return code, buf.getvalue()


# ---------------------------------------------------------------------------
# Tests: run_pass → exit 2 (INCOMPLETE), artifacts present, stdout RELEASE INCOMPLETE
# ---------------------------------------------------------------------------
# run_pass is a programmatic-only replay fixture. It evaluates negative_rejection
# (passes), but faithfulness, hallucination_rate, and advice_boundary require a live
# LLM judge / robustness run — those hard gates are unevaluated. Because all hard
# gates must be evaluated to PASS, the status is INCOMPLETE (exit 2). A full live
# run providing judge+robustness results will produce PASS (exit 0).

class TestRunPass:
    def test_exit_code_two(self, tmp_path: Path) -> None:
        code, _ = _run_main([
            "--replay", str(RUN_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-pass",
        ])
        assert code == 2, "run_pass should produce exit code 2 (INCOMPLETE)"

    def test_scorecard_json_written(self, tmp_path: Path) -> None:
        _run_main([
            "--replay", str(RUN_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-pass",
        ])
        json_path = tmp_path / "t-pass" / "scorecard.json"
        assert json_path.exists(), "scorecard.json must be written"
        # Must be valid JSON
        data = json.loads(json_path.read_text())
        assert "run_id" in data
        assert data["run_id"] == "t-pass"

    def test_scorecard_html_written(self, tmp_path: Path) -> None:
        _run_main([
            "--replay", str(RUN_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-pass",
        ])
        html_path = tmp_path / "t-pass" / "scorecard.html"
        assert html_path.exists(), "scorecard.html must be written"
        content = html_path.read_text()
        assert len(content) > 0, "scorecard.html must be non-empty"
        assert "<html" in content.lower() or "<!doctype" in content.lower()

    def test_stdout_contains_release_incomplete(self, tmp_path: Path) -> None:
        _, out = _run_main([
            "--replay", str(RUN_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-pass",
        ])
        assert "RELEASE INCOMPLETE" in out, (
            f"Expected 'RELEASE INCOMPLETE' in stdout; got:\n{out}"
        )

    def test_stdout_contains_overall(self, tmp_path: Path) -> None:
        _, out = _run_main([
            "--replay", str(RUN_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-pass",
        ])
        # render_text should include "overall" or "Overall" line
        assert "overall" in out.lower() or "Overall" in out, (
            f"Expected overall score line in stdout; got:\n{out}"
        )

    def test_stdout_contains_dimension(self, tmp_path: Path) -> None:
        _, out = _run_main([
            "--replay", str(RUN_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-pass",
        ])
        # At least one dimension name should appear in output
        assert any(
            dim in out
            for dim in [
                "financial_correctness",
                "faithfulness_grounding",
                "retrieval_quality",
                "safety_compliance",
            ]
        ), f"Expected dimension name in stdout; got:\n{out}"


# ---------------------------------------------------------------------------
# Tests: run_fail → exit 1, artifacts present, stdout RELEASE BLOCKED
# ---------------------------------------------------------------------------

class TestRunFail:
    def test_exit_code_one(self, tmp_path: Path) -> None:
        code, _ = _run_main([
            "--replay", str(RUN_FAIL),
            "--out", str(tmp_path),
            "--run-id", "t-fail",
        ])
        assert code == 1, "run_fail should produce exit code 1 (BLOCKED)"

    def test_stdout_contains_release_blocked(self, tmp_path: Path) -> None:
        _, out = _run_main([
            "--replay", str(RUN_FAIL),
            "--out", str(tmp_path),
            "--run-id", "t-fail",
        ])
        assert "RELEASE BLOCKED" in out, (
            f"Expected 'RELEASE BLOCKED' in stdout; got:\n{out}"
        )

    def test_artifacts_written_on_fail(self, tmp_path: Path) -> None:
        _run_main([
            "--replay", str(RUN_FAIL),
            "--out", str(tmp_path),
            "--run-id", "t-fail",
        ])
        assert (tmp_path / "t-fail" / "scorecard.json").exists()
        assert (tmp_path / "t-fail" / "scorecard.html").exists()

    def test_scorecard_json_valid_on_fail(self, tmp_path: Path) -> None:
        _run_main([
            "--replay", str(RUN_FAIL),
            "--out", str(tmp_path),
            "--run-id", "t-fail",
        ])
        data = json.loads((tmp_path / "t-fail" / "scorecard.json").read_text())
        assert data["status"] == "BLOCKED"


# ---------------------------------------------------------------------------
# Subprocess tests — exercise the real __main__ entry point
# ---------------------------------------------------------------------------

class TestSubprocess:
    def test_subprocess_run_pass_exits_2(self, tmp_path: Path) -> None:
        # Programmatic-only replay → INCOMPLETE (judge/robustness gates not measured)
        result = subprocess.run(
            [
                "uv", "run", "python", "-m", "src.eval.run_eval",
                "--replay", str(RUN_PASS),
                "--out", str(tmp_path),
                "--run-id", "sp-pass",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 2, (
            f"Expected exit 2 (INCOMPLETE); got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert (tmp_path / "sp-pass" / "scorecard.json").exists()

    def test_subprocess_run_fail_exits_1(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                "uv", "run", "python", "-m", "src.eval.run_eval",
                "--replay", str(RUN_FAIL),
                "--out", str(tmp_path),
                "--run-id", "sp-fail",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 1, (
            f"Expected exit 1; got {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "RELEASE BLOCKED" in result.stdout
