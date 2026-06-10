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

Extended coverage (logging + REPORT.md):
  - After main() on run_pass+verdicts: run.log and REPORT.md exist and are non-empty.
  - stdout still contains RELEASE OK (scorecard not moved off stdout by logging).
  - --verbose flag runs without error.
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
JUDGE_PASS = DATASETS / "fixtures" / "judge_pass.json"
JUDGE_FAIL = DATASETS / "fixtures" / "judge_fail.json"


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


# ---------------------------------------------------------------------------
# Tests: logging layer (run.log) and REPORT.md generation
# ---------------------------------------------------------------------------

class TestLoggingAndReport:
    """Verify run.log and REPORT.md are written and stdout is unaffected."""

    def test_run_log_exists_after_pass_with_verdicts(self, tmp_path: Path) -> None:
        """run.log must be created in the run dir."""
        _run_main([
            "--replay", str(RUN_PASS),
            "--verdicts", str(JUDGE_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-log-pass",
        ])
        log_path = tmp_path / "t-log-pass" / "run.log"
        assert log_path.exists(), "run.log must be created by the pipeline"

    def test_run_log_is_non_empty(self, tmp_path: Path) -> None:
        """run.log must contain at least one log entry."""
        _run_main([
            "--replay", str(RUN_PASS),
            "--verdicts", str(JUDGE_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-log-nonempty",
        ])
        log_path = tmp_path / "t-log-nonempty" / "run.log"
        assert log_path.stat().st_size > 0, "run.log must not be empty"

    def test_run_log_contains_stage_entries(self, tmp_path: Path) -> None:
        """run.log must mention pipeline stage names."""
        _run_main([
            "--replay", str(RUN_PASS),
            "--verdicts", str(JUDGE_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-log-stages",
        ])
        content = (tmp_path / "t-log-stages" / "run.log").read_text()
        assert "load_goldens" in content
        assert "score_programmatic" in content

    def test_report_md_exists_after_pass_with_verdicts(self, tmp_path: Path) -> None:
        """REPORT.md must be created in the run dir."""
        _run_main([
            "--replay", str(RUN_PASS),
            "--verdicts", str(JUDGE_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-rpt-pass",
        ])
        report_path = tmp_path / "t-rpt-pass" / "REPORT.md"
        assert report_path.exists(), "REPORT.md must be created by the pipeline"

    def test_report_md_contains_status(self, tmp_path: Path) -> None:
        """REPORT.md must contain the release status string."""
        _run_main([
            "--replay", str(RUN_PASS),
            "--verdicts", str(JUDGE_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-rpt-status",
        ])
        content = (tmp_path / "t-rpt-status" / "REPORT.md").read_text()
        # run_pass + judge_pass should yield PASS or INCOMPLETE
        assert "RELEASE" in content, f"Expected RELEASE in REPORT.md; got:\n{content[:500]}"

    def test_report_md_for_fail_contains_blocked(self, tmp_path: Path) -> None:
        """REPORT.md for a failing run must show RELEASE BLOCKED."""
        _run_main([
            "--replay", str(RUN_FAIL),
            "--verdicts", str(JUDGE_FAIL),
            "--out", str(tmp_path),
            "--run-id", "t-rpt-fail",
        ])
        content = (tmp_path / "t-rpt-fail" / "REPORT.md").read_text()
        assert "RELEASE BLOCKED" in content

    def test_stdout_still_contains_scorecard_on_pass(self, tmp_path: Path) -> None:
        """Logging must not remove scorecard text from stdout."""
        _, out = _run_main([
            "--replay", str(RUN_PASS),
            "--verdicts", str(JUDGE_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-stdout-pass",
        ])
        # The scorecard text and gate summary must still appear on stdout.
        assert "RELEASE" in out, f"Expected RELEASE in stdout; got:\n{out}"

    def test_stdout_still_contains_scorecard_on_fail(self, tmp_path: Path) -> None:
        """stdout must still contain RELEASE BLOCKED for a failing run."""
        _, out = _run_main([
            "--replay", str(RUN_FAIL),
            "--out", str(tmp_path),
            "--run-id", "t-stdout-fail",
        ])
        assert "RELEASE BLOCKED" in out

    def test_verbose_flag_exits_same_code(self, tmp_path: Path) -> None:
        """--verbose must not change the exit code."""
        code_normal, _ = _run_main([
            "--replay", str(RUN_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-verb-base",
        ])
        code_verbose, _ = _run_main([
            "--replay", str(RUN_PASS),
            "--verbose",
            "--out", str(tmp_path),
            "--run-id", "t-verb-v",
        ])
        assert code_normal == code_verbose, (
            f"Verbose changed exit code: {code_normal} vs {code_verbose}"
        )

    def test_verbose_short_flag(self, tmp_path: Path) -> None:
        """-v shorthand must work and run without error."""
        code, _ = _run_main([
            "--replay", str(RUN_PASS),
            "-v",
            "--out", str(tmp_path),
            "--run-id", "t-short-v",
        ])
        # Just confirm it ran (no exception) and returns a valid code.
        assert code in (0, 1, 2)

    def test_run_log_not_on_stdout(self, tmp_path: Path) -> None:
        """Log messages must not pollute stdout."""
        _, out = _run_main([
            "--replay", str(RUN_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-log-stdout",
        ])
        # Log format contains "eval.run_eval" — should not appear on stdout.
        assert "eval.run_eval" not in out, (
            "Log entries appeared on stdout; they must go to stderr only"
        )

    def test_report_exists_without_verdicts(self, tmp_path: Path) -> None:
        """REPORT.md must also be written in programmatic-only (no verdicts) mode."""
        _run_main([
            "--replay", str(RUN_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-rpt-no-verdicts",
        ])
        report_path = tmp_path / "t-rpt-no-verdicts" / "REPORT.md"
        assert report_path.exists(), "REPORT.md must be written even without verdicts"

    def test_run_log_exists_without_verdicts(self, tmp_path: Path) -> None:
        """run.log must be written in programmatic-only mode too."""
        _run_main([
            "--replay", str(RUN_PASS),
            "--out", str(tmp_path),
            "--run-id", "t-log-no-verdicts",
        ])
        log_path = tmp_path / "t-log-no-verdicts" / "run.log"
        assert log_path.exists(), "run.log must be written even without verdicts"


# ---------------------------------------------------------------------------
# Live pipeline — mocked SEC ingest + fake live judge provider.
#
# The live path needs network + an API key, so we mock its three external
# touchpoints: fetch_filing (SEC download), VectorStore (Chroma), run_live
# (the real SUT loop), and get_provider (the live LLM judge). No network, no
# key. These tests prove the live pipeline now scores the FULL stack
# (programmatic + robustness + live judge) and therefore reaches a real
# release decision instead of stalling at INCOMPLETE.
# ---------------------------------------------------------------------------


class _FakeLiveProvider:
    """Stand-in for the live OpenAI provider — returns a perfect judge verdict."""

    model = "fake-judge-model"

    def generate(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0) -> str:
        return '{"faithfulness": 1.0, "answer_relevance": 1.0, "hallucination": 0}'

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class _DummyStore:
    """No-op VectorStore replacement (the SUT loop is mocked, so it is unused)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


def _write_issuers(tmp_path: Path) -> Path:
    issuers = tmp_path / "issuers.yaml"
    issuers.write_text(
        "issuers:\n"
        "  - ticker: AAPL\n"
        "    cik: '0000320193'\n"
        "    forms: ['10-K']\n"
    )
    return issuers


class TestLivePipeline:
    def test_live_full_stack_reaches_decision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Live run scores judge + robustness → a real PASS/BLOCKED, not INCOMPLETE."""
        import src.eval.runner as runner_mod
        import src.sut.ingest as ingest_mod
        import src.sut.providers as providers_mod
        import src.sut.store as store_mod
        from src.eval.runner import load_replay

        monkeypatch.setattr(ingest_mod, "fetch_filing", lambda *a, **k: 7)
        monkeypatch.setattr(store_mod, "VectorStore", _DummyStore)
        monkeypatch.setattr(
            runner_mod, "run_live",
            lambda goldens, store, provider=None: load_replay(RUN_PASS),
        )
        monkeypatch.setattr(providers_mod, "get_provider", lambda mode=None: _FakeLiveProvider())
        # The judge calls get_judge_provider() — patch it to the fake too.
        monkeypatch.setattr(providers_mod, "get_judge_provider", lambda: _FakeLiveProvider())

        issuers = _write_issuers(tmp_path)
        code, out = _run_main([
            "--live",
            "--issuers", str(issuers),
            "--out", str(tmp_path),
            "--run-id", "t-live",
        ])

        # All hard gates evaluated → decided, never INCOMPLETE (exit 2).
        assert code in (0, 1), f"live run must be decided, not INCOMPLETE; got {code}\n{out}"

        data = json.loads((tmp_path / "t-live" / "scorecard.json").read_text())
        assert data["mode"] == "live"
        assert data["status"] in ("PASS", "BLOCKED")
        # Judge ran (faithfulness scored) and robustness ran (injection_resistance scored).
        assert data["metric_summary"]["faithfulness"] is not None
        assert data["metric_summary"]["injection_resistance"] is not None
        assert (tmp_path / "t-live" / "run.jsonl").exists()

    def test_live_empty_corpus_is_hard_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If no filings ingest (all skipped), the run fails instead of scoring nothing."""
        import src.sut.ingest as ingest_mod
        import src.sut.store as store_mod

        monkeypatch.setattr(ingest_mod, "fetch_filing", lambda *a, **k: 0)
        monkeypatch.setattr(store_mod, "VectorStore", _DummyStore)

        issuers = _write_issuers(tmp_path)
        code, _ = _run_main([
            "--live",
            "--issuers", str(issuers),
            "--out", str(tmp_path),
            "--run-id", "t-live-empty",
        ])

        assert code == 1, "empty corpus must be a hard error (exit 1)"
        # It fails before scoring, so no scorecard is written.
        assert not (tmp_path / "t-live-empty" / "scorecard.json").exists()

    def test_live_ingest_network_error_skips_filing_and_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A requests.RequestException for one filing is skipped; the run continues."""
        import requests

        import src.eval.runner as runner_mod
        import src.sut.ingest as ingest_mod
        import src.sut.providers as providers_mod
        import src.sut.store as store_mod
        from src.eval.runner import load_replay

        def _fetch(*args: object, **kwargs: object) -> int:
            if kwargs.get("issuer") == "AAPL":
                raise requests.RequestException("simulated SEC outage")
            return 7

        monkeypatch.setattr(ingest_mod, "fetch_filing", _fetch)
        monkeypatch.setattr(store_mod, "VectorStore", _DummyStore)
        monkeypatch.setattr(
            runner_mod, "run_live",
            lambda goldens, store, provider=None: load_replay(RUN_PASS),
        )
        monkeypatch.setattr(providers_mod, "get_judge_provider", lambda: _FakeLiveProvider())

        issuers = tmp_path / "issuers.yaml"
        issuers.write_text(
            "issuers:\n"
            "  - ticker: AAPL\n"
            "    cik: '0000320193'\n"
            "    forms: ['10-K']\n"
            "  - ticker: MSFT\n"
            "    cik: '0000789019'\n"
            "    forms: ['10-K']\n"
        )
        code, _ = _run_main([
            "--live",
            "--issuers", str(issuers),
            "--out", str(tmp_path),
            "--run-id", "t-live-skip",
        ])

        # The failed AAPL filing is skipped with a warning; MSFT still ingests,
        # so the run survives and reaches a real decision.
        assert code in (0, 1), f"run must survive a single failed filing; got {code}"
        stderr = capsys.readouterr().err
        assert "skipped AAPL 10-K" in stderr
        assert (tmp_path / "t-live-skip" / "scorecard.json").exists()

    def test_live_ingest_programming_error_is_not_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A non-ingest error (e.g. TypeError bug) must NOT be skipped per-filing —
        it fails the run instead of silently degrading the corpus."""
        import src.sut.ingest as ingest_mod
        import src.sut.store as store_mod

        def _fetch(*args: object, **kwargs: object) -> int:
            raise TypeError("simulated programming bug")

        monkeypatch.setattr(ingest_mod, "fetch_filing", _fetch)
        monkeypatch.setattr(store_mod, "VectorStore", _DummyStore)

        issuers = _write_issuers(tmp_path)
        code, _ = _run_main([
            "--live",
            "--issuers", str(issuers),
            "--out", str(tmp_path),
            "--run-id", "t-live-bug",
        ])

        assert code == 1
        stderr = capsys.readouterr().err
        # Failed as a pipeline error, not skipped as a per-filing ingest warning.
        assert "skipped" not in stderr
        assert "Live pipeline failed" in stderr
        assert not (tmp_path / "t-live-bug" / "scorecard.json").exists()
