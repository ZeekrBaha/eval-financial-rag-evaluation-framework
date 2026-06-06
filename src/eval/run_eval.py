"""run_eval.py — End-to-end eval orchestrator for T15 / N-03.

Wires the full offline (replay) and live evaluation pipelines into one command.

Usage:
    # Offline replay (default, no key/network required):
    uv run python -m src.eval.run_eval

    # Override replay fixture:
    uv run python -m src.eval.run_eval --replay datasets/fixtures/run_fail.jsonl

    # Live mode (requires OPENAI_API_KEY + network):
    uv run python -m src.eval.run_eval --live

Public surface:
    main(argv: list[str] | None = None) -> int
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from src.eval.obs import configure_run_logging, get_logger, timed


def _default_run_id() -> str:
    """Return a timestamp-based run id like 'run-20260606-154230'."""
    return datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval",
        description="Financial RAG evaluation pipeline (offline replay or live).",
    )
    p.add_argument(
        "--golden",
        default="datasets/golden_set.jsonl",
        help="Path to golden set JSONL (default: datasets/golden_set.jsonl).",
    )
    p.add_argument(
        "--replay",
        default="datasets/fixtures/run_pass.jsonl",
        help=(
            "Path to replay JSONL fixture (offline mode, default: "
            "datasets/fixtures/run_pass.jsonl). Ignored when --live is set."
        ),
    )
    p.add_argument(
        "--verdicts",
        default=None,
        help=(
            "Path to recorded judge verdicts JSON (offline replay mode). When "
            "provided, the judge metrics (faithfulness, hallucination_rate, "
            "answer_relevance) are evaluated from the fixture, enabling a full "
            "hard-gate decision. When omitted, those gates stay UNEVALUATED and "
            "the run is reported INCOMPLETE (the honest partial path)."
        ),
    )
    p.add_argument(
        "--live",
        action="store_true",
        help=(
            "Live mode: ingest filings from SEC EDGAR and run the real SUT. "
            "Requires OPENAI_API_KEY and network access."
        ),
    )
    p.add_argument(
        "--issuers",
        default="datasets/issuers.yaml",
        help="Path to issuers YAML (used only in --live mode).",
    )
    p.add_argument(
        "--out",
        default="reports",
        help="Output directory root; artifacts go to <out>/<run-id>/.",
    )
    p.add_argument(
        "--run-id",
        default=None,
        dest="run_id",
        help="Override run identifier (default: timestamp like run-YYYYMMDD-HHMMSS).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging to stderr (default: INFO only).",
    )
    return p


def _replay_pipeline(
    golden_path: str,
    replay_path: str,
    out_dir: Path,
    run_id: str,
    verdicts_path: str | None = None,
    verbose: bool = False,
) -> int:
    """Offline replay pipeline — no network, no API key.

    Scores programmatic + robustness metrics always; adds judge metrics from a
    recorded verdicts fixture when *verdicts_path* is given. With verdicts, all
    four hard gates can be evaluated (→ PASS/BLOCKED); without, faithfulness and
    hallucination_rate stay unevaluated (→ INCOMPLETE), which is the honest
    partial path.

    Returns exit code (0 = PASS, 1 = BLOCKED, 2 = INCOMPLETE).
    """
    from src.eval.aggregate import build_scorecard
    from src.eval.gates import enforce
    from src.eval.golden import load_goldens
    from src.eval.metrics.judge import score_judge
    from src.eval.metrics.programmatic import score_programmatic
    from src.eval.metrics.robustness import score_robustness
    from src.eval.report import write_report
    from src.eval.runner import load_replay
    from src.eval.scorecard import render_html, render_json, render_text

    # 0. Set up run directory and logging first.
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logging(run_dir, verbose=verbose)
    log = get_logger("run_eval")

    log.info("pipeline=replay run_id=%s", run_id)

    # 1. Load inputs
    with timed(log, "load_goldens"):
        goldens = load_goldens(golden_path)
    log.info("loaded %d goldens from %s", len(goldens), golden_path)

    with timed(log, "load_replay"):
        records = load_replay(replay_path)
    log.info("loaded %d replay records from %s", len(records), replay_path)

    # 2. Score — programmatic + robustness always; judge only with verdicts.
    with timed(log, "score_programmatic"):
        prog = score_programmatic(records, goldens)

    with timed(log, "score_robustness"):
        rob = score_robustness(records, goldens)

    judge = None
    if verdicts_path is not None:
        with timed(log, "score_judge"):
            judge = score_judge(
                records, goldens, mode="offline", verdicts_path=verdicts_path
            )

    n_prog = len(prog)
    n_rob = len(rob)
    n_judge = len(judge) if judge is not None else 0
    log.info(
        "scored %d programmatic / %d robustness / %d judge results",
        n_prog, n_rob, n_judge,
    )

    # 3. Aggregate
    with timed(log, "build_scorecard"):
        sc = build_scorecard(
            records,
            goldens,
            prog_results=prog,
            judge_results=judge,
            robustness_results=rob,
            run_id=run_id,
            mode="replay",
        )

    # 4. Gate enforcement (stamps sc.status, returns exit_code + summary_lines)
    with timed(log, "enforce_gates"):
        outcome = enforce(sc)

    log.info("status=%s exit=%d", outcome.status, outcome.exit_code)

    # 5. Write artifacts
    with timed(log, "write_artifacts"):
        render_json(sc, run_dir / "scorecard.json")
        render_html(sc, run_dir / "scorecard.html")
        all_metric_results = prog + rob + (judge or [])
        write_report(sc, outcome, goldens, all_metric_results, run_dir / "REPORT.md")

    log.info("wrote artifacts to %s", run_dir)

    # 6. Print scorecard text + gate summary (stdout — unchanged)
    print(render_text(sc))
    for line in outcome.summary_lines:
        print(line)

    return outcome.exit_code


def _live_pipeline(
    golden_path: str,
    issuers_path: str,
    out_dir: Path,
    run_id: str,
    verbose: bool = False,
) -> int:
    """Live pipeline — requires OPENAI_API_KEY and SEC EDGAR network access.

    Returns exit code (0 = PASS, 1 = BLOCKED, 2 = INCOMPLETE).
    Raises SystemExit with a descriptive message if key/network is unavailable.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: PyYAML is required for --live mode. Install with: uv add pyyaml",
            file=sys.stderr,
        )
        return 1

    from src.eval.aggregate import build_scorecard
    from src.eval.gates import enforce
    from src.eval.golden import load_goldens
    from src.eval.metrics.programmatic import score_programmatic
    from src.eval.report import write_report
    from src.eval.runner import run_live, write_run
    from src.eval.scorecard import render_html, render_json, render_text
    from src.sut.ingest import fetch_filing
    from src.sut.store import VectorStore

    # Set up run directory and logging before anything else.
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logging(run_dir, verbose=verbose)
    log = get_logger("run_eval")

    try:
        log.info("pipeline=live run_id=%s", run_id)

        with timed(log, "load_goldens"):
            goldens = load_goldens(golden_path)
        log.info("loaded %d goldens from %s", len(goldens), golden_path)

        with open(issuers_path) as fh:
            issuers_data = yaml.safe_load(fh)

        store = VectorStore()

        # Ingest each issuer's filings (best-effort — log failures, keep going)
        for issuer in issuers_data.get("issuers", []):
            ticker = issuer.get("ticker", "")
            cik = issuer.get("cik", "")
            for form in issuer.get("forms", ["10-K"]):
                try:
                    fetch_filing(
                        cik=cik,
                        accession="",   # fetch_filing will resolve latest
                        store=store,
                        form=form,
                        issuer=ticker,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("skipped %s %s: %s", ticker, form, exc)
                    print(
                        f"WARNING: skipped {ticker} {form}: {exc}",
                        file=sys.stderr,
                    )

        with timed(log, "run_live"):
            records = run_live(goldens, store)
        log.info("live run produced %d records", len(records))

        write_run(records, run_dir / "run.jsonl")

        with timed(log, "score_programmatic"):
            prog = score_programmatic(records, goldens)
        log.info("scored %d programmatic results", len(prog))

        with timed(log, "build_scorecard"):
            sc = build_scorecard(
                records,
                goldens,
                prog_results=prog,
                run_id=run_id,
                mode="live",
            )

        with timed(log, "enforce_gates"):
            outcome = enforce(sc)

        log.info("status=%s exit=%d", outcome.status, outcome.exit_code)

        with timed(log, "write_artifacts"):
            render_json(sc, run_dir / "scorecard.json")
            render_html(sc, run_dir / "scorecard.html")
            write_report(sc, outcome, goldens, prog, run_dir / "REPORT.md")

        log.info("wrote artifacts to %s", run_dir)

        print(render_text(sc))
        for line in outcome.summary_lines:
            print(line)

        return outcome.exit_code

    except Exception as exc:  # noqa: BLE001
        log.error("live pipeline failed: %s", exc)
        print(
            f"\nERROR: Live pipeline failed — {exc}\n"
            "Check that OPENAI_API_KEY is set and network access is available.",
            file=sys.stderr,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for the eval pipeline.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 = PASS / RELEASE OK, 1 = BLOCKED / error, 2 = INCOMPLETE.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_id = args.run_id or _default_run_id()
    out_dir = Path(args.out)

    if args.live:
        return _live_pipeline(
            golden_path=args.golden,
            issuers_path=args.issuers,
            out_dir=out_dir,
            run_id=run_id,
            verbose=args.verbose,
        )
    else:
        return _replay_pipeline(
            golden_path=args.golden,
            replay_path=args.replay,
            out_dir=out_dir,
            run_id=run_id,
            verdicts_path=args.verdicts,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    raise SystemExit(main())
