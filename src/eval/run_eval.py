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
    return p


def _replay_pipeline(
    golden_path: str,
    replay_path: str,
    out_dir: Path,
    run_id: str,
    verdicts_path: str | None = None,
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
    from src.eval.runner import load_replay
    from src.eval.scorecard import render_html, render_json, render_text

    # 1. Load inputs
    goldens = load_goldens(golden_path)
    records = load_replay(replay_path)

    # 2. Score — programmatic + robustness always; judge only with verdicts.
    prog = score_programmatic(records, goldens)
    rob = score_robustness(records, goldens)
    judge = None
    if verdicts_path is not None:
        judge = score_judge(
            records, goldens, mode="offline", verdicts_path=verdicts_path
        )

    # 3. Aggregate
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
    outcome = enforce(sc)

    # 5. Write artifacts
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    render_json(sc, run_dir / "scorecard.json")
    render_html(sc, run_dir / "scorecard.html")

    # 6. Print scorecard text + gate summary
    print(render_text(sc))
    for line in outcome.summary_lines:
        print(line)

    return outcome.exit_code


def _live_pipeline(
    golden_path: str,
    issuers_path: str,
    out_dir: Path,
    run_id: str,
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
    from src.eval.runner import run_live, write_run
    from src.eval.scorecard import render_html, render_json, render_text
    from src.sut.ingest import fetch_filing
    from src.sut.store import VectorStore

    try:
        goldens = load_goldens(golden_path)

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
                    print(
                        f"WARNING: skipped {ticker} {form}: {exc}",
                        file=sys.stderr,
                    )

        records = run_live(goldens, store)

        run_dir = out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_run(records, run_dir / "run.jsonl")

        prog = score_programmatic(records, goldens)
        sc = build_scorecard(
            records,
            goldens,
            prog_results=prog,
            run_id=run_id,
            mode="live",
        )
        outcome = enforce(sc)

        render_json(sc, run_dir / "scorecard.json")
        render_html(sc, run_dir / "scorecard.html")

        print(render_text(sc))
        for line in outcome.summary_lines:
            print(line)

        return outcome.exit_code

    except Exception as exc:  # noqa: BLE001
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
        )
    else:
        return _replay_pipeline(
            golden_path=args.golden,
            replay_path=args.replay,
            out_dir=out_dir,
            run_id=run_id,
            verdicts_path=args.verdicts,
        )


if __name__ == "__main__":
    raise SystemExit(main())
