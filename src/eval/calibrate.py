"""calibrate.py — Cohen's κ agreement between the LLM judge and reference labels.

This validates the GRADER, not the SUT. It loads the balanced calibration set
(planted pass + fail cases), obtains judge verdicts for each item (offline from a
recorded fixture, or live from the real judge), reduces both the judge and the
reference labels to binary per dimension, and reports Cohen's κ. κ ≥ KAPPA_TARGET
means the judge agrees with the reference well enough to be trusted.

IMPORTANT — single-annotator baseline:
    The reference labels in datasets/judge_calibration_set.jsonl are authored by a
    SINGLE annotator (the repo author), not an independent inter-rater panel. The κ
    reported here is therefore a *smoke test* of judge agreement, not a production
    calibration certificate. A real calibration uses ≥2 independent annotators on a
    larger balanced set. The harness is identical either way — only the label source
    changes.

Two dimensions are scored:
    faithfulness  — judge passes iff faithfulness score ≥ the faithfulness gate;
                    reference is the `faithful` bool.
    hallucination — judge flag (0/1) vs the reference `hallucinated` flag.

Usage:
    uv run python -m src.eval.calibrate                 # offline (recorded verdicts)
    uv run python -m src.eval.calibrate --live          # call the real judge
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from src.config import KAPPA_TARGET, gate_threshold

DEFAULT_SET = "datasets/judge_calibration_set.jsonl"
DEFAULT_VERDICTS = "datasets/fixtures/judge_calibration_verdicts.json"


@dataclass(frozen=True)
class CalibrationItem:
    """One labeled calibration scenario."""

    id: str
    question: str
    context: str
    answer: str
    ref_faithful: bool
    ref_hallucinated: int
    failure_type: str


def load_calibration_set(path: str | Path = DEFAULT_SET) -> list[CalibrationItem]:
    """Load the calibration JSONL into CalibrationItems.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if a row is missing required fields.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Calibration set not found at {p}. Author it before running calibration."
        )

    items: list[CalibrationItem] = []
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        row = json.loads(raw)
        try:
            ref = row["reference"]
            items.append(CalibrationItem(
                id=row["id"],
                question=row["question"],
                context=row["context"],
                answer=row["answer"],
                ref_faithful=bool(ref["faithful"]),
                ref_hallucinated=int(ref["hallucinated"]),
                failure_type=row.get("failure_type", "none"),
            ))
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Malformed calibration row at line {line_no} in {p}: {exc}"
            ) from exc

    if not items:
        raise ValueError(f"Calibration set {p} is empty.")
    return items


def cohen_kappa(rater_a: list[int], rater_b: list[int]) -> float:
    """Cohen's κ between two equal-length sequences of categorical labels.

    κ = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e is the
    agreement expected by chance from the marginal label frequencies. Returns 1.0
    when both raters are perfectly constant and identical (p_e == 1).

    Raises:
        ValueError: if the inputs are empty or of unequal length.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater label lists must be the same length")
    n = len(rater_a)
    if n == 0:
        raise ValueError("cannot compute kappa over zero items")

    p_o = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n

    categories = set(rater_a) | set(rater_b)
    p_e = sum(
        (rater_a.count(c) / n) * (rater_b.count(c) / n) for c in categories
    )

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def _judge_labels(
    items: list[CalibrationItem],
    *,
    mode: str,
    verdicts_path: str | None,
) -> tuple[list[int], list[int]]:
    """Run the judge over the calibration items and reduce to binary labels.

    Returns (judge_faithful_flags, judge_hallucinated_flags), aligned to *items*.
    Reuses score_judge so the exact production judge path is exercised.
    """
    # Local imports keep offline callers from importing the runner graph eagerly.
    from src.eval.metrics.judge import score_judge
    from src.eval.runner import RetrievedRef, RunRecord

    records: list[RunRecord] = [
        RunRecord(
            id=item.id,
            bucket="factual_lookup",
            question=item.question,
            answer=item.answer,
            retrieved=[
                RetrievedRef(
                    chunk_id=f"{item.id}#c1",
                    text=item.context,
                    similarity=1.0,
                    issuer="",
                    form="",
                    filing_date="",
                    accession="",
                    section="full",
                    source_url="",
                )
            ],
            citations={"c1": f"{item.id}#c1"},
            unmatched_citations=[],
            latency_ms=0,
            mode="replay",
        )
        for item in items
    ]

    results = score_judge(records, [], mode=mode, verdicts_path=verdicts_path)

    faith_by_id = {
        r.item_id: r.score for r in results if r.metric == "faithfulness"
    }
    halluc_by_id = {
        r.item_id: r.score for r in results if r.metric == "hallucination_rate"
    }

    faith_threshold = gate_threshold("faithfulness")
    judge_faithful = [
        int(faith_by_id[item.id] >= faith_threshold) for item in items
    ]
    judge_hallucinated = [int(halluc_by_id[item.id] == 1.0) for item in items]
    return judge_faithful, judge_hallucinated


def calibrate(
    set_path: str = DEFAULT_SET,
    *,
    mode: str = "offline",
    verdicts_path: str | None = DEFAULT_VERDICTS,
) -> dict[str, object]:
    """Compute judge-vs-reference Cohen's κ for the calibration set.

    Returns a report dict with per-dimension κ, observed agreement, n, and a
    per-dimension status ('CALIBRATED' / 'UNCALIBRATED') against KAPPA_TARGET.
    """
    items = load_calibration_set(set_path)

    ref_faithful = [int(i.ref_faithful) for i in items]
    ref_hallucinated = [i.ref_hallucinated for i in items]

    judge_faithful, judge_hallucinated = _judge_labels(
        items, mode=mode, verdicts_path=verdicts_path if mode == "offline" else None
    )

    kappa_faith = cohen_kappa(ref_faithful, judge_faithful)
    kappa_halluc = cohen_kappa(ref_hallucinated, judge_hallucinated)

    def _agreement(a: list[int], b: list[int]) -> float:
        return sum(1 for x, y in zip(a, b) if x == y) / len(a)

    return {
        "n": len(items),
        "mode": mode,
        "kappa_target": KAPPA_TARGET,
        "faithfulness": {
            "kappa": kappa_faith,
            "agreement": _agreement(ref_faithful, judge_faithful),
            "status": "CALIBRATED" if kappa_faith >= KAPPA_TARGET else "UNCALIBRATED",
        },
        "hallucination": {
            "kappa": kappa_halluc,
            "agreement": _agreement(ref_hallucinated, judge_hallucinated),
            "status": "CALIBRATED" if kappa_halluc >= KAPPA_TARGET else "UNCALIBRATED",
        },
    }


def _format_report(report: dict[str, object]) -> str:
    lines = [
        "─" * 62,
        f"  JUDGE CALIBRATION   n={report['n']}   mode={report['mode']}   "
        f"κ target={report['kappa_target']}",
        "  (single-annotator reference baseline — not an independent inter-rater study)",
        "─" * 62,
    ]
    for dim in ("faithfulness", "hallucination"):
        d = report[dim]
        assert isinstance(d, dict)
        lines.append(
            f"  {dim:<14} κ={d['kappa']:.3f}   agreement={d['agreement']:.0%}   "
            f"{d['status']}"
        )
    lines.append("─" * 62)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 if every dimension meets KAPPA_TARGET, else 1."""
    parser = argparse.ArgumentParser(
        prog="calibrate",
        description="Cohen's κ agreement between the LLM judge and reference labels.",
    )
    parser.add_argument("--set", default=DEFAULT_SET, dest="set_path",
                        help=f"Calibration JSONL (default: {DEFAULT_SET}).")
    parser.add_argument("--verdicts", default=DEFAULT_VERDICTS,
                        help="Recorded judge verdicts JSON (offline mode).")
    parser.add_argument("--live", action="store_true",
                        help="Call the real judge instead of recorded verdicts.")
    args = parser.parse_args(argv)

    mode = "live" if args.live else "offline"
    try:
        report = calibrate(args.set_path, mode=mode, verdicts_path=args.verdicts)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(_format_report(report))

    faith = report["faithfulness"]
    halluc = report["hallucination"]
    assert isinstance(faith, dict) and isinstance(halluc, dict)
    calibrated = faith["status"] == "CALIBRATED" and halluc["status"] == "CALIBRATED"
    return 0 if calibrated else 1


if __name__ == "__main__":
    raise SystemExit(main())
