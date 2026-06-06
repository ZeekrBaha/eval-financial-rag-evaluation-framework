"""LLM-as-judge metrics for T11 / E-07.

Two modes:
  offline — deterministic; reads pre-recorded verdict fixtures keyed by item id.
            No LLM key or network required. Safe for CI.
  live    — calls a real judge model via get_provider("live"). Thin wrapper;
            lazy-imports any judge library so offline never needs the package.

Public surface:
  score_judge(records, goldens, *, mode, verdicts_path) -> list[MetricResult]

Verdict fixture format (JSON object keyed by item id):
  {
    "fact-001": {"faithfulness": 1.0, "answer_relevance": 1.0, "hallucination": 0},
    "neg-001":  {"faithfulness": 1.0, "answer_relevance": 0.95, "hallucination": 0}
  }

  faithfulness     — float in [0, 1]. Score ≥0.95 to pass (HARD gate).
  answer_relevance — float in [0, 1]. Score ≥0.90 to pass (SOFT gate).
  hallucination    — 0 or 1. 1 means the answer contains an unsupported claim.
                     MetricResult.score = float(hallucination). aggregate_metric
                     over these gives the hallucination RATE; the HARD gate checks
                     rate ≤0.01.

CALIBRATION NOTE — the judge is UNCALIBRATED until the separate calibration
project (Cohen's κ agreement with human annotators) is complete. Faithfulness
and answer_relevance scores are working hypotheses, not production-grade
measurements. Do not treat verdict thresholds as certified until κ ≥0.7 is
established against a human-annotated reference set.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.golden import GoldenItem
from src.eval.runner import RunRecord

# Import MetricResult from programmatic — do NOT redefine it.
from src.eval.metrics.programmatic import MetricResult

# ---------------------------------------------------------------------------
# Thresholds (mirrored from src/config.py — avoid circular import)
# ---------------------------------------------------------------------------

_FAITHFULNESS_PASS = 0.95   # HARD gate: mean faithfulness ≥0.95
_ANSWER_RELEVANCE_PASS = 0.90  # SOFT gate: mean answer_relevance ≥0.90
# hallucination_rate is a HARD gate: mean ≤0.01; individual item passes iff score==0.0


# ---------------------------------------------------------------------------
# Offline path — read pre-recorded verdict fixture
# ---------------------------------------------------------------------------


def _score_offline(
    records: list[RunRecord],
    goldens: list[GoldenItem],  # noqa: ARG001 — kept for API symmetry with live
    verdicts: dict[str, dict],
) -> list[MetricResult]:
    """Emit 3 MetricResults per record from a pre-loaded verdict dict."""
    results: list[MetricResult] = []

    for record in records:
        item_id = record.id
        if item_id not in verdicts:
            raise ValueError(
                f"Judge verdict missing for item id={item_id!r}. "
                "All record ids must have a corresponding entry in the verdict fixture. "
                "Add an entry or regenerate the fixture before running eval."
            )

        v = verdicts[item_id]
        faith_score: float = float(v["faithfulness"])
        ar_score: float = float(v["answer_relevance"])
        halluc_flag: int = int(v["hallucination"])
        halluc_score: float = float(halluc_flag)

        results.append(MetricResult(
            metric="faithfulness",
            item_id=item_id,
            applicable=True,
            score=faith_score,
            passed=faith_score >= _FAITHFULNESS_PASS,
            detail=(
                f"Faithfulness score {faith_score:.3f} "
                f"({'≥' if faith_score >= _FAITHFULNESS_PASS else '<'}"
                f"{_FAITHFULNESS_PASS} threshold). "
                "Source: offline verdict fixture. "
                "UNCALIBRATED — see module docstring."
            ),
        ))

        results.append(MetricResult(
            metric="answer_relevance",
            item_id=item_id,
            applicable=True,
            score=ar_score,
            passed=ar_score >= _ANSWER_RELEVANCE_PASS,
            detail=(
                f"Answer relevance score {ar_score:.3f} "
                f"({'≥' if ar_score >= _ANSWER_RELEVANCE_PASS else '<'}"
                f"{_ANSWER_RELEVANCE_PASS} threshold). "
                "Source: offline verdict fixture. "
                "UNCALIBRATED — see module docstring."
            ),
        ))

        results.append(MetricResult(
            metric="hallucination_rate",
            item_id=item_id,
            applicable=True,
            score=halluc_score,
            passed=halluc_flag == 0,
            detail=(
                f"Hallucination flag={halluc_flag} "
                f"(score={halluc_score:.1f}; passed iff flag==0). "
                "Source: offline verdict fixture. "
                "UNCALIBRATED — see module docstring."
            ),
        ))

    return results


# ---------------------------------------------------------------------------
# Live path — real judge model call (thin; lazy imports; not run in tests)
# ---------------------------------------------------------------------------


def _score_live(
    records: list[RunRecord],
    goldens: list[GoldenItem],
) -> list[MetricResult]:
    """Call a real judge model to score faithfulness, answer_relevance, hallucination.

    Uses get_provider("live") from src.sut.providers. Any judge library (Ragas,
    DeepEval, etc.) is lazy-imported so offline mode never needs it installed.

    NOTE: UNCALIBRATED — judge verdicts are working hypotheses until Cohen's κ
    with human annotators is established (target κ ≥0.7). See module docstring.
    """
    try:
        from src.sut.providers import get_provider
        provider = get_provider("live")
    except Exception as exc:
        raise RuntimeError(
            "Live judge mode requires a valid provider key. "
            f"Original error: {exc}. "
            "Set the appropriate API key or switch to mode='offline'."
        ) from exc

    results: list[MetricResult] = []
    golden_map: dict[str, GoldenItem] = {g.id: g for g in goldens}

    for record in records:
        golden = golden_map.get(record.id)
        context = "\n".join(ref.text for ref in record.retrieved)

        # Prompt the judge model to evaluate faithfulness, answer relevance,
        # and hallucination. The prompt asks for a JSON response.
        judge_prompt = (
            "You are an expert financial QA evaluator. "
            "Given the QUESTION, CONTEXT (retrieved passages), and ANSWER, "
            "rate on a scale of 0.0–1.0:\n"
            "  faithfulness: fraction of answer claims supported by CONTEXT\n"
            "  answer_relevance: how directly the answer addresses the QUESTION\n"
            "  hallucination: 1 if the answer contains any claim NOT in CONTEXT, else 0\n\n"
            f"QUESTION: {record.question}\n"
            f"CONTEXT:\n{context}\n"
            f"ANSWER: {record.answer}\n\n"
            "Respond ONLY with JSON: "
            '{"faithfulness": <float>, "answer_relevance": <float>, "hallucination": <0|1>}'
        )

        try:
            raw = provider.generate(judge_prompt)
            verdict = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(
                f"Judge model call failed for item id={record.id!r}: {exc}. "
                "Check provider configuration and API key."
            ) from exc

        judge_note = f"judge_model={getattr(provider, 'model', 'unknown')}; UNCALIBRATED"

        faith_score = float(verdict["faithfulness"])
        ar_score = float(verdict["answer_relevance"])
        halluc_flag = int(verdict["hallucination"])
        halluc_score = float(halluc_flag)

        results.append(MetricResult(
            metric="faithfulness",
            item_id=record.id,
            applicable=True,
            score=faith_score,
            passed=faith_score >= _FAITHFULNESS_PASS,
            detail=(
                f"Faithfulness={faith_score:.3f}; {judge_note}. "
                "UNCALIBRATED — see module docstring."
            ),
        ))

        results.append(MetricResult(
            metric="answer_relevance",
            item_id=record.id,
            applicable=True,
            score=ar_score,
            passed=ar_score >= _ANSWER_RELEVANCE_PASS,
            detail=(
                f"Answer relevance={ar_score:.3f}; {judge_note}. "
                "UNCALIBRATED — see module docstring."
            ),
        ))

        results.append(MetricResult(
            metric="hallucination_rate",
            item_id=record.id,
            applicable=True,
            score=halluc_score,
            passed=halluc_flag == 0,
            detail=(
                f"Hallucination flag={halluc_flag}; {judge_note}. "
                "UNCALIBRATED — see module docstring."
            ),
        ))

    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def score_judge(
    records: list[RunRecord],
    goldens: list[GoldenItem],
    *,
    mode: str = "offline",
    verdicts_path: str | Path | None = None,
) -> list[MetricResult]:
    """Run LLM-as-judge metrics over a list of RunRecords.

    Args:
        records:       RunRecords to evaluate (one per golden item).
        goldens:       GoldenItems (used for API symmetry and live-mode context).
        mode:          "offline" (deterministic fixture) or "live" (real judge call).
        verdicts_path: Required for mode="offline". Path to the JSON verdict fixture.

    Returns:
        Flat list of MetricResult — exactly 3 per record:
          faithfulness, answer_relevance, hallucination_rate.

    Raises:
        ValueError: (offline) If a record id is missing from the verdict fixture.
        RuntimeError: (live) If the judge model call fails.
    """
    if mode == "offline":
        if verdicts_path is None:
            raise ValueError(
                "verdicts_path is required when mode='offline'. "
                "Pass the path to a JSON verdict fixture file."
            )
        path = Path(verdicts_path)
        try:
            verdicts: dict[str, dict] = json.loads(path.read_text())
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Verdict fixture not found at {path}. "
                "Create the fixture or check the path."
            ) from exc
        return _score_offline(records, goldens, verdicts)

    elif mode == "live":
        return _score_live(records, goldens)

    else:
        raise ValueError(
            f"Unknown mode={mode!r}. Must be 'offline' or 'live'."
        )
