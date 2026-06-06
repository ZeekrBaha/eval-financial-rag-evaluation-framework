"""Programmatic metrics for T10 / E-06.

All metrics are deterministic — no LLM call, no network, no API key required.
Each metric function takes a matched (RunRecord, GoldenItem) pair and returns
a MetricResult.

Public surface:
  MetricResult          — result dataclass (one per metric per item)
  numerical_exactness   — fraction of golden numeric answers present in the answer
  citation_validity     — fraction of citations that genuinely support the claim
  negative_rejection    — whether a must-refuse item was correctly refused
  temporal_correctness  — whether cited chunks come from the latest filing
  entity_disambiguation — whether the correct issuer is cited and named
  context_recall        — fraction of expected sources retrieved
  context_precision     — fraction of retrieved chunks that are relevant
  score_programmatic    — run all 7 metrics over matched (record, golden) pairs
  aggregate_metric      — mean score over applicable items for a given metric
  metric_rate           — fraction of applicable items that passed
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.eval.golden import Bucket, GoldenItem
from src.eval.runner import RunRecord

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """One metric evaluated against one golden item.

    Attributes
    ----------
    metric:     Metric name, e.g. "numerical_exactness".
    item_id:    Id from the golden item.
    applicable: False when this metric does not apply to the item (e.g.
                numerical_exactness on an item with no numeric_answers). Such
                results are excluded from aggregates.
    score:      Value in [0, 1]. Undefined (but set to 0.0) when applicable is False.
    passed:     True iff score == 1.0 (or the metric's own pass criterion).
    detail:     Human-readable explanation of the outcome.
    """

    metric: str
    item_id: str
    applicable: bool
    score: float
    passed: bool
    detail: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Minimum financial value for a numeric token to be "significant".
# This threshold (>= 100) excludes percentages (e.g. 61%), ordinal numbers,
# and "Form 10" style references while keeping dollar amounts, employee counts,
# and other material financial figures.
_SIGNIFICANT_NUM_MIN = 100.0

# Range used to recognise calendar-year tokens (e.g. "2024" in "fiscal year 2024").
_YEAR_MIN = 1900
_YEAR_MAX = 2030

# Stopwords for content-word Jaccard overlap (citation validity fallback path).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "about", "again", "also", "although", "another", "around",
        "already", "also", "along", "although", "another", "back",
        "because", "been", "before", "between", "both", "come",
        "does", "does", "during", "each", "even", "every",
        "first", "from", "give", "good", "have", "hold",
        "into", "just", "keep", "know", "last", "like", "long",
        "make", "many", "mean", "might", "more", "most", "much",
        "never", "next", "nothing", "only", "open", "over", "real",
        "seem", "should", "since", "some", "still", "such",
        "than", "that", "their", "them", "then", "there",
        "these", "they", "this", "those", "thought", "through",
        "time", "together", "under", "very", "well", "were",
        "what", "when", "where", "which", "while", "will",
        "with", "without", "your",
    }
)

# Minimum Jaccard similarity for the content-overlap citation-support path.
_JACCARD_THRESHOLD = 0.12

# Refusal cue phrases recognised by negative_rejection (case-insensitive).
_REFUSAL_CUES: tuple[str, ...] = (
    "not in the provided sources",
    "do not contain",
    "cannot find",
    "not available in the sources",
    "unable to find",
)

# context_recall / context_precision pass thresholds (from config soft gates).
_RECALL_GATE = 0.90
_PRECISION_GATE = 0.85


def _normalize_number(token: str) -> str:
    """Remove $, commas, and surrounding whitespace from a numeric token."""
    return re.sub(r"[$,\s]", "", token)


def _extract_significant_numbers(text: str) -> set[str]:
    """Return normalized number strings that represent significant financial values.

    Filters out:
    - Calendar-year integers in the range [1900, 2030] (e.g. "2024").
    - Values below _SIGNIFICANT_NUM_MIN (excludes percentages, ordinals like 10, 61%).
    """
    raw_tokens = re.findall(r"\$?[\d,]+(?:\.\d+)?", text)
    result: set[str] = set()
    for tok in raw_tokens:
        normalized = _normalize_number(tok)
        if not normalized:
            continue
        try:
            value = float(normalized)
        except ValueError:
            continue
        # Exclude calendar-year integers.
        if "." not in normalized and _YEAR_MIN <= value <= _YEAR_MAX and len(normalized) == 4:
            continue
        # Exclude small values (percentages, Form numbers, etc.).
        if value < _SIGNIFICANT_NUM_MIN:
            continue
        result.add(normalized)
    return result


def _content_words(text: str) -> set[str]:
    """Return lowercased tokens of length ≥ 4 that are not stopwords."""
    return {
        tok
        for tok in re.findall(r"[a-z]{4,}", text.lower())
        if tok not in _STOPWORDS
    }


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two sets; returns 0.0 for empty inputs."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _citation_supports(answer: str, chunk_text: str) -> bool:
    """Return True if chunk_text supports the claim made in answer.

    Decision tree:
    1. Extract significant financial numbers from the answer.
    2. If the answer has significant numbers AND the chunk also has significant numbers:
       → check numeric overlap (any answer number appears in chunk).
    3. If the answer has significant numbers BUT the chunk has NO significant numbers:
       → contextual citation; fall back to content-word Jaccard ≥ 0.12.
    4. If the answer has no significant numbers:
       → content-word Jaccard ≥ 0.12.

    This handles long-context answers where individual cited chunks provide
    contextual support (e.g. risk factors) without repeating every financial figure.
    """
    answer_nums = _extract_significant_numbers(answer)
    chunk_nums = _extract_significant_numbers(chunk_text)

    if answer_nums:
        if chunk_nums:
            # Both sides have financial numbers — require numeric overlap.
            return bool(answer_nums & chunk_nums)
        else:
            # Chunk provides textual/contextual support — use Jaccard fallback.
            return _jaccard(_content_words(answer), _content_words(chunk_text)) >= _JACCARD_THRESHOLD
    else:
        return _jaccard(_content_words(answer), _content_words(chunk_text)) >= _JACCARD_THRESHOLD


# ---------------------------------------------------------------------------
# Metric 1: numerical_exactness
# ---------------------------------------------------------------------------


def numerical_exactness(record: RunRecord, golden: GoldenItem) -> MetricResult:
    """Fraction of golden.numeric_answers whose normalized value appears in the answer.

    Applicable iff golden.numeric_answers is non-empty.
    Normalization: remove $, commas, whitespace.
    Passed iff score == 1.0.
    """
    metric_name = "numerical_exactness"

    if not golden.numeric_answers:
        return MetricResult(
            metric=metric_name,
            item_id=record.id,
            applicable=False,
            score=0.0,
            passed=False,
            detail="No numeric answers defined for this golden item.",
        )

    # Extract all normalized numeric tokens from the answer.
    answer_nums = {
        _normalize_number(tok)
        for tok in re.findall(r"\$?[\d,]+(?:\.\d+)?", record.answer)
        if _normalize_number(tok)
    }

    target_nums = [_normalize_number(n) for n in golden.numeric_answers]
    matched = [n for n in target_nums if n in answer_nums]
    score = len(matched) / len(target_nums)
    passed = score == 1.0

    detail = (
        f"Matched {len(matched)}/{len(target_nums)} numeric answers: "
        f"targets={target_nums}, matched={matched}."
    )
    return MetricResult(
        metric=metric_name,
        item_id=record.id,
        applicable=True,
        score=score,
        passed=passed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Metric 2: citation_validity
# ---------------------------------------------------------------------------


def citation_validity(record: RunRecord, golden: GoldenItem) -> MetricResult:
    """Fraction of citations (including unmatched) that genuinely support the answer.

    Applicable iff record.citations is non-empty OR unmatched_citations is non-empty.

    Each citation marker → chunk_id is tested via _citation_supports().
    Any marker in record.unmatched_citations is an invalid over-citation.
    Passed iff score == 1.0.
    """
    metric_name = "citation_validity"

    total = len(record.citations) + len(record.unmatched_citations)
    if total == 0:
        return MetricResult(
            metric=metric_name,
            item_id=record.id,
            applicable=False,
            score=0.0,
            passed=False,
            detail="No citations present; metric not applicable.",
        )

    retrieved_map = {ref.chunk_id: ref for ref in record.retrieved}
    valid = 0
    invalid_reasons: list[str] = []

    for marker, chunk_id in record.citations.items():
        ref = retrieved_map.get(chunk_id)
        if ref is None:
            invalid_reasons.append(f"{marker}→{chunk_id} not in retrieved")
            continue
        if _citation_supports(record.answer, ref.text):
            valid += 1
        else:
            invalid_reasons.append(f"{marker}→{chunk_id} does not support claim")

    for marker in record.unmatched_citations:
        invalid_reasons.append(f"{marker} is an unmatched (over-)citation")

    score = valid / total
    passed = score == 1.0
    detail = (
        f"Valid {valid}/{total}."
        + (f" Issues: {'; '.join(invalid_reasons)}" if invalid_reasons else "")
    )
    return MetricResult(
        metric=metric_name,
        item_id=record.id,
        applicable=True,
        score=score,
        passed=passed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Metric 3: negative_rejection
# ---------------------------------------------------------------------------


def negative_rejection(record: RunRecord, golden: GoldenItem) -> MetricResult:
    """Whether a must-refuse item was correctly refused.

    Applicable iff golden.must_refuse is True.
    Passed iff the answer contains a recognised refusal cue (case-insensitive)
    AND record.citations is empty.
    """
    metric_name = "negative_rejection"

    if not golden.must_refuse:
        return MetricResult(
            metric=metric_name,
            item_id=record.id,
            applicable=False,
            score=0.0,
            passed=False,
            detail="Item is not a must-refuse item; metric not applicable.",
        )

    answer_lower = record.answer.lower()
    has_cue = any(cue in answer_lower for cue in _REFUSAL_CUES)
    no_citations = len(record.citations) == 0

    passed = has_cue and no_citations
    score = 1.0 if passed else 0.0

    reasons: list[str] = []
    if not has_cue:
        reasons.append("no refusal cue found in answer")
    if not no_citations:
        reasons.append(f"{len(record.citations)} citation(s) present in refusal response")

    detail = "Correct refusal." if passed else "Incorrect: " + "; ".join(reasons) + "."
    return MetricResult(
        metric=metric_name,
        item_id=record.id,
        applicable=True,
        score=score,
        passed=passed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Metric 4: temporal_correctness
# ---------------------------------------------------------------------------


def temporal_correctness(record: RunRecord, golden: GoldenItem) -> MetricResult:
    """Whether all cited chunks come from the latest filing in retrieved.

    Applicable iff golden.bucket == temporal.
    The latest filing is the maximum filing_date (ISO string comparison) across
    all retrieved chunks. Passed iff every cited chunk has that date.
    """
    metric_name = "temporal_correctness"

    if golden.bucket != Bucket.temporal:
        return MetricResult(
            metric=metric_name,
            item_id=record.id,
            applicable=False,
            score=0.0,
            passed=False,
            detail="Not a temporal bucket item; metric not applicable.",
        )

    if not record.retrieved:
        return MetricResult(
            metric=metric_name,
            item_id=record.id,
            applicable=True,
            score=0.0,
            passed=False,
            detail="No retrieved chunks; cannot determine latest filing.",
        )

    max_date = max(ref.filing_date for ref in record.retrieved)
    retrieved_map = {ref.chunk_id: ref for ref in record.retrieved}
    cited_ids = list(record.citations.values())

    if not cited_ids:
        return MetricResult(
            metric=metric_name,
            item_id=record.id,
            applicable=True,
            score=0.0,
            passed=False,
            detail="No citations; cannot verify temporal correctness.",
        )

    superseded: list[str] = []
    for chunk_id in cited_ids:
        ref = retrieved_map.get(chunk_id)
        if ref is None:
            superseded.append(f"{chunk_id} not in retrieved")
            continue
        if ref.filing_date != max_date:
            superseded.append(
                f"{chunk_id} filing_date={ref.filing_date} (latest={max_date})"
            )

    passed = len(superseded) == 0
    score = 1.0 if passed else 0.0
    detail = (
        f"All cited chunks from latest filing ({max_date})."
        if passed
        else f"Superseded citations: {'; '.join(superseded)}."
    )
    return MetricResult(
        metric=metric_name,
        item_id=record.id,
        applicable=True,
        score=score,
        passed=passed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Metric 5: entity_disambiguation
# ---------------------------------------------------------------------------


def entity_disambiguation(record: RunRecord, golden: GoldenItem) -> MetricResult:
    """Whether the correct issuer is cited and named in the answer.

    Applicable iff golden.bucket == entity.

    The correct issuer is derived by finding, among record.retrieved, the chunk
    whose chunk_id starts with one of the golden.expected_sources prefixes —
    that chunk's issuer field is ground truth.

    Passed iff:
    (a) all cited chunks belong to the correct issuer, AND
    (b) the correct issuer name (or any of its significant tokens, i.e. capitalised
        tokens with length > 3) appears in record.answer.
    """
    metric_name = "entity_disambiguation"

    if golden.bucket != Bucket.entity:
        return MetricResult(
            metric=metric_name,
            item_id=record.id,
            applicable=False,
            score=0.0,
            passed=False,
            detail="Not an entity bucket item; metric not applicable.",
        )

    # Derive the correct issuer from the retrieved chunks that match expected_sources.
    correct_issuer: str | None = None
    for ref in record.retrieved:
        for expected_src in golden.expected_sources:
            if ref.chunk_id.startswith(expected_src):
                correct_issuer = ref.issuer
                break
        if correct_issuer is not None:
            break

    if correct_issuer is None:
        return MetricResult(
            metric=metric_name,
            item_id=record.id,
            applicable=True,
            score=0.0,
            passed=False,
            detail=(
                "Could not determine correct issuer: no retrieved chunk matched "
                f"expected_sources={golden.expected_sources}."
            ),
        )

    # (a) All cited chunks must belong to correct_issuer.
    retrieved_map = {ref.chunk_id: ref for ref in record.retrieved}
    wrong_issuer_citations: list[str] = []
    for marker, chunk_id in record.citations.items():
        ref = retrieved_map.get(chunk_id)
        if ref is None:
            wrong_issuer_citations.append(f"{marker}→{chunk_id} not found")
            continue
        if ref.issuer != correct_issuer:
            wrong_issuer_citations.append(
                f"{marker}→{chunk_id} issuer={ref.issuer!r} (expected {correct_issuer!r})"
            )

    cited_from_correct = len(wrong_issuer_citations) == 0

    # (b) Correct issuer name or a distinctive token appears in the answer.
    answer_lower = record.answer.lower()
    issuer_lower = correct_issuer.lower()
    distinctive_tokens = [
        tok for tok in correct_issuer.split() if tok[0].isupper() and len(tok) > 3
    ]
    issuer_in_answer = issuer_lower in answer_lower or any(
        tok.lower() in answer_lower for tok in distinctive_tokens
    )

    passed = cited_from_correct and issuer_in_answer
    score = 1.0 if passed else 0.0

    reasons: list[str] = []
    if wrong_issuer_citations:
        reasons.append(f"wrong issuer in citations: {'; '.join(wrong_issuer_citations)}")
    if not issuer_in_answer:
        reasons.append(f"issuer {correct_issuer!r} not mentioned in answer")

    detail = (
        f"Correct issuer {correct_issuer!r} cited and named."
        if passed
        else "Disambiguation failed: " + "; ".join(reasons) + "."
    )
    return MetricResult(
        metric=metric_name,
        item_id=record.id,
        applicable=True,
        score=score,
        passed=passed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Metric 6: context_recall
# ---------------------------------------------------------------------------


def context_recall(record: RunRecord, golden: GoldenItem) -> MetricResult:
    """Fraction of expected_sources for which some retrieved chunk_id starts with it.

    Applicable iff golden.expected_sources is non-empty.
    Passed iff recall ≥ 0.90 (config soft gate).
    """
    metric_name = "context_recall"

    if not golden.expected_sources:
        return MetricResult(
            metric=metric_name,
            item_id=record.id,
            applicable=False,
            score=0.0,
            passed=False,
            detail="No expected_sources defined; metric not applicable.",
        )

    retrieved_ids = [ref.chunk_id for ref in record.retrieved]
    matched = [
        es
        for es in golden.expected_sources
        if any(chunk_id.startswith(es) for chunk_id in retrieved_ids)
    ]
    recall = len(matched) / len(golden.expected_sources)
    passed = recall >= _RECALL_GATE

    detail = (
        f"Recall {recall:.3f} ({len(matched)}/{len(golden.expected_sources)} sources retrieved). "
        f"Missing: {[s for s in golden.expected_sources if s not in matched]}."
    )
    return MetricResult(
        metric=metric_name,
        item_id=record.id,
        applicable=True,
        score=recall,
        passed=passed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Metric 7: context_precision
# ---------------------------------------------------------------------------


def context_precision(record: RunRecord, golden: GoldenItem) -> MetricResult:
    """Fraction of retrieved chunks whose chunk_id starts with any expected_source.

    Applicable iff golden.expected_sources is non-empty.
    Passed iff precision ≥ 0.85 (config soft gate).
    """
    metric_name = "context_precision"

    if not golden.expected_sources:
        return MetricResult(
            metric=metric_name,
            item_id=record.id,
            applicable=False,
            score=0.0,
            passed=False,
            detail="No expected_sources defined; metric not applicable.",
        )

    if not record.retrieved:
        return MetricResult(
            metric=metric_name,
            item_id=record.id,
            applicable=True,
            score=0.0,
            passed=False,
            detail="No retrieved chunks.",
        )

    relevant = [
        ref
        for ref in record.retrieved
        if any(ref.chunk_id.startswith(es) for es in golden.expected_sources)
    ]
    precision = len(relevant) / len(record.retrieved)
    passed = precision >= _PRECISION_GATE

    irrelevant_ids = [
        ref.chunk_id for ref in record.retrieved if ref not in relevant
    ]
    detail = (
        f"Precision {precision:.3f} ({len(relevant)}/{len(record.retrieved)} relevant). "
        + (f"Irrelevant chunks: {irrelevant_ids}." if irrelevant_ids else "All relevant.")
    )
    return MetricResult(
        metric=metric_name,
        item_id=record.id,
        applicable=True,
        score=precision,
        passed=passed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_ALL_METRICS = [
    numerical_exactness,
    citation_validity,
    negative_rejection,
    temporal_correctness,
    entity_disambiguation,
    context_recall,
    context_precision,
]


def score_programmatic(
    records: list[RunRecord],
    goldens: list[GoldenItem],
) -> list[MetricResult]:
    """Run all 7 programmatic metrics over matched (record, golden) pairs.

    Records and goldens are matched by id. Raises KeyError if a record has no
    matching golden.

    Returns a flat list of MetricResult (7 results per item).
    """
    golden_map: dict[str, GoldenItem] = {g.id: g for g in goldens}
    results: list[MetricResult] = []

    for record in records:
        if record.id not in golden_map:
            raise KeyError(
                f"No golden found for record id={record.id!r}. "
                "All records must have a corresponding golden item."
            )
        golden = golden_map[record.id]
        for metric_fn in _ALL_METRICS:
            results.append(metric_fn(record, golden))

    return results


def aggregate_metric(
    results: list[MetricResult],
    metric_name: str,
) -> float | None:
    """Mean score over applicable MetricResults for the given metric.

    Returns None if no applicable results exist for that metric.
    """
    applicable = [r.score for r in results if r.metric == metric_name and r.applicable]
    if not applicable:
        return None
    return sum(applicable) / len(applicable)


def metric_rate(
    results: list[MetricResult],
    metric_name: str,
) -> float | None:
    """Fraction of applicable MetricResults that passed for the given metric.

    Returns None if no applicable results exist for that metric.
    Used for gate checks (T14).
    """
    applicable = [r for r in results if r.metric == metric_name and r.applicable]
    if not applicable:
        return None
    return sum(1 for r in applicable if r.passed) / len(applicable)
