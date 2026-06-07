"""Tests for the golden dataset fixtures (T8).

Validates:
- golden_set.jsonl parses cleanly via load_goldens
- At least 3 items per bucket
- run_pass.jsonl and run_fail.jsonl each have exactly one row per golden id
- Each run row has the required contract fields
- Every citation marker maps to a chunk_id present in that row's retrieved list
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.eval.golden import bucket_counts, load_goldens

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
GOLDEN_SET = DATASETS_DIR / "golden_set.jsonl"
RUN_PASS = DATASETS_DIR / "fixtures" / "run_pass.jsonl"
RUN_FAIL = DATASETS_DIR / "fixtures" / "run_fail.jsonl"

# Required fields on every run row (the run contract).
RUN_CONTRACT_FIELDS = {
    "id",
    "bucket",
    "question",
    "answer",
    "retrieved",
    "citations",
    "unmatched_citations",
    "latency_ms",
    "mode",
}

MIN_PER_BUCKET = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_run_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a run JSONL file and return a list of row dicts."""
    rows = []
    with path.open() as fh:
        for lineno, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                pytest.fail(f"Malformed JSON on line {lineno} of {path.name}: {exc}")
    return rows


# ---------------------------------------------------------------------------
# golden_set.jsonl — schema + bucket coverage
# ---------------------------------------------------------------------------


def test_golden_set_loads_clean() -> None:
    """golden_set.jsonl must parse without errors via load_goldens."""
    items = load_goldens(GOLDEN_SET)
    assert len(items) > 0, "golden_set.jsonl must have at least one item"


def test_golden_set_min_three_per_bucket() -> None:
    """Every bucket must have at least 3 items."""
    items = load_goldens(GOLDEN_SET)
    counts = bucket_counts(items)
    all_buckets = [
        "factual_lookup",
        "multi_source",
        "temporal",
        "negative",
        "entity",
        "adversarial",
        "long_context",
    ]
    for bucket in all_buckets:
        count = counts.get(bucket, 0)
        assert count >= MIN_PER_BUCKET, (
            f"Bucket '{bucket}' has {count} items, need >= {MIN_PER_BUCKET}"
        )


def test_golden_set_all_seven_buckets_present() -> None:
    """All 7 bucket types must be represented."""
    items = load_goldens(GOLDEN_SET)
    counts = bucket_counts(items)
    assert len(counts) == 7, f"Expected 7 buckets, got {sorted(counts.keys())}"


# ---------------------------------------------------------------------------
# run_pass.jsonl — contract conformance
# ---------------------------------------------------------------------------


def test_run_pass_one_row_per_golden_id() -> None:
    """run_pass.jsonl must have exactly one row per golden id."""
    golden_ids = {item.id for item in load_goldens(GOLDEN_SET)}
    rows = load_run_jsonl(RUN_PASS)
    run_ids = {row["id"] for row in rows}

    missing = golden_ids - run_ids
    extra = run_ids - golden_ids
    assert not missing, f"run_pass.jsonl missing golden ids: {sorted(missing)}"
    assert not extra, f"run_pass.jsonl has extra ids not in golden set: {sorted(extra)}"
    assert len(rows) == len(golden_ids), (
        f"run_pass.jsonl has {len(rows)} rows but golden set has {len(golden_ids)} ids "
        "(possible duplicate rows)"
    )


def test_run_pass_contract_fields() -> None:
    """Every row in run_pass.jsonl must have exactly the required contract fields."""
    rows = load_run_jsonl(RUN_PASS)
    for row in rows:
        missing = RUN_CONTRACT_FIELDS - set(row.keys())
        assert not missing, (
            f"run_pass row '{row.get('id', '?')}' missing contract fields: {missing}"
        )


def test_run_pass_citations_resolve_to_retrieved() -> None:
    """Every citation marker in run_pass must map to a chunk_id in that row's retrieved."""
    rows = load_run_jsonl(RUN_PASS)
    for row in rows:
        retrieved_ids = {chunk["chunk_id"] for chunk in row["retrieved"]}
        for marker, chunk_id in row["citations"].items():
            assert chunk_id in retrieved_ids, (
                f"run_pass row '{row['id']}': citation [{marker}] -> '{chunk_id}' "
                f"is not in retrieved chunk_ids {retrieved_ids}"
            )


# ---------------------------------------------------------------------------
# run_fail.jsonl — contract conformance
# ---------------------------------------------------------------------------


def test_run_fail_one_row_per_golden_id() -> None:
    """run_fail.jsonl must have exactly one row per golden id."""
    golden_ids = {item.id for item in load_goldens(GOLDEN_SET)}
    rows = load_run_jsonl(RUN_FAIL)
    run_ids = {row["id"] for row in rows}

    missing = golden_ids - run_ids
    extra = run_ids - golden_ids
    assert not missing, f"run_fail.jsonl missing golden ids: {sorted(missing)}"
    assert not extra, f"run_fail.jsonl has extra ids not in golden set: {sorted(extra)}"
    assert len(rows) == len(golden_ids), (
        f"run_fail.jsonl has {len(rows)} rows but golden set has {len(golden_ids)} ids "
        "(possible duplicate rows)"
    )


def test_run_fail_contract_fields() -> None:
    """Every row in run_fail.jsonl must have exactly the required contract fields."""
    rows = load_run_jsonl(RUN_FAIL)
    for row in rows:
        missing = RUN_CONTRACT_FIELDS - set(row.keys())
        assert not missing, (
            f"run_fail row '{row.get('id', '?')}' missing contract fields: {missing}"
        )


def test_run_fail_citations_resolve_to_retrieved() -> None:
    """Every citation marker in run_fail must map to a chunk_id in that row's retrieved.

    Note: run_fail has broken CITATION CONTENT (chunk text doesn't support the claim)
    but the chunk_id still exists in retrieved — citation pointer integrity must hold
    even for the fail fixture.
    """
    rows = load_run_jsonl(RUN_FAIL)
    for row in rows:
        retrieved_ids = {chunk["chunk_id"] for chunk in row["retrieved"]}
        for marker, chunk_id in row["citations"].items():
            assert chunk_id in retrieved_ids, (
                f"run_fail row '{row['id']}': citation [{marker}] -> '{chunk_id}' "
                f"is not in retrieved chunk_ids {retrieved_ids}"
            )
