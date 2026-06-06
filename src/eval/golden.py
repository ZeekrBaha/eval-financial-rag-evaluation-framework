"""Golden set loader for E-01.

Parses a JSONL file where each line is a :class:`GoldenItem`, validates every
field, and raises clear errors (including line numbers) on any violation.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator


class Bucket(str, Enum):
    factual_lookup = "factual_lookup"
    multi_source = "multi_source"
    temporal = "temporal"
    negative = "negative"
    entity = "entity"
    adversarial = "adversarial"
    long_context = "long_context"


class GoldenItem(BaseModel):
    """One row from the golden-set JSONL file."""

    id: str
    bucket: Bucket
    question: str
    reference_answer: str
    expected_sources: list[str]
    numeric_answers: list[str]
    must_refuse: bool
    injection: Optional[str]
    advice_boundary: bool

    @field_validator("id", "question", "reference_answer")
    @classmethod
    def must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field must be a non-empty string")
        return v

    # Expose bucket as plain string for convenience while keeping Enum validation.
    @property
    def bucket_str(self) -> str:
        return self.bucket.value


def load_goldens(path: str | Path) -> list[GoldenItem]:
    """Parse *path* (JSONL) and return a validated list of :class:`GoldenItem`.

    Raises
    ------
    ValueError
        On malformed JSON, unknown bucket, duplicate id, or invalid field — the
        message always includes the 1-based line number.
    """
    path = Path(path)
    items: list[GoldenItem] = []
    seen_ids: set[str] = set()

    with path.open() as fh:
        for lineno, raw in enumerate(fh, start=1):
            # Skip blank / whitespace-only lines.
            if not raw.strip():
                continue

            # Parse JSON.
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {lineno}: {exc}") from exc

            # Validate with Pydantic.
            try:
                item = GoldenItem.model_validate(data)
            except Exception as exc:
                raise ValueError(f"Validation error on line {lineno}: {exc}") from exc

            # Enforce unique ids.
            if item.id in seen_ids:
                raise ValueError(
                    f"Duplicate id '{item.id}' found on line {lineno}"
                )
            seen_ids.add(item.id)

            items.append(item)

    return items


def bucket_counts(items: list[GoldenItem]) -> dict[str, int]:
    """Return a mapping of bucket name → item count."""
    counts: dict[str, int] = {}
    for item in items:
        key = item.bucket.value
        counts[key] = counts.get(key, 0) + 1
    return counts
