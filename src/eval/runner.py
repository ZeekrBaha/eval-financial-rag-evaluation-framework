"""
runner.py — SUT runner for the Financial RAG evaluation framework (T9 / E-02).

Public surface:
  RetrievedRef  — Pydantic model for a single retrieved chunk in a run record.
  RunRecord     — Pydantic model for one row of a run.jsonl file.
  load_replay() — Parse a recorded run.jsonl; validate against contract. Offline/CI path.
  run_live()    — Run goldens through the live SUT; produce RunRecords with timing.
  write_run()   — Write a list of RunRecords to a JSONL file.

Single responsibility: produce and persist RunRecords. No scoring here.

Usage::

    # Offline/CI — parse a recorded run
    from src.eval.runner import load_replay
    records = load_replay("datasets/fixtures/run_pass.jsonl")

    # Live — run goldens through the SUT
    from src.eval.runner import run_live, write_run
    from src.eval.golden import load_goldens
    from src.sut.store import VectorStore

    goldens = load_goldens("datasets/golden_set.jsonl")
    store = VectorStore(persist_path="datasets/chroma")
    records = run_live(goldens, store)
    write_run(records, "datasets/run.jsonl")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from src.eval.golden import GoldenItem
from src.sut.generate import answer_question
from src.sut.providers import Provider
from src.sut.store import VectorStore


# ---------------------------------------------------------------------------
# RetrievedRef — a single retrieved chunk in a RunRecord
# ---------------------------------------------------------------------------


class RetrievedRef(BaseModel):
    """One retrieved chunk as serialised in run.jsonl.

    Uses ``similarity`` (1 - cosine distance, higher = more relevant) rather
    than the raw Chroma distance score, matching the authored fixture contract.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    similarity: float
    issuer: str
    form: str
    filing_date: str
    accession: str
    section: str
    source_url: str


# ---------------------------------------------------------------------------
# RunRecord — one row of run.jsonl
# ---------------------------------------------------------------------------


class RunRecord(BaseModel):
    """One row of a run.jsonl file, matching the authored fixture contract exactly.

    Fields
    ------
    id:                  Golden item id.
    bucket:              Bucket name string (e.g. "factual_lookup").
    question:            The question posed to the SUT.
    answer:              The SUT's answer text (may contain [cN] citation markers).
    retrieved:           Ordered list of retrieved chunks (closest first).
    citations:           Mapping from marker key (e.g. "c1") to chunk_id.
    unmatched_citations: Markers that appear in answer but lack a retrieved passage.
    latency_ms:          Wall-clock time for the SUT call, in milliseconds (≥ 0).
    mode:                "live" for real SUT calls, "replay" for recorded runs.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    bucket: str
    question: str
    answer: str
    retrieved: list[RetrievedRef]
    citations: dict[str, str]
    unmatched_citations: list[str]
    latency_ms: int = Field(ge=0)
    mode: Literal["live", "replay"]


# ---------------------------------------------------------------------------
# load_replay — offline / CI path
# ---------------------------------------------------------------------------


def load_replay(path: str | Path) -> list[RunRecord]:
    """Parse a recorded run.jsonl and return validated RunRecords.

    Args:
        path: Path to a run.jsonl file (run_pass.jsonl, run_fail.jsonl, etc.).

    Returns:
        List of RunRecord, one per non-blank line.

    Raises:
        ValueError: On malformed JSON, missing required fields, or invalid field
                    values — the message always includes the 1-based line number.
    """
    path = Path(path)
    records: list[RunRecord] = []

    with path.open() as fh:
        for lineno, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {lineno} of {path.name}: {exc}"
                ) from exc

            try:
                record = RunRecord.model_validate(data)
            except pydantic.ValidationError as exc:
                raise ValueError(
                    f"Validation error on line {lineno} of {path.name}: {exc}"
                ) from exc

            records.append(record)

    return records


# ---------------------------------------------------------------------------
# run_live — live SUT execution path
# ---------------------------------------------------------------------------


def run_live(
    goldens: list[GoldenItem],
    store: VectorStore,
    provider: Provider | None = None,
) -> list[RunRecord]:
    """Run each golden through the SUT and return RunRecords.

    Args:
        goldens:  List of GoldenItems from load_goldens().
        store:    VectorStore loaded with the target filing corpus.
        provider: Optional Provider to inject (defaults to get_provider() inside
                  answer_question, which uses OfflineProvider when no key is set).

    Returns:
        List of RunRecord with mode="live" and real latency_ms values.
    """
    records: list[RunRecord] = []

    for item in goldens:
        t0 = time.monotonic()
        answer = answer_question(item.question, store, provider=provider)
        latency_ms = int((time.monotonic() - t0) * 1000)

        retrieved_refs = [
            RetrievedRef(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                similarity=chunk.similarity,
                issuer=chunk.issuer,
                form=chunk.form,
                filing_date=chunk.filing_date,
                accession=chunk.accession,
                section=chunk.section,
                source_url=chunk.source_url,
            )
            for chunk in answer.retrieved
        ]

        record = RunRecord(
            id=item.id,
            bucket=item.bucket.value,
            question=item.question,
            answer=answer.answer,
            retrieved=retrieved_refs,
            citations=answer.citations,
            unmatched_citations=answer.unmatched_citations,
            latency_ms=latency_ms,
            mode="live",
        )
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# write_run — persist run records
# ---------------------------------------------------------------------------


def write_run(records: list[RunRecord], path: str | Path) -> None:
    """Write RunRecords to a JSONL file, creating parent directories as needed.

    Args:
        records: List of RunRecord to serialise.
        path:    Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# load_run — general alias (load_replay is the named offline entry point)
# ---------------------------------------------------------------------------

load_run = load_replay
