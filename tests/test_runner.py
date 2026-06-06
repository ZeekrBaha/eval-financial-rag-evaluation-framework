"""
Tests for src/eval/runner.py — T9: Runner (E-02).

TDD: tests written before implementation.
All tests run offline — no network, no API key required.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.eval.golden import Bucket, GoldenItem
from src.eval.runner import RunRecord, RetrievedRef, load_replay, run_live, write_run
from src.sut.ingest import ingest_fixture
from src.sut.store import VectorStore


# ---------------------------------------------------------------------------
# Paths to authored fixture datasets
# ---------------------------------------------------------------------------

_BASE = Path(__file__).parent.parent
_RUN_PASS = _BASE / "datasets" / "fixtures" / "run_pass.jsonl"
_RUN_FAIL = _BASE / "datasets" / "fixtures" / "run_fail.jsonl"


# ---------------------------------------------------------------------------
# Helper: build a minimal GoldenItem for offline tests
# ---------------------------------------------------------------------------


def _make_golden(
    id: str = "test-001",
    bucket: Bucket = Bucket.factual_lookup,
    question: str = "What was revenue?",
) -> GoldenItem:
    return GoldenItem(
        id=id,
        bucket=bucket,
        question=question,
        reference_answer="Revenue was $100M.",
        expected_sources=[],
        numeric_answers=[],
        must_refuse=False,
        injection=None,
        advice_boundary=False,
    )


# ---------------------------------------------------------------------------
# RetrievedRef model
# ---------------------------------------------------------------------------


class TestRetrievedRef:
    def test_has_all_contract_fields(self) -> None:
        ref = RetrievedRef(
            chunk_id="acc#item7#0",
            text="Some text.",
            similarity=0.91,
            issuer="AAPL",
            form="10-K",
            filing_date="2024-11-01",
            accession="acc-001",
            section="item7",
            source_url="https://example.com/doc",
        )
        assert ref.chunk_id == "acc#item7#0"
        assert ref.text == "Some text."
        assert ref.similarity == 0.91
        assert ref.issuer == "AAPL"
        assert ref.form == "10-K"
        assert ref.filing_date == "2024-11-01"
        assert ref.accession == "acc-001"
        assert ref.section == "item7"
        assert ref.source_url == "https://example.com/doc"


# ---------------------------------------------------------------------------
# RunRecord model
# ---------------------------------------------------------------------------


class TestRunRecord:
    def test_has_all_contract_fields(self) -> None:
        rec = RunRecord(
            id="fact-001",
            bucket="factual_lookup",
            question="What was revenue?",
            answer="Revenue was $58B [c1].",
            retrieved=[
                RetrievedRef(
                    chunk_id="NWM-10K-2024#item7#0",
                    text="Revenue text.",
                    similarity=0.94,
                    issuer="Northwind Motors Inc.",
                    form="10-K",
                    filing_date="2024-12-15",
                    accession="NWM-10K-2024",
                    section="item7",
                    source_url="https://example.invalid/nwm",
                )
            ],
            citations={"c1": "NWM-10K-2024#item7#0"},
            unmatched_citations=[],
            latency_ms=0,
            mode="replay",
        )
        assert rec.id == "fact-001"
        assert rec.mode == "replay"
        assert rec.latency_ms == 0
        assert isinstance(rec.retrieved, list)
        assert len(rec.retrieved) == 1

    def test_mode_must_be_live_or_replay(self) -> None:
        with pytest.raises(Exception):
            RunRecord(
                id="x",
                bucket="factual_lookup",
                question="Q?",
                answer="A.",
                retrieved=[],
                citations={},
                unmatched_citations=[],
                latency_ms=0,
                mode="invalid_mode",
            )

    def test_latency_ms_non_negative(self) -> None:
        # Negative latency should fail validation
        with pytest.raises(Exception):
            RunRecord(
                id="x",
                bucket="factual_lookup",
                question="Q?",
                answer="A.",
                retrieved=[],
                citations={},
                unmatched_citations=[],
                latency_ms=-1,
                mode="replay",
            )


# ---------------------------------------------------------------------------
# load_replay — run_pass.jsonl
# ---------------------------------------------------------------------------


class TestLoadReplayPass:
    def test_returns_list_of_runrecords(self) -> None:
        records = load_replay(_RUN_PASS)
        assert isinstance(records, list)
        assert all(isinstance(r, RunRecord) for r in records)

    def test_count_is_21(self) -> None:
        records = load_replay(_RUN_PASS)
        assert len(records) == 21

    def test_all_contract_fields_present(self) -> None:
        records = load_replay(_RUN_PASS)
        for rec in records:
            assert isinstance(rec.id, str) and rec.id
            assert isinstance(rec.bucket, str) and rec.bucket
            assert isinstance(rec.question, str) and rec.question
            assert isinstance(rec.answer, str)
            assert isinstance(rec.retrieved, list)
            assert isinstance(rec.citations, dict)
            assert isinstance(rec.unmatched_citations, list)
            assert isinstance(rec.latency_ms, int)
            assert rec.mode in ("live", "replay")

    def test_retrieved_items_have_all_fields(self) -> None:
        records = load_replay(_RUN_PASS)
        for rec in records:
            for ref in rec.retrieved:
                assert isinstance(ref.chunk_id, str)
                assert isinstance(ref.text, str)
                assert isinstance(ref.similarity, float)
                assert isinstance(ref.issuer, str)
                assert isinstance(ref.form, str)
                assert isinstance(ref.filing_date, str)
                assert isinstance(ref.accession, str)
                assert isinstance(ref.section, str)
                assert isinstance(ref.source_url, str)

    def test_first_row_matches_authored_fixture(self) -> None:
        records = load_replay(_RUN_PASS)
        first = records[0]
        assert first.id == "fact-001"
        assert first.bucket == "factual_lookup"
        assert first.mode == "replay"
        assert first.latency_ms == 0
        assert len(first.retrieved) == 1
        assert first.retrieved[0].chunk_id == "NWM-10K-2024#item7#0"
        assert first.retrieved[0].similarity == 0.94
        assert "c1" in first.citations
        assert first.citations["c1"] == "NWM-10K-2024#item7#0"

    def test_citations_chunk_ids_in_retrieved(self) -> None:
        """Every citation value must be a chunk_id present in that record's retrieved list."""
        records = load_replay(_RUN_PASS)
        for rec in records:
            retrieved_ids = {ref.chunk_id for ref in rec.retrieved}
            for marker, chunk_id in rec.citations.items():
                assert chunk_id in retrieved_ids, (
                    f"Record {rec.id!r}: citation {marker!r} → {chunk_id!r} not in retrieved"
                )

    def test_all_modes_are_replay(self) -> None:
        records = load_replay(_RUN_PASS)
        assert all(r.mode == "replay" for r in records)


# ---------------------------------------------------------------------------
# load_replay — run_fail.jsonl
# ---------------------------------------------------------------------------


class TestLoadReplayFail:
    def test_returns_21_records(self) -> None:
        records = load_replay(_RUN_FAIL)
        assert len(records) == 21

    def test_all_contract_fields_present(self) -> None:
        records = load_replay(_RUN_FAIL)
        for rec in records:
            assert isinstance(rec.id, str)
            assert isinstance(rec.bucket, str)
            assert isinstance(rec.question, str)
            assert isinstance(rec.citations, dict)
            assert isinstance(rec.unmatched_citations, list)
            assert isinstance(rec.latency_ms, int)
            assert rec.mode in ("live", "replay")

    def test_citations_chunk_ids_in_retrieved(self) -> None:
        records = load_replay(_RUN_FAIL)
        for rec in records:
            retrieved_ids = {ref.chunk_id for ref in rec.retrieved}
            for marker, chunk_id in rec.citations.items():
                assert chunk_id in retrieved_ids, (
                    f"Record {rec.id!r}: citation {marker!r} → {chunk_id!r} not in retrieved"
                )


# ---------------------------------------------------------------------------
# load_replay — malformed input
# ---------------------------------------------------------------------------


class TestLoadReplayMalformed:
    def test_missing_required_field_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        # Missing 'mode' field
        bad.write_text(json.dumps({
            "id": "x", "bucket": "factual_lookup", "question": "Q?",
            "answer": "A.", "retrieved": [], "citations": {},
            "unmatched_citations": [], "latency_ms": 0,
            # mode intentionally omitted
        }) + "\n")
        with pytest.raises(ValueError):
            load_replay(bad)

    def test_bad_json_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("not valid json\n")
        with pytest.raises(ValueError):
            load_replay(bad)

    def test_invalid_mode_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps({
            "id": "x", "bucket": "factual_lookup", "question": "Q?",
            "answer": "A.", "retrieved": [], "citations": {},
            "unmatched_citations": [], "latency_ms": 0,
            "mode": "broken",
        }) + "\n")
        with pytest.raises(ValueError):
            load_replay(bad)

    def test_error_message_is_clear(self, tmp_path: Path) -> None:
        """The ValueError message must mention the line number."""
        bad = tmp_path / "bad.jsonl"
        bad.write_text("not valid json\n")
        with pytest.raises(ValueError, match=r"line 1"):
            load_replay(bad)

    def test_extra_field_raises_value_error(self, tmp_path: Path) -> None:
        """RunRecord uses extra='forbid'; an unknown key must raise ValueError.

        This locks the extra=forbid contract so accidental schema drift is caught
        at load time rather than silently ignored.
        """
        bad = tmp_path / "extra_field.jsonl"
        row = {
            "id": "fact-001",
            "bucket": "factual_lookup",
            "question": "What was revenue?",
            "answer": "Revenue was $58B [c1].",
            "retrieved": [
                {
                    "chunk_id": "NWM-10K-2024#item7#0",
                    "text": "For fiscal year ended October 31, 2024, Northwind Motors Inc. recorded total revenue of $58,420 million.",
                    "similarity": 0.94,
                    "issuer": "Northwind Motors Inc.",
                    "form": "10-K",
                    "filing_date": "2024-12-15",
                    "accession": "NWM-10K-2024",
                    "section": "item7",
                    "source_url": "https://example.invalid/nwm-10k-2024",
                }
            ],
            "citations": {"c1": "NWM-10K-2024#item7#0"},
            "unmatched_citations": [],
            "latency_ms": 0,
            "mode": "replay",
            "surprise": "x",  # unknown key — must be rejected
        }
        bad.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError):
            load_replay(bad)

    def test_multi_line_error_reports_physical_line_number(self, tmp_path: Path) -> None:
        """load_replay must report the PHYSICAL line number of the bad row.

        File layout (1-indexed):
          line 1 — valid row
          line 2 — valid row
          line 3 — blank line (skipped by load_replay)
          line 4 — invalid row (missing required 'id' field)

        The raised ValueError must mention 'line 4'.
        """
        bad = tmp_path / "multiline.jsonl"
        valid_row = {
            "id": "fact-001",
            "bucket": "factual_lookup",
            "question": "What was revenue?",
            "answer": "Revenue was $58B [c1].",
            "retrieved": [
                {
                    "chunk_id": "NWM-10K-2024#item7#0",
                    "text": "For fiscal year ended October 31, 2024, Northwind Motors Inc. recorded total revenue of $58,420 million.",
                    "similarity": 0.94,
                    "issuer": "Northwind Motors Inc.",
                    "form": "10-K",
                    "filing_date": "2024-12-15",
                    "accession": "NWM-10K-2024",
                    "section": "item7",
                    "source_url": "https://example.invalid/nwm-10k-2024",
                }
            ],
            "citations": {"c1": "NWM-10K-2024#item7#0"},
            "unmatched_citations": [],
            "latency_ms": 0,
            "mode": "replay",
        }
        invalid_row = {
            # 'id' is intentionally omitted — required field missing
            "bucket": "factual_lookup",
            "question": "What was operating income?",
            "answer": "Operating income was $12B.",
            "retrieved": [],
            "citations": {},
            "unmatched_citations": [],
            "latency_ms": 0,
            "mode": "replay",
        }
        content = (
            json.dumps(valid_row) + "\n"   # line 1
            + json.dumps(valid_row) + "\n"  # line 2
            + "\n"                          # line 3 — blank
            + json.dumps(invalid_row) + "\n"  # line 4 — bad row
        )
        bad.write_text(content)
        with pytest.raises(ValueError, match=r"line 4"):
            load_replay(bad)


# ---------------------------------------------------------------------------
# run_live — offline (no key, no network)
# ---------------------------------------------------------------------------


class TestRunLive:
    def _build_store_and_golden(self) -> tuple[VectorStore, list[GoldenItem]]:
        """Ingest a tiny fixture and build matching GoldenItems."""
        store = VectorStore()
        # Create a tiny fixture inline via a temp file approach is not needed —
        # we call ingest_fixture on the real fixture text file if one exists,
        # or we add chunks directly.
        from src.sut.ingest import Chunk

        chunks = [
            Chunk(
                text="Northwind Motors reported revenue of $58B in FY2024.",
                issuer="Northwind Motors",
                form="10-K",
                filing_date="2024-12-15",
                accession="NWM-test-001",
                section="item7",
                source_url="https://example.invalid/nwm",
                chunk_id="NWM-test-001#item7#0",
            ),
            Chunk(
                text="Operating income was $12B for fiscal 2024.",
                issuer="Northwind Motors",
                form="10-K",
                filing_date="2024-12-15",
                accession="NWM-test-001",
                section="item7",
                source_url="https://example.invalid/nwm",
                chunk_id="NWM-test-001#item7#1",
            ),
        ]
        store.add(chunks)

        goldens = [
            _make_golden(id="test-001", question="What was revenue?"),
            _make_golden(id="test-002", bucket=Bucket.temporal, question="What was operating income?"),
        ]
        return store, goldens

    def test_returns_list_of_runrecords(self) -> None:
        store, goldens = self._build_store_and_golden()
        records = run_live(goldens, store)
        assert isinstance(records, list)
        assert all(isinstance(r, RunRecord) for r in records)

    def test_count_matches_goldens(self) -> None:
        store, goldens = self._build_store_and_golden()
        records = run_live(goldens, store)
        assert len(records) == len(goldens)

    def test_mode_is_live(self) -> None:
        store, goldens = self._build_store_and_golden()
        records = run_live(goldens, store)
        assert all(r.mode == "live" for r in records)

    def test_ids_match_goldens(self) -> None:
        store, goldens = self._build_store_and_golden()
        records = run_live(goldens, store)
        assert [r.id for r in records] == [g.id for g in goldens]

    def test_questions_match_goldens(self) -> None:
        store, goldens = self._build_store_and_golden()
        records = run_live(goldens, store)
        assert [r.question for r in records] == [g.question for g in goldens]

    def test_bucket_matches_golden_value(self) -> None:
        store, goldens = self._build_store_and_golden()
        records = run_live(goldens, store)
        for rec, golden in zip(records, goldens):
            assert rec.bucket == golden.bucket.value

    def test_retrieved_is_populated(self) -> None:
        store, goldens = self._build_store_and_golden()
        records = run_live(goldens, store)
        for rec in records:
            assert len(rec.retrieved) > 0
            for ref in rec.retrieved:
                assert isinstance(ref, RetrievedRef)

    def test_latency_ms_is_non_negative_number(self) -> None:
        store, goldens = self._build_store_and_golden()
        records = run_live(goldens, store)
        for rec in records:
            assert isinstance(rec.latency_ms, int)
            assert rec.latency_ms >= 0

    def test_retrieved_refs_have_similarity_not_score(self) -> None:
        """RetrievedRef uses similarity (0-1, higher = more relevant), not raw Chroma distance."""
        store, goldens = self._build_store_and_golden()
        records = run_live(goldens, store)
        for rec in records:
            for ref in rec.retrieved:
                # similarity must be between 0 and 1
                assert 0.0 <= ref.similarity <= 1.0

    def test_answer_is_string(self) -> None:
        store, goldens = self._build_store_and_golden()
        records = run_live(goldens, store)
        for rec in records:
            assert isinstance(rec.answer, str)

    def test_citations_is_dict(self) -> None:
        store, goldens = self._build_store_and_golden()
        records = run_live(goldens, store)
        for rec in records:
            assert isinstance(rec.citations, dict)

    def test_unmatched_citations_is_list(self) -> None:
        store, goldens = self._build_store_and_golden()
        records = run_live(goldens, store)
        for rec in records:
            assert isinstance(rec.unmatched_citations, list)


# ---------------------------------------------------------------------------
# write_run + load_replay roundtrip
# ---------------------------------------------------------------------------


class TestWriteRunRoundtrip:
    def test_write_and_reload_preserves_key_fields(self, tmp_path: Path) -> None:
        original = load_replay(_RUN_PASS)
        out = tmp_path / "run_out.jsonl"
        write_run(original, out)

        reloaded = load_replay(out)
        assert len(reloaded) == len(original)
        for orig, rel in zip(original, reloaded):
            assert orig.id == rel.id
            assert orig.bucket == rel.bucket
            assert orig.question == rel.question
            assert orig.answer == rel.answer
            assert orig.mode == rel.mode
            assert orig.latency_ms == rel.latency_ms
            assert orig.citations == rel.citations
            assert orig.unmatched_citations == rel.unmatched_citations

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        records = load_replay(_RUN_PASS)
        nested = tmp_path / "a" / "b" / "c" / "run.jsonl"
        write_run(records, nested)
        assert nested.exists()

    def test_written_file_is_valid_jsonl(self, tmp_path: Path) -> None:
        records = load_replay(_RUN_PASS)
        out = tmp_path / "run.jsonl"
        write_run(records, out)
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == len(records)
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            assert "id" in parsed
            assert "mode" in parsed
            assert "retrieved" in parsed

    def test_roundtrip_retrieved_similarity_preserved(self, tmp_path: Path) -> None:
        original = load_replay(_RUN_PASS)
        out = tmp_path / "run_rt.jsonl"
        write_run(original, out)
        reloaded = load_replay(out)
        for orig, rel in zip(original, reloaded):
            assert len(orig.retrieved) == len(rel.retrieved)
            for o_ref, r_ref in zip(orig.retrieved, rel.retrieved):
                assert o_ref.chunk_id == r_ref.chunk_id
                assert o_ref.similarity == r_ref.similarity
