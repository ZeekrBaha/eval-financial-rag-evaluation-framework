# implementation-plan.md — Financial RAG Evaluation Framework

Tasks are small, testable, and map to requirements + validation. Build order is dependency-ordered. Each task: files · steps · acceptance · tests · risk/rollback.

> Testing posture: **test-required + gated** (per skill). Tests accompany each task. If true TDD is desired, write the listed test first and watch it fail before implementing.

## Phase 0 — Scaffold

**T0. Repo + tooling**
- Files: `pyproject.toml`, `Makefile`, `.env.example`, `.gitignore`, `src/`, `tests/`, `README.md`.
- Steps: `uv init`; add deps (llama-index, chromadb, fastapi, uvicorn, ragas, deepeval, promptfoo via npx, pytest, pydantic, pyyaml); Make targets `ingest/run/score/eval/test`.
- Acceptance: `uv sync` ok; `make test` runs (0 tests ok); secret scan clean (N-02).
- Tests: smoke test imports.
- Risk: dep conflicts → pin versions.

**T1. Config + thresholds**
- Files: `src/config.py`.
- Steps: encode proposed hard/soft gates + dimension weights + k; single source of truth.
- Acceptance: importable; values match `requirements.md`.
- Tests: assert hard-gate set == {faithfulness, negative_rejection, hallucination, advice_boundary}.

## Phase 1 — SUT (RAG app)

**T2. Provider abstraction (live/offline)** — F-06
- Files: `src/sut/providers.py`, `datasets/fixtures/`.
- Steps: interface for embed() + generate(); `EVAL_MODE=offline` → fixture-backed deterministic.
- Acceptance: offline embed/generate identical across runs; no network in offline.
- Tests: determinism test; no-network assertion (monkeypatch).
- Risk: fixture drift → key by question hash.

**T3. Ingest + store** — F-01
- Files: `src/sut/ingest.py`, `src/sut/store.py`, `datasets/issuers.yaml`.
- Steps: pull EDGAR filings (user-agent, rate-limit), section-aware chunk, attach metadata, embed → Chroma.
- Acceptance: ≥5 issuers × (10-K+10-Q) ingested; each chunk has full metadata.
- Tests: chunk metadata schema test; ingest a fixture filing offline.
- Risk: EDGAR rate limits → cache raw downloads under `datasets/raw/`.

**T4. Retrieve** — F-02
- Files: `src/sut/retrieve.py`.
- Acceptance: top-k chunks + scores + metadata returned; k from config.
- Tests: retrieval returns expected fixture chunk for a known question.

**T5. Generate w/ citations + boundaries** — F-03/F-04/F-05
- Files: `src/sut/generate.py`, `src/sut/prompts.py`.
- Steps: prompt = answer-only-from-context + cite + refuse-if-absent + no-advice; return `{answer, citations}`.
- Acceptance: answers cite chunks; refuses on missing data; never emits advice.
- Tests: F-04 refusal case; F-05 advice-boundary case; citation mapping present.
- Risk: model ignores boundary → reinforce in system prompt + post-check in robustness metric.

**T6. FastAPI endpoint** — F-02/F-03
- Files: `src/sut/api.py`.
- Acceptance: `POST /query {question}` → `{answer, retrieved, citations, latency}`.
- Tests: TestClient happy path + refusal path.

## Phase 2 — Eval framework

**T7. Golden set loader** — E-01
- Files: `src/eval/golden.py`.
- Acceptance: loads + validates schema; per-bucket counts; rejects malformed rows.
- Tests: valid load; malformed row raises.

**T8. Author golden set (seed 50, scale to 200+)** — E-01
- Files: `datasets/golden_set.jsonl`.
- Steps: draft per bucket (mix per design §4), each with sources + reference + numeric/refuse/injection flags; manual review.
- Acceptance: ≥200 items, all buckets represented at target %.
- Tests: schema test over whole file; bucket-mix assertion within tolerance.
- Risk: authoring effort → start at 50 to unblock pipeline, grow.

**T9. Runner** — E-02
- Files: `src/eval/runner.py`.
- Acceptance: produces `run.jsonl` per contract.
- Tests: run over 3 fixture items → well-formed rows.

**T10. Programmatic metrics** — E-06 (E-03/E-04 parts)
- Files: `src/eval/metrics/programmatic.py`.
- Steps: numeric exactness (unit-normalized), citation validity (cited chunk supports claim), temporal, entity, negative-rejection.
- Acceptance: correct scores on labeled fixtures incl. a known-fail (citation 0.94).
- Tests: one per sub-metric, incl. deliberate fail case.
- Risk: numeric unit mismatch → normalization table + tests.

**T11. Judge metrics (+calibration hook)** — E-07
- Files: `src/eval/metrics/judge.py`.
- Steps: faithfulness/relevance/groundedness via Ragas/DeepEval; judge model+prompt versioned; offline stub.
- Acceptance: scores produced; judge config recorded in scorecard.
- Tests: offline stub deterministic; live path smoke (skipped without key).
- Risk: judge bias → calibration deferred to Portfolio Project 5; flag in report.

**T12. Robustness metrics** — E-05
- Files: `src/eval/metrics/robustness.py`.
- Steps: injection resistance (seeded items), advice-boundary, pass^k (k=5).
- Acceptance: injection items scored; advice violations caught; pass^k stable.
- Tests: injected item that SUT must ignore; advice-bait item.

**T13. Aggregate + scorecard** — E-08/E-10
- Files: `src/eval/aggregate.py`, `src/eval/scorecard.py`.
- Steps: dimension scores, weighted overall, per-bucket; render JSON + HTML (design-system tokens).
- Acceptance: scorecard.json + scorecard.html written; numbers in mono; states handled.
- Tests: aggregation math; HTML contains banner + dimension table.

**T14. Hard-gate enforcement (money shot)** — E-09
- Files: `src/eval/gates.py`.
- Steps: check hard gates; on fail print `RELEASE BLOCKED — <gate> <value> < <threshold>`; exit 1.
- Acceptance: passing run → exit 0; seeded failing run → exit 1 + correct message.
- Tests: both exit paths.
- Risk: false block → thresholds in config, documented as proposed.

## Phase 3 — Glue & polish

**T15. `make eval` end-to-end** — N-03
- Acceptance: one command ingest→run→score→gate; offline green, no secrets (N-01).
- Tests: subprocess test asserts exit code per fixtures.

**T16. README (honest framing)** — N-05
- Files: `README.md` (already drafted at repo root).
- Structure: follows the sibling hotel-bot harness's 14-section layout — (1) Mental model two-halves, (2) Where the SUT lives, (3) What the SUT does, (4) Metric stack, (5) Differentiator (hard gates + κ-calibrated judge), (6) Findings, (7) Golden data, (8) Reproducibility, (9) How to run, (10) Keys, (11) Repo map, (12) Why this stack, (13) Tech stack, (14) Limitations.
- Acceptance: public-data + designed-case-study framing; gates "proposed"; money-shot block; `(target)` markers replaced with measured numbers after first run; no unmeasured number presented as a result.

**T17. (Optional) Streamlit scorecard viewer**
- Acceptance: read-only viewer of `reports/<run-id>/scorecard.json`; off critical path.

**T18. (Stretch) Online drift note** — documented Phase 3 only.

## Requirement → task map

F-01→T3 · F-02→T4,T6 · F-03→T5,T6 · F-04→T5 · F-05→T5 · F-06→T2 · E-01→T7,T8 · E-02→T9 · E-03/04→T10,T11 · E-05→T12 · E-06→T10 · E-07→T11 · E-08→T13 · E-09→T14 · E-10→T13 · N-01→T2,T15 · N-02→T0 · N-03→T15 · N-04→all · N-05→T16.
