# research.md — Financial RAG Evaluation Framework

> Spec-first research. Decisions live in `design.md`, not here. Options below are marked.

## 1. User goal

Build a **portfolio-grade evaluation framework** for a financial RAG system, plus the RAG **System Under Test (SUT)** it evaluates. The deliverable proves I can build the test infrastructure that decides whether a GenAI system is safe to ship — grounded, gated, calibrated — on public data with my own baselines.

Two halves, both shipped in this repo:
1. **SUT** — a RAG assistant answering analyst questions over public SEC filings, with inline citations.
2. **Eval framework** — scores the SUT on grounding/retrieval/financial-correctness/robustness, renders a scorecard, and **blocks release** on hard-gate failure.

## 2. Audience

- Primary: hiring managers / recruiters for **AI QA Engineer → AI Evaluation Engineer → Test Infrastructure Engineer** roles.
- Secondary: me, as a reusable harness for later portfolio projects.

## 3. Success criteria

- `Evidence` One command ingests filings, answers the golden set, scores it, renders an HTML+JSON scorecard, and **exits non-zero** if any hard gate fails.
- `Evidence` Golden set ≥ 200 items (scale toward 500), across all 7 buckets.
- `Evidence` The "money shot": a scorecard that prints `RELEASE BLOCKED — citation validity 0.94 < 0.95 hard gate`.
- `Evidence` Reproducible without paid keys via a deterministic offline/fixture mode (CI runs green with no secrets).
- `Assumption` A live mode (OpenAI) demonstrates real scoring for the README screenshots.

## 4. Constraints

- `Constraint` **Stack override:** skill default is Next.js/web; this is a Python eval framework. Use **Python 3.12 + uv + FastAPI + pytest**. Recorded reason: the SUT and all eval tooling (Ragas, DeepEval, Promptfoo) are Python-native; a JS stack would add friction for zero benefit.
- `Constraint` **Public data only** — SEC EDGAR 10-K / 10-Q. No proprietary or Moody's data.
- `Constraint` **Honest framing** — Moody's is a *designed case study on a public product*, never insider knowledge. All gates labeled "proposed, calibrate against baseline."
- `Constraint` **No client-side secrets.** API keys in env vars only; documented by name. CI runs in offline/fixture mode.
- `Constraint` Deterministic, reproducible demos — fixtures for retrieval + LLM in test mode.

## 5. Data sources

- `Evidence` SEC EDGAR full-text + filing archive is public and free (`https://www.sec.gov/edgar`). Bulk/company filings accessible via the EDGAR REST API and the full-text search system.
- `Assumption` Start with 5–10 large issuers across 2 sectors (e.g. auto: F, GM; tech: AAPL, MSFT) — recent 10-K + most recent 10-Q each. Enough for temporal + entity buckets.
- `Repository fact` Empty repo; folder `eval-financial-rag-evaluation-framework` created, no code yet.

## 6. APIs / models (options — decided in design.md)

- Embeddings: `text-embedding-3-small` (OpenAI) — option A; `bge-small-en` local — option B (offline).
- Generator: `gpt-4o-mini` — option A; local small model — option B.
- Judge model: a stronger model than the generator (e.g. `gpt-4o`) to reduce self-preference bias; **must be calibrated** (links to Portfolio Project 5).
- Vector store: Chroma (local, no server) — option A; Postgres `pgvector` — option B (scale).
- Eval libs: Ragas + DeepEval (metrics), Promptfoo (regression/CI), pytest (harness).

## 7. Risks / unknowns

- `Risk` Judge bias (position/verbosity/self-preference) inflates faithfulness scores → mitigate with calibration sample + stronger judge.
- `Risk` Numerical claims in filings are easy to misread (units, thousands vs millions) → programmatic numeric extraction + exact-match check, not judge.
- `Risk` Temporal correctness: superseded figures across 10-K vs 10-Q → tag each chunk with filing date; test latest-vs-superseded explicitly.
- `Risk` Prompt-injection via filing text (rare in real filings but seedable) → inject test fixtures, measure resistance.
- `Unknown` Exact golden-set authoring effort for 200+ expert-quality items; mitigate with semi-automated draft + manual review.
- `Unknown` Whether to include the Streamlit dashboard in v1 or defer (see design non-goals).
