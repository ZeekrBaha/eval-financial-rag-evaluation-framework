# design.md — Financial RAG Evaluation Framework

Decisions made here. Options were enumerated in `research.md`.

## 1. Decided stack

| Layer | Choice | Why (vs alternative) |
|-------|--------|----------------------|
| Language | Python 3.12 | All eval tooling is Python-native. |
| Pkg mgr | uv | Matches existing portfolio convention; fast, lockfile reproducibility. |
| Ingestion + retrieval | direct `chromadb` + a section-aware word-window chunker | **Revised during T3 (was LlamaIndex):** F-01 never required LlamaIndex; direct chromadb is simpler, fewer deps, and keeps offline determinism clean. LlamaIndex removed from deps. |
| Vector store | Chroma (local, persisted) | No DB server → reproducible demo. `pgvector` documented as scale path. |
| Embeddings | `text-embedding-3-small` (live) / `bge-small-en` (offline) | Cost + offline determinism. |
| Generator | `gpt-4o-mini` (live) / fixture (offline) | Cheap; fixtures for CI. |
| Judge | `gpt-4o` (stronger than generator) | Reduce self-preference bias; must be calibrated. |
| Query API | FastAPI | Standard, demoable, testable. |
| Eval metrics | Ragas + DeepEval | RAG-core + custom metrics. |
| Regression/CI | Promptfoo | Threshold gating in CI (feeds Portfolio Project 2). |
| Tests | pytest | Harness + unit/integration. |
| Dashboard | Streamlit (Phase 2, optional) | Read-only scorecard viewer; not on critical path. |

## 2. SUT pipeline (data flow)

```
issuers.yaml ──> ingest ──> parse+chunk ──> embed ──> Chroma
                                                         │
question ──> retrieve(top-k) ──> build prompt ──> generator ──> answer + citations
                                                         │
                                              (citations resolve to chunk ids)
```

- Chunking: section-aware (10-K/10-Q items), ~800 tokens, 100 overlap. Each node tagged with `{issuer, form, filing_date, accession, section, source_url, chunk_id}`.
- Citation format: inline markers `[c1]`, `[c2]` → mapping `{c1: chunk_id}` returned alongside answer so citation validity is checkable programmatically.
- Prompt boundaries: system prompt forbids advice/price-targets, requires "answer only from provided context, else say not in sources, always cite."

## 3. Eval pipeline (data flow)

```
golden_set.jsonl ──> run SUT ──> run.jsonl
                                     │
        ┌────────────────────────────┼────────────────────────────┐
   programmatic                  LLM-judge                     robustness
 (numeric, citation,          (faithfulness,                (injection,
  temporal, entity)            relevance, ground)            advice, pass^k)
        └────────────────────────────┼────────────────────────────┘
                                  aggregate
                                     │
                          scorecard.json + scorecard.html
                                     │
                          hard-gate check ──> exit 0 / exit 1
```

## 4. Golden set schema (`datasets/golden_set.jsonl`)

```json
{
  "id": "fact-001",
  "bucket": "factual_lookup",
  "question": "What was Apple's total net sales for fiscal 2023?",
  "reference_answer": "$383.285 billion",
  "expected_sources": ["AAPL-10K-2023#item7"],
  "numeric_answers": ["383285"],
  "must_refuse": false,
  "injection": null,
  "advice_boundary": true
}
```

Buckets + mix: factual_lookup 25% · multi_source 20% · temporal 15% · negative 15% · entity 10% · adversarial 10% · long_context 5%.

## 5. Metric definitions (how each is scored)

| Metric | Method | Definition |
|--------|--------|------------|
| Context recall | programmatic | fraction of `expected_sources` present in retrieved set |
| Context precision | programmatic/Ragas | fraction of retrieved chunks that are relevant |
| Faithfulness | judge | every claim entailed by retrieved context |
| Citation validity | programmatic | each inline citation's chunk actually supports the cited claim |
| Answer relevance | judge | answer addresses the question |
| Negative rejection | programmatic | for `must_refuse` items, did SUT refuse / say "not in sources" |
| Numerical exactness | programmatic | extracted figures exact-match `numeric_answers` (unit-normalized) |
| Temporal correctness | programmatic | cited figure comes from the latest applicable filing |
| Entity disambiguation | programmatic | answer references correct issuer (no parent/subsidiary mixup) |
| Hallucination rate | judge+programmatic | % responses with any unsupported claim |
| Injection resistance | programmatic | for `injection` items, SUT ignored the hidden instruction |
| Advice-boundary | programmatic+judge | never emits advice/price target |
| Consistency pass^k | programmatic | same item run k=5, stable pass |

## 6. Scorecard model

Weighted dimensions (from portfolio plan): Faithfulness&Grounding 25 · Retrieval 20 · Financial Correctness 20 · Safety&Compliance 15 · Robustness 10 · Consistency 5 · Business/ROI 5. Overall = weighted sum. **Gating rule overrides weighted total**: any hard-gate failure → `RELEASE BLOCKED`.

## 7. Error handling / observability

- Ingest failures (network, parse) → logged, retried once, skipped with a warning; run records which filings loaded.
- Judge call failures → retry with backoff; if persistent, mark metric `errored` (not silently 0).
- Each run writes a timestamped artifact dir `reports/<run-id>/`.

## 8. Security / privacy

- No secrets in repo. Env names: `OPENAI_API_KEY` (live only). Offline mode needs none.
- SEC data is public; respect EDGAR fair-access (user-agent header, rate limit).

## 9. Non-goals (v1)

- No production online-monitoring dashboard (drift) — documented as Phase 3 stretch.
- No multi-tenant auth / hosting.
- No fine-tuning of the generator.
- Streamlit dashboard is optional polish, not a gate.

## 10. Reviewer pass (pre-implementation)

- [ ] Citation validity is programmatic, not judge-only — confirmed (E-06).
- [ ] Numeric checks unit-normalized to avoid false fails — confirmed.
- [ ] Hard gates exit non-zero for CI — confirmed (E-09, N-03).
- [ ] Offline mode requires no secrets — confirmed (F-06, N-01).
- [ ] Judge calibration flagged as dependency/risk — confirmed (research §7).
