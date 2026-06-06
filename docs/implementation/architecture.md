# architecture.md — Financial RAG Evaluation Framework

## 1. Module map

```
eval-financial-rag-evaluation-framework/
├── pyproject.toml            # uv-managed deps
├── Makefile                  # ingest / run / score / eval / test targets
├── .env.example              # OPENAI_API_KEY (live only); names only
├── src/
│   ├── sut/                  # System Under Test (the RAG app)
│   │   ├── ingest.py         # F-01 download+parse+chunk+embed
│   │   ├── store.py          # Chroma wrapper (F-01/F-02)
│   │   ├── retrieve.py       # F-02 top-k + metadata
│   │   ├── generate.py       # F-03/F-04/F-05 answer+citations, refusal, advice-boundary
│   │   ├── prompts.py        # system prompt (boundaries)
│   │   ├── providers.py      # live vs offline embeddings/LLM (F-06)
│   │   └── api.py            # FastAPI query endpoint
│   ├── eval/                 # Evaluation framework
│   │   ├── golden.py         # E-01 loader + schema validation
│   │   ├── runner.py         # E-02 run SUT over golden set -> run.jsonl
│   │   ├── metrics/
│   │   │   ├── programmatic.py  # E-06 numeric, citation, temporal, entity, negrej
│   │   │   ├── judge.py         # E-07 faithfulness, relevance, ground (Ragas/DeepEval)
│   │   │   └── robustness.py    # E-05 injection, advice, pass^k
│   │   ├── aggregate.py      # E-08 dimensions, weights, overall
│   │   ├── gates.py          # E-09 hard-gate enforcement -> exit code
│   │   └── scorecard.py      # E-08 JSON + HTML render (uses design-system tokens)
│   └── config.py             # thresholds (proposed gates), weights, k
├── datasets/
│   ├── issuers.yaml          # which filings to ingest
│   ├── golden_set.jsonl      # >=200 items
│   └── fixtures/             # offline-mode retrieval + LLM fixtures (F-06)
├── reports/                  # per-run artifacts (gitignored except samples)
├── tests/                    # pytest: one+ per F-0x/E-0x; AI-failure paths explicit
└── docs/                     # this package
```

## 2. Key boundaries

- **SUT ↔ Eval are decoupled.** Eval calls the SUT through one interface (`runner` → `api` or in-process `generate`). The eval framework could test *any* RAG SUT later (reuse value).
- **Provider abstraction** (`providers.py`) is the single switch between live and offline. `EVAL_MODE=offline` → fixtures, no network, deterministic (F-06, N-01).
- **Metric independence.** programmatic / judge / robustness are separate modules; a judge outage cannot zero a programmatic metric.

## 3. Run sequence (`make eval`)

1. `golden.load()` → validate, count per bucket (E-01).
2. `runner.run()` → for each item, query SUT, capture answer+retrieved+citations+latency → `run.jsonl` (E-02).
3. `metrics.programmatic` + `metrics.judge` + `metrics.robustness` score the run (E-03/04/05/06/07).
4. `aggregate.build()` → dimension scores + weighted overall (E-08, E-10).
5. `scorecard.render()` → `reports/<run-id>/scorecard.{json,html}` (E-08).
6. `gates.enforce()` → if any hard gate fails → print `RELEASE BLOCKED — <gate> <value> < <threshold>`, **exit 1** (E-09).

## 4. Data contracts

- `run.jsonl` row: `{id, bucket, question, answer, retrieved:[{chunk_id, score, meta}], citations:{c1:chunk_id}, latency_ms, mode}`.
- `scorecard.json`: `{run_id, mode, dimensions:[{name, weight, score, status}], buckets:{...}, hard_gates:[{name, value, threshold, passed}], overall, status}`.

## 5. Offline determinism

- Fixtures keyed by `hash(question)` → fixed retrieved chunks + fixed model output.
- Judge in offline mode → deterministic stub returning recorded verdicts (so CI is stable and free).
- Live mode used only to (a) author fixtures, (b) generate README screenshots.

## 6. CI hook (feeds Portfolio Project 2)

- `make eval` is the gate. CI runs it in offline mode; non-zero exit fails the build.
- `eval-gates.yaml` (Project 2) reads the same `config.py` thresholds → single source of truth.

## 7. Scale paths (documented, not built v1)

- Chroma → `pgvector` for larger corpora.
- Add online drift monitoring: re-embed on corpus update, alert on retrieval-recall drop.
- Swap in a hosted tracing backend (Langfuse/LangSmith) for live observability.
