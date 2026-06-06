# requirements.md — Financial RAG Evaluation Framework

Each requirement has an ID, is testable, and maps to tasks in `implementation-plan.md` and checks in `validation-plan.md`.

## Functional — SUT (RAG assistant)

| ID | Requirement | Acceptance criteria |
|----|-------------|---------------------|
| F-01 | Ingest public SEC filings | Given a list of issuer/form/date, the ingester downloads, parses, chunks, embeds, and stores filings; each chunk carries `{issuer, form, filing_date, accession, section, source_url}` metadata. |
| F-02 | Retrieve relevant passages | Given a question, return top-k chunks with scores and full metadata. k configurable. |
| F-03 | Answer with inline citations | The generated answer includes inline citation markers that resolve to specific stored passages. |
| F-04 | Refuse out-of-scope / missing data | If the answer is not supported by retrieved passages, the SUT says so ("not in sources") instead of guessing. |
| F-05 | Respect advice boundary | The SUT informs (ratings/figures/context) but never gives investment advice / price targets. |
| F-06 | Deterministic offline mode | With `EVAL_MODE=offline`, retrieval + generation use fixtures; identical inputs → identical outputs; no network/secrets. |

## Functional — Eval framework

| ID | Requirement | Acceptance criteria |
|----|-------------|---------------------|
| E-01 | Golden set loader | Loads `datasets/golden_set.jsonl`; validates schema; reports per-bucket counts. |
| E-02 | Run SUT over golden set | Produces a `run.jsonl` of `{id, question, answer, retrieved, citations, latency}`. |
| E-03 | Score RAG-core metrics | Context recall, context precision, faithfulness, citation validity, answer relevance, negative rejection. |
| E-04 | Score financial-correctness | Numerical exactness, temporal correctness, entity disambiguation, hallucination rate. |
| E-05 | Score robustness/safety | Prompt-injection resistance, advice-boundary adherence, consistency (pass^k). |
| E-06 | Programmatic checks | Numeric exactness, citation resolution (cited passage actually contains the claim), temporal/entity checks run deterministically, no judge. |
| E-07 | LLM-judge checks | Faithfulness, answer relevance, groundedness scored by judge; judge model + prompt versioned. |
| E-08 | Scorecard render | Emit JSON + HTML scorecard with per-dimension scores, weights, and overall. |
| E-09 | Hard-gate enforcement | Any hard-gate failure → overall status `RELEASE BLOCKED` + reason; process exits non-zero. |
| E-10 | Per-bucket breakdown | Scores reported per golden-set bucket, not just aggregate. |

## Non-functional

| ID | Requirement | Acceptance criteria |
|----|-------------|---------------------|
| N-01 | Reproducibility | `uv sync` + one command runs full offline eval green with no secrets. |
| N-02 | No secrets in repo | Only env-var names documented; secret scan clean. |
| N-03 | Single-command run | `make eval` (or `uv run eval`) does ingest→run→score→gate end to end. |
| N-04 | Test coverage | Every E-0x and F-0x has at least one automated test; AI-failure paths (F-04, F-05, E-05) tested explicitly. |
| N-05 | Honest README | States public-data + designed-case-study framing; gates labeled "proposed". |

## Hard gates (release-blocking)

| Gate | Threshold (proposed) |
|------|----------------------|
| Faithfulness / Groundedness | ≥ 0.95 |
| Negative Rejection | ≥ 0.95 |
| Hallucination rate | ≤ 1% |
| Advice-boundary adherence | 100% |

## Soft gates (reported, warn)

Context Recall ≥ 0.90 · Context Precision ≥ 0.85 · Citation validity ≥ 0.95 · Answer Relevance ≥ 0.90 · Numerical exactness ≥ 0.99 · Temporal correctness ≥ 0.98 · Entity disambiguation ≥ 0.98 · Prompt-injection resistance ≥ 0.95 · Consistency (pass^k) ≥ 0.90.

> All thresholds are **proposed starting gates**, calibrate against baseline. Not industry constants.
