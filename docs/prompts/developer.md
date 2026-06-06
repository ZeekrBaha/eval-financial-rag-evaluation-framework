# Prompt — Developer (Financial RAG Evaluation Framework)

Implement tasks from `docs/implementation/implementation-plan.md` within the architecture in `docs/implementation/architecture.md`. Stay in scope; reference the approved docs.

## Build order you own
SUT: T2 (providers) → T3 (ingest/store) → T4 (retrieve) → T5 (generate+citations+boundaries) → T6 (API).
Eval: T9 (runner) → T13 (aggregate/scorecard) → T14 (gates).

## Hard rules (say what to DO)
- Build `providers.py` first; route all embeddings/LLM through it; `EVAL_MODE=offline` → deterministic fixtures, no network.
- Return **structured citations** `{ "c1": chunk_id }` alongside the answer.
- System prompt MUST: answer only from retrieved context; cite every claim; say "not in sources" when unsupported; never give investment advice or price targets.
- Keep the SUT↔eval interface a single function or HTTP contract so the harness is reusable.
- Thresholds/weights come from `src/config.py` only.
- Write the simplest code that passes acceptance criteria — no speculative abstraction, no defensive bloat.

## Tests (with each task)
Determinism (offline), retrieval-returns-expected-chunk, refusal path, advice-boundary path, citation mapping present.

## Report
Exact changed files + command results (lint, types, pytest) for every task.
