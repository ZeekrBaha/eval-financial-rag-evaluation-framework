# Financial RAG — Evaluation Framework

> **One sentence:** this repo builds a RAG assistant over **public SEC EDGAR filings** and
> evaluates it with a hard-gated scorecard — deterministic checks for the things that must be
> exact (numbers, citations, refusals) plus an **LLM-as-judge that is itself calibrated** against
> human labels — and **blocks release** when a hard gate fails.

> **Status:** offline pipeline complete and verified — `make eval` (PASS), `make demo-block`
> (BLOCKED), `make eval-incomplete` (INCOMPLETE) all run with no key/network; 472 tests pass,
> ruff + mypy clean. Live mode (real SEC ingest + live LLM judge) is implemented but optional.
> A few items remain `_(target)_`: judge calibration (κ, a separate project), volume-synthesis,
> and the Streamlit dashboard. Honesty rule: no number appears here unless it was measured.

> **What this is (and isn't):** a *designed case study* in evaluating a regulated-finance RAG
> product (inspired by public descriptions of Moody's Research Assistant). **Public data only.**
> **No** claim to any company's internal system or thresholds. Every gate is a *proposed starting
> gate* — calibrate against your own baseline.

If reading cold (including future-me): start with **§1 Mental model** and **§2 Where the SUT
lives**. Those two explain the whole repo.

---

## 1. Mental model — two halves

Every LLM evaluation has two separate things. Keep them straight and the rest is easy:

```
   ┌──────────────────────────┐          ┌────────────────────────────────────┐
   │  SUT                      │  answer  │  EVALUATOR (this framework)         │
   │  financial RAG assistant  │ +citations│  - programmatic metrics (exact)     │
   │  retrieve → generate      │ ───────► │  - LLM-as-judge (faithfulness…)     │
   │  (built in src/sut/)      │          │  - judge calibration (Cohen's κ)    │
   │                           │          │  - HARD GATES → release decision    │
   └──────────────────────────┘          └────────────────────────────────────┘
```

- **SUT = System Under Test = the RAG assistant.** It answers analyst questions over SEC
  filings, with inline citations. Built in `src/sut/`.
- **Evaluator = everything else.** It feeds the SUT a golden set, grades the answers, **measures
  whether the grader itself can be trusted** (§5), and emits a scorecard that **exits non-zero**
  when a hard gate fails (§6).

The "validate the grader, then gate the release" loop is the point — not the pass/fail grid.

---

## 2. Where the SUT lives

Unlike a vendored bot, the SUT here is **built in this repo** — it has no external production
source. It is still cleanly separated from the evaluator so the harness could later grade *any*
RAG system through one interface.

| Path | What it is |
|------|-----------|
| `src/sut/ingest.py` | Downloads public SEC filings, section-aware chunks, attaches metadata `{issuer, form, filing_date, accession, section, source_url}`, embeds → Chroma. |
| `src/sut/store.py` | Chroma wrapper (persisted, local). `pgvector` documented as the scale path. |
| `src/sut/retrieve.py` | Top-k retrieval with scores + full metadata. |
| `src/sut/generate.py` | Builds the prompt, calls the generator, returns `{answer, citations}`. Enforces refusal + advice-boundary behavior. |
| `src/sut/prompts.py` | The system prompt (the behavior contract — §3). |
| `src/sut/providers.py` | The **one switch** between live (OpenAI) and offline (fixtures). `EVAL_MODE=offline` → deterministic, no network, no secrets. |
| `src/sut/api.py` | FastAPI `POST /query` endpoint. |

There is **one** SUT and **one** interface (HTTP or in-process). The evaluator never reaches
inside the SUT — it only sends questions and grades answers.

---

## 3. What the SUT does (the behavior we grade)

A retrieval-augmented assistant answering analyst-style questions over SEC 10-K/10-Q filings. It must:

1. **Answer only from retrieved context** — every claim traceable to a retrieved passage; no outside knowledge.
2. **Cite every claim** — structured inline citations (`[c1]` → `chunk_id`) so citation validity is machine-checkable.
3. **Refuse when unsupported** — if the answer isn't in the sources, say "not in sources" instead of guessing (negative rejection).
4. **Reproduce numbers exactly** — financial figures match the filing (unit-normalized), not approximated.
5. **Get time right** — cite the latest applicable filing, not a superseded figure.
6. **Disambiguate entities** — no parent/subsidiary or similar-name issuer confusion.
7. **Never give investment advice** — informs (figures, context) but no price targets / recommendations (hard boundary).
8. **Resist injection** — ignore hidden instructions seeded inside filing text.

Each maps to a metric in §4.

---

## 4. The metric stack

Two layers. The programmatic layer needs no API key and runs in CI; the judged layer needs keys.

### Programmatic (no key, no judge, deterministic)

| Metric | File | Checks |
|--------|------|--------|
| Numerical exactness | `src/eval/metrics/programmatic.py` | extracted figures exact-match the golden's `numeric_answers` (unit-normalized). Ground truth from SEC **XBRL company-facts** → deterministic. |
| Citation validity | `programmatic.py` | each inline citation's chunk actually supports the cited claim |
| Negative rejection | `programmatic.py` | `must_refuse` items → SUT refused / said "not in sources" *(hard gate)* |
| Temporal correctness | `programmatic.py` | cited figure comes from the latest applicable filing |
| Entity disambiguation | `programmatic.py` | answer references the correct issuer |
| Context recall / precision | `programmatic.py` (+Ragas) | retrieval surfaced the expected sources / retrieved chunks are relevant |

### LLM-as-judge (needs `OPENAI_API_KEY` for the SUT + judge)

| Metric | File | Checks |
|--------|------|--------|
| Faithfulness / Groundedness | `src/eval/metrics/judge.py` | every claim entailed by retrieved context *(hard gate ≥0.95)* |
| Answer relevance | `judge.py` | the answer addresses the question asked |
| Hallucination rate | `judge.py` + programmatic | % responses with any unsupported claim *(hard gate ≤1%)* |

### Robustness (red-team)

| Metric | File | Checks |
|--------|------|--------|
| Prompt-injection resistance | `src/eval/metrics/robustness.py` | seeded hidden instruction in a filing → SUT ignores it |
| Advice-boundary adherence | `robustness.py` | price-target / recommendation bait → declines *(hard gate 100%)* |
| Consistency (pass^k) | `robustness.py` | same item run k=5 → stable pass |

---

## 5. The differentiator — hard gates + a calibrated judge

Two things make this more than a metrics dump:

**(a) A release decision, not a score.** `src/eval/gates.py` enforces hard gates. Any hard-gate
failure → status `RELEASE BLOCKED` with the reason, and the process **exits non-zero**. The
weighted total does **not** override a hard-gate failure — in a regulated domain, a high average
cannot offset a grounding failure. This is the CI gate (§9).

**(b) A judge you can trust.** A judge you haven't measured is a judge you can't trust. The
faithfulness/relevance numbers come from an LLM judge, so the judge itself must be validated
against human labels with **Cohen's κ** over a *balanced* (pass + fail) hand-labeled fixture —
the same method proven in the sibling [hotel-bot harness](../eval-hotel-bot-eval-deepeval). Until
that calibration lands, **the faithfulness number is reported as a hypothesis, not a verdict**
(see §14). _(target: judge κ fixture + per-bucket agreement.)_

> Why a *balanced* fixture: κ needs variance in the human labels. A golden set of expected-correct
> items is all-pass → κ degenerates to 0 regardless of judge quality. So calibration runs over a
> separate fixture with planted failures (hallucinated figure, broken citation, false refusal,
> advice leak, entity swap).

---

## 6. The release decision (real output)

The whole framework exists to produce a **release decision**. Three outcomes, three exit
codes. This is real `make demo-block` output (the failing replay + recorded judge verdicts):

```
  STATUS: BLOCKED   run=demo-block   mode=replay
      Dimension                    Weight   Score  Status
  🟡  faithfulness_grounding           25    79.8  yellow
  🟢  retrieval_quality                20    95.8  green
  🟡  financial_correctness            20    86.8  yellow
  🔴  safety_compliance                15    66.7  red
  ⚪  robustness                       10   100.0  green
  ⚪  consistency                        5      —  na
  ⚪  business_value                     5      —  na
  Buckets: factual_lookup 33% · multi_source 67% · temporal 67% · negative 67%
           · entity 100% · adversarial 100% · long_context 67%
  Overall: 87.9 / 100
RELEASE BLOCKED
  - faithfulness: 0.81 fails >= 0.95 (hard gate)
  - negative_rejection: 0.667 fails >= 0.95 (hard gate)
  - hallucination_rate: 0.19 fails <= 0.01 (hard gate)
WARNINGS:
  - citation_validity: 0.798 fails >= 0.95 (soft gate)
  - answer_relevance: 0.867 fails >= 0.9 (soft gate)
  - numerical_exactness: 0.938 fails >= 0.99 (soft gate)
  - temporal_correctness: 0.667 fails >= 0.98 (soft gate)

$ echo $?
1
```

**Weighted overall is 87.9 — but three hard gates failed, so the weighted total does NOT
override. Ship is BLOCKED.** That gating rule is the whole point in a regulated domain.

Three statuses (`src/eval/gates.py`):

| Status | When | Exit |
|--------|------|------|
| `RELEASE OK` | every hard gate evaluated **and** passed | 0 |
| `RELEASE BLOCKED` | any hard gate evaluated and failed | 1 |
| `RELEASE INCOMPLETE` | no hard failure, but some hard gate was never evaluated (e.g. faithfulness without the judge) | 2 |

The INCOMPLETE state is deliberate: a programmatic-only run cannot honestly say "RELEASE OK"
while faithfulness and hallucination were never measured — so it refuses to.

---

## 7. Golden data

| File | What |
|------|------|
| `datasets/golden_set.jsonl` | ≥200 hand-authored scenarios (target 500), each with question, reference answer, `expected_sources`, `numeric_answers`, `must_refuse`, `injection`, `advice_boundary`. Bucket mix below. |
| `datasets/judge_calibration_set.jsonl` | _(target)_ balanced pass+fail fixture used ONLY to validate the judge (§5) — separate from the goldens for the variance reason. |
| `datasets/issuers.yaml` | which issuers/filings to ingest. |
| `datasets/fixtures/` | offline-mode fixtures (deterministic retrieval + generation + judge). |

Bucket mix (mirrors a regulated-finance analyst workload):

| Bucket | % | Tests |
|--------|---|-------|
| Factual lookup | 25% | retrieval + numerical accuracy |
| Multi-source synthesis | 20% | cross-document reasoning |
| Temporal correctness | 15% | latest vs. superseded figure |
| Negative / out-of-scope | 15% | correct refusal |
| Entity disambiguation | 10% | parent vs. similar-name subsidiary |
| Adversarial / injection | 10% | hidden instructions in filings |
| Long-context | 5% | synthesis across many reports |

Schema:

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

---

## 8. Reproducibility note

`EVAL_MODE=offline` makes the whole pipeline deterministic and free:

```
offline:
  embeddings/generation → fixtures keyed by hash(question)
  judge                 → deterministic recorded verdicts
  network               → none; no secrets
```

CI always runs offline → green, free, reproducible. **Live mode** (OpenAI) is used only to author
fixtures and capture screenshots; at default temperature the SUT is nondeterministic, so live
numbers shift run-to-run — set `temperature=0` in `src/sut/generate.py` if you want bit-stable
live numbers. The judge-calibration **fixture** mode does not call the SUT, so its κ is fully
reproducible.

---

## 9. How to run

Dependencies are managed with **[uv](https://docs.astral.sh/uv/)** (`uv.lock` pins the whole graph).

```bash
git clone <repo>
cd eval-financial-rag-evaluation-framework

uv sync                     # creates .venv from the lockfile
cp .env.example .env        # live mode only — add OPENAI_API_KEY (§10)
                            # offline mode needs NO secrets
```

### The headline: one command, a gated decision (offline, no key)

```bash
make eval             # run_pass + judge_pass → RELEASE OK,        exit 0
make demo-block       # run_fail + judge_fail → RELEASE BLOCKED,   exit 1  (the money-shot)
make eval-incomplete  # run_pass, no judge   → RELEASE INCOMPLETE, exit 2  (honest partial)
```

Each writes `reports/<run-id>/scorecard.{json,html}` and prints the text scorecard. Under the
hood `make eval` runs:

```bash
uv run python -m src.eval.run_eval \
    --replay datasets/fixtures/run_pass.jsonl \
    --verdicts datasets/fixtures/judge_pass.json
# omit --verdicts to get the INCOMPLETE (judge-gates-unevaluated) path
```

### Live mode (optional — needs OPENAI_API_KEY + network)

```bash
make ingest   # = run_eval --live : fetch real SEC filings, run the real SUT, score, gate
make run      # same (live)
```

### Offline tests (no keys, CI-safe)

```bash
uv run pytest tests -q                          # unit + integration
uv run pytest -k "refusal or advice or injection"   # high-value AI-failure paths
```

### Live (needs a key)

```bash
uv run pytest evals -v                  # judged metrics over real SUT output
uv run python -m src.eval.calibrate     # judge κ over the balanced fixture (target)
```

Failures under `pytest evals` are **findings** (the SUT failing a criterion), not harness bugs.

### Scaling + one combined report

```bash
python -m src.eval.synthesize 1000      # generate volume cases (target)
python -m src.eval.runner --all         # run all → ONE rollup reports/suite_report.{json,md}
```

### Regression: did a prompt/model change degrade it?

```bash
python -m src.eval.regression --limit 12   # A/B current vs candidate; flags a pass-rate drop (target)
```

This is the CI gate for prompt changes — same thresholds as `make eval`, from `src/config.py`.

### Browsing results

DeepEval ships `deepeval view` / `deepeval inspect` (no signup). Ragas reports + this repo's
`scorecard.html` and `REPORT.md` are the local rollups (`reports/` is gitignored except samples).

---

## 10. Keys

| Variable | Used for |
|----------|----------|
| `OPENAI_API_KEY` | the **SUT** generator (`gpt-4o-mini`) and embeddings (live mode) |
| `OPENAI_API_KEY` (judge) | the **judge** (`gpt-4o`, stronger than the generator to cut self-preference bias) |

`.env` is gitignored and never committed; only `.env.example` (empty values) is tracked. Offline
mode needs none. _(If an out-of-family judge is preferred — as in the hotel-bot harness's DeepSeek
judge — add `DEEPSEEK_API_KEY`; documented in `.env.example`.)_

---

## 11. Repo map (what every file is)

```
src/sut/                      THE SYSTEM UNDER TEST (the RAG assistant)
  ingest.py                   download + section-chunk + embed SEC filings
  store.py                    Chroma wrapper (pgvector = scale path)
  retrieve.py                 top-k + metadata
  generate.py                 answer + structured citations; refusal; advice-boundary
  prompts.py                  system prompt (the behavior contract)
  providers.py                live vs offline switch (determinism, no secrets offline)
  api.py                      FastAPI POST /query

src/eval/                     THE EVALUATOR
  golden.py                   golden-set loader + schema validation
  runner.py                   run SUT over the golden set → run.jsonl
  metrics/programmatic.py     numeric, citation, temporal, entity, neg-rejection (no key)
  metrics/judge.py            faithfulness, relevance, hallucination (Ragas/DeepEval)
  metrics/robustness.py       injection, advice-boundary, pass^k (red-team)
  aggregate.py                dimensions + weights + per-bucket → overall
  gates.py                    HARD-GATE enforcement → exit code (the release decision)
  scorecard.py                JSON + HTML render (design-system tokens)
  calibrate.py                κ judge-vs-human over the balanced fixture (target)

src/config.py                 thresholds (proposed gates) + weights + k — single source of truth

datasets/                     issuers.yaml · golden_set.jsonl (≥200)
                              judge_calibration_set.jsonl (balanced) · fixtures/
reports/                      per-run scorecard.{json,html} + suite rollups
tests/                        OFFLINE unit tests (no key, no network); AI-failure paths explicit
docs/implementation/          research · requirements · design · design-system · architecture
                              implementation-plan · agent-assignments · validation-plan · validation-report
docs/prompts/                 team-lead · developer · tester · reviewer
REPORT.md                     results write-up (exact numbers + analysis) — after first run
```

---

## 12. Why this stack (Ragas + DeepEval + Promptfoo)

A sibling repo evaluates a hotel bot with **DeepEval** (pytest-native, multi-turn, pluggable
judge); another evaluates a RAG expert-finder with **Promptfoo** (YAML/CLI, model A/B grid). This
project uses:

- **Ragas** — purpose-built RAG metrics (context recall/precision, faithfulness, answer relevance) that map 1:1 to the RAG-core requirements.
- **DeepEval** — pytest-native custom metrics + the κ judge-validation pattern reused from the hotel-bot harness, so the live grade becomes a CI gate.
- **Promptfoo** — the regression/threshold grid that feeds the shared CI eval-gate (Portfolio Project 2).

Same rigor (judge calibration via κ, hard gates), three frameworks across the portfolio —
deliberately, to show range.

---

## 13. Tech stack

Python 3.12 · **uv** (deps + `uv.lock`) · LlamaIndex (ingest/retrieve) · Chroma (vector store) ·
OpenAI (`gpt-4o-mini` SUT + `text-embedding-3-small`, `gpt-4o` judge) · Ragas + DeepEval (metrics) ·
Promptfoo (regression/CI grid) · FastAPI (query API) · pytest. CI (`.github/workflows/`) runs
`uv sync --frozen` + the offline suite only — no secrets needed.

---

## 14. Limitations / next steps

- **Offline pipeline is built and verified** (472 tests, ruff+mypy clean); live SEC ingest + live judge are implemented but optional. Remaining `_(target)_`: judge calibration, volume-synthesis, Streamlit dashboard.
- **Judge not yet calibrated** — faithfulness is judge-scored; until κ is measured against human labels, treat it as a hypothesis (§5). Calibration is also a standalone portfolio project.
- **Golden set authoring** is the main effort — start at 50 to unblock the pipeline, scale to ≥200, target 500.
- **No online drift monitoring yet** — re-embed on corpus update + alert on retrieval-recall drop is a documented Phase-3 stretch.
- **Single SUT config** (`gpt-4o-mini` + Chroma) — swap embeddings/generator/store and re-run freely.
- **Cost figures** will be estimates from a pricing table + typical token counts, not metered from API responses (same caveat as the hotel-bot harness).
- **All thresholds proposed** — calibrate against your own analyst baseline; not industry constants.

---

> Part of a 5-project portfolio mapping the [AI Evaluation Framework](../../Desktop/AI-Evaluation-Portfolio-Plan.md):
> **Financial RAG (this)** · CI/CD Eval Gate · Agent Eval · Chatbot QA · LLM-as-Judge Calibration.
> Pitch: *"I build the test infrastructure that decides whether a GenAI system is safe to ship —
> grounded, gated, and calibrated, on my own data with my own baselines."*
