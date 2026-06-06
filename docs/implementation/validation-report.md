# validation-report.md — Financial RAG Evaluation Framework

> Results of the offline validation run. All checks below run with NO API key and NO network.

## Run metadata
- Branch: `feat/implement-financial-rag`
- Mode: offline (replay + recorded judge verdicts)
- Toolchain: Python 3.12+, uv

## Commands run
| Check | Command | Result |
|-------|---------|--------|
| Install | `uv sync` | ✅ resolves, locked |
| Lint | `uv run ruff check src tests` | ✅ 0 errors |
| Types | `uv run mypy src` | ✅ 0 errors (21 source files) |
| Tests | `uv run pytest -q` | ✅ 472 passed |
| Full gate (PASS) | `make eval` | ✅ RELEASE OK, exit 0 |
| Money-shot (BLOCKED) | `make demo-block` | ✅ RELEASE BLOCKED, exit 1 |
| Honest partial | `make eval-incomplete` | ✅ RELEASE INCOMPLETE, exit 2 |
| Secret scan | grep for keys | ✅ none tracked; `.env` gitignored |

## Functional verification (runtime)
- F-01 ingest (offline fixture path): ✅ section-aware chunking, 7 metadata fields, TOC rejection, unique chunk_ids.
- F-02 retrieve: ✅ top-k + metadata via `VectorStore.query`; `GET/POST` API.
- F-03 citations: ✅ structured `{cN: chunk_id}`; hallucinated/out-of-range markers surfaced in `unmatched_citations`.
- F-04 refusal: ✅ negative-bucket items refuse (negative_rejection metric).
- F-05 advice-boundary: ✅ deterministic denylist; hard gate.
- F-06 offline determinism: ✅ provider fixtures, no network/secrets; determinism tests.

## Eval-framework verification
- E-01 golden loader: ✅ schema (`extra=forbid`), per-bucket counts, line-numbered errors.
- E-02 runner: ✅ replay + live; run-contract round-trip.
- E-03/04/06 programmatic: ✅ numeric exactness (scale-aware), citation validity (sentence-bound), temporal, entity, negative rejection, context recall/precision.
- E-05 robustness: ✅ advice_boundary, injection_resistance; consistency NA offline (needs live reruns).
- E-07 judge: ✅ faithfulness, answer_relevance, hallucination_rate from recorded verdicts; missing verdict raises; flagged UNCALIBRATED.
- E-08 scorecard: ✅ JSON + HTML + text; dimensions, weighted overall (renormalized over non-NA), per-bucket.
- E-09 hard-gate enforcement: ✅ PASS/BLOCKED/INCOMPLETE + exit 0/1/2; unevaluated hard gates surfaced, never silent-pass.
- E-10 per-bucket: ✅ gate-only per-item pass (soft retrieval excluded).

## Measured results (offline replay)
| Scenario | Status | Exit | Notes |
|----------|--------|------|-------|
| run_pass + judge_pass | PASS | 0 | all 4 hard gates evaluated & pass; overall 99.1 |
| run_fail + judge_fail | BLOCKED | 1 | faithfulness 0.81, negative_rejection 0.667, hallucination_rate 0.19 all fail (hard); soft warnings on citation_validity/answer_relevance/numerical/temporal |
| run_pass (no verdicts) | INCOMPLETE | 2 | faithfulness + hallucination_rate unevaluated (need live judge) |

## Anti-slop visual gate (HTML scorecard)
- ✅ IBM Plex Mono for numbers; dark backdrop; status by color **and** text label (not color-only); `role="status"` banner; no external CSS/JS.

## Honesty gate
- ✅ Public/fictional data only; gates labeled "proposed".
- ✅ Judge flagged UNCALIBRATED until the separate calibration project (Cohen's κ).
- ✅ INCOMPLETE status prevents a vacuous "RELEASE OK" when hard gates are unmeasured.

## Skipped / not built (by design)
- Live SEC ingest + live judge: implemented but not exercised in CI (need key/network).
- Streamlit dashboard (T17), online drift monitoring (T18): documented stretch, not built.
- consistency_passk: NA offline (requires k live reruns).

## Unresolved risks
- Judge calibration pending (separate portfolio project) — faithfulness is a hypothesis until measured against human labels.
- Golden set is 21 fictional items (3/bucket) — enough to exercise every metric + gate; scale to 200+ for a production-credible suite.
- advice_boundary `overweight` pattern is conservative (rare false-positive on "portfolio is overweight").
