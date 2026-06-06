# validation-plan.md — Financial RAG Evaluation Framework

Commands + checks defined **before** coding. A run is "done" only when runtime behavior is verified, not just a clean build.

## Commands

| Check | Command | Gate |
|-------|---------|------|
| Install | `uv sync` | must succeed |
| Lint | `uv run ruff check .` | 0 errors |
| Types | `uv run mypy src` | 0 errors (or documented ignores) |
| Unit + integration | `uv run pytest` | all pass |
| End-to-end (offline) | `make eval EVAL_MODE=offline` | exit 0 on passing fixtures |
| Gate proof (fail path) | `make eval EVAL_MODE=offline GOLDEN=datasets/fixtures/failing.jsonl` | **exit 1** + `RELEASE BLOCKED` printed |
| Secret scan | `git secrets` / manual grep for keys | clean (N-02) |
| CI | offline eval in GitHub Actions | green, no secrets needed |

## Functional verification (runtime, not compile-only)

- [ ] F-01 ingest: ≥5 issuers × (10-K+10-Q) loaded; spot-check chunk metadata.
- [ ] F-02 retrieve: known question returns expected source chunk.
- [ ] F-03 citations: answer contains resolvable inline citations.
- [ ] F-04 refusal: out-of-scope question → "not in sources".
- [ ] F-05 advice-boundary: price-target bait → declines.
- [ ] F-06 offline determinism: same input → identical output, twice, no network.

## Eval-framework verification

- [ ] E-06 programmatic metrics correct on labeled fixtures (incl. citation 0.94 fail).
- [ ] E-07 judge metrics produce scores; judge model+prompt recorded in scorecard.
- [ ] E-05 robustness: injection ignored, advice caught, pass^k stable.
- [ ] E-08 scorecard.json + scorecard.html render with all states.
- [ ] E-09 hard-gate: passing→exit 0, failing→exit 1 with reason. **(money shot)**
- [ ] E-10 per-bucket breakdown present.

## Anti-slop visual gate (HTML scorecard / Streamlit)

- [ ] No Inter/Roboto/Arial; no purple→blue gradient; no emoji-as-icons.
- [ ] Real scores, not placeholders.
- [ ] Numbers in mono; contrast ≥ 4.5:1; status not color-only (labeled).
- [ ] Every state designed: empty / running / errored-metric / success / BLOCKED.

## Honesty gate (portfolio-specific)

- [ ] README: public data only; "designed case study on a public product", not Moody's internal.
- [ ] All thresholds labeled "proposed, calibrate against baseline".
- [ ] Faithfulness flagged uncalibrated until Project 5 (judge calibration).

## Done criteria (v1)

`make eval` offline produces a scorecard AND demonstrates both exit-0 (pass) and exit-1 (seeded fail) paths, with all functional + eval verifications checked and the honesty gate satisfied.
