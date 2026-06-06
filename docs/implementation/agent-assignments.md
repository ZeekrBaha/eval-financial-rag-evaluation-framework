# agent-assignments.md — role review notes

Per-role review of the plan before implementation. Each role flags concerns from its lens.

## PM
- Scope is two halves (SUT + eval). Keep SUT *minimal* — it exists to be evaluated, not to be a product. Guard against gold-plating retrieval.
- The money-shot (RELEASE BLOCKED scorecard) is the primary portfolio artifact. Prioritize T13–T14 even if golden set is still at 50 items.

## Developer
- Provider abstraction (T2) must land before SUT logic, or offline determinism is bolted on late and leaks network in tests.
- Keep SUT↔eval interface a single function/HTTP contract so the eval harness is reusable for other SUTs later.
- Citations must be structured (`{c1: chunk_id}`), not free-text, or citation-validity (T10) becomes a parsing nightmare.

## Junior developer
- Numeric exactness: watch units (thousands vs millions, $ vs %). Build a normalization helper + tests first.
- Don't compute faithfulness in code — that's the judge's job (T11). Don't let programmatic and judge metrics overlap silently.

## Tester
- Every AI-failure path needs an explicit test: refusal (F-04), advice-boundary (F-05), injection (T12). These are the high-value tests.
- Need a deliberately failing fixture (citation 0.94) so T14's exit-1 path is proven, not assumed.
- pass^k (k=5) test must be deterministic offline or it flakes — use seeded fixtures.

## Reviewer
- Check README does not overclaim (no "Moody's internal", gates labeled proposed).
- Check no secrets committed; offline mode truly needs none.
- Watch for judge-bias overclaim — faithfulness number is uncalibrated until Project 5; say so in report.

## Team lead
- Critical path: T0→T1→T2→T3→T4→T5→T9→T10→T13→T14→T15. Judge (T11), robustness (T12), and golden-set scale (T8) can proceed in parallel once T9 lands.
- Ship gate-working end-to-end with 50 golden items first; scale to 200+ after.
- Definition of done for v1: `make eval` offline → scorecard + correct exit code, both pass and seeded-fail demonstrated.
