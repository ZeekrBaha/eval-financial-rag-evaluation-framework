# datasets/

**IMPORTANT: All data in this directory is entirely fictional and for evaluation purposes only. These are NOT real SEC filings, real companies, or real financial figures.**

---

## Fictional Corpus

Three fictional issuers are used throughout the golden set and run fixtures:

| Issuer | Abbrev | Sector | Filings |
|--------|--------|--------|---------|
| Northwind Motors Inc. | NWM | Automotive | 10-K 2024, 10-K 2023 |
| Northwind Auto Finance LLC | NAF | Auto Finance (subsidiary of NWM) | 10-K 2024 |
| Cascade Semiconductor Corp. | CSC | Semiconductor | 10-K 2024, 10-K 2023 |

Northwind Auto Finance LLC is deliberately similarly named to Northwind Motors Inc. to create entity disambiguation test cases (parent vs. subsidiary). All financial figures are invented and do not correspond to any real company or SEC filing.

Accession number format: `{ISSUER}-{FORM}-{YEAR}` (e.g. `NWM-10K-2024`).
Chunk id format: `{ACCESSION}#{SECTION}#{INDEX}` (e.g. `NWM-10K-2024#item7#0`).

---

## Golden Set (`golden_set.jsonl`)

21 items, 3 per bucket across all 7 evaluation buckets:

| Bucket | IDs | What it tests |
|--------|-----|---------------|
| `factual_lookup` | fact-001..003 | Exact numeric figures from a single filing section |
| `multi_source` | multi-001..003 | Comparison or aggregation across two issuers or sections |
| `temporal` | temp-001..003 | Must cite the LATEST filing; older superseded figure is a trap |
| `negative` | neg-001..003 | Topic not in corpus; SUT must refuse (`must_refuse=true`) |
| `entity` | ent-001..003 | Parent (NWM) vs. subsidiary (NAF) disambiguation |
| `adversarial` | adv-001..003 | Hidden prompt-injection strings; SUT must ignore them |
| `long_context` | long-001..003 | Synthesis across 2-3 sections or 2 fiscal years |

---

## Replay Fixtures

The fixtures in `fixtures/` are **pre-recorded, offline-deterministic run files**. They let the evaluator and gate logic run in CI without any API keys, network calls, or live model inference.

In production, the live SUT records real runs to `run.jsonl` format. These fixtures simulate that output format exactly so the metrics and scorecard logic can be exercised end-to-end offline.

### `fixtures/run_pass.jsonl`

All 21 rows are authored to pass every gate:
- Factual answers contain the exact numeric figures matching `numeric_answers`.
- Citations map to retrieved chunks whose text genuinely supports the claim.
- Negative-bucket rows refuse with "not in the provided sources" and empty citations.
- Adversarial rows answer the factual question and ignore the injection string.
- Temporal rows cite only the LATEST filing (2024), not superseded 2023 figures.
- Entity rows name the correct issuer (NAF vs. NWM).
- No investment advice or price targets appear anywhere.

### `fixtures/run_fail.jsonl`

Same 21 row ids, but with deliberate defects to block release:

| Row | Defect type | Expected gate failure |
|-----|-------------|----------------------|
| `fact-001` | Cited chunk text is generic brand copy, not revenue figures | citation_validity |
| `fact-002` | Cited chunk text describes capex/fabs, not gross profit | citation_validity |
| `multi-001` | NWM citation chunk contains workforce headcount, not revenue | citation_validity |
| `neg-001` | Does NOT refuse; hallucinates a submarine revenue figure of $1,250M | negative_rejection |
| `temp-001` | Cites and reports the superseded 2023 figure ($54,180M) instead of 2024 ($58,420M) | temporal_correctness |

All other rows in `run_fail.jsonl` pass their individual checks. This creates a realistic partial-failure scenario that exercises the gate logic without making the entire run look broken.

---

## Run Row Contract

Each row in a run JSONL file has exactly these fields:

```json
{
  "id": "fact-001",
  "bucket": "factual_lookup",
  "question": "...",
  "answer": "... [c1].",
  "retrieved": [
    {
      "chunk_id": "NWM-10K-2024#item7#0",
      "text": "...passage text...",
      "similarity": 0.94,
      "issuer": "Northwind Motors Inc.",
      "form": "10-K",
      "filing_date": "2024-12-15",
      "accession": "NWM-10K-2024",
      "section": "item7",
      "source_url": "https://example.invalid/nwm-10k-2024"
    }
  ],
  "citations": {"c1": "NWM-10K-2024#item7#0"},
  "unmatched_citations": [],
  "latency_ms": 0,
  "mode": "replay"
}
```

Every citation marker (e.g. `[c1]`) must map to a `chunk_id` present in `retrieved`. A citation is valid only when the cited chunk's `text` genuinely contains or supports the claimed fact.
