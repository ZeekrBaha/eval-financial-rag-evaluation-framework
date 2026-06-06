# validation-report.md — Financial RAG Evaluation Framework

> Stub. Fill after implementation. Do not mark done from compile/build success alone.

## Run metadata
- Date: _TBD_
- Commit: _TBD_
- Mode: offline / live

## Commands run
| Check | Command | Result | Notes |
|-------|---------|--------|-------|
| Install | `uv sync` | ⬜ | |
| Lint | `uv run ruff check .` | ⬜ | |
| Types | `uv run mypy src` | ⬜ | |
| Tests | `uv run pytest` | ⬜ | |
| E2E offline | `make eval EVAL_MODE=offline` | ⬜ | |
| Gate fail-path | `make eval ... failing.jsonl` | ⬜ | expect exit 1 |
| Secret scan | grep keys | ⬜ | |

## Functional verification results
- F-01..F-06: _TBD_

## Eval-framework verification results
- E-01..E-10: _TBD_

## Anti-slop visual gate
- _TBD_

## Honesty gate
- _TBD_

## Skipped checks
- _list + reason_

## Unresolved risks
- Judge calibration pending (Portfolio Project 5).
- Golden set at _N_ items (target ≥200).
