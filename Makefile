.PHONY: ingest run score eval eval-incomplete demo-block test typecheck lint check calibrate live-ingest-smoke

# Replay + judge-verdict fixtures for `make eval` (override on command line).
# Defaults → run_pass + judge_pass so the full hard-gate decision is PASS (exit 0).
REPLAY ?= datasets/fixtures/run_pass.jsonl
VERDICTS ?= datasets/fixtures/judge_pass.json

# ---------------------------------------------------------------------------
# eval / score — offline replay, no API key required.
# Programmatic + robustness + judge (recorded verdicts) → all 4 hard gates
# evaluated. Default (run_pass + judge_pass) → RELEASE OK, exit 0.
# ---------------------------------------------------------------------------
eval:
	uv run python -m src.eval.run_eval --replay $(REPLAY) --verdicts $(VERDICTS)

# score is an alias for eval (same pipeline)
score: eval

# ---------------------------------------------------------------------------
# demo-block — failing fixtures show the RELEASE BLOCKED money-shot.
# Non-zero exit (1) is expected; prefix with - so make itself does not error.
# ---------------------------------------------------------------------------
demo-block:
	-uv run python -m src.eval.run_eval --replay datasets/fixtures/run_fail.jsonl --verdicts datasets/fixtures/judge_fail.json

# ---------------------------------------------------------------------------
# eval-incomplete — programmatic-only (no judge verdicts): the honest partial
# path. faithfulness/hallucination/advice stay UNEVALUATED → RELEASE INCOMPLETE
# (exit 2). Demonstrates that a high programmatic score cannot certify release.
# ---------------------------------------------------------------------------
eval-incomplete:
	-uv run python -m src.eval.run_eval --replay datasets/fixtures/run_pass.jsonl

# ---------------------------------------------------------------------------
# ingest / run — live paths; require OPENAI_API_KEY + network access.
# ---------------------------------------------------------------------------
ingest:
	uv run python -m src.eval.run_eval --live

run:
	uv run python -m src.eval.run_eval --live

# ---------------------------------------------------------------------------
# calibrate — Cohen's κ agreement between the judge and the reference labels
# (offline, recorded verdicts). Add --live to call the real judge.
# ---------------------------------------------------------------------------
calibrate:
	uv run python -m src.eval.calibrate

# ---------------------------------------------------------------------------
# live-ingest-smoke — real-network SEC ingest smoke test (quarantined).
# Requires network access; resolves + downloads + chunks a live filing.
# ---------------------------------------------------------------------------
live-ingest-smoke:
	RUN_LIVE_INGEST=1 uv run pytest -m live -v

# ---------------------------------------------------------------------------
# test — run the full offline test suite
# ---------------------------------------------------------------------------
test:
	uv run pytest

# ---------------------------------------------------------------------------
# typecheck — mypy (strict, configured in pyproject.toml) over src AND tests.
# The whole repo is type-clean, so both are checked.
# ---------------------------------------------------------------------------
typecheck:
	uv run mypy src tests

# ---------------------------------------------------------------------------
# lint — ruff over the whole repo.
# ---------------------------------------------------------------------------
lint:
	uv run ruff check .

# ---------------------------------------------------------------------------
# check — the full local gate: lint + typecheck + tests (what CI runs offline).
# ---------------------------------------------------------------------------
check: lint typecheck test
