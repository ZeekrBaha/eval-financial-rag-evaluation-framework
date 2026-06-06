.PHONY: ingest run score eval demo-block test

# Replay fixture to use for `make eval` (override on command line if needed).
# Default → run_pass.jsonl so `make eval` exits 0 in CI.
REPLAY ?= datasets/fixtures/run_pass.jsonl

# ---------------------------------------------------------------------------
# eval / score — offline replay, no API key required.
# Default fixture (run_pass) exits 0; swap REPLAY= to run_fail for a 1.
# ---------------------------------------------------------------------------
eval:
	uv run python -m src.eval.run_eval --replay $(REPLAY)

# score is an alias for eval (same pipeline)
score: eval

# ---------------------------------------------------------------------------
# demo-block — run the failing fixture to show the RELEASE BLOCKED money-shot.
# Non-zero exit is expected here; prefix with - so make itself does not error.
# ---------------------------------------------------------------------------
demo-block:
	-uv run python -m src.eval.run_eval --replay datasets/fixtures/run_fail.jsonl

# ---------------------------------------------------------------------------
# ingest / run — live paths; require OPENAI_API_KEY + network access.
# ---------------------------------------------------------------------------
ingest:
	uv run python -m src.eval.run_eval --live

run:
	uv run python -m src.eval.run_eval --live

# ---------------------------------------------------------------------------
# test — run the full offline test suite
# ---------------------------------------------------------------------------
test:
	uv run pytest
