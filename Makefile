.PHONY: ingest run score eval eval-incomplete demo-block test

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
# test — run the full offline test suite
# ---------------------------------------------------------------------------
test:
	uv run pytest
