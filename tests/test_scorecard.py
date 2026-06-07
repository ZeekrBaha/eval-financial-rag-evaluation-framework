"""Tests for T13 — scorecard.py (E-08/E-10).

TDD: tests written before the implementation.

Coverage:
  - render_json: writes valid JSON with expected structure.
  - render_text: ASCII scorecard contains STATUS, dimension row, overall.
  - render_html: contains status banner text, dimension table, IBM Plex Mono font,
    and does NOT convey status by color alone (text label present).
  - NA dimension rendered gracefully in all three outputs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.eval.golden import load_goldens
from src.eval.runner import load_replay
from src.eval.metrics.programmatic import score_programmatic
from src.eval.aggregate import Dimension, Scorecard, build_scorecard
from src.eval.scorecard import render_html, render_json, render_text

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

DATASETS = Path(__file__).parent.parent / "datasets"
GOLDEN_SET = DATASETS / "golden_set.jsonl"
RUN_PASS = DATASETS / "fixtures" / "run_pass.jsonl"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scorecard_run_pass() -> Scorecard:
    goldens = load_goldens(GOLDEN_SET)
    records = load_replay(RUN_PASS)
    prog_results = score_programmatic(records, goldens)
    return build_scorecard(
        records,
        goldens,
        prog_results=prog_results,
        run_id="sc_test",
        mode="replay",
    )


@pytest.fixture
def minimal_all_na_scorecard() -> Scorecard:
    """A scorecard where every dimension is NA and overall is None."""
    from src.config import DIMENSION_WEIGHTS
    dims = [
        Dimension(name=n, weight=w, score=None, status="na", metrics={})
        for n, w in DIMENSION_WEIGHTS.items()
    ]
    return Scorecard(
        run_id="na-test",
        mode="replay",
        dimensions=dims,
        buckets={"factual_lookup": 0.5},
        overall=None,
        metric_summary={"citation_validity": None},
        status="PENDING",
    )


# ---------------------------------------------------------------------------
# render_json
# ---------------------------------------------------------------------------


class TestRenderJson:
    def test_writes_valid_json(self, scorecard_run_pass: Scorecard, tmp_path: Path) -> None:
        out = tmp_path / "scorecard.json"
        render_json(scorecard_run_pass, out)
        with out.open() as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_json_has_run_id(self, scorecard_run_pass: Scorecard, tmp_path: Path) -> None:
        out = tmp_path / "scorecard.json"
        render_json(scorecard_run_pass, out)
        data = json.loads(out.read_text())
        assert data["run_id"] == "sc_test"

    def test_json_has_dimensions_list(self, scorecard_run_pass: Scorecard, tmp_path: Path) -> None:
        out = tmp_path / "scorecard.json"
        render_json(scorecard_run_pass, out)
        data = json.loads(out.read_text())
        assert "dimensions" in data
        assert isinstance(data["dimensions"], list)
        from src.config import DIMENSION_WEIGHTS
        assert len(data["dimensions"]) == len(DIMENSION_WEIGHTS)

    def test_json_has_overall(self, scorecard_run_pass: Scorecard, tmp_path: Path) -> None:
        out = tmp_path / "scorecard.json"
        render_json(scorecard_run_pass, out)
        data = json.loads(out.read_text())
        assert "overall" in data
        assert isinstance(data["overall"], float)

    def test_json_has_buckets(self, scorecard_run_pass: Scorecard, tmp_path: Path) -> None:
        out = tmp_path / "scorecard.json"
        render_json(scorecard_run_pass, out)
        data = json.loads(out.read_text())
        assert "buckets" in data
        assert isinstance(data["buckets"], dict)

    def test_json_has_metric_summary(self, scorecard_run_pass: Scorecard, tmp_path: Path) -> None:
        out = tmp_path / "scorecard.json"
        render_json(scorecard_run_pass, out)
        data = json.loads(out.read_text())
        assert "metric_summary" in data

    def test_json_has_status(self, scorecard_run_pass: Scorecard, tmp_path: Path) -> None:
        out = tmp_path / "scorecard.json"
        render_json(scorecard_run_pass, out)
        data = json.loads(out.read_text())
        assert "status" in data

    def test_json_floats_rounded(self, scorecard_run_pass: Scorecard, tmp_path: Path) -> None:
        """Floats must be rounded (not long decimals like 98.95833333333333)."""
        out = tmp_path / "scorecard.json"
        render_json(scorecard_run_pass, out)
        raw = out.read_text()
        # Find any decimal numbers with more than 3 decimal places
        long_decimals = re.findall(r"\d+\.\d{4,}", raw)
        assert not long_decimals, f"Unrounded floats found: {long_decimals}"

    def test_json_na_dimension_score_is_null(
        self, minimal_all_na_scorecard: Scorecard, tmp_path: Path
    ) -> None:
        out = tmp_path / "scorecard.json"
        render_json(minimal_all_na_scorecard, out)
        data = json.loads(out.read_text())
        for dim in data["dimensions"]:
            assert dim["score"] is None

    def test_json_none_overall_is_null(
        self, minimal_all_na_scorecard: Scorecard, tmp_path: Path
    ) -> None:
        out = tmp_path / "scorecard.json"
        render_json(minimal_all_na_scorecard, out)
        data = json.loads(out.read_text())
        assert data["overall"] is None


# ---------------------------------------------------------------------------
# render_text
# ---------------------------------------------------------------------------


class TestRenderText:
    def test_returns_string(self, scorecard_run_pass: Scorecard) -> None:
        text = render_text(scorecard_run_pass)
        assert isinstance(text, str)

    def test_contains_status(self, scorecard_run_pass: Scorecard) -> None:
        text = render_text(scorecard_run_pass)
        assert "PENDING" in text

    def test_contains_run_id(self, scorecard_run_pass: Scorecard) -> None:
        text = render_text(scorecard_run_pass)
        assert "sc_test" in text

    def test_contains_a_dimension_row(self, scorecard_run_pass: Scorecard) -> None:
        """At least one dimension name must appear in the text output."""
        text = render_text(scorecard_run_pass)
        assert "retrieval_quality" in text or "financial_correctness" in text

    def test_contains_overall(self, scorecard_run_pass: Scorecard) -> None:
        """The word 'overall' or 'Overall' must appear."""
        text = render_text(scorecard_run_pass)
        assert "overall" in text.lower()

    def test_contains_bucket_info(self, scorecard_run_pass: Scorecard) -> None:
        """At least one bucket name must appear."""
        text = render_text(scorecard_run_pass)
        assert any(
            b in text
            for b in ["factual_lookup", "negative", "entity", "temporal"]
        )

    def test_na_dimension_shown_gracefully(self, scorecard_run_pass: Scorecard) -> None:
        """NA dimensions must render with a placeholder, not crash."""
        text = render_text(scorecard_run_pass)
        # NA is signalled by the ⚪ emoji or a dash/NA label
        assert "na" in text.lower() or "⚪" in text or "—" in text or "N/A" in text

    def test_none_overall_renders_gracefully(
        self, minimal_all_na_scorecard: Scorecard
    ) -> None:
        text = render_text(minimal_all_na_scorecard)
        assert isinstance(text, str)
        # Should contain a placeholder for absent overall
        assert "—" in text or "N/A" in text or "n/a" in text.lower() or "none" in text.lower()

    def test_green_dim_has_green_emoji(self, scorecard_run_pass: Scorecard) -> None:
        """Green dimensions must include the 🟢 emoji."""
        text = render_text(scorecard_run_pass)
        assert "🟢" in text

    def test_na_dim_has_white_circle(self, scorecard_run_pass: Scorecard) -> None:
        """NA dimensions must include the ⚪ emoji."""
        text = render_text(scorecard_run_pass)
        assert "⚪" in text


# ---------------------------------------------------------------------------
# render_html
# ---------------------------------------------------------------------------


class TestRenderHtml:
    def test_writes_file(self, scorecard_run_pass: Scorecard, tmp_path: Path) -> None:
        out = tmp_path / "scorecard.html"
        render_html(scorecard_run_pass, out)
        assert out.exists()

    def test_is_valid_html_skeleton(self, scorecard_run_pass: Scorecard, tmp_path: Path) -> None:
        out = tmp_path / "scorecard.html"
        render_html(scorecard_run_pass, out)
        html = out.read_text()
        assert "<html" in html
        assert "</html>" in html

    def test_html_contains_status_text_label(
        self, scorecard_run_pass: Scorecard, tmp_path: Path
    ) -> None:
        """Status must be conveyed as text (not just color), for accessibility."""
        out = tmp_path / "scorecard.html"
        render_html(scorecard_run_pass, out)
        html = out.read_text()
        assert "PENDING" in html

    def test_html_references_ibm_plex_mono(
        self, scorecard_run_pass: Scorecard, tmp_path: Path
    ) -> None:
        out = tmp_path / "scorecard.html"
        render_html(scorecard_run_pass, out)
        html = out.read_text()
        assert "IBM Plex Mono" in html

    def test_html_references_ibm_plex_sans(
        self, scorecard_run_pass: Scorecard, tmp_path: Path
    ) -> None:
        out = tmp_path / "scorecard.html"
        render_html(scorecard_run_pass, out)
        html = out.read_text()
        assert "IBM Plex Sans" in html

    def test_html_has_dark_background_token(
        self, scorecard_run_pass: Scorecard, tmp_path: Path
    ) -> None:
        """Dark bg color #0F1115 must appear in the HTML."""
        out = tmp_path / "scorecard.html"
        render_html(scorecard_run_pass, out)
        html = out.read_text()
        # Case-insensitive check
        assert "0f1115" in html.lower() or "#0F1115" in html

    def test_html_dimension_table_present(
        self, scorecard_run_pass: Scorecard, tmp_path: Path
    ) -> None:
        """HTML must contain a table or grid for dimensions."""
        out = tmp_path / "scorecard.html"
        render_html(scorecard_run_pass, out)
        html = out.read_text()
        assert "<table" in html.lower() or "dimension" in html.lower()

    def test_html_contains_bucket_section(
        self, scorecard_run_pass: Scorecard, tmp_path: Path
    ) -> None:
        out = tmp_path / "scorecard.html"
        render_html(scorecard_run_pass, out)
        html = out.read_text()
        assert "bucket" in html.lower() or "factual_lookup" in html

    def test_html_status_not_color_only(
        self, scorecard_run_pass: Scorecard, tmp_path: Path
    ) -> None:
        """Status must appear as a TEXT label (not just a background color).

        Accessibility requirement: status conveyed by both color AND text.
        We verify this by confirming 'PENDING' (or another status string)
        appears as visible text content, not just inside a color property.
        """
        out = tmp_path / "scorecard.html"
        render_html(scorecard_run_pass, out)
        html = out.read_text()
        # 'PENDING' must appear outside a CSS rule (i.e., as content text)
        # Simplest check: it appears in a non-style context.
        # We strip <style> blocks and check if PENDING still appears.
        without_style = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
        assert "PENDING" in without_style

    def test_html_status_pass_color_present(
        self, scorecard_run_pass: Scorecard, tmp_path: Path
    ) -> None:
        """At least one design-system status color token must appear in CSS."""
        out = tmp_path / "scorecard.html"
        render_html(scorecard_run_pass, out)
        html = out.read_text()
        status_colors = ["2ebd85", "e8b339", "e5484d"]
        assert any(c in html.lower() for c in status_colors)

    def test_html_na_dimension_renders_gracefully(
        self, minimal_all_na_scorecard: Scorecard, tmp_path: Path
    ) -> None:
        out = tmp_path / "scorecard.html"
        render_html(minimal_all_na_scorecard, out)
        html = out.read_text()
        assert isinstance(html, str)
        assert len(html) > 100

    def test_html_none_overall_renders_gracefully(
        self, minimal_all_na_scorecard: Scorecard, tmp_path: Path
    ) -> None:
        out = tmp_path / "scorecard.html"
        render_html(minimal_all_na_scorecard, out)
        html = out.read_text()
        assert "N/A" in html or "—" in html or "n/a" in html.lower()

    def test_html_no_external_resources(
        self, scorecard_run_pass: Scorecard, tmp_path: Path
    ) -> None:
        """self-contained: no <script src=...>, no <link rel=stylesheet href=...>."""
        out = tmp_path / "scorecard.html"
        render_html(scorecard_run_pass, out)
        html = out.read_text()
        assert not re.search(r'<script\s[^>]*src\s*=', html, re.IGNORECASE)
        assert not re.search(r'<link\s[^>]*rel\s*=\s*["\']stylesheet', html, re.IGNORECASE)
