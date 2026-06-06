"""Scorecard renderers for T13 / E-08.

Three output formats for a Scorecard produced by build_scorecard():
  render_json(scorecard, path) — machine-readable JSON
  render_text(scorecard) -> str — ASCII console scorecard (what `make eval` prints)
  render_html(scorecard, path) — self-contained HTML using design-system tokens

Design-system tokens (from docs/implementation/design-system.md):
  Font body:   IBM Plex Sans
  Font mono:   IBM Plex Mono  (all numbers)
  Dark bg:     #0F1115
  Status pass: #2EBD85
  Status warn: #E8B339
  Status fail: #E5484D

Accessibility requirement (E-08):
  Status is conveyed by BOTH color AND a text label — never color alone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.eval.aggregate import Dimension, Scorecard
from src.eval.golden import Bucket

# ---------------------------------------------------------------------------
# Design-system tokens
# ---------------------------------------------------------------------------

_BG = "#0F1115"
_FG = "#E8E8E8"
_COLOR_PASS = "#2EBD85"
_COLOR_WARN = "#E8B339"
_COLOR_FAIL = "#E5484D"
_COLOR_NA = "#888888"
_FONT_SANS = "IBM Plex Sans, ui-sans-serif, system-ui, sans-serif"
_FONT_MONO = "IBM Plex Mono, ui-monospace, monospace"

# Status → color mapping (used in HTML; also applied in text with emoji fallback).
_STATUS_COLOR: dict[str, str] = {
    "PASS": _COLOR_PASS,
    "WARN": _COLOR_WARN,
    "FAIL": _COLOR_FAIL,
    "PENDING": _COLOR_WARN,
}

_DIM_STATUS_EMOJI: dict[str, str] = {
    "green": "🟢",
    "yellow": "🟡",
    "red": "🔴",
    "na": "⚪",
}

_DIM_STATUS_COLOR: dict[str, str] = {
    "green": _COLOR_PASS,
    "yellow": _COLOR_WARN,
    "red": _COLOR_FAIL,
    "na": _COLOR_NA,
}


# ---------------------------------------------------------------------------
# Rounding helper
# ---------------------------------------------------------------------------


def _round_float(v: float | None, ndigits: int = 2) -> float | None:
    """Round a float to *ndigits* decimal places, or return None unchanged."""
    if v is None:
        return None
    return round(v, ndigits)


def _fmt_score(score: float | None) -> str:
    """Format a score (0-100) for display; '—' when None."""
    if score is None:
        return "—"
    return f"{score:.1f}"


def _fmt_rate(rate: float | None) -> str:
    """Format a rate (0-1) as a percentage; '—' when None."""
    if rate is None:
        return "—"
    return f"{rate * 100:.0f}%"


# ---------------------------------------------------------------------------
# render_json
# ---------------------------------------------------------------------------


def _scorecard_to_dict(scorecard: Scorecard) -> dict[str, Any]:
    """Convert Scorecard to a JSON-serialisable dict with rounded floats."""
    dims = []
    for d in scorecard.dimensions:
        dims.append({
            "name": d.name,
            "weight": d.weight,
            "score": _round_float(d.score, 2),
            "status": d.status,
            "metrics": {
                k: _round_float(v, 3)
                for k, v in d.metrics.items()
            },
        })

    buckets = {
        k: _round_float(v, 3)
        for k, v in scorecard.buckets.items()
    }

    metric_summary = {
        k: _round_float(v, 3)
        for k, v in scorecard.metric_summary.items()
    }

    return {
        "run_id": scorecard.run_id,
        "mode": scorecard.mode,
        "status": scorecard.status,
        "hard_gate_failures": scorecard.hard_gate_failures,
        "overall": _round_float(scorecard.overall, 2),
        "dimensions": dims,
        "buckets": buckets,
        "metric_summary": metric_summary,
    }


def render_json(scorecard: Scorecard, path: str | Path) -> None:
    """Write *scorecard* as a pretty-printed JSON file at *path*.

    Floats are rounded to 1-3 decimal places depending on field type.
    None values are serialised as JSON null.
    """
    path = Path(path)
    data = _scorecard_to_dict(scorecard)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# render_text
# ---------------------------------------------------------------------------


def render_text(scorecard: Scorecard) -> str:
    """Return an ASCII scorecard string suitable for console output.

    Structure:
      ┌─ STATUS banner ─────────────────────────────────────────┐
      │ Run:   <run_id>   Mode: <mode>                          │
      ├─ Dimensions ────────────────────────────────────────────┤
      │ Emoji  Dimension               Weight   Score  Status   │
      │  🟢   retrieval_quality          20     95.8   green    │
      │  ⚪   robustness                 10      —      na       │
      │ ...                                                      │
      ├─ Buckets ───────────────────────────────────────────────┤
      │  factual_lookup  100%    temporal  33%  ...              │
      ├─ Overall ───────────────────────────────────────────────┤
      │  98.96 / 100                                             │
      └──────────────────────────────────────────────────────────┘
    """
    lines: list[str] = []
    width = 70
    sep = "─" * width

    status_color = _STATUS_COLOR.get(scorecard.status, _COLOR_NA)

    lines.append(sep)
    lines.append(f"  STATUS: {scorecard.status}   run={scorecard.run_id}   mode={scorecard.mode}")
    lines.append(sep)

    # Dimension table header
    col_hdr = f"  {'':2}  {'Dimension':<28} {'Weight':>6}  {'Score':>6}  {'Status':<8}"
    lines.append(col_hdr)
    lines.append("  " + "─" * (width - 2))

    for d in scorecard.dimensions:
        emoji = _DIM_STATUS_EMOJI.get(d.status, "⚪")
        score_str = _fmt_score(d.score)
        row = (
            f"  {emoji}  {d.name:<28} {d.weight:>6}  {score_str:>6}  {d.status:<8}"
        )
        lines.append(row)

    lines.append(sep)

    # Buckets — rendered in canonical Bucket enum order
    lines.append("  Buckets (item pass-rate):")
    bucket_parts = []
    for b in Bucket:
        rate = scorecard.buckets.get(b.value)
        bucket_parts.append(f"{b.value}: {_fmt_rate(rate)}")
    # Wrap into lines of ~width
    bucket_line = "  " + "   ".join(bucket_parts)
    lines.append(bucket_line)

    lines.append(sep)

    # Overall
    overall_str = _fmt_score(scorecard.overall)
    lines.append(f"  Overall: {overall_str} / 100")
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# render_html
# ---------------------------------------------------------------------------


def _html_dim_row(d: Dimension) -> str:
    """Return a <tr> for one dimension."""
    emoji = _DIM_STATUS_EMOJI.get(d.status, "⚪")
    color = _DIM_STATUS_COLOR.get(d.status, _COLOR_NA)
    score_cell = (
        f'<span style="font-family:{_FONT_MONO}">{_fmt_score(d.score)}</span>'
    )
    label_cell = f'<span style="color:{color}">{d.status.upper()}</span>'
    return (
        f"<tr>"
        f"<td>{emoji}</td>"
        f"<td>{d.name}</td>"
        f'<td style="text-align:right;font-family:{_FONT_MONO}">{d.weight}</td>'
        f'<td style="text-align:right">{score_cell}</td>'
        f"<td>{label_cell}</td>"
        f"</tr>"
    )


def _html_bucket_rows(buckets: dict[str, float | None]) -> str:
    rows: list[str] = []
    # Render in canonical Bucket enum order; None values display as "—"
    for b in Bucket:
        rate = buckets.get(b.value)
        pct_str = f'<span style="font-family:{_FONT_MONO}">{_fmt_rate(rate)}</span>'
        rows.append(
            f"<tr><td>{b.value}</td><td style='text-align:right'>{pct_str}</td></tr>"
        )
    return "\n".join(rows)


def render_html(scorecard: Scorecard, path: str | Path) -> None:
    """Write *scorecard* as a self-contained HTML file at *path*.

    Design-system compliance:
      - Dark background #0F1115
      - IBM Plex Sans for body text
      - IBM Plex Mono for all numeric values
      - Status colors #2EBD85 / #E8B339 / #E5484D
      - Status conveyed by BOTH color AND a text label (accessibility)
      - No external CSS or JS dependencies
    """
    path = Path(path)

    status = scorecard.status
    banner_color = _STATUS_COLOR.get(status, _COLOR_NA)
    overall_str = _fmt_score(scorecard.overall)

    dim_rows_html = "\n".join(_html_dim_row(d) for d in scorecard.dimensions)
    bucket_rows_html = _html_bucket_rows(scorecard.buckets)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Financial RAG Scorecard — {scorecard.run_id}</title>
<style>
  /* Design-system tokens */
  :root {{
    --bg:           {_BG};
    --fg:           {_FG};
    --color-pass:   {_COLOR_PASS};
    --color-warn:   {_COLOR_WARN};
    --color-fail:   {_COLOR_FAIL};
    --color-na:     {_COLOR_NA};
    --font-sans:    {_FONT_SANS};
    --font-mono:    {_FONT_MONO};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--fg);
    font-family: var(--font-sans);
    padding: 2rem;
    line-height: 1.6;
  }}
  h1, h2, h3 {{ margin-bottom: 0.5rem; }}
  h2 {{ margin-top: 2rem; font-size: 1rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: #888; }}
  .banner {{
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 4px;
    background: {banner_color};
    color: #000;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
  }}
  .meta {{
    color: #999;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
    font-family: var(--font-mono);
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin-top: 0.5rem;
  }}
  th {{
    text-align: left;
    padding: 0.4rem 0.8rem;
    border-bottom: 1px solid #333;
    color: #aaa;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}
  td {{
    padding: 0.45rem 0.8rem;
    border-bottom: 1px solid #1e2025;
    font-size: 0.9rem;
  }}
  tr:last-child td {{ border-bottom: none; }}
  .overall-box {{
    margin-top: 2rem;
    padding: 1rem 1.5rem;
    border: 1px solid #333;
    border-radius: 6px;
    display: inline-block;
  }}
  .overall-label {{ color: #999; font-size: 0.8rem; text-transform: uppercase;
                    letter-spacing: 0.08em; }}
  .overall-value {{
    font-family: var(--font-mono);
    font-size: 2rem;
    color: var(--fg);
    margin-top: 0.2rem;
  }}
</style>
</head>
<body>

<h1>Financial RAG Evaluation Scorecard</h1>

<!-- STATUS BANNER — status conveyed by both color (background) and text label -->
<div class="banner" role="status" aria-label="Evaluation status: {status}">{status}</div>

<div class="meta">
  run_id: {scorecard.run_id} &nbsp;|&nbsp; mode: {scorecard.mode}
</div>

<h2>Dimensions</h2>
<table>
  <thead>
    <tr>
      <th></th>
      <th>Dimension</th>
      <th style="text-align:right">Weight</th>
      <th style="text-align:right">Score (0-100)</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
{dim_rows_html}
  </tbody>
</table>

<h2>Per-Bucket Pass Rate</h2>
<table>
  <thead>
    <tr>
      <th>Bucket</th>
      <th style="text-align:right">Pass Rate</th>
    </tr>
  </thead>
  <tbody>
{bucket_rows_html}
  </tbody>
</table>

<div class="overall-box">
  <div class="overall-label">Overall Score</div>
  <div class="overall-value">{overall_str if scorecard.overall is not None else "N/A"} <span style="font-size:1rem;color:#888">/ 100</span></div>
</div>

</body>
</html>"""

    path.write_text(html, encoding="utf-8")
