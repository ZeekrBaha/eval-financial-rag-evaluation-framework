# design-system.md — Scorecard & Dashboard tokens

> Scope: this project is a Python eval framework. The only UI surfaces are (1) the **HTML scorecard** (always built) and (2) an optional **Streamlit dashboard** (Phase 2). Tokens pinned here so output is not AI-slop default. Reuse this block verbatim in any screen/report spec.

## Design tokens

```yaml
# Typography — NOT Inter/Roboto/Arial (slop ban)
font_sans:  "IBM Plex Sans", system-ui, sans-serif      # finance/data feel
font_mono:  "IBM Plex Mono", ui-monospace, monospace     # numbers, scores, gate lines
type_scale:  [12, 14, 16, 20, 28, 40]                     # modular, two faces max

# Color — neutral ramp + one accent + semantic status (no purple/indigo gradients)
bg:          "#0F1115"   # real backdrop, not #fff (dark report)
surface:     "#171A21"
border:      "#262B36"
text:        "#E6E9EF"
text_muted:  "#9AA4B2"
accent:      "#2D7FF9"   # single accent (links, headers)
status_pass: "#2EBD85"   # 🟢
status_warn: "#E8B339"   # 🟡
status_fail: "#E5484D"   # 🔴 (RELEASE BLOCKED)

# Shape / spacing / elevation
radius:      [4, 8]                       # by role, not 16-on-everything
spacing:     [4, 8, 12, 16, 24, 32]       # 4pt grid
elevation:   "0 1px 2px rgba(0,0,0,.4)"   # by role, not one shadow everywhere

# Icons — single SVG set, one stroke width; no emoji-as-icons (status dots are OK as legend)
icons:       "Lucide, 1.5px stroke"
```

## Scorecard layout rules

- Top banner = overall status, full-width, colored by `status_*`. `RELEASE BLOCKED` in `font_mono`, `status_fail`, with the failing gate + value inline.
- Dimension table: name · weight · score/100 · status dot. Numbers in `font_mono`, right-aligned.
- Per-bucket breakdown below the dimension table.
- Every state designed: empty (no run yet), running, errored metric, success, blocked.

## Anti-slop checklist (UI gate)

- [ ] No Inter/Roboto/Arial; no purple→blue gradient; no emoji-as-icons; no hero→3-cards template.
- [ ] Real data (actual scores), never lorem ipsum.
- [ ] Contrast ≥ 4.5:1 on text; status colors distinguishable + labeled (not color-only).
- [ ] Mono font for all numeric/score/gate output.
