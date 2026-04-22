# I3 Command Center (Advanced UI)

A premium, single-view command-center dashboard for the
Implicit Interaction Intelligence demo. Alongside the existing
`web/` UI (unchanged), this advanced layer lives at
`web/advanced/` and is served at:

    http://localhost:8000/advanced/

## What it shows

Seven panels arranged in a single CSS Grid:

1. **Top bar** — title, live clock, connection status, tour and
   record-mode controls.
2. **Left — Chat** with a live composition-intensity meter that
   pulses with inter-key interval.
3. **Centre — 3D embedding space** (Three.js). A rotating sphere of
   the last 200 user-state embeddings colour-coded by persona. The
   current state glows and pulses; older states fade.
4. **Top-right — Adaptation gauges**. Eight radial SVG gauges with
   animated fills, target markers, and uncertainty bands from the
   G2 UQ head.
5. **Mid-right — Cross-attention heatmap** (4 × 4), live-animated
   from `/api/explain/adaptation` with a synthetic fallback.
6. **Bottom-right — Router dashboard**. Donut of
   `local_slm` / `cloud` / `local_reflect` counts, P50/P95 latency
   line chart, and a live cost counter in GBP with per-provider
   breakdown.
7. **Bottom strip — Interpretability**. Counterfactual
   "why this response" text, uncertainty pills, refusal badges,
   and an active-preference A/B status readout.

## Guided tour

Press **Alt+T** (or click *Guided Tour*) to start a cinematic,
autonomous walkthrough of the four demo phases:

| Phase            | Typing speed | Adaptation highlight     |
|------------------|--------------|--------------------------|
| Cold Start       | 60 WPM       | Neutral defaults         |
| Energetic        | 100 WPM      | High emotional tone      |
| Fatigue          | 30 WPM       | High cognitive load      |
| Accessibility    | 15 WPM       | Accessibility mode on    |

The tour is isolated to `user_id = "demo_user"` and can be
cancelled at any time with **Escape**.

## Screen-recording preset

Press **Alt+S** (or click *Record Mode*) to:

* Hide dev / ghost controls.
* Bump base font size by ~15 %.
* Slow transitions by ~30 % for smoother capture.
* Use a high-visibility cursor that survives video codecs.
* Freeze grid track sizes to prevent mid-recording layout shift.

## Keyboard shortcuts

| Key         | Action                         |
|-------------|--------------------------------|
| `Alt+T`     | Guided tour (toggle)           |
| `Alt+R`     | Reset demo user                |
| `Alt+W`     | What-if panel                  |
| `Alt+A`     | A11y contrast audit (console)  |
| `Alt+M`     | Toggle reduced motion          |
| `Alt+S`     | Screen-recording preset        |
| `?`         | Shortcut help overlay          |
| `Esc`       | Cancel / close overlay         |

## Accessibility

* WCAG 2.2 AA minimum contrast on all text/background pairs
  (self-audited with `Alt+A`).
* Every dynamic region has `aria-live="polite"`.
* Fully keyboard-navigable; focus-visible outlines everywhere.
* `prefers-reduced-motion: reduce` collapses all non-essential
  motion to 0 ms.
* Skip-to-main-content link at the top of the document.

## Customising the palette

The entire UI is driven from six CSS custom properties in the
`:root` block of `css/command_center.css`:

```css
--bg:      #1a1a2e;
--panel:   #16213e;
--accent:  #0f3460;
--hot:     #e94560;
--muted:   #a0a0b0;
--active:  #f0f0f0;
```

Change these and the rest of the UI follows. Do *not* introduce
a seventh colour: the palette discipline is what makes the demo
look cinematic.

## How the UI is served

`server/app.py` mounts this directory in addition to the existing
`web/` mount:

```python
app.mount("/", StaticFiles(directory="web", html=True), name="static")
# Advanced cinematic demo UI (Batch G9) — served at /advanced.
app.mount("/advanced", StaticFiles(directory="web/advanced", html=True), name="advanced_ui")
```

No other server changes were required for Batch G9.
