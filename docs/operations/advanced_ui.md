# Advanced Command Center UI (Batch G9) — Operations Guide

This document describes the **advanced cinematic demo UI** that
ships alongside the existing I3 web demo. The original demo under
`web/` remains the canonical "recruiter-friendly" entry point. The
advanced UI at `web/advanced/` is designed for a single purpose:
to compress every axis of the I3 system — chat, adaptation,
interpretability, routing, preference learning, safety — into one
visually stunning command-center view that can be driven through
a guided tour on stage, demoed live to an interviewer, or captured
as a backup video without a single click.

## 1. Design philosophy

### 1.1 Dark command-center aesthetic

The advanced UI chooses a dark, glossy command-center aesthetic over
the friendlier dark-but-forgiving palette of the original demo. The
goal is a single read at a glance: you should be able to stop the
screen at any frame and see, simultaneously, what the user is
typing, how fast they're typing it, what persona the system thinks
they're in, what adaptation dimensions it has reconfigured, which
arm the router picked, what the current P95 latency is, what it's
costing, and why the model answered the way it did. Seven panels,
one glance.

### 1.2 Palette discipline

The entire UI is restricted to **six** colours:

```
--bg        #1a1a2e   page
--panel     #16213e   panel surfaces
--accent    #0f3460   inputs, secondary bars
--hot       #e94560   live highlights, CTA, current state
--muted     #a0a0b0   secondary text, history points
--active    #f0f0f0   primary text, target markers
```

Any additional visual emphasis (uncertainty bands, faded history,
soft shadows) is produced by alpha-compositing one of the six, not
by introducing a new hue. The discipline is what makes the UI feel
deliberate: a seventh colour would break the illusion of a bespoke
piece of hardware.

### 1.3 Typography

Body type is rendered in the system sans-serif stack aliased as
"Inter"; numeric / data values use the system monospace stack
aliased as "JetBrains Mono". We do **not** load Google Fonts or
any other remote typography, because (a) the demo must work
offline on a projector with flaky conference WiFi, and (b) a
third-party font request is an information-disclosure vector that
does not add enough value to justify the exposure.

### 1.4 Motion

* Every transition uses the same `ease-out-cubic`
  (`cubic-bezier(0.215, 0.61, 0.355, 1)`) curve so the UI feels
  like one organism, not a collage of widgets.
* Micro interactions (button hover, gauge fill) complete in
  **250 ms**; panel-level transitions (phase change during the
  guided tour) complete in **600 ms**.
* `prefers-reduced-motion: reduce` collapses *all* durations to
  near zero. Purely decorative animations (the pulsing brand dot,
  the current-embedding mesh pulse) are disabled entirely.

## 2. Panel-by-panel walkthrough

### 2.1 Top bar

A 60-pixel strip containing: brand title (I³), a live
`HH:MM:SS` clock, the WebSocket connection indicator (dot + label,
with three states: `Connecting…` / `Live` / `Reconnecting…`), and
the operator controls (guided tour button, record-mode toggle,
shortcuts help). Focus-visible outlines are hot-coloured so the
keyboard user always knows where they are.

### 2.2 Chat panel (left)

The primary interaction surface. A rolling message log with
`role="log"` and `aria-live="polite"` so screen-reader users hear
incoming responses. The composition-intensity meter reads
inter-key intervals in real time, normalises them into `[0, 1]`,
and smooths them with a 40/60 EMA against the previous value. When
typing is fast the meter fills and glows; when typing slows or
stops it decays exponentially every 250 ms. This is the UX proxy
for the cognitive-load signal the backend computes server-side —
giving the user a visible, honest tell of what the system is
sensing.

### 2.3 3D embedding space (centre)

A `Three.js` orbital scene showing the last 200 64-dim user-state
embeddings projected into 3D via a seeded deterministic random
projection (xorshift-seeded `0x13ADEAD`). The projection matrix is
built once at page load and never changes, so a given embedding
maps to the same point every time — crucial for demoing
stability. The current embedding is a pulsing `#e94560` sphere;
historical points fade from `#a0a0b0` toward `#f0f0f0` with age.
If WebGL is unavailable or the Three.js CDN is blocked, the panel
soft-fails to a 2D canvas projection using the same matrix.

### 2.4 Adaptation gauges (top-right)

Eight radial SVG gauges, one per adaptation dimension:
`cognitive_load`, `style.{formality,verbosity,emotionality,directness}`,
`emotional_tone`, `accessibility`, and a `reserved` slot for the
next dimension we add. Each gauge has three visible arcs: a fixed
background, a translucent uncertainty band from the G2 UQ head
(width = 2σ), and the foreground value. A white tick indicates the
target value when present.

### 2.5 Cross-attention heatmap (mid-right)

A 4 × 4 grid where each cell's alpha encodes an attention weight.
Data comes from `/api/explain/adaptation` when fresh, from WS
"attention" frames when pushed, and from a synthetic soft-noise
pattern when both are absent. The cell colour is the hot colour
with alpha == value, so even the heatmap stays inside the palette.

### 2.6 Router dashboard (bottom-right)

Chart.js `doughnut` of arm counts (`local_slm`, `cloud`,
`local_reflect`) plus a line chart showing P50/P95 latency over
the last 60 samples, and a live cost counter in GBP. The cost
counter polls `/api/metrics` every 2 s when the endpoint exists
and falls back to a streaming-sum of `router` WS frames otherwise.

### 2.7 Interpretability strip (bottom)

Four columns:

* **Why this response** — the most recent counterfactual string.
  Also triggers an ephemeral toast in the bottom-right.
* **Uncertainty** — pills for the top-3 high-variance adaptation
  dimensions.
* **Safety** — refusal / moderation badges.
* **Preference A/B** — pulls `/api/preference/query/:user_id`
  every 5 s; renders the current comparison status when the
  active-preference selector fires.

## 3. Guided tour mechanics

Triggered by `Alt+T` or the top-bar button. The tour runs against
`user_id = "demo_user"` (sent via a `{type:"set_user"}` frame) so
that no real user state is ever mutated by running the demo.

Each phase consists of: (a) a 1.4 s full-screen overlay naming
the phase and a WCAG-friendly caption of what's happening, (b) a
synthetic adaptation frame that causes the gauges to swing even
when the backend is offline (useful for rehearsals), and (c) a
scripted sequence of typed lines at a phase-specific WPM. The
typing simulation dispatches real `keydown` events so the
composition-intensity meter reacts naturally — no special-casing.
Submission is via a synthetic `submit` event so the chat module
remains the single source of truth for outgoing messages.

Escape cancels immediately; the abort controller terminates any
pending timer and clears the overlay.

## 4. Screen-recording preset

Triggered by `Alt+S` or the *Record Mode* button. Adds a
`.record-mode` class to `<body>` that, via CSS custom properties:

* Bumps `--fs-base` from 14 px to 16 px (~15 %).
* Lengthens `--t-micro` and `--t-panel` by ~30 %.
* Hides dev/ghost controls (shortcut help, etc.).
* Switches the cursor to a thick SVG data-URL cursor that survives
  video codecs.
* Freezes grid `grid-template-columns` with `minmax(0, …)` so
  dynamic content can never push a panel sideways mid-capture.

The preference is persisted in `sessionStorage` so re-opening the
page during a recording session restores the preset.

## 5. Accessibility self-audit

We target WCAG 2.2 AA across the board. Self-audit results:

| Check                             | Status |
|-----------------------------------|--------|
| 1.3.1 Info and relationships      | Pass — `<h1>/<h2>`, `role=`, `aria-label` on every region. |
| 1.4.3 Contrast (minimum)          | Pass — all foreground text on palette backgrounds exceeds 4.5:1; large text exceeds 3:1. `Alt+A` re-runs the audit live. |
| 1.4.11 Non-text contrast          | Pass — focus outlines are 3:1 against the surface. |
| 1.4.12 Text spacing               | Pass — no fixed line-height below 1.5, no fixed letter-spacing that resists user overrides. |
| 2.1.1 Keyboard                    | Pass — every control is reachable via `Tab`; no mouse-only affordances. |
| 2.4.1 Bypass blocks               | Pass — skip-to-main link first in tab order. |
| 2.4.7 Focus visible               | Pass — `:focus-visible` 2 px outline in hot colour, 2 px offset. |
| 2.5.3 Label in name               | Pass — accessible name includes visible text for every button. |
| 2.5.8 Target size                 | Pass — all interactive elements ≥ 24 × 24 px. |
| 3.2.6 Consistent help             | Pass — `?` always opens the same shortcuts dialog. |
| 4.1.3 Status messages             | Pass — all dynamic regions use `aria-live="polite"`. |

The runtime contrast auditor in `a11y.js` walks every text-bearing
element, computes the relative-luminance ratio against its nearest
opaque ancestor, and warns in the console if any pair fails AA.
Run it with `Alt+A` during a rehearsal; a clean run is logged as a
single info message.

## 6. Integration with existing routes

The advanced UI does not own any new backend routes. It reads from
the existing ones:

| Route                                   | Usage                               |
|-----------------------------------------|-------------------------------------|
| `/ws`                                   | Live frames (chat, adaptation, embedding, attention, router). |
| `/api/explain/adaptation`               | Counterfactual + cross-attention matrix (fallback). |
| `/api/preference/query/:user_id`        | Active A/B comparison status. |
| `/api/metrics`                          | Cost and latency ground-truth (optional). |
| `/api/admin/reset`                      | Alt+R reset (no-op if admin is disabled via `I3_DISABLE_ADMIN`). |

## 7. Trade-offs

### 7.1 Vanilla JS vs React

We deliberately chose vanilla ES modules over React for three
reasons:

1. **Consistency with `web/`.** The existing demo is vanilla; a
   React island would introduce a second framework and a build
   step just for one corner of the UI.
2. **No build step.** The entire Batch G9 diff is static files
   plus one `app.mount` line. There is no webpack, rollup, vite,
   package.json, or node_modules. The only server change is a
   single additive line in `server/app.py`.
3. **Offline-safe.** With no remote dependency resolution (npm,
   CDNs for framework code) the UI ships fully functional on a
   plane-WiFi laptop. The CDN-loaded vendors (Three.js, Chart.js)
   are strictly optional — every panel degrades gracefully.

The cost is hand-written imperative DOM code in several panels
(`radial_gauges.js`, `attention_heatmap.js`). We mitigate by
keeping each panel in its own module with a narrow public API.

### 7.2 CDN-loaded vendors vs local copies

See `web/advanced/vendor/README.md`. In short: CDN + SRI is
smaller, auditable, and good enough for a demo. Replace SRI
placeholders with real `sha384-...` hashes before any production
deployment.

### 7.3 Synthetic fallbacks vs hard errors

Every remote data source has a synthetic fallback so that a panel
never appears broken. This is the right trade-off for a *demo*: a
panel that says "warming up" while showing plausible values is a
vastly better experience than one that shows a red error state
and makes the audience wonder whether the whole system is down.
In production, we would still render the fallback but also log a
structured metric so operators know the real source was unreachable.

## 8. Deploying changes

The UI is a pure static-file bundle. To change it:

1. Edit files under `D:/implicit-interaction-intelligence/web/advanced/`.
2. Reload `http://localhost:8000/advanced/` in the browser.

No rebuild. No restart. No cache-bust. If you change the CSP or
add a new vendor script, regenerate the SRI hash (see
`vendor/README.md`) and update both `integrity` and the CSP
`script-src` origin simultaneously.
