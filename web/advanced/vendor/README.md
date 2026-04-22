# Vendor assets — CDN pinning policy

The advanced command-center UI ships *no* bundled third-party
JavaScript. All vendor libraries are loaded at runtime from a
pinned CDN URL and verified with a Subresource Integrity (SRI)
`integrity="sha384-..."` attribute. This directory therefore
contains **no code** — it is documentation only.

## Why CDN instead of local copies?

* **Offline-safe fallback**: every panel in `web/advanced/js` is
  written so it soft-fails when its vendor is unavailable
  (`embedding_3d.js` falls back to a 2D canvas; `router_dashboard.js`
  falls back to hand-drawn placeholders). The UI is therefore
  never "broken" if a CDN is unreachable.
* **No build step**: vanilla ES modules only. A local `vendor/`
  copy would require a publishing step or a build script — out of
  scope for Batch G9.
* **Smaller git repo**: Three.js r164 alone is 1.2 MB minified.

## Pinned versions

| Library   | Version | URL                                                                   |
|-----------|---------|-----------------------------------------------------------------------|
| Three.js  | r164.1  | `https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.min.js`       |
| Chart.js  | 4.4.4   | `https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.js`       |
| d3 (opt.) | 7.9.0   | `https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js` *(not loaded in G9)* |

## Computing the SRI hashes

`index.html` currently ships with placeholder hashes
(`sha384-placeholder-...`) to make the policy obvious during code
review. Before a production deploy, regenerate real hashes with:

```
curl -sL https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.min.js \
  | openssl dgst -sha384 -binary \
  | openssl base64 -A
```

Prefix the output with `sha384-` and paste into the `integrity`
attribute in `web/advanced/index.html`.

## Content Security Policy

The `<meta http-equiv="Content-Security-Policy">` header in
`index.html` restricts `script-src` to `'self'` +
`https://cdn.jsdelivr.net` + `https://unpkg.com`. If you change
the CDN, update the CSP *and* the SRI hashes simultaneously.

## No runtime network surprises

* No Google Fonts. Typography uses system stacks aliased as
  "Inter" and "JetBrains Mono".
* No analytics. No tracking pixels.
* No WebSocket endpoints other than the app's own `/ws`.
