# I3 Advanced Insights Panels

These optional panels augment the main I3 demo UI with five interpretability and control surfaces:

1. Cross-attention heatmap
2. What-if adaptation comparator
3. Persona override chip row
4. Floating reset-session pill
5. Self-running WCAG 2.2 audit

The panels live in a right-side drawer and do not touch any of the original UI modules.

## Enabling / disabling

The drawer is enabled by default. To turn it off without removing any files, open the browser devtools console and run:

```js
localStorage.setItem('i3AdvancedUI', '0');
location.reload();
```

To re-enable:

```js
localStorage.setItem('i3AdvancedUI', '1');
location.reload();
```

The helpers `window.i3Advanced.enable()` and `window.i3Advanced.disable()` are also exposed once the module has loaded.

## Keyboard shortcut

Press `Alt+A` anywhere in the page to toggle the drawer open and closed. The shortcut is registered only when the advanced UI is enabled.

## Query-string flags

- `?a11y=1` activates the WCAG 2.2 audit. Results are printed to the devtools console as a pass/fail table plus an overall `AA` and `AAA` percentage score, and are also stashed on `window.__i3_a11y_report`.

## Admin token

The reset-session pill and the persona chips POST to admin-scoped endpoints. If present, the token from `localStorage.i3AdminToken` is sent as an `Authorization: Bearer ...` header. Set it with:

```js
localStorage.setItem('i3AdminToken', '<your-token-here>');
```

## Wire-protocol assumptions

The advanced modules assume these backend endpoints. Each endpoint is optional; the frontend degrades gracefully if it 404s or times out.

| Endpoint                                  | Method | Purpose                                                      |
|-------------------------------------------|--------|--------------------------------------------------------------|
| `GET /api/attention?session_id=<id>`      | GET    | Returns `{ "attention": number[4][4] }` in [0, 1].           |
| `POST /whatif/compare`                    | POST   | Body `{ prompt, alternatives: [{id, label, adaptation}] }`. Returns `{ responses: [{ label, text, route?, latency_ms? }] }`. |
| `POST /admin/persona/{name}`              | POST   | Biases the user profile toward a named archetype.            |
| `POST /admin/reset`                       | POST   | Clears working memory and persona bias for the session.      |

The attention viz module additionally hooks the existing WebSocket client's `response` event; no new socket messages are required.

## Files

- `css/advanced.css` -- styling, palette-matched with `style.css`
- `js/advanced_init.js` -- entry point, imports everything
- `js/attention_viz.js` -- heatmap (ES module)
- `js/whatif.js` -- comparator (ES module)
- `js/persona_switcher.js` -- chip row (ES module)
- `js/reset_button.js` -- floating pill (ES module)
- `js/wcag_audit.js` -- audit script (plain IIFE, self-running)
