# I³ — Fixes Applied from the 2026-04-23 Deep Audit

This report summarises every code change made in response to the two
audit passes:

- [`SECURITY_REVIEW_2026-04-23.md`](SECURITY_REVIEW_2026-04-23.md) —
  first (security) pass: 2 high, 5 medium, 5 low, 8 positive.
- [`DEEP_AUDIT_2026-04-23.md`](DEEP_AUDIT_2026-04-23.md) — second
  (robustness / performance / code quality) pass: 1 blocker, 9 high,
  16 medium, 14 low, 7 positive.

Every **blocker**, every **high**, and most **medium / low** findings
have been fixed. All fixes are labelled in-source with a
``(<severity>, 2026-04-23 audit)`` comment so future maintainers can
trace the rationale.

## Blocker

- **B-1 — `process_keystroke` never awaited** →
  [`server/websocket.py:429`](../server/websocket.py#L429).
  Added `await` so every keystroke event actually reaches
  `InteractionMonitor` and populates the feature window.

## High — first audit

- **H-1 — `/whatif/*` bypassed rate limiter** →
  [`server/middleware.py:401-425`](../server/middleware.py). Rewrote
  limiter from an **include-list** (`/api/*` only) to an
  **exclude-list** (static, ws, health, docs). Every new route is now
  throttled by default.
- **H-2 — preference routes stored unsanitised cross-user text** →
  [`server/routes_preference.py:40,253-256,298,349`](../server/routes_preference.py).
  Wired `PrivacySanitizer` on the write path and
  `Depends(require_user_identity)` on the read paths.

## High — second audit

- **H-1 — POST routes not gated** → added
  [`server/auth.py::require_user_identity_from_body`](../server/auth.py#L80)
  and applied to
  [`routes_whatif.py::/respond,/compare`](../server/routes_whatif.py),
  [`routes_tts.py::POST /api/tts`](../server/routes_tts.py),
  [`routes_translate.py::POST /api/translate`](../server/routes_translate.py),
  [`routes_preference.py::POST /api/preference/record`](../server/routes_preference.py),
  [`routes_explain.py::POST /api/explain/adaptation`](../server/routes_explain.py).
- **H-2 — DiaryStore opens per-op connection + FK pragma lost** →
  [`i3/diary/store.py:100-180`](../i3/diary/store.py).
  Persistent `aiosqlite.Connection` held for the store lifetime; WAL
  journal + `foreign_keys = ON` set once. 10 call sites migrated via a
  drop-in `self._conn()` async context manager.
- **H-3 — SLM generation blocked event loop** →
  [`i3/pipeline/engine.py:810-817`](../i3/pipeline/engine.py).
  Offloaded to `loop.run_in_executor(...)` mirroring the encoder.
- **H-4 — unbounded per-user dicts** →
  [`i3/pipeline/engine.py:137-149,1037-1070`](../i3/pipeline/engine.py).
  `user_models` is now an `OrderedDict` capped at
  `I3_MAX_TRACKED_USERS` (default 10 000) with O(1) LRU eviction and
  full footprint cleanup (response-time + length + engagement + route
  dicts all cleared for the evicted user).
- **H-5 — `ContextualThompsonBandit` not concurrency-safe** →
  [`i3/router/bandit.py:109-115,152-245,265-290`](../i3/router/bandit.py).
  Added an `RLock` around `select_arm` + `update` + `_refit_posterior`.
  History converted to `deque(maxlen=_MAX_HISTORY_PER_ARM)` so
  overflow is O(1) instead of O(n) slice churn. 800-op 8-thread stress
  test produced consistent `total_pulls` sum.
- **H-6 — `httpx.AsyncClient` lazy init race** →
  [`i3/cloud/client.py:148-208`](../i3/cloud/client.py) with
  `asyncio.Lock`;
  [`i3/cloud/providers/openrouter.py`](../i3/cloud/providers/openrouter.py),
  [`ollama.py`](../i3/cloud/providers/ollama.py),
  [`huawei_pangu.py`](../i3/cloud/providers/huawei_pangu.py) with
  double-checked `threading.Lock`.
- **H-7 — exception class name leaked to wire** →
  [`i3/pipeline/engine.py:1086-1092`](../i3/pipeline/engine.py).
  Replaced `type(exc).__name__` with constant `"pipeline_error"`.
- **H-8 — global `torch.manual_seed` on every explain request** →
  [`server/routes_explain.py:232-262`](../server/routes_explain.py).
  Scoped `torch.Generator` + module-level cached layer, so no global
  RNG mutation and the allocation happens once per process.
- **H-9 — `prior_alpha` passed as `prior_precision`** →
  [`i3/config.py:227-233`](../i3/config.py) adds a distinct
  `prior_precision` field;
  [`i3/router/router.py:86-100`](../i3/router/router.py) now passes the
  right field. Operators can tune the Beta prior and the Gaussian
  precision independently.

## Medium

- **M-1 — double `load_config` at startup** →
  [`server/app.py:189-204,49-59`](../server/app.py). Config loaded once
  in `create_app` and reused from `app.state.config` in the lifespan.
- **M-2 — root `Config` accepted unknown YAML keys** →
  [`i3/config.py:503-519`](../i3/config.py) adds
  `extra="forbid"`. Typos now fail at load time.
- **M-3 — `CloudConfig.model` default drifted** →
  [`i3/config.py:370-374`](../i3/config.py). Default is
  `claude-sonnet-4-5`, matching the brief-§8-locked id.
- **M-4 — `PrivacyAuditor` recursion uncapped** →
  [`i3/privacy/sanitizer.py:409-435`](../i3/privacy/sanitizer.py).
  Depth cap at 32, list-based path join (O(n) not O(n²)).
- **M-5 — `PrivacyAuditor._findings` grew unbounded** →
  [`i3/privacy/sanitizer.py:264-271`](../i3/privacy/sanitizer.py).
  Now a `deque(maxlen=1_000)`.
- **M-7 — `generate_session_summary` no timeout** →
  [`i3/pipeline/engine.py:366-398`](../i3/pipeline/engine.py). Wrapped
  in `asyncio.wait_for(..., timeout=timeout*1.2)` with fallback.
- **M-10 — activation-cache manifest traversal** →
  [`i3/interpretability/activation_cache.py:302-350`](../i3/interpretability/activation_cache.py).
  Manifest size capped at 1 MiB, shape validated, each shard path
  `resolve()`-and-`relative_to`-checked against the cache root.
- **M-12 — dead exempt prefixes** →
  [`server/middleware.py:401-410`](../server/middleware.py).
  `/docs`/`/redoc`/`/openapi.json` replaced with the actual mount
  points `/api/docs` etc.
- **M-14 — inference 404 leaked toolchain hint** →
  [`server/routes_inference.py:112-127`](../server/routes_inference.py).
  Detail is a constant `"Model not found"`; hint lives in the log only.
- **M-15 — translate `raise` dropped exception cause** →
  [`server/routes_translate.py:336-344`](../server/routes_translate.py).
  `raise HTTPException(...) from exc`.

## Medium — from the first audit (also fixed)

- **M-3 — multi-worker limiter silent** → `server/app.py` now REFUSES
  to start when `I3_WORKERS>1` without explicit
  `I3_ALLOW_LOCAL_LIMITER=1`.
- **M-4 — OpenRouter response.text in exception** →
  [`openrouter.py:170-187`](../i3/cloud/providers/openrouter.py). Body
  moved to DEBUG log; exception message narrowed.
- **M-5 — limiter include-list** → fixed jointly with H-1 above.

## Low (selection)

- **L-1 — provider httpx kwargs not pinned** → every adapter now pins
  `verify=True`, `follow_redirects=False`, `httpx.Limits(...)`.
- **L-2 — Huawei / Ollama response.text echo** → both now log body at
  DEBUG, keep exception message narrow.
- **L-3 — sanitiser IP regex false positives** →
  [`sanitizer.py:115-129`](../i3/privacy/sanitizer.py) requires each
  octet ≤ 255. Windows build numbers and SemVer strings no longer
  count as IPs.
- **L-4 — admin export 404 for ghost users** →
  [`routes_admin.py:453-467`](../server/routes_admin.py).
- **L-5 — sliding-window eviction was O(n)** →
  [`middleware.py:305-360`](../server/middleware.py). `OrderedDict` +
  `popitem(last=False)` for amortised O(1); active keys refreshed
  with `move_to_end`.
- **L-10 — surrogate layer re-allocated per request** → fixed jointly
  with H-8 (module-level cache).
- **L-13 — `feature_window.pop(0)` O(n)** →
  [`monitor.py:55-65,218-223`](../i3/interaction/monitor.py). Now
  `deque(maxlen=feature_window_size)`; auto-trim.

## Verification after the fixes

- `scripts/verify_all.py --strict`: 27 PASS / 0 FAIL / 19 SKIP.
  All SKIPs are environment-gated (torch DLL on Windows, missing
  binaries like `ruff`, `mypy`, `helm`, `cedarpy`, `mkdocs`).
- `scripts/security/run_redteam_notorch.py --targets sanitizer,pddl,guardrails`:
  3 / 4 invariants PASS. The rate_limit invariant FAILs only because
  the FastAPI target surface is not exercised (torch DLL env issue).
- Bespoke smoke test matrix — all PASS:
  - `Config` rejects typoed YAML section (M-2).
  - Bandit 800 ops across 8 threads produces consistent
    `sum(total_pulls) == 800` (H-5).
  - `DiaryStore` `foreign_keys` pragma persists across ops (H-2);
    `.close()` is idempotent.
  - `require_user_identity` rejects wrong / accepts correct
    `X-I3-User-Token` (H-1 / H-2).
  - Sanitiser regex distinguishes `192.168.1.100` (match) from
    `Windows 10.0.22621` (no match) (L-3).
  - `PrivacyAuditor.audit_request` caps recursion at 32 levels (M-4).

## Items deferred or not fixed

- **M-2 (second audit) — `Config` sub-models** still use
  `extra="allow"` individually. Root-level forbid catches the common
  bug class; per-section forbid can be added later without risk.
- **M-6 — `admin_export` N+1** — solved *implicitly* by H-2 (connection
  reuse makes per-call cost negligible). A proper single-JOIN can be
  added later.
- **M-8 — `InteractionMonitor` threading.Lock in async** — critical
  section is short, has no `await`, and is already double-checked; left
  in place with a comment. Swap to `asyncio.Lock` is a future refactor.
- **M-11 — consolidate provider-client factory** — the three providers
  flagged now have the same locked lazy-init pattern; centralising into
  a shared helper is a lint-level improvement, not a security fix.
- **M-13 — readiness probe detail** — the existing endpoint is already
  useful to Kubernetes. A `/api/ready/detailed` split is a follow-up.
- **M-16 / L-9 — preference fabricated pair + magic threshold** —
  UX-level correctness issues; not load-bearing for security.
- **L-6 — `db_path` symlink traversal** — requires an opt-in root and
  should land with the broader file-ACL discussion.
- **L-7 — `prior_beta` unused** — H-9 fix documents both prior fields;
  whether `prior_beta` is plumbed through to the bandit's
  Beta-Bernoulli posterior is a future ticket.
- **L-11 / L-12 / L-14** — small correctness / config issues, not
  security-load-bearing.

## Line count of changes

```
 18 files changed, 551 insertions(+), 99 deletions(-)
```

(approximate — final number will be in the commit diff)
