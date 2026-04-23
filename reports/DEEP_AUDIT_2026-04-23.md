# I³ Deep Audit (Robustness / Performance / Code Quality) — 2026-04-23

Auditor: Claude Opus 4.7 (1M context), manual static review on D:\implicit-interaction-intelligence.
Scope: production code under `server/`, `i3/pipeline/`, `i3/privacy/`, `i3/user_model/`,
`i3/diary/`, `i3/router/`, `i3/cloud/`, `i3/interaction/`, `i3/adaptation/`,
`i3/interpretability/`, `i3/safety/`, `i3/tts/`, `i3/config.py`. Tests skimmed
only to note coverage gaps; not audited for their own robustness.

This is the **second** pass; HIGH/MEDIUM/LOW findings from
`reports/SECURITY_REVIEW_2026-04-23.md` are not re-flagged.

## Executive summary

Counts by severity (new findings only):

- **Blocker**: 1
- **High**: 9
- **Medium**: 16
- **Low**: 14
- **Info / positive**: 7

Top five concerns:

1. **B-1 — `process_keystroke` coroutine is called without `await` in the
   WebSocket handler.** The call in `server/websocket.py:429` discards
   the coroutine object, so *every keystroke event is silently dropped*.
   Keystroke dynamics is the product's core implicit signal; the entire
   TCN-encoded behavioural baseline is fed the all-zeros fallback.
   Python emits a `RuntimeWarning: coroutine ... was never awaited`
   that operators are not watching.
2. **H-1 — Several user-scoped routes bypass the `require_user_identity`
   gate the server itself already defined.** `POST /whatif/respond`,
   `POST /whatif/compare`, `POST /api/tts`, `POST /api/translate`,
   `POST /api/preference/record`, and `POST /api/explain/adaptation`
   all accept a `user_id` in the body or path but do NOT depend on
   `require_user_identity`. Since `POST /api/explain/adaptation`
   writes to the `_CACHE` that `GET /api/explain/last-decision/{user_id}`
   reads, an unauthenticated caller can poison a victim's cached
   explanation payload.
3. **H-2 — `DiaryStore` opens and closes an `aiosqlite` connection on
   every single operation.** Ten call sites (`initialize`,
   `create_session`, `end_session`, `log_exchange`, two queries, etc.)
   all do `async with aiosqlite.connect(self.db_path)` per call. On the
   hot response path this is ~10–40 ms per message of pure connection
   overhead. Worse, the `PRAGMA foreign_keys = ON` is per-connection,
   so every subsequent op runs with FK enforcement **off** — the
   declared FK on `exchanges(session_id)` is not actually enforced.
4. **H-3 — Heavy CPU/GPU work runs on the asyncio event loop.**
   The encoder `_encode_features` is properly offloaded via
   `run_in_executor`, but `_slm_generator.generate(...)` in the same
   `_generate_response` method is NOT. The what-if routes
   (`whatif_respond` / `whatif_compare`) and the TTS route
   (`TTSEngine.speak`) all call synchronous model/synth code directly
   inside `async def`, blocking the event loop for other users. In
   `whatif_compare` variants run sequentially rather than in parallel,
   multiplying the latency hit.
5. **H-4 — Unbounded in-memory dicts on the long-running server.**
   `Pipeline.user_models`, `Pipeline._last_response_time`,
   `Pipeline._last_response_length`, `Pipeline._previous_engagement`,
   `Pipeline._previous_route`, `InteractionMonitor._sessions`, and
   `ContextualThompsonBandit.history` are all keyed by user_id or arm
   with no eviction policy. A long-lived server (or one receiving
   user-id rotation, intentional or otherwise) grows these
   monotonically until OOM.

Overall verdict: the server has strong **perimeter** controls
(the first-pass audit covered those), but its **internals** have a
systemic pattern of sync-in-async, unbounded state, and per-call
database connection churn. The keystroke bug is the one that should
be fixed before the next demo; everything else is recoverable.

## Methodology

- Read every server route file end-to-end (`server/*.py`).
- Read `i3/pipeline/engine.py`, `i3/privacy/sanitizer.py`,
  `i3/privacy/encryption.py`, `i3/user_model/store.py`,
  `i3/diary/store.py`, `i3/interaction/monitor.py`,
  `i3/router/bandit.py`, `i3/router/router.py`,
  `i3/router/preference_learning.py` (first 280 lines),
  `i3/cloud/client.py`, `i3/cloud/providers/ollama.py`,
  `i3/interpretability/activation_cache.py`, `i3/config.py`,
  `i3/adaptation/uncertainty.py` (header), `i3/tts/engine.py`
  (header).
- Cross-checked grep scans for: `async def` without `await`,
  `threading.Lock` inside async code, bare `except Exception`,
  unbounded dicts, `run_in_executor` coverage, sync DB connects,
  double config loads.
- Skimmed `tests/` directory listing to identify coverage gaps.

What I did NOT audit for robustness issues (per brief): `tests/`,
`demo/`, `scripts/*.py` beyond `scripts/verification/`, training
scripts, edge-export modules, MCP server, federated/authz modules,
observability beyond `instrumentation.py`, notebooks.

## Findings

---

### BLOCKER

#### B-1 — `process_keystroke` coroutine is not awaited in the WebSocket handler

- **Severity**: blocker
- **Category**: correctness
- **File and line range**: `server/websocket.py:429`; definition at
  `i3/interaction/monitor.py:135-150`
- **Quote**:
  ```python
  # server/websocket.py:429
  pipeline.monitor.process_keystroke(user_id, ks)
  ```
  ```python
  # i3/interaction/monitor.py:135-150
  async def process_keystroke(
      self, user_id: str, event: KeystrokeEvent
  ) -> None:
      ...
      session = self._get_or_create_session(user_id)
      async with session.lock:
          session.keystroke_buffer.append(event)
  ```
- **Failure scenario**: `process_keystroke` is `async def`, so the call
  at line 429 returns a coroutine object that is immediately discarded
  without ever being scheduled. Every keystroke event from every
  WebSocket client is silently dropped on the floor. The per-session
  `keystroke_buffer` in `_UserSession` therefore stays empty, meaning
  `_compute_keystroke_metrics` always takes the zero-metrics branch
  (`monitor.py:270-282`), meaning `mean_iki_ms`, `std_iki_ms`,
  `mean_burst_length`, `backspace_ratio`, and `composition_speed_cps`
  are always 0.0. The TCN encoder is fed a near-constant feature
  vector; the whole behavioural-baseline story collapses to "all users
  look identical". Python emits a `RuntimeWarning: coroutine ... was
  never awaited` but nothing in the deployment surfaces it.
- **Suggested fix**: `await pipeline.monitor.process_keystroke(user_id, ks)`
  (or queue to a bounded background worker if you want to keep the
  receive loop cheap). Add a regression test that exercises
  `/ws/{user_id}` with a sequence of keystroke frames and asserts the
  resulting feature window is non-zero.

---

### HIGH

#### H-1 — Cross-user-writable routes not gated by `require_user_identity`

- **Severity**: high
- **Category**: correctness / api
- **File and line range**:
  `server/routes_whatif.py:310, 358` (both POST routes);
  `server/routes_tts.py:366` (`POST /api/tts`);
  `server/routes_translate.py:309` (`POST /api/translate`);
  `server/routes_preference.py:227` (`POST /api/preference/record`);
  `server/routes_explain.py:355` (`POST /api/explain/adaptation`)
- **Quote** (one example):
  ```python
  # server/routes_explain.py:355-356
  @router.post("/adaptation")
  async def explain_adaptation(request: Request) -> JSONResponse:
  ```
  No `dependencies=[Depends(require_user_identity)]`. The body's
  `user_id` then feeds `_CACHE.set(body.user_id, payload)` at
  line 417 — which the `require_user_identity`-gated
  `GET /last-decision/{user_id}` reads back at line 434.
- **Failure scenario**: The first-pass audit introduced
  `server/auth.py::require_user_identity` and wired it on the three
  GET routes in `server/routes.py`, both GET routes in
  `server/routes_preference.py`, and the GET route in
  `server/routes_explain.py`. **Every POST route that mutates
  per-user state was missed.** Concretely:
  - `POST /api/explain/adaptation` writes an `ExplainResponse` to the
    per-user cache; an attacker can overwrite any user's cached
    payload. The `I3_REQUIRE_USER_AUTH=1` gate on the GET then reads
    attacker-controlled data back.
  - `POST /api/preference/record` appends a `PreferencePair` to
    `_CACHE.get_or_create(body.user_id)`; an attacker can pollute any
    user's preference dataset.
  - `POST /whatif/*` and `POST /api/tts` run generation under any
    `user_id`, burning compute and filling logs tagged with the
    victim's id.
  The fix the first audit applied is therefore incomplete and the
  auth invariant is one-sided (read-only).
- **Suggested fix**: Add
  `dependencies=[Depends(require_user_identity)]` to each POST route
  whose payload carries a `user_id`. For routes that read `user_id`
  from the request body, either (a) move it to a path parameter and
  validate via the dependency, or (b) extend `require_user_identity`
  to accept a body-resolved user id.

#### H-2 — `DiaryStore` opens a fresh `aiosqlite` connection on every call

- **Severity**: high
- **Category**: perf / correctness
- **File and line range**: `i3/diary/store.py:110, 191, 239, 326,
  374, 406, 441, 476, 559, 624` (ten call sites)
- **Quote**:
  ```python
  # i3/diary/store.py:110
  async with aiosqlite.connect(self.db_path) as db:
      await db.execute("PRAGMA foreign_keys = ON")
      ...
  ```
- **Failure scenario**: Every `create_session`, `log_exchange`,
  `end_session`, `get_session`, `get_user_sessions`,
  `get_session_exchanges`, `prune_old_entries` (etc.) call opens a
  brand-new SQLite connection, runs schema negotiation, and closes it.
  On the hot response path (`_log_exchange_safe` →
  `diary_store.log_exchange`) this is ~5–30 ms of pure DB overhead
  per message. The `admin_export` endpoint is worse: for every
  session it fetches, it calls `get_session_exchanges` in a loop —
  N+1 connections per export.
  Correctness consequence: `PRAGMA foreign_keys = ON` is set only in
  `initialize()` (line 114). PRAGMA statements apply to the current
  connection only. **All subsequent operations run with FK
  enforcement OFF**, so the declared
  `FOREIGN KEY (session_id) REFERENCES sessions(session_id)` on the
  `exchanges` table is decorative — orphan exchanges can be written,
  and `delete_session` won't cascade (if it existed).
- **Suggested fix**: Hold a persistent `aiosqlite.Connection` in
  `DiaryStore._db` opened in `initialize()`, close it in
  `close()` (mirror the pattern in `UserModelStore`). Set `PRAGMA
  foreign_keys = ON` (and `PRAGMA journal_mode = WAL`) once at open
  time. Re-use the connection across calls; wrap each method in its
  own transaction if the current per-op commit boundary matters.

#### H-3 — Blocking CPU/GPU work on the event loop in multiple async routes

- **Severity**: high
- **Category**: perf / concurrency
- **File and line range**:
  `i3/pipeline/engine.py:800-805` (SLM generation);
  `server/routes_whatif.py:285-289, 382-389`;
  `server/routes_tts.py:313`;
  `server/routes_explain.py:387, 400-404`
- **Quote** (one representative):
  ```python
  # i3/pipeline/engine.py:798-805
  if self._slm_generator is not None:
      try:
          return self._slm_generator.generate(
              prompt=message,
              adaptation_vector=adaptation.to_tensor().unsqueeze(0),
              user_state=user_state.unsqueeze(0),
          )
  ```
  Note that two lines earlier the encoder IS wrapped in
  `loop.run_in_executor` (line 478-481), so the pattern is known —
  the SLM / TTS / MC-Dropout paths were just not ported.
- **Failure scenario**: `_slm_generator.generate` is a synchronous
  PyTorch generation loop. Under any concurrency (second user, second
  WS message, a `/whatif/compare` batch) it blocks the single event
  loop thread for the entire duration. A 500 ms SLM call means every
  other active request on the process is stalled for 500 ms. The
  `whatif_compare` route iterates `adaptation_variants` sequentially
  (`routes_whatif.py:379`), so a 4-variant request takes 4× longer
  than it would if variants ran in a thread pool. MC-Dropout in
  `routes_explain.py` runs 30 samples synchronously — multiply by the
  encoder size and it's easily several seconds. TTS synthesis
  (`TTSEngine.speak`) is similarly synchronous CPU.
- **Suggested fix**: Replace each direct synchronous call with
  `await asyncio.get_running_loop().run_in_executor(None, fn, args)`,
  mirroring the encoder pattern at `engine.py:478-481`. For
  `whatif_compare`, wrap the per-variant block in `asyncio.gather`
  over the thread-pool futures. For the MC-Dropout hot loop in
  `routes_explain`, make the whole `explain_adaptation` inner block
  a single `run_in_executor` call.

#### H-4 — Unbounded per-user / per-arm in-memory state across long-running server

- **Severity**: high
- **Category**: resource / perf
- **File and line range**:
  `i3/pipeline/engine.py:137` (`user_models`),
  `:195-198` (four engagement dicts),
  `i3/interaction/monitor.py:98` (`_sessions`),
  `i3/router/bandit.py:110-112` (`history` — per-arm is bounded but
  `len(history)` scans are O(n)).
- **Quote**:
  ```python
  # i3/pipeline/engine.py:137, 195-198
  self.user_models: dict[str, Any] = {}
  ...
  self._last_response_time: dict[str, float] = {}
  self._last_response_length: dict[str, int] = {}
  self._previous_engagement: dict[str, float] = {}
  self._previous_route: dict[str, int] = {}
  ```
  ```python
  # i3/interaction/monitor.py:98
  self._sessions: dict[str, _UserSession] = {}
  ```
- **Failure scenario**: Only `admin_delete_user` and `admin_reset`
  remove entries (and only for `user_id == "demo_user"`). Every other
  `user_id` ever seen stays resident. Each `_UserSession` holds a
  feature-window list and a `BaselineTracker`; each `UserModel` holds
  three EMAs. In a multi-tenant deployment, or a single-user
  deployment where the client rotates ids during testing, these maps
  grow without bound. Combined with the `websocket.py` handler
  creating a session on any hit to `/ws/{id}`, this is a linear-in-
  request-history memory leak. On the bandit, `history[arm]` is
  already capped at `_MAX_HISTORY_PER_ARM = 10_000` but the slice-and-
  replace pattern `self.history[arm] = self.history[arm][-N:]` reallocates
  the whole list on every overflow (O(n) churn).
- **Suggested fix**: Wrap each of these dicts in an
  `OrderedDict` + LRU cap (mirror `_SlidingWindowLimiter` and
  `_UserCache`). `max_users = 10_000` is a reasonable default; evict
  LRU when full. For bandit history use
  `collections.deque(maxlen=_MAX_HISTORY_PER_ARM)` so overflow is O(1).

#### H-5 — `ContextualThompsonBandit` is not concurrency-safe

- **Severity**: high
- **Category**: concurrency
- **File and line range**: `i3/router/bandit.py:154-218`
  (`select_arm`), `:224-283` (`update`), `:289-405`
  (`_refit_posterior`); caller `i3/pipeline/engine.py:678,
  734-745`.
- **Quote**:
  ```python
  # i3/router/bandit.py:272-283
  self.history[arm].append((context.copy(), reward_f))
  if len(self.history[arm]) > _MAX_HISTORY_PER_ARM:
      self.history[arm] = self.history[arm][-_MAX_HISTORY_PER_ARM:]
  self.total_pulls[arm] += 1
  self.total_rewards[arm] += reward_f
  if self.total_pulls[arm] % self.refit_interval == 0:
      self._refit_posterior(arm)
  ```
- **Failure scenario**: `ContextualThompsonBandit` is a single shared
  instance on `pipeline.router`, called concurrently by every in-
  flight `process_message` (for `select_arm`) and every
  `compute_engagement` (for `update`). Individual list/dict ops are
  atomic under the GIL, but the multi-step pattern above is not:
  between `history[arm].append` and `total_pulls[arm] += 1`, another
  coroutine can run `select_arm` against inconsistent state; between
  `self.total_pulls[arm] % self.refit_interval == 0` and
  `_refit_posterior(arm)`, another coroutine can mutate
  `self.weight_means[arm]` under the feet of the Newton loop.
  `select_arm` can also read `weight_means[arm]` and
  `weight_covs[arm]` mid-write. There is no lock anywhere in the
  module. This is exactly the lost-update / mid-write-read race the
  engine's `_user_models_lock` guards against elsewhere.
- **Suggested fix**: Add `self._lock = asyncio.Lock()` (or
  `threading.Lock()` since the surface is already called from both
  asyncio and a potential bandit-update thread) and wrap both
  `select_arm` and `update` in `with self._lock:`. Refit can stay
  inside `update`; it is already short (`_NEWTON_ITERS=8`).

#### H-6 — `httpx.AsyncClient` lazy init lacks a lock — concurrent first-hit leaks clients

- **Severity**: high
- **Category**: resource / concurrency
- **File and line range**:
  `i3/cloud/client.py:169-208` (`_ensure_client`);
  `i3/cloud/providers/ollama.py:75-92`; same pattern exists in
  OpenRouter / Huawei / others per the first audit.
- **Quote**:
  ```python
  # i3/cloud/client.py:178-187
  if self._client is None:
      timeout = httpx.Timeout(...)
      self._client = httpx.AsyncClient(...)
  ```
- **Failure scenario**: Two concurrent `generate()` calls on a fresh
  pipeline each pass the `if self._client is None` check before either
  assigns. Both construct an `httpx.AsyncClient` (each opens a TLS
  cert store, sets up a connection pool, etc.). One is stored in
  `self._client`; the other is orphaned — its connection pool is
  never closed and its sockets leak until the interpreter tears them
  down. On a server that cold-starts under load, this leaks a handful
  of connections on first warm-up.
- **Suggested fix**: Either instantiate the client eagerly in
  `__init__` (no lazy path needed), or guard the lazy path with
  `asyncio.Lock`:
  ```python
  async with self._init_lock:
      if self._client is None:
          self._client = httpx.AsyncClient(...)
  ```

#### H-7 — `PipelineOutput.adaptation` leaks Python exception class name to clients

- **Severity**: high
- **Category**: api / correctness
- **File and line range**: `i3/pipeline/engine.py:1076-1092`
  (`_build_error_output`)
- **Quote**:
  ```python
  # i3/pipeline/engine.py:1085
  "error": type(exc).__name__,
  ```
- **Failure scenario**: On any exception inside `process_message`,
  the handler constructs a degraded `PipelineOutput` whose
  `adaptation` field has an extra `"error"` key set to the exception
  class name (`"ValueError"`, `"KeyError"`, `"TorchRuntimeError"`,
  …). This ends up in the WebSocket `state_update` frame and in any
  REST response that serialises the pipeline output. The server's
  other exception handlers (`server/app.py:274-283`) are careful to
  never leak exception metadata to the wire; this one-off violates
  the invariant.
- **Suggested fix**: Replace with a constant string (e.g.
  `"pipeline_error"`) or omit the field entirely. Keep the Python
  class name in the structured log only.

#### H-8 — `torch.manual_seed(0xA11CE)` mutates global RNG on every explain request

- **Severity**: high
- **Category**: correctness / concurrency
- **File and line range**: `server/routes_explain.py:240-246`
  (`_surrogate_mapping_fn`); called from `_build_payload` and
  `explain_adaptation` per request.
- **Quote**:
  ```python
  # server/routes_explain.py:242-246
  torch.manual_seed(0xA11CE)
  layer = nn.Linear(32, 8, bias=False)
  for p in layer.parameters():
      p.requires_grad_(True)
  return layer
  ```
- **Failure scenario**: Every call to `explain_adaptation` (and a
  second call from `_build_payload`) resets PyTorch's global RNG
  seed. Other coroutines running in parallel — the pipeline's SLM
  generator using `torch.multinomial`, the MC-Dropout estimator, the
  router bandit's `torch`-backed paths — will observe a suddenly
  deterministic, identical random stream because the global seed was
  just reset. Under concurrency this produces repeated samples
  across unrelated requests and silently breaks Thompson sampling's
  exploration.
- **Suggested fix**: Use a seeded `torch.Generator` scoped to the
  layer construction:
  ```python
  gen = torch.Generator().manual_seed(0xA11CE)
  layer = nn.Linear(32, 8, bias=False)
  with torch.no_grad():
      for p in layer.parameters():
          p.copy_(torch.randn(p.shape, generator=gen))
  ```
  Or cache the layer module-level since it is deterministic anyway.

#### H-9 — `IntelligentRouter` passes `prior_alpha` as `prior_precision` — semantic mismatch

- **Severity**: high
- **Category**: correctness
- **File and line range**: `i3/router/router.py:86-93`
- **Quote**:
  ```python
  # i3/router/router.py:88-93
  self.bandit = ContextualThompsonBandit(
      n_arms=2,
      context_dim=config.router.context_dim,
      prior_precision=config.router.prior_alpha,
      exploration_bonus=config.router.exploration_bonus,
  )
  ```
- **Failure scenario**: `config.router.prior_alpha` is the **Beta**
  posterior's α prior (used by the Beta-Bernoulli cold-start path
  inside the bandit). `prior_precision` in the Laplace-logistic path
  is the inverse variance of the Gaussian weight prior. These are
  unrelated quantities: Beta's `alpha=1.0` default works as a
  uniform prior; a Gaussian precision of `1.0` shrinks weights with
  standard deviation 1. The ops operator tuning `prior_alpha` to
  strengthen the Beta prior simultaneously (and unexpectedly)
  strengthens the Gaussian shrinkage on the logistic-regression
  weights — producing coupled misbehaviour that a practitioner
  cannot diagnose without reading the router.py constructor.
  `config.router.prior_beta` is silently unused.
- **Suggested fix**: Add a distinct `prior_precision` field to
  `RouterConfig` (default 1.0, `gt=0.0`), pass it explicitly, and
  either consume `prior_beta` or remove it from the schema with a
  deprecation warning.

---

### MEDIUM

#### M-1 — `load_config` is called twice at server startup, mutating global RNG seeds both times

- **Severity**: medium
- **Category**: config / perf
- **File and line range**: `server/app.py:52, 194`;
  `i3/config.py:684-685`
- **Quote**:
  ```python
  # server/app.py:194 (inside create_app)
  config = load_config("configs/default.yaml")
  # server/app.py:52 (inside lifespan, runs AFTER)
  config = load_config("configs/default.yaml")
  ```
  ```python
  # i3/config.py:684-685
  if set_seeds:
      _set_seeds(config.project.seed)
  ```
- **Failure scenario**: Both calls default to `set_seeds=True`, so
  both invoke `_set_seeds` which mutates `os.environ["PYTHONHASHSEED"]`,
  `random`, `np.random`, and `torch.manual_seed`. The lifespan's
  second call also fires AFTER the middleware stack has already
  done any RNG-sensitive setup. File I/O is wasted too — the config
  file is read, parsed, and validated twice on every server start.
- **Suggested fix**: Load the config once in `create_app` and stash
  it on `app.state` for the lifespan to reuse. Pass `set_seeds=False`
  from the server path entirely (the server is not trying to make
  its RNG reproducible with a training run) or at least from the
  second call.

#### M-2 — `Config` Pydantic model accepts unknown top-level YAML keys silently

- **Severity**: medium
- **Category**: config / correctness
- **File and line range**: `i3/config.py:497-517`
- **Quote**:
  ```python
  # i3/config.py:497-504
  class Config(BaseModel):
      model_config = ConfigDict(frozen=True)
  ```
  No `extra="forbid"`.
- **Failure scenario**: A YAML typo (e.g. `saftey:` instead of
  `safety:`, or `priavcy:` instead of `privacy:`) is silently
  accepted — the real section keeps its defaults, the typoed section
  is dropped. The operator believes they configured something they
  did not. Every sub-section model is `frozen=True` but none is
  `extra="forbid"`, so typos inside a section have the same bug.
- **Suggested fix**: Add `ConfigDict(frozen=True, extra="forbid")`
  to `Config` and to every sub-model. Provide a migration note for
  any vendor-specific extras if present.

#### M-3 — `CloudConfig.model` default in code drifts from `configs/default.yaml`

- **Severity**: medium
- **Category**: config / docs
- **File and line range**: `i3/config.py:364`; `configs/default.yaml:89`
- **Quote**:
  ```python
  # i3/config.py:364
  model: str = Field(default="claude-sonnet-4-20250514", min_length=1)
  ```
  ```yaml
  # configs/default.yaml:89
  model: "claude-sonnet-4-5"            # brief §8: locked model id
  ```
- **Failure scenario**: Anything that instantiates `Config()` without
  loading the YAML (tests, isolated import, CI smoke) gets the old
  snapshot-dated id `claude-sonnet-4-20250514`. The brief's
  §8 pin and the CHANGELOG's `claude-sonnet-4-5` entry are bypassed
  silently. Bug class: config drift between two sources of truth.
- **Suggested fix**: Change the code default to `claude-sonnet-4-5`,
  or (better) make the field required (`...`) so the YAML is the
  sole source of truth.

#### M-4 — `PrivacyAuditor.audit_request._scan_value` recurses without depth cap

- **Severity**: medium
- **Category**: correctness / resource
- **File and line range**: `i3/privacy/sanitizer.py:409-422`
- **Quote**:
  ```python
  # i3/privacy/sanitizer.py:409-422
  def _scan_value(value, field_path: str) -> None:
      if isinstance(value, str):
          ...
      elif isinstance(value, dict):
          for k, v in value.items():
              _scan_value(v, f"{field_path}.{k}")
      elif isinstance(value, list):
          for i, item in enumerate(value):
              _scan_value(item, f"{field_path}[{i}]")
  ```
- **Failure scenario**: A deeply nested dict/list (adversarially
  crafted, or accidentally produced by a misbehaving cloud payload)
  recurses until `RecursionError`. The field-path string is also
  built with `{field_path}.{k}` which is O(n²) in depth because each
  level appends to the accumulator. An attacker with any toehold on
  the cloud-request shape (e.g. through the guardrail input surface)
  can bring down the audit.
- **Suggested fix**: Add a `depth` parameter with a cap of 32; use
  `list.append` + `".".join` instead of repeated string
  concatenation.

#### M-5 — `PrivacyAuditor._findings` grows unbounded over the auditor's lifetime

- **Severity**: medium
- **Category**: resource
- **File and line range**: `i3/privacy/sanitizer.py:260, 387, 466`
- **Quote**:
  ```python
  # i3/privacy/sanitizer.py:260, 387, 466
  self._findings: list[dict] = []
  ...
  self._findings.append(result)  # x2
  ```
- **Failure scenario**: Every `audit_database` and `audit_request`
  call appends to the list, never trims. A long-lived auditor
  (e.g. attached to the running pipeline) accumulates GBs of
  violation records if the sanitiser is ever misconfigured.
- **Suggested fix**: Bound with `collections.deque(maxlen=1_000)`
  or trim on each append.

#### M-6 — `admin_export` does N+1 SQLite connections per export

- **Severity**: medium
- **Category**: perf
- **File and line range**: `server/routes_admin.py:401-439`
- **Quote**:
  ```python
  # server/routes_admin.py:401-405
  sessions = await pipeline.diary_store.get_user_sessions(user_id, limit=1_000)
  for session in sessions:
      exchanges = await pipeline.diary_store.get_session_exchanges(
          session["session_id"]
      )
  ```
  Each inner call opens a fresh `aiosqlite.connect` per H-2.
- **Failure scenario**: Exporting a 1000-session user requires 1001
  database connections (1 for `get_user_sessions`, 1000 for the
  per-session loop). Combined with H-2 this is several seconds of
  admin export latency with the event loop pinned.
- **Suggested fix**: Add `DiaryStore.get_user_full_export(user_id,
  limit)` that does the JOIN in SQL and returns one flat result set;
  or fix H-2 and let connection reuse handle the bulk of it.

#### M-7 — `cloud_client.generate_session_summary` has no per-call timeout wrapper

- **Severity**: medium
- **Category**: robustness
- **File and line range**: `i3/pipeline/engine.py:368-377`
- **Quote**:
  ```python
  # i3/pipeline/engine.py:368-371
  if self.cloud_client.is_available:
      summary_text = await self.cloud_client.generate_session_summary(
          session_summary
      )
  ```
- **Failure scenario**: `generate_session_summary` calls
  `self.generate(...)`, which has up to `(_MAX_RETRIES+1)` attempts
  with per-attempt timeouts up to `_MAX_TIMEOUT_SECONDS` and up to
  `_MAX_TOTAL_BACKOFF_SECONDS` of sleep on top. So end-session can
  block for up to ~45 s. The caller (`WebSocket session_end`
  handler) doesn't wrap this in `asyncio.wait_for`, so a slow cloud
  turns into a long tail for the client's session-end round-trip.
- **Suggested fix**: Wrap the call in
  `asyncio.wait_for(..., timeout=self.cloud_client.timeout * 1.2)`
  with a graceful fallback to `_build_fallback_summary` on timeout.

#### M-8 — `InteractionMonitor` uses `threading.Lock` inside async code

- **Severity**: medium
- **Category**: concurrency
- **File and line range**: `i3/interaction/monitor.py:104,
  122-129`
- **Quote**:
  ```python
  # i3/interaction/monitor.py:104
  self._sessions_lock = threading.Lock()
  ```
  ```python
  # i3/interaction/monitor.py:122-128
  with self._sessions_lock:
      session = self._sessions.get(user_id)
      if session is None:
          session = _UserSession(...)
          self._sessions[user_id] = session
  ```
- **Failure scenario**: `_get_or_create_session` is called from both
  `async def process_keystroke` and `async def process_message`.
  `threading.Lock.__enter__` blocks the calling thread if contended
  — inside an asyncio coroutine this stalls the event loop rather
  than yielding. Today the critical section is short, but mixing
  `threading.Lock` and asyncio is a footgun: any future refactor
  that expands the critical section instantly regresses the event-
  loop responsiveness guarantee.
- **Suggested fix**: Switch to `asyncio.Lock` and make
  `_get_or_create_session` an `async def`; call it with `await`.

#### M-9 — `_UserCache.get_or_create` eviction blast radius not tracked

- **Severity**: medium
- **Category**: correctness
- **File and line range**: `server/routes_preference.py:201-210`
- **Quote**:
  ```python
  # server/routes_preference.py:207-209
  while len(self._store) > self._max:
      evicted, _ = self._store.popitem(last=False)
      logger.debug("Evicted preference state for user_id=%s", evicted)
  ```
- **Failure scenario**: When `_UserCache` evicts the oldest user,
  the `_UserState` (dataset, reward model, selector) is dropped
  silently. Any in-flight learning for that user is lost. The
  `stats` endpoint (`server/routes_preference.py:348`) subsequently
  returns 404 for that user, confusing the UI (which had just been
  showing "12 pairs collected"). No metric or WARNING log signals
  the eviction to operators.
- **Suggested fix**: Promote the eviction log to WARNING with
  `extra={"event": "preference_eviction", "user_id": evicted}`,
  and consider persisting the evictees to SQLite (the class already
  imports `aiosqlite` hooks via `PreferenceDataset.persist`).

#### M-10 — `manifest` in `ActivationCache.load` is trusted without shape validation or traversal check

- **Severity**: medium
- **Category**: correctness
- **File and line range**: `i3/interpretability/activation_cache.py:304-324`
- **Quote**:
  ```python
  # activation_cache.py:309-318
  manifest = json.loads(manifest_path.read_text())
  for name, shards in manifest.items():
      parts: list[torch.Tensor] = []
      for shard_name in shards:
          parts.append(
              torch.load(
                  src / shard_name, map_location="cpu", weights_only=True
              )
          )
  ```
- **Failure scenario**: `shard_name` is taken verbatim from
  `index.json`. If the manifest contains `"../../etc/foo.pt"`,
  `src / shard_name` traverses out of `src` because `Path.__truediv__`
  does not resolve `..`. `weights_only=True` blocks RCE via pickle,
  but an attacker who can write `index.json` (a model marketplace
  scenario) can at minimum cause `torch.load` to read arbitrary
  `.pt` files on disk and bundle them as "activations" — information
  disclosure. Also, no size check on `index.json` itself; no type
  check that `shards` is a list of strings.
- **Suggested fix**: After computing `candidate = src / shard_name`,
  call `candidate = candidate.resolve()` and assert
  `candidate.is_relative_to(src.resolve())`. Validate the JSON
  structure: `dict[str, list[str]]`. Cap `manifest_path.stat().st_size`.

#### M-11 — `_ensure_client` double-open not only races; client persists after server reload

- **Severity**: medium
- **Category**: resource / concurrency
- **File and line range**: `i3/cloud/providers/ollama.py:75-92`
  (illustrative; the same pattern recurs across providers).
- **Quote**:
  ```python
  # ollama.py:75-77
  def _ensure_client(self) -> httpx.AsyncClient:
      if self._client is None:
          self._client = httpx.AsyncClient(...)
      return self._client
  ```
  Sync method called from async code without a lock. The OpenRouter,
  Anthropic, Huawei variants all have the same shape (first-pass
  audit L-1 covered the kwargs but not the race).
- **Failure scenario**: Identical to H-6; demoted to medium here
  because Ollama is local-only in the expected deployment. But
  applying the fix to Anthropic (H-6) and not here is an
  inconsistency that will regress.
- **Suggested fix**: Centralise in
  `i3/cloud/providers/base.py::make_client()` that holds a
  `asyncio.Lock` and a single shared `httpx.AsyncClient` factory
  per adapter.

#### M-12 — `RateLimitMiddleware.DEFAULT_EXEMPT_PREFIXES` contains dead entries

- **Severity**: medium
- **Category**: config / correctness
- **File and line range**: `server/middleware.py:401-410`
- **Quote**:
  ```python
  # server/middleware.py:406-409
  "/docs",          # FastAPI autodoc
  "/redoc",         # FastAPI redoc
  "/openapi.json",  # FastAPI OpenAPI schema
  ```
- **Failure scenario**: In `server/app.py:176-178` the app actually
  mounts docs at `/api/docs`, `/api/redoc`, `/api/openapi.json` —
  so the bare-prefix exempts in the limiter are **never** hit, and
  the docs surfaces ARE rate-limited. That may be desirable, but
  the dead exempts are actively misleading — a future dev who
  reads the exempt list will believe the docs are off-limit to the
  limiter.
- **Suggested fix**: Either remove the dead `/docs`, `/redoc`,
  `/openapi.json` entries, or update them to `/api/docs`,
  `/api/redoc`, `/api/openapi.json` if the intent is to exempt
  them.

#### M-13 — Readiness probe leaks encryption-key status to unauthenticated clients

- **Severity**: medium
- **Category**: api / config
- **File and line range**: `server/routes_health.py:118-151`
- **Quote**:
  ```python
  # routes_health.py:125-149
  checks = {
      "pipeline": pipeline_status,
      "encryption_key": key_status,  # "ok" / "disabled" / "missing"
      "disk": disk_status,           # "ok" / "low" / "unknown"
  }
  ```
- **Failure scenario**: `/api/ready` is unauthenticated (in the
  limiter's exempt set). The body exposes whether encryption is
  configured, whether the disk is low, and whether the pipeline is
  still initialising. For a public endpoint this is useful to
  Kubernetes, but it also lets a scanner fingerprint the deployment
  posture. The `details` dict further leaks the raw detail strings
  like `"I3_DISABLE_ENCRYPTION=1"` and `"pipeline.is_ready == False"`.
- **Suggested fix**: Split the probe: emit a minimal
  `{"status": "ready"}` / `{"status": "not_ready"}` on
  `/api/ready`, and keep the detailed dict on a separate
  `/api/ready/detailed` gated by the admin token or a loopback-only
  ACL.

#### M-14 — `routes_inference` 404 detail leaks expected filename and directory

- **Severity**: medium
- **Category**: api
- **File and line range**: `server/routes_inference.py:117-124`
- **Quote**:
  ```python
  # routes_inference.py:117-124
  raise HTTPException(
      status_code=404,
      detail=(
          "Model not found. Export the ONNX artefact with "
          "`python -m i3.encoder.onnx_export` and drop it into "
          "web/models/."
      ),
      ...
  )
  ```
- **Failure scenario**: Other 404s in the same function return the
  constant string "Model not found" precisely to not leak anything.
  This one hint is inconsistent and tells a scanner both the
  filesystem layout (`web/models/`) and the toolchain used
  (`python -m i3.encoder.onnx_export`).
- **Suggested fix**: Collapse both 404 branches to the same
  constant `"Model not found"` string; keep the operator hint in
  the server log.

#### M-15 — `routes_translate` exception re-raise drops cause

- **Severity**: medium
- **Category**: type / correctness
- **File and line range**: `server/routes_translate.py:332-337`
- **Quote**:
  ```python
  # routes_translate.py:332-337
  try:
      body = TranslateRequest.model_validate_json(raw_body)
  except ValueError as exc:
      logger.debug("translate: validation failure: %s", exc)
      raise HTTPException(status_code=422, detail="Invalid request payload")
  ```
  Missing `from exc` — the original exception chain is lost.
- **Failure scenario**: Server-side log shows `HTTPException` with
  a synthetic traceback instead of pointing at the original Pydantic
  failure. Minor correctness / observability issue. (The same
  pattern IS correct in `routes_preference.py:237`.)
- **Suggested fix**: `raise HTTPException(...) from exc`. Same
  pattern is used correctly in peer files — worth making uniform.

#### M-16 — Preference route fabricates a "Response A"/"Response B" pair but stores it as the user's `last_candidate`

- **Severity**: medium
- **Category**: correctness
- **File and line range**: `server/routes_preference.py:161-191`
- **Quote**:
  ```python
  # routes_preference.py:170-190
  fabricated = PreferencePair(
      prompt="Which response feels more natural right now?",
      response_a="Response A",
      response_b="Response B",
      winner="tie",
      ...
      user_id="anonymous",
  )
  return [fabricated]
  ```
- **Failure scenario**: The fabricated pair's `user_id="anonymous"`
  — if `register_labelled` is later called, it writes with that id
  rather than the querying user's id, breaking `filter_by_user`.
  Moreover, the neutral pair is regenerated every call because
  `candidate_pairs()` constructs it fresh; two back-to-back
  `GET /query/{u}` can return different `information_gain` values.
- **Suggested fix**: Either don't fabricate (return 204 / empty
  `should_query=False`) or mark fabricated pairs so they cannot
  accidentally be recorded. Set `user_id=user_id` rather than
  `"anonymous"`.

---

### LOW

#### L-1 — `_safe_int`/`_safe_float` have an unused `default` parameter

- **Severity**: low
- **Category**: type / docs
- **File and line range**: `server/websocket.py:189, 204`
- **Quote**:
  ```python
  # websocket.py:189, 204
  def _safe_int(value: Any, default: int = 0) -> int: ...
  def _safe_float(value: Any, default: float = 0.0) -> float: ...
  ```
  `default` is never referenced inside either function body.
- **Suggested fix**: Drop the parameter (no call site passes it) or
  implement the fallback behaviour it implies.

#### L-2 — `_install_health_router` path-prefix shadowing of `routes.py::/health`

- **Severity**: low
- **Category**: correctness
- **File and line range**:
  `server/routes.py:95-104`; `server/routes_health.py:102-109`
  (both define `/health`); `i3/observability/instrumentation.py:186-202`
- **Quote**: `routes.py:95` registers `@router.get("/health")` on the
  `api_router`, `routes_health.py:102` registers `@router.get("/health")`
  on the health router; both are mounted under `/api`, so FastAPI
  silently picks the first-registered route.
- **Failure scenario**: Two different `/api/health` handlers with
  slightly different response shapes exist in the codebase;
  whichever is registered first wins. Order is non-obvious (depends
  on observability setup running before or after `include_router`).
- **Suggested fix**: Delete `routes.py::health_check` (the richer
  `routes_health.py` version already covers it) or explicitly alias.

#### L-3 — `server/app.py` `setup_observability` runs **before** exception handlers and routers are installed

- **Severity**: low
- **Category**: build
- **File and line range**: `server/app.py:248-321`
- **Quote**: order is middleware → observability → exception handlers →
  routers.
- **Failure scenario**: Observability instrumentation patches
  FastAPI's `__call__` and Starlette routing. If the patch hooks
  into route registration, routes added AFTER the patch may not be
  observed. The pattern is subtle and depends on what
  `setup_observability` actually patches; worth moving it after
  all `include_router` calls for determinism.
- **Suggested fix**: Move `setup_observability` to the very end of
  `create_app`, after the static mounts.

#### L-4 — `Pipeline.process_message` parameter name `input` shadows builtin

- **Severity**: low
- **Category**: type
- **File and line range**: `i3/pipeline/engine.py:410, 440, 456`
- **Quote**: `async def process_message(self, input: PipelineInput)`
- **Failure scenario**: `input` shadows the builtin `input()`. Dead
  bug class, but ruff `A` rule flags it and the next developer
  that types `input("…")` in that function gets an AttributeError.
- **Suggested fix**: Rename to `pipeline_input` or `req`.

#### L-5 — `_REPORTS_DIR` declared then unused

- **Severity**: low
- **Category**: type / docs
- **File and line range**: `server/routes_admin.py:619`
- **Quote**:
  ```python
  _REPORTS_DIR = Path("reports")
  ```
- **Failure scenario**: Dead code; the comment claims it supports an
  offline CLI but no code in this module references it.
- **Suggested fix**: Remove, or move to the CLI module where it is
  actually used.

#### L-6 — `DiaryConfig.db_path` traversal check uses `Path(v).parts` — bypassed by symlinks

- **Severity**: low
- **Category**: config
- **File and line range**: `i3/config.py:382-390`
- **Quote**:
  ```python
  # i3/config.py:388-390
  if ".." in Path(v).parts:
      raise ValueError(f"db_path must not contain '..' components: {v!r}")
  ```
- **Failure scenario**: `Path(v).parts` only catches literal `..` in
  the string. An attacker who controls a symlink inside `data/`
  pointing out of the project can redirect the write. Also, absolute
  paths (`"/etc/diary.db"`) are not rejected.
- **Suggested fix**: After validation, resolve the path and assert
  it is inside a configured root directory; reject absolute paths
  unless an explicit opt-in flag is set.

#### L-7 — `RouterConfig.prior_beta` is never consumed

- **Severity**: low
- **Category**: config / docs
- **File and line range**: `i3/config.py:228` (declared);
  nothing reads it (see H-9 for `prior_alpha` misuse).
- **Failure scenario**: Operators tune `prior_beta` in YAML
  believing it affects the bandit; it does not.
- **Suggested fix**: Either pass it to `ContextualThompsonBandit`
  (which currently hardcodes `alpha=1.0, beta_param=1.0`) or remove
  it with a deprecation note.

#### L-8 — `CloudLLMClient._parse_response` echoes upstream response keys in exception

- **Severity**: low
- **Category**: api
- **File and line range**: `i3/cloud/client.py:256-261`
- **Quote**:
  ```python
  raise ValueError(
      f"Unexpected response structure: missing 'content' list.  "
      f"Keys received: {list(data.keys())}"
  )
  ```
- **Failure scenario**: If the upstream ever returns an unexpected
  payload (e.g. a new error-envelope shape), its top-level keys
  are echoed into the `ValueError`. The callers log at DEBUG so
  this mostly stays in logs, but it is a mild upstream-shape leak.
- **Suggested fix**: Log the keys at DEBUG in a separate statement;
  keep the exception message constant.

#### L-9 — `PreferenceStatsResponse.reward_model_ready` threshold hardcoded

- **Severity**: low
- **Category**: config
- **File and line range**: `server/routes_preference.py:378`
- **Quote**:
  ```python
  reward_model_ready=bool(pairs >= 8),
  ```
- **Failure scenario**: Magic number `8` is not tied to the
  `state.target_labels` budget (20) or any field on
  `BradleyTerryRewardModel`. Moving the readiness threshold requires
  a code change, not a config change.
- **Suggested fix**: Expose as a config field
  (`preference.reward_model_min_pairs`) or attach to the reward
  model itself.

#### L-10 — `_surrogate_mapping_fn` is re-allocated on every explain call

- **Severity**: low
- **Category**: perf
- **File and line range**: `server/routes_explain.py:231-246`
- **Quote**: Inside `_build_payload` and `explain_adaptation`, each
  branch calls `CounterfactualExplainer(mapping_fn=_surrogate_mapping_fn(), …)`.
- **Failure scenario**: Small `nn.Linear(32, 8, bias=False)` is
  cheap but still ~1k floats allocated per request. Stacked with
  H-8 (global-seed pollution), it's wasted work.
- **Suggested fix**: Cache the layer at module scope once.

#### L-11 — `DiaryStore._initialized` flag is read without a memory barrier

- **Severity**: low
- **Category**: concurrency
- **File and line range**: `i3/diary/store.py:167-173`
- **Quote**: `_ensure_initialized` check is a plain attribute read.
- **Failure scenario**: Under concurrent initialisation / reset, the
  `_initialized` flag could be observed stale; harmless in asyncio
  but the module advertises thread-safety implicitly.
- **Suggested fix**: `initialize()` already runs serially at
  startup; note in the docstring that the store is not reusable
  concurrently with `initialize()`.

#### L-12 — `pipeline._previous_route.pop(user_id, None)` called in `start_session` but not `process_message`

- **Severity**: low
- **Category**: correctness
- **File and line range**: `i3/pipeline/engine.py:328`
- **Quote**: Only `start_session` clears this dict. `process_message`
  updates it (line 585) but `end_session` clears only the length
  and time dicts (lines 399-401), leaving `_previous_route` stale
  until the next `start_session`.
- **Suggested fix**: Clear `_previous_route` alongside the others
  in `end_session`.

#### L-13 — `InteractionMonitor.process_message` appends to `feature_window` via `list.pop(0)`

- **Severity**: low
- **Category**: perf
- **File and line range**: `i3/interaction/monitor.py:211-213`
- **Quote**:
  ```python
  session.feature_window.append(fv)
  if len(session.feature_window) > self._feature_window_size:
      session.feature_window.pop(0)
  ```
- **Failure scenario**: `list.pop(0)` is O(n); for a feature window
  of default 10 this is negligible, but the same pattern is a
  footgun if the window is raised. `collections.deque(maxlen=…)` is
  O(1) and self-trimming.
- **Suggested fix**: Convert `feature_window` to
  `collections.deque(maxlen=self._feature_window_size)`.

#### L-14 — `ModelEncryptor` silently falls back to an ephemeral in-memory key

- **Severity**: low
- **Category**: config
- **File and line range**: `i3/privacy/encryption.py:90-100`
- **Quote**:
  ```python
  # encryption.py:92
  self._fernet = Fernet(Fernet.generate_key())
  logger.warning(
      "No encryption key found in %s. "
      ...
  )
  ```
- **Failure scenario**: Production deployment without
  `I3_ENCRYPTION_KEY` gets a warning but still writes "encrypted"
  blobs that cannot be decrypted after the next restart. The
  readiness probe does check for the key (`routes_health.py:72-85`),
  but only the `I3_DISABLE_ENCRYPTION=1` branch is explicit — the
  ephemeral-fallback case returns `"missing"` yet the encryptor
  happily continues. Inconsistent failure modes.
- **Suggested fix**: Add a `strict: bool = True` parameter;
  `initialize()` raises when strict and no key is present, unless
  `I3_ALLOW_EPHEMERAL_KEY=1` is set. Default to strict for
  production.

---

### INFO / positive findings

- **I-1** — The background-task pattern in `Pipeline._background_tasks`
  (`engine.py:568-580, 1041-1059`) correctly holds strong refs and
  consumes `task.exception()` in the done-callback so silent failures
  are impossible. This is exactly the pattern the first-pass audit
  complimented.
- **I-2** — `CloudLLMClient` caps retries, cumulative backoff, per-phase
  timeouts, response-body size, and request-body size — a
  disciplined defensive profile that most outbound HTTP clients
  skip. (`i3/cloud/client.py:292-560`.)
- **I-3** — `ConnectionManager` in the WebSocket layer is race-free on
  eviction: it passes the `WebSocket` identity to `disconnect()` so a
  stale handler's cleanup cannot clobber a freshly installed slot.
  (`server/websocket.py:99-141`.)
- **I-4** — Every Pydantic request model on the new routes uses
  `ConfigDict(extra="forbid")` — consistent and correct. The root
  `Config` in `i3/config.py` is the sole outlier (see M-2).
- **I-5** — The sanitizer's ordering comment
  (`i3/privacy/sanitizer.py:67-68`) documents the URL-before-email and
  SSN-before-phone dependency, which is exactly the non-obvious
  invariant a future maintainer would trip over.
- **I-6** — `load_config` correctly validates CORS origins
  (`i3/config.py:459-480`) — scheme must be `http(s)`, netloc must be
  present. This catches the `localhost:8000` (missing scheme) class
  of bugs at config-load time, not at Starlette-dispatch time.
- **I-7** — `routes_whatif` correctly clamps each override dimension
  to `[0, 1]` AND rejects NaN defensively (`routes_whatif.py:186-213`),
  even though the Pydantic field validators already bound the range.
  Defence-in-depth on a numeric input surface.

---

## Files audited

- `server/app.py` — FastAPI bootstrap; feeds M-1 (double-load),
  L-3 (observability order).
- `server/auth.py` — token map is re-parsed per request (minor perf).
- `server/middleware.py` — feeds M-12 (dead exempt prefixes);
  positive on sliding-window OrderedDict after first-audit fix.
- `server/routes.py` — three GET routes correctly gated; positive.
- `server/routes_admin.py` — feeds M-6 (N+1 on export), L-5 (dead
  `_REPORTS_DIR`).
- `server/routes_health.py` — feeds L-2 (dup `/health`), M-13
  (detail leak).
- `server/routes_inference.py` — feeds M-14 (404 detail).
- `server/routes_explain.py` — feeds H-1 (no auth on POST),
  H-8 (global seed), L-10 (mapping fn alloc).
- `server/routes_preference.py` — feeds H-1 (no auth on record),
  M-9 (silent eviction), M-16 (fabricated pair).
- `server/routes_translate.py` — feeds H-1 (no auth), M-15 (exception
  chaining).
- `server/routes_tts.py` — feeds H-1 (no auth), H-3 (sync TTS).
- `server/routes_whatif.py` — feeds H-1 (no auth), H-3 (sync SLM),
  positive on numeric validation.
- `server/websocket.py` — feeds **B-1** (keystroke never awaited),
  L-1 (unused `default` kwarg), positive on connection race-freedom.
- `i3/config.py` — feeds M-1 (seed side effect), M-2
  (`extra="forbid"`), M-3 (model id drift), L-6 (db_path traversal),
  L-7 (`prior_beta` unused).
- `i3/pipeline/engine.py` — feeds H-3 (sync SLM), H-4 (unbounded
  dicts), H-7 (error class leak), L-4 (`input` shadow), L-12
  (`_previous_route` cleanup).
- `i3/privacy/sanitizer.py` — feeds M-4 (recursion depth), M-5
  (`_findings` growth); positive I-5.
- `i3/privacy/encryption.py` — feeds L-14 (ephemeral-key fallback).
- `i3/user_model/store.py` — exception strings embed `user_id`
  (minor; not raised as finding since the caller is admin-only).
- `i3/diary/store.py` — feeds H-2 (per-call connections + missing
  PRAGMA on reused conns), L-11 (`_initialized` flag).
- `i3/router/router.py` — feeds H-9 (`prior_alpha` / `prior_precision`
  semantic swap).
- `i3/router/bandit.py` — feeds H-5 (concurrency), H-4 (`history`
  slice O(n) churn).
- `i3/router/preference_learning.py` — `PreferenceDataset._pairs`
  unbounded (same class of H-4).
- `i3/cloud/client.py` — feeds H-6 (`_ensure_client` race),
  L-8 (response-key leak); positive I-2.
- `i3/cloud/providers/ollama.py` — feeds M-11 (same race).
- `i3/interpretability/activation_cache.py` — feeds M-10 (manifest
  traversal).
- `i3/interaction/monitor.py` — feeds B-1 (consumer of unawaited
  coroutine), M-8 (threading.Lock in async), L-13 (`pop(0)`).
- `i3/adaptation/uncertainty.py` — header only; not a source of
  findings.
- `i3/tts/engine.py` — header only; sync dispatch referenced by H-3.
- `i3/safety/pddl_planner.py` — header only; positive posture.

---

## Test coverage gaps

Not audited for their own robustness, but enumeration of what the
`tests/` directory *does not* cover on the production surface:

- No tests for `server/routes_whatif.py` (the rate-limit bypass the
  first audit found lives here; still no coverage of the non-bypass
  path).
- No tests for `server/routes_explain.py` (the H-8 global-seed
  mutation and H-1 auth gap would be caught by a concurrent-request
  test).
- No tests for `server/routes_admin.py` (M-6 N+1 would be visible in
  any timing test).
- No tests for `server/auth.py` (the token-map parsing corner cases
  are unverified).
- No tests for `server/routes_inference.py` other than an
  `advanced_ui_static` smoke test.
- No WebSocket test that asserts keystroke data actually reaches
  `InteractionMonitor` — which is why B-1 has been shippable.
- The `DiaryStore` test set does not appear to exercise concurrent
  writers (where the missing `PRAGMA foreign_keys` would reveal
  itself).

---

## Suggested next priorities

1. **Fix B-1 (keystroke never awaited).** Single-line fix in
   `websocket.py:429`. Add a WebSocket integration test that sends
   keystroke frames and asserts `feature_window` is populated. Without
   this, the product's headline behavioural-baseline claim is
   uninstantiated.

2. **Extend `require_user_identity` to the five POST routes that
   accept `user_id` (H-1).** Mechanical change but essential for the
   auth invariant to hold end-to-end.

3. **Persist `DiaryStore._db` across calls and re-enable FK
   enforcement on the reused connection (H-2).** One open, one close;
   every hot path gets faster and correctness of orphan-exchange
   prevention is restored.

4. **Lift `_slm_generator.generate` and `TTSEngine.speak` off the
   event loop via `run_in_executor` (H-3).** Follow the encoder's
   pattern at `engine.py:478`; parallelise `whatif_compare` variants
   with `asyncio.gather`.

5. **Add LRU caps to `Pipeline.user_models` + friends (H-4) and lock
   `ContextualThompsonBandit` (H-5).** Both are straightforward and
   prevent long-run degradation on production traffic.

Everything else is deferrable; the first five fixes land the product
in a meaningfully better state without blocking downstream work.
