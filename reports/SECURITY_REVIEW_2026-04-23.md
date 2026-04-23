# I³ Deep Manual Security Audit — 2026-04-23

Auditor: Claude Opus 4.7 (manual review pass, beyond the 46-check verification
harness and the 55-attack red-team harness).
Scope: production code in `server/` and `i3/`, plus infrastructure manifests
and `.env*` / `configs/default.yaml`. Tests and scripts skipped per brief.

## Executive Summary

Counts by severity (new findings only; items already covered by the
automated harnesses are not re-listed):

- **Blocker**: 0
- **High**: 2
- **Medium**: 5
- **Low**: 5
- **Info / positive**: 8

Top three concerns:

1. **H-1 — `/whatif/*` endpoints bypass the rate limiter entirely.** They are
   mounted under `/whatif` instead of `/api/whatif`, and
   `RateLimitMiddleware.dispatch()` in `server/middleware.py:413` short-
   circuits anything that does not start with `/api`. These endpoints run SLM
   generation (`whatif/compare` can fan out to four variants per request) and
   are reachable with zero throttling. Trivial DoS + GPU-burn vector.
2. **H-2 — `routes_preference` stores unsanitised free-text and exposes it
   cross-user without authentication.** `prompt`, `response_a`,
   `response_b` (4 096 chars each) are accepted verbatim, NEVER passed
   through `PrivacySanitizer`, persisted in an in-memory `_UserCache`, and
   then **returned** by the unauthenticated `GET /api/preference/query/
   {user_id}`. Any client that knows a user_id can read what another user
   submitted, including any PII it contained.
3. **M-1 — `torch.load` on an attacker-reachable path without
   `weights_only=True`.** `i3/interpretability/activation_cache.py:279,308`
   loads activation caches with the default pickle deserialiser. If a
   future feature ingests user-supplied caches (research workflow, model
   marketplace, etc.) this is RCE-in-waiting. It is not currently exposed
   over the network, hence medium rather than high.

Overall posture verdict: **solid, with a small number of real gaps.** The
server's defence-in-depth is strong — explicit CORS allow-list, negotiated
HSTS, request-size + sliding-window limiters, per-route body caps, a PII
sanitiser module-level singleton on the critical routes, a PDDL-grounded
safety planner, `secrets.compare_digest` for the admin token, Fernet with
authenticated encryption for embeddings, a disciplined `torch.load(..,
weights_only=True)` pattern on inference paths, and a correctly
hardened outbound `httpx` client for the main Anthropic path. The issues
found are about the **edges** of that posture — the new routes
(`routes_preference`, `routes_whatif`) that were added later and did not
inherit every invariant the older routes have.

## Methodology

For each file listed in the audit brief I read the full source one file at
a time, with focus on (1) input validation, (2) authentication/authorisation
around state-changing routes, (3) PII flow into persistence or outbound
cloud calls, (4) cryptographic primitives and key handling, (5) path
traversal / SSRF / command injection, (6) deserialisation, (7) rate-limit
and DoS exposure, (8) secret disclosure, (9) supply chain, (10) CORS /
header hygiene, (11) concurrency, and (12) fail-open vs fail-closed
behaviour on sanitiser / limiter / planner error paths.

I cross-checked grep scans for `pickle.loads`, `yaml.load`, `eval`, `exec`,
`subprocess shell=True`, `requests.get`, `urlopen`, `torch.load(` without
`weights_only`, `allow_origins=["*"]`, `allow_credentials=True`,
`compare_digest`, `Fernet`, raw `open(user_...)`, and prints of request
bodies.

What I did NOT do: dynamic exploitation (no live server — torch DLL
issue), SBOM diffing (out of scope), or review of tests, scripts,
notebooks, docs.

## Findings

---

### HIGH

#### H-1 — `/whatif/*` endpoints are not rate-limited

- **Severity**: high
- **File**: `server/routes_whatif.py:53`, `server/routes_whatif.py:435-438`
  (`include_whatif_routes`); `server/middleware.py:413-419`
  (`RateLimitMiddleware.dispatch`).
- **Quote**:
  ```python
  # server/middleware.py:414-419
  if (
      not path.startswith("/api")
      or path in self._exempt_paths
      or path.startswith("/ws/")
  ):
      return await call_next(request)
  ```
  ```python
  # server/routes_whatif.py:53
  router = APIRouter(prefix="/whatif", tags=["whatif"])
  ```
  ```python
  # server/routes_whatif.py:435-438
  # SEC: mounted under /whatif (no /api prefix) so the interpretability
  # panel can namespace itself distinctly from the main REST API in
  # reverse-proxy rules.
  app.include_router(router)
  ```
- **Attack**: An unauthenticated attacker POSTs a continuous stream of
  requests to `/whatif/compare` (up to 4 adaptation variants each — so up
  to 4× `SLMGenerator.generate` per request). No per-IP limit, no
  circuit-breaker, no account gate. On a GPU-backed deployment this is
  cheap DoS and expensive GPU-burn; on a CPU-only deployment it is cheap
  CPU-burn and latency amplification against legitimate users.
  `whatif/respond` is the same shape minus the fan-out.
- **Fix**: Either (a) change the prefix to `/api/whatif` so the existing
  limiter catches it, or (b) change the middleware gate to `path.startswith
  (("/api", "/whatif"))`. Option (a) is the less surprising one and already
  matches the WebSocket rate-limit convention.

#### H-2 — Preference endpoints store unsanitised free text and return it cross-user without authentication

- **Severity**: high
- **File**: `server/routes_preference.py:68-81` (Pydantic model),
  `:219-275` (record handler), `:278-327` (query handler).
- **Quote**:
  ```python
  # server/routes_preference.py:73-76
  prompt: str = Field(..., min_length=1, max_length=4096)
  response_a: str = Field(..., min_length=1, max_length=4096)
  response_b: str = Field(..., min_length=1, max_length=4096)
  ```
  The handler stores the values as-is — no `PrivacySanitizer.sanitize`
  anywhere in the module (grep `PrivacySanitizer` in the file confirms
  it is never imported):
  ```python
  # server/routes_preference.py:249-268
  pair = PreferencePair(
      prompt=body.prompt,
      response_a=body.response_a,
      response_b=body.response_b,
      ...
  )
  ...
  state.dataset.append(pair)
  state.selector.register_labelled(pair)
  state.last_candidate = pair
  ```
  And `GET /api/preference/query/{user_id}` returns them verbatim:
  ```python
  # server/routes_preference.py:311-319
  payload = PreferenceQueryResponse(
      user_id=user_id,
      should_query=bool(ig >= threshold),
      prompt=chosen.prompt,
      response_a=chosen.response_a,
      response_b=chosen.response_b,
      ...
  )
  ```
- **Attack**:
  (a) **PII leak on the wire.** A caller submits a pair that includes an
  email / phone / SSN in `response_a`. It is stored as-is. Any subsequent
  `/api/preference/query/<that_user>` GET returns it — the sanitiser
  contract advertised throughout the rest of the product does not hold
  here.
  (b) **Cross-user read.** `routes_preference.py` does not authenticate
  the GET caller against the path `user_id`. Combined with (a), an
  attacker who knows or guesses a user id can harvest whatever prose the
  target submitted. This is the same "any client can read any user_id"
  limitation noted at the top of `server/routes.py` but here the payload
  is **free text**, not embeddings + scalars.
  (c) **Memory amplification.** Three 4 KiB fields × the 256-entry cache
  × future cache-growth = unbounded memory pressure over time if a
  scripted client rotates user_ids; partially mitigated by the LRU but
  the bodies are large enough that a handful of writes per key can push
  the server over a reasonable working-set.
- **Fix**: Add
  `_SANITIZER = PrivacySanitizer(enabled=True)` at module scope and pass
  `prompt` / `response_a` / `response_b` through `.sanitize()` in
  `record_preference` before constructing `PreferencePair`. Separately,
  gate `GET /api/preference/query/{user_id}` on the same authentication
  rule the rest of the server plans to enforce — at a minimum add the
  same "Known limitation" docstring `routes.py` has and file a followup.

---

### MEDIUM

#### M-1 — `torch.load` without `weights_only=True` on potentially attacker-reachable paths

- **Severity**: medium
- **File**: `i3/interpretability/activation_cache.py:279`, `:308`;
  `i3/slm/train.py:535`.
- **Quote**:
  ```python
  # activation_cache.py:279
  payload = torch.load(src, map_location="cpu")
  ...
  # activation_cache.py:308
  parts.append(torch.load(src / shard_name, map_location="cpu"))
  ```
  ```python
  # slm/train.py:535
  checkpoint = torch.load(
      path, map_location=self.device, weights_only=False
  )
  ```
- **Attack**: `torch.load` without `weights_only=True` is a pickle
  deserialisation sink — any `.pt` / `.pkl` under the loaded path can
  trigger arbitrary code execution via `__reduce__`. Today neither
  path is reachable from the HTTP surface, but the activation-cache
  module is designed to be hydrated from a directory that could
  plausibly come from a model marketplace, a shared research bucket, or
  a user upload in a future feature. The train.py callsite is already
  commented as "trusted local" — that is the right posture, but the
  activation-cache file has no such disclaimer.
- **Fix**: In `activation_cache.py`, pass `weights_only=True` at both
  `torch.load` sites (the loaded artefacts are pure `torch.Tensor`
  dicts per the surrounding code's `isinstance(tensor, torch.Tensor)`
  check, so `weights_only=True` is a drop-in). For `slm/train.py`, add
  a file-integrity check (sha256 against a manifest) before the load,
  mirroring `i3/mlops/checkpoint.py:309`.

#### M-2 — Generic REST read endpoints serve any user's data without auth

- **Severity**: medium
- **File**: `server/routes.py:109-196` (`get_user_profile`,
  `get_user_diary`, `get_user_stats`).
- **Quote**:
  ```python
  # server/routes.py:18-23 (the authors already know):
  # Known limitation (demo build):
  #     There is no caller authentication, so any client can read any user_id's
  #     data. This is acceptable for the on-device demo but MUST be revisited
  #     before any multi-tenant deployment ...
  ```
- **Attack**: Any client that knows a user id can retrieve that user's
  profile, diary, and stats. The `/api/demo/*` routes are gated on
  `I3_DEMO_MODE`, but the read routes are not. User enumeration (404
  vs 200 shape) is also possible. Note: the **diary content** is
  embeddings + scalars — no raw text by design — so the blast radius is
  behavioural-fingerprint disclosure, not literal content leak.
- **Fix**: Introduce a caller-identity dependency that matches the
  path `user_id` against an authenticated claim. The admin surface
  already shows the pattern (`require_admin_token`); mirror it for
  `/api/user/*`.

#### M-3 — Rate-limit state is per-process; multi-worker deployments silently multiply the limit

- **Severity**: medium
- **File**: `server/middleware.py:266-290` (`_SlidingWindowLimiter`),
  `Dockerfile:148` (`I3_WORKERS=1`), `.env.example:97`
  (`I3_WORKERS=1` commented-out hint).
- **Quote**:
  ```python
  # server/middleware.py:268-271
  Not suitable for multi-process deployments (each worker gets its own
  state), but fine for the single-process demo server.  For production,
  replace with a Redis-backed implementation.
  ```
  The container ENV defaults to `I3_WORKERS=1` which is the right
  choice; however, nothing **refuses** to start when `I3_WORKERS > 1`.
- **Attack**: Operator sets `I3_WORKERS=4` for throughput. Each worker
  independently tracks `60 req/min/IP`, so the effective per-IP rate
  climbs to 240/min (on the worst-case hash distribution more, since
  Linux's SO_REUSEPORT will not guarantee sticky-IP-to-worker). The
  limiter fails open against distributed attack but fails silent —
  operators have no warning.
- **Fix**: Add a startup log at WARNING when `I3_WORKERS > 1` and a
  `redis` URL is not configured. Gate multi-worker launch behind an
  explicit `I3_ALLOW_LOCAL_LIMITER=1` override. Long-term, add the
  Redis-backed limiter hinted at in the docstring.

#### M-4 — OpenRouter adapter echoes upstream response body into the error

- **Severity**: medium
- **File**: `i3/cloud/providers/openrouter.py:158-162`.
- **Quote**:
  ```python
  if status >= 400:
      raise PermanentError(
          f"OpenRouter HTTP {status}: {response.text[:200]}",
          provider=self.provider_name,
      )
  ```
- **Attack**: If OpenRouter ever reflects a request header or the
  API key back (their 4xx bodies frequently include the offending
  `Authorization:` prefix), that fragment ends up in the
  `PermanentError` message. The cloud client — `i3/cloud/client.py:
  512-515` — sets the opposite standard: "never log / never echo the
  body". The inconsistency is not itself a breach but raises the
  probability of a future secret-leak through exception strings.
- **Fix**: Match the Anthropic policy — include only the status code
  and the provider name; log the body to the **logger** at DEBUG if
  needed, never to the exception message.

#### M-5 — `/whatif` and preference / translate routes bypass HTTP rate limit (related to H-1)

- **Severity**: medium
- **File**: `server/routes_translate.py:144` (`prefix="/api/translate"` —
  IS covered), `server/routes_preference.py:215` (`prefix="/api/
  preference"` — IS covered), `server/routes_tts.py:211` (covered),
  `server/routes_explain.py:134` (covered), `server/routes_whatif.py:
  53` (**NOT covered** — `/whatif`, no `/api`).
- **Attack**: See H-1. Listed separately because of the systemic
  observation: every other route family on the API surface **is**
  correctly namespaced under `/api`; `/whatif` is the sole outlier.
  This is more likely an oversight than intentional, given that the
  comment at `routes_whatif.py:435-438` invokes a reverse-proxy
  namespacing argument that is not load-bearing once the route is
  deployed behind a single ingress.
- **Fix**: Same as H-1. Mentioned again here because the limiter
  middleware's narrow prefix check is the common upstream: a defence-
  in-depth fix is to switch the middleware to an explicit **exclude-
  list** (static, websocket) rather than an **include-list** (`/api/`
  only).

---

### LOW

#### L-1 — Per-provider adapters do not uniformly assert `verify=True` / `follow_redirects=False` on their httpx clients

- **Severity**: low
- **File**: `i3/cloud/providers/openrouter.py:92-96`, vs
  `i3/cloud/client.py:200-207` and
  `i3/cloud/providers/huawei_pangu.py:141-142` which do.
- **Quote**:
  ```python
  # openrouter.py:92-96
  self._client = httpx.AsyncClient(
      base_url=_BASE_URL,
      headers=headers,
      timeout=httpx.Timeout(self._timeout),
  )
  ```
  (no `verify=True`, no `follow_redirects=False`, no size caps,
  no connection-pool `limits=`)
- **Attack**: httpx defaults to `verify=True` and
  `follow_redirects=False`, so nothing is actually broken today. But
  the regression risk is real — a future refactor that standardises
  on a helper which forgets these, or a library upgrade that flips
  defaults, would silently weaken the OpenRouter / Mistral / Cohere /
  OpenAI paths. The main Anthropic client gets this right explicitly.
- **Fix**: Either (a) add the same explicit kwargs to every adapter
  or (b) funnel every provider through a shared
  `_make_provider_client()` helper that sets them once.

#### L-2 — Error body `response.text` flows into log / exception strings in two provider paths

- **Severity**: low (covered separately in M-4 for OpenRouter; the
  other callsite is the Huawei adapter).
- **File**: `i3/cloud/providers/huawei_pangu.py` (error-path
  `response.text` handling; not quoted here).
- **Fix**: Redact before raising; log at DEBUG only.

#### L-3 — `contains_pii` IP-address regex will match non-IP dotted quads such as `10.0.22621` (Windows build numbers), SemVer fragments, or telemetry counters

- **Severity**: low (false-positive noise, not a miss)
- **File**: `i3/privacy/sanitizer.py:115-117`.
- **Quote**:
  ```python
  ("ip_address", re.compile(
      r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
  ), "[IP_ADDRESS]"),
  ```
- **Attack**: Not exploitable; the sanitiser is already bounded
  (`MAX_INPUT_LENGTH=50_000`) so ReDoS is out. The concern is
  correctness: any time a system prompt or user message contains a
  version string with three or more dots, the sanitiser reports a
  spurious PII hit, and downstream audit counters (`stats["pii_found"
  ]`) become noisy. The downstream cloud-audit in
  `PrivacyAuditor.audit_request` uses `contains_pii` as its sole
  signal, so the false positive inflates `pii_fields` there too.
- **Fix**: Require each octet to be `<= 255` (`(?:25[0-5]|2[0-4][0-9]|
  [01]?[0-9][0-9]?)`) and exclude leading-zero octets. Keeps ReDoS
  safety (patterns remain linear in length).

#### L-4 — Admin-export endpoint returns a non-404 shape for unknown `user_id`, leaking "exists" vs "doesn't exist" via the `profile_present` log field

- **Severity**: low
- **File**: `server/routes_admin.py:336-471` (`admin_export`), and the
  `"profile_present": profile_payload is not None` field emitted to
  the structured log.
- **Attack**: An admin-token holder who enumerates user IDs can tell
  which ones exist by watching the log telemetry — but since the
  client-visible JSON carries the same information (`profile: null`
  vs a populated dict), the enumeration is also visible on the wire.
  Only exploitable with the admin token, so a defence-in-depth
  concern rather than a breach.
- **Fix**: Return 404 when **both** `profile_payload is None` and
  `diary_payload == []` and `bandit_stats == {}` (all three signals
  absent → user genuinely does not exist). Low-effort; aligns with
  the "generic 404, do not echo user id" posture in `routes.py:133`.

#### L-5 — `_ws_rate_limiter` runs outside the `_SlidingWindowLimiter`'s `max_keys` cap guardrail **for the WebSocket user-id keyspace**

- **Severity**: low
- **File**: `server/websocket.py:80`, `server/middleware.py:450-464`.
- **Quote**:
  ```python
  # middleware.py:450-464
  @staticmethod
  def ws_limiter(
      limit: int = DEFAULT_WS_USER_RATE_LIMIT,
      window_seconds: int = DEFAULT_WINDOW_SECONDS,
      max_keys: int = DEFAULT_MAX_TRACKED_KEYS,
  ) -> _SlidingWindowLimiter:
      ...
  ```
  This is fine, but the WebSocket handler keys on the
  `user_id`-regex-validated name. The user_id regex allows
  `[a-zA-Z0-9_-]{1,64}` → 64^62 distinct identifiers, vs the IP
  keyspace which is naturally bounded. The eviction path keeps
  memory bounded, but a hostile client that rapidly rotates user_ids
  will churn the limiter's eviction loop (every new key takes the
  `_evict_oldest` path when the cap is saturated, which is O(n) over
  the dict).
- **Attack**: Memory stays bounded, but CPU cost per accept scales
  with `DEFAULT_MAX_TRACKED_KEYS` in the worst case. Combined with
  the WS origin-check it is hard to exploit from a browser, but a
  direct `websockets`-library client in the same origin allow-list
  could mount it.
- **Fix**: Switch `_evict_oldest` from `min(..., key=...)` (O(n)) to
  an OrderedDict insertion-order eviction (amortised O(1)). Same
  functional outcome, substantially cheaper under hostile churn.

---

### INFO / positive findings

- **I-1** — Admin auth uses `secrets.compare_digest` (`routes_admin.py:91`)
  and the token is never logged (`_redact_api_key`-style discipline).
- **I-2** — CORS wildcard is actively refused unless
  `I3_ALLOW_CORS_WILDCARD=1`, with `allow_credentials` forced to False
  whenever the allow-list contains `*` (`app.py:164-215`).
- **I-3** — Every route with free-text input on the v2 surface
  (`translate`, `tts`, the WebSocket handler) pipes it through
  `PrivacySanitizer.sanitize` before any persistence or cloud leg. The
  module-level `_SANITIZER` pattern reuses compiled regexes correctly.
- **I-4** — The Fernet wrapper uses AES-128-CBC + HMAC-SHA256 via the
  vetted `cryptography` library; no custom crypto, `MultiFernet` key
  rotation is implemented and documented (`encryption.py:314-378`).
- **I-5** — The outbound Anthropic client explicitly pins
  `verify=True`, `follow_redirects=False`, per-phase timeouts, response
  size ceiling (`_MAX_RESPONSE_BYTES = 2 MiB`), request size ceiling
  (`_MAX_REQUEST_BYTES = 256 KiB`), and a jittered-backoff + Retry-After-
  clamped retry budget (`client.py:169-208, 292-560`).
- **I-6** — Diary SQLite schema structurally excludes raw text
  (`diary/store.py:115-158` comment + no text column besides `summary`
  which is metadata-derived); the auditor in `privacy/sanitizer.py:
  226-459` sweeps real DB files for natural-language leaks defensively.
- **I-7** — PDDL planner encodes the cloud-route-forbidden invariant as
  a checkable safety certificate and re-verifies each step at
  certification time (`pddl_planner.py:465-556`).  The forward-search
  planner is a short, readable, dependency-free implementation that
  can itself be audited.
- **I-8** — Inference-path `torch.load` calls use `weights_only=True`
  (`encoder/inference.py:97-101`, `pipeline/engine.py:1252`,
  `eval/ablation_experiment.py:427-428`, `slm/quantize.py:421`,
  `encoder/onnx_export.py:197`, `slm/onnx_export.py:298`,
  `serving/ray_serve_app.py:112`). The training-path exception at
  `slm/train.py:535` is explicitly annotated and the file is never
  touched by the server process.

---

## Files audited

- `server/app.py` — FastAPI bootstrap; middleware ordering, CORS, exception
  handlers; localhost-by-default binding; `I3_DISABLE_OPENAPI` gate. No new
  findings beyond positive observations.
- `server/middleware.py` — security headers, body-size cap, sliding-window
  limiter. Feeds H-1 / M-5 (limiter predicate); L-5 (eviction complexity).
- `server/routes.py` — user read endpoints. Feeds M-2 (known-limitation
  still unfixed).
- `server/routes_admin.py` — token gate + GDPR export/delete. Feeds L-4
  (enumeration via export shape).
- `server/routes_translate.py` — PII-sanitise-then-cloud pattern.
  Positive (I-3).
- `server/routes_preference.py` — preference learning. Feeds H-2
  (unsanitised PII, cross-user read).
- `server/routes_explain.py` — uncertainty + counterfactual. Clean — no
  raw text read or returned.
- `server/routes_whatif.py` — override experiments. Feeds H-1 / M-5
  (rate-limit bypass via non-`/api` prefix).
- `server/routes_tts.py` — adaptation-conditioned TTS. Positive (I-3).
- `server/routes_health.py` — liveness / readiness / metrics. Does NOT
  leak hostname, Python version, or uptime-derived entropy in a
  uniquely-identifying way; disk check reports bytes only.
- `server/routes_inference.py` — ONNX file server. Regex + `resolve()`
  + `relative_to()` double-check stops path traversal; `Content-Type`
  locked to `application/octet-stream`; COOP/COEP explicitly set.
- `server/websocket.py` — WebSocket handler. Hard caps on session
  duration, message count, per-message bytes, keystroke-buffer length,
  numeric-coercion guards against NaN/inf, race-safe connection
  eviction. Feeds L-5 (minor churn-cost concern) only.
- `i3/privacy/sanitizer.py` — 10-pattern regex battery. ReDoS-guarded
  by `MAX_INPUT_LENGTH`; bounded quantifiers; module-level lock on
  stats. Feeds L-3 (IP-address false positives only).
- `i3/privacy/encryption.py` — Fernet wrapper. Positive (I-4).
- `i3/privacy/differential_privacy.py` — DP-SGD sketch (not active).
- `i3/safety/pddl_planner.py` — positive (I-7).
- `i3/safety/certificates.py` — Pydantic `frozen=True, extra="forbid"`
  certificate envelope; `yaml.safe_load` contract in the round-trip.
- `i3/cloud/guardrails.py` — input/output guardrail pair. API-key regex
  battery in the output path catches Anthropic / OpenAI / AWS / GitHub
  / Slack / Google prefixes. Stateless OutputGuardrail so concurrent
  share is safe.
- `i3/cloud/client.py` — positive (I-5).
- `i3/cloud/providers/anthropic.py` — wraps `CloudLLMClient`; no new
  network surface.
- `i3/cloud/providers/openrouter.py` — feeds M-4 (response.text in
  exception) and L-1 (no explicit verify/follow_redirects).
- `i3/cloud/providers/google.py` — `genai.configure(api_key=...)` is a
  module-global side effect; not a security bug but a correctness nit
  when multiple Google-keyed providers are constructed in the same
  process. Not raised as a finding.
- `i3/cloud/providers/huawei_pangu.py` — explicit `verify=True`,
  `follow_redirects=False`. Positive.
- `i3/diary/store.py` — structural privacy guarantee in the schema; FK
  enforcement turned on (PRAGMA). Positive (I-6).
- `.env.example` / `.env.providers.example` / `configs/default.yaml` —
  no hardcoded secrets, placeholder format only; model id locked to
  `claude-sonnet-4-5`.
- `Dockerfile` — multi-stage build, non-root user (uid 10001), no
  build toolchain in the runtime, `RUN curl | sh` NOT used, HEALTHCHECK
  via Python stdlib (no curl in runtime), Poetry pinned. Positive.
- `deploy/k8s/deployment.yaml` — `runAsNonRoot`, `readOnlyRootFilesystem`,
  `allowPrivilegeEscalation: false`, `capabilities: drop ALL`,
  `automountServiceAccountToken: false`, `seccompProfile: RuntimeDefault`.
  Positive.

## Coverage gaps

- **Live server not reachable** — the `torch.load(c10.dll)` issue on this
  Windows host blocked any dynamic probing. All findings are from static
  review; no active validation that (e.g.) a crafted payload to
  `/whatif/compare` actually returns before the limiter, and no timing
  measurements against the admin token's `compare_digest` path.
- **SBOM / dependency audit** — scope excluded. The verify harness's
  `config.no_hardcoded_secrets`, `config.yaml_parse_all`, `config.
  toml_parse_all` checks already pass cleanly, so the pinning surface
  is at least syntactically valid, but I did not cross-check
  `poetry.lock` against vulnerability feeds.
- **GitHub Actions workflows** — enumerated (`.github/workflows/*.yml`)
  but not audited in depth for `pull_request_target` misuse,
  `permissions: write-all`, or third-party action pinning. The six
  workflow files listed are the obvious ones to inspect in a follow-up
  pass.
- **Pipeline engine + user-model store** — called transitively by every
  route I audited, but their ~1 500 LOC were not read end-to-end. The
  route-level contracts are the load-bearing security boundary; the
  pipeline operates on already-sanitised, already-bounded inputs.
- **Helm chart / Terraform / Cedar policies** — listed in the brief but
  only `deploy/k8s/deployment.yaml` was inspected. Helm / Terraform can
  reintroduce privilege-escalation gaps that the raw Kubernetes manifest
  avoids.
- **Test-only surfaces** — per the brief, `tests/` / `scripts/` / `docs/`
  were skipped. Anything leaking via a test-only endpoint or a
  helper-script shell-out is outside the scope of this review.
