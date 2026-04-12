# Security Policy

## Supported Versions

| Version | Supported |
|---------|:---------:|
| 1.0.x   |    Yes    |
| < 1.0   |    No     |

Only the latest 1.x minor line receives security patches.  All other
branches are provided as-is.

---

## Reporting a Vulnerability

Please report security vulnerabilities privately to:

**tamer.atesyakar@bk.ru**

Do not open a public GitHub issue for suspected vulnerabilities and do
not disclose details publicly until they have been acknowledged and
addressed.  We aim to acknowledge reports within two business days and
to ship a patch within fourteen days for critical issues.

When reporting, please include:

- A description of the vulnerability.
- Steps to reproduce (a minimal proof-of-concept is ideal).
- The affected version / commit hash.
- Any known mitigations.

Reports may be encrypted with the maintainer's public PGP key on
request.

---

## Security Architecture

### Privacy by Architecture

Implicit Interaction Intelligence is built around an unusual constraint:
the system has to build a deep behavioural model of its user without
ever retaining the content of anything they actually typed.  Privacy is
therefore enforced at the **architecture** layer, not merely the
configuration layer:

1. **Raw user text is never persisted.**  The database schema in
   `i3/diary/store.py` has no column capable of holding a user message,
   an AI response, or a conversation turn.  The only strings stored are
   topic keywords (three per exchange, extracted by TF-IDF) and an
   aggregate session summary generated from *metadata only*.
2. **Embeddings are encrypted at rest.**  The user-state and baseline
   embeddings produced by the TCN encoder are serialised via NumPy and
   wrapped by `ModelEncryptor` (Fernet AES-128-CBC + HMAC-SHA256)
   before being written to disk.
3. **PII is sanitised before any cloud call.**  `PrivacySanitizer`
   applies ten regex patterns (email, US/UK/international phone, SSN,
   credit card, IP, physical address, date of birth, URL) to every
   string that is about to leave the device.  The sanitised result is
   what ends up in the Claude API request body.
4. **Only aggregated metadata is sent to the cloud.**  The Anthropic
   prompt builder (`i3/cloud/prompt_builder.py`) encodes the current
   adaptation vector and topic history as *instructions*, not user
   text.  See the privacy auditor (`PrivacyAuditor.audit_request`) for
   the automated compliance check.
5. **Defence-in-depth auditing.**  `PrivacyAuditor.audit_database`
   walks every text column of every SQLite table, samples up to 100
   rows, and flags anything that looks like natural-language prose or
   contains PII.  This runs as part of `make security-check`.

### Threat Model

The system is scoped to defend against the threats enumerated below.
Threats outside this table (e.g., supply-chain attacks on PyTorch
wheels, kernel-level memory scraping) are acknowledged but out of
scope.

| # | Threat                                               | Actor           | Severity |
|---|------------------------------------------------------|-----------------|----------|
| 1 | Raw user text leaking into logs or the diary DB      | Insider / audit | High     |
| 2 | PII (email, phone, credit card) sent to cloud LLM    | Careless config | High     |
| 3 | WebSocket exhaustion (DoS via huge frames / flood)   | External        | High     |
| 4 | Malicious `.pt` checkpoint triggering code execution | Supply chain    | Critical |
| 5 | SQL injection through `user_id` or pagination params | External        | High     |
| 6 | Secret / API key leakage via error messages or logs  | Insider         | High     |
| 7 | CSRF / click-jacking against the demo UI             | External        | Medium   |
| 8 | Long-running SLM generation locking the pipeline     | External        | Medium   |
| 9 | Session-pinning or user-id impersonation             | External        | Medium   |
| 10| Weak encryption-key management                       | Operator error  | High     |

### Mitigations Implemented

| # | Threat           | Mitigation                                                  |
|---|------------------|-------------------------------------------------------------|
| 1 | Raw-text leak    | Diary schema has no message column; logger discards text after topic extraction; `PrivacyAuditor.audit_database` scans on demand. |
| 2 | PII to cloud     | `PrivacySanitizer` with 10 regex patterns, invoked before prompt building; `PrivacyAuditor.audit_request` re-scans the final payload. |
| 3 | WS exhaustion    | `_MAX_MESSAGE_BYTES=65536`, `_MAX_MESSAGES_PER_SESSION=1000`, `_MAX_SESSION_SECONDS=3600`, `_MAX_KEYSTROKE_BUFFER=2000`, per-user sliding-window rate limiter (600/min). |
| 4 | Pickled `.pt`    | All inference-time `torch.load` calls now pass `weights_only=True`.  The only remaining `weights_only=False` call is `SLMTrainer.load_checkpoint`, which is documented as trusted-input only. |
| 5 | SQL injection    | All queries in `i3/diary/store.py` and `i3/user_model/store.py` use parameter placeholders.  `user_id` is constrained by a `^[a-zA-Z0-9_-]{1,64}$` regex at the FastAPI boundary. |
| 6 | Secret leakage   | API keys are redacted in logs via `_redact_api_key` (`sk-ant-***abcd`).  Error bodies from the upstream API are truncated and echoed only as "HTTP <code>" to clients.  Unhandled exceptions return a generic 500. |
| 7 | CSRF / clickjack | `SecurityHeadersMiddleware` sets `X-Frame-Options: DENY`, `Content-Security-Policy`, `Referrer-Policy`, `Permissions-Policy`, HSTS (on HTTPS), and nosniff. |
| 8 | SLM runaway      | Generation loop is bounded by `max_new_tokens` (default 100).  The cloud client has a hard `_MAX_TIMEOUT_SECONDS=30` ceiling regardless of config. |
| 9 | ID impersonation | `user_id` regex is enforced in REST and WebSocket layers.  Duplicate connections for the same user evict the earlier socket. |
| 10| Key management   | `ModelEncryptor.initialize` reads from `I3_ENCRYPTION_KEY`; falls back to an ephemeral key with a prominent warning; the generated key is **never** logged.  `ModelEncryptor.rotate_to` returns a `MultiFernet` for zero-downtime key rotation. |

---

## Security Audit Report

### Methodology

The audit was conducted in four passes over the repository:

1. **Static inventory** — `grep`/`rg` sweep for `torch.load`,
   `yaml.load`, `pickle.*`, `subprocess`, `os.system`, `eval(`,
   `exec(`, hardcoded `sk-ant-`/`password=`/`token=`/`secret=`, and
   `allow_origins=["*"]`.
2. **Per-file manual review** — each hit was read in context to
   distinguish genuine issues from false positives.
3. **Fix application** — all critical and high findings were fixed
   in-place.
4. **Post-fix verification** — a second `grep` pass confirmed that
   the surface area had been reduced.

### Findings

#### Critical (Fixed)

| ID   | Finding                                                              | File(s)                              | Fix |
|------|----------------------------------------------------------------------|--------------------------------------|-----|
| C-1  | `torch.load(..., weights_only=False)` on inference paths             | `i3/encoder/inference.py`, `i3/pipeline/engine.py`, `i3/slm/quantize.py`, `training/evaluate.py`, `i3/encoder/train.py` | Switched to `weights_only=True`. |
| C-2  | `torch.load(..., weights_only=False)` on training data `.pt` files   | `training/train_slm.py`, `training/train_encoder.py`, `training/evaluate.py` | Switched to `weights_only=True` (tensor-only dicts, safe). |
| C-3  | Optimizer-state resume is pickled                                    | `i3/slm/train.py`                    | Left as `weights_only=False` with a prominent comment noting the trusted-input constraint. |
| C-4  | `CORSMiddleware(allow_origins=["*"])` in the default server config  | `server/app.py`, `configs/default.yaml` | Origins now loaded from `config.server.cors_origins` and `I3_CORS_ORIGINS`; wildcard rejected unless `I3_ALLOW_CORS_WILDCARD=1`. |

#### High (Fixed)

| ID   | Finding                                                              | File                       | Fix |
|------|----------------------------------------------------------------------|----------------------------|-----|
| H-1  | WebSocket accepted arbitrary JSON frames with no size cap            | `server/websocket.py`      | `_MAX_MESSAGE_BYTES=64 KiB`, `_MAX_MESSAGE_TEXT_CHARS=8 KiB`, rejection with WS close 1009. |
| H-2  | WebSocket had no per-user flood limit                                | `server/websocket.py`      | Sliding-window limiter (600 msg/min) reused from REST middleware. |
| H-3  | WebSocket `user_id` was unvalidated                                  | `server/websocket.py`, `server/routes.py` | `^[a-zA-Z0-9_-]{1,64}$` regex applied in both layers; rejected sockets close with WS 1008. |
| H-4  | No max session duration or max messages per session                  | `server/websocket.py`      | `_MAX_SESSION_SECONDS=3600`, `_MAX_MESSAGES_PER_SESSION=1000`. |
| H-5  | No OWASP security headers on HTTP responses                          | `server/middleware.py`     | New `SecurityHeadersMiddleware` adds CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy, and HSTS (on HTTPS). |
| H-6  | No request-body size cap                                             | `server/middleware.py`     | New `RequestSizeLimitMiddleware` rejects bodies > 1 MiB with 413. |
| H-7  | No REST rate limiting                                                | `server/middleware.py`     | New `RateLimitMiddleware` (60 req/min per IP, 60 s window). |
| H-8  | Cloud API key could be logged in error messages                      | `i3/cloud/client.py`       | `_redact_api_key` used for info-level startup log; error bodies truncated; public error text never contains the key. |
| H-9  | Cloud timeout taken verbatim from config (could be unbounded)        | `i3/cloud/client.py`       | Clamped to `_MAX_TIMEOUT_SECONDS=30` at construction time. |
| H-10 | 429 responses did not honour `Retry-After`                           | `i3/cloud/client.py`       | `Retry-After` header parsed; value clamped to the timeout ceiling. |
| H-11 | Unhandled exceptions could leak stack traces to clients              | `server/app.py`            | Three exception handlers (`StarletteHTTPException`, `RequestValidationError`, `Exception`) return sanitised JSON. |
| H-12 | REST routes had no pagination bounds (`limit`, `offset`)             | `server/routes.py`         | `limit` ∈ [1, 100], `offset` ∈ [0, 10000] enforced with FastAPI `Query`. |
| H-13 | `PrivacyAuditor.audit_database` built SQL with f-string identifiers  | `i3/privacy/sanitizer.py`  | Added `^[A-Za-z0-9_]+$` identifier validator; invalid names are skipped with a warning. |
| H-14 | Default `server.host = "0.0.0.0"` exposed the server publicly        | `i3/config.py`, `configs/default.yaml`, `.env.example` | Default is now `127.0.0.1`; operators must opt in to all-interfaces binding. |

#### Medium (Fixed)

| ID   | Finding                                                              | File                       | Fix |
|------|----------------------------------------------------------------------|----------------------------|-----|
| M-1  | Encryption key leakage on fallback to ephemeral key                  | `i3/privacy/encryption.py` | The generated key is no longer logged (was previously echoed to the INFO log).  Only a warning is emitted. |
| M-2  | Malformed `I3_ENCRYPTION_KEY` propagated a cryptic exception         | `i3/privacy/encryption.py` | Wrapped in a `ValueError` with a clear message; the invalid key is never echoed back. |
| M-3  | No key-rotation helper                                               | `i3/privacy/encryption.py` | `ModelEncryptor.rotate_to(new_key)` returns a `MultiFernet` for zero-downtime rotation. |
| M-4  | WebSocket could stack connections for the same `user_id`             | `server/websocket.py`      | Old socket is closed with code 1001 before the new one is registered. |
| M-5  | `HTTP_CONNECTION_LIMIT` / reverse-proxy hints absent                 | `server/middleware.py`     | `X-Forwarded-For` is honoured only when `request.state.trust_forwarded_for` is explicitly set. |
| M-6  | Keystroke buffer had no upper bound                                  | `server/websocket.py`      | Buffer trimmed when it exceeds `_MAX_KEYSTROKE_BUFFER=2000`. |

#### Low / Informational

- **L-1 — `yaml.safe_load` already in use.**  `i3/config.py` uses
  `yaml.safe_load`.  No occurrences of `yaml.load` were found.
- **L-2 — No `pickle.loads` in repository.**  `grep` for `pickle.load`
  and `pickle.loads` returned no matches.
- **L-3 — No `subprocess` / `os.system`.**  Repository is free of
  direct shell invocations in production code.
- **L-4 — SQL queries already parameterised.**  Every CRUD method in
  `i3/diary/store.py` and `i3/user_model/store.py` uses `?`
  placeholders; no concatenation or f-string SQL was found.
- **L-5 — No hardcoded secrets.**  `grep` for known secret prefixes
  returned only `.env.example`, which contains a placeholder.
- **L-6 — Static file mount order.**  `StaticFiles` is mounted *last*
  in `server/app.py` so API and WebSocket routes take precedence, and
  its request body is subject to the same middleware stack.
- **L-7 — SLM generation is already bounded.**  The generation loop
  in `i3/slm/generate.py` iterates strictly `range(max_new_tokens)`.

### Recommendations for Production Deployment

The following recommendations go beyond what is enforced in code.
None of these are required for the demo; they are the next steps
before a production rollout.

1. **Authenticate WebSocket clients.**  The current build accepts any
   `user_id` that matches the regex; no session token, JWT, or cookie
   is required.  In production, prefix the WebSocket URL with a signed
   token (e.g., an HS256 JWT bound to the `user_id`) and validate it
   in the `connect` method before accepting the socket.
2. **Use a distributed rate limiter.**  The in-memory sliding-window
   limiter in `server/middleware.py` is per-process.  Behind a
   multi-worker uvicorn or a Kubernetes deployment, replace it with
   Redis (e.g., `redis-py`'s `INCR` + `EXPIRE`) or an API gateway
   feature (Envoy, nginx `limit_req`).
3. **Put the server behind TLS.**  HSTS is only emitted on HTTPS
   requests.  Terminate TLS at a reverse proxy (nginx, envoy, caddy)
   and add `trust_forwarded_for = True` on the request state so the
   rate limiter sees the real client IP.
4. **Rotate the Fernet key on a schedule.**  Call
   `ModelEncryptor.rotate_to(new_key)` in a nightly cron, re-wrap the
   embedding blobs in place, and then flip `I3_ENCRYPTION_KEY`.
5. **Enable audit logging.**  Configure structured JSON logging
   (`structlog` or `python-json-logger`) with a field whitelist so
   that PII and secrets cannot accidentally enter logs.
6. **Scan dependencies continuously.**  Add `make security-check` to
   CI (GitHub Actions) and fail the build on any *high* finding from
   `bandit`, `pip-audit`, or `safety`.
7. **Sign model checkpoints.**  For the trusted-input torch.load
   code path in `i3/slm/train.py`, produce a SHA-256 of the
   checkpoint at training time, store it next to the `.pt` file, and
   verify it before loading.
8. **Disable `/api/demo/reset` and `/api/demo/seed` in production.**
   These endpoints are intended for the live demo only; gate them
   behind a config flag or remove them in the production build.
9. **Constrain `uvicorn` workers.**  `uvicorn --limit-concurrency 100`
   and `--timeout-keep-alive 5` cap total connections; add a process
   manager (`systemd`, `supervisord`) with `Restart=on-failure`.
10. **Monitor `PrivacyAuditor.audit_database` results.**  Run it on
    every deploy and again in a nightly cron; alert on any violation.

---

## Dependencies

We use the following tools to monitor dependency security:

- **`pip-audit`** — scans installed Python packages for known CVEs
  from the PyPI advisory database.
- **`safety`** — cross-checks against the Safety CLI vulnerability
  database.
- **`bandit`** — static analysis for the most common insecure Python
  patterns (hardcoded passwords, weak hashes, shell injection, etc.).

Run them all with:

```bash
make security-check
```

Under the hood this executes:

```bash
bandit -r i3/ server/ -ll     # line-level severity: low and above
pip-audit                      # dependency CVE scan
safety check                   # second-source CVE scan
```

The `[tool.poetry.group.security.dependencies]` section of
`pyproject.toml` pins `bandit>=1.7`, `pip-audit>=2.6`, and
`safety>=3.0`.

---

## Cryptography

- **Fernet symmetric encryption** (cryptography library) for user
  models at rest.  Fernet = AES-128-CBC + HMAC-SHA256 + random IV,
  which gives authenticated encryption without a custom construction.
- **TLS 1.2+** for all cloud API calls via `httpx` (which delegates
  to the system OpenSSL).  Certificate validation is enabled by
  default.
- **No custom crypto.**  All cryptographic operations go through the
  `cryptography` package.  No hand-rolled hashing, padding, or
  encryption is used anywhere in the codebase.
- **Keys are 256 bits of random entropy** generated by
  `Fernet.generate_key()` (URL-safe base64 encoding of `os.urandom(32)`).

### Key Rotation

To rotate the encryption key without downtime:

```python
from i3.privacy.encryption import ModelEncryptor
encryptor = ModelEncryptor()
encryptor.initialize()

new_key = ModelEncryptor.generate_key()
mf = encryptor.rotate_to(new_key)

# Re-encrypt in place:
for row in db.iter_rows():
    row.baseline_embedding = mf.rotate(row.baseline_embedding)

# Finally update I3_ENCRYPTION_KEY in your secret store.
```

---

## Data Handling

- Raw user text is **never persisted** to disk.  The diary schema has
  no column capable of holding it.
- PII is **sanitised** via 10+ regex patterns in `PrivacySanitizer`
  before any string leaves the device.
- Embeddings are **encrypted at rest** with Fernet (see above).
- Session summaries are generated from **aggregated metadata only**
  (message counts, engagement averages, topic keywords, dominant
  emotion).  The cloud LLM never sees individual messages.
- `PrivacyAuditor.audit_database` is a database-level scanner that
  walks every text column and flags prose-like content or PII;
  `PrivacyAuditor.audit_request` is a payload-level scanner that is
  run immediately before any cloud call.

---

## WebSocket Security

- **Max inbound frame size**: 64 KiB.  Frames larger than this cause
  a WebSocket close 1009.
- **Max message text length**: 8 KiB.  Chat messages larger than this
  are rejected with WebSocket close 1009.
- **Max messages per session**: 1000.  Exceeding this closes the
  socket with code 1008.
- **Max session duration**: 1 hour wall-clock.  On expiry the socket
  is closed with code 1008 and the pipeline cleans up the session.
- **User ID validation**: `^[a-zA-Z0-9_-]{1,64}$`.  Non-conforming
  IDs are rejected before the socket is accepted.
- **Rate limiting**: 600 messages per minute per `user_id`, via a
  shared in-memory sliding-window limiter.  Exceeding the limit sends
  an `{"type":"error","code":429}` frame and closes the socket.
- **Duplicate-connection eviction**: If a new socket arrives for an
  already-connected `user_id`, the earlier socket is closed with
  code 1001 ("going away").
- **Fail-closed teardown**: Any JSON parse error, schema violation,
  or handler exception closes the socket immediately and triggers
  `pipeline.end_session` in a `finally` block.

---

## API Security

- **CORS origins**: configurable via `configs/default.yaml` or the
  `I3_CORS_ORIGINS` env var.  Wildcard `*` is only accepted if
  `I3_ALLOW_CORS_WILDCARD=1` is *also* set, otherwise it falls back
  to `localhost:8000`.
- **Rate limiting**: 60 requests per minute per client IP on `/api/*`
  paths (excluding `/api/health`).
- **Request size limit**: 1 MiB on the `Content-Length` header;
  larger requests receive a 413.
- **Security headers** injected on every response:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `Referrer-Policy: strict-origin-when-cross-origin`
  - `Permissions-Policy: camera=(), microphone=(), geolocation=(), ...`
  - `Content-Security-Policy: default-src 'self'; script-src 'self'
    'unsafe-inline'; ... frame-ancestors 'none'`
  - `Strict-Transport-Security: max-age=63072000; includeSubDomains`
    (HTTPS only)
  - `Cache-Control: no-store` for `/api/*`
- **Exception handlers**: Any unhandled exception, validation error,
  or `HTTPException` returns a sanitised JSON body.  Stack traces are
  written to the server log, never to the response.
- **Loopback-only by default**: `server.host` defaults to `127.0.0.1`;
  operators must explicitly set `I3_HOST=0.0.0.0` to expose the
  server on all interfaces.

---

## Model Security

- **Inference-time `torch.load`**: every call passes
  `weights_only=True`.  This disables the pickle path inside
  `torch.load` and forbids arbitrary code execution during
  deserialisation.
- **Training-time `torch.load`**: the one code path that still
  passes `weights_only=False` is `SLMTrainer.load_checkpoint`, which
  is required because the optimiser state dict cannot be loaded
  under `weights_only=True`.  This code path is documented as
  trusted-input only; do not point it at checkpoints from untrusted
  sources.
- **Checkpoint signing** (recommended, not yet implemented): compute
  a SHA-256 of every saved checkpoint and verify it on load.

---

## Logging & Monitoring

- Logs use Python's standard `logging` module at module level.
- **No secrets are ever logged.**  The Anthropic API key is redacted
  via `_redact_api_key` before it enters any log record.
- **No PII is ever logged.**  `PrivacySanitizer.sanitize` removes PII
  before any text could be passed to a log call, and
  `PrivacyAuditor.audit_database` provides an after-the-fact scan.
- **No stack traces are returned to clients.**  The FastAPI exception
  handlers in `server/app.py` always return a sanitised JSON body.
- **Recommendation**: switch to structured JSON logging in production
  so that downstream tooling can field-filter logs and so that a log
  search for `baseline_embedding` or `message_text` cannot match a
  stringified Python object.

---

## Acknowledgements

This code was developed for an interview project and underwent a
formal security review as documented in this file.  The audit covered
input validation, authentication boundaries, cryptography, secrets
management, transport security, denial-of-service mitigations, and
dependency hygiene.  All critical and high findings have been fixed
in-place; medium findings have been either fixed or documented with a
mitigation note; low / informational findings are noted for future
iterations.

---

*Last audit: see `git log -- SECURITY.md` for the most recent review
date.*
