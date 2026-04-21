# Runbook — Incident Response

Step-by-step playbooks for the seven incident classes that can take down,
degrade, or compromise an I³ instance. Each scenario is structured as
**Symptom → Detection → Triage → Remediation → Verification → Postmortem
checklist**.

!!! warning "Escalation paths"
    - **Privacy incidents** → stop the service immediately. Privacy is the
      product.
    - **Data loss risk** → snapshot first, act second.
    - **Cloud cost blow-up** → cap first, diagnose second.

## Scenario index { #index }

1. [Privacy auditor triggered](#s1-privacy-auditor)
2. [Router stuck on cloud (or local)](#s2-router-stuck)
3. [Pipeline P95 breach](#s3-latency-breach)
4. [Cloud LLM error spike](#s4-cloud-errors)
5. [WebSocket flood / DoS attempt](#s5-ws-flood)
6. [SQLite corruption / profile loss](#s6-sqlite-corruption)
7. [Fernet key rotation / loss](#s7-fernet-rotation)

---

## S1 · Privacy auditor triggered { #s1-privacy-auditor }

### Symptom

Alert `I3PrivacyAuditorHit` fires. `i3_privacy_auditor_hits_total > 0`.

### Detection

- Alert from Prometheus.
- Log lines: `level=CRITICAL event=privacy.auditor.hit`.

### Triage

1. **Stop the service** immediately to prevent further writes.
   ```bash
   kubectl -n i3 scale deploy/i3 --replicas=0
   ```
2. Snapshot the SQLite DB for forensics:
   ```bash
   kubectl -n i3 cp <pod>:/app/data/profiles.sqlite ./profiles-$(date -u +%FT%TZ).sqlite
   ```
3. Note the `entry_id` and `trace_id` from the CRITICAL log line.

### Remediation

1. Identify the offending writer via the trace:
   ```
   SELECT span_name, attributes FROM spans WHERE trace_id = '…';
   ```
2. Patch the writer. Most likely causes:
   - A logger format string leaked a user message.
   - A new adapter started persisting a raw field.
3. Purge the offending rows:
   ```sql
   DELETE FROM diary WHERE entry_id IN (…);
   ```
4. Redeploy with the patch. Verify auditor is clean before scaling up.

### Verification

- `i3_privacy_auditor_hits_total` stable at 0 for 24 h.
- `tests/test_privacy.py` passes with the patched writer.

### Postmortem checklist

- [ ] Incident timeline posted internally.
- [ ] Added a regression test that reproduces the leak pattern.
- [ ] Reviewed every writer added in the same release window.
- [ ] Disclosed per security policy if any production data was affected.

---

## S2 · Router stuck on cloud (or local) { #s2-router-stuck }

### Symptom

Alert `I3RouterStuckOnCloud` or `I3RouterStuckOnLocal` fires. One arm's
share > 90 % for 15 minutes.

### Detection

- Prometheus alert.
- `i3_router_decisions_total{arm="cloud"} / i3_router_decisions_total{} > 0.9`.

### Triage

1. Inspect the Laplace posterior:
   ```bash
   curl -s .../metrics | grep i3_router_posterior_mean
   ```
2. Check the privacy-override share — a spike means a keyword list change
   pushed everything to local.
3. Inspect recent reward distribution — if reward stalled at 0 for one
   arm, the posterior will converge there.

### Remediation

| Cause | Action |
|:------|:-------|
| Laplace blew up (posterior mean extreme) | Reset: `DELETE FROM router_state WHERE user_id = '…'` (falls back to Beta-Bernoulli) |
| Reward signal dead (all zeros) | Check the reward computer in `i3/router/router.py`; validate engagement feed |
| Cost config wrong | Tune `alpha_cost` in `configs/default.yaml::router` |
| Privacy keyword list change | Revert or narrow the list |

### Verification

- Route share returns to configured prior (e.g. 65/35) within 30 min.
- No router exceptions in logs.

### Postmortem checklist

- [ ] Added a test for the degenerate posterior case.
- [ ] Confirmed reward pipeline is end-to-end tested.
- [ ] Considered widening `prior_variance` for slower convergence.

---

## S3 · Pipeline P95 breach { #s3-latency-breach }

### Symptom

`I3PipelineP95Over500ms` fires.

### Detection

- Prometheus.
- Feedback: users report "feels sluggish".

### Triage

1. Dashboard: which layer inflated?
   - Encoder — unlikely. Small model, CPU is dominated elsewhere.
   - SLM — most likely. Check `i3_slm_generation_latency_seconds`.
   - Cloud — check `i3_cloud_latency_seconds` and `i3_cloud_retries_total`.
2. Check CPU throttling (K8s `cpu_cfs_throttled_seconds_total`).
3. Check concurrency (`i3_session_active`, `i3_requests_in_flight`).

### Remediation

| Cause | Action |
|:------|:-------|
| CPU saturation | Scale out (`kubectl scale`) or bump `resources.limits.cpu` |
| Cloud latency regression | Fail back to local via config `router.force_local=true` temporarily |
| Quantization regression | Re-enable INT8 if disabled; verify `slm.quantize_int8=true` |
| Memory pressure (swap) | Bump `resources.limits.memory` |

### Verification

- P95 back below the 200 ms target for 30 min.
- No OOM events in pod events.

### Postmortem checklist

- [ ] If SLM regressed, compare perplexity and latency vs last-known-good
      checkpoint.
- [ ] Added a CI gate if one was missing.

---

## S4 · Cloud LLM error spike { #s4-cloud-errors }

### Symptom

`I3CloudErrorRate > 10 %` for 10 minutes.

### Triage

1. Check Anthropic status page.
2. `kubectl logs` for `cloud.complete` errors; count by `status`.
3. Validate `ANTHROPIC_API_KEY` still works:
   ```bash
   curl -sH "x-api-key: $ANTHROPIC_API_KEY" \
     https://api.anthropic.com/v1/messages ...
   ```

### Remediation

| Cause | Action |
|:------|:-------|
| Upstream outage | Flip router to `force_local=true` in config, redeploy |
| Expired / revoked key | Rotate: update Secret + `kubectl rollout restart` |
| Rate limit at Anthropic | Reduce concurrency, implement a queue, or negotiate capacity |
| Client-side bug | Inspect `tests/test_cloud.py`; hotfix + release |

### Verification

- Cloud error rate under 1 % for 30 min.
- User-facing complaints cleared.

---

## S5 · WebSocket flood / DoS attempt { #s5-ws-flood }

### Symptom

`i3_rate_limit_exceeded_total` spikes; connection count rises without
engagement-score movement; `_CLOSE_POLICY_VIOLATION` close codes surge.

### Triage

1. Identify offending origins / IPs via access log.
2. Check CPU / memory — is the server still stable?
3. Inspect `_ws_rate_limiter` state; confirm caps are enforcing.

### Remediation

1. Drop abusive IP ranges at the ingress (nginx `deny`, NetworkPolicy,
   WAF).
2. Tighten `I3_CORS_ORIGINS` — remove any `*`.
3. Reduce `_MAX_MESSAGES_PER_SESSION` / `_MAX_SESSION_SECONDS` in
   `server/websocket.py` if abuse is sophisticated.
4. Enable body-size limit at the ingress (128 KiB recommended).

### Verification

- Abuse traffic shed; legitimate sessions unaffected.
- Limiter metrics stabilise.

### Postmortem checklist

- [ ] Reviewed whether the cap sizes are still right.
- [ ] Considered adding Fail2Ban-style automated blocking.
- [ ] Added alert on `i3_rate_limit_exceeded_total` rate > baseline × 10.

---

## S6 · SQLite corruption / profile loss { #s6-sqlite-corruption }

### Symptom

Pipeline start-up fails: `sqlite3.DatabaseError: database disk image is malformed`.
Or: `UserProfile` decryption raises `InvalidToken`.

### Triage

1. **Stop the service**.
2. Snapshot `profiles.sqlite` to a safe location.
3. Attempt integrity check:
   ```bash
   sqlite3 profiles.sqlite "PRAGMA integrity_check;"
   ```

### Remediation

| Cause | Action |
|:------|:-------|
| Minor corruption | `sqlite3 profiles.sqlite ".recover" \| sqlite3 profiles-recovered.sqlite` |
| Fernet key mismatch | See [S7](#s7-fernet-rotation) |
| Filesystem failure | Restore from the latest backup (Velero / cron) |
| Concurrent write from a crashed pod | Enable WAL mode (`PRAGMA journal_mode=WAL`) + single-writer |

### Verification

- Service starts cleanly.
- `/user/{user_id}/stats` returns data for a known user.
- `i3_privacy_auditor_hits_total` remains 0 after recovery.

---

## S7 · Fernet key rotation / loss { #s7-fernet-rotation }

### Symptom

- **Rotation**: you intentionally want to change keys.
- **Loss**: `cryptography.fernet.InvalidToken` on profile load.

### Rotation procedure

1. Generate the new key:
   ```bash
   python scripts/generate_encryption_key.py > new.key
   ```
2. Assemble a `MultiFernet` key chain: `new_key,old_key` (new first).
3. Deploy with `I3_ENCRYPTION_KEY=<new>,<old>`.
4. Run the background re-encryptor:
   ```bash
   python -m i3.privacy.encryption --rotate
   ```
5. Remove the old key from the list; redeploy.

### Loss procedure

The key cannot be recovered. The encrypted rows are gone. Steps:

1. Accept the loss of historical `UserProfile`s.
2. Deploy with a fresh key.
3. Profiles rebuild from the next session's observations (cold start).
4. Do **not** attempt to read stale rows — they will corrupt the fresh DB
   if you re-insert ciphertext.

### Postmortem checklist

- [ ] Back-up policy documented and tested.
- [ ] Key stored in two independent secrets stores.
- [ ] Rotation rehearsed at least quarterly.

---

## Reference — alert → section { #alert-map }

| Alert | Section |
|:------|:--------|
| `I3PrivacyAuditorHit`       | [S1](#s1-privacy-auditor) |
| `I3RouterStuckOnCloud`      | [S2](#s2-router-stuck) |
| `I3RouterStuckOnLocal`      | [S2](#s2-router-stuck) |
| `I3PipelineP95Over500ms`    | [S3](#s3-latency-breach) |
| `I3CloudErrorRate`          | [S4](#s4-cloud-errors) |
| `I3RateLimitStorm`          | [S5](#s5-ws-flood) |
| `I3SQLiteUnhealthy`         | [S6](#s6-sqlite-corruption) |
| `I3FernetInvalid`           | [S7](#s7-fernet-rotation) |

## Further reading { #further }

- [Observability](observability.md)
- [Troubleshooting](troubleshooting.md)
- [SECURITY.md](https://github.com/abailey81/implicit-interaction-intelligence/blob/main/SECURITY.md)
