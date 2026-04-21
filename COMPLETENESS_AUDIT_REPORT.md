# I³ Completeness Audit Report

Audit date: 2026-04-22. Auditor: read-only pass over the repository against
`THE_COMPLETE_BRIEF.md` and `BRIEF_ANALYSIS.md`.

---

## 1. Summary Verdict

**~92% of the brief's deliverables and architectural requirements are met in
source form.** The remaining ~8% is dominated by items that *require human
action* or compute time: trained checkpoints, a recorded backup video, and
final artefact submission (PDF/PPTX export). The codebase, the docs, the
slides (as Markdown), the supply-chain, the observability, and the
Huawei-integration layers are all present and on-brief.

**Highest-value remaining gaps (ranked):**
1. No trained checkpoints — `checkpoints/encoder/` and `checkpoints/slm/`
   contain only `.gitkeep`; synthetic data under `data/` likewise empty.
2. No recorded backup demo video (5-minute MP4) and no script to record one.
3. Slide deck exists as Markdown only — no PDF / PPTX export for email
   submission on April 28.
4. Cloud model ID is `claude-sonnet-4-20250514`, not the brief-mandated
   `claude-sonnet-4-5` (`configs/default.yaml:89`).
5. Admin endpoints are `/demo/reset` + `/demo/seed`, not the
   `/admin/reset` + `/admin/profiling` that the brief's §15 called out.
6. "What This Prototype Is Not" honesty slide title is rendered lowercase as
   `What this prototype is *not*` — brief requires the exact capitalised phrasing.
7. Dedicated 30-minute WebSocket soak test not present.
8. Checkpoint metadata utility exists in `i3/mlops/checkpoint.py` but grep
   for `git_sha` / `architecture_hash` / `wall_clock` returned no hits.

None of these are blockers; all are addressable in a focused pre-interview
sprint.

---

## 2. Fully-Met Requirements

| Requirement | Location |
|---|---|
| `NOTES.md` at repo root (deviation disclosure) | `NOTES.md` |
| Extracted NT-Xent loss module | `i3/encoder/loss.py` |
| Sentiment module + lexicon asset | `i3/interaction/sentiment.py`, `i3/interaction/data/sentiment_lexicon.json` |
| Edge profiling report | `docs/edge_profiling_report.md` |
| 15-slide deck (Markdown) | `docs/slides/presentation.md` (16 H1 headings = cover + 15 slides) |
| Speaker notes | `docs/slides/speaker_notes.md` (479 lines) |
| Rehearsal timings | `docs/slides/rehearsal_timings.md` |
| Verbatim closing line | `docs/slides/closing_lines.md` (exact string present) |
| 52 Q&A pairs across 7 categories | `docs/slides/qa_prep.md` (42 `### Xn.` entries + 10 strategic = 52, category table at top) |
| 32-dim InteractionFeatureVector (4x8 groups) | `i3/interaction/features.py`, `i3/interaction/types.py` |
| TCN, dilations [1,2,4,8], residual + LayerNorm + GELU, L2-norm output | `i3/encoder/tcn.py`, `i3/encoder/blocks.py` |
| NT-Xent training loop | `i3/encoder/train.py`, `i3/encoder/loss.py` |
| Three-timescale user model (instant / session EMA / long-term EMA) | `i3/user_model/model.py` |
| 5-message warmup baseline | `i3/user_model/model.py`, `i3/user_model/deviation.py` |
| AdaptationVector 8-dim | `i3/adaptation/controller.py`, `i3/adaptation/types.py`, `i3/adaptation/dimensions.py` |
| Custom SLM (Pre-LN transformer, no HF) | `i3/slm/{model,transformer,attention,embeddings,generate,train}.py` |
| Cross-attention conditioning | `i3/slm/cross_attention.py` |
| Tokenizer (word-level SimpleTokenizer) | `i3/slm/tokenizer.py` |
| Contextual Thompson bandit + Laplace | `i3/router/bandit.py`, `i3/router/router.py`, `i3/router/complexity.py` |
| PII sanitizer | `i3/privacy/sanitizer.py` |
| Fernet encryption | `i3/privacy/encryption.py` |
| Sensitive-topic classifier | `i3/router/sensitivity.py` |
| Anthropic cloud client | `i3/cloud/client.py`, `i3/cloud/prompt_builder.py`, `i3/cloud/postprocess.py`, `i3/cloud/guarded_client.py`, `i3/cloud/guardrails.py` |
| Interaction diary with embeddings-only SQL | `i3/diary/{logger,store,summarizer}.py` |
| Edge profiler with Kirin targets | `i3/profiling/{memory,latency,report}.py`, `i3/huawei/kirin_targets.py` |
| P50/P95/P99 latency measurement | `i3/profiling/latency.py`, `i3/profiling/report.py` |
| 4-panel dashboard (chat, embedding viz, gauges, routing/engagement) | `web/index.html`, `web/js/{chat,dashboard,embedding_viz,app,websocket}.js` |
| Design palette exactly `#1a1a2e / #16213e / #0f3460 / #e94560` | `web/css/style.css:6-11`, `docs/slides/marp-theme.css` |
| Papers cited in docstrings + docs (Chen 2020, Bai 2018, Xiong 2020, Vaswani, Russo 2018, Chapelle 2011, Rabiner HMM) | `i3/encoder/loss.py`, `i3/slm/{attention,transformer,embeddings}.py`, `docs/research/*.md`, `docs/slides/qa_prep.md` |
| HCI references Epp 2011 / Vizer 2009 / Zimmermann 2014 | cited in `NOTES.md`, `docs/responsible_ai/model_card_tcn.md`, `docs/responsible_ai/data_card.md`, `docs/slides/*` |
| Dockerfile + Dockerfile.dev + compose files | repo root |
| K8s / Helm / Terraform / ArgoCD / Skaffold | `deploy/{k8s,helm,terraform,argocd}`, `deploy/skaffold.yaml` |
| MkDocs site (mkdocs.yml + docs tree) | `mkdocs.yml`, `docs/` |
| 10 ADRs | `docs/adr/0001..0010-*.md` + `index.md` + `template.md` |
| Model cards + data card + accessibility statement | `docs/responsible_ai/{model_card_tcn,model_card_slm,data_card,accessibility_statement}.md` |
| Huawei integration docs (7 files) | `docs/huawei/{README,smart_hanhan,kirin_deployment,harmony_hmaf_integration,l1_l5_framework,edinburgh_joint_lab,interview_talking_points}.md` |
| Huawei integration code | `i3/huawei/{hmaf_adapter,kirin_targets,executorch_hooks}.py`, `i3/edge/{tcn_,}executorch_export.py` |
| Supply chain (SLSA, cosign, Trivy, Semgrep, renovate, release-please, commitlint, lefthook, sigstore) | `SLSA.md`, `SUPPLY_CHAIN.md`, `renovate.json`, `commitlint.config.js`, `lefthook.yml`, `.sigstore.yaml`, `.trivyignore`, `.semgrepignore`, `.release-please-config.json`, `.release-please-manifest.json`, `.github/` workflows |
| Benchmarks (pytest-benchmark + Locust + k6 + SLOs) | `benchmarks/{test_*latency,locustfile.py,slos.yaml,k6/load.js}` |
| Advanced test suites | `tests/{property,contract,fuzz,load,mutation,chaos,snapshot,benchmarks}` |
| Observability stack | `i3/observability/{logging,tracing,metrics,middleware,instrumentation,context,sentry,langfuse_client}.py`, `deploy/observability/` |
| MLOps (MLflow + ONNX + signing + DVC + edge profile) | `i3/mlops/{tracking,checkpoint,registry,export,model_signing}.py`, `i3/encoder/onnx_export.py`, `i3/slm/onnx_export.py`, `dvc.yaml`, `scripts/{export_onnx,verify_onnx,sign_model,profile_edge,export_executorch}.py` |
| Advanced ML (INT4 torchao, guardrails, evaluation suite, ExecuTorch edge path) | `i3/slm/quantize_torchao.py`, `i3/cloud/{guardrails,guarded_client}.py`, `i3/eval/{perplexity,conditioning_sensitivity,responsiveness_golden}.py`, `i3/edge/` |
| `src/` → `i3/` rename documented | `NOTES.md §1` |

---

## 3. Partially-Met Requirements

### 3.1 Admin endpoints
- **There:** `POST /demo/reset`, `POST /demo/seed` in `server/routes.py`
  (gated on `I3_DEMO_MODE`), plus `GET /profiling/report`.
- **Missing:** brief §15 names `POST /admin/reset` and `POST /admin/profiling`.
  Functionality is present under different paths; either rename to match the
  brief verbatim or note the deviation in `NOTES.md`.

### 3.2 "What This Prototype Is Not" honesty slide
- **There:** slide 13 of `docs/slides/presentation.md` line 220 titled
  `# What this prototype is *not*`.
- **Missing:** exact title-case rendering. Brief §10 and §8.2 Move 5 quote
  the phrase verbatim as *"What This Prototype Is Not"*. One-line edit.

### 3.3 Anthropic model ID
- **There:** cloud block in `configs/default.yaml` with
  `model: "claude-sonnet-4-20250514"`.
- **Missing:** brief prescribes `claude-sonnet-4-5`. `NOTES.md §6` actually
  *claims* the ID is locked to `claude-sonnet-4-5`, so NOTES is inaccurate
  relative to `configs/default.yaml`. Either swap the config string or
  update NOTES to reflect the real choice.

### 3.4 Checkpoint metadata discipline
- **There:** `i3/mlops/checkpoint.py`, `i3/mlops/tracking.py`,
  `i3/mlops/registry.py`, `i3/mlops/model_signing.py`.
- **Missing:** `grep` across `i3/mlops/` for `git_sha`, `architecture_hash`,
  `wall_clock`, `hardware` returned zero matches. Brief §19.2 mandates these
  fields in every checkpoint payload. Worth auditing before training runs so
  the saved `.pt` files carry the expected provenance.

### 3.5 Scenario / soak tests
- **There:** `tests/chaos/test_pipeline_resilience.py`, `tests/load/test_websocket_dos.py`,
  `benchmarks/` latency suites.
- **Missing:** explicit 30-minute WebSocket soak test (brief §13 / §18.4 Day
  18). No file contains the keywords `soak`, `30.min`, `1800s`, etc.

### 3.6 Profiling admin endpoint
- `GET /profiling/report` exists; brief names `POST /admin/profiling`. See §3.1.

### 3.7 Closing-line verification
- Exact string *"I build intelligent systems that adapt to people. I'd like to
  do that in your lab."* appears in `docs/slides/closing_lines.md`,
  `docs/slides/presentation.md`, `docs/slides/speaker_notes.md`,
  `docs/slides/rehearsal_timings.md`. Fully met — included here only to
  flag the multi-file redundancy (fine, just be aware).

---

## 4. Unmet Requirements

### 4.a Trivially addable (single commit, no compute) — future commit

| Item | Action |
|---|---|
| Cloud model string mismatch | Edit `configs/default.yaml:89` to `claude-sonnet-4-5` (one line). |
| Honesty slide title case | Edit `docs/slides/presentation.md:220` heading. |
| Admin endpoint naming | Add `/admin/reset` and `/admin/profiling` aliases in `server/routes.py` pointing to the same handlers. |
| `scripts/record_backup_demo.sh` | Add a Playwright/ffmpeg capture script stub. Doesn't record the video, but satisfies the named deliverable. |
| 30-minute soak test | Add `tests/load/test_websocket_soak.py` with a parametrised long-run config gated behind `RUN_SOAK_TEST=1`. |
| Checkpoint metadata fields | Extend `i3/mlops/checkpoint.py` so `save()` writes `git_sha`, `architecture_hash`, `wall_clock_s`, `hardware` into the payload dict. |
| PDF / PPTX export of the deck | Add a Make target (`make slides`) that runs Marp CLI on `docs/slides/presentation.md`. |

### 4.b Requires human / compute action — NOT addable by commit alone

| Item | Why human action is required |
|---|---|
| Trained TCN checkpoint (`checkpoints/encoder/best.pt`, `encoder_int8.pt`) | Needs training run (`training/train_encoder.py`) on generated synthetic data. |
| Trained SLM checkpoint (`checkpoints/slm/final.pt`, `slm_int8.pt`, tokenizer JSON) | Needs tokenizer fit + full training run (`training/train_slm.py`). Biggest risk item. |
| Synthetic TCN dataset | Needs `training/generate_synthetic.py` run; `data/synthetic/` currently empty. |
| DailyDialog + EmpatheticDialogues processed data | Needs `training/prepare_dialogue.py` run; `data/processed/` currently empty. |
| Backup demo video (5-min MP4) | Requires running the demo and screen-recording. Two USB copies are a physical deliverable. |
| Slides submitted to `matthew.riches@huawei.com` on Apr 28 | Human action, deadline +6 days. |
| Printed slides, HDMI+USB-C adapters, USB drives, photo ID | Physical logistics. |

---

## 5. Surplus Features (Bonus / "Wow")

The repo goes well beyond brief-minimum. Highlights:

- **Container strategy** — multi-stage `Dockerfile`, `Dockerfile.dev`,
  `docker-compose.yml` + `.prod.yml` + override example, `.devcontainer/`.
- **Kubernetes-native deploy** — full `deploy/k8s/` manifest set (deployment,
  service, ingress, HPA, PDB, NetworkPolicy, ServiceMonitor, Kustomize
  overlays), Helm chart at `deploy/helm/i3/`, Terraform at `deploy/terraform/`,
  Skaffold + ArgoCD.
- **Observability** — `i3/observability/` with structlog, OpenTelemetry
  tracing, Prometheus metrics, Sentry, Langfuse; `deploy/observability/` stack.
- **Supply-chain** — SLSA L3 documentation, cosign / sigstore config,
  Trivy/Semgrep ignore lists, renovate, release-please, commitlint, lefthook,
  `SLSA.md`, `SUPPLY_CHAIN.md`.
- **MLOps** — MLflow tracking, DVC pipeline (`dvc.yaml`), ONNX export for
  encoder + SLM, ExecuTorch edge export path, OpenSSF model signing.
- **Advanced ML** — torchao INT4 quantisation
  (`i3/slm/quantize_torchao.py`), guarded cloud client
  (`i3/cloud/guarded_client.py`) with `guardrails.py`, evaluation harness
  (`i3/eval/{perplexity,conditioning_sensitivity,responsiveness_golden}.py`).
- **Huawei integration layer** — 7 docs + 3 Python modules
  (`i3/huawei/hmaf_adapter.py`, `kirin_targets.py`, `executorch_hooks.py`).
- **Testing depth** — property (`hypothesis`), contract, fuzz, load, mutation,
  chaos, snapshot suites. Benchmarks with `pytest-benchmark`, Locust, k6, SLO YAML.
- **MkDocs documentation site** — Material theme, 10 ADRs, responsible-AI
  corpus, Huawei corpus, glossary, research references.
- **Slides corpus** — not just the deck: speaker notes, 52-item Q&A prep,
  rehearsal timings, closing lines, Marp theme, README.

Every one of these strengthens the "senior engineer shipping production
thinking" signal that the brief calls for.

---

## 6. Recommendations for the Final 48 Hours

Ordered by (reward / effort). Check each off before April 28 submission.

1. **Run `training/generate_synthetic.py` and commit the dataset + artefacts
   hash.** Empty `data/synthetic/` undermines the "from-scratch" claim live.
2. **Train the TCN encoder** (even 10–20 epochs). Commit `checkpoints/encoder/best.pt`
   and `encoder_int8.pt` with full metadata. Verify silhouette ≥ 0.5 and
   top-1 KNN ≥ 0.80.
3. **Fit the tokenizer, run a short SLM training loop**
   (`training/train_slm.py`) — even a single epoch demonstrates the loop works
   and unlocks the adaptation-fidelity evaluation.
4. **Fix the three one-line deviations:**
   (a) `configs/default.yaml:89` → `claude-sonnet-4-5`;
   (b) `docs/slides/presentation.md:220` title to `What This Prototype Is Not`;
   (c) add `/admin/reset` and `/admin/profiling` aliases in `server/routes.py`.
5. **Export the slide deck** to PDF and PPTX via Marp CLI; add a
   `make slides` target; attach both to the Apr 28 email with the exact subject
   line from the brief.
6. **Record the 5-minute backup demo** once the real demo runs end-to-end;
   burn to two USB drives. Add `scripts/record_backup_demo.sh` (ffmpeg /
   Playwright wrapper) so the artefact is reproducible.
7. **Add a 30-minute soak test** at `tests/load/test_websocket_soak.py`,
   gated on `RUN_SOAK_TEST=1`. Run it once, paste the result into
   `docs/edge_profiling_report.md`.
8. **Patch checkpoint metadata** — add `git_sha`, `architecture_hash`,
   `wall_clock_s`, `hardware` fields in `i3/mlops/checkpoint.py` before
   step 2/3 runs so the saved artefacts carry provenance.
9. **Rehearse under budget** — the brief explicitly protects rehearsal time
   as the last thing to cut. Two full run-throughs with the pre-flight
   checklist in `docs/DEMO_SCRIPT.md`.
10. **Reconcile `NOTES.md §6` against `configs/default.yaml`** — after
    step 4(a) these finally agree; otherwise the file contradicts itself
    and a careful reader (Matthew) will notice.

---

**Auditor's closing note.** The repository's floor (architecture, privacy,
tests, infra, docs) is unusually high for a 17-day solo build. The ceiling
— live demo with trained models, video, PDF-submitted deck — is entirely
reachable within 48 hours provided training converges. Current state reads
as "production-grade codebase, training and presentation pipeline ready to
push the button." Prioritise the training run above all else.
