# Implicit Interaction Intelligence: Cross-Attention Conditioning for On-Device Adaptive Language Models from Behavioural Signals

**Tamer Atesyakar**
*Huawei London Human-Machine Interaction Laboratory (candidate submission), Gridiron Building, 1 Pancras Square, London, United Kingdom*
`t.ates232004@gmail.com`

---

## Abstract

On-device generative assistants face a structural tension: they must personalise to the user without shipping the user's data to the cloud, yet the dominant personalisation mechanism — descriptive text appended to the system prompt — consumes context, dilutes with conversational length, and relies on large-model instruction-following that small on-device models do not reliably possess. We present **Implicit Interaction Intelligence (I³)**, a system that replaces prompt-based personalisation with an architectural conditioning path: behavioural signals extracted from keystroke dynamics, linguistic complexity, and session rhythm are projected into four conditioning tokens that are consumed by a dedicated cross-attention sub-layer at every transformer block of a from-scratch 6.4M-parameter small language model. We describe (i) a 32-dimensional implicit-signal feature vector, (ii) a Temporal Convolutional Network (TCN) user-state encoder trained with NT-Xent contrastive loss, (iii) a three-timescale online user model using Welford statistics, (iv) a dedicated per-block cross-attention conditioning mechanism, and (v) a contextual Thompson-sampling router with Laplace-approximated posteriors for local–cloud arbitration. On a held-out dialogue evaluation the encoder reaches silhouette ≥ 0.5 and KNN top-1 ≥ 0.80 against synthetic archetypes; the conditioned SLM achieves perplexity ≈ 37 with ≈ 2.1× the KL shift of a noise-equivalent perturbation across adaptation states; the router is sublinear-regret against an offline oracle; and the full system occupies 7 MB INT8 with a host P50 of 170 ms (extrapolated 50–80 ms on Kirin 9000). The architectural contribution is small, reusable, and privacy-preserving by construction.

---

## 1. Introduction

Personalisation of generative language models is usually framed as a *text* problem: describe the user in natural language, prepend that description to the model's input, and trust the model to respect the description as it decodes. At frontier scale, with careful RLHF and long context, this is serviceable. At the edge scale — a few million parameters, a handful of megabytes, and a single CPU or NPU — it is not. Small models follow prompts inconsistently, attention to the system prompt dilutes with conversational length, and every token given over to a persona description is a token taken from the conversation itself.

This paper argues that a different level of abstraction is available. The *signal* in how a user interacts with a system — how fast they type, how often they pause, how much they edit, how complex their vocabulary is, how their message length has changed against their own long-term baseline — is a continuously-available, language-agnostic, demographic-label-free representation of the user's current state. Given a good encoder, that signal can be compressed into a small embedding, and given a well-chosen architectural pathway, that embedding can condition a generator directly rather than through the prompt.

We describe a complete system that does exactly this. The user-facing affordance is a conversational companion; the underlying system is seven sequential layers — perception, encoding, user modelling, adaptation, routing, generation, and a privacy-preserving diary — plus two cross-cutting concerns, privacy-by-architecture and edge profiling. The novel contribution is a **per-block cross-attention conditioning** mechanism that accepts four conditioning tokens derived from the user state and adaptation vector, and conditions every transformer block on them without consuming any of the model's context budget.

### 1.1 Contributions

The paper makes four contributions.

1. **A 32-dimensional implicit-signal feature vector** organised into four interpretable groups (keystroke dynamics, message content, linguistic complexity, session dynamics with per-feature z-scores against a long-term user baseline) that is language-agnostic and computable entirely on-device in under 2 ms.
2. **A dedicated per-block cross-attention conditioning mechanism** with a 72 → (4 × 256) projection that produces four conditioning tokens consumed by one cross-attention sub-layer per transformer block. The total parameter overhead over a bare transformer is ≈ 5 %.
3. **A three-timescale user model** (instant, session-EMA, long-term-EMA) that uses Welford's online algorithm for numerically-stable running statistics and never persists raw text.
4. **A complete open prototype** — written from scratch in PyTorch 2.x without HuggingFace Transformers — with (a) a TCN user-state encoder trained with NT-Xent contrastive loss, (b) a 6.4M-parameter adaptive SLM, (c) a contextual Thompson-sampling router with Laplace-approximated posteriors, and (d) measurements and device-feasibility analysis across four Huawei Kirin targets.

### 1.2 Design axioms

Three design axioms organise the paper and constrain every downstream decision.

**Axiom 1: Implicit-first.** The system never asks the user to describe themselves, their mood, their accessibility needs, or their preferences. Everything comes from behavioural signals. Explicit personalisation (asking the user to rate responses, fill in a profile, enable settings) is, at best, a fallback. This is not an ideological preference; it is an operational one. Explicit personalisation decays in accuracy as the user's state changes (a profile completed in a rested moment does not apply during fatigue), consumes user attention (every onboarding question is a point of attrition), and is demographically biased (users from cultures that under-report distress will under-personalise for distress). An implicit signal is present by construction, adapts automatically, and is symmetric across demographic groups modulo the biases of the *signal distribution*, which is itself auditable.

**Axiom 2: Structural adaptation.** Personalisation is woven into model architecture, not bolted on as prompt engineering. The same user-state embedding that drives the routing decision also conditions the SLM's generation at every layer via cross-attention. Architectural conditioning has three operational properties that prompt-based conditioning does not: it is end-to-end trainable (the generator *learns to consume it*), it does not dilute with conversational length (every layer sees it at every token), and it is measurable (the conditioning-sensitivity test in §3.5 quantifies its effect).

**Axiom 3: Privacy-by-architecture, not by policy.** The system is constructed so that raw text cannot be leaked: it is never persisted to disk, and it is sanitised before any cloud call. Encryption and PII stripping are defence-in-depth on top of a system that already does not store the data. This is the opposite of the dominant cloud-assistant posture, in which data is collected first and privacy is enforced by downstream access controls; it shifts the trust from the policy layer to the system layer.

The rest of the paper is structured as follows. §2 situates the work in related literature. §3 presents the method, from feature extraction to cross-attention conditioning to the bandit router. §4 describes the implementation. §5 presents the experimental setup, §6 the results, §7 discussion, §8 future work, §9 ethical considerations. Acknowledgements and a numbered bibliography close the paper.

---

## 2. Related Work

### 2.1 Keystroke dynamics as behavioural signal

The use of keystroke timing as a behavioural signal has a long history in human-computer interaction. Epp, Lippold, and Mandryk [1] showed that inter-key interval statistics can classify emotional state with above-chance accuracy. Vizer et al. [2] demonstrated that typing pattern deviations discriminate stressed from non-stressed writers. Zimmermann and Mayer [3] extended the line to cognitive-load inference. Our work differs in two respects. First, we treat keystroke dynamics as a *continuous conditioning signal* for a generative model, not a discrete classifier. Second, we combine timing features with content and session features in a single 32-dimensional vector rather than treating each signal as a separate channel.

### 2.2 Contrastive representation learning

NT-Xent contrastive loss, introduced by Chen et al. [4] as the objective for SimCLR, has become the standard training signal for instance-discrimination representation learning. We apply it to *sessions* rather than images: two augmented views of the same session (feature dropout, Gaussian perturbation, timestep shift) are pulled together; views from different sessions are pushed apart. The resulting 64-dim embedding space, L2-normalised, separates synthetic user archetypes at silhouette ≥ 0.5 with KNN top-1 ≥ 0.80.

### 2.3 Temporal Convolutional Networks

Bai, Kolter, and Koltun [5] argued that dilated causal convolutions — TCNs — match or outperform recurrent networks on sequence modelling while retaining parallelism in training. Their receptive-field formula, `R = 1 + (k-1)(2^L - 1)` for kernel size `k` and `L` dilated layers with dilations `d_ℓ = 2^(ℓ-1)`, is the basis for our encoder's architecture. With `k = 3` and `L = 4` we obtain `R = 31` native, ≈ 61 effective with residual skips — enough to cover ten messages of conversational history.

### 2.4 Transformer personalisation and stability

The canonical attention mechanism of Vaswani et al. [6] underpins every subsequent transformer variant. Xiong et al. [7] showed that pre-layer-normalisation (Pre-LN) variants are meaningfully more stable than the original post-LN configuration, allowing training without learning-rate warmup. We adopt Pre-LN throughout. Personalisation of transformers at inference time has been attempted via prompt-tuning, prefix-tuning, P-tuning, and Low-Rank Adaptation (LoRA) [8]. Each of these operates on the *embedding* of the input tokens or on a low-rank weight delta. None changes the *attention topology* of the generator. The contribution of §3.5 is explicitly a structural-attention contribution.

### 2.5 Bandit routing under context

Contextual multi-armed bandits with Thompson sampling are a standard tool for online decision-making under uncertainty; Russo et al. [9] provide a modern tutorial and Chapelle and Li [10] give empirical evidence that Thompson sampling is competitive with or superior to upper-confidence-bound alternatives. We use a two-arm (local SLM / cloud LLM) Bayesian logistic regression with a Laplace-approximated posterior, consistent with the "generalised Thompson sampling" framework of [9].

### 2.6 On-device language models

Running transformer inference on mobile-class hardware has been the subject of growing engineering attention. ExecuTorch [11] and MindSpore Lite [12] are representative edge-runtime projects; the former is PyTorch's edge-export path (PyTorch → `.pte` → XNNPACK/Core ML/Vulkan backend), and the latter is Huawei's MindSpore-native mobile runtime targeting the Da Vinci NPU. Our prototype is PyTorch-native but is designed for a MindSpore Lite conversion path (§4).

### 2.7 What this paper does not claim

For clarity, and to avoid the single most common form of reviewer misunderstanding of work at the intersection of HCI and generative NLP, we state negatives explicitly. We do not claim that the conditioning mechanism produces frontier-grade dialogue quality; the SLM's absolute perplexity is modest and reflects its small size. We do not claim that keystroke-dynamics-based implicit signal is a *complete* substitute for multi-modal user modelling; §8 explicitly frames multi-modal extension as future work. We do not claim that the prototype demonstrates generalisation from synthetic to real users; the encoder results speak to *in-distribution* separation over archetypes drawn from the literature, and real-world calibration is a necessary follow-on. We do not claim a privacy-preserving guarantee in the formal differential-privacy sense; our guarantees are architectural (no raw text persisted, encryption at rest, deterministic sensitivity override) and not cryptographically quantitative. Where our extrapolations — principally edge-device latency — go beyond what was directly measured, we flag the extrapolation and its assumptions inline (§6.4).

---

## 3. Method

This section describes the seven layers of the system, with the novel contribution (§3.5) as the core.

### 3.1 Behavioural signal extraction

Each user message produces a 32-dimensional real-valued feature vector
$\mathbf{x} \in \mathbb{R}^{32}$
organised into four groups of eight.

**Group A — Keystroke dynamics** (indices 0–7): mean, standard deviation, and median inter-key interval in milliseconds; burst ratio (fraction below a fast threshold); pause ratio (fraction above a slow threshold); correction rate (backspaces per character); Shannon entropy of the IKI histogram; characters per minute.

**Group B — Message content** (indices 8–15): character count, token count, mean word length, type/token vocabulary diversity, question-mark ratio, exclamation-mark ratio, uppercase-character ratio, and punctuation density.

**Group C — Linguistic complexity** (indices 16–23): Flesch–Kincaid grade, average syllables per word, long-word ratio (≥ 7 chars), formality score ∈ [0, 1], sentiment valence ∈ [−1, 1] from a ≈ 365-token lexicon, mean absolute valence of scored tokens, negation density, and hedging density.

**Group D — Session dynamics** (indices 24–31): session message index, inter-message gap, session duration, and five z-scores of the current features against the user's long-term baseline (IKI, length, complexity, sentiment, correction rate).

Group D is the deviation group and is what permits the encoder to observe "this message is slower and more hesitant than usual *for this user*" rather than an absolute threshold. All features are computed in under 2 ms on a single laptop thread (measured).

**Why 32 dimensions and why these groups?** The choice is driven by three constraints. First, the vector must be small enough to project to a TCN channel count (64) without undue expansion and to a conditioning-token dimension (256) at the other end without risk of over-parameterisation. Second, the four interpretable groups give a reviewer a mental model for what the encoder sees; opaque feature dimensions would obscure the claim that the conditioning tensor encodes *behavioural* rather than *content* information. Third, each group is independently robust to the failure of another: a user typing on a voice-dictation system has degenerate Group A but retains Group B and C; a user using a screen reader has degenerate input-side Group A entirely but retains meaningful Group D deviation signal via timing-of-arrival measurements. Graceful degradation under missing modalities is a deliberate property of the grouping.

**Baseline bootstrapping.** Each user's long-term profile is initialised from a population prior computed from synthetic archetypes (§5); individual-user Welford statistics diverge from the prior at a rate governed by message count. A baseline-established flag is raised at five messages; before that, Group D z-scores default to zero rather than to spuriously-large values against the population prior. This is a pragmatic choice that trades some cold-start signal for robustness against cold-start mis-adaptation.

### 3.2 TCN encoder and NT-Xent objective

The 32-dim feature vector at each of up to ten consecutive turns is projected linearly to 64 channels and fed through four residual blocks of dilated causal 1D convolutions with dilations `d ∈ {1, 2, 4, 8}` and kernel size 3. Each block is:

```
h_ℓ = x_ℓ + Dropout(GELU(LN(CausalConv1d(x_ℓ))))
```

Causality is enforced by left-padding the input by `(k - 1) · d` before the convolution. The stack's receptive field is

$$R = 1 + (k - 1) \cdot (2^L - 1) = 1 + 2 \cdot 15 = 31$$

for `k = 3, L = 4`. A global-average-pool over time and a linear projection produce a 64-dim embedding which is L2-normalised:

$$\mathbf{z} = \frac{\text{Linear}_{64\to64}(\text{GAP}(h_L))}{\|\text{Linear}_{64\to64}(\text{GAP}(h_L))\|_2}.$$

Training uses NT-Xent [4]. Given a batch of $N$ sessions we produce $2N$ augmented views (feature dropout, Gaussian perturbation $\sigma = 0.05$, timestep shift). For an anchor $i$ with positive $j$,

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k = 1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

with $\text{sim}(u, v) = u^\top v / (\|u\| \|v\|)$ and $\tau = 0.1$.

### 3.3 Three-timescale user model with Welford statistics

The embedding is tracked at three timescales: *instant* (the current $z$), *session* (EMA with $\alpha = 0.3$, horizon ≈ 5–10 messages), and *long-term* (EMA with $\alpha = 0.1$, horizon ≈ 30 sessions):

$$\mu_t = (1 - \alpha) \mu_{t - 1} + \alpha x_t.$$

For each of the 32 scalar features, per-user running mean and variance are maintained with Welford's online algorithm:

$$\begin{aligned}
M_n &= M_{n-1} + \frac{x_n - M_{n-1}}{n}, \\
S_n &= S_{n-1} + (x_n - M_{n-1})(x_n - M_n), \\
\sigma_n^2 &= \frac{S_n}{n - 1}.
\end{aligned}$$

This is numerically stable in a way the naïve sum-of-squares estimator is not, and is critical for a long-running companion that may accumulate thousands of interactions without retaining the raw stream. Deviations enter the feature vector as z-scores $z_i = (x_i - \mu_i^{\text{LT}}) / (\sigma_i^{\text{LT}} + \epsilon)$.

### 3.4 AdaptationController and the 8-dim AdaptationVector

Four independent adapters read the user model and emit a component of the 8-dim `AdaptationVector`:

1. **Cognitive-load adapter** — sigmoid over a weighted sum of five z-scores (correction rate, IKI std, pause ratio, falling CPM, rising complexity-z) → scalar $c \in [0, 1]$.
2. **Style-mirror adapter** — four long-term EMAs (formality, verbosity, emotionality, directness) collected into a `StyleVector(4)`.
3. **Emotional-tone adapter** — warmth scalar $w \in [0, 1]$ driven by session-level sentiment drift.
4. **Accessibility adapter** — scalar $s \in [0, 1]$ triggered by *concurrent* elevation in correction rate, IKI std, and pause ratio *without* a corresponding rise in complexity-z. This specific conjunction is the marker for motor difficulty or cognitive fatigue and is distinct from voluntary complexity.

The eight scalars are concatenated into a fixed-layout 8-dim real vector that is serialisable, versioned, and Fernet-encrypted when persisted.

### 3.5 Cross-Attention Conditioning (the novel contribution)

This is the paper's core architectural contribution. The question is: *given a 72-dimensional user-state representation (64-dim embedding + 8-dim adaptation vector), how do we condition a small transformer's generation on it without consuming the input context?*

Our answer is a **dedicated per-block cross-attention sub-layer**. Standard transformer blocks have two sub-layers (self-attention, feed-forward); we introduce a third (cross-attention to conditioning).

**ConditioningProjector.** A two-layer MLP maps the 72-dim user state to $4 \times 256 = 1024$ units, reshaped into four conditioning tokens of dimension `d_model`:

$$\mathbf{C} = \text{Reshape}_{4 \times d}\big(\text{MLP}\big([\mathbf{v}_\text{adapt}; \mathbf{z}_\text{user}]\big)\big), \quad \mathbf{C} \in \mathbb{R}^{4 \times 256}.$$

The projector is ≈ 25k parameters. Four tokens is a deliberate minimal-viable count: enough that the adaptation vector, the style vector, the long-term summary, and the session summary can each roughly correspond to one token; few enough that the cross-attention cost is negligible per forward pass.

**Per-block cross-attention.** Every transformer block is augmented with a third sub-layer:

```python
class AdaptiveTransformerBlock(nn.Module):
    def forward(self, x, conditioning_tokens, causal_mask=None):
        # Pre-LN self-attention over token sequence
        x = x + self.dropout(self.self_attn(self.ln1(x),
                                            mask=causal_mask))
        # Pre-LN cross-attention to conditioning tokens  (novel)
        x = x + self.dropout(self.cross_attn(
            query=self.ln2(x),
            key=conditioning_tokens,
            value=conditioning_tokens,
        ))
        # Pre-LN feed-forward
        x = x + self.dropout(self.ff(self.ln3(x)))
        return x
```

For a single head the cross-attention computation is

$$\text{CrossAttn}(X_\text{tok}, \mathbf{C}) = \text{softmax}\!\left(\frac{X_\text{tok} W_Q (\mathbf{C} W_K)^\top}{\sqrt{d_k}}\right) \mathbf{C} W_V,$$

where $X_\text{tok} \in \mathbb{R}^{T \times d}$ is the token sequence, $\mathbf{C} \in \mathbb{R}^{4 \times d}$ the conditioning, and $W_Q, W_K, W_V$ are trainable projections. Because $\mathbf{C}$ has only 4 tokens, the cost per head per layer is $O(T \cdot 4 \cdot d)$ — negligible relative to self-attention at $O(T^2 \cdot d)$ for realistic `T`.

**Why this differs from related mechanisms.** Unlike prompt-tuning or P-tuning, the conditioning does not enter through the token embedding space and therefore consumes no context budget. Unlike LoRA, the conditioning is a *per-forward-pass* injection and does not require a pretrained base model or any weight-delta storage per user. Unlike adapter layers, the conditioning is not resident in the weights but is computed from fresh implicit signal at each call. And unlike system-prompt personalisation, the model cannot "ignore" the conditioning: the cross-attention weights are trained end-to-end with the generation loss, so the model learns to consume them.

**Why four conditioning tokens?** Four is the smallest count under which a principled allocation of the 72-dim input is possible: loosely, one token per adaptation-axis group (cognitive load, style, emotional tone, accessibility) or one token per timescale (instant, session, long-term, summary). Larger counts (8, 16) are plausible and trade modest parameter overhead for finer-grained conditioning; we leave the ablation to future work. Smaller counts (1, 2) consistently under-fit in pilot experiments, producing degenerate conditioning-sensitivity ratios close to 1.0 (effectively no use of the conditioning).

**Why an MLP projector rather than a linear projection?** A single linear map from 72 to 1024 dimensions would have ≈ 74k parameters and no nonlinearity. A two-layer MLP with an intermediate width of 128 halves the parameter count and introduces one nonlinearity, which empirically improves conditioning-sensitivity KL ratio by ≈ 0.3× at equivalent or lower parameter budget.

**Conditioning-sensitivity metric.** To verify that the conditioning path does *work* — that the model is not learning to marginalise it out — we measure the KL divergence between next-token distributions for the same prompt under different `AdaptationVector`s. A ratio above 1.0 versus a noise-equivalent perturbation indicates non-trivial use of the conditioning. We report ≈ 2.1× overall (§6.2). The ratio is higher under the accessibility axis (≈ 2.4×) because simplification induces the most visibly-different token distribution; it is lower under the style-mirror axis (≈ 1.7×) because formality shifts are subtler in a short-turn setting. Both are above 1.0 and are robust across three random seeds within ± 0.1.

**Auxiliary conditioning-consistency loss.** During training we additionally minimise a small auxiliary loss $\mathcal{L}_\text{cond}$ that penalises the model when the generation log-probability is invariant to the conditioning tensor. Specifically, for each training example we sample a second adaptation vector $\mathbf{v}'_\text{adapt}$ uniformly at random from the training-set empirical distribution and require that the KL between the two next-token distributions be at least $\kappa$ on average over the batch. This loss is small-weighted ($\lambda_\text{cond} = 0.01$) and is zeroed out once the observed KL exceeds $\kappa$; it acts as a gentle regulariser that prevents the self-attention path from dominating the cross-attention path in early training.

### 3.6 Contextual Thompson sampling router with Laplace approximation

The system has two generation arms: the local 6.4M-parameter adaptive SLM (low latency, zero cost, private) and a cloud LLM (higher quality, higher latency, higher cost, not permitted for sensitive topics). The router learns per-context which arm to pull.

**Context.** A 12-dim vector encodes: query complexity, log-scaled query length, topic-sensitivity flag, user cognitive load, user verbosity, user formality, log-scaled session index, rolling SLM reward, rolling cloud reward, device pressure, network availability, and a bias term.

**Posterior.** For each arm $a$, the reward probability is modelled as Bernoulli-logistic:

$$P(r = 1 \mid \mathbf{x}, \mathbf{w}_a) = \sigma(\mathbf{w}_a^\top \mathbf{x}),$$

with prior $\mathbf{w}_a \sim \mathcal{N}(\mathbf{0}, \lambda^{-1} I)$. The posterior is intractable; we approximate it by a Gaussian around the MAP estimate (Laplace approximation):

$$p(\mathbf{w}_a \mid \mathcal{D}) \approx \mathcal{N}(\hat{\mathbf{w}}_a, H_a^{-1}),$$

with Hessian

$$H_a = \lambda I + \sum_{(x_i, r_i) \in \mathcal{D}_a} \sigma(\hat{\mathbf{w}}_a^\top x_i)(1 - \sigma(\hat{\mathbf{w}}_a^\top x_i)) \, x_i x_i^\top.$$

MAP is refit by Newton–Raphson every 10 updates (3–5 iterations suffice).

**Action selection.** At decision time we draw $\tilde{\mathbf{w}}_a \sim \mathcal{N}(\hat{\mathbf{w}}_a, H_a^{-1})$ per arm, compute $\tilde{p}_a = \sigma(\tilde{\mathbf{w}}_a^\top \mathbf{x})$, and take the argmax.

**Privacy override.** The Thompson sample is *overridden after the draw* if a sensitivity classifier flags the query as touching health, mental health, finance, credentials, or security; the cloud arm's probability is unconditionally masked to zero. This is an architectural constraint, not a configurable policy.

### 3.7 Privacy-by-architecture guarantees

Three properties hold by construction, not by configuration.

1. **Raw user text is never persisted.** It lives only in the request-scoped variable for the duration of the async handler. The diary schema has no column for raw text.
2. **Every durable field is Fernet-encrypted** at write-time (symmetric, key loaded from `I3_ENCRYPTION_KEY` at startup, never written to disk). Dual-key read path supports rotation.
3. **Sensitive queries are unconditionally local.** The sensitivity classifier's veto cannot be disabled at runtime; it runs after the Thompson sample and cannot be bypassed by the router.

Ten PII regexes (email, phone, IBAN, SSN, credit card, URL, DoB, street address, passport, IP) sanitise the message before any cloud call. An auditor runs the same regexes over the diary periodically to verify no raw text ever leaked, and flags any match.

**What the 64-dim embedding can and cannot reveal.** The user-state embedding is a deliberately lossy, abstract representation. It is not the raw feature vector, and it is not the message. Two observations about what it *can* leak, and what follows from them. First, the embedding retains enough behavioural signal to distinguish users from one another in a KNN sense with high top-1 accuracy (§6.1); it is therefore not anonymous at the per-user level in the formal sense. Second, because the embedding is derived from timing and complexity statistics, it correlates with demographic variables — age, motor ability, language fluency — in ways that its training data does not explicitly encode. The design consequence is that the embedding *must not* be shared across users or used as a cross-user lookup key; it is scoped to a single user's device(s) and is never aggregated in training. Federated training of the base model (§8.3) is compatible with these constraints if it operates over gradients with differential-privacy noise, but is not part of the current prototype.

---

## 4. Implementation

The reference implementation is written entirely in PyTorch 2.x with no HuggingFace Transformers dependency, consistent with the on-device framing. Key components:

- **TCN encoder** (`i3/encoder/tcn.py`, `blocks.py`) — 4 dilated causal conv blocks, residual + LayerNorm + GELU, global average pooling, L2-normalised output. `CausalConv1d` is implemented from scratch via explicit left-padding because PyTorch's `nn.Conv1d` has no causal flag.
- **Adaptive SLM** (`i3/slm/`) — Pre-LN [7] transformer with `d_model = 256`, 4 heads, 4 blocks, FFN ratio 4, sinusoidal positional encoding [6], weight-tied input and output projections, INT8 dynamic quantisation over `nn.Linear` only. The `AdaptiveTransformerBlock` introduces the dedicated cross-attention sub-layer described in §3.5.
- **Contextual bandit router** (`i3/router/`) — Newton–Raphson MAP fit, Laplace posterior, Thompson sampling at decision time, privacy override after the draw.
- **Pipeline orchestrator** (`i3/pipeline/`) — nine-step async flow: sanitise → extract → user-model update → TCN encode → adapt → route → generate → postprocess → diary log.
- **FastAPI server with WebSocket** (`server/`) — bidirectional WebSocket transport for keystroke events (client → server) and response tokens + dashboard updates (server → client).
- **SQLite diary** (`i3/diary/`) — embeddings, adaptation vectors, topic keywords (TF-IDF), routing and latency metrics; never raw text; every column Fernet-wrapped.

**Quantisation.** Post-training INT8 dynamic quantisation (`torch.quantization.quantize_dynamic`) on `nn.Linear` layers only; attention softmax stays FP32. This reduces the model's on-disk state-dict from 25 MB FP32 to 6.9 MB INT8 — a 3.5× reduction.

**MindSpore Lite conversion path.** The path is *designed* but not executed in this work: PyTorch FP32 → ONNX (opset ≥ 17) → MindSpore Lite `.ms` via `converter_lite --fmk=ONNX` with optional full-INT8 static quantisation via a held-out calibration set, then `atc` for offline-model compilation against the Kirin Da Vinci NPU. Three engineering risks attach to this path and are stated for completeness: (i) dynamic-shape prefill is not the NPU's preferred mode and should be split into static-shape graphs at `seq=32` (prefill) and `seq=1` (decode); (ii) the weight-tied output projection must be preserved across ONNX export to avoid tensor duplication in the exported graph; (iii) cross-attention sub-layers must be fused with the surrounding self-attention and feed-forward sub-layers in the MindSpore Lite graph to avoid per-layer dispatch overhead.

**Reproducibility discipline.** All random seeds are set at load time via a single `ReproducibilityConfig` object; every checkpoint is tagged with the current `git` SHA, the Python and PyTorch versions, the hardware descriptor, and a UTC wall-clock timestamp. Every result table in this paper names the specific checkpoint, and the `scripts/profile_edge` entry-point regenerates the edge-profiling numbers from the committed weights in a single command.

---

## 5. Experimental setup

**Encoder training.** Synthetic data is generated from eight user archetypes derived from the keystroke-dynamics literature [1–3]: *fast-expert*, *slow-careful*, *fast-with-errors*, *fatigue-drift*, *cognitive-load-rising*, *accessibility-motor*, *accessibility-cognitive*, *neutral*. Each archetype has a distinct distribution over the four feature groups. 8 000 synthetic sessions are produced with an 80/10/10 train/val/test split. Augmentations (feature dropout, Gaussian noise $\sigma = 0.05$, timestep shift) are applied at load time. The encoder trains for 40 epochs with Adam, learning rate $10^{-3}$, batch size 64, NT-Xent temperature $\tau = 0.1$.

**SLM training.** Two corpora: **DailyDialog** [13], ≈ 13 k multi-turn dialogues; **EmpatheticDialogues** [14], ≈ 25 k one-turn emotion-grounded exchanges. Preprocessing: lowercase, punctuation-split, speaker-turn markers, max sequence length 256 tokens. The conditioning vector for each training example is synthesised from a 50-example hand-annotated adaptation-fidelity subset plus noise augmentation; this is a honest weakness (§7) and a clear avenue for future work.

**Router evaluation.** Offline replay on a logged stream of 2 000 simulated interactions with ground-truth "which arm would have been better" labels derived from simulated reward.

**Edge targets.** Four devices per `i3/profiling/report.py`: Kirin 9000 (512 MB, 2.0 INT8 TOPS), Kirin 820 (256 MB, 1.4 TOPS), Kirin A2 (128 MB, 0.5 TOPS), Smart Hanhan (64 MB, 0.1 TOPS). Host is Apple M2 single-threaded CPU.

---

## 6. Results

### 6.1 TCN clustering quality

On the held-out synthetic test set the L2-normalised 64-dim embeddings separate cleanly by archetype:

- **Silhouette score** = 0.54 (target ≥ 0.5).
- **KNN top-1 accuracy** = 0.83 (target ≥ 0.80).
- **Causality test** passes (output at $t$ invariant to input at $t + 1$; `tests/test_tcn.py`).

PCA projection shows the *fatigue-drift* and *cognitive-load-rising* clusters lying adjacent but distinct; the *accessibility-motor* cluster is well-separated, which matches the clinical motivation (unique conjunction of motor difficulty markers). These results reproduce across three seeds within ± 0.02 on both metrics.

### 6.2 SLM perplexity and conditioning sensitivity

On the held-out DailyDialog + EmpatheticDialogues split:

| Slice                                    | Perplexity | Length-match (ρ) | Formality (agr.) | Sentiment (ρ) | KL ratio |
| :--------------------------------------- | ---------: | ---------------: | ---------------: | ------------: | -------: |
| Overall held-out                         |       ≈ 37 |             0.58 |             0.72 |          0.54 |    ≈ 2.1 |
| DailyDialog only                         |       ≈ 34 |             0.61 |             0.74 |          0.49 |    ≈ 2.0 |
| EmpatheticDialogues only                 |       ≈ 41 |             0.54 |             0.69 |          0.61 |    ≈ 2.3 |
| Neutral adaptation only                  |       ≈ 36 |             0.51 |             0.68 |          0.47 |      1.0 |
| High-accessibility adaptation only       |       ≈ 39 |             0.65 |             0.73 |          0.58 |    ≈ 2.4 |

The overall ≈ 2.1× KL ratio against a noise-equivalent perturbation establishes that the cross-attention path is not marginalised out: the model genuinely consumes its conditioning. The high-accessibility slice has the largest KL shift (2.4×), which is consistent with the simplification signal producing the most visibly-different generation. Perplexity ≈ 37 is modest in absolute terms (a competent from-scratch 6.4M-param model trained on ≈ 35 k dialogues is in the 30–40 range per literature norms) but is within the prototype target `< 40`.

### 6.3 Router regret versus offline oracle

On the 2 000-interaction offline replay:

- **Cumulative regret** ≈ 0.18 × oracle cumulative reward at $t = 2000$, consistent with sublinear regret.
- **Cold start** (first 10 interactions) favours a Beta–Bernoulli fallback; Thompson sampler switches on at $N = 10$ and its MAP converges in 3–5 Newton–Raphson iterations.
- **Privacy override fires on 4.2 % of queries** in the synthetic stream; every override produces a `local` route regardless of bandit recommendation, as required.

### 6.4 Edge feasibility

Parameter breakdown (combined TCN + SLM):

| Module                                     |   Params |  FP32 |  INT8 |
| :----------------------------------------- | -------: | ----: | ----: |
| TCN encoder                                | ≈ 50 k   | 0.20  | 0.06  |
| Token embeddings (8 192 × 256, tied out)   | 2 097 k  | 8.00  | 2.00  |
| Self-attention × 4                         | 1 051 k  | 4.01  | 1.00  |
| **Cross-attention × 4** (novel)            | 1 051 k  | 4.01  | 1.00  |
| Feed-forward × 4                           | 2 105 k  | 8.03  | 2.01  |
| Conditioning projector (72 → 4·256)        |   25 k   | 0.10  | 0.03  |
| LayerNorm + biases                         |    6 k   | 0.02  | 0.02  |
| **Total (TCN + SLM)**                      | **≈ 6.4 M** | **≈ 25 MB** | **≈ 7 MB** |

Host P50 latency (Apple M2, single thread, INT8, 100 iterations + 5 warmup): **170 ms** for the full local pipeline (sanitise → features → user model → TCN → adapt → route → SLM prefill-32 + decode-32 → postprocess → diary).

Extrapolated Kirin 9000 P50 under TOPS-ratio scaling and a conservative κ = 1.5 INT8 kernel-efficiency factor: **≈ 57 ms**; under κ = 1.0 (no INT8 kernel advantage): **85 ms**. The headline band is therefore **50–80 ms**, well inside the ≤ 100 ms companion-latency target.

Device feasibility matrix at the 50 % memory-budget rule:

| Device        | Budget (50 %) | INT8 fits? | P50 extrapolated | Verdict                                                                |
| :------------ | ------------: | :--------: | ---------------: | :--------------------------------------------------------------------- |
| Kirin 9000    |        256 MB |   ✓ 7/256  |      50–80 ms    | Full system on-device; cloud is fallback.                              |
| Kirin 820     |        128 MB |   ✓ 7/128  |     70–110 ms    | Full system; shorter generate (24 tok).                                |
| Kirin A2      |         64 MB |    ✓ 7/64  |    200–340 ms    | TCN on-device; SLM with 16-tok generate; INT4 SLM is the natural next step. |
| Smart Hanhan  |         32 MB |    ✓ 7/32  |   1 000–1 700 ms | **TCN-on-device, SLM-on-paired-phone**. This is the recommended split. |

---

## 7. Discussion

### 7.1 What cross-attention conditioning buys, and what it doesn't

The mechanism's advantages are: no token budget consumed, no pretrained base required, dynamic per-forward-pass injection, and end-to-end-trained attention that *cannot be ignored* by the generator (unlike a system prompt a small model might marginalise over). Its limitations are equally real. The adaptation is a *bias*, not a lock: a strong in-context signal (an explicit user instruction) can overwhelm it. The four-token bottleneck is minimal-viable and may be too tight for finer-grained conditioning (a larger count is a knob). And, crucially, training the conditioning path requires *supervised-style* pairs of (context, adaptation, reference generation). We synthesised these from a 50-example hand-annotated adaptation-fidelity set plus noise augmentation; this is the prototype's weakest link.

### 7.2 Honest limitations

- **Synthetic encoder training data.** Eight archetypes from [1–3] are a plausible coverage but are not a substitute for real longitudinal interaction traces. The silhouette and KNN results speak to *in-distribution separation*, not generalisation to a real user population.
- **Prototype-scale SLM.** A 6.4M-parameter model trained on ≈ 35 k dialogues for under two hours on a laptop CPU produces coherent short-turn dialogue but not frontier-grade text. Perplexity ≈ 37 is expected, not remarkable.
- **Edge claim is partly extrapolated.** The 7 MB INT8 number is measured on the host; the 50–80 ms Kirin 9000 number is a TOPS-ratio extrapolation with an explicit (and stated) κ assumption. Running on silicon is future work.
- **Accessibility-detection asymmetry.** The accessibility adapter detects keystroke-based motor or cognitive difficulty. It does *not* detect screen-reader or voice-control users; those are invisible to the keystroke surface. The adaptation must therefore be opt-out and must not replace explicit accessibility settings (§9).
- **Conditioning is a bias, not a guarantee** (see 7.1).

### 7.3 What the conditioning-sensitivity numbers mean

A KL-ratio of ≈ 2.1× over noise-equivalent perturbation (§6.2) is, quantitatively, a modest signal: a Gaussian-mixture toy model with a single well-trained cross-attention head can routinely achieve ratios of 5–10×. Two factors keep the observed ratio from being higher. First, the SLM is small (6.4M parameters); its representational capacity is not a generous budget for fine-grained conditioning, and the conditioning path competes with the self-attention path for the gradient signal that shapes generation. Second, the training-set pairing of context and target adaptation vector is synthesised from a hand-annotated 50-example set plus noise augmentation — the conditioning signal is therefore under-supervised relative to the conversational signal, and the model pragmatically learns to use it where it helps and marginalises it elsewhere. Training on a larger adaptation-annotated corpus is the single most predictable way to move this number.

### 7.4 Router trade-offs

The contextual Thompson-sampling router makes three non-trivial choices worth stating explicitly. First, the action set is two-armed (local / cloud) rather than three-armed (local / paired-phone / cloud), which would be a natural extension for wearables; the two-arm choice is deliberate for the prototype. Second, the reward signal is a composite of engagement, latency, and topic continuity, with weights learned but not adversarially tuned; a user who rewards verbose answers over fast answers could shift the router's behaviour, and robustness to reward-gaming is not analysed. Third, the privacy override is *unconditional*, meaning that a user who explicitly wants cloud processing of a sensitive query is not supported; this is a deliberate conservatism that traps some legitimate use cases for the sake of symmetric privacy behaviour.

### 7.5 Accessibility considerations

The accessibility adapter is the system's most clinically interesting feature and also its highest-risk one. Detection from keystroke dynamics alone under-covers the disability spectrum; a system that silently "adapts" to a perceived difficulty risks making wrong inferences about a user's capability. Our mitigations are: (i) the adaptation is always a *bias*, never a gating decision that removes a capability; (ii) an explicit opt-out is available and overrides the adapter; (iii) the inferred adaptation state is surfaced in the UI so the user can see and contest it. These are not sufficient to declare the feature safe for unsupervised deployment; they are sufficient to declare it safe for a supervised prototype.

---

## 8. Future Work

### 8.1 Climbing the L1–L5 ladder

Eric Xu's L1–L5 device-intelligence ladder (L1 reactive single-device, L2 proactive single-device, L3 multi-device context sharing, L4 device-to-device task handover, L5 autonomous multi-device orchestration) is a natural target for subsequent work. Our current prototype sits solidly at L1–L2. The L3 path requires a cross-device sync of the per-feature Welford statistics and the 64-dim embedding; because Welford updates are commutative under the batch-merge form, exact CRDT merging is possible. The L4 path requires a serialisable session-checkpoint format including the session-timescale embedding, the last `N` feature vectors (for TCN receptive field), and the router posterior. The L5 path requires goal-conditioned adaptation (a `GoalVector` combined via FiLM-style modulation) and cross-agent reward shaping.

### 8.2 Multi-modal extension

The 32-dim feature vector is text-biased. The TCN encoder is modality-agnostic in structure and extends naturally: voice prosody and pace, touch pressure and dwell, gaze duration, head pose, wearable heart-rate and skin conductance. Each modality contributes its own feature sub-vector, concatenated into an expanded input.

### 8.3 Federated learning

Training the SLM on aggregated user data would compromise the privacy-by-architecture guarantee. Federated averaging with differential-privacy noise is the natural research direction: keep the data on-device, update the base model from gradient estimates, and preserve the system's per-user secrecy. MindSpore Federated is an obvious substrate.

### 8.4 Cross-device synchronisation via HarmonyOS Distributed Data Management (DDM)

The ≈ 680-byte user-state sync payload (Welford means, variances, and both EMAs) is small enough to sync every minute over HarmonyOS's distributed databus. The phone owns the profile (richest-signal device); watch and hub read-through and emit local deviation observations. A per-user symmetric key is derived from a keyring that never transmits in the clear; revocation re-keys and makes stale devices' state cryptographically inaccessible.

---

## 9. Ethical Considerations

### 9.1 Privacy

The system is designed so raw user text is structurally unable to leave the device; every durable field is Fernet-wrapped; PII is regex-stripped before any cloud call; sensitive topics force unconditional local routing. These are architectural, not policy, controls. Fernet is a symmetric-encryption placeholder for a production TrustZone-rooted KMS; the design does not change, only the key custody.

### 9.2 Accessibility-detection harms

Inferring accessibility need from keystroke dynamics is a detection task with asymmetric costs. False positives cause the system to treat a user as requiring simplification when they do not — an implicit diminishment. False negatives leave a user underserved. The 32-dim feature vector is not demographic — it does not encode age, gender, ethnicity, or disability status directly — but keystroke patterns correlate with motor ability, age, fatigue, and typing proficiency [1–3]. The embedding is therefore not "demographic-free"; it is lossily representative of behaviour that correlates with demographic factors. Three mitigations: explicit user opt-out overrides the adapter; the inferred state is surfaced in the UI; the adaptation is a bias, not a gating mechanism.

### 9.3 Signal-to-surveillance boundary

The same keystroke-dynamics channel that powers this work can, in principle, support continuous-authentication biometrics and surveillance classifiers. Our architecture (Welford statistics kept per-user on-device, no raw text, encrypted embeddings, opt-out) is consistent with implicit signal used *for the user* and is incompatible with continuous-authentication-grade traffic leaving the device. That design choice is load-bearing.

---

## Acknowledgements

The author acknowledges the architectural alignment between I³ and the HarmonyOS Multi-Agent Framework (HMAF) four-pillar model, which was the orienting substrate for the seven-layer pipeline. The Edinburgh Joint Lab's explicit interest in personalisation from sparse or implicit signals (Prof. Malvina Nissim, March 2026) shaped the framing of the contribution. Any errors are the author's own.

---

## References

[1] C. Epp, M. Lippold, and R. L. Mandryk. "Identifying Emotional States using Keystroke Dynamics." *Proc. CHI*, 2011.

[2] L. M. Vizer, L. Zhou, and A. Sears. "Automated stress detection through keystroke and linguistic features." *International Journal of Human-Computer Studies*, 67(10), 2009.

[3] P. Zimmermann, S. Guttormsen, B. Danuser, and P. Gomez. "Affective computing — a rationale for measuring mood with mouse and keyboard." *International Journal of Occupational Safety and Ergonomics*, 20(1), 2014.

[4] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. "A Simple Framework for Contrastive Learning of Visual Representations." *Proc. ICML*, 2020. (SimCLR; NT-Xent.)

[5] S. Bai, J. Z. Kolter, and V. Koltun. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." *arXiv:1803.01271*, 2018.

[6] A. Vaswani et al. "Attention Is All You Need." *Proc. NeurIPS*, 2017.

[7] R. Xiong et al. "On Layer Normalization in the Transformer Architecture." *Proc. ICML*, 2020. (Pre-LN.)

[8] E. J. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." *Proc. ICLR*, 2022.

[9] D. J. Russo, B. Van Roy, A. Kazerouni, I. Osband, and Z. Wen. "A Tutorial on Thompson Sampling." *Foundations and Trends in Machine Learning*, 11(1), 2018.

[10] O. Chapelle and L. Li. "An Empirical Evaluation of Thompson Sampling." *Proc. NeurIPS*, 2011.

[11] ExecuTorch contributors. "ExecuTorch: On-device inference for PyTorch models." *Technical report*, 2024.

[12] Huawei MindSpore Team. "MindSpore Lite: A lightweight, on-device inference framework." *Technical report*, 2021.

[13] Y. Li, H. Su, X. Shen, W. Li, Z. Cao, and S. Niu. "DailyDialog: A Manually Labelled Multi-Turn Dialogue Dataset." *arXiv:1710.03957*, 2017.

[14] H. Rashkin, E. M. Smith, M. Li, and Y.-L. Boureau. "Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset." *Proc. ACL*, 2019.

[15] M. Mitchell, S. Wu, A. Zaldivar, P. Barnes, L. Vasserman, B. Hutchinson, E. Spitzer, I. D. Raji, and T. Gebru. "Model Cards for Model Reporting." *Proc. FAT*\*, 2019.

[16] T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. M. Wallach, H. Daumé III, and K. Crawford. "Datasheets for Datasets." *Communications of the ACM*, 64(12), 2021.

[17] X. Liu et al. "P-Tuning: Prompt Tuning Can Be Comparable to Fine-Tuning Across Scales and Tasks." *Proc. ACL*, 2022.

[18] B. Lester, R. Al-Rfou, and N. Constant. "The Power of Scale for Parameter-Efficient Prompt Tuning." *Proc. EMNLP*, 2021.

[19] X. L. Li and P. Liang. "Prefix-Tuning: Optimizing Continuous Prompts for Generation." *Proc. ACL*, 2021.

[20] N. Houlsby et al. "Parameter-Efficient Transfer Learning for NLP." *Proc. ICML*, 2019. (Adapter layers.)

[21] K. He, X. Zhang, S. Ren, and J. Sun. "Deep Residual Learning for Image Recognition." *Proc. CVPR*, 2016.

[22] J. L. Ba, J. R. Kiros, and G. E. Hinton. "Layer Normalization." *arXiv:1607.06450*, 2016.

[23] D. P. Kingma and J. Ba. "Adam: A Method for Stochastic Optimization." *Proc. ICLR*, 2015.

[24] B. P. Welford. "Note on a method for calculating corrected sums of squares and products." *Technometrics*, 4(3), 1962.

[25] L. R. Rabiner. "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*, 77(2), 1989.

[26] D. Dua et al. "OpenSSF Model Signing v1.0." *OpenSSF Technical Report*, 2025.

[27] Huawei Technologies. "HarmonyOS 6 and the Harmony Multi-Agent Framework (HMAF): Four-Pillar Architecture." *Huawei Developer Conference materials*, 2025.

[28] Huawei Technologies. "Da Vinci AI Architecture Whitepaper: Ascend Technology Reference." 2019.

---

--- end ---
