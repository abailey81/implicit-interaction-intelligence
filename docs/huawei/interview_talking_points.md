# Interview Talking Points — I³ × Huawei HMI Lab

> **Thesis.** This is the cheat sheet. Not a document. Not marketing. A
> cheat sheet you re-read the morning of the interview. Every section is
> optimised to be scannable under pressure, not pretty.

---

## 0. One line about *why* you built this

> "I wanted to build the thing the HMI Lab would actually hire someone
> to build — a small-language-model personalisation stack that runs on
> Kirin, plugs into HMAF, and has privacy baked into its architecture."

If you only say one sentence all interview, say that one.

---

## 1. The 60-second project pitch

Memorise this cold. Deliver it conversationally, not as a recitation.

> "I³ — Implicit Interaction Intelligence — is an on-device
> personalisation stack with three language-model arms. It learns the
> user from *how* they type — keystroke rhythm, word-choice variety,
> session drift — never from what they explicitly say about themselves.
> It compresses that into a 64-dim user-state embedding via a 50 KB TCN
> encoder, turns that into an 8-dim adaptation vector, and conditions a
> **204 M-parameter from-scratch custom transformer** (the chat arm —
> 12 layers, 12 heads, MoE + ACT, BPE 32 k vocab) via cross-attention
> at every token position. A second arm — **Qwen 1.7 B + LoRA** — is
> the deterministic JSON intent parser for vehicle commands
> (`set_timer`, `play_music`, `navigate`, …) with `val_loss = 5.4×10⁻⁶`
> on a 4 545 / 252 split. A third optional arm — **Gemini 2.5 Flash via
> Google AI Studio** — is the OOD-command safety net; when Qwen flunks
> a phrasing the cascade routes to Gemini and a slot-normaliser maps
> the result back to the canonical schema. A contextual Thompson
> sampling bandit learns local-vs-cloud routing per user, per message,
> with a privacy override that forces local on sensitive topics. The
> from-scratch chat arm is the differentiator; the cascade is what
> makes the demo robust to phrasings nobody trained for."

Beats to hit: implicit-first, structural conditioning, edge-feasible,
HMAF-native, privacy-by-architecture. In that order.

---

## 2. Per-layer 30-second technical one-liners

| Layer | One-liner |
|:---|:---|
| **1. Perception** | "32-dim feature vector, four groups of eight: keystroke dynamics, message content, linguistic complexity, session dynamics. Regex-stripped for PII before extraction." |
| **2. Encoding** | "4-layer dilated causal TCN, 50 K parameters, exponentially growing receptive field to 31 timesteps. Trained with NT-Xent contrastive loss on augmented session pairs." |
| **3. User model** | "Three timescales: instant, session (α=0.3), long-term (α=0.1). Welford online statistics for numerical stability. Z-scores against the long-term baseline drive the deviation signal." |
| **4. Adaptation** | "Four adapters: cognitive load, style mirror, emotional tone, accessibility. Produces an 8-dim vector that conditions everything downstream." |
| **5. Routing** | "Contextual Thompson sampling. Bayesian logistic regression per arm, Laplace-approximated posterior, Newton-Raphson MAP refit. 12-dim context. Privacy override vetoes the bandit on sensitive topics." |
| **6a. Local SLM (chat arm)** | "204 M parameters, 12-layer × 12-head Pre-LN transformer with self-attention + cross-attention + FFN per block, MoE (2 experts) + ACT halting, BPE 32 K vocab, weight-tied output. From-scratch — no Llama / Qwen base." |
| **6b. Intent parser (LoRA arm)** | "Qwen3-1.7B + LoRA (DoRA rank 16, NEFTune α=5, 8-bit AdamW, cosine warm restarts), val_loss 5.4×10⁻⁶. Deterministic JSON output validated against `ACTION_SLOTS`. The cascade calls this on commands; mis-parses fall through to Gemini." |
| **6b. Cloud LLM** | "Anthropic Claude with dynamically constructed system prompts from the `AdaptationVector`. Payload sanitised before the HTTPS call." |
| **7. Diary** | "Metadata-only logging: adaptation vectors, latencies, rewards, topic keywords via TF-IDF. No raw text ever persisted. Fernet-wrapped SQLite." |

---

## 3. Panel question bank — three per layer with outlined answers

### Layer 1 — Perception

1. **"Why 32 features and not 64?"**
   - Starting point: four natural groups of eight.
   - Dimensionality is small enough to interpret each feature individually.
   - Adding more risks spurious correlation at our data scale.
   - The TCN lifts this to 64-dim in its first projection, so downstream
     capacity is not capped.

2. **"How do you handle non-Latin scripts?"**
   - Keystroke dynamics are script-independent (they're timing stats).
   - Linguistic complexity features use character-level metrics that
     generalise (e.g. average char-per-token).
   - Some features (Flesch-Kincaid) are English-biased; those are
     localisable with a per-locale adapter.

3. **"Can you explain the correction_rate feature in 20 seconds?"**
   - Backspaces per character typed.
   - Strong predictor of cognitive load and motor difficulty.
   - Combined with std_iki_ms and pause_ratio in the accessibility
     adapter.

### Layer 2 — Encoding

1. **"Why TCN instead of Transformer?"**
   - Causal by construction, parallel over time during training.
   - Exactly computable receptive field.
   - 50 K parameters vs 200 K+ for a comparable-quality Transformer at
     our input size.
   - The contextual attention transformers buy us isn't valuable when
     the input is a 32-dim feature vector rather than tokens.

2. **"How does the contrastive loss work on behavioural data?"**
   - NT-Xent on pairs of augmented views of the same session.
   - Augmentations preserve user identity: feature dropout, Gaussian
     noise, timestep shifting.
   - Training on 10 K synthetic sessions, ~30 min on a laptop CPU.

3. **"What's the receptive field?"**
   - Kernel 3, dilation {1, 2, 4, 8}: 1 + 2×(2⁴−1) = 31 timesteps.
   - Residual skip paths give an effective coverage of ~61.
   - Enough to see a full 10-message conversational window.

### Layer 3 — User model

1. **"Why three timescales?"**
   - Instant captures the current message.
   - Session captures within-session drift — e.g. the user is losing
     focus over this conversation.
   - Long-term captures the user's baseline — what "normal" is for them.
   - Z-scores against long-term require the baseline; deviation
     detection requires session.

2. **"Welford's algorithm — what and why?"**
   - Online mean and variance update with O(1) memory per feature.
   - Numerically stable (no catastrophic cancellation).
   - Critical because the user model lives for years — you cannot store
     all history and batch-recompute.

3. **"How do you cold-start?"**
   - Long-term EMA initialises to a population prior.
   - Z-scores are meaningless for the first ~10 messages — that's why
     the router also uses a Beta-Bernoulli cold-start bandit.
   - After ~30 messages, the long-term profile is reliable.

### Layer 4 — Adaptation

1. **"Why accessibility as a separate adapter?"**
   - It is a *pattern* signal, not a single-feature signal: concurrent
     rise in correction rate, IKI variance, and pause ratio, *without*
     a rise in linguistic complexity.
   - That specific pattern is a marker for motor or cognitive difficulty
     typing ordinary sentences.
   - Merging it into "cognitive load" would blur a clinically
     interesting signal.

2. **"How do you prevent the style mirror from making the bot sycophantic?"**
   - Smoothing lag — style mirrors *long-term* profile, not the current
     message.
   - Bounded dimensions (each in [0, 1]) and clamped updates.
   - The mirror is along four axes only (formality, verbosity,
     emotionality, directness) — none of which are "agree with the
     user".

3. **"What's in the 8th dimension of AdaptationVector?"**
   - Reserved, always 0.0. Deliberate slot for future dimensions
     without breaking the tensor shape or conditioning projector.

### Layer 5 — Routing

1. **"Thompson sampling versus UCB?"**
   - UCB has a tunable confidence parameter that's dataset-dependent.
   - Thompson is Bayes-optimal in the stationary setting.
   - Thompson handles non-stationarity more gracefully.
   - We already need a Laplace posterior for credible intervals;
     Thompson composes with it.

2. **"What's your reward function?"**
   - Weighted combination of user edits to the response, follow-up
     rate, thumbs up/down.
   - Future work: a learned reward model from the same implicit signals
     we use for perception.

3. **"What stops it from routing everything to the cloud?"**
   - Cloud has a per-token *cost* baked into the reward.
   - Cloud has higher latency in the context vector (feature 9, device
     pressure).
   - The privacy override *cannot* be overridden by the bandit.

### Layer 6 — SLM

1. **"Why word-level tokenizer?"**
   - At 6.3 M params trained on DailyDialog + EmpatheticDialogues, an
     8 K word vocabulary is simpler, faster, and more interpretable
     than BPE.
   - We can print the OOV tokens and reason about them.
   - BPE's advantages are for diverse web-scale corpora — we have a
     narrower distribution.

2. **"Why Pre-LN over Post-LN?"**
   - Pre-LN transformers are significantly more stable to train without
     learning-rate warmup since Xiong et al. 2020.
   - Post-LN at our scale would need careful LR scheduling to avoid
     divergence.

3. **"Explain cross-attention conditioning in 30 seconds."**
   - 2-layer MLP projects [AdaptationVector (8); UserState (64)] into
     4 × d_model conditioning tokens.
   - Every transformer block cross-attends from token sequence to those
     4 conditioning tokens.
   - Every token in every layer "sees" the user state in its attention
     context.
   - Cannot be ignored: the cross-attention weights are trained
     end-to-end to minimise generation loss.

### Layer 7 — Diary

1. **"What's in the diary?"**
   - Adaptation vectors, latencies, rewards, TF-IDF topic keywords.
   - No raw text.
   - All Fernet-encrypted.

2. **"Why SQLite?"**
   - Embedded, zero-config, async via aiosqlite.
   - Atomic transactions — matters for user-model updates where a
     partial write would corrupt running statistics.
   - Redis would add a dependency and still need manual transactions.

3. **"How often does the auditor run?"**
   - Every 1 000 exchanges.
   - Runs the same 10 PII regexes over the stored metadata.
   - Should always report clean because raw text is never stored.

---

## 4. Deep-dive zones — be ready to spend 5+ minutes on these

Expect the panel to pick one and go deep. Know these cold.

### 4.1 Cross-attention conditioning

- **Math:** `CrossAttn(X_tok, C) = softmax(X_tok W_Q (C W_K)^T / √d_k) C W_V`
- **Cost:** `O(T × 4 × d)` per head per layer. Rounding error.
- **Why it isn't ignored:** weights trained end-to-end on the generation
  objective.
- **Why it's dynamic:** the projector recomputes conditioning every
  forward pass.
- **Why it's edge-friendly:** ~5 % parameter overhead.
- **The conditioning sensitivity test:** feed the same prompt with
  different AdaptationVectors, measure KL divergence of next-token
  distributions. High divergence → model actually uses its conditioning.

### 4.2 Thompson sampling math

- **Setup:** two arms (local, cloud), 12-dim context, Bayesian logistic
  regression per arm with Gaussian prior.
- **Posterior:** Laplace approximation around the MAP weight vector.
- **Hessian:** `H = λI + Σ σ(ŵ^T x)(1−σ(ŵ^T x)) x x^T`.
- **Sample:** `w̃ ∼ N(ŵ, H⁻¹)`, pick `argmax σ(w̃^T x)`.
- **Refit:** Newton-Raphson every 10 updates, 3–5 iters converge.
- **Cold-start:** Beta-Bernoulli on raw reward for first 10 interactions.
- **Privacy override:** deterministically vetoes the bandit on sensitive
  topics.

### 4.3 Privacy architecture

- **What is stored:** 64-dim user state (enc), 32-dim feature stats (enc),
  adaptation vectors (enc), scalar metrics (enc), TF-IDF keywords (enc).
- **What is never stored:** raw user text, generated responses.
- **Sanitiser:** 10 regex patterns (email, phone, SSN, credit card, IBAN,
  IP, URL, DoB, street address, passport). Compiled at module load.
- **Encryption:** Fernet symmetric, key from `$I3_ENCRYPTION_KEY`, never
  persisted. Dual-key rotation path.
- **Auditor:** runs the 10 regexes against the diary every 1 000
  exchanges.

### 4.4 Edge profile math

- Params: 6.4 M → FP32 25 MB → INT8 7 MB → INT4 weight-only 2.6 MB.
- Latency (M2 baseline, single thread): 170 ms P50 local-path.
- Scale to Kirin 9000: TOPS-ratio gives ~95 ms; NPU efficiency on
  small matmuls likely better.
- KV cache: 2 × 4 layers × 4 heads × 64 head_dim × 256 ctx × 1 byte =
  524 KB flat.
- 50 % memory budget rule: I³ fits on Kirin 9000/9010 (97 % free),
  A2 (90 % free), and encoder-only on Smart Hanhan.

---

## 5. Red-flag questions and clean deflections

These are the questions designed to test whether you know *why* you
made a decision. Don't get defensive. Show you considered the
alternative.

### 5.1 "Why not HuggingFace? Just fine-tune a small open model."

> "Two reasons. First, cross-attention conditioning cannot be
> retrofitted to a pre-trained transformer without retraining from a
> random init — the cross-attention weights don't exist in any
> pre-trained model. At that point the weight-loading complexity isn't
> buying me anything. Second, at edge scale every byte counts. A custom
> SLM with sinusoidal positions, weight-tied output, and a
> task-specific word-level tokenizer is ~2× smaller than a distilled
> HuggingFace equivalent at comparable quality for this task. For a
> classroom project I'd take the HuggingFace shortcut; for a Kirin
> deployment, I wouldn't."

### 5.2 "Why not fine-tune the base model on user data?"

> "Because the whole point of the architecture is that personalisation
> is *not* a weight update. The weights stay fixed; the user signal
> enters as a conditioning tensor. That gives three wins: zero risk of
> catastrophic forgetting, every device runs the same weights (audit-
> and update-friendly), and the personalisation signal is explicit and
> inspectable. Fine-tuning per user on-device has none of those
> properties."

### 5.3 "Why a custom byte-level BPE rather than a SentencePiece / HuggingFace tokenizer?"

> "Two reasons. (1) The JD literally asks for SLM development without
> heavy frameworks; pulling in `tokenizers` undoes that. The BPE in
> `i3/slm/bpe_tokenizer.py` is ~460 LOC of pure Python — Sennrich-style
> merges with byte-level fallback so any UTF-8 byte is representable.
> 32 k vocab trained on the 977 k-pair corpus. (2) Owning the tokeniser
> means I own its serialisation format, special tokens, and the failure
> modes — debugging an OOM during training is a question I can answer
> in code rather than a stack trace through someone else's library.
> Earlier (v1) the model used a word-level tokenizer with frequency
> pruning; the BPE swap landed in v2 and is what's deployed today."

### 5.4 "Isn't implicit signal capture kind of creepy?"

> "This is the question the privacy architecture answers. Nothing I³
> observes about the user leaves the device in raw form. The only
> things persisted are the things I³ needs to do its job — a 64-dim
> embedding, feature statistics, adaptation vectors — and all of them
> are Fernet-encrypted. There is no central server holding user text.
> The architecture is constructed so that the 'creepy' failure mode
> is a physical impossibility, not a policy promise."

### 5.5 "You wrote your own Thompson sampler — isn't there a library for that?"

> "There are libraries, but they assume scalar or discrete contexts.
> Contextual Thompson sampling with Bayesian logistic regression and
> Laplace approximation is a specific combination that fits our
> context-vector shape and our need for a private, on-device posterior.
> The implementation is ~300 lines and fully typed. Using a library
> would have been heavier."

### 5.6 "Cross-attention conditioning — isn't this just a fancy embedding concatenation?"

> "No. Concatenation puts the conditioning in the *prompt*, competing
> for attention mass with the conversation. Cross-attention puts it in
> a separate tensor that every layer at every token position attends
> to. The gradient from the generation loss flows directly into the
> conditioning-producing MLP. And because conditioning is only 4 tokens
> vs a growing prompt, it doesn't dilute as the conversation grows."

### 5.7 "How do you know you're not just overfitting to synthetic personas?"

> "Honest answer: I don't, not fully. The conditioning-sensitivity test
> shows the model uses the conditioning distinctively across personas
> in training distribution, but generalisation to human users is the
> next experiment. That's precisely the hole a Joint Lab collaboration
> would fill — anonymised dialogue corpora with verified persona
> labels."

---

## 6. The ten questions *you* ask at the end

Pick four. Don't ask all ten. These are designed to signal competence
and to get useful information for a decision.

1. "How is the HMI Lab prioritising on-device vs cloud workloads for
   the next generation of HMAF agents — is I³'s 'local-by-default with
   bandit cloud escalation' pattern the direction the Lab is headed?"

2. "The Edinburgh Joint Lab has done interesting work on personalisation
   from sparse signals. Is there a path for London-based work to feed
   into that collaboration, or are they parallel tracks?"

3. "What's the realistic timeline for a Da Vinci backend in ExecuTorch —
   is that a HiSilicon-internal effort, or is there room for open-source
   contribution?"

4. "How does the HMI Lab balance custom ML — the 'traditional ML
   pipelines, SLMs, and foundational-model solutions' in the job
   description — versus building on top of Huawei's centralised model
   releases?"

5. "What does the typical first-6-months look like for an intern in
   the HMI Lab? Prototyping, literature review, infrastructure work, or
   a mix?"

6. "How does the London lab interact with the broader Huawei device
   ecosystem — are you mostly consumed by current-quarter roadmaps, or
   is there genuine research headroom?"

7. "HMAF launched with HarmonyOS 6 in late 2025 — what have you seen in
   the first six months of developer uptake that surprised you?"

8. "What's the Lab's position on model signing and provenance — are you
   aligned with OpenSSF Model Signing, or using an internal Huawei
   PKI?"

9. "For an intern project ending in a paper, what venues are standard
   for the Lab — HCI venues like CHI, ML venues like NeurIPS, or
   systems venues like MobiSys?"

10. "What do good interns do in their first week that makes you think
    'this one's going to work out'?"

---

## 7. The leave-behind checklist

Things to have ready to show:

- [ ] **Working demo.** The local server running, web UI up, a saved
      session that demonstrates the accessibility adapter kicking in.
      Rehearse the 4-phase demo script from `DEMO_SCRIPT.md`.
- [ ] **Architecture diagram.** Either the Mermaid in ARCHITECTURE.md
      rendered, or a hand-drawn version on a clean sheet.
- [ ] **One code sample open on laptop.** Recommendation:
      `i3/slm/cross_attention.py` or the AdaptiveTransformerBlock in
      `i3/slm/transformer.py`. That is the highest-signal code you
      have written.
- [ ] **Profiling report printout.** A `to_markdown()` from the
      `EdgeProfiler` for the TCN and SLM. This is physical proof the
      edge-feasibility claim is real.
- [ ] **This talking-points page printed.** You will not look at it
      during the interview but the act of having it makes the pitch
      smoother.
- [ ] **FAQ page.** The one-page FAQ covering the six red-flag
      questions above.
- [ ] **Huawei dossier printouts.** The five pages in `docs/huawei/`,
      stapled, for the panel to keep.

---

## 8. Things to NOT say

- "I used ChatGPT to write it." Even if true, irrelevant. Talk about
  what the system does, not how you wrote it.
- "It's production-ready." It isn't, nor does it need to be.
- "This could scale to trillions of users." Nobody cares; scale is not
  the point of this project.
- Marketing adjectives ("revolutionary", "state-of-the-art", "best-in-class").
  The panel will mentally discount every one.

---

## 9. Things to absolutely say

- **"Privacy by architecture, not by policy."** The tagline. Use it.
- **"The personalisation signal enters architecturally, not as a prompt."**
  The technical tagline. Use it.
- **"L1–L2 today, with L3 designed and L4–L5 sketched."** The roadmap
  tagline. Use it honestly.
- **"I built this *for* the HMI Lab's research scope."** If asked why
  I³ specifically. True and correct.
- **"36 scenarios, 170 turns, 100 % pass on the multi-turn drift test
  — that's the HMI bar, not single-shot retrieval."** The
  conversational-coherence metric. Use it when the panel asks how
  you measure quality, or "how do you know it's good?". Reference
  test: `D:/tmp/context_drift_test.py`. Up from 20/29 = 69 % at
  iter 40 — the curve is in `memory/project_pipeline_quality_guards.md`.
  Now includes 8-slot multi-fact session memory (name, colour, food,
  music, occupation, location, hobby, age, pet) — recall across
  any number of turns.
- **"Cross-session personalisation, encrypted at rest, with a
  one-utterance wipe."** The privacy-pillar payoff (iter 50).
  Personal facts the user *explicitly* shares are persisted to the
  Interaction Diary's `user_facts` table, Fernet-encrypted with the
  same key as the embedding columns. Loaded on every
  `start_session(user_id)`, cleared by saying *"forget my facts"*.
  4/4 pass on the cross-session recall test
  (`D:/tmp/cross_session_test.py`).  Use this when asked about
  privacy + personalisation + edge — it's the "can it remember me
  without leaking" answer.

---

## 10. The last 30 seconds of the interview

If they ask "anything else you'd like us to know?" — pick one:

- "I think the most interesting thing in the project is the conditioning-
  sensitivity test. Most personalisation systems don't have a way to
  prove the personalisation signal actually flows through to generation.
  I³ does, and the test is three lines of code."

- "The thing I'm most proud of is the Thompson sampling router's
  interaction with the privacy override — the bandit can choose
  freely, but the system cannot choose wrong on sensitive topics. That
  took four iterations to get right."

- "If I were to start over tomorrow, I'd ship the L3 cross-device sync
  first. It's the thing a real user would actually feel — watch knowing
  what the phone knows — and it's where the architecture's privacy
  properties pay the biggest dividend."

Each one ends on a positive technical note. Pick the one that matches
the panel's energy.

---

*End. Close the sheet, pour a glass of water, go.*
