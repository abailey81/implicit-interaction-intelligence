# Q&A Preparation — 52 Prepared Pairs

Seven categories. Each answer is structured as:

1. **One-line compressed answer** — deployable if time is short.
2. **Two-sentence elaboration** — the default delivery.
3. **If pressed further** — the third beat, only if asked.

Category counts:

| Category             | Count |
|----------------------|-------|
| Strategic            | `10`  |
| Architecture         | `12`  |
| Privacy              | `8`   |
| Data                 | `6`   |
| Deployment           | `6`   |
| Behavioural          | `6`   |
| Depth-probing        | `4`   |
| **Total**            | `52`  |

Citation style — use the exact phrasing:
- *"that's the Bai/Kolter/Koltun receptive-field formula"*
- *"NT-Xent as in Chen 2020, SimCLR"*
- *"Russo 2018 for the tutorial, Chapelle and Li 2011 for the empirical"*
- *"as Xu put it — experience over computing power"*
- *"as Nissim framed it in the Edinburgh talk — sparse or implicit signals"*

---

## Strategic (10)

### S1. What's the point of a conversational AI? Why a chatbot?
- **Compressed:** The conversation is the demo vehicle, not the point.
- **Elaboration:** What I actually built is three transferable capabilities — a behavioural signal encoder, a three-timescale user model, and an edge-cloud router. Chat just proves them end-to-end in 17 days; the same machinery sits behind Smart Hanhan emotion, AI Glasses gaze-adaptation, or any HarmonyOS agent personalising from sparse signals.
- **If pressed further:** The screening questions asked for traditional ML, SLM, orchestration, and edge — this integrates all four in one demo without conflating them into a research project.

### S2. Why not a large LLM with clever prompting?
- **Compressed:** The user model is orthogonal to the language model.
- **Elaboration:** Even with infinite LLM capability, you still decide what to tell it about the user — which is the TCN and adaptation vector. Beyond that: latency for wearables, and privacy — routing every utterance to cloud contradicts on-device-first.
- **If pressed further:** Prompt engineering on a large LLM is the least portable, most latency-heavy, least private answer to the wrong problem.

### S3. How is this different from Spotify / iPhone Focus / Netflix personalisation?
- **Compressed:** Implicit signals, continuous adaptation, personal baseline.
- **Elaboration:** Nothing requires the user to declare preferences or flip a switch. Adaptation is continuous and multi-dimensional, not discrete modes; and the model learns what's normal for this specific individual rather than matching global clusters.
- **If pressed further:** Spotify recommends what others-like-you enjoyed; I³ recognises that you specifically are different today than yesterday.

### S4. Why now? Why couldn't this have been built five years ago?
- **Compressed:** Research tools, product architecture, commercial thesis all converged recently.
- **Elaboration:** Cross-attention conditioning became standard in the last 2–3 years; contrastive temporal encoders via NT-Xent became practical around 2020. And HarmonyOS 6's three-tier experience-first architecture is a 2025 development.
- **If pressed further:** Before that, you'd prepend conditioning tokens or fine-tune per-user — both inefficient at edge scale.

### S5. Why HMI Lab, not a language-model lab?
- **Compressed:** This is an interaction-experience hypothesis, not a text-quality hypothesis.
- **Elaboration:** Every technical choice — three-tier architecture, three-timescale user model, accessibility axis, privacy constraints — is driven by user-experience reasoning, not ML benchmarks. An LLM lab might build a better SLM; an HMI lab builds the system around the SLM that makes it feel like understanding.
- **If pressed further:** As Xu put it — experience over computing power — that's the compass here.

### S6. If you had six months, not 17 days, what would you build differently?
- **Compressed:** Real-user study, distilled SLM, second modality, MindSpore port, A/B outcomes study.
- **Elaboration:** Twenty participants over a week for archetype validation; distil the SLM from a stronger teacher; add voice or gaze to prove the encoder is modality-agnostic in practice; port to MindSpore Lite and profile on a real Kirin; ship a controlled A/B on task completion with vs without the adaptation layer.
- **If pressed further:** The real metric is outcome, not a held-out loss — that study is the one missing link.

### S7. Why does this fit Huawei specifically?
- **Compressed:** Experience-first + on-device + distributed devices — the thesis matches.
- **Elaboration:** Huawei's 2025 pivot to experience-first, on-device, agent-orchestrated AI with HarmonyOS 6 aligns with where the field is heading. The HMI Lab — small, concept-driven, London–Edinburgh–Shenzhen pipeline — is an organisational pattern that turns research insight into a production path.
- **If pressed further:** Other companies are still building cloud-centric; Huawei's local-AI bet is both differentiated and architecturally sounder.

### S8. The JD says SLMs, traditional ML, foundational-model apps. Why all three?
- **Compressed:** Each tier solves a problem the others can't.
- **Elaboration:** Traditional ML builds encoders for novel modalities where no pretrained model exists; SLMs run under latency and memory constraints on-device; foundational models handle complex reasoning via cloud. I³ puts one of each in the pipeline so each proves its own screening question.
- **If pressed further:** The tier order in the JD — custom to off-the-shelf — is itself a judgement prompt; the router is the piece that enforces that judgement at runtime.

### S9. Did you talk to designers before writing code?
- **Compressed:** Yes — started from the experience moment, not the architecture.
- **Elaboration:** Before the first line I asked — what does this interaction feel like to the user? The answer was "understood without having to explain yourself," and every component was reverse-engineered from that feeling back to models and features.
- **If pressed further:** I present architecture to designers only when they ask for it; the shared vocabulary is experience moments, not layer diagrams.

### S10. Is this really a prototype or a product pitch?
- **Compressed:** Prototype — the honesty slide is there for a reason.
- **Elaboration:** A prototype is a feasibility argument; a product pitch claims readiness. Slide 13 exists because I want you to evaluate this as the former, not mistake it for the latter — a production trajectory is named, but not claimed.
- **If pressed further:** I'd rather be hired for the thinking than for the polish.

---

## Architecture (12)

### A11. Why TCN over Transformer for the encoder?
- **Compressed:** Causal, interpretable receptive field, parameter-efficient.
- **Elaboration:** TCN respects temporal causality natively without masking, has fixed memory per inference regardless of history length, and an interpretable receptive field — that's the Bai/Kolter/Koltun receptive-field formula on the slide. Under 500K parameters versus 2M+ for a comparable Transformer encoder at 10-message windows.
- **If pressed further:** The Transformer's long-range strengths don't earn their quadratic cost at this sequence length.

### A12. Why cross-attention conditioning, not prepending tokens?
- **Compressed:** Dedicated mechanism vs. competing with self-attention bandwidth.
- **Elaboration:** Prepending forces the model to look back at conditioning through self-attention, competing with generation context. Cross-attention gives a dedicated Q/K/V for conditioning at every layer — stronger, more consistent signal at lower capacity cost, which matters most in an 8M-parameter model.
- **If pressed further:** Same reason CLIP-guided generation uses cross-attention — conditioning is a first-class citizen, not a prefix.

### A13. Why three timescales in the user model?
- **Compressed:** Three kinds of decision need three kinds of estimate.
- **Elaboration:** Instant drives the current response, session EMA detects within-conversation drift, long-term EMA is the baseline that lets you detect "today is unusual." A single EMA collapses these and loses structure.
- **If pressed further:** It mirrors how humans track each other — what did you just say, what's the vibe of this conversation, and is this how you usually are.

### A14. How does the router learn? What's the reward?
- **Compressed:** Composite engagement signal — continuation, sentiment, latency, topic.
- **Elaboration:** After a response I compute engagement from the user's next interaction: continuation flag, reply latency, reply length, sentiment delta, topic continuity. Positive ≈ 1, negative ≈ 0, linearly combined; posterior updates per rollout via Laplace + Newton–Raphson.
- **If pressed further:** Not binary, not immediate-only — single-turn engagement is noisy, so there's a k-turn lookahead component.

### A15. Why Thompson sampling over UCB or epsilon-greedy?
- **Compressed:** Continuous context + asymmetric cost + clean Bayesian updates.
- **Elaboration:** UCB's confidence bounds are hard in 12-dim continuous context; Thompson handles it via posterior sampling. Asymmetric costs — cloud calls cost latency and money — reward probability-matching behaviour: explore uncertain, exploit confident.
- **If pressed further:** That's Russo 2018 for the tutorial and Chapelle and Li 2011 for the empirical on this class of problem.

### A16. Why not LoRA over a pre-trained base?
- **Compressed:** LoRA presupposes a large base — breaks edge deployment.
- **Elaboration:** LoRA still requires a large pre-trained base, defeating the whole point that the SLM fits in a Smart Hanhan. Personalising the model itself implies per-user weights — not scalable across millions of devices; I wanted conditioning at inference-time only, one model, all users.
- **If pressed further:** Screening question two specifically excludes HuggingFace-style frameworks — LoRA presupposes that framework.

### A17. How does this scale to HarmonyOS distributed architecture?
- **Compressed:** 64-dim embedding syncs via DDM; each device contributes via native modality.
- **Elaboration:** The user model is architecturally suited — compact syncable state via HarmonyOS Distributed Data Management. Phone: text. Glasses: gaze. Watch: motion. Smart Hanhan: voice/touch. Federated averaging updates long-term profile without raw data leaving device.
- **If pressed further:** Adding "route to paired smartphone" as a third bandit arm is a config change, not an architecture change — exactly the AI Glasses use case.

### A18. Why four dimensions in the AdaptationVector?
- **Compressed:** Four balances expressiveness and identifiability.
- **Elaboration:** Cognitive load (effort), communication style (register), emotional tone (affect), accessibility (capability) — approximately orthogonal. Started with six; pruned anything with Pearson r > 0.4 on synthetic scenarios to avoid identifiability problems in training.
- **If pressed further:** Fewer collapses distinctions designers care about; more correlates and introduces training instability.

### A19. Why four conditioning tokens specifically?
- **Compressed:** 72 conditioning scalars → ample headroom at d_model=256.
- **Elaboration:** 64-dim user state plus 8-dim adaptation is 72 scalars. Four tokens at d_model=256 gives 1024 dimensions of representation — ample headroom for 72 input dimensions, with redundancy for head specialisation.
- **If pressed further:** Converged via ablation — fewer bottlenecks the conditioning, more wastes compute on empty channels.

### A20. Why Pre-LN instead of Post-LN in the SLM?
- **Compressed:** Training stability at small scale without warmup tricks.
- **Elaboration:** Xiong et al. 2020 showed Pre-LN gives well-bounded gradients at initialisation — matters for training an 8M-parameter SLM from scratch on a laptop without heavy warmup scheduling.
- **If pressed further:** Post-LN is still slightly better at the asymptotic optimum; at this scale and budget, stability wins.

### A21. Why not a diffusion model or state-space model instead of TCN?
- **Compressed:** Diffusion is overkill for a 64-dim classification-style encoder; SSMs are plausible.
- **Elaboration:** Diffusion models solve generation, not encoding — wrong tool for this slot. State-space models (Mamba) are genuinely plausible at longer sequences; at 10-message windows a TCN ties on accuracy with lower implementation risk for a 17-day build.
- **If pressed further:** I'd consider Mamba for a multi-modal extension where sequences are thousands of timesteps, not tens.

### A22. Why keep the cloud arm at all — why not go full on-device?
- **Compressed:** Because honesty about capability beats ideology about locality.
- **Elaboration:** An 8M-parameter SLM will not match Claude on complex reasoning or long-form coherence. The router exists so that complex queries without privacy sensitivity get the quality they deserve, and quick intimate ones stay local — the architecture matches the claim.
- **If pressed further:** Full-local is a brand position, not a product one; the router is the product-grade answer.

---

## Privacy (8)

### P23. What about privacy risks? Embeddings encode information.
- **Compressed:** Lossy, abstract, still identity-signalling — named honestly, mitigated structurally.
- **Elaboration:** You cannot reconstruct typed text from the 64-dim embedding — it's trained on dynamics, not content. Still, it encodes identity signal, which is why it's Fernet-encrypted at rest, PII-stripped before any cloud call, and sensitive topics forced local.
- **If pressed further:** Production moves Fernet's key into TrustZone; the Fernet-in-software here is a placeholder, not a claim.

### P24. Could this be used for surveillance?
- **Compressed:** The architecture supports privacy-respecting deployment; it does not force one.
- **Elaboration:** Non-persistence of raw text is enforced at the storage layer — no toggle. Profile is device-local and encrypted. Whether this becomes surveillance is a product-policy question about retention, consent, and what signals leave the device.
- **If pressed further:** Those architectural choices are mitigations, not guarantees — legal, product, and security review is part of any deployment path.

### P25. What stops a user manipulating the adaptation by typing oddly?
- **Compressed:** Adaptation is not an access-control layer.
- **Elaboration:** Someone can type slowly to get "low cognitive load" responses — that just changes style, not privileges. The adaptation layer is not a security boundary; any safety-critical decision needs harder authentication signals, not interaction dynamics.
- **If pressed further:** I'd decline to use adaptation alone for triage, medical complexity routing, or anything consequential.

### P26. What adversarial robustness have you built in?
- **Compressed:** Three partial defences; no formal adversarial testing yet.
- **Elaboration:** TCN trained on synthetic data covering abrupt transitions so it doesn't collapse on pattern shifts. Sensitive-topic classifier forces local routing on adversarial "extract cloud-side data" prompts. Router has a confidence floor below which it falls back to safe policy.
- **If pressed further:** Prompt injection against the SLM, embedding inversion, model extraction via router — all real surfaces I have not formally tested.

### P27. How do you prevent discrimination against out-of-distribution users?
- **Compressed:** I don't fully — production needs diverse data and surfaced confidence.
- **Elaboration:** The TCN is trained on archetypes derived from HCI literature, which undersamples non-native typists, atypical motor patterns, elderly users. Poor encoding propagates to poor adaptation — honest limit.
- **If pressed further:** Mitigation is diverse real-user data stratified by demographics, per-subgroup evaluation, and a surfaced confidence score so the system admits when it doesn't know.

### P28. If a user has a disability you don't detect, could adaptation harm them?
- **Compressed:** Yes — that's exactly why it must stay opt-out capable.
- **Elaboration:** Keystroke-only detection works for motor difficulty, misses vision, cognitive, voice-control users. Design principle: adaptation complements explicit accessibility settings, it must not replace them — the user can always say "treat me normally" and the system respects it.
- **If pressed further:** The worst failure mode is adapting someone out of the experience they actually want — that's the Matthew-values alignment here.

### P29. Can you reconstruct typed text from the embedding?
- **Compressed:** Not under this encoder — the features are dynamics, not content.
- **Elaboration:** The encoder is trained on typing speed, pause distribution, backspace ratio, burst rate — none of which carry token-level content. Inverting to text is not technically possible from this representation.
- **If pressed further:** Different story for a multi-modal encoder that included token streams — that's why the SLM's attention stays on raw text, and only the compressed embedding is stored.

### P30. Why Fernet and not AES-GCM or TrustZone?
- **Compressed:** Fernet is AES-128-CBC + HMAC — fine for software; TrustZone is the target in production.
- **Elaboration:** Fernet gives authenticated encryption with sensible defaults for a prototype. Production on Kirin moves key storage to hardware-backed TrustZone — that's where the `I3_ENCRYPTION_KEY` placeholder goes, not an environment variable.
- **If pressed further:** Fernet's main advantage here is "no way to configure it insecurely" — production needs the hardware root-of-trust regardless.

---

## Data (6)

### D31. What data did you train the TCN on?
- **Compressed:** Synthetic sequences from 8 state archetypes derived from HCI literature.
- **Elaboration:** Markov-chain state transitions with per-state emission distributions sourced from Epp 2011, Vizer 2009, Zimmermann 2014. Synthetic ensures coverage of rare states — motor difficulty, stress — that real data undersamples.
- **If pressed further:** Production validation needs an internal study — 20–30 participants over a week would meaningfully refine and probably extend the archetype set.

### D32. The 8 archetypes — aren't you begging the question?
- **Compressed:** Partly, yes — it's a prototype hypothesis, not a proven space.
- **Elaboration:** The archetypes span the four adaptation dimensions roughly orthogonally and map to typing-dynamics literature plus empathy-dialogue corpora. If real variation doesn't fit, the TCN encodes a narrower space than needed.
- **If pressed further:** Archetype engineering is explicitly a phase-two problem — I'd expect the first real-user study to refine or replace several.

### D33. How do you evaluate SLM quality beyond perplexity?
- **Compressed:** Perplexity + adaptation-fidelity — length, formality, vocab, sentiment match to target.
- **Elaboration:** Axis one: perplexity under 40 on held-out dialogue. Axis two: adaptation fidelity — generate with a target adaptation vector, score 0–1 on length matching, formality score, vocab complexity, sentiment alignment; track per-epoch.
- **If pressed further:** Weak evaluation — ideal is human preference studies with a trained annotator panel; for 17 days this is the quantitative signal that conditioning shapes output rather than just the prompt.

### D34. How did you validate the TCN embeddings are meaningful?
- **Compressed:** Silhouette ~0.62, KNN top-1 ~87%, interpretable PCA clusters.
- **Elaboration:** Silhouette score on held-out synthetic sequences for cluster separation; KNN classification on archetype labels for linear informativeness; visual 2D PCA inspection for interpretable neighbour structure.
- **If pressed further:** What this doesn't validate is downstream adaptation quality on real users — that needs a human study on outcome metrics.

### D35. What's your held-out split? Are you overfitting?
- **Compressed:** 80/10/10 synthetic for TCN, 80/20 dialogue for SLM plus 50-example fidelity set, bandit offline replay.
- **Elaboration:** Dropout, early stopping, weight decay on the SLM; final val loss 1.2× training loss — within typical non-overfitting range at this scale. 8M parameters on ~100K dialogues is within the overfit zone, so regularisation matters.
- **If pressed further:** Honest answer: overfit risk dominates and only real data + federated retraining truly resolves it.

### D36. What about keystroke data from multiple languages or input methods?
- **Compressed:** Core dynamics generalise; language-specific linguistic features don't.
- **Elaboration:** Typing speed, pause distribution, correction rate generalise across Latin scripts. Type-token ratio, Flesch–Kincaid, formality scoring are English-specific and need per-language pipelines; CJK and RTL scripts need rewritten tokenisation and IME-aware timing.
- **If pressed further:** The TCN is the most portable piece across languages; the SLM and linguistic features are the least — that's the internationalisation plan's order.

---

## Deployment (6)

### Dep37. What would it take to deploy in a Huawei product?
- **Compressed:** Roughly 4–6 months across conversion, integration, real-user study, security review.
- **Elaboration:** Phase one: convert PyTorch to MindSpore Lite via ONNX. Phase two: HarmonyOS DDM integration and Kirin-specific INT8 re-calibration. Phase three: 50–100 participants over 4 weeks. Phase four: TrustZone key integration and sensitive-topic audit.
- **If pressed further:** After that, field pilot — the architecture is product-shaped already; the missing pieces are validation, not redesign.

### Dep38. How do you handle model drift?
- **Compressed:** EMA decays naturally; bandit learns online; encoder periodically retrained federated.
- **Elaboration:** The long-term EMA downweights old preferences naturally. The bandit's posterior updates per rollout. The encoder is frozen post-training — periodic federated retraining, maybe quarterly, on aggregated anonymised signals.
- **If pressed further:** The federated scheme plugs into MindSpore Federated — I'd integrate existing infrastructure rather than invent.

### Dep39. Cold start for a new user?
- **Compressed:** Five-message calibration, cloud-biased routing, "getting to know you" indicator.
- **Elaboration:** Long-term profile initialised to population average; session EMA uses higher learning rate; router defaults to cloud arm to avoid brittle SLM responses before personalisation has signal. After calibration, personalisation kicks in.
- **If pressed further:** The failure mode to avoid is aggressive adaptation in the first 30 seconds — that produces uncanny guessing, which feels worse than neutral handling.

### Dep40. Shared devices — multi-user scenarios?
- **Compressed:** Currently single-user; keystroke biometrics + explicit auth for production.
- **Elaboration:** One profile per device in the prototype. For shared devices, session-start identification — explicit (PIN, face unlock) plus implicit (keystroke biometrics at ~90% per the literature). Profile-per-session design means switching is clean.
- **If pressed further:** Implicit user ID via typing is research-grade — I wouldn't ship it as the only identity signal; combined with explicit is the product answer.

### Dep41. How does this interact with XiaoYi / Celia?
- **Compressed:** Complementary — I³ is the understanding layer, XiaoYi is the speaking layer.
- **Elaboration:** The adaptation vector feeds into XiaoYi's prompt construction as a side channel — style, verbosity, pacing — while XiaoYi owns the conversational capability and tool use. No architectural conflict; clean integration surface.
- **If pressed further:** In a HarmonyOS deployment, the AdaptationController's output is the bridge — XiaoYi retains its existing agent semantics.

### Dep42. How would you port to Kirin NPU specifically?
- **Compressed:** PyTorch → ONNX → MindSpore Lite → Kirin NPU runtime with re-calibrated INT8.
- **Elaboration:** Pre-LN transformer, cross-attention, causal conv are all standard ONNX ops. MindSpore Lite has its own quantisation calibration — I'd re-run it in MS Lite since quantised ops aren't always bit-identical to PyTorch's.
- **If pressed further:** Validate with under 1% generation divergence on a held-out set; larger divergence indicates a conversion bug, not quantisation noise.

---

## Behavioural (6)

### B43. Tell me about an open-ended, exploratory context you worked in.
- **Compressed:** This project — vague JD, 17 days, MSc coursework alongside.
- **Elaboration:** I analysed the JD and the products Huawei is actually building — Smart Hanhan, AI Glasses, HarmonyOS 6 — and identified that their intersection requires all four screening criteria simultaneously. That insight decided the project; everything else followed from it.
- **If pressed further:** The exploratory part was figuring out what "implicit signals" meant technically — that led to dynamics-based TCN rather than content-based sentiment analysis. That decision was the hinge.

### B44. How do you collaborate with designers?
- **Compressed:** Experience-first vocabulary, then reveal engineering depth when asked.
- **Elaboration:** Before any code I ask — what does this feel like to the user? Then I work backward to signals, models, and adaptations. When I talk to designers I lead with experience moments, not architecture diagrams.
- **If pressed further:** My conversational AI platform experience taught me: if the designer can't describe the user's moment, I can't build it. Shared vocabulary is the unblocker.

### B45. Tell me about a time something you built didn't work.
- **Compressed:** Thompson sampling router — reward signal was the bug, not the algorithm.
- **Elaboration:** Early in a conversational platform, the router was too exploratory. I debugged assuming an implementation bug; turned out users sometimes keep talking because the AI frustrated them, so "continuation" alone was a misleading reward.
- **If pressed further:** I redesigned the reward as a composite — continuation, sentiment, latency. Every reward-shaping problem since, I've started by assuming my reward signal is the bug. That's why I³'s reward is composite by default.

### B46. A time you pushed back on a technical approach.
- **Compressed:** MSc group project — argued to prune 20+ correlated features.
- **Elaboration:** A group member wanted 20+ engineered features, mostly derivatives of each other. I ran a correlation analysis — most were r > 0.9 with one of three primitives. We reduced to 8 features; attention maps became interpretable; validation F1 went up.
- **If pressed further:** Lesson — pushback must come with data, not opinion. If I can't show it, I don't say it.

### B47. A time you had to learn something quickly.
- **Compressed:** HMMs for the Crypto Stat-Arb system, Rabiner 1989 from scratch, 3 days.
- **Elaboration:** Regime detection needed HMMs; I'd studied them but never implemented. Rabiner's 1989 tutorial, forward-backward from scratch, Viterbi, Baum-Welch, toy-sequence validation with known states.
- **If pressed further:** That pattern — primary sources, first-principles implementation, toy-case validation — is how I approach anything new. It's why I built the TCN and SLM from scratch.

### B48. What environment makes you do your best work?
- **Compressed:** Small teams, direct communication, ownership, deadlines after exploration.
- **Elaboration:** Access to people who know more than me about adjacent things — designer, researcher, engineer — each challenging a different axis. Freedom to explore the problem, hard deadlines once committed.
- **If pressed further:** A team that wants to build something that hasn't existed rather than optimising a known product. The HMI Lab description reads like exactly that.

---

## Depth-probing (4)

### DP49. Walk me through Thompson sampling with Bayesian logistic regression + Laplace.
- **Compressed:** Gaussian posterior per arm via MAP mean + inverse-Hessian covariance; sample, argmax.
- **Elaboration:** Each arm a has weight w_a ∈ R^d, expected reward sigmoid(w_a · x). Posterior on w_a approximated Gaussian via Laplace — mean μ_a from MAP gradient ascent, covariance Σ_a from inverse Hessian of log-posterior at μ_a. Hessian for logistic with Gaussian prior N(0, λI) is H = λI + Σ_t σ(μ_a·x_t)(1 − σ(μ_a·x_t)) x_t x_tᵀ.
- **If pressed further:** At decision time, sample w'_a ~ N(μ_a, Σ_a) independently per arm, pick argmax_a σ(w'_a · x). Online update: one or two Newton steps on μ_a, Sherman–Morrison rank-1 Hessian update on Σ_a. O(d²) per step.

### DP50. Derive NT-Xent and explain why it works.
- **Compressed:** Contrastive softmax over cosine similarities, temperature τ — that's NT-Xent as in Chen 2020, SimCLR.
- **Elaboration:** For a batch of 2N with N positive pairs, L_{i,j} = −log[ exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) ], sim cosine on L2-normalised z. Average over positive pairs. It maximises mutual information between augmented views — numerator — while pushing unrelated examples apart — denominator.
- **If pressed further:** Temperature τ controls softmax sharpness; low τ focuses gradient on hardest negatives, high τ treats all negatives equally. τ ≈ 0.1 empirically. For the TCN, positives are two augmentations of the same archetype sequence.

### DP51. Cross-attention compute cost vs self-attention-only?
- **Compressed:** O(N²d) self vs O(NMd) cross with M=4 — <5% overhead per layer.
- **Elaboration:** Self-attention is O(N²d) dominated by the N×N softmax. Cross-attention between query length N and key/value length M is O(NMd). In I³, M=4 conditioning tokens, so cross-attention is O(4Nd) per layer — essentially linear in N, negligible vs quadratic self-attention.
- **If pressed further:** The alternative — concatenating conditioning into self-attention — raises N by 4 and balloons self-attention quadratically. Cross-attention is strictly cheaper at this regime.

### DP52. What's the hardest part technically, and did you solve it?
- **Compressed:** Making cross-attention conditioning measurably shape generation — partially solved.
- **Elaboration:** An 8M-parameter SLM has limited representational capacity; four conditioning tokens can get lost in self-attention noise. What worked: conditioning injected at every layer, careful initialisation of cross-attn weights so early training has gradient signal, and an auxiliary loss penalising conditioning-agnostic outputs.
- **If pressed further:** Partial solution — conditioning effect is measurable but not as strong as I'd want. Larger model or distillation from a conditioned teacher would likely improve it substantially. The honesty slide owns this.
