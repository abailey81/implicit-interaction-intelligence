# Interview Q&A — rapid recall

> **Purpose.** A fast lookup of the most likely questions tomorrow, each
> with a 30-to-60-second answer. Print this and have it on the table.
> Different from `COMPLETE_GUIDE.md` Part 9 — that's the long-form prep;
> this is the *cheat sheet*.
>
> **Organisation.** Four sections matching the interview blocks:
> §1 mid-presentation interruptions, §2 dedicated tech Q&A,
> §3 behavioural questions, §4 candidate-asking-them. Plus §5 emergency
> answers if something goes sideways.
>
> **How to use.** Don't read off the page. Glance for the thesis line,
> then deliver in your own voice.

---

## §1 — Mid-presentation interruptions

These come during the 30-minute presentation. They're a sign of
engagement. Answer in 30–60 seconds, then return to your thread.

### Q. *"What's MoE? Why two experts?"*

> Mixture-of-experts — Shazeer 2017. The dense FFN at each layer is
> replaced by K parallel FFNs plus a gating network that routes each
> token to one of them. Top-1 routing here. Capacity scales with the
> number of experts; FLOPs-per-token stay constant. Two experts is
> the maximum that fit in the 6 GB laptop GPU's training-side
> activation budget — each expert is a 3 072-dimensional projection,
> so two is roughly 50 M extra params. With an A100 we'd run 4–8.

### Q. *"What's ACT halting?"*

> Adaptive Computation Time, Graves 2016. Each layer produces a
> halting probability; once they sum past 1.0 the model exits.
> Average halting depth on the held-out set is around 7.4 layers out
> of 12 — simple turns halt early, hard turns run the full depth. A
> small ponder-cost penalty in the loss encourages early halting.

### Q. *"Why byte-level BPE and not SentencePiece?"*

> Two reasons. First, the JD asks specifically for SLM development
> *without heavy frameworks* — pulling in `tokenizers` undoes that.
> The BPE in `i3/slm/bpe_tokenizer.py` is about 460 LOC of Python.
> Sennrich-style merges with byte-level fallback so any UTF-8 byte
> is representable. Second, owning the tokeniser means I own the
> serialisation format and the failure modes — I can debug an
> issue in code rather than through someone else's library.

### Q. *"What's the 8-axis adaptation vector?"*

> Eight scalars in [0, 1]: cognitive load, verbosity, formality,
> directness, emotionality, simplification, accessibility, emotional
> tone. The TCN encodes a sliding 32-dim feature window of typing
> dynamics into a 64-dim user state. An `AdaptationController` MLP
> projects that to the 8-dim vector. The vector cross-attends into
> every transformer block during generation *and* applies as a
> post-generation surface rewrite.

### Q. *"Two perplexity numbers — which is real?"*

> Both. The 147 is `exp(best_eval_loss)` from the checkpoint blob —
> training-time, response-token-only, same-distribution holdout.
> That's the headline; the apples-to-apples number versus published
> small LMs. The 1 725 is from a stress-test eval — 500 pairs from
> the *full* 977 k corpus, scoring all non-padding tokens including
> the unconditioned first history token. Both ship in
> `reports/slm_v2_eval.md` with their definitions side by side.

### Q. *"How is cross-attention conditioning different from a prompt prefix?"*

> Three reasons. Prompt prefixes are brittle — the model can ignore
> the prefix; the prefix inflates the context window; it must be
> re-emitted on every call. Cross-attention puts the user-state
> vector in a separate tensor consumed by the same gradient-flowing
> mechanism that consumes content. The model can no more ignore it
> than its own input. Ablation: zeroing the cross-attention head
> drops generation divergence under different adaptation vectors
> from over 0.6 Levenshtein distance to under 0.05.

### Q. *"What's DoRA?"*

> Weight-decomposed LoRA, Liu 2024. The frozen weight matrix W is
> decomposed into a magnitude vector and a normalised direction
> matrix. LoRA adapts only the *direction*; the magnitude is updated
> as a small dense vector. DoRA matches full fine-tuning quality
> more closely than vanilla LoRA at the same rank. Cost: a small
> magnitude-vector overhead per layer — negligible at this scale.

### Q. *"What's NEFTune doing?"*

> Jain 2023. Adds Gaussian noise to input embeddings during training,
> scaled by α / sqrt(seq_len × d_model), with α=5 here. Acts as a
> regulariser on instruction-tuning datasets that are tiny relative
> to pre-training. Empirically lifts AlpacaEval by 5–10 points; on
> our intent-parsing data the effect is harder to see because the
> task is too easy for any meaningful generalisation gap, but it's
> known-good practice and zero inference-time cost.

### Q. *"How did 204 M fit on a 6 GB laptop GPU for training?"*

> Three tricks compounded. bf16 mixed precision halves activation
> memory. 8-bit AdamW from `bitsandbytes` quarters the optimiser
> state (it's normally 2× the parameter count in fp32). Gradient
> checkpointing trades roughly 30% wall-clock time for 50% activation
> memory. Combined: peak training memory 3.15 GB, well inside the
> 6.4 GB usable budget.

---

## §2 — Tech Q&A block (10 dedicated minutes)

This is the dedicated technical-questions slot. Likely 4–7 questions.
Use the structural answer template from BRIEFING §2.5: thesis →
evidence → artefact → caveat.

### Q. *"Why three arms instead of one bigger model?"*

> Each arm has a different role and a different cost-quality trade.
> The from-scratch SLM is the differentiator. The Qwen LoRA is
> deterministic JSON for actuators — mixing it into the SLM would
> sacrifice chat quality. Gemini is the OOD safety net. Mixing them
> in one model would either over-budget the SLM or sacrifice the
> from-scratch claim. The cascade matches the JD's both/and bullet:
> build *and* fine-tune.

### Q. *"Why is your perplexity so high?"*

> The architecture is data-bound at this scale. 204 M from-scratch
> on a 300 k subset of synthetic dialogue. The headline 147 is
> respectable for that combination — the published Cornell-only
> baselines around the same parameter range land at 90–200. Open
> problem #2 is the full-corpus retrain — 4 epochs on the full 977 k,
> targeting headline ppl below 80, stress-test below 600.

### Q. *"You say zero HuggingFace, but you fine-tuned Qwen — that's HuggingFace."*

> Correct, and I'm specific about the scope of that claim. **Zero
> HuggingFace dependencies in the from-scratch SLM's *generation
> path*.** The Qwen LoRA arm is a separate component that exists
> precisely *because* the JD also asks about fine-tuning pre-trained
> models. The two arms have different roles. Honest framing: HF is
> used for one arm of the cascade, deliberately, with explicit
> scoping.

### Q. *"How do you know the conditioning isn't decorative?"*

> Ablation. With the cross-attention head firing: generation
> divergence under different adaptation vectors is over 0.6
> Levenshtein distance and a 3.2× length ratio. With the cross-
> attention head zeroed: divergence drops to under 0.05 — same up to
> sampling noise. The conditioning is load-bearing. Reproduction in
> `scripts/benchmarks/evaluate_conditioning.py`.

### Q. *"You haven't run a real user study. How do you know the adaptation works?"*

> Honest answer: I don't, on real users. Validation is currently on
> synthetic personas in `tests/test_simulation_personas.py` and the
> 36-scenario / 170-turn drift test (170/170 pass). User-state
> validation is open problem #5: n=20 within-subjects with self-
> report Likert per axis, Pearson correlation between live
> adaptation vector and self-report. 3 weeks, IRB-blocked. I'd
> rather flag the gap than claim validation I don't have.

### Q. *"Why a contextual bandit instead of full RL?"*

> Three reasons. Sparse rewards: routing only gets a reward signal
> when the user reacts — maybe 1 in 10 turns. RL needs dense rewards
> or a learned reward model; both impractical. Action space: routing
> is one binary-ish choice per turn. The bandit posterior converges
> in ~10 turns; RL would need 10 k+. Interpretability: the LinUCB
> posterior is a Gaussian over linear weights — I can plot it. An
> RL critic is opaque.

### Q. *"What about the SLM not being on a watch yet?"*

> Open problem #1. Encoder ships today (162 KB INT8 ONNX, in-browser
> via WebGPU/WASM). The 204 M SLM at INT8 would be ~200 MB — fits a
> Kirin 9000-class NPU, exceeds a Kirin A2 watch's 8 MB peak
> resident budget. The watch path is distillation to a 10–20 M
> student with a smaller vocab. ONNX export plumbing exists; field
> deployment is week-1 of the internship if I had a Kirin dev kit.

### Q. *"Wouldn't a single GPT-4 call beat your whole cascade?"*

> On chat quality today, yes. On latency, privacy, cost, and
> provable on-device behaviour for HMI, no. GPT-4 is 800–1500 ms
> network round-trip; the local SLM is 600 ms on CPU and would be
> ~50 ms on a Kirin NPU. GPT-4 sees every keystroke; the local arm
> sees none. GPT-4 is $5–15 per 1 M output tokens; local is
> electricity. And I can show DevTools with zero network requests;
> OpenAI can't. The cascade hits Gemini *only* when the local arms
> can't, *only* when the user opted in. That's the HMI shape.

### Q. *"Walk me through one chat turn."*

> User types "set timer for 30 seconds". Stage 1 sanitises the input
> — strips PII, validates UTF-8. Stage 2 resolves coref — nothing to
> resolve here. Stage 3 pulls the recent 10-event keystroke buffer
> and encodes it through the TCN to a 64-dim user state. Stage 4
> projects to the 8-dim adaptation vector. Stage 5 the smart router
> classifies: command pattern fires — `set_timer` regex matches.
> Stage 7 Qwen LoRA emits `{"action": "set_timer", "params":
> {"duration_seconds": 30}}` — schema-valid. Stage 14 the dispatcher
> schedules an asyncio task with 30-second delay. The reply text is
> the actuator template: "OK, setting a 30-second timer." The chip
> shows Qwen·1.0. Total latency around 600 ms cold, 50 ms warm.
> 30 seconds later the asyncio task fires; an `actuator_event` frame
> goes back; the gold-pulse banner drops in the chat.

### Q. *"What's the scariest bug you've shipped?"*

> The perplexity misquote. I had been quoting 1 725 as the headline
> perplexity in some docs and 147 in others — they're different
> metrics on different distributions, both real, but I'd been
> sloppy about which one was which. A reviewer caught it. I wrote
> `scripts/verify_numbers.py` immediately after — re-runnable
> verifier that loads each artefact and asserts 22 specific claims.
> Now any future drift fails the audit immediately. 22/22 PASS as
> of this morning.

### Q. *"Why MoE if you're going to deploy on edge?"*

> Two reasons. First, MoE is *cheaper* on edge per token, not more
> expensive — only one expert fires per token, so FLOPs scale with
> sparse rather than dense FFN. Second, MoE is the path to scaling
> capacity without scaling FLOPs, which is exactly the trade-off a
> wearable target needs. The blocker on edge isn't the architecture;
> it's the parameter count. Distilling to a 20 M student with the
> same MoE+ACT scaffolding is the watch story.

---

## §3 — Behavioural questions (10 dedicated minutes)

Use STAR: Situation → Task → Action → Result. Spend 60% of your
answer on Action.

### Q. *"Tell me about a difficult technical problem you faced."*

> **Iter 51 phase 6 — the warm-restart attempt.** The v2 SLM had
> landed at perplexity 147; I wanted to push it lower. I wrote
> `--resume` plumbing into the trainer to load
> `model_state_dict + optimizer_state + global_step`. Re-launched at
> lr=3e-5 (one tenth of the original peak), warmup-ratio 0.001,
> +3 000 polish steps. Monitored eval-loss every 500 steps. At step
> 18 750 the loss landed at 5.10, ppl 164.7 — *worse* than the 4.987
> baseline. Halted the run. Wrote up the negative result in
> `reports/slm_v2_eval.md` honestly, with the conclusion that the
> architecture is data-bound, not epoch-bound. The path forward is
> the full-corpus retrain, scoped as open problem #2. **Lesson: warm
> restarts don't help when the model has already converged on a
> small slice. The right move was to admit the negative result and
> re-scope.**

### Q. *"Tell me about something you learned quickly."*

> **The LoRA / DoRA / NEFTune / 8-bit AdamW stack** for the Qwen
> fine-tune. Forty-eight hours from "no prior experience with
> bitsandbytes 8-bit AdamW or DoRA specifically" to a working
> training script with all four techniques stacked correctly.
> Read Hu 2021, Liu 2024, Jain 2023, Dettmers 2022 — picked a
> recipe that fits the 6 GB laptop GPU. Built a synthetic
> 5 050-example HMI command dataset. Trained at 100% test
> accuracy on a held-out 253-example test set. **Lesson: stacking
> known-good techniques is faster than re-deriving first-principles
> solutions for each one.**

### Q. *"Tell me about a disagreement."*

> If real example: use it. Default fallback: **"During iter 49 I
> oscillated between deterministic vs learned smart-router. The
> honest disagreement was with my earlier self — the deterministic
> version was easier to debug; a learned router would generalise
> better. I landed on deterministic for v1 with the learned version
> flagged as future work. Resolving the disagreement meant admitting
> both positions had merit and scoping the trade-off explicitly."**

### Q. *"Tell me about something that didn't go as planned."*

> **The iPhone deployment attempt.** Trying to demo I3 on a phone
> over LAN, I hit Windows firewall + network issues blocking
> `localhost:8000`. Spent ~30 minutes troubleshooting, hit a dead
> end. Pivoted: the encoder ONNX path was already running in-browser
> via WebGPU/WASM, so I leaned into that as the edge story rather
> than pursuing a phone runtime. **The deck's edge slide is now
> anchored on the in-browser INT8 encoder demo, which is *more*
> reliable than a phone demo would have been.** Kirin watch
> deployment is open problem #1 with a clear hardware blocker.
> **Lesson: when a path closes, the right move is often a smaller,
> cleaner demo that's actually shippable, not a heroic recovery.**

### Q. *"Tell me about a time you took initiative."*

> **`scripts/verify_numbers.py`** — written after I caught myself
> misquoting the SLM perplexity. **Situation:** A reviewer caught me
> quoting 1 725 as the headline number in one place and 147 in
> another — different metrics on different distributions, both real,
> but easy to confuse. **Task:** prevent the class of error.
> **Action:** wrote a re-runnable verifier that loads each artefact
> and asserts 22 specific claims against the on-disk values: model
> config, training step, eval loss, parameter count, encoder size,
> parity MAE. Tied every recruiter-facing claim to one of those
> assertions. Added a mandatory re-run before any release.
> **Result:** 22/22 PASS as of this morning. Any future drift fails
> immediately. **Lesson: build the audit before you need it. A
> 2-hour script paid for itself the first time it caught drift.**

### Q. *"Tell me about a time you had to give difficult feedback."*

> Default if no real example: **"I'd lean on a project-review
> moment. When the v1 SLM hit perplexity 148 I had to convince myself
> — and then justify in writing — that 148 was good enough to ship
> the v1 demo and that pushing for a v2 retrain was the higher-
> leverage move. The 'difficult feedback' was telling my earlier
> self that more polish on v1 wasn't the path. Wrote the
> conclusion explicitly in the iter 51 summary so future-me would
> remember the reasoning."**

### Q. *"Why this role at Huawei?"*

> Three reasons. First, the JD's both/and — building from scratch
> *and* fine-tuning pre-trained — is rare; most labs do one or the
> other. I3 is built around exactly that combination. Second, HMI
> is where implicit-interaction signals matter most: vehicles,
> wearables, AR glasses — contexts where cognitive bandwidth is
> scarce. Third, the Edinburgh Joint Lab has published on small-LLM
> personalisation, which is the direct extension of I3's TCN-encoder
> + adaptation-vector path. The forward roadmap maps each I3
> component to a HarmonyOS / Kirin extension.

### Q. *"What's your biggest weakness?"*

> Honestly: scope. I'm a generalist by instinct — I3 covers SLM,
> BPE, encoder, bandit, safety, UI, edge — and it'd be easy to
> dilute focus by accepting too many directions. I'm working on it
> by leaning more heavily on tech leads to scope ruthlessly, and by
> writing open-problems docs early so I can park future work without
> losing it. The six PR-shaped open problems on slide 15 are the
> habit in action.

### Q. *"Where do you see yourself in five years?"*

> Doing work that combines the research depth I'm building now with
> the deployment scale Huawei has access to. Not planning a PhD next
> — I want to ship things to real users and validate research
> questions at scale. The HMI Lab is exactly that combination — the
> research questions on implicit interaction matter, and the
> deployment surface (HarmonyOS, Kirin, the wearable line) lets you
> validate them.

### Q. *"What attracted you to AI/ML specifically?"*

> The combination of mathematical structure and observable behaviour.
> I can reason about a transformer block on paper, then I can run
> it and watch what actually happens. Most fields force you to
> choose — pure mathematics has structure but no embodiment;
> systems engineering has embodiment but limited structure. ML has
> both. HMI is a particularly satisfying corner because the
> *behaviour* part is human, which keeps the work honest.

### Q. *"How do you handle stress?"*

> Honest answer: I write things down. The 51-iteration log of I3 is
> the artefact of a habit — when something feels stressful or
> stuck, I open a markdown file and write the situation down,
> including the specific failure mode, what I tried, and what's
> open. Within a few sentences the stress usually resolves into a
> concrete next action. The 88 commits on this project are not a
> performance — they're how I think.

### Q. *"Tell me about a time you failed."*

> The warm-restart attempt above is one. Another: **iter 40
> baseline** — the multi-turn drift test was passing 20 out of 29
> scenarios — about 69%. The model would lose track of entities
> after 5 turns. I tried four architectural fixes that didn't move
> the number. The eventual fix was a *deterministic* coref resolver
> (`i3/dialogue/coref.py`) plus a recency stack of typed entities.
> Got to 170/170 = 100% over 9 iterations. **Lesson: when a
> learned approach fails to converge, the structure problem is
> usually upstream of the model.**

---

## §4 — Candidate Q&A block (5 dedicated minutes)

Pick 2–4 of these depending on flow.

### Strong opening question

> **"What does a typical week look like for an HMI Lab intern?"**

Why: genuine, surfaces whether the role is research-shaped or
engineering-shaped, gives them an opening to talk about specifics.

### Research / vision questions

> **"What's the most interesting open research question the lab is
> working on right now?"**

> **"How does the lab decide what gets shipped to a HarmonyOS
> product vs what stays research?"**

> **"How does the UK lab collaborate with Edinburgh and Shenzhen?"**

### Concrete / deliverable questions

> **"What does success look like for an intern by end of summer?"**

> **"Who would I be working with most closely, and what's their
> focus?"**

### The closing question (save for last)

> **"Based on what we discussed today, is there anything about my
> background or experience you'd want me to clarify or expand on?"**

Why this lands: it (a) signals you can take feedback in real time,
(b) surfaces concerns before they become reasons not to extend an
offer, (c) is confident without being arrogant. If they say "no,
covered it well", positive signal. If they raise a concern,
address it directly using the structural answer template.

### Don't ask in this block

- Salary (HR conversation later).
- Working hours / vacation.
- Anything you could have learned from the website.
- "What do you like most about working here?" (too soft).

---

## §5 — Emergency answers

### "I don't actually know that."

> "Let me think about that for a moment."

(Five seconds of silence. Then either answer or:)

> "I want to give you a precise answer rather than a sloppy one — can
> I come back to this in a moment, or would you like me to think out
> loud?"

(If they say "think out loud", you've turned a blank into a
problem-solving exercise — they want to see how you reason.)

### "That's something I haven't worked with."

> "I haven't worked with [thing] specifically. Walk me through the
> basics?"

(Then listen and tie to something you do know: "That sounds related
to [adjacent concept] — the key difference would be…")

### "I disagree with what you said."

> "That's a fair pushback — let me make sure I understand. You're
> saying [their critique]?"

Then either: "Here's why I still think [position] — [evidence]. But
I take the concern about [their concern]." OR: "You're right, I
hadn't considered [aspect]; I'd update my answer to [updated]."

### "Did you use AI to build this?"

> Yes — Claude Code as a pair-programmer for some of the iter 50–51
> work, and I'm transparent about that in the commit messages
> (the trailers say `Co-Authored-By: Claude Opus`). Every
> architectural decision, every design doc, every published claim
> is mine. The AI accelerated the typing; the project is mine.
> The repo's commit graph is open if you want to inspect the
> cadence.

### "Could you reproduce this from scratch?"

> Yes. The orchestrator at `scripts/run_everything.py` is a 21-stage
> wave-based DAG that goes from a clean checkout to a running
> system in roughly 10 minutes — minus the two long training runs.
> The training runs are reproducible; the iter 51 phase 6 work
> documented every hyperparameter and every result, including the
> negative one.

### "What if I told you we don't have a Kirin dev kit available?"

> Then open problem #1 becomes "browser-side INT8 SLM via ONNX
> Runtime Web with WebGPU acceleration" — same proof of edge
> feasibility, different target hardware. The decision tree is in
> `docs/huawei/forward_roadmap.md`. I'd love to do the watch path
> but I'd happily pivot to a different concrete target.

### "Your demo just crashed."

> "No problem — let me park that and come back to the slides. The
> reproduction recipe is in [file path]; happy to walk through it
> after the call."

(Switch back to slides; do **not** troubleshoot the demo in front
of them. Most candidates lose composure here. Don't.)

---

## §6 — Five things you will quote no matter what

If you blank on every other number, these five will carry you:

1. **204 M parameters** — from-scratch transformer; 12L × 12H,
   d_model 768, MoE-2 + ACT, BPE 32 k.
2. **5.36 × 10⁻⁶ val loss** — Qwen3-1.7B + LoRA; 100% on 253-example
   test set; DoRA + NEFTune α=5 + 8-bit AdamW.
3. **22 / 22 routing classifications** correct on the precision
   smoke; six deterministic signals → five route classes.
4. **162 KB INT8 encoder, 460 µs p50** — runs in-browser via ONNX
   Runtime Web; 12.5× under Kirin A2 watch budget; DevTools shows
   zero `/api/encode` when the toggle is on.
5. **Timer-actually-fires latency: 30 s exact** — `set_timer` →
   asyncio task → `actuator_event` frame → gold-pulse banner.

If you can quote only one, **#4** — that's the JD's edge-deployment
question answered with a live demonstrable artefact.

---

## §7 — Final morning verification

Run before you leave the house:

```bash
python scripts/verify_numbers.py
```

If it returns `ALL CLAIMS VERIFY OK`, every number in this document
and the deck is provable on disk. If it fails, fix it before you
leave.

---

*Last updated: 2026-04-28 for the 29 April 2026 in-person interview at
Huawei R&D UK, MR1, Gridiron Building, Pancras Square.*
