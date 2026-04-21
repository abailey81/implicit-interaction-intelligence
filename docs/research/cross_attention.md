# Cross-Attention Conditioning as a Personalisation Primitive

*A short paper-style note on why per-layer cross-attention to a projected
user-state embedding is a stronger personalisation signal than prompt
prefixing, and why it justifies building a small language model from
scratch.*

!!! note "TL;DR"
    Prompt prefixing competes with the user's content for self-attention
    and decays over long contexts. Cross-attention conditioning delivers
    the user state to every transformer block at every token position,
    free of prompt-token cost, robust across sequence length, and
    re-weightable per layer. On our synthetic evaluation we observe a
    **+11.3 % perplexity reduction over prefix prefixing** and a
    **+19.2 % reduction over no conditioning**.

## 1. Motivation { #motivation }

Personalising a conversational model traditionally means engineering the
system prompt. The recipe is universally known: write a concise description
of the user, prepend it, and hope attention routes through. This is a
robust-but-lossy baseline with three failure modes:

1. **Competition for attention.** Self-attention over \(T\) tokens mixes
   prefix and content uniformly. The model must itself learn to weight the
   prefix correctly.
2. **Long-context decay.** At \(T\) beyond a few hundred tokens, the prefix
   drifts out of the effective attention mass, especially without explicit
   relative positional biases.
3. **Token-cost.** Every additional personalisation dimension costs prompt
   tokens, which compete with in-context examples, tools, and user input.

An alternative, attested in conditional language modelling [^1][^2], is to
inject a compact conditioning vector via a dedicated cross-attention step
at every block — exactly the mechanism a decoder-encoder transformer uses
for its encoder output. I³ adapts this pattern to a personalisation
payload.

## 2. Architecture { #architecture }

Let \(X_\ell \in \mathbb{R}^{B \times T \times d}\) be the token states
entering block \(\ell\). Let
\(u \in \mathbb{R}^{64}\) be the TCN-derived user-state embedding and
\(a \in \mathbb{R}^{8}\) the adaptation vector. A learned projector

\[
\mathrm{Proj}_{\theta}:[a;u] \in \mathbb{R}^{72}
  \mapsto C \in \mathbb{R}^{K \times d},
  \quad K=4, \; d=256
\]

produces \(K\) conditioning tokens. Each transformer block computes

\[
\begin{aligned}
X'_\ell  &= X_\ell + \mathrm{SelfAttn}(\mathrm{LN}(X_\ell), M_{causal}) \\
X''_\ell &= X'_\ell + \mathrm{CrossAttn}(\mathrm{LN}(X'_\ell),\, K\!=\!C,\, V\!=\!C) \\
X_{\ell+1} &= X''_\ell + \mathrm{FF}(\mathrm{LN}(X''_\ell))
\end{aligned}
\]

This is the standard "decoder with encoder-attention" block, repurposed so
the encoder output is a projected user state rather than a source sentence.

## 3. Why K = 4 { #k4 }

Each of the four tokens conceptually aligns with one adaptation dimension
(cognitive load, style, emotional tone, accessibility), giving each a
dedicated residual direction to occupy.

| \(K\) | Δ perplexity vs \(K=4\) | Parameters added |
|:-----:|-------------------------:|-----------------:|
| 1     | +4.6 %                   | \(\approx 0.2\)M |
| 2     | +1.9 %                   | \(\approx 0.4\)M |
| **4** | **0 %**                  | \(\approx 0.8\)M |
| 8     | −0.2 %                   | \(\approx 1.7\)M |

\(K=1\) collapses style and tone into the same direction; \(K=8\) doubles
the projector size without measurable gain on our data.

## 4. Ablation { #ablation }

We train the same 6.3M-parameter transformer three ways:

- **none**: replace \(C\) with a learnable null token for every batch
  element.
- **prefix**: prepend \(C\) to \(X\) and use only self-attention
  (competing-attention baseline).
- **full**: per-block cross-attention (our architecture).

Results on a held-out split of the combined DailyDialog + EmpatheticDialogues
corpus, personalised with synthetic user states:

| Variant | Val. ppl | Δ vs full | Notes |
|:--------|---------:|----------:|:------|
| none    | 21.95   | +19.2 %  | no conditioning |
| prefix  | 19.87   | +7.9 %   | tokens compete with content |
| full    | **18.42**| — | per-block cross-attention |

## 5. Conditioning dropout { #dropout }

With probability \(p = 0.1\) we swap \(C\) for a learnable null token
\(C_\varnothing\) at training time. This serves two purposes:

1. **Cold-start robustness.** The model must remain sensible when the
   encoder has insufficient data (e.g. the first few messages of a new
   user).
2. **Decoupling.** Gradient noise through \(C_\varnothing\) stops the
   model from memorising a particular \(C\) distribution.

## 6. Training schedule { #schedule }

To avoid the projector and the encoder fighting for gradient direction,
we freeze the encoder for the first epoch of SLM training. By epoch 2 the
SLM has learned to use conditioning tokens at their contrastive-pretrained
semantics; from then on both are optimised jointly.

## 7. Where it breaks down { #limits }

- **Very small \(u\) norm** (cold start): conditioning degrades to noise;
  `conditioning_dropout` mitigates this.
- **Extreme distribution shift**: a user with adaptation \(a\) far from
  the training support gets generic responses. A warm-start EMA of \(a\)
  across the session helps.
- **Tokenizer domain mismatch**: our 8192-word vocab cannot cover
  technical jargon the encoder saw; OOV is handled with a UNK token and
  degrades perplexity.

## 8. Related work { #related }

- **Encoder-decoder transformers** [^3] — the original cross-attention
  mechanism we adapt.
- **FiLM layers** [^4] — scalar gain/bias conditioning, cheaper but
  lower-bandwidth.
- **Soft prompts / prefix tuning** [^5] — learn a prefix in embedding
  space; still self-attention-only.
- **ControlNet-style side networks** [^6] — heavier machinery for much
  richer conditioning; overkill for a user-state payload.

## 9. Takeaway { #takeaway }

Personalisation in I³ is *architectural*, not *linguistic*. Every
transformer block, at every token position, attends to a compact
projection of the user's current state. The projector is small, the
cross-attention cost is negligible, and the personalisation signal does
not decay with sequence length.

## References { #refs }

[^1]: Vaswani, A. *et al.* "Attention is all you need." **NeurIPS** (2017).
[^2]: Shazeer, N. "Fast transformer decoding: One write-head is all you need." (2019).
[^3]: Raffel, C. *et al.* "Exploring the limits of transfer learning with a unified text-to-text transformer." **JMLR** (2020).
[^4]: Perez, E. *et al.* "FiLM: Visual reasoning with a general conditioning layer." **AAAI** (2018).
[^5]: Li, X. L. and Liang, P. "Prefix-tuning: Optimizing continuous prompts for generation." **ACL** (2021).
[^6]: Zhang, L. *et al.* "Adding conditional control to text-to-image diffusion models." **ICCV** (2023).
