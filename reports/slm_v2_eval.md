# I3 SLM v2 — held-out evaluation

Checkpoint: `D:\implicit-interaction-intelligence\checkpoints\slm_v2\best_model.pt`  
Val set: `D:\implicit-interaction-intelligence\data\dialogue\val.pt` (n=500, seq_len=127)  
Device: `cpu`  

## Architecture
- params: **204.41 M**
- d_model=768, n_layers=12, n_heads=12
- vocab_size=32000

## Aggregate
- cross-entropy: **7.4531**
- perplexity: **1725.278**
- top-1 next-token accuracy: **0.1027**
- tokens evaluated: 33,909
- wall: 84.3 s  (402.0 tok/s)

## Per-quartile perplexity (sequence position)

| Quartile | tokens | loss | ppl |
|---|---|---|---|
| Q1 | 14,551 | 7.3435 | 1546.038 |
| Q2 | 11,603 | 7.6513 | 2103.325 |
| Q3 | 6,584 | 7.3983 | 1633.165 |
| Q4 | 1,171 | 7.1615 | 1288.843 |

## Generation samples (greedy + nucleus)

- **prompt**: `hello, how are you doing today?`
  **gen** (1036.7 ms): "hello, how are you doing today?i'm well , how are doing doing doing doing great good good today, you ?"
- **prompt**: `explain photosynthesis in one sentence`
  **gen** (383.8 ms): 'explain photosynthesis in one sentenceThe first of the way of them to keep the city'
- **prompt**: `what is a transformer in machine learning?`
  **gen** (170.1 ms): 'what is a transformer in machine learning?the national name of the first'
- **prompt**: `set timer for 5 minutes`
  **gen** (430.9 ms): 'set timer for 5 minutesoh that way of a week ! i love with my dog !'

## Iter 51 phase 6 — extended-fine-tune experiment

**Question:** can we squeeze meaningfully better perplexity out of the
existing checkpoint by continuing training from step 18 000?

**Method:** added `--resume` to `i3.slm.train_v2` (loads
`model_state_dict` + `optimizer_state_dict` + `global_step` from
`best_model.pt`; LR schedule restarts as a fresh cosine over the new
total).  Re-launched with `lr=3e-5` (1/10 of original peak),
`warmup-ratio=0.001` (essentially no warmup), `n-epochs=3`,
`max-steps=21000` — i.e. +3 000 polish steps on the same 300 k subset.

**Result:** at step 18 750 (~750 steps post-resume) the in-loop eval
returned `loss=5.10, ppl=164.7` vs the baseline `4.99, 146.5` — i.e.
**slightly worse** in the immediate post-restart window.  The training
run was halted at step ~18 900 (loss bouncing in `[4.5, 5.8]`) when
the laptop's swap pressure started failing concurrent CPU jobs.

**Interpretation:** the original cosine-decayed LR at step 18 000 was
already near-zero, so the model had effectively converged on the 300 k
subset under the v2 architecture.  A warm restart with a fresh peak
(even a small one) destabilises the learned minimum without enough
new data for the model to discover a better one.  The path forward
for genuinely better perplexity is *more data* (the full 974 k corpus
or a curated higher-quality subset), not more epochs on the same slice.

**Artefacts kept:** the `--resume` plumbing in `i3/slm/train_v2.py`
remains for future runs; `best_model.pt` is unchanged (the warm-restart
never beat its eval).
