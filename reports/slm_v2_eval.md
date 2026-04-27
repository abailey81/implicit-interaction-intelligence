# I3 SLM v2 — held-out evaluation

Checkpoint: `D:\implicit-interaction-intelligence\checkpoints\slm_v2\best_model.pt`  
Val set: `D:\implicit-interaction-intelligence\data\dialogue\val.pt` (n=200, seq_len=127)  
Device: `cpu`  

## Architecture
- params: **204.41 M**
- d_model=768, n_layers=12, n_heads=12
- vocab_size=32000

## Aggregate
- cross-entropy: **7.4235**
- perplexity: **1674.841**
- top-1 next-token accuracy: **0.1024**
- tokens evaluated: 13,228
- wall: 38.3 s  (345.2 tok/s)

## Per-quartile perplexity (sequence position)

| Quartile | tokens | loss | ppl |
|---|---|---|---|
| Q1 | 5,795 | 7.284 | 1456.84 |
| Q2 | 4,585 | 7.6418 | 2083.562 |
| Q3 | 2,502 | 7.3988 | 1633.983 |
| Q4 | 346 | 7.044 | 1145.996 |
