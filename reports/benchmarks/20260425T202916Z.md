# I3 benchmark report — 20260425T202916Z

## Headline

| metric | value |
| --- | ---: |
| perplexity | 25.33 |
| slm_params | 55885824.00 |
| safety_params | 47620.00 |
| adaptation_faithfulness_pct | 0.00 |

## Per-suite rows
### adaptation

| metric | value | unit | n | notes |
| --- | ---: | --- | ---: | --- |
| pct_correct_direction | 0.0000 | pct | 0 | skipped — pipeline not available in this run |

### coherence

| metric | value | unit | n | notes |
| --- | ---: | --- | ---: | --- |
| error | 0.0000 | pct | 0 | coherence suite produced no audit |

### latency

| metric | value | unit | n | notes |
| --- | ---: | --- | ---: | --- |
| error | 0.0000 | ms | 0 | latency suite produced no samples |

### memory

| metric | value | unit | n | notes |
| --- | ---: | --- | ---: | --- |
| on_disk_mb__slm_v2 | 16499.0728 | MB | 1 | on-disk size of checkpoints\slm_v2 |
| on_disk_mb__slm | 4887.8316 | MB | 1 | on-disk size of checkpoints\slm |
| on_disk_mb__encoder | 2.9449 | MB | 1 | on-disk size of checkpoints\encoder |
| on_disk_mb__safety | 0.1851 | MB | 1 | on-disk size of checkpoints\safety |
| on_disk_mb__personalisation | 0.0057 | MB | 1 | on-disk size of checkpoints\personalisation |
| on_disk_mb__gaze | 0.2867 | MB | 1 | on-disk size of checkpoints\gaze |
| params__safety_classifier | 47620.0000 | count | 1 | char-CNN safety classifier (constitutional layer) |
| params__slm | 55885824.0000 | count | 1 | custom decoder transformer (from-scratch) |
| rss_mb | 395.7031 | MB | 1 | resident set size of the benchmark process |

### perplexity

| metric | value | unit | n | notes |
| --- | ---: | --- | ---: | --- |
| final_eval_ppl | 25.3264 | ppl | 6 | final eval perplexity at step 31500 |
| best_eval_ppl | 24.1204 | ppl | 6 | best (lowest) eval perplexity over full training run |

## Plots
- [latency_breakdown](latency_breakdown.svg)
- [perplexity_curve](perplexity_curve.svg)
- [coherence_categories](coherence_categories.svg)
- [adaptation_faithfulness](adaptation_faithfulness.svg)