# Intent-parser eval report

Iter 51 (2026-04-27).  Test set: `D:\implicit-interaction-intelligence\data\processed\intent\test.jsonl`.

| Backend | n | JSON valid | Action acc | Slots valid | Full match | Macro slot F1 | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **qwen3.5-2b-lora** | 253 | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 7020.9 | 9768.6 |

## Confusion matrix — qwen3.5-2b-lora

| expected ↓ \ predicted → | call | cancel | control_device | navigate | play_music | send_message | set_alarm | set_reminder | set_timer | unsupported | weather_query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **call** | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **cancel** | 0 | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **control_device** | 0 | 0 | 23 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **navigate** | 0 | 0 | 0 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **play_music** | 0 | 0 | 0 | 0 | 22 | 0 | 0 | 0 | 0 | 0 | 0 |
| **send_message** | 0 | 0 | 0 | 0 | 0 | 27 | 0 | 0 | 0 | 0 | 0 |
| **set_alarm** | 0 | 0 | 0 | 0 | 0 | 0 | 26 | 0 | 0 | 0 | 0 |
| **set_reminder** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 28 | 0 | 0 | 0 |
| **set_timer** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 21 | 0 | 0 |
| **unsupported** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **weather_query** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 28 |
