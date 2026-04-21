# Python SDK

In-process access to the full pipeline. Use this for scripting, training,
evaluation, batch inference, and unit tests.

!!! tip "Install first"
    `poetry install` — see [Installation](../getting-started/installation.md).

## Five-line example { #five-line }

```python
import asyncio
from i3.config import load_config
from i3.pipeline.engine import PipelineEngine
from i3.pipeline.types import PipelineInput

async def main():
    cfg = load_config("configs/default.yaml")
    engine = await PipelineEngine.create(cfg)
    out = await engine.process(PipelineInput(
        user_id="alice",
        message="Explain cross-attention briefly.",
        keystroke_intervals_ms=[120, 145, 98, 210, 88],
        timestamp=1712534400.0,
    ))
    print(out.response)

asyncio.run(main())
```

## Top-level surface { #surface }

| Module | Purpose |
|:-------|:--------|
| `i3.config`          | `load_config(path, overlay=...)` — returns a frozen `I3Config` |
| `i3.pipeline.engine` | `PipelineEngine.create(config)` — constructs the full stack |
| `i3.pipeline.types`  | `PipelineInput`, `PipelineOutput` |
| `i3.interaction`     | Feature extraction, linguistic analysis |
| `i3.encoder`         | TCN encoder + inference wrapper |
| `i3.user_model`      | Three-timescale EMAs, Welford stats, store |
| `i3.adaptation`      | Dimension adapters, controller |
| `i3.router`          | Thompson sampling bandit, sensitivity, complexity |
| `i3.slm`             | Adaptive SLM + tokenizer + generation |
| `i3.cloud`           | Async Claude client |
| `i3.diary`           | Diary store, logger, summariser |
| `i3.privacy`         | Sanitiser, encryption, auditor |
| `i3.profiling`       | Latency, memory, feasibility report |

## Auto-generated reference { #auto-ref }

Rendered live by [`mkdocstrings`](https://mkdocstrings.github.io/). If a
symbol has a docstring in the source tree, it appears below with
signatures, parameters, and source links.

### Pipeline { #ref-pipeline }

::: i3.pipeline.engine
    options:
      show_root_heading: true
      show_root_full_path: false
      members_order: source
      filters: ["!^_"]

::: i3.pipeline.types
    options:
      show_root_heading: true
      show_root_full_path: false
      members_order: source

### Configuration { #ref-config }

::: i3.config
    options:
      show_root_heading: true
      show_root_full_path: false
      members_order: source

### Interaction (Layer 1) { #ref-interaction }

::: i3.interaction.types
    options:
      show_root_heading: true
      show_root_full_path: false

::: i3.interaction.features
    options:
      show_root_heading: true
      show_root_full_path: false

::: i3.interaction.linguistic
    options:
      show_root_heading: true
      show_root_full_path: false
      filters: ["!^_"]

### Encoder (Layer 2) { #ref-encoder }

::: i3.encoder.tcn
    options:
      show_root_heading: true
      show_root_full_path: false

::: i3.encoder.inference
    options:
      show_root_heading: true
      show_root_full_path: false

### User model (Layer 3) { #ref-user-model }

::: i3.user_model.types
    options:
      show_root_heading: true
      show_root_full_path: false

::: i3.user_model.model
    options:
      show_root_heading: true
      show_root_full_path: false

### Adaptation (Layer 4) { #ref-adaptation }

::: i3.adaptation.types
    options:
      show_root_heading: true
      show_root_full_path: false

::: i3.adaptation.controller
    options:
      show_root_heading: true
      show_root_full_path: false

### Router (Layer 5) { #ref-router }

::: i3.router.router
    options:
      show_root_heading: true
      show_root_full_path: false

::: i3.router.bandit
    options:
      show_root_heading: true
      show_root_full_path: false

### SLM (Layer 6a) { #ref-slm }

::: i3.slm.model
    options:
      show_root_heading: true
      show_root_full_path: false

::: i3.slm.cross_attention
    options:
      show_root_heading: true
      show_root_full_path: false

::: i3.slm.generate
    options:
      show_root_heading: true
      show_root_full_path: false

### Cloud (Layer 6b) { #ref-cloud }

::: i3.cloud.client
    options:
      show_root_heading: true
      show_root_full_path: false

::: i3.cloud.prompt_builder
    options:
      show_root_heading: true
      show_root_full_path: false

### Diary (Layer 7) { #ref-diary }

::: i3.diary.store
    options:
      show_root_heading: true
      show_root_full_path: false

::: i3.diary.logger
    options:
      show_root_heading: true
      show_root_full_path: false

### Privacy { #ref-privacy }

::: i3.privacy.sanitizer
    options:
      show_root_heading: true
      show_root_full_path: false

::: i3.privacy.encryption
    options:
      show_root_heading: true
      show_root_full_path: false

### Profiling { #ref-profiling }

::: i3.profiling.latency
    options:
      show_root_heading: true
      show_root_full_path: false

::: i3.profiling.memory
    options:
      show_root_heading: true
      show_root_full_path: false

## Further reading { #further }

- [REST API](rest.md)
- [WebSocket API](websocket.md)
- [Architecture: Layers](../architecture/layers.md)
