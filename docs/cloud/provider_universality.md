# Provider Universality: The I³ Multi-LLM Architecture

> Batch G7 of the I³ cloud layer ships a **provider-neutral LLM protocol**:
> the same `CompletionRequest` flows through Anthropic, OpenAI, Google,
> Azure, AWS Bedrock, Mistral, Cohere, Ollama, OpenRouter, LiteLLM, and
> Huawei Cloud PanGu behind an identical adapter surface. One config edit
> re-homes the entire pipeline from one vendor to another, and a
> `MultiProviderClient` fallback chain lets the pipeline degrade
> gracefully when any single upstream fails.

## 1. Motivation

### 1.1 Vendor neutrality

The original `i3/cloud/client.py` targets Anthropic's Messages API directly.
That is the right *default* — Claude 4.5 is a strong instruction-following
model — but the rest of the I³ architecture (on-device encoder, adaptation
vector, contextual router) has *nothing* to do with which cloud answers the
few messages that escape device. Hard-coupling to one provider is a
strategic fragility: a price change, an outage, a regional ban, or a new
enterprise-compliance requirement should never require touching encoder
training, routing logic, or database schemas. Batch G7 decouples the two.

### 1.2 Graceful degradation

The `MultiProviderClient` wraps an ordered list of providers with a
circuit-breaker fallback strategy (after Nygard, *Release It!*, 2007). If
the primary provider fails three times in a row, we skip it for 60 seconds
and slide to the next one in the chain — ideally a different vendor, so a
single-provider incident cannot take I³ offline. A *parallel* strategy
(after Dean & Barroso, "The Tail at Scale," CACM 56(2), 2013) hedges
against tail latency by racing all providers and returning the fastest
success; a *best-of-N* strategy collects every provider's response and
picks the highest-scoring one by a simple heuristic.

### 1.3 Research reproducibility

Cross-provider experiments (same adaptation vector, different backbone)
are a first-class research workflow for I³. Without a neutral protocol
each comparison demands a bespoke wiring exercise; with it, the I³
researcher changes one YAML key and re-runs. The `CostTracker` closes
the loop by normalising token and dollar accounting so that side-by-side
per-model reports are a function call away.

### 1.4 The Huawei story

The I³ roadmap (see `docs/huawei/`) is explicitly HarmonyOS-native and
Kirin-class. A first-class Huawei Cloud PanGu adapter turns "we could
run on Huawei" into "we already run on Huawei." PanGu Deep Thinking 5.5
(announced at the Shanghai World AI Conference, June 2025) is the
flagship reasoning model behind Huawei Cloud's agentic / embodied AI
platform; pointing I³'s cloud layer at it is a one-line config change.
That is the strongest possible proof of integration readiness.

---

## 2. The Protocol

All providers implement the same narrow Python typing protocol defined
in `i3.cloud.providers.base`:

```python
class CloudProvider(Protocol):
    provider_name: str
    async def complete(self, request: CompletionRequest) -> CompletionResult: ...
    async def close(self) -> None: ...
```

The request / result dataclasses are frozen Pydantic models so they are
safe to share across the fallback chain:

- `CompletionRequest(system, messages, max_tokens, temperature, stop, metadata)`
- `CompletionResult(text, provider, model, usage, latency_ms, finish_reason)`
- `ChatMessage(role: "system"|"user"|"assistant", content: str)`
- `TokenUsage(prompt_tokens, completion_tokens, total_tokens, cached_tokens)`

Errors are a small closed hierarchy: `ProviderError` with `AuthError`,
`RateLimitedError`, `TransientError`, and `PermanentError` subclasses.
Every adapter MUST map upstream failures onto these four so that
downstream code can route by error class instead of status code.

The `prompt_translator` module centralises the shape-shifting between
the neutral request and the provider-specific payloads (Anthropic's
separate `system` field, Gemini's `contents`/`parts`, Cohere's
preamble/chat_history/message split, Bedrock's per-family body
dispatch, …).

---

## 3. Per-Provider Quickstart

Each provider ships with a small YAML fragment in
`configs/providers/*.yaml`. Merge it into the `cloud:` block of your
main config.

### Anthropic (default)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
# cloud.provider: anthropic, model: claude-sonnet-4-5
```

### OpenAI
```bash
export OPENAI_API_KEY=sk-...
# cloud.provider: openai, model: gpt-4.1 | gpt-5-turbo | o3 | o4-mini
```

### Google Gemini
```bash
export GOOGLE_API_KEY=...
# cloud.provider: google, model: gemini-2.5-pro | gemini-2.5-flash
pip install "google-generativeai>=0.7"
```

### Azure OpenAI
```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
# cloud.provider: azure, deployment: <your-deployment>
```

### AWS Bedrock
```bash
# Use the default AWS credential chain (env vars, ~/.aws/credentials,
# IAM role, or SSO).  Adapter uses boto3.
export AWS_REGION=us-east-1
# cloud.provider: bedrock, model: anthropic.claude-sonnet-4-5
pip install "boto3>=1.34"
```

### Mistral AI
```bash
export MISTRAL_API_KEY=...
# cloud.provider: mistral, model: mistral-large-latest | codestral-latest
pip install "mistralai>=1.0"
```

### Cohere
```bash
export COHERE_API_KEY=...
# cloud.provider: cohere, model: command-r-plus | command-r
pip install "cohere>=5.0"
```

### Ollama (local / offline)
```bash
ollama pull llama3.3
# cloud.provider: ollama, model: llama3.3, base_url: http://localhost:11434
# No API key required.  Great for CI and developer-laptop fallback.
```

### OpenRouter (200+ models, one key)
```bash
export OPENROUTER_API_KEY=...
# cloud.provider: openrouter, model: anthropic/claude-sonnet-4.5
# (routing prefix picks the upstream provider)
```

### LiteLLM (universal fallback, 100+ providers)
```bash
pip install "litellm>=1.50"
# cloud.provider: litellm, model: openai/gpt-4.1  (etc.)
# Each upstream still needs its own credentials.
```

### Huawei Cloud PanGu (strategic first-party)
```bash
export HUAWEI_CLOUD_PANGU_APIKEY=...
export HUAWEI_CLOUD_PANGU_REGION=cn-southwest-2
# cloud.provider: huawei_pangu, model: pangu-deepthink-5.5
```

---

## 4. Huawei PanGu — Deep Dive

The `huawei_pangu` adapter is the highest-strategic-value provider in
I³. It targets Huawei Cloud's PanGu service and ships with
**PanGu Deep Thinking 5.5** as the default model.

**What is Deep Thinking 5.5?** At the Shanghai World AI Conference
(WAIC, June 2025) Huawei Cloud announced the PanGu 5.5 family — five
foundation models covering natural language, multimodal, prediction,
scientific computing, and computer vision. The headline model is
**Deep Thinking 5.5**, a **718 B-parameter Mixture-of-Experts (MoE)**
reasoning model that integrates a *fast thinking* pipeline (low-latency
System-1 style responses) with a *slow thinking* pipeline (deliberative
chain-of-thought reasoning) behind a single interface. It is the
reasoning engine underpinning Huawei Cloud's Agentic AI and
**Embodied AI** platforms, both of which were announced alongside it at
WAIC 2025.

**Why it matters for I³.** I³'s adaptation vector, cross-attention
conditioning, and style-mirror are *precisely* the kind of implicit,
user-adaptive capability surfaces that Huawei's agent framework (HMAF)
needs to expose. Wiring I³'s cloud layer to PanGu means an I³ agent
running under HMAF can lift reasoning to Deep Thinking 5.5 without
leaving the Huawei Cloud perimeter — no cross-cloud egress, no
jurisdictional surprises, no vendor discontinuity between encoder
training (which can use public Huawei Cloud ModelArts infrastructure)
and production inference.

**Access requirements.** Huawei Cloud PanGu API access requires (1) a
Huawei Cloud account with (2) the **PanGu Large Models service enabled**
in the target region (3) an API key provisioned under the PanGu
workspace. Regions with the broadest PanGu model catalogue are
`cn-southwest-2` (Guiyang, default), `cn-north-4` (Beijing), and
`cn-east-3` (Shanghai). Without an enabled PanGu workspace the
adapter raises `AuthError` at client-construction time with a clear
hint pointing operators at Huawei Cloud onboarding.

**Transport.** The adapter uses raw `httpx` (not a third-party SDK).
This is intentional: there is no official, stable PyPI-distributed
PanGu SDK at time of writing, and keeping the transport explicit means
there is no ImportError-at-import-time footgun, and the wire format
is auditable in a single 260-line file.

---

## 5. Multi-provider fallback

```python
from i3.cloud.provider_registry import ProviderRegistry
from i3.cloud.multi_provider import MultiProviderClient

primary   = ProviderRegistry.get("anthropic")
secondary = ProviderRegistry.get("google")
tertiary  = ProviderRegistry.get("ollama")  # local safety net

client = MultiProviderClient(
    [primary, secondary, tertiary],
    strategy="sequential",          # or "parallel" or "best_of_n"
    per_provider_timeout_s=10.0,
    failure_threshold=3,            # 3 consecutive failures → open
    cool_down_s=60.0,               # stays open 60s, then half-open probe
)
result = await client.complete(request)
```

**Sequential (default).** Try providers in order. On each failure, bump
the provider's consecutive-failure counter; at the threshold open the
circuit breaker. Skip open-breaker providers for `cool_down_s`, then
probe once. Return the first success; raise `AllProvidersFailedError`
if every provider fails.
*Use when:* cost matters and P99 latency does not — you only pay for
the first successful call.

**Parallel.** Race all eligible providers concurrently; return the
first success, cancel the rest. Embodies the Dean-Barroso "hedged
request" pattern: trades a constant factor in cost for a step change
in tail latency.
*Use when:* user-facing latency is the top constraint (interactive UI,
voice, real-time agents).

**best_of_N.** Run every eligible provider, then score results with a
cheap heuristic (length-vs-target + truncation penalty) and return
the winner. This is a provider-neutral perplexity proxy for
scenarios where we lack logprobs.
*Use when:* quality matters more than cost — e.g. long-form summary
generation, where you can afford 4× the spend for a measurably better
answer.

---

## 6. Cost tracking

```python
from i3.cloud.cost_tracker import CostTracker

tracker = CostTracker()
result = await provider.complete(request)
tracker.record(result.provider, result.model, result.usage, result.latency_ms)
report = tracker.report()

print(report.to_dict())
# {"total_cost_usd": 0.012, "by_provider": {...}, "by_model": {...}, ...}
```

Prices come from `i3/cloud/pricing_2026.json` (schema_version 1,
as_of 2026-04-22). The table covers every (provider, model) pair I³
ships with. Unknown pairs are tracked at zero cost and surfaced under
`unknown_models` so operators can extend the table.

The PanGu entries are marked `"estimate_only": true` — they reflect an
informed public-pricing estimate, not a binding Huawei Cloud rate card.
Treat them as order-of-magnitude guidance until a contracted rate is
in place.

---

## 7. Switching providers with one edit

```yaml
# configs/default.yaml
cloud:
  provider: "anthropic"         # ← change this key
  model: "claude-sonnet-4-5"
  max_tokens: 200
  timeout: 10.0
  fallback_on_error: true
  fallback_chain: []            # or e.g. [anthropic, google, ollama]
  cost_tracking_enabled: true
```

Supported values for `provider`: `anthropic`, `openai`, `google`,
`azure`, `bedrock`, `mistral`, `cohere`, `ollama`, `openrouter`,
`litellm`, `huawei_pangu`. If `fallback_chain` is non-empty the
registry builds a `MultiProviderClient` wrapping the named providers
in order; otherwise a single adapter is used.

---

## 8. Testing strategy

**Unit tests.** Every adapter is unit-tested against SDK mocks
(`pytest.importorskip` for each optional SDK). The registry, the
translator, the cost tracker, and the multi-provider fallback chain
have dedicated test modules with 40+ cases total. See:

- `tests/test_provider_registry.py`
- `tests/test_multi_provider.py`
- `tests/test_prompt_translator.py`
- `tests/test_cost_tracker.py`

**Live-API integration tests.** Gated behind `I3_TEST_LIVE_PROVIDERS=1`.
When that env var is set (plus the relevant provider credentials), the
live tests call real endpoints with a deterministic small prompt and
assert a non-empty response. In CI we leave the env var unset; the
live tests are skipped, all other tests run.

**Mocking a provider.**

```python
from i3.cloud.provider_registry import ProviderRegistry
from i3.cloud.providers.base import CompletionResult, TokenUsage

class FakeProvider:
    provider_name = "fake"
    async def complete(self, request):
        return CompletionResult(
            text="hello", provider=self.provider_name, model="fake-1",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            latency_ms=10, finish_reason="stop",
        )
    async def close(self): pass

ProviderRegistry.register("fake", lambda opts: FakeProvider())
```

---

## 9. References

- Huawei Cloud, *PanGu 5.5 model family announcement*, Shanghai WAIC,
  June 2025 — see also the Huawei Cloud Agentic and Embodied AI
  platform launches from the same event.
- Michael T. Nygard, *Release It!: Design and Deploy Production-Ready
  Software*, Pragmatic Bookshelf, 2007, ISBN 978-0-9787392-1-8 —
  circuit-breaker pattern.
- Jeffrey Dean and Luiz André Barroso, "The Tail at Scale,"
  *Communications of the ACM*, 56(2), Feb 2013, pp. 74-80 —
  hedged / parallel-request pattern.
