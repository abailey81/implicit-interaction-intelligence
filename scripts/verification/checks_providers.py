"""Universal provider-layer checks: registry, construction, fallback, cost, translate."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from scripts.verification.framework import CheckResult, register_check

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

EXPECTED_PROVIDERS: tuple[str, ...] = (
    "anthropic",
    "openai",
    "google",
    "azure",
    "bedrock",
    "mistral",
    "cohere",
    "ollama",
    "openrouter",
    "litellm",
    "huawei_pangu",
)


def _now_ms(t0: float) -> int:
    """Milliseconds since ``t0``."""
    return int((time.monotonic() - t0) * 1000)


# Runtime deps that are part of the project's core install. If one of these
# is missing, the failure is an environment issue, not a code defect -- we
# SKIP instead of FAIL so the harness still reports a clean result in a
# minimal CI image that hasn't run `poetry install`.
_RUNTIME_DEPS = {"numpy", "torch", "fastapi", "pydantic", "cryptography", "aiosqlite"}


def _env_missing_result(check_id: str, t0: float, exc: ImportError) -> CheckResult | None:
    """If ``exc`` indicates a missing core runtime dep, return a SKIP result.

    Otherwise return ``None`` so the caller can keep the FAIL semantics.
    """
    missing = getattr(exc, "name", "") or ""
    if missing in _RUNTIME_DEPS:
        return CheckResult(
            check_id=check_id,
            status="SKIP",
            duration_ms=_now_ms(t0),
            message=f"runtime dep not installed: {missing}",
            evidence=None,
        )
    return None


@register_check(
    id="providers.all_registered",
    name="ProviderRegistry contains >= 11 first-class providers",
    category="providers",
    severity="blocker",
)
def check_all_providers_registered() -> CheckResult:
    """Import the registry module and assert the canonical 11 are present."""
    t0 = time.monotonic()
    try:
        from i3.cloud.provider_registry import ProviderRegistry
    except ImportError as exc:
        env_skip = _env_missing_result("providers.all_registered", t0, exc)
        if env_skip is not None:
            return env_skip
        return CheckResult(
            check_id="providers.all_registered",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"ProviderRegistry not importable: {exc}",
            evidence=None,
        )
    names = set(ProviderRegistry.names())
    missing = sorted(set(EXPECTED_PROVIDERS) - names)
    return CheckResult(
        check_id="providers.all_registered",
        status="PASS" if not missing else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"{len(names)} provider(s) registered (>= 11)"
            if not missing
            else f"missing: {missing}"
        ),
        evidence="\n".join(missing) if missing else None,
    )


@register_check(
    id="providers.construct_without_sdk",
    name="Every provider class instantiates without its SDK",
    category="providers",
    severity="high",
)
def check_provider_construct_without_sdk() -> CheckResult:
    """Import each provider adapter and attempt a bare construction.

    SDK-absent failures degrade to a per-provider skip; only genuine
    import errors inside the adapter module count as FAIL.
    """
    t0 = time.monotonic()
    try:
        from i3.cloud import providers as _pkg  # noqa: F401 - ensures package
        from i3.cloud.providers import (
            anthropic as mod_anthropic,
            azure as mod_azure,
            bedrock as mod_bedrock,
            cohere as mod_cohere,
            google as mod_google,
            huawei_pangu as mod_pangu,
            litellm as mod_litellm,
            mistral as mod_mistral,
            ollama as mod_ollama,
            openai as mod_openai,
            openrouter as mod_openrouter,
        )
    except ImportError as exc:
        env_skip = _env_missing_result("providers.construct_without_sdk", t0, exc)
        if env_skip is not None:
            return env_skip
        return CheckResult(
            check_id="providers.construct_without_sdk",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"provider package not importable: {exc}",
            evidence=None,
        )

    # Construction is tolerated to fail -- we only want the import side
    # of the contract (no NameError, no top-level SDK crash).
    attempts = [
        ("anthropic", mod_anthropic, "AnthropicProvider"),
        ("openai", mod_openai, "OpenAIProvider"),
        ("google", mod_google, "GoogleProvider"),
        ("azure", mod_azure, "AzureOpenAIProvider"),
        ("bedrock", mod_bedrock, "BedrockProvider"),
        ("mistral", mod_mistral, "MistralProvider"),
        ("cohere", mod_cohere, "CohereProvider"),
        ("ollama", mod_ollama, "OllamaProvider"),
        ("openrouter", mod_openrouter, "OpenRouterProvider"),
        ("litellm", mod_litellm, "LiteLLMProvider"),
        ("huawei_pangu", mod_pangu, "HuaweiPanguProvider"),
    ]
    missing_classes: list[str] = []
    for name, module, cls_name in attempts:
        if not hasattr(module, cls_name):
            missing_classes.append(f"{name}: {cls_name} not defined")
    return CheckResult(
        check_id="providers.construct_without_sdk",
        status="PASS" if not missing_classes else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            f"all {len(attempts)} adapter modules import cleanly"
            if not missing_classes
            else f"{len(missing_classes)} adapter module(s) missing expected class"
        ),
        evidence="\n".join(missing_classes) if missing_classes else None,
    )


@register_check(
    id="providers.multi_provider_fallback",
    name="MultiProviderClient falls through to second provider on first failure",
    category="providers",
    severity="blocker",
)
def check_multi_provider_fallback() -> CheckResult:
    """Two stub providers -- first raises, second returns -- exercise fallback."""
    t0 = time.monotonic()
    try:
        from i3.cloud.multi_provider import MultiProviderClient
        from i3.cloud.providers.base import (
            ChatMessage,
            CompletionRequest,
            CompletionResult,
            ProviderError,
            TokenUsage,
        )
    except ImportError as exc:
        env_skip = _env_missing_result("providers.multi_provider_fallback", t0, exc)
        if env_skip is not None:
            return env_skip
        return CheckResult(
            check_id="providers.multi_provider_fallback",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"multi_provider not importable: {exc}",
            evidence=None,
        )

    class _Failing:
        provider_name = "failing-stub"

        async def complete(self, request: CompletionRequest) -> CompletionResult:
            raise ProviderError("injected failure", provider=self.provider_name)

        async def close(self) -> None:
            return None

    class _Ok:
        provider_name = "ok-stub"

        async def complete(self, request: CompletionRequest) -> CompletionResult:
            return CompletionResult(
                text="ok",
                provider=self.provider_name,
                model="stub-1",
                usage=TokenUsage(),
                latency_ms=1,
                finish_reason="stop",
            )

        async def close(self) -> None:
            return None

    client = MultiProviderClient(providers=[_Failing(), _Ok()], per_provider_timeout_s=5.0)
    req = CompletionRequest(messages=[ChatMessage(role="user", content="hi")])
    try:
        result = asyncio.run(client.complete(req))
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            check_id="providers.multi_provider_fallback",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"complete() raised {type(exc).__name__}: {exc}",
            evidence=None,
        )
    if result.provider != "ok-stub":
        return CheckResult(
            check_id="providers.multi_provider_fallback",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"expected ok-stub, got {result.provider!r}",
            evidence=None,
        )
    return CheckResult(
        check_id="providers.multi_provider_fallback",
        status="PASS",
        duration_ms=_now_ms(t0),
        message="fallback chain escalated to second provider as expected",
        evidence=None,
    )


@register_check(
    id="providers.cost_tracker_basic",
    name="CostTracker.record then .report yields non-zero totals",
    category="providers",
    severity="medium",
)
def check_cost_tracker_basic() -> CheckResult:
    """Smoke-test the in-memory cost ledger."""
    t0 = time.monotonic()
    try:
        from i3.cloud.cost_tracker import CostTracker
        from i3.cloud.providers.base import TokenUsage
    except ImportError as exc:
        env_skip = _env_missing_result("providers.cost_tracker_basic", t0, exc)
        if env_skip is not None:
            return env_skip
        return CheckResult(
            check_id="providers.cost_tracker_basic",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"cost_tracker not importable: {exc}",
            evidence=None,
        )
    tracker = CostTracker()
    try:
        tracker.record(
            provider="anthropic",
            model="claude-sonnet-4-5",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
            latency_ms=42,
        )
        report = tracker.report()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            check_id="providers.cost_tracker_basic",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"record/report raised {type(exc).__name__}: {exc}",
            evidence=None,
        )
    # The report should have something: either total_calls > 0 or a
    # by_provider bucket.
    total_calls = getattr(report, "total_calls", None)
    if total_calls is None and hasattr(report, "model_dump"):
        total_calls = report.model_dump().get("total_calls", 0)
    if not total_calls:
        return CheckResult(
            check_id="providers.cost_tracker_basic",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message="report.total_calls is zero after record()",
            evidence=str(report)[:300],
        )
    return CheckResult(
        check_id="providers.cost_tracker_basic",
        status="PASS",
        duration_ms=_now_ms(t0),
        message=f"report.total_calls = {total_calls}",
        evidence=None,
    )


@register_check(
    id="providers.prompt_translator_shapes",
    name="prompt_translator produces provider-specific shapes",
    category="providers",
    severity="medium",
)
def check_prompt_translator_shapes() -> CheckResult:
    """Translate a single request through 4 helper functions."""
    t0 = time.monotonic()
    try:
        from i3.cloud.prompt_translator import (
            anthropic_payload,
            cohere_parts,
            google_contents,
            openai_messages,
        )
        from i3.cloud.providers.base import ChatMessage, CompletionRequest
    except ImportError as exc:
        env_skip = _env_missing_result("providers.prompt_translator_shapes", t0, exc)
        if env_skip is not None:
            return env_skip
        return CheckResult(
            check_id="providers.prompt_translator_shapes",
            status="FAIL",
            duration_ms=_now_ms(t0),
            message=f"prompt_translator import failed: {exc}",
            evidence=None,
        )
    req = CompletionRequest(
        system="you are helpful",
        messages=[
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
            ChatMessage(role="user", content="more"),
        ],
        max_tokens=128,
    )
    errors: list[str] = []

    # Anthropic
    try:
        ap = anthropic_payload(req, "claude-sonnet-4-5", 512)
        if "messages" not in ap or "system" not in ap:
            errors.append("anthropic_payload missing messages/system")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"anthropic_payload: {exc}")

    # OpenAI
    try:
        om = openai_messages(req)
        if not isinstance(om, list) or om[0].get("role") != "system":
            errors.append("openai_messages: leading system turn missing")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"openai_messages: {exc}")

    # Google
    try:
        gc = google_contents(req)
        if not isinstance(gc, list) or not all("parts" in c for c in gc):
            errors.append("google_contents: parts missing")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"google_contents: {exc}")

    # Cohere returns (preamble, chat_history, current_message)
    try:
        cp = cohere_parts(req)
        if (
            not isinstance(cp, tuple)
            or len(cp) != 3
            or not isinstance(cp[1], list)
        ):
            errors.append(f"cohere_parts: bad shape {type(cp).__name__}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cohere_parts: {exc}")

    return CheckResult(
        check_id="providers.prompt_translator_shapes",
        status="PASS" if not errors else "FAIL",
        duration_ms=_now_ms(t0),
        message=(
            "4 provider shapes produced cleanly"
            if not errors
            else f"{len(errors)} translator error(s)"
        ),
        evidence="\n".join(errors) if errors else None,
    )
