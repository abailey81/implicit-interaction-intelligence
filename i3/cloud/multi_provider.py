"""Multi-provider fallback chain for the I3 cloud layer.

Implements three execution strategies over a list of
:class:`~i3.cloud.providers.base.CloudProvider` instances:

- **sequential**: try each in order; return the first success; apply
  a per-provider circuit breaker so that a repeatedly failing
  provider is skipped for a cool-down window.
- **parallel**: race all providers concurrently and return the
  fastest successful result; cancel the rest.
- **best_of_n**: run all providers, collect successful results, and
  pick the one with the best heuristic "perplexity" score (shorter
  responses relative to prompt size score better -- a cheap proxy
  for confidence when we have no logprobs).

Design references
-----------------
- **Circuit breaker pattern**: Michael T. Nygard, *Release It!*
  (Pragmatic Bookshelf, 2007, ISBN 978-0-9787392-1-8), chapter on
  "Stability Patterns".
- **Hedged / parallel requests**: Jeffrey Dean and Luiz Andre Barroso,
  "The Tail at Scale," *Communications of the ACM*, vol. 56 no. 2,
  Feb 2013, pp. 74-80.  Motivates the ``parallel`` strategy below
  as tail-latency mitigation at the cost of extra backend load.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from i3.cloud.providers.base import (
    CloudProvider,
    CompletionRequest,
    CompletionResult,
    ProviderError,
    RateLimitedError,
)

logger = logging.getLogger(__name__)


Strategy = Literal["sequential", "parallel", "best_of_n"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AllProvidersFailedError(Exception):
    """Raised when every provider in the chain failed.

    Attributes:
        errors: Mapping of ``provider_name -> Exception`` recording
            the last error from each provider that was attempted.
    """

    def __init__(
        self, errors: dict[str, BaseException] | None = None
    ) -> None:
        self.errors: dict[str, BaseException] = errors or {}
        names = ", ".join(self.errors) or "<none tried>"
        super().__init__(f"All providers failed: {names}")


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


@dataclass
class _BreakerState:
    """Minimal Nygard-style circuit breaker state (per provider).

    Fields:
        consecutive_failures: Count of back-to-back failures.
        opened_at: Monotonic timestamp when the breaker tripped; 0.0
            when the breaker is closed.
    """

    consecutive_failures: int = 0
    opened_at: float = 0.0


class _CircuitBreakers:
    """Per-provider breaker store with a simple failure threshold.

    Args:
        failure_threshold: Consecutive failures that trip the breaker.
        cool_down_s: Seconds to stay open before allowing a probe.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cool_down_s: float = 60.0,
    ) -> None:
        self._states: dict[str, _BreakerState] = {}
        self._threshold = failure_threshold
        self._cool_down = cool_down_s

    def is_open(self, provider: str) -> bool:
        state = self._states.get(provider)
        if state is None or state.opened_at == 0.0:
            return False
        if time.monotonic() - state.opened_at > self._cool_down:
            # Half-open: allow the next attempt through.
            state.opened_at = 0.0
            state.consecutive_failures = 0
            return False
        return True

    def record_success(self, provider: str) -> None:
        self._states[provider] = _BreakerState()

    def record_failure(self, provider: str) -> None:
        state = self._states.setdefault(provider, _BreakerState())
        state.consecutive_failures += 1
        if state.consecutive_failures >= self._threshold:
            state.opened_at = time.monotonic()
            logger.warning(
                "Circuit breaker OPEN for provider %s (%d failures)",
                provider,
                state.consecutive_failures,
            )


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


def _heuristic_score(
    request: CompletionRequest, result: CompletionResult
) -> float:
    """Lower is better.

    We don't have per-token logprobs from most providers, so this is a
    deliberately simple proxy for "quality vs. waffle":

    - Prefer responses with non-trivial length.
    - Penalise very long responses (likely verbose hallucination).
    - Penalise "stop='length'" -- the response was truncated.
    """
    text_len = len(result.text)
    if text_len == 0:
        return float("inf")
    target = max(20, (request.max_tokens or 512) * 2)  # ~2 chars per token
    length_penalty = abs(text_len - target) / max(1.0, target)
    truncation_penalty = 0.5 if result.finish_reason == "length" else 0.0
    return length_penalty + truncation_penalty


# ---------------------------------------------------------------------------
# MultiProviderClient
# ---------------------------------------------------------------------------


@dataclass
class _ChainStats:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    skipped_by_breaker: int = 0
    last_errors: dict[str, str] = field(default_factory=dict)


class MultiProviderClient:
    """Fallback-chain / hedged client over multiple providers.

    Args:
        providers: Ordered list of providers.  The first is the
            preferred upstream; subsequent ones are fallbacks.
        strategy: ``"sequential"`` (default), ``"parallel"``, or
            ``"best_of_n"``.  See module docstring.
        per_provider_timeout_s: Hard per-call timeout applied on top
            of whatever the provider's own client uses.
        failure_threshold: Circuit-breaker trip threshold (default 3).
        cool_down_s: Circuit-breaker open-state duration (default 60).
    """

    def __init__(
        self,
        providers: list[CloudProvider],
        *,
        strategy: Strategy = "sequential",
        per_provider_timeout_s: float = 10.0,
        failure_threshold: int = 3,
        cool_down_s: float = 60.0,
    ) -> None:
        if not providers:
            raise ValueError("MultiProviderClient requires >=1 provider")
        if strategy not in ("sequential", "parallel", "best_of_n"):
            raise ValueError(f"Unknown strategy: {strategy!r}")
        self._providers = list(providers)
        self._strategy = strategy
        self._timeout = per_provider_timeout_s
        self._breakers = _CircuitBreakers(
            failure_threshold=failure_threshold,
            cool_down_s=cool_down_s,
        )
        self._stats = _ChainStats()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    async def complete(
        self, request: CompletionRequest
    ) -> CompletionResult:
        """Execute ``request`` according to the configured strategy."""
        if self._strategy == "sequential":
            return await self._sequential(request)
        if self._strategy == "parallel":
            return await self._parallel(request)
        return await self._best_of_n(request)

    async def close(self) -> None:
        """Close every provider.  Idempotent; errors are swallowed."""
        for p in self._providers:
            try:
                await p.close()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug(
                    "Provider %s close() raised %s",
                    p.provider_name,
                    type(exc).__name__,
                )

    @property
    def stats(self) -> dict[str, object]:
        """Return a dict snapshot of per-chain counters."""
        return {
            "attempts": self._stats.attempts,
            "successes": self._stats.successes,
            "failures": self._stats.failures,
            "skipped_by_breaker": self._stats.skipped_by_breaker,
            "last_errors": dict(self._stats.last_errors),
        }

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    async def _sequential(
        self, request: CompletionRequest
    ) -> CompletionResult:
        errors: dict[str, BaseException] = {}
        for p in self._providers:
            if self._breakers.is_open(p.provider_name):
                self._stats.skipped_by_breaker += 1
                logger.info(
                    "Skipping %s: circuit breaker open",
                    p.provider_name,
                )
                continue
            self._stats.attempts += 1
            try:
                result = await asyncio.wait_for(
                    p.complete(request), timeout=self._timeout
                )
                self._breakers.record_success(p.provider_name)
                self._stats.successes += 1
                # Iter 88: bill the global CostTracker so /api/cost/report
                # reflects every cloud call going through the chain.
                # Best-effort: never block the result on telemetry.
                try:
                    from i3.cloud.cost_tracker import get_global_cost_tracker
                    get_global_cost_tracker().record(
                        provider=p.provider_name,
                        model=getattr(result, "model", "") or "",
                        usage=getattr(result, "usage", None),
                        latency_ms=int(getattr(result, "latency_ms", 0) or 0),
                    )
                except Exception:  # pragma: no cover - never block on telemetry
                    pass
                return result
            except asyncio.TimeoutError as exc:
                self._record_error(p, exc, errors)
            except RateLimitedError as exc:
                # Rate-limited providers trip the breaker too; the
                # chain should slide to the next one immediately.
                self._record_error(p, exc, errors)
            except ProviderError as exc:
                self._record_error(p, exc, errors)
            except Exception as exc:  # pragma: no cover - defensive
                self._record_error(p, exc, errors)
        raise AllProvidersFailedError(errors)

    async def _parallel(
        self, request: CompletionRequest
    ) -> CompletionResult:
        """Race all eligible providers; return the fastest success."""
        eligible = [
            p
            for p in self._providers
            if not self._breakers.is_open(p.provider_name)
        ]
        if not eligible:
            raise AllProvidersFailedError({})

        tasks: dict[asyncio.Task[CompletionResult], CloudProvider] = {}
        for p in eligible:
            self._stats.attempts += 1
            tasks[
                asyncio.create_task(
                    asyncio.wait_for(p.complete(request), self._timeout)
                )
            ] = p

        errors: dict[str, BaseException] = {}
        try:
            while tasks:
                done, _pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    p = tasks.pop(task)
                    try:
                        result = task.result()
                    except Exception as exc:
                        self._record_error(p, exc, errors)
                        continue
                    self._breakers.record_success(p.provider_name)
                    self._stats.successes += 1
                    return result
        finally:
            # Cancel anything still running.
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        raise AllProvidersFailedError(errors)

    async def _best_of_n(
        self, request: CompletionRequest
    ) -> CompletionResult:
        """Run all eligible providers and pick the lowest-penalty result."""
        eligible = [
            p
            for p in self._providers
            if not self._breakers.is_open(p.provider_name)
        ]
        if not eligible:
            raise AllProvidersFailedError({})

        coros = [
            asyncio.wait_for(p.complete(request), self._timeout)
            for p in eligible
        ]
        self._stats.attempts += len(eligible)
        raw = await asyncio.gather(*coros, return_exceptions=True)

        candidates: list[tuple[float, CompletionResult]] = []
        errors: dict[str, BaseException] = {}
        for p, outcome in zip(eligible, raw):
            if isinstance(outcome, BaseException):
                self._record_error(p, outcome, errors)
                continue
            self._breakers.record_success(p.provider_name)
            self._stats.successes += 1
            score = _heuristic_score(request, outcome)
            candidates.append((score, outcome))

        if not candidates:
            raise AllProvidersFailedError(errors)
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def _record_error(
        self,
        provider: CloudProvider,
        exc: BaseException,
        errors: dict[str, BaseException],
    ) -> None:
        self._breakers.record_failure(provider.provider_name)
        self._stats.failures += 1
        errors[provider.provider_name] = exc
        self._stats.last_errors[provider.provider_name] = (
            f"{type(exc).__name__}: {exc}"
        )
        logger.warning(
            "Provider %s failed: %s",
            provider.provider_name,
            type(exc).__name__,
        )


__all__ = [
    "AllProvidersFailedError",
    "MultiProviderClient",
    "Strategy",
]
