"""Per-provider token and cost accounting.

:class:`CostTracker` aggregates :class:`~i3.cloud.providers.base.TokenUsage`
records across a session and prices them against a static table of
per-model per-MTok rates (:file:`pricing_2026.json`).

Design notes
------------
- Prices are loaded lazily on first :meth:`report` call; :meth:`record`
  does not trigger I/O.
- Unknown (provider, model) pairs are tracked (counted, token-summed)
  but priced at zero.  An ``unknown_models`` field in the report flags
  them so callers can extend the price table.
- All monetary values are USD, per the ``pricing_2026.json`` schema.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from i3.cloud.providers.base import TokenUsage

logger = logging.getLogger(__name__)


_PRICING_PATH = Path(__file__).parent / "pricing_2026.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    """Per-(provider, model) cumulative counters."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0


@dataclass
class CostReport:
    """Aggregate cost / usage snapshot for a session."""

    total_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cached_tokens: int
    total_cost_usd: float
    by_provider: dict[str, dict[str, Any]]
    by_model: dict[str, dict[str, Any]]
    unknown_models: list[tuple[str, str]]
    as_of: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the report."""
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "by_provider": self.by_provider,
            "by_model": self.by_model,
            "unknown_models": [
                {"provider": p, "model": m} for p, m in self.unknown_models
            ],
            "as_of": self.as_of,
        }


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Thread-safe per-provider, per-model token + cost accumulator.

    Args:
        pricing_path: Override the :file:`pricing_2026.json` path (used
            by tests to supply a fixture pricing table).
    """

    def __init__(self, pricing_path: Path | str | None = None) -> None:
        self._path = Path(pricing_path) if pricing_path else _PRICING_PATH
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._unknown: set[tuple[str, str]] = set()
        self._pricing: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Pricing table
    # ------------------------------------------------------------------

    def _load_pricing(self) -> dict[str, Any]:
        if self._pricing is not None:
            return self._pricing
        try:
            with self._path.open("r", encoding="utf-8") as fp:
                self._pricing = json.load(fp)
        except FileNotFoundError:
            logger.warning(
                "Pricing table not found at %s; using empty table",
                self._path,
            )
            self._pricing = {
                "schema_version": 1,
                "as_of": "unknown",
                "models": {},
            }
        except json.JSONDecodeError as exc:
            logger.error("Pricing table JSON decode failed: %s", exc)
            self._pricing = {
                "schema_version": 1,
                "as_of": "unknown",
                "models": {},
            }
        return self._pricing

    def _lookup_rate(
        self, provider: str, model: str
    ) -> tuple[float, float, float, bool]:
        """Return ``(input_rate, output_rate, cached_rate, known)``.

        Rates are per million tokens.  ``known=False`` means the
        (provider, model) pair is missing from the price table; the
        report will surface it in ``unknown_models``.
        """
        pricing = self._load_pricing()
        models = pricing.get("models", {}) or {}
        provider_table = models.get(provider) or {}
        entry = provider_table.get(model)
        if entry is None:
            # Allow a wildcard fallback (used for Ollama).
            entry = provider_table.get("*")
            if entry is None:
                return (0.0, 0.0, 0.0, False)
        input_rate = float(entry.get("input", 0.0) or 0.0)
        output_rate = float(entry.get("output", 0.0) or 0.0)
        cached_rate = float(
            entry.get("cached_input", input_rate * 0.1) or 0.0
        )
        return input_rate, output_rate, cached_rate, True

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        provider: str,
        model: str,
        usage: TokenUsage,
        latency_ms: int,
    ) -> float:
        """Add a completion to the ledger.  Returns its cost (USD)."""
        input_rate, output_rate, cached_rate, known = self._lookup_rate(
            provider, model
        )
        non_cached = max(0, usage.prompt_tokens - usage.cached_tokens)
        cost = (
            (non_cached * input_rate)
            + (usage.cached_tokens * cached_rate)
            + (usage.completion_tokens * output_rate)
        ) / 1_000_000.0

        with self._lock:
            if not known:
                self._unknown.add((provider, model))
                logger.debug(
                    "Unknown pricing for (%s, %s); recording at zero cost",
                    provider,
                    model,
                )
            entry = self._entries.setdefault((provider, model), _Entry())
            entry.calls += 1
            entry.prompt_tokens += usage.prompt_tokens
            entry.completion_tokens += usage.completion_tokens
            entry.cached_tokens += usage.cached_tokens
            entry.total_latency_ms += int(latency_ms)
            entry.total_cost_usd += cost
        return cost

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> CostReport:
        """Aggregate the ledger into a :class:`CostReport`."""
        pricing = self._load_pricing()
        as_of = str(pricing.get("as_of", "unknown"))

        by_provider: dict[str, dict[str, Any]] = {}
        by_model: dict[str, dict[str, Any]] = {}
        total_calls = 0
        total_prompt = 0
        total_completion = 0
        total_cached = 0
        total_cost = 0.0

        with self._lock:
            for (provider, model), entry in self._entries.items():
                total_calls += entry.calls
                total_prompt += entry.prompt_tokens
                total_completion += entry.completion_tokens
                total_cached += entry.cached_tokens
                total_cost += entry.total_cost_usd

                prov_bucket = by_provider.setdefault(
                    provider,
                    {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cached_tokens": 0,
                        "cost_usd": 0.0,
                    },
                )
                prov_bucket["calls"] += entry.calls
                prov_bucket["prompt_tokens"] += entry.prompt_tokens
                prov_bucket["completion_tokens"] += entry.completion_tokens
                prov_bucket["cached_tokens"] += entry.cached_tokens
                prov_bucket["cost_usd"] = round(
                    prov_bucket["cost_usd"] + entry.total_cost_usd, 6
                )

                key = f"{provider}:{model}"
                model_bucket = by_model.setdefault(
                    key,
                    {
                        "provider": provider,
                        "model": model,
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cached_tokens": 0,
                        "avg_latency_ms": 0.0,
                        "cost_usd": 0.0,
                    },
                )
                model_bucket["calls"] += entry.calls
                model_bucket["prompt_tokens"] += entry.prompt_tokens
                model_bucket["completion_tokens"] += entry.completion_tokens
                model_bucket["cached_tokens"] += entry.cached_tokens
                model_bucket["cost_usd"] = round(
                    model_bucket["cost_usd"] + entry.total_cost_usd, 6
                )
                model_bucket["avg_latency_ms"] = round(
                    entry.total_latency_ms / max(1, entry.calls), 2
                )

            unknown = sorted(self._unknown)

        return CostReport(
            total_calls=total_calls,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cached_tokens=total_cached,
            total_cost_usd=total_cost,
            by_provider=by_provider,
            by_model=by_model,
            unknown_models=unknown,
            as_of=as_of,
        )

    def reset(self) -> None:
        """Clear the ledger (useful for per-session isolation)."""
        with self._lock:
            self._entries.clear()
            self._unknown.clear()


__all__ = ["CostReport", "CostTracker"]
