"""Langfuse LLM observability for the I3 cloud client.

Provides a thin, privacy-preserving wrapper around the Langfuse Python
SDK (https://langfuse.com) that traces calls to the Anthropic Claude
Messages API. Captures:

* Model name
* Prompt / completion token counts
* Cost in USD (computed from Anthropic Sonnet 4.5 pricing constants)
* End-to-end latency (milliseconds)
* Optional ``user_id`` and ``trace_id`` context

The module is a **soft dependency**: ``langfuse`` is imported inside a
``try``/``except ImportError`` block so the rest of I3 remains
importable even if the SDK is absent. The tracer is also a no-op when
either ``LANGFUSE_PUBLIC_KEY`` or ``LANGFUSE_SECRET_KEY`` is missing
from the environment.

Usage::

    from i3.observability.langfuse_client import LangfuseTracer

    tracer = LangfuseTracer()  # no-op unless env vars + SDK present

    @tracer.trace_generation(name="claude.generate", user_id="alice")
    async def call_claude(prompt: str) -> dict:
        return await cloud_client.generate(prompt, system_prompt="...")

    # or as a context manager:
    async with tracer.span("claude.generate", user_id="alice") as span:
        result = await cloud_client.generate(prompt, system_prompt="...")
        span.record(result)

No raw user prompts are forwarded to Langfuse unless the caller
explicitly passes ``capture_io=True`` — I3 defaults to metadata-only
tracing to stay aligned with the project's privacy architecture.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import of the langfuse SDK
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent import
    import langfuse as _langfuse_module  # type: ignore[import-not-found]

    _LANGFUSE_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised when dep absent
    _langfuse_module = None  # type: ignore[assignment]
    _LANGFUSE_AVAILABLE = False
    logger.info(
        "langfuse SDK not installed; LangfuseTracer will operate as a no-op. "
        "Install with: pip install langfuse>=2.40"
    )

# ---------------------------------------------------------------------------
# Anthropic Claude Sonnet 4.5 pricing (USD per million tokens)
# Sourced from https://www.anthropic.com/pricing (April 2026).
# ---------------------------------------------------------------------------

INPUT_USD_PER_MTOK: float = 3.0
OUTPUT_USD_PER_MTOK: float = 15.0


def compute_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Compute the per-call Anthropic cost in USD.

    Args:
        input_tokens: Tokens consumed by the prompt.
        output_tokens: Tokens produced in the completion.

    Returns:
        The estimated cost in US dollars (float, may be zero).
    """
    cost_in = (max(0, input_tokens) / 1_000_000.0) * INPUT_USD_PER_MTOK
    cost_out = (max(0, output_tokens) / 1_000_000.0) * OUTPUT_USD_PER_MTOK
    return cost_in + cost_out


# Type variables for the decorator.
T = TypeVar("T")
AsyncFn = Callable[..., Awaitable[T]]


class _NoopSpan:
    """Null-object span used when Langfuse is unavailable."""

    def __init__(self, name: str, trace_id: str) -> None:
        self.name: str = name
        self.trace_id: str = trace_id
        self._start: float = time.monotonic()

    def record(self, result: Any, **extra: Any) -> None:
        """Ignore the payload — no-op mode."""
        return

    def end(self) -> None:
        """Ignore — no-op mode."""
        return


class LangfuseTracer:
    """LLM-specific tracer for Anthropic Claude calls.

    The tracer is intentionally tolerant: if the ``langfuse`` SDK is
    missing or the required public/secret keys are not in the
    environment, every public method becomes a no-op. This keeps I3
    deployments that do not use Langfuse completely free of the
    dependency at runtime.

    Attributes:
        enabled: ``True`` iff the SDK is importable AND both
            ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set.
        default_model: Model name recorded when the wrapped call does
            not supply one in the return value.
    """

    _PUBLIC_KEY_ENV: str = "LANGFUSE_PUBLIC_KEY"
    _SECRET_KEY_ENV: str = "LANGFUSE_SECRET_KEY"
    _HOST_ENV: str = "LANGFUSE_HOST"

    def __init__(
        self,
        *,
        default_model: str = "claude-sonnet-4-5",
        capture_io: bool = False,
    ) -> None:
        """Initialise the tracer.

        Args:
            default_model: Model label used when the wrapped callable
                does not provide an explicit ``model`` field.
            capture_io: If ``True``, forward the raw prompt and
                completion strings to Langfuse. Default ``False`` to
                match the I3 privacy contract.
        """
        self.default_model: str = default_model
        self.capture_io: bool = capture_io
        self._client: Any | None = None
        self.enabled: bool = self._init_client()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_client(self) -> bool:
        """Instantiate the Langfuse client if possible.

        Returns:
            ``True`` if a working Langfuse client was created.
        """
        if not _LANGFUSE_AVAILABLE:
            return False
        public_key = os.environ.get(self._PUBLIC_KEY_ENV, "")
        secret_key = os.environ.get(self._SECRET_KEY_ENV, "")
        if not public_key or not secret_key:
            logger.info(
                "Langfuse env vars missing (%s/%s); tracer disabled.",
                self._PUBLIC_KEY_ENV,
                self._SECRET_KEY_ENV,
            )
            return False
        try:
            # langfuse exposes a top-level Langfuse() class in 2.x.
            langfuse_cls = getattr(_langfuse_module, "Langfuse", None)
            if langfuse_cls is None:
                logger.warning(
                    "langfuse module has no Langfuse class; disabling tracer."
                )
                return False
            kwargs: dict[str, Any] = {
                "public_key": public_key,
                "secret_key": secret_key,
            }
            host = os.environ.get(self._HOST_ENV, "")
            if host:
                kwargs["host"] = host
            self._client = langfuse_cls(**kwargs)
            logger.info("LangfuseTracer initialised (host=%s)", host or "default")
            return True
        except Exception as exc:
            logger.warning(
                "Failed to initialise Langfuse client (%s); tracer disabled.",
                type(exc).__name__,
            )
            self._client = None
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_metrics(
        self, result: Any, elapsed_ms: float
    ) -> dict[str, Any]:
        """Pull token / cost / latency metrics from a generate() result.

        Args:
            result: The value returned by the wrapped async function.
                Expected to be a ``dict`` with ``input_tokens`` /
                ``output_tokens`` keys (the shape returned by
                :class:`i3.cloud.client.CloudLLMClient.generate`).
            elapsed_ms: Wall-clock latency measured by the tracer.

        Returns:
            A dict with ``input_tokens``, ``output_tokens``,
            ``cost_usd``, ``latency_ms``, and ``model``.
        """
        if isinstance(result, dict):
            input_tokens = int(result.get("input_tokens", 0) or 0)
            output_tokens = int(result.get("output_tokens", 0) or 0)
            model = str(result.get("model", self.default_model))
        else:
            input_tokens = 0
            output_tokens = 0
            model = self.default_model
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": compute_cost_usd(input_tokens, output_tokens),
            "latency_ms": elapsed_ms,
            "model": model,
        }

    def _emit(
        self,
        *,
        name: str,
        trace_id: str,
        user_id: str | None,
        metrics: dict[str, Any],
        prompt: str | None,
        completion: str | None,
        error: BaseException | None,
    ) -> None:
        """Send a single generation record to Langfuse (best-effort)."""
        if not self.enabled or self._client is None:
            return
        try:
            payload: dict[str, Any] = {
                "name": name,
                "trace_id": trace_id,
                "model": metrics["model"],
                "usage": {
                    "input": metrics["input_tokens"],
                    "output": metrics["output_tokens"],
                    "total": metrics["input_tokens"]
                    + metrics["output_tokens"],
                    "unit": "TOKENS",
                },
                "metadata": {
                    "latency_ms": metrics["latency_ms"],
                    "cost_usd": metrics["cost_usd"],
                },
            }
            if user_id:
                payload["user_id"] = user_id
            if self.capture_io and prompt is not None:
                payload["input"] = prompt
            if self.capture_io and completion is not None:
                payload["output"] = completion
            if error is not None:
                payload["level"] = "ERROR"
                payload["status_message"] = type(error).__name__
            # langfuse SDK uses .generation() for LLM calls.
            gen_fn = getattr(self._client, "generation", None)
            if callable(gen_fn):
                gen_fn(**payload)
            else:
                trace_fn = getattr(self._client, "trace", None)
                if callable(trace_fn):
                    trace_fn(**payload)
        except Exception as exc:
            logger.debug(
                "Langfuse emit failed (%s); swallowing.", type(exc).__name__
            )

    def flush(self) -> None:
        """Flush pending events to Langfuse (no-op if disabled)."""
        if not self.enabled or self._client is None:
            return
        try:
            flush_fn = getattr(self._client, "flush", None)
            if callable(flush_fn):
                flush_fn()
        except Exception as exc:
            logger.debug(
                "Langfuse flush failed (%s); swallowing.", type(exc).__name__
            )

    # ------------------------------------------------------------------
    # Public API: decorator
    # ------------------------------------------------------------------

    def trace_generation(
        self,
        *,
        name: str = "cloud.generate",
        user_id: str | None = None,
    ) -> Callable[[AsyncFn[T]], AsyncFn[T]]:
        """Decorator wrapping any async function that calls ``generate``.

        The decorator measures wall-clock latency, extracts usage from
        the wrapped callable's return value (expected to be a dict with
        ``input_tokens`` / ``output_tokens``), computes the USD cost,
        and emits one Langfuse generation event per call.

        Args:
            name: Span name recorded in Langfuse.
            user_id: Optional stable user identifier.

        Returns:
            A decorator that preserves the wrapped coroutine's
            signature and return type.
        """

        def decorator(fn: AsyncFn[T]) -> AsyncFn[T]:
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                trace_id = str(uuid.uuid4())
                started = time.monotonic()
                err: BaseException | None = None
                result: Any = None
                try:
                    result = await fn(*args, **kwargs)
                    return result  # type: ignore[no-any-return]
                except BaseException as exc:
                    err = exc
                    raise
                finally:
                    elapsed_ms = (time.monotonic() - started) * 1000.0
                    metrics = self._extract_metrics(result, elapsed_ms)
                    prompt_arg = kwargs.get("user_message") or (
                        args[0] if args else None
                    )
                    prompt = (
                        prompt_arg if isinstance(prompt_arg, str) else None
                    )
                    completion = (
                        result.get("text")
                        if isinstance(result, dict)
                        else None
                    )
                    self._emit(
                        name=name,
                        trace_id=trace_id,
                        user_id=user_id,
                        metrics=metrics,
                        prompt=prompt,
                        completion=completion,
                        error=err,
                    )

            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Public API: context manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def span(
        self,
        name: str = "cloud.generate",
        *,
        user_id: str | None = None,
    ) -> AsyncIterator[_NoopSpan]:
        """Async context manager form of :meth:`trace_generation`.

        Example::

            async with tracer.span("claude.generate", user_id="u1") as s:
                result = await cloud_client.generate(...)
                s.record(result)

        The span records the elapsed time automatically on exit. If the
        caller invokes :meth:`_NoopSpan.record` the payload is used as
        the authoritative source for token counts.
        """
        trace_id = str(uuid.uuid4())
        started = time.monotonic()
        span = _NoopSpan(name=name, trace_id=trace_id)
        _last_record: dict[str, Any] = {}

        original_record = span.record

        def record(result: Any, **extra: Any) -> None:
            _last_record["result"] = result
            _last_record["extra"] = extra
            original_record(result, **extra)

        span.record = record  # type: ignore[method-assign]
        err: BaseException | None = None
        try:
            yield span
        except BaseException as exc:
            err = exc
            raise
        finally:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            result = _last_record.get("result")
            metrics = self._extract_metrics(result, elapsed_ms)
            completion = (
                result.get("text") if isinstance(result, dict) else None
            )
            self._emit(
                name=name,
                trace_id=trace_id,
                user_id=user_id,
                metrics=metrics,
                prompt=None,
                completion=completion,
                error=err,
            )

    # ------------------------------------------------------------------
    # Sync helpers for test harnesses
    # ------------------------------------------------------------------

    def record_generation(
        self,
        *,
        name: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model: str | None = None,
        user_id: str | None = None,
        trace_id: str | None = None,
    ) -> str:
        """Emit a single generation record without wrapping a callable.

        Useful for test harnesses or batch-evaluation scripts.

        Returns:
            The trace id used (generated if not supplied).
        """
        tid = trace_id or str(uuid.uuid4())
        metrics = {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost_usd": compute_cost_usd(input_tokens, output_tokens),
            "latency_ms": float(latency_ms),
            "model": model or self.default_model,
        }
        self._emit(
            name=name,
            trace_id=tid,
            user_id=user_id,
            metrics=metrics,
            prompt=None,
            completion=None,
            error=None,
        )
        return tid


__all__ = [
    "INPUT_USD_PER_MTOK",
    "OUTPUT_USD_PER_MTOK",
    "LangfuseTracer",
    "compute_cost_usd",
]


# ---------------------------------------------------------------------------
# Best-effort sanity check: ensure ``asyncio`` is available (it is part of
# the stdlib so this is only here to silence an unused-import warning
# when consumers re-export this module through a narrow __all__).
# ---------------------------------------------------------------------------

_ = asyncio
