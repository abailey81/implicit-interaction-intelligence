"""Outlines constrained generation for the I3 *local* SLM path.

`Outlines <https://dottxt-ai.github.io/outlines/>`_ enforces
grammar / regex / JSON-schema constraints on LLM output by masking the
logits at each decoding step so that only tokens satisfying the target
automaton remain.  Because it operates on raw logits it only works with
*local* models where we own the forward pass — the I3 on-device SLM,
an HF model, llama.cpp via its Python bindings, etc.  It is **not**
applicable to Claude, which we call through a remote HTTP API.

References:
    * Willard, B. T. & Louf, R. (2023). **Efficient Guided Generation
      for Large Language Models.** arXiv:2307.09702.  The KV-cache-
      friendly FSM construction implemented by Outlines.
    * Outlines documentation:
      https://dottxt-ai.github.io/outlines/latest/

Install hint::

    pip install "outlines>=0.1"
"""

from __future__ import annotations

import logging
from typing import Any, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent
    import outlines  # type: ignore[import-not-found]
    import outlines.generate as _ol_generate  # type: ignore[import-not-found]

    _OUTLINES_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised when dep absent
    outlines = None  # type: ignore[assignment]
    _ol_generate = None  # type: ignore[assignment]
    _OUTLINES_AVAILABLE = False


_INSTALL_HINT: str = (
    "outlines is not installed. Install with: "
    '`pip install "outlines>=0.1"`.'
)


def is_available() -> bool:
    """Return ``True`` iff ``outlines`` is importable."""
    return _OUTLINES_AVAILABLE


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

JsonSchema = Union[str, dict[str, Any], type]
"""Accepted types for the JSON-schema constraint:

* ``str``  — raw JSON-schema as a JSON string.
* ``dict`` — parsed JSON-schema as a Python mapping.
* ``type`` — a Pydantic ``BaseModel`` subclass (Outlines inspects it).
"""


def constrained_generate(
    model: Any,
    prompt: str,
    regex_or_schema: str | JsonSchema,
    *,
    max_tokens: int = 256,
    sampler: Any | None = None,
) -> Any:
    """Generate text that satisfies a regex or JSON-schema constraint.

    Dispatches between :func:`outlines.generate.regex` and
    :func:`outlines.generate.json` based on the shape of
    *regex_or_schema*:

    * If it is a ``str`` and does not parse as JSON, it is treated as
      a regex pattern.
    * If it is a ``dict``, a Pydantic class, or a JSON string, it is
      treated as a JSON schema.

    Args:
        model: An Outlines-wrapped model (``outlines.models.transformers``,
            ``outlines.models.llamacpp``, etc.).  Must expose a tokenizer
            and logits access; remote HTTP models are unsupported.
        prompt: The prompt text.
        regex_or_schema: The constraint.
        max_tokens: Maximum new tokens to sample.
        sampler: Optional Outlines sampler (greedy / multinomial / beam).

    Returns:
        The constrained completion.  For JSON schemas Outlines parses
        and returns a Python object (dict or Pydantic instance); for
        regex constraints it returns the raw string.

    Raises:
        ImportError: If ``outlines`` is not installed.
        ValueError: If *regex_or_schema* cannot be classified.
    """
    if not _OUTLINES_AVAILABLE:
        raise ImportError(_INSTALL_HINT)
    assert _ol_generate is not None

    kind = _classify_constraint(regex_or_schema)
    if kind == "regex":
        gen = _ol_generate.regex(model, regex_or_schema, sampler=sampler)  # type: ignore[arg-type]
    elif kind == "json":
        gen = _ol_generate.json(model, regex_or_schema, sampler=sampler)  # type: ignore[arg-type]
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported constraint kind: {kind!r}")

    return gen(prompt, max_tokens=max_tokens)


def _classify_constraint(constraint: Any) -> str:
    """Classify *constraint* as ``"regex"`` or ``"json"``.

    - ``dict`` or subclass of ``pydantic.BaseModel`` -> ``"json"``.
    - ``str`` that parses as JSON beginning with ``{`` -> ``"json"``.
    - any other ``str`` -> ``"regex"``.
    """
    if isinstance(constraint, dict):
        return "json"
    # Pydantic BaseModel subclass (without importing pydantic eagerly)
    if isinstance(constraint, type):
        try:
            from pydantic import BaseModel  # local import to keep soft
        except ImportError:  # pragma: no cover
            return "json"
        if issubclass(constraint, BaseModel):
            return "json"
        return "json"
    if isinstance(constraint, str):
        stripped = constraint.lstrip()
        if stripped.startswith("{"):
            return "json"
        return "regex"
    return "json"


__all__ = [
    "JsonSchema",
    "constrained_generate",
    "is_available",
]
