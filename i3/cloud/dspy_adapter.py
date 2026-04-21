"""DSPy adapter for compile-time optimisation of the I3 adaptive prompt.

This module replaces the hand-written :class:`PromptBuilder` with a
DSPy program.  In the DSPy paradigm the developer writes a declarative
*signature* plus a :class:`~dspy.Module`, and a **teleprompter** compiles
the program into an optimised set of few-shot demonstrations and
instructions that maximise a supplied metric.  For I3 the metric is
conditioning-fidelity drawn from
:mod:`i3.eval.responsiveness_golden`.

The adapter is a strict *addition*: the core cloud client and prompt
builder in ``i3/cloud/client.py`` and ``i3/cloud/prompt_builder.py`` are
untouched, so the pipeline continues to boot even if ``dspy-ai`` is
absent.

References:
    Khattab, O. *et al.* (2023). **DSPy: Compiling Declarative Language
    Model Calls into State-of-the-Art Pipelines.**  arXiv:2310.03714.
    (See Sec. 4 for the teleprompter formulation this module follows.)

Install hint::

    pip install "dspy-ai>=2.5"
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft import of the DSPy SDK
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment-dependent
    import dspy  # type: ignore[import-not-found]

    _DSPY_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - exercised when dep absent
    dspy = None  # type: ignore[assignment]
    _DSPY_AVAILABLE = False


_INSTALL_HINT: str = (
    "DSPy is not installed. Install it with: "
    '`pip install "dspy-ai>=2.5"`  '
    "(or add it to the [tool.poetry.group.llm-ecosystem.dependencies] group)."
)


def is_available() -> bool:
    """Return ``True`` iff the ``dspy`` SDK is importable."""
    return _DSPY_AVAILABLE


def _require_dspy() -> None:
    """Raise :class:`ImportError` with a friendly install hint if DSPy absent."""
    if not _DSPY_AVAILABLE:
        raise ImportError(_INSTALL_HINT)


# ---------------------------------------------------------------------------
# Signatures (input/output contract)
# ---------------------------------------------------------------------------

if _DSPY_AVAILABLE:  # pragma: no cover - only exercised when SDK present

    class AdaptiveResponseSignature(dspy.Signature):  # type: ignore[misc, name-defined]
        """Declarative I/O contract for I3's adaptive responder.

        The signature deliberately mirrors the ``AdaptationVector`` axes
        used by the on-device SLM so that the compiled program produces
        responses consistent with the local model.
        """

        user_state = dspy.InputField(  # type: ignore[attr-defined]
            desc=(
                "Compact, privacy-safe summary of the user's current "
                "cognitive and emotional state. Never contains raw text."
            )
        )
        adaptation_vector = dspy.InputField(  # type: ignore[attr-defined]
            desc=(
                "8-dimensional adaptation vector rendered as a dict of "
                "(formality, verbosity, emotionality, directness, "
                "cognitive_load, simplification, urgency, empathy)."
            )
        )
        message = dspy.InputField(desc="Current user message.")  # type: ignore[attr-defined]
        response = dspy.OutputField(  # type: ignore[attr-defined]
            desc=(
                "Assistant reply, concise by default, mirroring the "
                "adaptation axes. No PII, no safety-flagged content."
            )
        )
else:  # pragma: no cover - placeholder so attribute access does not crash
    AdaptiveResponseSignature = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Program definition
# ---------------------------------------------------------------------------


@dataclass
class CompiledProgramInfo:
    """Lightweight descriptor of a compiled DSPy program on disk.

    Attributes:
        path: Absolute path to the JSON artefact.
        teleprompter: Name of the teleprompter that produced it (e.g.
            ``"BootstrapFewShot"`` or ``"MIPROv2"``).
        metric_name: Human-readable name of the metric that was maximised.
        train_size: Number of training examples supplied.
        dev_size: Number of dev examples supplied.
    """

    path: Path
    teleprompter: str
    metric_name: str
    train_size: int
    dev_size: int


class I3AdaptivePromptProgram:
    """DSPy module that generates an adaptive response.

    Replaces the hand-written ``PromptBuilder`` at compile-time.  Uses a
    :class:`dspy.ChainOfThought` sub-module so the teleprompter can
    optimise the intermediate reasoning prompt alongside the final
    response format.

    Args:
        lm: Optional configured :class:`dspy.LM`.  If omitted, callers
            must have already run ``dspy.settings.configure(lm=...)``.

    Raises:
        ImportError: If ``dspy-ai`` is not installed.
    """

    def __init__(self, lm: Optional[Any] = None) -> None:
        _require_dspy()
        # Inherit from dspy.Module dynamically so the class is still
        # importable (for type checking, docstrings, etc.) when dspy is
        # not present.
        assert dspy is not None
        self._module = _make_module(lm)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def forward(
        self,
        user_state: str,
        adaptation_vector: dict[str, float] | str,
        message: str,
    ) -> Any:
        """Run the program and return a DSPy prediction.

        Args:
            user_state: Privacy-safe summary string (no raw text).
            adaptation_vector: Eight-dimensional vector as a dict or
                its JSON-encoded string representation.
            message: The current user message.

        Returns:
            A :class:`dspy.Prediction` with attribute ``response``.
        """
        if isinstance(adaptation_vector, dict):
            av_str = json.dumps(adaptation_vector, sort_keys=True)
        else:
            av_str = str(adaptation_vector)
        return self._module(
            user_state=user_state,
            adaptation_vector=av_str,
            message=message,
        )

    # Callable alias -- matches DSPy convention
    def __call__(
        self,
        user_state: str,
        adaptation_vector: dict[str, float] | str,
        message: str,
    ) -> Any:
        return self.forward(user_state, adaptation_vector, message)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | os.PathLike[str]) -> None:
        """Persist the compiled demonstrations/instructions to disk.

        The JSON format is the DSPy-native format produced by
        ``dspy.Module.save`` so the artefact can be loaded by any other
        DSPy runtime.
        """
        _require_dspy()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # dspy.Module.save signature differs across minor versions; we
        # call the most generic form.
        self._module.save(str(target))

    def load(self, path: str | os.PathLike[str]) -> None:
        """Load demonstrations/instructions from a previous compile."""
        _require_dspy()
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"No compiled DSPy program at: {source}")
        self._module.load(str(source))


def _make_module(lm: Optional[Any]) -> Any:
    """Construct the underlying :class:`dspy.Module` instance.

    Kept as a free function so it can be monkey-patched in tests.
    """
    _require_dspy()
    assert dspy is not None

    class _Inner(dspy.Module):  # type: ignore[misc, name-defined]
        def __init__(self, lm_inner: Optional[Any]) -> None:
            super().__init__()
            if lm_inner is not None:
                # Per-module LM override (dspy >=2.5)
                try:
                    self.set_lm(lm_inner)
                except AttributeError:  # pragma: no cover - legacy dspy
                    dspy.settings.configure(lm=lm_inner)
            self.think = dspy.ChainOfThought(AdaptiveResponseSignature)

        def forward(
            self,
            user_state: str,
            adaptation_vector: str,
            message: str,
        ) -> Any:
            return self.think(
                user_state=user_state,
                adaptation_vector=adaptation_vector,
                message=message,
            )

    return _Inner(lm)


# ---------------------------------------------------------------------------
# Teleprompter-driven optimisation
# ---------------------------------------------------------------------------


MetricFn = Callable[[Any, Any, Optional[Any]], float]


def optimize_program(
    train_set: Sequence[Any],
    dev_set: Sequence[Any],
    metric: MetricFn,
    *,
    program: Optional[I3AdaptivePromptProgram] = None,
    teleprompter: str = "BootstrapFewShot",
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 8,
    save_path: Optional[str | os.PathLike[str]] = None,
) -> tuple[I3AdaptivePromptProgram, CompiledProgramInfo]:
    """Compile :class:`I3AdaptivePromptProgram` with a teleprompter.

    Args:
        train_set: Sequence of ``dspy.Example`` for bootstrapping.
        dev_set: Sequence of ``dspy.Example`` for metric evaluation.
        metric: Callable ``(gold, pred, trace) -> float``.  Higher is
            better.  For I3 use the conditioning-fidelity metric from
            :mod:`i3.eval.responsiveness_golden`.
        program: Optional existing program instance to compile.  A new
            one is constructed when omitted.
        teleprompter: One of ``"BootstrapFewShot"`` or ``"MIPROv2"``.
        max_bootstrapped_demos: Passed through to the teleprompter.
        max_labeled_demos: Passed through to the teleprompter.
        save_path: Optional location at which to persist the compiled
            artefact.  When omitted the artefact is held in-memory only.

    Returns:
        Tuple ``(compiled_program, info)``.

    Raises:
        ImportError: If ``dspy-ai`` is not installed.
        ValueError: If *teleprompter* is not recognised.
    """
    _require_dspy()
    assert dspy is not None

    prog = program if program is not None else I3AdaptivePromptProgram()

    tele_name = teleprompter.strip()
    if tele_name == "BootstrapFewShot":
        # dspy.teleprompt namespace has been stable since v2.3
        from dspy.teleprompt import BootstrapFewShot  # type: ignore[import-not-found]

        opt = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
        )
        compiled = opt.compile(prog._module, trainset=list(train_set))
    elif tele_name == "MIPROv2":
        from dspy.teleprompt import MIPROv2  # type: ignore[import-not-found]

        opt = MIPROv2(metric=metric, auto="light")
        compiled = opt.compile(
            prog._module,
            trainset=list(train_set),
            valset=list(dev_set),
        )
    else:
        raise ValueError(
            f"Unknown teleprompter: {teleprompter!r}. "
            "Use 'BootstrapFewShot' or 'MIPROv2'."
        )

    # Re-wrap the compiled module
    prog._module = compiled

    if save_path is not None:
        prog.save(save_path)
        artefact_path = Path(save_path).resolve()
    else:
        artefact_path = Path("<in-memory>")

    info = CompiledProgramInfo(
        path=artefact_path,
        teleprompter=tele_name,
        metric_name=getattr(metric, "__name__", "custom_metric"),
        train_size=len(train_set),
        dev_size=len(dev_set),
    )
    logger.info(
        "DSPy program compiled (teleprompter=%s, train=%d, dev=%d, path=%s)",
        tele_name,
        info.train_size,
        info.dev_size,
        info.path,
    )
    return prog, info


__all__ = [
    "AdaptiveResponseSignature",
    "CompiledProgramInfo",
    "I3AdaptivePromptProgram",
    "MetricFn",
    "is_available",
    "optimize_program",
]
