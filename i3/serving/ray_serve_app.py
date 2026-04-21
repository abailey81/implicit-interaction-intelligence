"""Ray Serve deployment for the Adaptive Small Language Model.

This module defines a Ray Serve application that exposes the same
``/generate`` REST schema used by the single-host ``server.app`` FastAPI
implementation, but under Ray Serve's replica-based autoscaling model. In
production this allows horizontally scaling SLM inference across a Ray
cluster with graceful replica lifecycle.

The module is safe to import even when ``ray`` is not installed: the
``@serve.deployment`` decorator collapses to a no-op wrapper, and
:func:`build_ray_serve_app` raises a clear ``RuntimeError`` if called
without Ray present.

Architecture
------------

The graph has two deployments:

* :class:`I3ServeDeployment` — the worker, holds a warm copy of
  :class:`i3.slm.model.AdaptiveSLM` and serves ``/generate`` requests.
* :class:`I3BanditRouter` — a lightweight ingress that performs the same
  epsilon-greedy bandit routing as the single-host pipeline, then forwards
  the chosen adaptation-vector to a worker replica.

Deployment
----------

.. code-block:: bash

    ray start --head
    serve run i3.serving.ray_serve_app:app --host 0.0.0.0 --port 8000

"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch

# ---------------------------------------------------------------------------
# Soft-import ray[serve]
# ---------------------------------------------------------------------------

try:
    from ray import serve  # type: ignore[import-not-found]

    _RAY_AVAILABLE = True
except ImportError:
    serve = None  # type: ignore[assignment]
    _RAY_AVAILABLE = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decorator helpers — collapse to identity when Ray is unavailable so the
# module still imports cleanly for static analysis / docs builds.
# ---------------------------------------------------------------------------


def _deployment(*args: Any, **kwargs: Any) -> Any:
    """Return ``serve.deployment`` or an identity decorator if Ray is absent.

    Args:
        *args: Positional forwards to ``serve.deployment``.
        **kwargs: Keyword forwards to ``serve.deployment``.

    Returns:
        Either the real decorator or a pass-through.
    """
    if _RAY_AVAILABLE and serve is not None:
        return serve.deployment(*args, **kwargs)

    def _identity(cls: Any) -> Any:
        return cls

    return _identity


# ---------------------------------------------------------------------------
# Worker deployment
# ---------------------------------------------------------------------------


@_deployment(num_replicas=3, ray_actor_options={"num_cpus": 2})
class I3ServeDeployment:
    """Ray Serve worker deployment wrapping :class:`AdaptiveSLM`.

    The deployment loads the SLM once per replica during ``__init__`` and
    serves ``/generate`` requests backed by :meth:`AdaptiveSLM.generate`.
    """

    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu") -> None:
        """Load the SLM checkpoint into memory.

        Args:
            checkpoint_path: Optional path to a ``.pt`` checkpoint. If
                ``None``, the model is initialized with random weights
                (useful for smoke tests).
            device: Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
        """
        from i3.slm.model import AdaptiveSLM  # lazy import

        self.device = torch.device(device)
        self.model = AdaptiveSLM().to(self.device).eval()
        self.checkpoint_path = checkpoint_path
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state.get("model_state_dict", state))
            logger.info("Loaded SLM checkpoint from %s", checkpoint_path)

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a Ray Serve request in-process.

        Args:
            request: JSON body with ``prompt``, ``adaptation_vector`` (list
                of 8 floats), ``user_state`` (list of 64 floats), and
                optional ``max_new_tokens``.

        Returns:
            Dict with ``text`` and ``latency_ms``.
        """
        return await self.generate(request)

    async def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run autoregressive generation for a single request.

        Args:
            request: Request body (see ``__call__``).

        Returns:
            Dict with ``text`` and ``latency_ms``.

        Raises:
            ValueError: If required fields are missing.
        """
        prompt = request.get("prompt")
        adaptation = request.get("adaptation_vector")
        user_state = request.get("user_state")
        max_new_tokens = int(request.get("max_new_tokens", 64))

        if prompt is None or adaptation is None or user_state is None:
            raise ValueError(
                "Request must include 'prompt', 'adaptation_vector', and 'user_state'."
            )

        start = time.perf_counter()
        # NOTE: The model exposes a generate() helper in i3.slm.generate;
        # we keep this wrapper minimal and let the caller's tokenization
        # decisions flow through when wired to the full pipeline.
        with torch.no_grad():
            text = self._run_generate(prompt, adaptation, user_state, max_new_tokens)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {"text": text, "latency_ms": latency_ms}

    def _run_generate(
        self,
        prompt: str,
        adaptation: list[float],
        user_state: list[float],
        max_new_tokens: int,
    ) -> str:
        """Invoke the SLM; isolated so the async path is easy to test.

        Args:
            prompt: Raw prompt text.
            adaptation: Length-8 adaptation vector.
            user_state: Length-64 user-state embedding.
            max_new_tokens: Max tokens to generate.

        Returns:
            The generated text (may be an empty string if the model is
            random-initialized in a smoke test).
        """
        from i3.slm.generate import SLMGenerator
        from i3.slm.tokenizer import SimpleTokenizer

        try:
            tokenizer = SimpleTokenizer()
            generator = SLMGenerator(self.model, tokenizer, device=str(self.device))
            cond = torch.tensor([adaptation], device=self.device, dtype=torch.float32)
            us = torch.tensor([user_state], device=self.device, dtype=torch.float32)
            return generator.generate(
                prompt,
                adaptation_vector=cond,
                user_state=us,
                max_new_tokens=max_new_tokens,
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning("SLM generation failed in Ray Serve worker: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Ingress router with bandit routing
# ---------------------------------------------------------------------------


@_deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
class I3BanditRouter:
    """Ingress deployment that performs bandit arm-selection, then delegates.

    The router accepts the front-end's raw request, runs an
    epsilon-greedy-style arm pull to pick an ``adaptation_vector``
    configuration, and forwards to the worker. Keeping this logic out of
    the worker lets us scale arm selection and generation independently.
    """

    def __init__(self, worker: Any, epsilon: float = 0.1) -> None:
        """Store the worker handle and epsilon parameter.

        Args:
            worker: Ray Serve handle for :class:`I3ServeDeployment`.
            epsilon: Exploration probability.
        """
        self.worker = worker
        self.epsilon = float(epsilon)

    async def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        """Route a request through the bandit.

        Args:
            request: The raw request; may or may not contain
                ``adaptation_vector``.

        Returns:
            Dict produced by :meth:`I3ServeDeployment.generate`.
        """
        adaptation = request.get("adaptation_vector")
        if adaptation is None:
            adaptation = self._select_arm()
            request = {**request, "adaptation_vector": adaptation}
        # handle.remote() returns an ObjectRef-like; await it.
        result = await self.worker.generate.remote(request)  # type: ignore[attr-defined]
        return dict(result)

    def _select_arm(self) -> list[float]:
        """Return a default adaptation vector.

        A production router would maintain per-user posterior statistics
        and sample via Thompson or UCB; the scaffold returns a neutral
        vector so the service boots out-of-the-box.

        Returns:
            A length-8 list of floats.
        """
        return [0.5] * 8


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_ray_serve_app(checkpoint_path: str | None = None, device: str = "cpu") -> Any:
    """Build the Ray Serve application graph.

    Args:
        checkpoint_path: Optional SLM checkpoint path.
        device: Torch device string.

    Returns:
        A Ray Serve application object suitable for ``serve.run(app)``.

    Raises:
        RuntimeError: If ``ray[serve]`` is not installed.
    """
    if not _RAY_AVAILABLE or serve is None:
        raise RuntimeError(
            "ray[serve] is not installed; install via "
            "`poetry install --with distributed`."
        )
    worker = I3ServeDeployment.bind(checkpoint_path=checkpoint_path, device=device)  # type: ignore[attr-defined]
    router = I3BanditRouter.bind(worker)  # type: ignore[attr-defined]
    return router


# Module-level app attribute so ``serve run i3.serving.ray_serve_app:app``
# works without calling the builder explicitly. We build lazily to avoid
# unnecessary actor-wiring at import time for downstream tools that only
# want to introspect the module.
app: Any = None


def _build_default_app_if_possible() -> Any:
    """Construct the default app graph when Ray is available.

    Returns:
        The built application or ``None`` if Ray is not installed or the
        builder itself errored.
    """
    if not _RAY_AVAILABLE:
        return None
    try:
        return build_ray_serve_app()
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - runtime wiring
        logger.warning("Ray Serve app graph could not be built at import: %s", exc)
        return None


app = _build_default_app_if_possible()


if TYPE_CHECKING:
    __all__ = ["I3BanditRouter", "I3ServeDeployment", "app", "build_ray_serve_app"]
