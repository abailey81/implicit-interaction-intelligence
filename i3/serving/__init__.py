"""High-throughput serving surfaces for I3.

This package collects the *scale-out* serving paths for the Implicit
Interaction Intelligence system. All integrations are soft-imported: the
package can be imported even when the heavy dependencies (``ray``,
``vllm``, ``tritonclient``) are absent. The primary ``server.app`` FastAPI
application remains the single-host reference; these modules layer on
additional deployment options as the user base grows.

Public surface
--------------

:func:`build_ray_serve_app`
    Returns the Ray Serve application graph for the SLM.
:class:`I3ServeDeployment`
    The Ray Serve deployment class that wraps :class:`AdaptiveSLM`.
:func:`generate_triton_config`
    Emits a Triton Inference Server ``config.pbtxt`` for the TCN encoder.
:class:`VLLMServer`
    vLLM-backed OpenAI-compatible server wrapper for the SLM (scaffold).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "I3ServeDeployment",
    "VLLMServer",
    "build_ray_serve_app",
    "generate_triton_config",
]

if TYPE_CHECKING:
    # Static-only imports to keep runtime light and avoid hard failures
    # when optional dependencies are missing.
    from i3.serving.ray_serve_app import I3ServeDeployment, build_ray_serve_app
    from i3.serving.triton_config import generate_triton_config
    from i3.serving.vllm_server import VLLMServer


def __getattr__(name: str) -> object:
    """Lazily import public names to keep optional deps soft.

    Args:
        name: The attribute name requested.

    Returns:
        The resolved symbol.

    Raises:
        AttributeError: If the name is not part of the public surface.
    """
    if name in {"I3ServeDeployment", "build_ray_serve_app"}:
        from i3.serving import ray_serve_app as _mod

        return getattr(_mod, name)
    if name == "generate_triton_config":
        from i3.serving import triton_config as _mod

        return _mod.generate_triton_config
    if name == "VLLMServer":
        from i3.serving import vllm_server as _mod

        return _mod.VLLMServer
    raise AttributeError(f"module 'i3.serving' has no attribute {name!r}")
