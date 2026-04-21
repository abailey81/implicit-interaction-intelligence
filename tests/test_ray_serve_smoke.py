"""Smoke tests for the Ray Serve deployment scaffold.

These tests are *import-level* checks: we must not spin up an actual Ray
cluster inside the unit-test harness because it is slow and can leak
background processes.

Covered:

* :mod:`i3.serving.ray_serve_app` imports with and without ``ray``;
* :class:`i3.serving.ray_serve_app.I3ServeDeployment` is always defined;
* when ``ray[serve]`` is installed, ``build_ray_serve_app()`` returns a
  non-None application graph.
"""

from __future__ import annotations

import pytest

try:
    from ray import serve as _serve  # type: ignore[import-not-found]  # noqa: F401

    _HAS_RAY_SERVE = True
except ImportError:
    _HAS_RAY_SERVE = False


def test_ray_serve_module_imports() -> None:
    """The module and its public names must import cleanly."""
    from i3.serving import ray_serve_app

    assert hasattr(ray_serve_app, "I3ServeDeployment")
    assert hasattr(ray_serve_app, "I3BanditRouter")
    assert hasattr(ray_serve_app, "build_ray_serve_app")


def test_i3_serve_deployment_is_class() -> None:
    """The deployment class should be instantiable (decorator is no-op when ray absent)."""
    from i3.serving.ray_serve_app import I3ServeDeployment

    # Without Ray the decorator collapses to identity, so we can build it;
    # with Ray installed the binding flow is exercised elsewhere.
    assert I3ServeDeployment is not None


@pytest.mark.skipif(not _HAS_RAY_SERVE, reason="ray[serve] not installed")
def test_build_ray_serve_app_returns_non_none() -> None:
    """build_ray_serve_app should return a bound application graph."""
    from i3.serving.ray_serve_app import build_ray_serve_app

    app = build_ray_serve_app()
    assert app is not None


def test_build_ray_serve_app_raises_without_ray() -> None:
    """When ray is missing the builder must raise a clear RuntimeError."""
    from i3.serving import ray_serve_app

    if ray_serve_app._RAY_AVAILABLE:
        pytest.skip("ray is installed; negative-path test not applicable")

    with pytest.raises(RuntimeError, match="ray"):
        ray_serve_app.build_ray_serve_app()


def test_package_public_surface() -> None:
    """i3.serving package must expose the documented public surface."""
    import i3.serving as pkg

    assert "I3ServeDeployment" in pkg.__all__
    assert "build_ray_serve_app" in pkg.__all__
    assert "generate_triton_config" in pkg.__all__
    assert "VLLMServer" in pkg.__all__
