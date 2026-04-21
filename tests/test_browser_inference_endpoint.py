"""Tests for ``server.routes_inference``.

Covers three properties:

1. A valid request for an existing model returns ``200`` with the
   full cross-origin-isolation header set.
2. A missing model returns ``404`` with a clear, non-revealing detail.
3. A path-traversal payload (``..`` in the filename) is rejected
   before the filesystem is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes_inference import include_inference_routes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def inference_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Return a minimal FastAPI ``TestClient`` with the inference router.

    The ``web/models`` root is redirected at ``tmp_path / web / models``
    via a monkey-patch of :data:`server.routes_inference._MODELS_SUBPATH`
    so the tests do not clobber the real repository checkout.
    """
    # Redirect the models root inside a temporary working tree.
    models_dir = tmp_path / "web" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    import server.routes_inference as ri

    monkeypatch.setattr(ri, "_MODELS_SUBPATH", models_dir)

    app = FastAPI()
    include_inference_routes(app)
    client = TestClient(app)
    # Stash the tmp dir on the client for the individual tests.
    client.models_dir = models_dir  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_200_with_isolation_headers_when_model_exists(
    inference_app: TestClient,
) -> None:
    """A valid model request streams bytes with full COOP/COEP headers."""
    models_dir: Path = inference_app.models_dir  # type: ignore[attr-defined]
    dummy = models_dir / "encoder_int8.onnx"
    dummy.write_bytes(b"ONNXDUMMYBYTES")

    resp = inference_app.get("/api/onnx/encoder_int8.onnx")

    assert resp.status_code == 200
    assert resp.content == b"ONNXDUMMYBYTES"
    # Content-Type is locked to octet-stream so browsers never sniff
    # a model blob into text/HTML.
    assert resp.headers["content-type"].startswith("application/octet-stream")
    # Cross-origin isolation pair — both required for SharedArrayBuffer.
    assert resp.headers["cross-origin-opener-policy"] == "same-origin"
    assert resp.headers["cross-origin-embedder-policy"] == "require-corp"
    # Defence-in-depth hardening.
    assert resp.headers.get("x-content-type-options") == "nosniff"


def test_returns_404_when_model_missing(inference_app: TestClient) -> None:
    """Missing model yields a 404 with a friendly, non-revealing detail."""
    resp = inference_app.get("/api/onnx/does_not_exist.onnx")

    assert resp.status_code == 404
    body = resp.json()
    assert "detail" in body
    # The detail should be a helpful user-facing message — we only check
    # that it does not leak the resolved absolute path.
    assert "Model not found" in body["detail"]
    assert "/tmp" not in body["detail"] and "C:\\" not in body["detail"]


def test_rejects_path_traversal(inference_app: TestClient) -> None:
    """``..`` in the filename must be rejected before disk access."""
    # FastAPI normalises URL path segments, so a raw ``..`` in the path
    # is ambiguous across HTTP clients.  We therefore exercise two
    # realistic attacker shapes: a traversal in the filename itself,
    # and an encoded traversal.
    attacks = [
        "/api/onnx/..%2F..%2Fetc%2Fpasswd",
        "/api/onnx/..%2Fapp.py",
        "/api/onnx/evil..onnx",  # double-dot embedded in filename
    ]
    for url in attacks:
        resp = inference_app.get(url)
        # We never want a 200 for any of these.
        assert resp.status_code in (404, 422), (url, resp.status_code)
        if resp.status_code == 404:
            assert "Model not found" in resp.json().get("detail", "")
