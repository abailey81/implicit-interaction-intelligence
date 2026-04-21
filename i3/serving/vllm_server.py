"""vLLM wrapper for serving the I3 SLM at high throughput.

vLLM is relevant once the SLM has been exported to a compatible
checkpoint; currently included as an integration scaffold.

vLLM's PagedAttention is designed for transformer language models with
HuggingFace-compatible checkpoints (config.json + model weights in a
supported layout). The I3 SLM is currently a bespoke nn.Module defined
in :mod:`i3.slm.model` with custom cross-attention conditioning, so it is
**not yet** directly loadable by ``vllm.LLM``. We keep this module as a
forward-looking integration point:

* the :class:`VLLMServer` wrapper shows exactly how we would launch the
  OpenAI-compatible server once an export path exists;
* :func:`build_openai_server_args` produces the argv that would be passed
  to ``vllm.entrypoints.openai.api_server``.

See the :mod:`docs.research.distributed_training` note for a discussion of
when vLLM's overhead is worth paying (concurrent users, long contexts)
and when it is not (tiny models where a simple FastAPI loop wins).

"""

# TODO: PagedAttention requires a format export step.
# vLLM loads models via ``vllm.LLM(model=<hf-path-or-hub-id>)`` which
# expects either a HuggingFace Hub id or a local directory with
# ``config.json``, ``tokenizer.json``, and ``*.safetensors``. The I3 SLM
# is a custom architecture (AdaptationVector + UserStateEmbedding cross-
# attention conditioning) that has no counterpart in the HF model zoo.
# Before this module is live we must either:
#   1. Register a custom vLLM model class that mirrors ``AdaptiveSLM`` and
#      expose it via vLLM's model registry; OR
# 2. Rewrite the SLM on top of a vanilla decoder architecture that maps
#    onto an existing vLLM-supported family (e.g. LlamaForCausalLM).
# Option (2) is cleaner but loses the cross-attention conditioning -- the
# defining feature of the Adaptive SLM. Option (1) preserves the
# architecture at the cost of maintaining the custom vLLM class.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Soft-import vllm
# ---------------------------------------------------------------------------

try:
    import vllm  # type: ignore[import-not-found]

    _VLLM_AVAILABLE = True
except ImportError:
    vllm = None  # type: ignore[assignment]
    _VLLM_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    """Configuration for the vLLM-backed SLM server.

    Attributes:
        model_path: Path or Hub id for the (future) exported SLM.
        host: HTTP bind host.
        port: HTTP bind port.
        dtype: Model dtype, e.g. ``"bfloat16"``.
        tensor_parallel_size: Number of GPUs to shard the model across.
        max_model_len: Max context length the server will accept.
        gpu_memory_utilization: Fraction of GPU memory to reserve for KV
            cache.
    """

    model_path: str
    host: str = "0.0.0.0"  # noqa: S104 - container-local server bind
    port: int = 8000
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9


class VLLMServer:
    """High-throughput vLLM wrapper for the SLM.

    This is an integration scaffold -- see the module-level TODO for the
    export work required before it can load real checkpoints. The class
    intentionally keeps all vLLM imports lazy so static analysis, docs
    builds, and CI on machines without CUDA can still load this module.
    """

    def __init__(self, config: VLLMConfig) -> None:
        """Store configuration; defer engine construction until ``start``.

        Args:
            config: The vLLM server configuration.
        """
        self.config = config
        self._engine: Any = None

    def is_available(self) -> bool:
        """Return whether the ``vllm`` package is importable.

        Returns:
            ``True`` if vLLM can be imported in this interpreter.
        """
        return _VLLM_AVAILABLE

    def start_engine(self) -> Any:
        """Instantiate the offline ``vllm.LLM`` engine.

        Returns:
            The vLLM engine instance.

        Raises:
            RuntimeError: If ``vllm`` is not installed.
        """
        if not _VLLM_AVAILABLE or vllm is None:
            raise RuntimeError(
                "vllm is not installed; install via "
                "`poetry install --with distributed -E vllm` "
                "(note: requires CUDA)."
            )
        self._engine = vllm.LLM(
            model=self.config.model_path,
            dtype=self.config.dtype,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )
        logger.info("vLLM engine started for %s", self.config.model_path)
        return self._engine

    def build_openai_server_args(self) -> list[str]:
        """Return the argv for the OpenAI-compatible server.

        The vLLM package ships ``vllm.entrypoints.openai.api_server`` as a
        runnable module. This helper produces the exact argv list we would
        pass to it.

        Returns:
            A list of strings suitable for ``sys.argv``.
        """
        return [
            "--model",
            self.config.model_path,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--dtype",
            self.config.dtype,
            "--tensor-parallel-size",
            str(self.config.tensor_parallel_size),
            "--max-model-len",
            str(self.config.max_model_len),
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
        ]

    def run_openai_server(self) -> None:
        """Launch the OpenAI-compatible HTTP server in-process.

        Raises:
            RuntimeError: If ``vllm`` is not installed.
        """
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "vllm is not installed; install via "
                "`poetry install --with distributed -E vllm`."
            )
        # Lazy import because the entrypoints subpackage itself needs CUDA.
        from vllm.entrypoints.openai import api_server  # type: ignore[import-not-found]

        argv = self.build_openai_server_args()
        logger.info("Launching vLLM OpenAI server with argv=%s", argv)
        api_server.run_server(argv)  # type: ignore[attr-defined]


def main() -> None:
    """CLI entry point for ad-hoc launching."""
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Launch the vLLM-backed I3 SLM server.")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF hub id.")
    parser.add_argument("--host", type=str, default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    server = VLLMServer(
        VLLMConfig(
            model_path=args.model,
            host=args.host,
            port=args.port,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    )
    if not server.is_available():
        logger.error("vllm is not installed; see the module docstring.")
        return
    server.run_openai_server()


if __name__ == "__main__":
    main()
