"""CLI entry point for the I3 MCP server.

Example
-------

Run over stdio (Claude Desktop integration)::

    python -m scripts.run_mcp_server --transport stdio

Run over HTTP + SSE::

    python -m scripts.run_mcp_server --transport sse --port 8765

Run over streamable HTTP::

    python -m scripts.run_mcp_server --transport http --port 8765

Environment variables
---------------------

``I3_MCP_LOG_LEVEL``
    Override the Python log level (default ``INFO``).  Every MCP tool
    invocation is already logged at ``INFO`` with the user id only.

``I3_MCP_NAME``
    Override the advertised server name (default ``i3-hmi-companion``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Sequence


_TRANSPORT_ALIASES: dict[str, str] = {
    "stdio": "stdio",
    "sse": "sse",
    "http": "streamable_http",
    "streamable-http": "streamable_http",
    "streamable_http": "streamable_http",
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional pre-split argv for testing.  Defaults to ``sys.argv``.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        prog="run_mcp_server",
        description="Run the I3 Model Context Protocol server.",
    )
    parser.add_argument(
        "--transport",
        choices=sorted(_TRANSPORT_ALIASES),
        default="stdio",
        help="MCP transport. 'http' is an alias for 'streamable_http'.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for HTTP-based transports (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port for HTTP-based transports (default: 8765).",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("I3_MCP_LOG_LEVEL", "INFO"),
        help="Python logging level (default: INFO).",
    )
    parser.add_argument(
        "--name",
        default=os.environ.get("I3_MCP_NAME", "i3-hmi-companion"),
        help="Server name advertised to MCP clients.",
    )
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help=(
            "Do not attach an I3 Pipeline. Useful for smoke tests; "
            "user-scoped tools will return {'error': 'pipeline_unavailable'}."
        ),
    )
    return parser.parse_args(argv)


def _configure_logging(level_name: str) -> None:
    """Configure root logging for the MCP server process.

    Args:
        level_name: Python logging level name (e.g. ``"INFO"``).
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    # stdio transport reserves STDOUT for framing — route logs to STDERR.
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_server(name: str, no_pipeline: bool):  # type: ignore[no-untyped-def]
    """Construct :class:`I3MCPServer`, attaching a pipeline on request.

    Args:
        name: Server name.
        no_pipeline: If ``True``, do not try to initialise the pipeline.

    Returns:
        A ready-to-serve :class:`I3MCPServer`.
    """
    from i3.mcp.server import I3MCPServer

    pipeline = None
    if not no_pipeline:
        try:
            from i3.config import load_config
            from i3.pipeline.engine import Pipeline

            cfg = load_config()
            pipeline = Pipeline(cfg)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "Pipeline unavailable (%s). Starting MCP server in standalone mode.",
                exc,
            )
            pipeline = None

    return I3MCPServer(pipeline=pipeline, name=name)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Optional pre-split argv for testing.

    Returns:
        Process exit code (``0`` on graceful shutdown).
    """
    args = _parse_args(argv)
    _configure_logging(args.log_level)
    transport = _TRANSPORT_ALIASES[args.transport]

    try:
        server = _build_server(name=args.name, no_pipeline=args.no_pipeline)
    except RuntimeError as exc:
        sys.stderr.write(f"MCP startup error: {exc}\n")
        return 2

    logging.getLogger(__name__).info(
        "i3_mcp_server ready name=%s transport=%s host=%s port=%s",
        server.name,
        transport,
        args.host,
        args.port,
    )
    try:
        server.run(transport=transport)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("i3_mcp_server shutdown (SIGINT)")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
