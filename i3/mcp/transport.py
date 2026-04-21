"""Configurable transports for :class:`i3.mcp.server.I3MCPServer`.

Anthropic's MCP specification defines several transports; this module
switches between them by name:

* ``stdio`` — default transport used by Claude Desktop / Claude Code.
  Spawns the MCP server as a child process and communicates over its
  standard input/output streams.
* ``sse`` — HTTP + Server-Sent Events.  Useful for hosting the MCP server
  behind a reverse proxy when clients cannot spawn a local subprocess.
* ``streamable_http`` — the newer chunked-streaming transport introduced
  in 2025 which replaces SSE for long-lived sessions.

Each transport is soft-imported so that we can degrade cleanly when the
installed ``mcp`` SDK does not ship that particular backend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from i3.mcp.server import I3MCPServer


#: Supported transport names (kept in sync with the CLI wrapper).
SUPPORTED_TRANSPORTS: tuple[str, ...] = ("stdio", "sse", "streamable_http")


def run_transport(
    server: I3MCPServer,
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Run *server* using the requested transport (blocking call).

    Args:
        server: A fully-constructed :class:`I3MCPServer`.
        transport: One of :data:`SUPPORTED_TRANSPORTS`.
        host: Bind host for HTTP-based transports (ignored for stdio).
        port: Bind port for HTTP-based transports (ignored for stdio).

    Raises:
        ValueError: If *transport* is not recognised.
        RuntimeError: If the installed mcp SDK is missing the backend.
    """
    transport = transport.lower().strip()
    if transport not in SUPPORTED_TRANSPORTS:
        raise ValueError(
            f"Unsupported transport {transport!r}; "
            f"choose one of {list(SUPPORTED_TRANSPORTS)}"
        )

    logger.info(
        "mcp_transport starting transport=%s host=%s port=%s",
        transport,
        host,
        port,
    )

    if transport == "stdio":
        _run_stdio(server)
    elif transport == "sse":
        _run_sse(server, host=host, port=port)
    else:  # "streamable_http"
        _run_streamable_http(server, host=host, port=port)


# ---------------------------------------------------------------------------
# Individual transport implementations
# ---------------------------------------------------------------------------


def _run_stdio(server: I3MCPServer) -> None:
    """Run the server over stdio — the Claude Desktop default."""
    mcp_app = server.mcp
    # Preferred: FastMCP exposes a convenience runner.
    runner = getattr(mcp_app, "run", None)
    if callable(runner):
        try:
            runner(transport="stdio")
            return
        except TypeError:
            # Older FastMCP signatures had no ``transport`` kwarg and
            # default to stdio already.
            runner()
            return
    # Fallback to the low-level stdio helper.
    try:
        from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Installed mcp SDK does not expose stdio_server; "
            "upgrade mcp[cli]."
        ) from exc

    import anyio

    async def _main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            # ``FastMCP`` exposes an ``_mcp_server`` attribute with an async
            # ``run`` method that takes (read, write, init_options).
            low_level = getattr(mcp_app, "_mcp_server", None) or mcp_app
            init_opts = getattr(low_level, "create_initialization_options", None)
            if callable(init_opts):
                await low_level.run(read_stream, write_stream, init_opts())
            else:
                await low_level.run(read_stream, write_stream)

    anyio.run(_main)


def _run_sse(server: I3MCPServer, *, host: str, port: int) -> None:
    """Run the server over HTTP + Server-Sent Events."""
    mcp_app = server.mcp
    runner = getattr(mcp_app, "run", None)
    if callable(runner):
        try:
            runner(transport="sse", host=host, port=port)
            return
        except TypeError:
            pass  # Older signature — fall through.

    try:
        from mcp.server.sse import SseServerTransport  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Installed mcp SDK does not expose SseServerTransport; "
            "upgrade mcp[cli] or use --transport stdio."
        ) from exc

    try:
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "SSE transport requires uvicorn + starlette. "
            "Install with `pip install uvicorn starlette`."
        ) from exc

    sse = SseServerTransport("/messages/")
    low_level = getattr(mcp_app, "_mcp_server", None) or mcp_app

    async def _handle_sse(request):  # type: ignore[no-untyped-def]
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (read, write):
            init_opts = getattr(low_level, "create_initialization_options", None)
            if callable(init_opts):
                await low_level.run(read, write, init_opts())
            else:
                await low_level.run(read, write)

    app = Starlette(
        routes=[
            Route("/sse", endpoint=_handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ]
    )
    uvicorn.run(app, host=host, port=port)


def _run_streamable_http(server: I3MCPServer, *, host: str, port: int) -> None:
    """Run the server over the streamable HTTP transport."""
    mcp_app = server.mcp
    runner = getattr(mcp_app, "run", None)
    if callable(runner):
        try:
            runner(transport="streamable-http", host=host, port=port)
            return
        except TypeError:
            try:
                runner(transport="streamable_http", host=host, port=port)
                return
            except TypeError:
                pass

    raise RuntimeError(
        "Installed mcp SDK does not expose the streamable-http transport; "
        "upgrade mcp[cli] (>=1.0) or use --transport stdio|sse."
    )


__all__ = ["SUPPORTED_TRANSPORTS", "run_transport"]
