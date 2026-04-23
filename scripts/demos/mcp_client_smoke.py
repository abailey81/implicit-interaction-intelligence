"""Smoke test that exercises the I3 MCP server as a real MCP client.

The script:

1. Spawns ``python -m scripts.run_mcp_server --transport stdio --no-pipeline``
   as a child process.
2. Connects to it as an MCP client using the official ``mcp`` SDK over
   stdio.
3. Lists tools, resources, and prompts.
4. Calls ``get_device_profile("kirin_9000")`` and asserts the response
   shape.
5. Exits with status ``0`` on success, ``1`` on assertion failure, ``2``
   if the ``mcp`` package is not installed.

Run with::

    python -m scripts.mcp_client_smoke
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


def _require_sdk() -> None:
    """Abort with exit code ``2`` if the mcp SDK is unavailable."""
    try:
        import mcp  # noqa: F401
    except Exception:
        sys.stderr.write("mcp package not installed. `pip install mcp[cli]`\n")
        sys.exit(2)


async def _run_smoke() -> int:
    """Execute the smoke test.

    Returns:
        ``0`` on success, ``1`` on assertion failure.
    """
    from mcp import ClientSession, StdioServerParameters  # type: ignore[import-not-found]
    from mcp.client.stdio import stdio_client  # type: ignore[import-not-found]

    # Spawn the server as a child process over stdio.
    cmd = sys.executable
    args = [
        "-m",
        "scripts.run_mcp_server",
        "--transport",
        "stdio",
        "--no-pipeline",
    ]
    params = StdioServerParameters(
        command=cmd,
        args=args,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    logger.info("mcp_smoke connecting cmd=%s args=%s", cmd, args)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # -- List tools -------------------------------------------
            tools = await session.list_tools()
            tool_names = {t.name for t in getattr(tools, "tools", [])}
            expected_tools = {
                "get_user_adaptation_vector",
                "get_user_state_embedding",
                "get_feature_vector",
                "get_session_metadata",
                "get_device_profile",
                "route_recommendation",
                "explain_adaptation",
            }
            missing = expected_tools - tool_names
            if missing:
                sys.stderr.write(
                    f"missing tools: {sorted(missing)}; saw: {sorted(tool_names)}\n"
                )
                return 1
            logger.info("mcp_smoke tools_ok count=%d", len(tool_names))

            # -- List resources ---------------------------------------
            try:
                resources = await session.list_resources()
                resource_count = len(getattr(resources, "resources", []) or [])
            except Exception as exc:  # noqa: BLE001 — optional in some SDKs
                logger.warning("mcp_smoke list_resources_err %s", exc)
                resource_count = 0
            logger.info("mcp_smoke resources_listed count=%d", resource_count)

            # -- Call get_device_profile -------------------------------
            result = await session.call_tool(
                "get_device_profile", {"target": "kirin_9000"}
            )
            payload = _extract_payload(result)
            if not isinstance(payload, dict):
                sys.stderr.write(f"unexpected tool result type: {type(payload)}\n")
                return 1
            if payload.get("target") != "kirin_9000":
                sys.stderr.write(f"unexpected target: {payload.get('target')}\n")
                return 1
            profile = payload.get("profile")
            if not isinstance(profile, dict):
                sys.stderr.write(f"profile missing or wrong type: {profile!r}\n")
                return 1
            required_fields = {"name", "device_class", "ram_mb", "int8_tops"}
            for field in required_fields:
                if field not in profile:
                    sys.stderr.write(
                        f"profile missing field {field!r}: {list(profile.keys())}\n"
                    )
                    return 1
            logger.info("mcp_smoke device_profile_ok name=%s", profile.get("name"))

    return 0


def _extract_payload(tool_result: Any) -> Any:
    """Pull the dict payload out of an MCP ``call_tool`` result.

    FastMCP wraps JSON results as either ``structuredContent`` or as a list
    of ``TextContent`` blocks with ``type="text"``.

    Args:
        tool_result: Whatever ``ClientSession.call_tool`` returned.

    Returns:
        Best-effort-parsed dict/list/primitive.
    """
    structured = getattr(tool_result, "structuredContent", None)
    if isinstance(structured, dict) and structured:
        return structured
    content = getattr(tool_result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return None


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=os.environ.get("I3_MCP_SMOKE_LOG_LEVEL", "INFO"),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _require_sdk()
    try:
        return asyncio.run(_run_smoke())
    except Exception as exc:  # noqa: BLE001 — report and bail
        logger.exception("mcp_smoke_failed %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
