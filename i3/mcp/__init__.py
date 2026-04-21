"""Model Context Protocol (MCP) integration for Implicit Interaction Intelligence.

This package exposes a subset of I3's read-only APIs to MCP clients such as
Claude Desktop, Claude Code, or any custom client built on Anthropic's
`Model Context Protocol`_.  MCP is an open standard for connecting AI
assistants to data sources and tools; servers expose **tools**, **resources**,
and **prompts** via stdio, HTTP+SSE, or streamable HTTP transports.

The I3 MCP server (``i3-hmi-companion``) lets a Claude-based client inspect a
user's current adaptation state, pull a feature-attribution breakdown, or
query device profiles -- all WITHOUT ever reading raw user text.  Every read
of user data routes through :class:`i3.privacy.sanitizer.PrivacySanitizer`
and is audit-logged at ``INFO`` with only the ``user_id``.

The MCP SDK is soft-imported: importing this package without ``mcp``
installed still succeeds; constructing :class:`I3MCPServer` without the SDK
raises a clear ``RuntimeError`` telling the user to
``pip install "mcp[cli]"``.

Public surface
--------------
:class:`I3MCPServer`
    The MCP server wrapper that registers tools, resources, and prompts.

.. _Model Context Protocol: https://modelcontextprotocol.io/
"""

from __future__ import annotations

from i3.mcp.server import I3MCPServer

__all__ = ["I3MCPServer"]
