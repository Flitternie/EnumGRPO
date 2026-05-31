"""OpenHands tool adapters for this repo.

For fair DB-agent comparisons, we keep only the MCP proxy tool adapter here.
All database functionality is provided by MCP tools exposed from `mcp_server.py`.
"""

from codebase.tools.mcp_proxy import McpProxyTool  # noqa: F401
