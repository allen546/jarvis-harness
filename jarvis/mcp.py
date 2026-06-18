import json
import os
import sys
import logging
import datetime
from contextlib import AsyncExitStack
from typing import Any
from mcp import StdioServerParameters
from mcp.client.session_group import ClientSessionGroup, SseServerParameters, StreamableHttpParameters
from jarvis.tools import Tool

logger = logging.getLogger(__name__)

class McpClientManager:
    def __init__(self, config_path: str = "config/mcp_settings.json", proxy_env: dict[str, str] | None = None) -> None:
        self.config_path = config_path
        self.group: ClientSessionGroup | None = None
        self.exit_stack = AsyncExitStack()
        self._proxy_env = proxy_env or {}
    async def initialize(self) -> list[Tool]:
        if not os.path.exists(self.config_path):
            return []
            
        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as exc:
                print(f"Error parsing MCP settings JSON: {exc}", file=sys.stderr)
                return []
                
        servers = data.get("mcpServers", {})
        if not servers:
            return []
            
        self.group = ClientSessionGroup(exit_stack=self.exit_stack)
        await self.exit_stack.enter_async_context(self.group)
        
        for name, cfg in servers.items():
            try:
                if "url" in cfg:
                    transport = cfg.get("transport", "sse")
                    if transport == "streamable_http":
                        params = StreamableHttpParameters(
                            url=cfg["url"],
                            headers=cfg.get("headers"),
                            timeout=datetime.timedelta(seconds=cfg.get("timeout", 30)),
                            sse_read_timeout=datetime.timedelta(seconds=cfg.get("sse_read_timeout", 300)),
                            terminate_on_close=cfg.get("terminate_on_close", True),
                        )
                    else:
                        params = SseServerParameters(
                            url=cfg["url"],
                            headers=cfg.get("headers"),
                            timeout=cfg.get("timeout", 5.0),
                            sse_read_timeout=cfg.get("sse_read_timeout", 300.0)
                        )
                else:
                    server_env = cfg.get("env") or {}
                    if self._proxy_env:
                        # Only strip ALL_PROXY if it points to a SOCKS endpoint,
                        # since stdio processes lack socksio.  HTTP/HTTPS proxies
                        # are always safe to pass.
                        filtered = {}
                        for k, v in self._proxy_env.items():
                            if k.lower() in ("all_proxy",):
                                if v.startswith("socks"):
                                    continue  # skip SOCKS — no socksio
                            filtered[k] = v
                        server_env = {**filtered, **server_env}
                    params = StdioServerParameters(
                        command=cfg["command"],
                        args=cfg.get("args", []),
                        env=server_env or None,
                        cwd=cfg.get("cwd", None)
                    )
                await self.group.connect_to_server(params)
            except Exception as exc:
                print(f"Failed to connect to MCP server {name}: {exc}")
                
        jarvis_tools: list[Tool] = []
        for tool_name, mcp_tool in self.group.tools.items():
            jarvis_tools.append(Tool(
                name=tool_name,
                description=mcp_tool.description or "",
                parameters=mcp_tool.inputSchema,
                handler=lambda args, name=tool_name: self.execute_tool(name, args)
            ))
        return jarvis_tools
        
    async def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        if not self.group:
            raise RuntimeError("MCP Client Group not initialized")
        res = await self.group.call_tool(name, args)
        
        parts = []
        for block in res.content:
            if hasattr(block, "text") and block.text:
                parts.append(block.text)
            elif hasattr(block, "data") and block.data:
                parts.append(block.data)
            else:
                parts.append(str(block))
        text_result = "\n".join(parts)
        
        if res.isError:
            raise ValueError(text_result)
        return text_result
        
    async def close(self) -> None:
        try:
            await self.exit_stack.aclose()
        except RuntimeError as exc:
            # anyio cancel scopes are task-bound; close() may be called from
            # a different task (e.g. main shutdown vs. QQ handler).  The OS
            # reaps orphaned child processes on exit, so this is safe during
            # shutdown — log rather than crash.
            logger.warning("mcp: exit_stack close skipped (%s)", exc)
        self.group = None
