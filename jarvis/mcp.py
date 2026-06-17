import json
import os
from typing import Any
from mcp import StdioServerParameters
from mcp.client.session_group import ClientSessionGroup, SseServerParameters
from jarvis.tools import Tool

class McpClientManager:
    def __init__(self, config_path: str = "config/mcp_settings.json") -> None:
        self.config_path = config_path
        self.group: ClientSessionGroup | None = None
        
    async def initialize(self) -> list[Tool]:
        if not os.path.exists(self.config_path):
            return []
            
        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                return []
                
        servers = data.get("mcpServers", {})
        if not servers:
            return []
            
        self.group = ClientSessionGroup()
        await self.group.__aenter__()
        
        for name, cfg in servers.items():
            try:
                if "url" in cfg:
                    params = SseServerParameters(
                        url=cfg["url"],
                        headers=cfg.get("headers"),
                        timeout=cfg.get("timeout", 5.0),
                        sse_read_timeout=cfg.get("sse_read_timeout", 300.0)
                    )
                else:
                    params = StdioServerParameters(
                        command=cfg["command"],
                        args=cfg.get("args", []),
                        env=cfg.get("env", None),
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
        if self.group:
            await self.group.__aexit__(None, None, None)
            self.group = None
