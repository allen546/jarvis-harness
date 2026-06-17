import pytest
import json
from pathlib import Path
from jarvis.runtime import AgentSession
from jarvis.tools import ToolRegistry, ToolCall

MOCK_MCP_SETTINGS = {
    "mcpServers": {
        "mock_server": {
            "command": "python",
            "args": ["-c", "import sys; print('mock tool init')"]
        }
    }
}

@pytest.mark.asyncio
async def test_mcp_settings_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jarvis.mcp import McpClientManager
    config_file = tmp_path / "mcp_settings.json"
    config_file.write_text(json.dumps(MOCK_MCP_SETTINGS))
    
    class MockGroup:
        def __init__(self):
            from types import SimpleNamespace
            self.tools = {
                "mcp_hello": SimpleNamespace(
                    name="mcp_hello",
                    description="test tool",
                    inputSchema={"type": "object", "properties": {}}
                )
            }
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def connect_to_server(self, params): pass
        async def call_tool(self, name, args):
            from types import SimpleNamespace
            if name == "mcp_error_tool":
                return SimpleNamespace(content=[SimpleNamespace(text="mcp error msg")], isError=True)
            return SimpleNamespace(content=[SimpleNamespace(text="mcp response")], isError=False)
    
    monkeypatch.setattr("jarvis.mcp.ClientSessionGroup", MockGroup)
    
    manager = McpClientManager(config_path=str(config_file))
    tools = await manager.initialize()
    assert len(tools) == 1
    assert tools[0].name == "mcp_hello"
    
    res = await tools[0].handler({})
    assert res == "mcp response"
    
    manager.group.tools["mcp_error_tool"] = manager.group.tools["mcp_hello"]
    with pytest.raises(ValueError, match="mcp error msg"):
        await manager.execute_tool("mcp_error_tool", {})
