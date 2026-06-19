import pytest
import json
import sys
from pathlib import Path
from jarvis.runtime import AgentSession, AgentContext, RuntimeConfig, SessionState
from jarvis.tools import ToolRegistry, ToolCall, Tool
from jarvis.events import MessageEvent
from jarvis.models.base import Message

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
        def __init__(self, *args, **kwargs):
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

@pytest.mark.asyncio
async def test_agent_session_lazy_init_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_tools_returned = [
        Tool(
            name="lazy_mcp_tool",
            description="mock",
            parameters={},
            handler=lambda x: "hello lazy"
        )
    ]
    
    initialized_called = 0
    close_called = 0
    
    class MockMcpClientManager:
        def __init__(self, config_path="config/mcp_settings.json", proxy_env=None):
            pass
        async def initialize(self):
            nonlocal initialized_called
            initialized_called += 1
            return mock_tools_returned
        async def close(self):
            nonlocal close_called
            close_called += 1
            
    monkeypatch.setattr("jarvis.mcp.McpClientManager", MockMcpClientManager)
    
    class MockKernel:
        async def run_turn(self, ctx, message):
            yield MessageEvent(
                session_id=ctx.session.id,
                message=Message(role="assistant", content="done")
            )
            
    ctx = AgentContext(
        config=RuntimeConfig(),
        session=SessionState(id="session_test"),
        model=None, # type: ignore
        tools=ToolRegistry(),
    )
    
    session = AgentSession(ctx=ctx, kernel=MockKernel())
    
    # MCP not loaded at startup
    assert not session._mcp_initialized
    assert ctx.mcp_manager is None
    assert "lazy_mcp_tool" not in ctx.tools._tools
    
    # load_mcp tool is registered
    assert "load_mcp" in ctx.tools._tools
    
    # Call load_mcp to trigger lazy init
    result = await ctx.tools._tools["load_mcp"].handler({})
    assert "lazy_mcp_tool" in result
    assert initialized_called == 1
    
    # Now MCP is initialized
    assert session._mcp_initialized
    assert ctx.mcp_manager is not None
    assert "lazy_mcp_tool" in ctx.tools._tools
    
    await session.close()
    
    assert not session._mcp_initialized
    assert ctx.mcp_manager is None
    assert "lazy_mcp_tool" not in ctx.tools._tools
    assert close_called == 1

@pytest.mark.asyncio
async def test_mcp_config_edge_cases(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from jarvis.mcp import McpClientManager
    
    missing_file = tmp_path / "nonexistent.json"
    manager = McpClientManager(config_path=str(missing_file))
    tools = await manager.initialize()
    assert tools == []
    
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("this is not { valid json }")
    manager2 = McpClientManager(config_path=str(invalid_file))
    tools2 = await manager2.initialize()
    assert tools2 == []
    
    captured = capsys.readouterr()
    assert "Error parsing MCP settings JSON" in captured.err
