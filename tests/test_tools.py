from pathlib import Path

import pytest

from jarvis.models.base import ToolCall
from jarvis.tools import Tool, ToolRegistry, builtin_tools


async def echo(args: dict[str, object]) -> str:
    return str(args["value"])


@pytest.mark.asyncio
async def test_registry_executes_registered_tool() -> None:
    registry = ToolRegistry()
    registry.register(Tool(name="echo", description="Echo a value.", parameters={"type": "object"}, handler=echo))
    result = await registry.execute(ToolCall(call_id="1", tool_name="echo", arguments={"value": "hi"}))
    assert result.content == "hi"
    assert result.is_error is False


@pytest.mark.asyncio
async def test_registry_returns_error_for_unknown_tool() -> None:
    registry = ToolRegistry()
    result = await registry.execute(ToolCall(call_id="1", tool_name="missing", arguments={}))
    assert result.is_error is True
    assert "Unknown tool: missing" in result.content


def test_registry_schemas_include_registered_tools() -> None:
    registry = ToolRegistry()
    registry.register(Tool(name="echo", description="Echo a value.", parameters={"type": "object"}, handler=echo))
    assert registry.schemas() == [{"name": "echo", "description": "Echo a value.", "parameters": {"type": "object"}}]


@pytest.mark.asyncio
async def test_builtin_read_file_and_search_text(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("alpha\nbeta\n", encoding="utf-8")
    registry = ToolRegistry(builtin_tools(root=tmp_path))
    read = await registry.execute(ToolCall(call_id="1", tool_name="read_file", arguments={"path": "sample.txt"}))
    search = await registry.execute(ToolCall(call_id="2", tool_name="search_text", arguments={"query": "beta"}))
    assert read.content == "alpha\nbeta\n"
    assert "sample.txt:2:beta" in search.content
