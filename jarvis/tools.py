from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from jarvis.models.base import ToolCall

ToolHandler = Callable[[dict[str, Any]], Awaitable[str] | str]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass(slots=True)
class ToolResult:
    call_id: str
    tool_name: str
    content: str
    is_error: bool = False


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.tool_name)
        if tool is None:
            return ToolResult(call_id=call.call_id, tool_name=call.tool_name, content=f"Unknown tool: {call.tool_name}", is_error=True)
        try:
            value = tool.handler(call.arguments)
            if not isinstance(value, str):
                value = await value
            return ToolResult(call_id=call.call_id, tool_name=call.tool_name, content=str(value), is_error=False)
        except Exception as exc:
            return ToolResult(call_id=call.call_id, tool_name=call.tool_name, content=f"{type(exc).__name__}: {exc}", is_error=True)


def _resolve(root: Path, raw_path: str) -> Path:
    candidate = (root / raw_path).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"path escapes root: {raw_path}")
    return candidate


def builtin_tools(root: Path | str = ".") -> list[Tool]:
    base = Path(root)

    def list_files(args: dict[str, Any]) -> str:
        raw_path = str(args.get("path", "."))
        target = _resolve(base, raw_path)
        if target.is_file():
            return target.name
        return "\n".join(sorted(str(path.relative_to(base.resolve())) for path in target.rglob("*") if path.is_file()))

    def read_file(args: dict[str, Any]) -> str:
        target = _resolve(base, str(args["path"]))
        return target.read_text(encoding="utf-8")

    def search_text(args: dict[str, Any]) -> str:
        query = str(args["query"])
        raw_path = str(args.get("path", "."))
        target = _resolve(base, raw_path)
        files = [target] if target.is_file() else [path for path in target.rglob("*") if path.is_file()]
        matches: list[str] = []
        for path in files:
            try:
                for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                    if query in line:
                        matches.append(f"{path.relative_to(base.resolve())}:{line_no}:{line}")
            except UnicodeDecodeError:
                continue
        return "\n".join(matches)

    async def run_command(args: dict[str, Any]) -> str:
        command = str(args["command"])
        proc = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output, _ = await proc.communicate()
        return output.decode("utf-8", errors="replace")

    object_params = {"type": "object", "properties": {}, "additionalProperties": True}
    from jarvis.memory_store import search_semantic_memory_tool, purge_semantic_memory_tool
    search_params = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "tag": {"type": "string", "description": "Optional tag filter (e.g. 'truths', 'history')."}
        },
        "required": ["query"]
    }
    purge_params = {
        "type": "object",
        "properties": {
            "tag": {"type": "string", "description": "Tag to purge memories by (e.g. 'truths', 'history')."},
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of memory item IDs to purge."
            }
        }
    }

    async def spawn_subagent_handler(args: dict[str, Any]) -> str:
        from jarvis.runtime import current_context
        from jarvis.subagent import spawn_subagent_tool
        import json
        ctx = current_context.get()
        if ctx is None:
            raise RuntimeError("No active agent context")
        res = await spawn_subagent_tool(ctx, args)
        return json.dumps(res)

    async def send_subagent_message_handler(args: dict[str, Any]) -> str:
        from jarvis.runtime import current_context
        from jarvis.subagent import send_subagent_message_tool
        import json
        ctx = current_context.get()
        if ctx is None:
            raise RuntimeError("No active agent context")
        res = await send_subagent_message_tool(ctx, args)
        return json.dumps(res)

    async def close_subagent_handler(args: dict[str, Any]) -> str:
        from jarvis.runtime import current_context
        from jarvis.subagent import close_subagent_tool
        import json
        ctx = current_context.get()
        if ctx is None:
            raise RuntimeError("No active agent context")
        res = await close_subagent_tool(ctx, args)
        return json.dumps(res)

    spawn_subagent_params = {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of prompts/messages for the subagent to process."
            },
            "prompt": {
                "type": "string",
                "description": "Alternative single prompt for the subagent to process."
            },
            "task_name": {
                "type": "string",
                "description": "Name or description of the subtask."
            },
            "system_override": {
                "type": "string",
                "description": "Optional custom system prompt override for the subagent."
            }
        },
        "required": ["task_name"]
    }

    send_subagent_message_params = {
        "type": "object",
        "properties": {
            "sub_session_id": {
                "type": "string",
                "description": "The session ID of the target subagent."
            },
            "message": {
                "type": "string",
                "description": "The follow-up message to send."
            }
        },
        "required": ["sub_session_id", "message"]
    }

    close_subagent_params = {
        "type": "object",
        "properties": {
            "sub_session_id": {
                "type": "string",
                "description": "The session ID of the subagent to close."
            }
        },
        "required": ["sub_session_id"]
    }

    return [
        Tool("list_files", "List files under a workspace path.", object_params, list_files),
        Tool("read_file", "Read a UTF-8 text file under the workspace.", object_params, read_file),
        Tool("search_text", "Search for text under the workspace.", object_params, search_text),
        Tool("run_command", "Run a shell command. Policy hooks decide whether it is allowed.", object_params, run_command),
        Tool("search_semantic_memory", "Search semantic memory for previously stored facts and history.", search_params, search_semantic_memory_tool),
        Tool("purge_semantic_memory", "Purge specific items or tags from semantic memory.", purge_params, purge_semantic_memory_tool),
        Tool("spawn_subagent", "Spawn a collaborative subagent to handle a specific task.", spawn_subagent_params, spawn_subagent_handler),
        Tool("send_subagent_message", "Send a message to an active subagent.", send_subagent_message_params, send_subagent_message_handler),
        Tool("close_subagent", "Close an active subagent and clean up resources.", close_subagent_params, close_subagent_handler),
    ]
