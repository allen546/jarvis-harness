from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

from jarvis.models.base import Message, ModelResponse, ToolCall
from jarvis.tools import ToolResult


@dataclass(slots=True)
class HookResult:
    messages: list[Message] | None = None
    skip_tool: bool = False
    stop: bool = False
    reason: str | None = None


class TurnHook(Protocol):
    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult: ...
    async def after_model(self, ctx: object, response: ModelResponse) -> HookResult: ...
    async def before_tool(self, ctx: object, tool_call: ToolCall) -> HookResult: ...
    async def after_tool(self, ctx: object, tool_call: ToolCall, result: ToolResult) -> HookResult: ...
    async def after_turn(self, ctx: object, message: Message) -> HookResult: ...


class NoopTurnHook:
    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
        return HookResult()

    async def after_model(self, ctx: object, response: ModelResponse) -> HookResult:
        return HookResult()

    async def before_tool(self, ctx: object, tool_call: ToolCall) -> HookResult:
        return HookResult()

    async def after_tool(self, ctx: object, tool_call: ToolCall, result: ToolResult) -> HookResult:
        return HookResult()

    async def after_turn(self, ctx: object, message: Message) -> HookResult:
        return HookResult()


class JSONLHistoryHook(NoopTurnHook):
    def __init__(self, storage_dir: str = "storage") -> None:
        self.storage_dir = storage_dir

    def _get_file_path(self, session_id: str) -> Path:
        return Path(self.storage_dir) / "sessions" / session_id / "history.jsonl"

    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
        session = getattr(ctx, "session")
        if not session.history:
            file_path = self._get_file_path(session.id)
            if file_path.exists():
                history = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            history.append(Message.model_validate(json.loads(line)))
                session.history = history
        return HookResult()

    async def after_turn(self, ctx: object, message: Message) -> HookResult:
        session = getattr(ctx, "session")
        file_path = self._get_file_path(session.id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            for m in session.history[-2:]:  # User message and final response
                f.write(json.dumps(m.model_dump()) + "\n")
        return HookResult()

