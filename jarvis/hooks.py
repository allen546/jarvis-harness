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
    __slots__ = ()

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
    __slots__ = ("storage_dir", "_loaded_sessions", "_turn_start_index")

    def __init__(self, storage_dir: str = "storage") -> None:
        self.storage_dir = storage_dir
        self._loaded_sessions: set[int] = set()
        self._turn_start_index: dict[int, int] = {}

    def _get_file_path(self, session_id: str) -> Path:
        return Path(self.storage_dir) / "sessions" / session_id / "history.jsonl"

    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
        session = getattr(ctx, "session")
        session_key = id(session)
        
        if session_key not in self._loaded_sessions:
            file_path = self._get_file_path(session.id)
            loaded_history = []
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            loaded_history.append(Message.model_validate(json.loads(line)))
            
            if loaded_history:
                session.history = loaded_history + session.history
                
                # Prepend the loaded history to the messages list being sent to the model
                if messages and messages[0].role == "system":
                    messages = [messages[0]] + loaded_history + messages[1:]
                else:
                    messages = loaded_history + messages
            
            self._loaded_sessions.add(session_key)

        # Track the start of the current turn in session.history
        self._turn_start_index[session_key] = len(session.history) - 1
        
        return HookResult(messages=messages)

    async def after_turn(self, ctx: object, message: Message) -> HookResult:
        session = getattr(ctx, "session")
        session_key = id(session)
        file_path = self._get_file_path(session.id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_idx = self._turn_start_index.get(session_key, len(session.history) - 1)
        start_idx = max(0, min(start_idx, len(session.history) - 1))
        
        new_messages = session.history[start_idx:]
        
        with open(file_path, "a", encoding="utf-8") as f:
            for m in new_messages:
                f.write(json.dumps(m.model_dump()) + "\n")
                
        self._turn_start_index.pop(session_key, None)
        return HookResult()


class ContextCompressionHook(NoopTurnHook):
    __slots__ = ("threshold", "compress_count")

    def __init__(self, threshold: int = 20, compress_count: int = 10) -> None:
        self.threshold = threshold
        self.compress_count = compress_count

    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
        session = getattr(ctx, "session")
        model = getattr(ctx, "model")
        if len(session.history) >= self.threshold:
            to_compress = session.history[:self.compress_count]
            remaining = session.history[self.compress_count:]
            
            prompt_msgs = [
                Message(role="system", content="Summarize the following chat history concisely:"),
                *to_compress
            ]
            response = await model.generate(prompt_msgs, [])
            summary_content = response.content or "No summary"
            summary_msg = Message(role="system", content=f"[Summary of previous conversation: {summary_content}]")
            
            session.history = [summary_msg] + remaining
            
            # Rebuild the messages list that we return to the kernel loop
            # Note: the kernel uses messages (which is built using history and system prompt).
            # If the user's turn has a system prompt, kernel._with_system_prompt prepend it.
            # We can update the messages to contain the new history (including summary).
            new_messages = list(session.history)
            if any(m.role == "system" for m in messages):
                # If there's an active system prompt, preserve it at the very top.
                sys_msgs = [m for m in messages if m.role == "system" and not m.content.startswith("[Summary of")]
                new_messages = sys_msgs + new_messages
            
            return HookResult(messages=new_messages)
        return HookResult()


def __getattr__(name: str):
    if name == "SemanticMemoryHook":
        from jarvis.memory_store import SemanticMemoryHook
        return SemanticMemoryHook
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")






