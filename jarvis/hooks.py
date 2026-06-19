from __future__ import annotations

from dataclasses import dataclass
import inspect
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
            new_messages = list(session.history)
            if any(m.role == "system" for m in messages):
                # Preserve the active system prompt at the top, but skip summaries
                # that are now in session.history to avoid duplication
                sys_msgs = [m for m in messages if m.role == "system" and not m.content.startswith("[Summary of")]
                new_messages = sys_msgs + new_messages
            
            return HookResult(messages=new_messages)
        return HookResult()


class BudgetGuardHook(NoopTurnHook):
    __slots__ = ("_counts", "_last_history_len")
    
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._last_history_len: dict[str, int] = {}
        
    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
        session = getattr(ctx, "session")
        session_id = session.id
        hist_len = len(session.history)
        if self._last_history_len.get(session_id) != hist_len:
            self._counts[session_id] = 0
            self._last_history_len[session_id] = hist_len
        return HookResult()
        
    async def before_tool(self, ctx: object, tool_call: ToolCall) -> HookResult:
        session = getattr(ctx, "session")
        config = getattr(ctx, "config")
        count = self._counts.get(session.id, 0)
        limit = getattr(config, "max_consecutive_tools", 5)
        if count >= limit:
            return HookResult(stop=True, reason=f"Tool execution budget limit exceeded: max {limit} consecutive calls")
        return HookResult()
        
    async def after_tool(self, ctx: object, tool_call: ToolCall, result: ToolResult) -> HookResult:
        session = getattr(ctx, "session")
        self._counts[session.id] = self._counts.get(session.id, 0) + 1
        return HookResult()

    async def after_turn(self, ctx: object, message: Message) -> HookResult:
        session = getattr(ctx, "session")
        self._counts.pop(session.id, None)
        self._last_history_len.pop(session.id, None)
        return HookResult()


class ToolApprovalHook(NoopTurnHook):
    __slots__ = ()
    
    async def before_tool(self, ctx: object, tool_call: ToolCall) -> HookResult:
        config = getattr(ctx, "config")
        require_approval = getattr(config, "require_tool_approval", False)
        if not require_approval:
            return HookResult()
            
        handler = getattr(ctx, "approval_handler", None)
        if handler is None:
            return HookResult(skip_tool=True, reason="Tool approval required but no handler registered")
            
        approved = handler(tool_call)
        if inspect.isawaitable(approved):
            approved = await approved
            
        if not approved:
            return HookResult(skip_tool=True, reason="Tool call rejected by user")
        return HookResult()


class RepeatedLoopHook(NoopTurnHook):
    """Stop the turn when the agent calls the same tool with identical arguments repeatedly."""
    __slots__ = ("_consecutive", "_last_keys", "_max_repeats")

    def __init__(self, max_repeats: int = 3) -> None:
        self._max_repeats = max_repeats
        self._consecutive: dict[str, int] = {}  # session_id -> count of same call
        self._last_keys: dict[str, str] = {}    # session_id -> last tool call key

    async def after_tool(self, ctx: object, tool_call: ToolCall, result: ToolResult) -> HookResult:
        session = getattr(ctx, "session")
        key = f"{tool_call.tool_name}:{json.dumps(tool_call.arguments, sort_keys=True)}"
        prev_key = self._last_keys.get(session.id)
        if key == prev_key:
            count = self._consecutive.get(session.id, 0) + 1
            self._consecutive[session.id] = count
            if count >= self._max_repeats:
                # Clean up so next turn starts fresh
                self._consecutive.pop(session.id, None)
                self._last_keys.pop(session.id, None)
                return HookResult(
                    stop=True,
                    reason=f"Repeated tool loop detected: {tool_call.tool_name} called {count + 1} times with identical arguments",
                )
        else:
            self._consecutive[session.id] = 1
        self._last_keys[session.id] = key
        return HookResult()

    async def after_turn(self, ctx: object, message: Message) -> HookResult:
        session = getattr(ctx, "session")
        self._consecutive.pop(session.id, None)
        self._last_keys.pop(session.id, None)
        return HookResult()


def _char_ngrams(text: str, n: int = 3) -> dict[str, int]:
    """Count character n-grams in text."""
    counts: dict[str, int] = {}
    lower = text.lower()
    for i in range(len(lower) - n + 1):
        gram = lower[i : i + n]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _jaccard_ngrams(a: str, b: str, n: int = 3) -> float:
    """Jaccard similarity of character n-gram sets. Returns 1.0 for identical strings."""
    if a == b:
        return 1.0
    grams_a = set(_char_ngrams(a, n).keys())
    grams_b = set(_char_ngrams(b, n).keys())
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / len(grams_a | grams_b)


class RepeatedContentHook(NoopTurnHook):
    """Stop the turn when the assistant generates near-identical content repeatedly."""
    __slots__ = ("_threshold", "_window", "_recent")

    def __init__(self, threshold: float = 0.8, window: int = 3) -> None:
        self._threshold = threshold
        self._window = window
        self._recent: dict[str, list[str]] = {}  # session_id -> recent assistant contents

    async def after_turn(self, ctx: object, message: Message) -> HookResult:
        session = getattr(ctx, "session")
        content = message.content or ""
        if not content:
            return HookResult()
        recent = self._recent.setdefault(session.id, [])
        # Evict oldest before checking, so only the sliding window is compared
        if len(recent) >= self._window:
            recent.pop(0)
        for prev in recent:
            if _jaccard_ngrams(content, prev) >= self._threshold:
                return HookResult(
                    stop=True,
                    reason=f"Repeated content detected (similarity >= {self._threshold})",
                )
        recent.append(content)
        return HookResult()


def __getattr__(name: str):
    if name == "SemanticMemoryHook":
        from jarvis.memory_store import SemanticMemoryHook
        return SemanticMemoryHook
    if name == "SkillInstructionsHook":
        from jarvis.skills import SkillInstructionsHook
        return SkillInstructionsHook
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
