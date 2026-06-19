from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import warnings
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
    """Append-only raw history writer. Sessions start fresh — no replay from disk."""

    __slots__ = ("storage_dir", "_turn_start_index")

    def __init__(self, storage_dir: str = "storage") -> None:
        self.storage_dir = storage_dir
        self._turn_start_index: dict[int, int] = {}

    def _get_file_path(self, session_id: str) -> Path:
        return Path(self.storage_dir) / "sessions" / session_id / "history.jsonl"

    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
        # Sessions start fresh — no history loaded from disk into context.
        session = getattr(ctx, "session")
        session_key = id(session)
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


class SessionRefreshHook(NoopTurnHook):
    """Compress undistilled messages in-place when context grows too long.
    
    Extracts facts/procedures + summary in one model call.
    Marks compressed messages as distilled. Stores history_summary in semantic memory.
    """

    __slots__ = ("storage_dir", "threshold", "keep_messages")

    def __init__(self, storage_dir: str = "storage", threshold: int = 24, keep_messages: int = 12) -> None:
        self.storage_dir = storage_dir
        self.threshold = threshold
        self.keep_messages = keep_messages

    def _get_refresh_file_path(self, session_id: str) -> Path:
        return Path(self.storage_dir) / "sessions" / session_id / "refreshes.jsonl"

    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
        session = getattr(ctx, "session")
        model = getattr(ctx, "model", None)
        if model is None or len(session.history) < self.threshold:
            return HookResult()

        # Find undistilled messages in the compressible zone (everything except recent N)
        compressible = session.history[:-self.keep_messages] if self.keep_messages > 0 else session.history
        undistilled = [m for m in compressible if not m.metadata.get("distilled")]

        if not undistilled:
            return HookResult()

        # One model call: compress + distill
        prompt_msgs = [
            Message(role="system", content=(
                "Compress and distill this Jarvis conversation segment. "
                "Return strict JSON only, no markdown.\n"
                "Schema: {\"summary\": str, \"facts\": [{\"text\": str, \"tags\": [str], \"confidence\": float}], "
                "\"procedures\": [{\"name\": str, \"trigger\": str, \"summary\": str, \"steps\": [str], "
                "\"tools\": [str], \"confidence\": float}]}\n"
                "Summary should preserve user preferences, durable facts, unresolved tasks, decisions, and procedures. "
                "Facts are stable truths. Procedures are reusable task patterns. Do not invent details."
            )),
            *undistilled,
        ]

        try:
            response = await model.generate(prompt_msgs, [])
            response_text = response.content or ""
        except Exception as exc:
            warnings.warn(f"SessionRefreshHook model call failed: {exc}")
            return HookResult()

        # Parse response
        summary_content = response_text
        facts = []
        procedures = []
        try:
            parsed = json.loads(response_text)
            summary_content = parsed.get("summary", response_text)
            facts = parsed.get("facts", [])
            procedures = parsed.get("procedures", [])
        except (json.JSONDecodeError, AttributeError):
            pass  # Use raw text as summary, no fact/procedure extraction

        # Mark all processed messages as distilled
        for m in undistilled:
            m.metadata["distilled"] = True

        # Replace undistilled messages with summary in session.history
        recent = session.history[-self.keep_messages:] if self.keep_messages > 0 else []
        summary_msg = Message(
            role="system",
            content=f"[Session refresh: {summary_content}]",
            metadata={"memory_kind": "session_refresh", "distilled": True},
        )
        session.history = [summary_msg] + recent

        # Store facts/procedures in semantic memory
        try:
            from jarvis.memory_store import SemanticMemoryStore
            memory_config = getattr(getattr(ctx, "config", None), "memory", None)
            scope = getattr(memory_config, "scope", "global") if memory_config else "global"
            storage_dir = getattr(memory_config, "storage_dir", self.storage_dir) if memory_config else self.storage_dir
            store = SemanticMemoryStore(storage_dir=storage_dir)

            for fact in facts:
                text = fact.get("text", "")
                if text:
                    await store.add_memory(
                        session.id, text, fact.get("tags", ["truths"]),
                        kind="fact", scope=scope,
                        metadata={"source": "session_refresh"},
                        confidence=fact.get("confidence", 1.0),
                    )
            for proc in procedures:
                name = proc.get("name", "")
                steps = proc.get("steps", [])
                if name and steps:
                    await store.add_memory(
                        session.id, proc.get("summary", name),
                        ["procedure", *proc.get("tools", [])],
                        kind="procedure", scope=scope,
                        metadata={
                            "name": name,
                            "trigger": proc.get("trigger", ""),
                            "steps": steps,
                            "tools": proc.get("tools", []),
                            "source": "session_refresh",
                        },
                        confidence=proc.get("confidence", 1.0),
                    )

            # Store history summary
            await store.add_memory(
                session.id, summary_content,
                ["history", "session_refresh"],
                kind="history_summary", scope=scope,
                metadata={"refreshed_message_count": len(undistilled)},
            )
        except Exception as exc:
            warnings.warn(f"Error storing session refresh memory: {exc}")

        # Append refresh record to refreshes.jsonl
        try:
            refresh_path = self._get_refresh_file_path(session.id)
            refresh_path.parent.mkdir(parents=True, exist_ok=True)
            import datetime
            record = {
                "session_id": session.id,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "summary": summary_content,
                "refreshed_message_count": len(undistilled),
                "kept_message_count": len(recent),
            }
            with open(refresh_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            warnings.warn(f"Error writing refresh record: {exc}")

        # Rebuild messages preserving active system prompt at index 0
        new_messages = list(session.history)
        if messages and messages[0].role == "system":
            # Keep the original system prompt, skip any refresh summaries now in history
            sys_msg = messages[0]
            non_sys = [m for m in new_messages if m is not summary_msg]
            new_messages = [sys_msg] + non_sys

        return HookResult(messages=new_messages)


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
        self._consecutive: dict[str, int] = {}
        self._last_keys: dict[str, str] = {}

    async def after_tool(self, ctx: object, tool_call: ToolCall, result: ToolResult) -> HookResult:
        session = getattr(ctx, "session")
        key = f"{tool_call.tool_name}:{json.dumps(tool_call.arguments, sort_keys=True)}"
        prev_key = self._last_keys.get(session.id)
        if key == prev_key:
            count = self._consecutive.get(session.id, 0) + 1
            self._consecutive[session.id] = count
            if count >= self._max_repeats:
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
        self._recent: dict[str, list[str]] = {}

    async def after_turn(self, ctx: object, message: Message) -> HookResult:
        session = getattr(ctx, "session")
        content = message.content or ""
        if not content:
            return HookResult()
        recent = self._recent.setdefault(session.id, [])
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
    if name == "MemoryDistillationHook":
        from jarvis.memory_store import MemoryDistillationHook
        return MemoryDistillationHook
    if name == "MemoryInjectionHook":
        from jarvis.memory_store import MemoryInjectionHook
        return MemoryInjectionHook
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
