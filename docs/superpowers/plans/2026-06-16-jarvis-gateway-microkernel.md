# Jarvis Gateway Microkernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken over-designed harness with a clean gateway-first microkernel, serialized session runtime, hook checkpoints, tool registry, CLI transport, and SSE gateway.

**Architecture:** Transports submit messages to `AgentSession`, which serializes turns and calls `AgentKernel`. `AgentKernel` owns only the turn loop, hook checkpoints, tool dispatch, and event emission. Channel quirks stay outside the kernel through transport-scoped tools, native actions, metadata, and transport renderers.

**Tech Stack:** Python 3.14, dataclasses, asyncio, FastAPI, pytest, pytest-asyncio, httpx ASGI transport.

---

## File Structure

This is a clean replacement. Do not preserve stale harness, channel, memory, or subagent module contracts just because tests still reference them. Keep useful model adapter/config code; remove old architectural centers.

- Create `jarvis/events.py`: event dataclasses and `event_to_dict`.
- Create `jarvis/hooks.py`: `TurnHook`, `HookResult`, and no-op base hook.
- Create `jarvis/tools.py`: `Tool`, `ToolResult`, `ToolRegistry`, and built-in tools.
- Create `jarvis/kernel.py`: `AgentKernel.run_turn`.
- Create `jarvis/runtime.py`: `RuntimeConfig`, `SessionState`, `AgentContext`, `AgentSession`, and runtime factory helpers.
- Create `jarvis/transports/__init__.py`: transport package marker.
- Create `jarvis/transports/cli.py`: minimal CLI transport that submits to `AgentSession`.
- Modify `jarvis/config.py`: keep the current config shape only where it helps model/session loading.
- Modify `jarvis/models/base.py`: keep the existing message/model dataclasses and ensure native actions are preserved in dumps.
- Modify `main.py`: replace `AgentHarness`, channels, and memory engine usage with `AgentSession` and SSE event serialization.
- Modify `run.py`: use the new CLI transport.
- Delete `jarvis/agent.py` after `kernel.py`, `runtime.py`, and CLI replace it.
- Delete or replace stale tests that import removed modules.

---

### Task 1: Event Contracts

**Files:**
- Create: `jarvis/events.py`
- Test: `tests/test_events.py`

- [ ] **Step 1: Write the failing event tests**

Create `tests/test_events.py`:

```python
from jarvis.events import (
    ErrorEvent,
    MessageEvent,
    NativeActionEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    event_to_dict,
)
from jarvis.models.base import Message, NativeAction, ToolCall


def test_text_delta_event_serializes() -> None:
    event = TextDeltaEvent(session_id="s1", content="hello")
    assert event_to_dict(event) == {
        "event": "text_delta",
        "session_id": "s1",
        "content": "hello",
    }


def test_message_event_serializes_message() -> None:
    msg = Message(role="assistant", content="done")
    event = MessageEvent(session_id="s1", message=msg)
    assert event_to_dict(event)["message"] == msg.model_dump()


def test_tool_events_serialize() -> None:
    call = ToolCall(call_id="c1", tool_name="read_file", arguments={"path": "x"})
    call_event = ToolCallEvent(session_id="s1", tool_call=call)
    result_event = ToolResultEvent(session_id="s1", call_id="c1", tool_name="read_file", content="ok", is_error=False)
    assert event_to_dict(call_event)["tool_call"] == call.model_dump()
    assert event_to_dict(result_event)["content"] == "ok"
    assert event_to_dict(result_event)["is_error"] is False


def test_native_action_and_error_events_serialize() -> None:
    action = NativeAction(action_type="reaction", params={"emoji": "thumbs_up"})
    native = NativeActionEvent(session_id="s1", action=action)
    error = ErrorEvent(session_id="s1", message="failed")
    assert event_to_dict(native)["action"] == action.model_dump()
    assert event_to_dict(error) == {
        "event": "error",
        "session_id": "s1",
        "message": "failed",
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_events.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.events'`.

- [ ] **Step 3: Implement event dataclasses**

Create `jarvis/events.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Union

from jarvis.models.base import Message, NativeAction, ToolCall


@dataclass(slots=True)
class TextDeltaEvent:
    session_id: str
    content: str
    event: Literal["text_delta"] = "text_delta"


@dataclass(slots=True)
class MessageEvent:
    session_id: str
    message: Message
    event: Literal["message"] = "message"


@dataclass(slots=True)
class ToolCallEvent:
    session_id: str
    tool_call: ToolCall
    event: Literal["tool_call"] = "tool_call"


@dataclass(slots=True)
class ToolResultEvent:
    session_id: str
    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    event: Literal["tool_result"] = "tool_result"


@dataclass(slots=True)
class NativeActionEvent:
    session_id: str
    action: NativeAction
    event: Literal["native_action"] = "native_action"


@dataclass(slots=True)
class ErrorEvent:
    session_id: str
    message: str
    event: Literal["error"] = "error"


AgentEvent = Union[
    TextDeltaEvent,
    MessageEvent,
    ToolCallEvent,
    ToolResultEvent,
    NativeActionEvent,
    ErrorEvent,
]


def event_to_dict(event: AgentEvent) -> dict[str, object]:
    data = asdict(event)
    if isinstance(event, MessageEvent):
        data["message"] = event.message.model_dump()
    elif isinstance(event, ToolCallEvent):
        data["tool_call"] = event.tool_call.model_dump()
    elif isinstance(event, NativeActionEvent):
        data["action"] = event.action.model_dump()
    return data
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_events.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jarvis/events.py tests/test_events.py
git commit -m "feat: add agent event contracts"
```

---

### Task 2: Tool Registry And Built-ins

**Files:**
- Create: `jarvis/tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write the failing tool registry tests**

Create `tests/test_tools.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_tools.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.tools'`.

- [ ] **Step 3: Implement tool registry and safe built-ins**

Create `jarvis/tools.py`:

```python
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
            if hasattr(value, "__await__"):
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
    return [
        Tool("list_files", "List files under a workspace path.", object_params, list_files),
        Tool("read_file", "Read a UTF-8 text file under the workspace.", object_params, read_file),
        Tool("search_text", "Search for text under the workspace.", object_params, search_text),
        Tool("run_command", "Run a shell command. Policy hooks decide whether it is allowed.", object_params, run_command),
    ]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_tools.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jarvis/tools.py tests/test_tools.py
git commit -m "feat: add tool registry and built-ins"
```

---

### Task 3: Hooks And Runtime Session Serialization

**Files:**
- Create: `jarvis/hooks.py`
- Create: `jarvis/runtime.py`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: Write failing runtime and hook tests**

Create `tests/test_runtime.py`:

```python
import asyncio

import pytest

from jarvis.events import MessageEvent
from jarvis.hooks import HookResult, NoopTurnHook
from jarvis.models.base import Message, ModelResponse
from jarvis.runtime import AgentContext, AgentSession, RuntimeConfig, SessionState
from jarvis.tools import ToolRegistry


class SlowModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(self, messages: list[Message], tools: list[object]) -> ModelResponse:
        self.calls.append(messages[-1].content)
        await asyncio.sleep(0.01)
        return ModelResponse(content=f"reply:{messages[-1].content}")


class StopHook(NoopTurnHook):
    async def before_model(self, ctx: AgentContext, messages: list[Message]) -> HookResult:
        return HookResult(stop=True, reason="stopped")


class FakeKernel:
    async def run_turn(self, ctx: AgentContext, message: Message):
        ctx.session.history.append(message)
        await ctx.model.generate(ctx.session.history, [])
        yield MessageEvent(session_id=ctx.session.id, message=Message(role="assistant", content=f"reply:{message.content}"))


@pytest.mark.asyncio
async def test_agent_session_serializes_turns() -> None:
    model = SlowModel()
    ctx = AgentContext(
        config=RuntimeConfig(),
        session=SessionState(id="s1"),
        model=model,
        tools=ToolRegistry(),
        hooks=[],
    )
    session = AgentSession(ctx=ctx, kernel=FakeKernel())

    async def collect(content: str) -> list[object]:
        return [event async for event in session.submit(Message(role="user", content=content))]

    first, second = await asyncio.gather(collect("one"), collect("two"))
    assert [event.message.content for event in first if isinstance(event, MessageEvent)] == ["reply:one"]
    assert [event.message.content for event in second if isinstance(event, MessageEvent)] == ["reply:two"]
    assert [message.content for message in ctx.session.history if message.role == "user"] == ["one", "two"]


def test_hook_result_defaults() -> None:
    result = HookResult()
    assert result.messages is None
    assert result.skip_tool is False
    assert result.stop is False
    assert result.reason is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_runtime.py -v`

Expected: FAIL with `ModuleNotFoundError` for `jarvis.hooks` or `jarvis.runtime`.

- [ ] **Step 3: Implement hook and runtime contracts**

Create `jarvis/hooks.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
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
```

Create `jarvis/runtime.py`:

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

from jarvis.config import SessionConfig
from jarvis.events import AgentEvent
from jarvis.hooks import TurnHook
from jarvis.models.base import BaseModelClient, Message, get_model_class
from jarvis.tools import ToolRegistry


@dataclass(slots=True)
class RuntimeConfig:
    system_prompt: str | None = None


@dataclass(slots=True)
class SessionState:
    id: str
    history: list[Message] = field(default_factory=list)


@dataclass(slots=True)
class AgentContext:
    config: RuntimeConfig
    session: SessionState
    model: BaseModelClient
    tools: ToolRegistry
    hooks: list[TurnHook] = field(default_factory=list)


class AgentSession:
    def __init__(self, ctx: AgentContext, kernel: object) -> None:
        self.ctx = ctx
        self.kernel = kernel
        self._lock = asyncio.Lock()

    async def submit(self, message: Message) -> AsyncIterator[AgentEvent]:
        async with self._lock:
            async for event in self.kernel.run_turn(self.ctx, message):  # type: ignore[attr-defined]
                yield event


def context_from_config(config: SessionConfig, tools: ToolRegistry, hooks: list[TurnHook] | None = None) -> AgentContext:
    provider = config.model.provider.lower()
    model_cls = get_model_class(provider)
    return AgentContext(
        config=RuntimeConfig(system_prompt=config.harness.system_prompt),
        session=SessionState(id=config.session_id),
        model=model_cls.from_cfg(config),
        tools=tools,
        hooks=hooks or [],
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_runtime.py -v`

Expected: PASS.

- [ ] **Step 5: Commit hook/runtime contracts**

```bash
git add jarvis/hooks.py jarvis/runtime.py tests/test_runtime.py
git commit -m "feat: add hook contracts and serialized sessions"
```

---

### Task 4: Agent Kernel Core Loop

**Files:**
- Create: `jarvis/kernel.py`
- Test: `tests/test_kernel.py`
- Modify: `tests/test_runtime.py`

- [ ] **Step 1: Write failing kernel tests**

Create `tests/test_kernel.py`:

```python
import pytest

from jarvis.events import ErrorEvent, MessageEvent, NativeActionEvent, ToolCallEvent, ToolResultEvent
from jarvis.hooks import HookResult, NoopTurnHook
from jarvis.kernel import AgentKernel
from jarvis.models.base import Message, ModelResponse, NativeAction, ToolCall
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState
from jarvis.tools import Tool, ToolRegistry, ToolResult


class SequenceModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.calls: list[list[Message]] = []

    async def generate(self, messages: list[Message], tools: list[object]) -> ModelResponse:
        self.calls.append(list(messages))
        return self.responses.pop(0)


async def echo(args: dict[str, object]) -> str:
    return str(args["value"])


def ctx(model: SequenceModel, hooks: list[object] | None = None) -> AgentContext:
    return AgentContext(
        config=RuntimeConfig(system_prompt="system"),
        session=SessionState(id="s1"),
        model=model,
        tools=ToolRegistry([Tool("echo", "Echo value.", {"type": "object"}, echo)]),
        hooks=hooks or [],
    )


@pytest.mark.asyncio
async def test_kernel_emits_final_message() -> None:
    model = SequenceModel([ModelResponse(content="hello")])
    events = [event async for event in AgentKernel().run_turn(ctx(model), Message(role="user", content="hi"))]
    assert [event.message.content for event in events if isinstance(event, MessageEvent)] == ["hello"]


@pytest.mark.asyncio
async def test_kernel_executes_tool_and_loops_back() -> None:
    call = ToolCall(call_id="c1", tool_name="echo", arguments={"value": "ok"})
    model = SequenceModel([ModelResponse(content="", tool_calls=[call]), ModelResponse(content="done")])
    events = [event async for event in AgentKernel().run_turn(ctx(model), Message(role="user", content="hi"))]
    assert any(isinstance(event, ToolCallEvent) for event in events)
    assert [event.content for event in events if isinstance(event, ToolResultEvent)] == ["ok"]
    assert [event.message.content for event in events if isinstance(event, MessageEvent)] == ["done"]


class StopAfterModel(NoopTurnHook):
    async def after_model(self, ctx: object, response: ModelResponse) -> HookResult:
        return HookResult(stop=True, reason="repeat detected")


@pytest.mark.asyncio
async def test_kernel_stops_when_hook_requests_stop() -> None:
    model = SequenceModel([ModelResponse(content="same")])
    events = [event async for event in AgentKernel().run_turn(ctx(model, [StopAfterModel()]), Message(role="user", content="hi"))]
    assert [event.message for event in events if isinstance(event, ErrorEvent)] == ["repeat detected"]


class SkipToolHook(NoopTurnHook):
    async def before_tool(self, ctx: object, tool_call: ToolCall) -> HookResult:
        return HookResult(skip_tool=True, reason="blocked")


@pytest.mark.asyncio
async def test_kernel_skips_tool_when_hook_requests_skip() -> None:
    call = ToolCall(call_id="c1", tool_name="echo", arguments={"value": "ok"})
    model = SequenceModel([ModelResponse(content="", tool_calls=[call]), ModelResponse(content="done")])
    events = [event async for event in AgentKernel().run_turn(ctx(model, [SkipToolHook()]), Message(role="user", content="hi"))]
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert results[0].is_error is True
    assert results[0].content == "blocked"


@pytest.mark.asyncio
async def test_kernel_emits_native_action_events() -> None:
    action = NativeAction(action_type="reaction", params={"emoji": "thumbs_up"})
    model = SequenceModel([ModelResponse(content="ok")])
    message = Message(role="user", content="hi", native_actions=[action])
    events = [event async for event in AgentKernel().run_turn(ctx(model), message)]
    assert any(isinstance(event, NativeActionEvent) for event in events)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kernel.py tests/test_runtime.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.kernel'`.

- [ ] **Step 3: Implement `AgentKernel`**

Create `jarvis/kernel.py`:

```python
from __future__ import annotations

from typing import AsyncIterator

from jarvis.events import ErrorEvent, MessageEvent, NativeActionEvent, TextDeltaEvent, ToolCallEvent, ToolResultEvent
from jarvis.hooks import HookResult
from jarvis.models.base import Message, ModelResponse
from jarvis.runtime import AgentContext
from jarvis.tools import ToolResult


class AgentKernel:
    async def run_turn(self, ctx: AgentContext, user_message: Message) -> AsyncIterator[object]:
        ctx.session.history.append(user_message)
        messages = self._with_system_prompt(ctx, list(ctx.session.history))
        try:
            while True:
                hook_result = await self._run_before_model(ctx, messages)
                if hook_result.messages is not None:
                    messages = hook_result.messages
                if hook_result.stop:
                    yield ErrorEvent(session_id=ctx.session.id, message=hook_result.reason or "turn stopped")
                    return

                response = await ctx.model.generate(messages, ctx.tools.schemas())
                after_model = await self._run_after_model(ctx, response)
                if after_model.stop:
                    yield ErrorEvent(session_id=ctx.session.id, message=after_model.reason or "turn stopped")
                    return

                assistant = Message(role="assistant", content=response.content or "")
                if assistant.content:
                    yield TextDeltaEvent(session_id=ctx.session.id, content=assistant.content)
                for action in user_message.native_actions + assistant.native_actions:
                    yield NativeActionEvent(session_id=ctx.session.id, action=action)

                if not response.tool_calls:
                    messages.append(assistant)
                    ctx.session.history = self._without_system_prompt(messages)
                    for hook in ctx.hooks:
                        result = await hook.after_turn(ctx, assistant)
                        if result.stop:
                            yield ErrorEvent(session_id=ctx.session.id, message=result.reason or "turn stopped")
                            return
                    yield MessageEvent(session_id=ctx.session.id, message=assistant)
                    return

                messages.append(Message(role="assistant", content=assistant.content))
                for tool_call in response.tool_calls:
                    yield ToolCallEvent(session_id=ctx.session.id, tool_call=tool_call)
                    before_tool = await self._run_before_tool(ctx, tool_call)
                    if before_tool.stop:
                        yield ErrorEvent(session_id=ctx.session.id, message=before_tool.reason or "turn stopped")
                        return
                    if before_tool.skip_tool:
                        result = ToolResult(tool_call.call_id, tool_call.tool_name, before_tool.reason or "tool skipped", True)
                    else:
                        result = await ctx.tools.execute(tool_call)
                    yield ToolResultEvent(
                        session_id=ctx.session.id,
                        call_id=result.call_id,
                        tool_name=result.tool_name,
                        content=result.content,
                        is_error=result.is_error,
                    )
                    messages.append(Message(role="tool", content=result.content, metadata={"tool_call_id": tool_call.call_id, "tool_name": tool_call.tool_name}))
                    after_tool = await self._run_after_tool(ctx, tool_call, result)
                    if after_tool.stop:
                        yield ErrorEvent(session_id=ctx.session.id, message=after_tool.reason or "turn stopped")
                        return
        except Exception as exc:
            yield ErrorEvent(session_id=ctx.session.id, message=f"{type(exc).__name__}: {exc}")
            return

    def _with_system_prompt(self, ctx: AgentContext, messages: list[Message]) -> list[Message]:
        if ctx.config.system_prompt and not any(message.role == "system" for message in messages):
            return [Message(role="system", content=ctx.config.system_prompt), *messages]
        return messages

    def _without_system_prompt(self, messages: list[Message]) -> list[Message]:
        return [message for message in messages if message.role != "system"]

    async def _run_before_model(self, ctx: AgentContext, messages: list[Message]) -> HookResult:
        current = messages
        for hook in ctx.hooks:
            result = await hook.before_model(ctx, current)
            if result.messages is not None:
                current = result.messages
            if result.stop:
                return HookResult(messages=current, stop=True, reason=result.reason)
        return HookResult(messages=current)

    async def _run_after_model(self, ctx: AgentContext, response: ModelResponse) -> HookResult:
        for hook in ctx.hooks:
            result = await hook.after_model(ctx, response)
            if result.stop:
                return result
        return HookResult()

    async def _run_before_tool(self, ctx: AgentContext, tool_call: object) -> HookResult:
        for hook in ctx.hooks:
            result = await hook.before_tool(ctx, tool_call)  # type: ignore[arg-type]
            if result.stop or result.skip_tool:
                return result
        return HookResult()

    async def _run_after_tool(self, ctx: AgentContext, tool_call: object, result: ToolResult) -> HookResult:
        for hook in ctx.hooks:
            hook_result = await hook.after_tool(ctx, tool_call, result)  # type: ignore[arg-type]
            if hook_result.stop:
                return hook_result
        return HookResult()
```

- [ ] **Step 4: Run kernel/runtime tests**

Run: `pytest tests/test_kernel.py tests/test_runtime.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jarvis/kernel.py tests/test_kernel.py tests/test_runtime.py
git commit -m "feat: add microkernel turn loop"
```

---

### Task 5: CLI Transport

**Files:**
- Create: `jarvis/transports/__init__.py`
- Create: `jarvis/transports/cli.py`
- Modify: `run.py`
- Test: `tests/test_cli_transport.py`

- [ ] **Step 1: Write failing CLI transport test**

Create `tests/test_cli_transport.py`:

```python
import pytest

from jarvis.events import MessageEvent, NativeActionEvent
from jarvis.models.base import Message, NativeAction
from jarvis.transports.cli import render_cli_event


def test_render_cli_message_event(capsys: pytest.CaptureFixture[str]) -> None:
    render_cli_event(MessageEvent(session_id="s1", message=Message(role="assistant", content="hello")))
    assert capsys.readouterr().out == "hello\n"


def test_render_cli_native_action_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    action = NativeAction(action_type="reaction", params={"emoji": "thumbs_up"})
    render_cli_event(NativeActionEvent(session_id="s1", action=action))
    assert "native_action reaction" in capsys.readouterr().out
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cli_transport.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.transports'`.

- [ ] **Step 3: Implement CLI transport**

Create `jarvis/transports/__init__.py`:

```python
"""Transport adapters for Jarvis."""
```

Create `jarvis/transports/cli.py`:

```python
from __future__ import annotations

import asyncio

from jarvis.events import ErrorEvent, MessageEvent, NativeActionEvent, TextDeltaEvent, ToolCallEvent, ToolResultEvent
from jarvis.models.base import Message
from jarvis.runtime import AgentSession


def render_cli_event(event: object) -> None:
    if isinstance(event, MessageEvent):
        print(event.message.content)
    elif isinstance(event, TextDeltaEvent):
        print(event.content, end="", flush=True)
    elif isinstance(event, ToolCallEvent):
        print(f"\n[tool_call] {event.tool_call.tool_name} {event.tool_call.arguments}")
    elif isinstance(event, ToolResultEvent):
        prefix = "[tool_error]" if event.is_error else "[tool_result]"
        print(f"\n{prefix} {event.tool_name}: {event.content}")
    elif isinstance(event, NativeActionEvent):
        print(f"[native_action {event.action.action_type}] {event.action.params}")
    elif isinstance(event, ErrorEvent):
        print(f"[error] {event.message}")


async def run_cli(session: AgentSession) -> None:
    while True:
        line = await asyncio.to_thread(input, "> ")
        if line.lower() in {"exit", "quit"}:
            return
        async for event in session.submit(Message(role="user", content=line)):
            render_cli_event(event)
```

Modify `run.py`:

```python
import asyncio
from pathlib import Path

from jarvis.config import ModelConfig, SessionConfig
from jarvis.kernel import AgentKernel
from jarvis.runtime import AgentSession, context_from_config
from jarvis.tools import ToolRegistry, builtin_tools
from jarvis.transports.cli import run_cli


async def main() -> None:
    config = SessionConfig(
        session_id="cli",
        model=ModelConfig(provider="openai", model_name="gpt-4o"),
    )
    ctx = context_from_config(config, tools=ToolRegistry(builtin_tools(Path.cwd())))
    session = AgentSession(ctx=ctx, kernel=AgentKernel())
    await run_cli(session)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run CLI transport test**

Run: `pytest tests/test_cli_transport.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jarvis/transports/__init__.py jarvis/transports/cli.py run.py tests/test_cli_transport.py
git commit -m "feat: add cli transport"
```

---

### Task 6: Gateway SSE Runtime

**Files:**
- Modify: `main.py`
- Test: `tests/test_gateway.py`

- [ ] **Step 1: Write failing gateway SSE tests**

Create `tests/test_gateway.py`:

```python
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from jarvis.models.base import Message, ModelResponse
from main import app


class FakeModel:
    @classmethod
    def from_cfg(cls, cfg: object) -> "FakeModel":
        return cls()

    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        return ModelResponse(content="hello gateway")


@pytest.mark.asyncio
async def test_gateway_streams_kernel_events(monkeypatch: pytest.MonkeyPatch) -> None:
    from jarvis.config import ModelConfig, SessionConfig

    monkeypatch.setattr("main.get_model_class", lambda provider: FakeModel)
    monkeypatch.setattr("main.load_session_config", lambda session_id: SessionConfig(session_id=session_id, model=ModelConfig(provider="fake", model_name="fake")))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/sessions/s1/turns", json={"content": "hi", "channel": "sse"})

    assert response.status_code == 200
    assert "event: message" in response.text
    assert "hello gateway" in response.text
```

- [ ] **Step 2: Run gateway test to verify it fails**

Run: `pytest tests/test_gateway.py -v`

Expected: FAIL because `main.py` still imports deleted harness/channel/memory modules.

- [ ] **Step 3: Replace `main.py` with session-backed SSE gateway**

Modify `main.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from jarvis.config import SessionConfig, load_session_config
from jarvis.events import event_to_dict
from jarvis.kernel import AgentKernel
from jarvis.models.base import Message, TurnRequest, get_model_class
from jarvis.runtime import AgentContext, AgentSession, RuntimeConfig, SessionState
from jarvis.tools import ToolRegistry, builtin_tools

app = FastAPI(title="Jarvis Gateway")
app.state.sessions = {}


def build_session(config: SessionConfig) -> AgentSession:
    model_cls = get_model_class(config.model.provider)
    ctx = AgentContext(
        config=RuntimeConfig(system_prompt=config.harness.system_prompt),
        session=SessionState(id=config.session_id),
        model=model_cls.from_cfg(config),
        tools=ToolRegistry(builtin_tools(Path.cwd())),
        hooks=[],
    )
    return AgentSession(ctx=ctx, kernel=AgentKernel())


def get_or_create_session(session_id: str) -> AgentSession:
    sessions: dict[str, AgentSession] = app.state.sessions
    if session_id not in sessions:
        config = load_session_config(session_id)
        if config.session_id == "default":
            config.session_id = session_id
        sessions[session_id] = build_session(config)
    return sessions[session_id]


def sse_line(event: dict[str, Any]) -> str:
    name = str(event.pop("event"))
    return f"event: {name}\ndata: {json.dumps(event)}\n\n"


@app.post("/sessions/{session_id}/turns")
async def execute_session_turn(session_id: str, request: TurnRequest) -> StreamingResponse:
    if request.channel.lower() != "sse":
        raise HTTPException(status_code=400, detail="Only sse channel is implemented in the microkernel gateway")
    try:
        session = get_or_create_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {exc}") from exc

    async def stream() -> object:
        async for event in session.submit(Message(role="user", content=request.content, metadata={"channel": request.channel})):
            yield sse_line(event_to_dict(event))

    return StreamingResponse(stream(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run gateway test**

Run: `pytest tests/test_gateway.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_gateway.py
git commit -m "feat: rebuild gateway on agent sessions"
```

---

### Task 7: Remove Legacy Architecture Debris

**Files:**
- Delete: `jarvis/agent.py`
- Delete: `tests/test_harness.py`
- Delete: `tests/test_memory.py`
- Delete: `tests/test_subagent.py`
- Delete: `tests/test_integration.py`
- Modify: `tests/test_channels.py`

- [ ] **Step 1: Delete stale deleted-module tests**

Run:

```bash
rm -f tests/test_harness.py tests/test_memory.py tests/test_subagent.py tests/test_integration.py
```

Expected: stale tests that import `jarvis.harness`, `jarvis.memory`, or `jarvis.subagent` are gone. Their replacement coverage is in `tests/test_kernel.py`, `tests/test_runtime.py`, `tests/test_tools.py`, and `tests/test_gateway.py`.

- [ ] **Step 2: Replace channel tests with transport metadata contract**

Replace `tests/test_channels.py` with:

```python
from jarvis.models.base import Message


def test_message_metadata_preserves_transport_fields() -> None:
    message = Message(role="user", content="hi", metadata={"channel_id": "c1", "message_id": "m1"})
    assert message.model_dump()["metadata"] == {"channel_id": "c1", "message_id": "m1"}
```

- [ ] **Step 3: Delete broken transitional agent module**

Run:

```bash
rm -f jarvis/agent.py
```

Expected: `jarvis/agent.py` is removed. This file is replaced by `jarvis/kernel.py`, `jarvis/runtime.py`, and `jarvis/transports/cli.py`.

- [ ] **Step 4: Run the focused suite**

Run:

```bash
pytest \
  tests/test_events.py \
  tests/test_tools.py \
  tests/test_runtime.py \
  tests/test_kernel.py \
  tests/test_cli_transport.py \
  tests/test_gateway.py \
  tests/test_channels.py \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit cleanup**

```bash
git add -A jarvis/agent.py tests/test_harness.py tests/test_channels.py tests/test_memory.py tests/test_subagent.py tests/test_integration.py
git commit -m "test: replace stale harness tests with microkernel contracts"
```

---

### Task 8: Final Verification

**Files:**
- No planned file edits.

- [ ] **Step 1: Run all tests**

Run: `pytest -v`

Expected: PASS.

- [ ] **Step 2: Run type check**

Run: `pyright`

Expected: PASS. If `pyright` is not installed in the active environment, run `uv run pyright` and expect PASS.

- [ ] **Step 3: Inspect imports for deleted core modules**

Run: `rg -n "jarvis\\.(harness|memory|subagent|agent)|jarvis\\.channels" . -g "*.py"`

Expected: no matches.

- [ ] **Step 4: Commit verification fixes**

If verification required file changes, run:

```bash
git add -A
git commit -m "fix: align microkernel verification"
```

If verification produced no file changes, record that no commit is needed in the implementation notes.
