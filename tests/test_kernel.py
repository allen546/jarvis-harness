import pytest
from typing import Any

from jarvis.events import ErrorEvent, MessageEvent, NativeActionEvent, ToolCallEvent, ToolResultEvent
from jarvis.hooks import HookResult, NoopTurnHook, TurnHook
from jarvis.kernel import AgentKernel
from jarvis.models.base import BaseModelClient, Message, ModelResponse, NativeAction, ToolCall
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState
from jarvis.tools import Tool, ToolRegistry, ToolResult


class SequenceModel(BaseModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.calls: list[list[Message]] = []

    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        self.calls.append(list(messages))
        return self.responses.pop(0)


async def echo(args: dict[str, object]) -> str:
    return str(args["value"])


def ctx(model: BaseModelClient, hooks: list[TurnHook] | None = None) -> AgentContext:
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
