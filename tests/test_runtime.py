import asyncio

import pytest

from typing import Any
from jarvis.events import MessageEvent
from jarvis.hooks import HookResult, NoopTurnHook
from jarvis.models.base import BaseModelClient, Message, ModelResponse
from jarvis.runtime import AgentContext, AgentSession, RuntimeConfig, SessionState
from jarvis.tools import ToolRegistry


class SlowModel(BaseModelClient):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
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
