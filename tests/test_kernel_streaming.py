import pytest
from typing import AsyncGenerator, Any
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState
from jarvis.tools import ToolRegistry
from jarvis.kernel import AgentKernel
from jarvis.events import TextDeltaEvent, MessageEvent, ToolCallEvent
from jarvis.models.base import BaseModelClient, Message, ModelResponse, ToolCall

class MockStreamingModel(BaseModelClient):
    @classmethod
    def from_cfg(cls, cfg): return cls()
    
    async def generate(self, messages, tools):
        return ModelResponse(content="fallback completed response")
        
    async def generate_stream(self, messages, tools) -> AsyncGenerator[ModelResponse, None]:
        if any(m.role == "assistant" and m.metadata and "tool_calls" in m.metadata for m in messages):
            return
        yield ModelResponse(content="hello ")
        yield ModelResponse(content="streaming world")
        yield ModelResponse(content=None, tool_calls=[ToolCall(call_id="c1", tool_name="ls", arguments={})])

class MockNoStreamingModel(BaseModelClient):
    @classmethod
    def from_cfg(cls, cfg): return cls()
    
    async def generate(self, messages, tools):
        return ModelResponse(content="one-shot completion")
        
    async def generate_stream(self, messages, tools) -> AsyncGenerator[ModelResponse, None]:
        raise NotImplementedError("Streaming is not supported")

@pytest.mark.asyncio
async def test_kernel_run_turn_streams_events() -> None:
    ctx = AgentContext(
        config=RuntimeConfig(stream=True),
        session=SessionState(id="s1"),
        model=MockStreamingModel(),
        tools=ToolRegistry()
    )
    kernel = AgentKernel()
    
    events = []
    async for event in kernel.run_turn(ctx, Message(role="user", content="hi")):
        events.append(event)
        
    # Verify text deltas were yielded
    deltas = [ev.content for ev in events if isinstance(ev, TextDeltaEvent)]
    assert deltas == ["hello ", "streaming world"]
    
    # Verify tool call was accumulated and emitted
    tcalls = [ev.tool_call.tool_name for ev in events if isinstance(ev, ToolCallEvent)]
    assert tcalls == ["ls"]

@pytest.mark.asyncio
async def test_kernel_generate_stream_fallback() -> None:
    ctx = AgentContext(
        config=RuntimeConfig(stream=True),
        session=SessionState(id="s1"),
        model=MockNoStreamingModel(),
        tools=ToolRegistry()
    )
    kernel = AgentKernel()
    
    events = []
    async for event in kernel.run_turn(ctx, Message(role="user", content="hi")):
        events.append(event)
        
    # Fallback should call generate()
    msg_events = [ev.message.content for ev in events if isinstance(ev, MessageEvent)]
    assert msg_events == ["one-shot completion"]
