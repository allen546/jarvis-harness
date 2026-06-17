import pytest
from jarvis.hooks import ContextCompressionHook
from jarvis.models.base import Message, ModelResponse, BaseModelClient
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState
from jarvis.tools import ToolRegistry
from jarvis.kernel import AgentKernel

class MockModel(BaseModelClient):
    @classmethod
    def from_cfg(cls, cfg): return cls()
    async def generate(self, messages, tools):
        return ModelResponse(content="SUMMARY_OF_CHAT")

@pytest.mark.asyncio
async def test_compression_hook():
    hook = ContextCompressionHook(threshold=5, compress_count=3)
    state = SessionState(id="sess_comp")
    # 6 messages (exceeds threshold 5)
    state.history = [
        Message(role="user", content="m1"),
        Message(role="assistant", content="m2"),
        Message(role="user", content="m3"),
        Message(role="assistant", content="m4"),
        Message(role="user", content="m5"),
        Message(role="assistant", content="m6"),
    ]
    ctx = AgentContext(config=RuntimeConfig(), session=state, model=MockModel(), tools=ToolRegistry(), hooks=[])

    # Trigger compression hook
    # Simulate list(messages) that would go to before_model hook
    messages = list(state.history)
    res = await hook.before_model(ctx, messages)
    
    # Check that oldest 3 messages are replaced by a system summary
    assert len(state.history) == 4  # 1 summary + remaining 3 messages
    assert state.history[0].role == "system"
    assert "SUMMARY_OF_CHAT" in state.history[0].content
    assert state.history[1].content == "m4"
    assert state.history[2].content == "m5"
    assert state.history[3].content == "m6"

@pytest.mark.asyncio
async def test_compression_hook_below_threshold():
    hook = ContextCompressionHook(threshold=5, compress_count=3)
    state = SessionState(id="sess_no_comp")
    state.history = [
        Message(role="user", content="m1"),
        Message(role="assistant", content="m2"),
        Message(role="user", content="m3"),
    ]
    ctx = AgentContext(config=RuntimeConfig(), session=state, model=MockModel(), tools=ToolRegistry(), hooks=[])
    messages = list(state.history)
    res = await hook.before_model(ctx, messages)
    assert len(state.history) == 3
    assert state.history[0].content == "m1"
    assert state.history[1].content == "m2"
    assert state.history[2].content == "m3"
    assert res.messages is None

@pytest.mark.asyncio
async def test_compression_hook_preserves_system_prompt():
    hook = ContextCompressionHook(threshold=5, compress_count=3)
    state = SessionState(id="sess_sys")
    state.history = [
        Message(role="user", content="m1"),
        Message(role="assistant", content="m2"),
        Message(role="user", content="m3"),
        Message(role="assistant", content="m4"),
        Message(role="user", content="m5"),
        Message(role="assistant", content="m6"),
    ]
    ctx = AgentContext(config=RuntimeConfig(system_prompt="SYS_PROMPT"), session=state, model=MockModel(), tools=ToolRegistry(), hooks=[])
    
    # Simulate list(messages) that would go to before_model hook (prepopulated with system prompt at the top)
    messages = [Message(role="system", content="SYS_PROMPT")] + list(state.history)
    res = await hook.before_model(ctx, messages)
    
    # The return value HookResult should have the messages where active system prompt is preserved at the top
    assert res.messages is not None
    assert len(res.messages) == 5  # 1 system prompt + 1 summary + 3 remaining
    assert res.messages[0].role == "system"
    assert res.messages[0].content == "SYS_PROMPT"
    assert res.messages[1].role == "system"
    assert "SUMMARY_OF_CHAT" in res.messages[1].content
    assert res.messages[2].content == "m4"

@pytest.mark.asyncio
async def test_compression_hook_integration_with_kernel():
    hook = ContextCompressionHook(threshold=5, compress_count=3)
    state = SessionState(id="sess_integration")
    state.history = [
        Message(role="user", content="m1"),
        Message(role="assistant", content="m2"),
        Message(role="user", content="m3"),
        Message(role="assistant", content="m4"),
    ]
    
    model = MockModel()
    ctx = AgentContext(
        config=RuntimeConfig(system_prompt="SYS_PROMPT"),
        session=state,
        model=model,
        tools=ToolRegistry(),
        hooks=[hook],
    )
    
    kernel = AgentKernel()
    user_msg = Message(role="user", content="m5")
    
    events = [event async for event in kernel.run_turn(ctx, user_msg)]
    
    # Verify that the summary message is preserved in history at the end of the turn
    assert len(state.history) == 4
    assert state.history[0].role == "system"
    assert "SUMMARY_OF_CHAT" in state.history[0].content
    assert state.history[1].content == "m4"
    assert state.history[2].content == "m5"
    assert state.history[3].role == "assistant"
    assert state.history[3].content == "SUMMARY_OF_CHAT"
