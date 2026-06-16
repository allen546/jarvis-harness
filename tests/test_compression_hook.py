import pytest
from jarvis.hooks import ContextCompressionHook
from jarvis.models.base import Message, ModelResponse, BaseModelClient
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState
from jarvis.tools import ToolRegistry

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
