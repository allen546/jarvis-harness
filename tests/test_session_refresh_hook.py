import json
import pytest
from jarvis.hooks import SessionRefreshHook
from jarvis.models.base import Message, ModelResponse, BaseModelClient
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState
from jarvis.tools import ToolRegistry
from jarvis.kernel import AgentKernel


class MockModel(BaseModelClient):
    @classmethod
    def from_cfg(cls, cfg):
        return cls()

    async def generate(self, messages, tools):
        return ModelResponse(content=json.dumps({
            "summary": "Chat compressed: discussed project setup and deployment.",
            "facts": [
                {"text": "User prefers Python for backend.", "tags": ["preference"], "confidence": 0.9}
            ],
            "procedures": [
                {
                    "name": "deploy-service",
                    "trigger": "when deploying a service",
                    "summary": "Deploy the service with docker-compose",
                    "steps": ["build", "push", "restart"],
                    "tools": ["bash"],
                    "confidence": 0.85
                }
            ]
        }))


@pytest.mark.asyncio
async def test_session_refresh_hook():
    hook = SessionRefreshHook(threshold=5, keep_messages=3)
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

    messages = list(state.history)
    res = await hook.before_model(ctx, messages)

    # Check that undistilled messages (all except last keep_messages) are compressed
    # 6 messages, keep=3, so first 3 are compressed into summary
    assert len(state.history) == 4  # 1 summary + remaining 3 messages
    assert state.history[0].role == "system"
    assert "[Session refresh:" in state.history[0].content
    assert state.history[0].metadata.get("distilled") is True
    assert state.history[1].content == "m4"
    assert state.history[2].content == "m5"
    assert state.history[3].content == "m6"

    # Returned messages: when no system prompt in input, summary_msg is included
    assert res.messages is not None
    assert len(res.messages) == 4  # summary + m4 + m5 + m6
    assert res.messages[0].role == "system"
    assert "[Session refresh:" in res.messages[0].content


@pytest.mark.asyncio
async def test_session_refresh_below_threshold():
    hook = SessionRefreshHook(threshold=5, keep_messages=3)
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
async def test_session_refresh_already_distilled():
    """Already-distilled messages should not be re-processed."""
    hook = SessionRefreshHook(threshold=5, keep_messages=3)
    state = SessionState(id="sess_distilled")
    state.history = [
        Message(role="user", content="m1", metadata={"distilled": True}),
        Message(role="assistant", content="m2", metadata={"distilled": True}),
        Message(role="user", content="m3", metadata={"distilled": True}),
        Message(role="assistant", content="m4"),
        Message(role="user", content="m5"),
        Message(role="assistant", content="m6"),
    ]
    ctx = AgentContext(config=RuntimeConfig(), session=state, model=MockModel(), tools=ToolRegistry(), hooks=[])

    messages = list(state.history)
    res = await hook.before_model(ctx, messages)

    # All compressible messages are already distilled -> no-op
    assert res.messages is None
    assert len(state.history) == 6  # unchanged


@pytest.mark.asyncio
async def test_session_refresh_preserves_system_prompt():
    hook = SessionRefreshHook(threshold=5, keep_messages=3)
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

    # The returned messages preserve the active system prompt at index 0,
    # followed by recent non-compressed messages (summary_msg is filtered out by identity)
    assert res.messages is not None
    assert len(res.messages) == 4  # 1 system prompt + 3 remaining messages
    assert res.messages[0].role == "system"
    assert res.messages[0].content == "SYS_PROMPT"
    assert res.messages[1].content == "m4"
    assert res.messages[2].content == "m5"
    assert res.messages[3].content == "m6"


@pytest.mark.asyncio
async def test_session_refresh_integration_with_kernel():
    hook = SessionRefreshHook(threshold=5, keep_messages=3)
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

    # After the turn, history contains: m3 (overlap from keep), m4, m5, assistant response
    # The summary message is role="system" so _without_system_prompt strips it from session.history
    assert len(state.history) == 4
    assert state.history[0].content == "m3"
    assert state.history[1].content == "m4"
    assert state.history[2].content == "m5"
    assert state.history[3].role == "assistant"
