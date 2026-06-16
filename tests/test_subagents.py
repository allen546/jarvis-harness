import pytest
from typing import Any
from jarvis.events import TextDeltaEvent, MessageEvent
from jarvis.models.base import BaseModelClient, ModelResponse, Message
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState, AgentSession
from jarvis.tools import ToolRegistry
from jarvis.subagent import spawn_subagent_tool, send_subagent_message_tool, close_subagent_tool

class FakeModel(BaseModelClient):
    @classmethod
    def from_cfg(cls, cfg): return cls()
    async def generate(self, messages, tools):
        return ModelResponse(content="subagent reply")

@pytest.mark.asyncio
async def test_collaborative_subagents():
    # Setup parent session context
    parent_state = SessionState(id="parent_sess")
    parent_called_events = []
    def parent_cb(ev):
        parent_called_events.append(ev)

    parent_ctx = AgentContext(
        config=RuntimeConfig(),
        session=parent_state,
        model=FakeModel(),
        tools=ToolRegistry(),
        hooks=[],
        emit_event=parent_cb
    )

    # 1. Test Spawn Subagent
    resp = await spawn_subagent_tool(parent_ctx, {"prompt": "subtask prompt", "task_name": "task1"})
    assert "sub_session_id" in resp
    assert resp["response"] == "subagent reply"
    sub_id = resp["sub_session_id"]

    # Check that events from subagent bubbled up to parent context callback
    assert len(parent_called_events) > 0
    assert any(isinstance(ev, MessageEvent) and ev.message.content == "subagent reply" for ev in parent_called_events)

    # 2. Test Send Message to Subagent
    reply_resp = await send_subagent_message_tool(parent_ctx, {"sub_session_id": sub_id, "message": "follow up"})
    assert reply_resp["response"] == "subagent reply"

    # 3. Test Close Subagent
    close_resp = await close_subagent_tool(parent_ctx, {"sub_session_id": sub_id})
    assert "closed" in close_resp["message"]


@pytest.mark.asyncio
async def test_subagent_boundary_isolation():
    # Create two isolated parent contexts
    ctx_a = AgentContext(
        config=RuntimeConfig(),
        session=SessionState(id="session_a"),
        model=FakeModel(),
        tools=ToolRegistry(),
        hooks=[]
    )
    ctx_b = AgentContext(
        config=RuntimeConfig(),
        session=SessionState(id="session_b"),
        model=FakeModel(),
        tools=ToolRegistry(),
        hooks=[]
    )

    # Spawn subagent in Session A
    resp_a = await spawn_subagent_tool(ctx_a, {"prompt": "task a", "task_name": "taskA"})
    sub_id = resp_a["sub_session_id"]

    # Try to send a message to subagent using Session B's context - should fail with ValueError
    with pytest.raises(ValueError, match="No active subagent found with ID"):
        await send_subagent_message_tool(ctx_b, {"sub_session_id": sub_id, "message": "hello"})

    # Try to close subagent using Session B's context - should not remove it from Session A's active subagents
    await close_subagent_tool(ctx_b, {"sub_session_id": sub_id})
    # Verification: it is still present in Session A's sessions
    assert sub_id in ctx_a.session.metadata.get("subagent_sessions", {})


@pytest.mark.asyncio
async def test_subagent_empty_input_validation():
    ctx = AgentContext(
        config=RuntimeConfig(),
        session=SessionState(id="parent_sess"),
        model=FakeModel(),
        tools=ToolRegistry(),
        hooks=[]
    )

    # Empty prompt spawn
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        await spawn_subagent_tool(ctx, {"prompt": "", "task_name": "task1"})

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        await spawn_subagent_tool(ctx, {"prompt": "   ", "task_name": "task1"})

    # First spawn a valid subagent
    resp = await spawn_subagent_tool(ctx, {"prompt": "valid prompt", "task_name": "task1"})
    sub_id = resp["sub_session_id"]

    # Empty message send
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        await send_subagent_message_tool(ctx, {"sub_session_id": sub_id, "message": ""})

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        await send_subagent_message_tool(ctx, {"sub_session_id": sub_id, "message": "   "})
