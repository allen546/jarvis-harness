import pytest
from unittest.mock import AsyncMock, MagicMock
from jarvis.subagent import SubagentManager
from jarvis.memory.base import SessionContext
from jarvis.models.base import Message, ModelResponse
from jarvis.harness import TurnResult

def test_subagent_manager_tool_definition():
    manager = SubagentManager(harness_factory=MagicMock())
    tool_def = manager.get_tool_definition()
    assert tool_def["name"] == "spawn_subagent"
    assert "properties" in tool_def["parameters"]
    assert "prompt" in tool_def["parameters"]["properties"]
    assert "task_name" in tool_def["parameters"]["properties"]

@pytest.mark.asyncio
async def test_execute_subagent():
    # Setup mocks
    harness_mock = MagicMock()
    turn_result = TurnResult(
        response=ModelResponse(content="Subagent completed the task successfully."),
        tool_results=[],
        has_more_actions=False
    )
    harness_mock.execute_turn = AsyncMock(return_value=turn_result)
    harness_factory = MagicMock(return_value=harness_mock)

    manager = SubagentManager(harness_factory=harness_factory)
    
    parent_ctx = SessionContext(session_id="parent-123", scope={"parent_key": "val"})
    channel_mock = MagicMock()
    
    result = await manager.execute_subagent(
        parent_ctx=parent_ctx,
        channel=channel_mock,
        prompt="Please run the task",
        task_name="subtask-alpha"
    )
    
    assert result == "Subagent completed the task successfully."
    harness_factory.assert_called_once()
    
    # Check that execute_turn was called with the correct SessionContext
    harness_mock.execute_turn.assert_called_once()
    called_ctx = harness_mock.execute_turn.call_args[0][0]
    
    assert isinstance(called_ctx, SessionContext)
    assert called_ctx.session_id != parent_ctx.session_id
    assert called_ctx.parent_session_id == parent_ctx.session_id
    assert called_ctx.scope == {"task_name": "subtask-alpha"}
    
    # Check that channel and message are forwarded
    called_channel = harness_mock.execute_turn.call_args[0][1]
    called_msg = harness_mock.execute_turn.call_args[0][2]
    
    assert called_channel == channel_mock
    assert isinstance(called_msg, Message)
    assert called_msg.role == "user"
    assert called_msg.content == "Please run the task"
