import pytest
from unittest.mock import AsyncMock, MagicMock, call
from jarvis.harness import AgentHarness, TurnResult
from jarvis.config import HarnessConfig
from jarvis.memory.base import SessionContext
from jarvis.models.base import Message, ModelResponse

@pytest.mark.asyncio
async def test_execute_turn_streamed_filtered():
    config = HarnessConfig(system_prompt="system instructions")
    
    model_client = MagicMock()
    async def mock_stream(msgs, tools):
        yield ModelResponse(content="Reading file: chunk1", tool_calls=[], raw_response=None)
        yield ModelResponse(content="Reading file: chunk2", tool_calls=[], raw_response=None)
    model_client.generate_stream = mock_stream
    
    memory = MagicMock()
    memory.load_history = AsyncMock(return_value=[])
    memory.save_history = AsyncMock()

    harness = AgentHarness(
        config=config,
        model_client=model_client,
        memory_engine=memory,
        mcp_manager=MagicMock(),
        skills_manager=MagicMock()
    )
    
    ctx = SessionContext(session_id="session-stream")
    channel = MagicMock()
    channel.send_stream_chunk = AsyncMock()
    channel.send_message = AsyncMock()
    # Stub filter stripping thoughts
    channel.filter_content = lambda x: x.replace("Reading file: ", "")

    user_message = Message(role="user", content="Hello")
    result = await harness.execute_turn(ctx, channel, user_message)
    
    # 1. Loads history
    memory.load_history.assert_called_once_with(ctx)
    
    # 2. Saves history (user message, then assistant response)
    # The first save should be for the user message
    # The second save should be for the assistant message
    assert memory.save_history.call_count == 2
    memory.save_history.assert_has_calls([
        call(ctx, [user_message]),
        call(ctx, [Message(role="assistant", content="Reading file: chunk1Reading file: chunk2")]),
    ])

    # 3. Streams chunks & applies channel-side content filtering
    # Check result content matches original unfiltered assistant content
    assert result.response.content == "Reading file: chunk1Reading file: chunk2"
    
    # Verify that filtered content was sent to stream
    channel.send_stream_chunk.assert_any_call("session-stream", "chunk1")
    channel.send_stream_chunk.assert_any_call("session-stream", "chunk2")
    
    # 4. Filtered assistant response message sent to channel
    channel.send_message.assert_called_once()
    sent_msg = channel.send_message.call_args[0][1]
    assert sent_msg.role == "assistant"
    assert sent_msg.content == "chunk1chunk2"


@pytest.mark.asyncio
async def test_execute_turn_hooks():
    config = HarnessConfig(system_prompt="system instructions")
    
    model_client = MagicMock()
    async def mock_stream(msgs, tools):
        # We check that the model client received the modified history from pre_turn_hook
        assert len(msgs) == 3
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"
        assert msgs[2].role == "user"
        assert msgs[2].content == "hook_added"
        yield ModelResponse(content="response", tool_calls=[], raw_response=None)
    model_client.generate_stream = mock_stream
    
    memory = MagicMock()
    memory.load_history = AsyncMock(return_value=[])
    memory.save_history = AsyncMock()

    harness = AgentHarness(
        config=config,
        model_client=model_client,
        memory_engine=memory,
        mcp_manager=MagicMock(),
        skills_manager=MagicMock()
    )
    
    # Pre-turn hook that appends a message to history
    pre_hook_called = False
    async def pre_hook(session_ctx, history):
        nonlocal pre_hook_called
        pre_hook_called = True
        history.append(Message(role="user", content="hook_added"))
        return history
        
    harness.pre_turn_hooks.append(pre_hook)

    # Post-message hook that observes the response
    post_hook_called = False
    async def post_hook(session_ctx, response):
        nonlocal post_hook_called
        post_hook_called = True
        assert response.content == "response"

    harness.post_message_hooks.append(post_hook)
    
    ctx = SessionContext(session_id="session-hooks")
    channel = MagicMock()
    channel.send_stream_chunk = AsyncMock()
    channel.send_message = AsyncMock()
    channel.filter_content = lambda x: x

    user_message = Message(role="user", content="Hello")
    result = await harness.execute_turn(ctx, channel, user_message)
    
    assert pre_hook_called
    assert post_hook_called
    assert result.response.content == "response"
