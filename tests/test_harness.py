import pytest
from unittest.mock import AsyncMock, MagicMock, call
from jarvis.harness import AgentHarness, TurnResult
from jarvis.config import HarnessConfig
from jarvis.memory.base import SessionContext
from jarvis.models.base import Message, ModelResponse
from jarvis.channels.base import StatefulFilter
from jarvis.channels.qq import QQChannel

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
    # Stub stream filter stripping thoughts
    channel.filter_stream_chunk = lambda sid, x: x.replace("Reading file: ", "")
    channel.filter_content = lambda x: x.replace("Reading file: ", "")

    user_message = Message(role="user", content="Hello")
    result = await harness.execute_turn(ctx, channel, user_message)
    
    # 1. Loads history
    memory.load_history.assert_called_once_with(ctx)
    
    # 2. Saves history (user message, then assistant response)
    assert memory.save_history.call_count == 2
    memory.save_history.assert_has_calls([
        call(ctx, [user_message]),
        call(ctx, [Message(role="assistant", content="Reading file: chunk1Reading file: chunk2")]),
    ])

    # 3. Streams chunks & applies channel-side content filtering
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
    
    pre_hook_called = False
    async def pre_hook(session_ctx, history):
        nonlocal pre_hook_called
        pre_hook_called = True
        history.append(Message(role="user", content="hook_added"))
        return history
        
    harness.pre_turn_hooks.append(pre_hook)

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
    channel.filter_stream_chunk = lambda sid, x: x
    channel.filter_content = lambda x: x

    user_message = Message(role="user", content="Hello")
    result = await harness.execute_turn(ctx, channel, user_message)
    
    assert pre_hook_called
    assert post_hook_called
    assert result.response.content == "response"


def test_stateful_filter_split_tokens():
    f = StatefulFilter()
    
    # 1. Test target prefix split across chunks
    assert f.filter_chunk("Now let ") == ""
    assert f.filter_chunk("me read: success") == "success"
    
    # 2. Test normal text output
    assert f.filter_chunk(" and normal output") == " and normal output"
    
    # 3. Test another prefix split across chunks with whitespace stripping
    f2 = StatefulFilter()
    assert f2.filter_chunk("Executing ") == ""
    assert f2.filter_chunk("command:\n  ") == ""
    assert f2.filter_chunk("working") == "working"
    
    # 4. Test normal text that happens to match prefixes partially but not fully
    f3 = StatefulFilter()
    assert f3.filter_chunk("Now let me ") == ""
    assert f3.filter_chunk("go home") == "Now let me go home"


def test_qq_channel_stateful_filter():
    qq = QQChannel(app_id="app123", app_secret="sec123")
    
    # Test filtering content with re.sub (filter_content works on full string)
    assert qq.filter_content("Now let me read: Hello!") == "Hello!"
    
    # Test streaming chunks
    assert qq.filter_stream_chunk("session-1", "Now let ") == ""
    assert qq.filter_stream_chunk("session-1", "me read: Hello!") == "Hello!"
    
    # Different session should not interfere
    assert qq.filter_stream_chunk("session-2", "Executing ") == ""
    assert qq.filter_stream_chunk("session-1", " and normal") == " and normal"
    assert qq.filter_stream_chunk("session-2", "command: working") == "working"


@pytest.mark.asyncio
async def test_execute_turn_stream_error_handling():
    config = HarnessConfig(system_prompt="system instructions")
    
    model_client = MagicMock()
    async def mock_stream(msgs, tools):
        yield ModelResponse(content="Partial text", tool_calls=[], raw_response=None)
        raise ValueError("Stream failed!")
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
    
    ctx = SessionContext(session_id="session-error")
    channel = MagicMock()
    channel.send_stream_chunk = AsyncMock()
    channel.send_message = AsyncMock()
    channel.filter_stream_chunk = lambda sid, x: x

    user_message = Message(role="user", content="Hello")
    
    with pytest.raises(ValueError, match="Stream failed!"):
        await harness.execute_turn(ctx, channel, user_message)
    
    # Assert partial response was saved as assistant message
    assert memory.save_history.call_count == 2
    memory.save_history.assert_has_calls([
        call(ctx, [user_message]),
        call(ctx, [Message(role="assistant", content="Partial text")]),
    ])
    
    # Assert channel was notified of the error
    channel.send_message.assert_called_once()
    error_msg = channel.send_message.call_args[0][1]
    assert error_msg.role == "assistant"
    assert "Error: Stream failed!" in error_msg.content


@pytest.mark.asyncio
async def test_execute_turn_pre_hook_returns_none():
    config = HarnessConfig(system_prompt="system instructions")
    
    model_client = MagicMock()
    async def mock_stream(msgs, tools):
        # Verify hook did NOT change history to None
        assert msgs is not None
        assert len(msgs) == 2
        yield ModelResponse(content="OK", tool_calls=[], raw_response=None)
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
    
    async def pre_hook(session_ctx, history):
        return None  # Should be ignored
        
    harness.pre_turn_hooks.append(pre_hook)
    
    ctx = SessionContext(session_id="session-hooks-none")
    channel = MagicMock()
    channel.send_stream_chunk = AsyncMock()
    channel.send_message = AsyncMock()
    channel.filter_stream_chunk = lambda sid, x: x
    channel.filter_content = lambda x: x

    user_message = Message(role="user", content="Hello")
    await harness.execute_turn(ctx, channel, user_message)
