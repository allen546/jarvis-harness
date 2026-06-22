import pytest
from jarvis.transports.discord import DiscordTransport, MockDiscordAPI
from jarvis.models.base import NativeAction

@pytest.mark.asyncio
async def test_discord_transport_mock():
    api = MockDiscordAPI()
    transport = DiscordTransport(client=api)
    
    # 1. Test reply action
    reply_action = NativeAction(action_type="discord_reply", params={"message_id": "msg1", "content": "hello discord"})
    await transport.execute_native_action("channel1", reply_action)
    assert api.sent_messages[0] == ("channel1", "hello discord", "msg1")

    # 2. Test reaction action
    react_action = NativeAction(action_type="discord_reaction", params={"message_id": "msg1", "emoji": "👍", "action": "add"})
    await transport.execute_native_action("channel1", react_action)
    assert api.reactions[0] == ("channel1", "msg1", "👍", "add")

    # 3. Test send embed action
    embed_action = NativeAction(action_type="discord_send_embed", params={"title": "Embed Title", "description": "Embed Desc"})
    await transport.execute_native_action("channel1", embed_action)
    assert api.embeds[0] == ("channel1", {"title": "Embed Title", "description": "Embed Desc"})

    # 4. Test create thread action
    thread_action = NativeAction(action_type="discord_create_thread", params={"name": "New Thread", "message_id": "msg1"})
    await transport.execute_native_action("channel1", thread_action)
    assert api.threads[0] == ("channel1", "New Thread", "msg1")

    # 5. Verify cross-platform filtering: DiscordTransport ignores QQ action
    qq_action = NativeAction(action_type="qq_reply", params={"message_id": "msg2", "content": "ignore me"})
    await transport.execute_native_action("channel1", qq_action)
    assert len(api.sent_messages) == 1  # Should not add another message


def test_snowflake_validation():
    from jarvis.transports.discord import _to_snowflake
    assert _to_snowflake(12345) == 12345
    assert _to_snowflake("12345") == 12345
    with pytest.raises(ValueError, match="Invalid Discord snowflake ID"):
        _to_snowflake("abc")


@pytest.mark.asyncio
async def test_qq_fallback_to_plaintext():
    from unittest.mock import AsyncMock, MagicMock
    import botpy.errors
    from jarvis.transports.qq import QQBot
    from jarvis.models.base import Message

    async def mock_on_message(session_id, msg):
        return Message(role="assistant", content="test response")

    from botpy import Intents
    bot = QQBot(
        intents=Intents(public_messages=True),
        on_message=mock_on_message,
        allowed_senders=["user123"],
    )

    mock_message = MagicMock()
    mock_message.content = "hello"
    mock_message.author.user_openid = "user123"
    mock_message.attachments = []
    mock_message.id = "msg123"

    mock_api = AsyncMock()
    mock_message._api = mock_api

    call_count = 0
    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("msg_type") == 2:
            raise botpy.errors.ServerError("无效 markdown content")
        return "success"

    mock_api.post_c2c_message.side_effect = side_effect

    await bot.on_c2c_message_create(mock_message)

    assert call_count == 2
    mock_api.post_c2c_message.assert_any_call(
        openid="user123",
        msg_type=2,
        markdown={"content": "test response"},
        msg_id="msg123"
    )
    mock_api.post_c2c_message.assert_any_call(
        openid="user123",
        msg_type=0,
        content="test response",
        msg_id="msg123"
    )


@pytest.mark.asyncio
async def test_qq_voice_attachment_processing(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from jarvis.transports.qq import QQBot
    from jarvis.models.base import Message
    
    monkeypatch.setattr("jarvis.media.transcode_amr_to_mp3", lambda d: b"mock-mp3-bytes")
    
    received_msg = None
    async def mock_on_message(session_id, msg):
        nonlocal received_msg
        received_msg = msg
        return Message(role="assistant", content="reply")

    from botpy import Intents
    bot = QQBot(
        intents=Intents(public_messages=True),
        on_message=mock_on_message,
        allowed_senders=["user123"],
    )
    
    mock_message = MagicMock()
    mock_message.content = ""
    mock_message.author.user_openid = "user123"
    mock_message.id = "msg123"
    
    mock_att = MagicMock()
    mock_att.url = "http://qq.com/voice.amr"
    mock_att.content_type = "voice"
    mock_att.filename = "test.amr"
    
    # Simulate payload with raw dictionary fields
    mock_att._raw_data = {
        "content_type": "voice",
        "url": "http://qq.com/voice.amr",
        "asr_refer_text": "hello from QQ ASR"
    }
    mock_message.attachments = [mock_att]
    
    mock_api = AsyncMock()
    mock_message._api = mock_api
    
    monkeypatch.setattr(bot, "_download", AsyncMock(return_value=b"mock-amr-bytes"))
    
    await bot.on_c2c_message_create(mock_message)
    
    assert received_msg is not None
    assert len(received_msg.attachments) == 1
    att = received_msg.attachments[0]
    assert att.mime_type == "voice"
    assert "audio/mpeg" not in att.mime_type
    assert "base64," in att.url
    assert received_msg.metadata.get("asr_text") == "hello from QQ ASR"
