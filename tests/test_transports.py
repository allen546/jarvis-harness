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
