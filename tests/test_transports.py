import pytest
from jarvis.transports.discord import DiscordTransport, MockDiscordAPI
from jarvis.transports.qq import QQTransport, MockQQAPI
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

@pytest.mark.asyncio
async def test_qq_transport_mock():
    api = MockQQAPI()
    transport = QQTransport(client=api)
    
    # 1. Test reply action
    reply_action = NativeAction(action_type="qq_reply", params={"message_id": "msg1", "content": "hello qq"})
    await transport.execute_native_action("user1", reply_action)
    assert api.replies[0] == ("user1", "hello qq", "msg1")

    # 2. Test markdown action
    md_action = NativeAction(action_type="qq_send_markdown", params={"content": "md content", "template_id": "temp1"})
    await transport.execute_native_action("user1", md_action)
    assert api.markdowns[0] == ("user1", "md content", "temp1")
