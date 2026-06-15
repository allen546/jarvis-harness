import pytest
from jarvis.channels.discord import DiscordChannel
from jarvis.channels.qq import QQChannel
from jarvis.models.base import Message

def test_channels_initialization():
    discord = DiscordChannel(bot_token="token123")
    qq = QQChannel(app_id="app123", app_secret="sec123")
    assert discord.bot_token == "token123"
    assert qq.app_id == "app123"

    # Test content filtering on QQ channel
    assert qq.filter_content("Now let me read main.py: Hello!") == "Hello!"
    # Test content filtering on Discord (should preserve the thoughts)
    assert discord.filter_content("Now let me read main.py: Hello!") == "Now let me read main.py: Hello!"

@pytest.mark.asyncio
async def test_cli_channel(capsys):
    from jarvis.channels.cli import CLIChannel
    
    # Reset state to ensure clean test environment
    CLIChannel._last_active_session = None
    
    channel = CLIChannel()
    await channel.send_stream_chunk("session1", "Hello ")
    await channel.send_stream_chunk("session1", "World")
    
    captured = capsys.readouterr()
    assert captured.out == "[session1] Hello World"
    
    msg = Message(role="assistant", content="ignored")
    await channel.send_message("session1", msg)
    
    captured = capsys.readouterr()
    assert captured.out == "\n"

    # Test switching session
    await channel.send_stream_chunk("session2", "New session text")
    captured = capsys.readouterr()
    assert captured.out == "\n[session2] New session text"

    # Test error output
    error_msg = Message(role="assistant", content="Error: System overload")
    await channel.send_message("session2", error_msg)
    captured = capsys.readouterr()
    assert captured.out == "\nError: System overload"

@pytest.mark.asyncio
async def test_discord_stateful_channel(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock, AsyncMock
    
    # Setup mock discord client
    class MockDiscordClient:
        def __init__(self, intents=None):
            self.event_handlers = {}
            self.start = AsyncMock()
            self.get_channel = MagicMock()
            self.fetch_channel = AsyncMock()
            self.user = MagicMock()

        def event(self, func):
            self.event_handlers[func.__name__] = func
            return func
            
    mock_discord = MagicMock()
    mock_discord.Intents.default = MagicMock()
    mock_discord.Client = MockDiscordClient
    
    # Mock importlib.import_module
    def mock_import(name):
        if name == "discord":
            return mock_discord
        raise ImportError()
        
    monkeypatch.setattr("importlib.import_module", mock_import)
    
    # 1. Start the channel and check event registration
    channel = DiscordChannel(bot_token="test_discord_token")
    
    received_messages = []
    async def on_message_callback(channel_id: str, content: str):
        received_messages.append((channel_id, content))
        
    await channel.start(on_message_callback)
    
    # Wait briefly for asyncio task to schedule
    await asyncio.sleep(0.01)
    
    assert channel.client is not None
    assert "on_message" in channel.client.event_handlers
    
    # Trigger on_message (ignoring own bot messages)
    mock_msg_own = MagicMock()
    mock_msg_own.author = channel.client.user
    mock_msg_own.channel.id = 12345
    mock_msg_own.content = "bot message"
    
    await channel.client.event_handlers["on_message"](mock_msg_own)
    assert len(received_messages) == 0
    
    mock_msg_other = MagicMock()
    mock_msg_other.author = MagicMock() # not channel.client.user
    mock_msg_other.channel.id = 54321
    mock_msg_other.content = "hello jarvis"
    
    await channel.client.event_handlers["on_message"](mock_msg_other)
    assert len(received_messages) == 1
    assert received_messages[0] == ("54321", "hello jarvis")
    
    # 2. Sending messages utilizes the active client connection to dispatch the payload
    # Case A: channel is cached (get_channel returns it)
    mock_channel = MagicMock()
    mock_channel.send = AsyncMock()
    channel.client.get_channel.return_value = mock_channel
    
    await channel.send_message("54321", Message(role="assistant", content="response from jarvis"))
    channel.client.get_channel.assert_called_with(54321)
    mock_channel.send.assert_called_with("response from jarvis")
    
    # Case B: channel is not cached (get_channel returns None, fetch_channel is called)
    channel.client.get_channel.return_value = None
    channel.client.fetch_channel.reset_mock()
    
    mock_channel_fetched = MagicMock()
    mock_channel_fetched.send = AsyncMock()
    channel.client.fetch_channel.return_value = mock_channel_fetched
    
    await channel.send_message("99999", Message(role="assistant", content="fetched response"))
    channel.client.get_channel.assert_called_with(99999)
    channel.client.fetch_channel.assert_called_with(99999)
    mock_channel_fetched.send.assert_called_with("fetched response")

@pytest.mark.asyncio
async def test_qq_stateful_channel(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock, AsyncMock
    
    # Setup mock botpy client
    class MockBotpyClient:
        def __init__(self, intents=None):
            self.start = AsyncMock()
            self.api = MagicMock()
            self.api.post_message = AsyncMock()

    mock_botpy = MagicMock()
    mock_botpy.Intents.default = MagicMock()
    mock_botpy.Client = MockBotpyClient
    
    # Mock importlib.import_module
    def mock_import(name):
        if name == "botpy":
            return mock_botpy
        raise ImportError()
        
    monkeypatch.setattr("importlib.import_module", mock_import)
    
    # 1. Start the channel and check event registration
    channel = QQChannel(app_id="app123", app_secret="sec123")
    
    received_messages = []
    async def on_message_callback(channel_id: str, content: str):
        received_messages.append((channel_id, content))
        
    await channel.start(on_message_callback)
    
    # Wait briefly for asyncio task to schedule
    await asyncio.sleep(0.01)
    
    assert channel.client is not None
    assert hasattr(channel.client, "on_at_message_create")
    
    # Trigger on_at_message_create
    mock_msg = MagicMock()
    mock_msg.channel_id = "qq_channel_555"
    mock_msg.content = "hello qq bot"
    
    await channel.client.on_at_message_create(mock_msg)
    assert len(received_messages) == 1
    assert received_messages[0] == ("qq_channel_555", "hello qq bot")
    
    # 2. Sending messages utilizes the active client connection to dispatch the payload
    await channel.send_message("qq_channel_555", Message(role="assistant", content="response to qq"))
    channel.client.api.post_message.assert_called_with(channel_id="qq_channel_555", content="response to qq")


