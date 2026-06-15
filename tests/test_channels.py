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


