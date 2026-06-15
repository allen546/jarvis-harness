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
    
    channel = CLIChannel()
    await channel.send_stream_chunk("session1", "Hello ")
    await channel.send_stream_chunk("session1", "World")
    
    captured = capsys.readouterr()
    assert captured.out == "Hello World"
    
    msg = Message(role="assistant", content="ignored")
    await channel.send_message("session1", msg)
    
    captured = capsys.readouterr()
    assert captured.out == "\n"

