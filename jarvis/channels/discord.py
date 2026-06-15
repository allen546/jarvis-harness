import importlib
from typing import Optional, Any
from jarvis.channels.base import BaseChannel
from jarvis.models.base import Message

class DiscordChannel(BaseChannel):
    def __init__(self, bot_token: str, guild_id: Optional[str] = None):
        self.bot_token = bot_token
        self.guild_id = guild_id

    async def send_message(self, session_id: str, message: Message):
        discord = importlib.import_module("discord")
        pass

    async def send_stream_chunk(self, session_id: str, chunk: str):
        pass
