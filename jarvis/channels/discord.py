import importlib
import asyncio
from typing import Optional, Any, Callable, Coroutine
from jarvis.channels.base import BaseChannel
from jarvis.models.base import Message

class DiscordChannel(BaseChannel):
    def __init__(self, bot_token: str, guild_id: Optional[str] = None):
        self.bot_token = bot_token
        self.guild_id = guild_id
        self.client = None

    async def start(self, on_message_callback: Callable[[str, str], Coroutine]):
        discord = importlib.import_module("discord")
        intents = discord.Intents.default()
        intents.message_content = True
        
        client = discord.Client(intents=intents)
        self.client = client

        @client.event
        async def on_message(message):
            if message.author == client.user:
                return
            await on_message_callback(str(message.channel.id), message.content)

        async def run_bot():
            try:
                await client.start(self.bot_token)
            except Exception:
                pass

        asyncio.create_task(run_bot())

    async def send_message(self, session_id: str, message: Message):
        if not self.client:
            discord = importlib.import_module("discord")
            intents = discord.Intents.default()
            intents.message_content = True
            self.client = discord.Client(intents=intents)
        
        channel_id = int(session_id)
        channel = self.client.get_channel(channel_id)
        if channel is None:
            fetch_coro = self.client.fetch_channel(channel_id)
            if fetch_coro is not None and (asyncio.iscoroutine(fetch_coro) or hasattr(fetch_coro, "__await__")):
                channel = await fetch_coro
            else:
                channel = fetch_coro
        
        if channel is not None:
            send_coro = channel.send(message.content)
            if send_coro is not None and (asyncio.iscoroutine(send_coro) or hasattr(send_coro, "__await__")):
                await send_coro

    async def send_stream_chunk(self, session_id: str, chunk: str):
        pass
