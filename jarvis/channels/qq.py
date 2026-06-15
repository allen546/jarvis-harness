import importlib
import asyncio
import re
from typing import Callable, Coroutine
from jarvis.channels.base import BaseChannel, StatefulFilter
from jarvis.models.base import Message

class QQChannel(BaseChannel):
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self._stream_filters = {}
        self.client = None

    async def start(self, on_message_callback: Callable[[str, str], Coroutine]):
        botpy = importlib.import_module("botpy")
        
        class JarvisQQClient(botpy.Client):
            async def on_at_message_create(self, message):
                await on_message_callback(str(message.channel_id), message.content)

        try:
            intents = botpy.Intents.default()
            client = JarvisQQClient(intents=intents)
        except AttributeError:
            client = JarvisQQClient()

        self.client = client

        async def run_bot():
            try:
                await client.start(appid=self.app_id, secret=self.app_secret)
            except Exception:
                pass

        asyncio.create_task(run_bot())

    async def send_message(self, session_id: str, message: Message):
        self._stream_filters.pop(session_id, None)
        if not self.client:
            botpy = importlib.import_module("botpy")
            try:
                intents = botpy.Intents.default()
                self.client = botpy.Client(intents=intents)
            except (AttributeError, TypeError):
                self.client = botpy.Client()
        coro = self.client.api.post_message(channel_id=session_id, content=message.content)
        if coro is not None and (asyncio.iscoroutine(coro) or hasattr(coro, "__await__")):
            await coro

    async def send_stream_chunk(self, session_id: str, chunk: str):
        pass

    def filter_content(self, content: str) -> str:
        # Filter internal monologue thought logs (e.g. "Now let me read file:")
        pattern = r"(?:Now let me read|Reading file|Executing command|Calling tool).*?:\s*"
        return re.sub(pattern, "", content)

    def filter_stream_chunk(self, session_id: str, chunk: str) -> str:
        if session_id not in self._stream_filters:
            self._stream_filters[session_id] = StatefulFilter()
        return self._stream_filters[session_id].filter_chunk(chunk)
