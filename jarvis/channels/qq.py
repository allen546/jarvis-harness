import importlib
import re
from jarvis.channels.base import BaseChannel
from jarvis.models.base import Message

class QQChannel(BaseChannel):
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret

    async def send_message(self, session_id: str, message: Message):
        botpy = importlib.import_module("botpy")
        pass

    async def send_stream_chunk(self, session_id: str, chunk: str):
        pass

    def filter_content(self, content: str) -> str:
        # Filter internal monologue thought logs (e.g. "Now let me read file:")
        pattern = r"(?:Now let me read|Reading file|Executing command|Calling tool).*?:\s*"
        return re.sub(pattern, "", content)
