from jarvis.models.base import Message
from typing import Any

class BaseChannel:
    async def send_message(self, session_id: str, message: Message):
        raise NotImplementedError

    async def send_stream_chunk(self, session_id: str, chunk: str):
        pass

    def filter_content(self, content: str) -> str:
        return content

    def get_channel_tools(self, session_id: str) -> list[Any]:
        return []
