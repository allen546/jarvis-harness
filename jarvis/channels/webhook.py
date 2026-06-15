from jarvis.channels.base import BaseChannel
from jarvis.models.base import Message
import httpx

class WebhookChannel(BaseChannel):
    _shared_client = None

    @classmethod
    def _get_client(cls):
        if cls._shared_client is None:
            cls._shared_client = httpx.AsyncClient()
        return cls._shared_client

    def __init__(self, callback_url: str):
        self.callback_url = callback_url

    async def send_message(self, session_id: str, message: Message):
        client = self._get_client()
        await client.post(self.callback_url, json={
            "session_id": session_id,
            "message": message.model_dump()
        })

    async def send_stream_chunk(self, session_id: str, chunk: str):
        client = self._get_client()
        await client.post(self.callback_url + "/stream", content=chunk.encode("utf-8"))

