import sys
from jarvis.channels.base import BaseChannel
from jarvis.models.base import Message

class CLIChannel(BaseChannel):
    async def send_stream_chunk(self, session_id: str, chunk: str):
        sys.stdout.write(chunk)
        sys.stdout.flush()

    async def send_message(self, session_id: str, message: Message):
        sys.stdout.write("\n")
        sys.stdout.flush()
