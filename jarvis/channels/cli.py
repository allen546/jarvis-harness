import sys
from jarvis.channels.base import BaseChannel
from jarvis.models.base import Message

class CLIChannel(BaseChannel):
    _last_active_session = None

    async def send_stream_chunk(self, session_id: str, chunk: str):
        if session_id != CLIChannel._last_active_session:
            if CLIChannel._last_active_session is not None:
                sys.stdout.write("\n")
            sys.stdout.write(f"[{session_id}] ")
            CLIChannel._last_active_session = session_id
        sys.stdout.write(chunk)
        sys.stdout.flush()

    async def send_message(self, session_id: str, message: Message):
        if message.content.startswith("Error:"):
            sys.stdout.write(f"\n{message.content}")
        else:
            sys.stdout.write("\n")
        sys.stdout.flush()
