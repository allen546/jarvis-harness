import asyncio
from jarvis.models.base import Message


class CLIReceiver:
    def __init__(self):
        self._queue: asyncio.Queue[Message] = asyncio.Queue()

    async def start(self):
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, input, "> ")
            if line.lower() in ("quit", "exit"):
                break
            await self._queue.put(Message(role="user", content=line))

    async def get(self) -> Message | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None


class CLISender:
    async def send(self, content: str) -> None:
        print(content)
