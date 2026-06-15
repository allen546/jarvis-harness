import json
import os
import aiofiles
from jarvis.memory.base import BaseMemoryEngine, SessionContext
from jarvis.models.base import Message

class JSONLMemoryEngine(BaseMemoryEngine):
    def __init__(self, file_path: str = "history.jsonl"):
        self.file_path = file_path

    async def load_history(self, context: SessionContext) -> list[Message]:
        if not os.path.exists(self.file_path):
            return []
        messages = []
        async with aiofiles.open(self.file_path, mode="r") as f:
            async for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("session_id") == context.session_id:
                    messages.append(Message(**data["message"]))
        return messages

    async def save_history(self, context: SessionContext, messages: list[Message]):
        async with aiofiles.open(self.file_path, mode="a") as f:
            for m in messages:
                line = {"session_id": context.session_id, "message": m.model_dump()}
                await f.write(json.dumps(line) + "\n")
