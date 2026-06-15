from jarvis.models.base import Message
from typing import Any
import asyncio

class StatefulFilter:
    TARGET_PREFIXES = ["Now let me read", "Reading file", "Executing command", "Calling tool"]

    def __init__(self):
        self.buffer = ""
        self.state = "NORMAL"  # "NORMAL", "SUPPRESSING", "STRIPPING_WHITESPACE"

    def filter_chunk(self, chunk: str) -> str:
        self.buffer += chunk
        output = []

        while True:
            if self.state == "NORMAL":
                found_action = False
                for i in range(len(self.buffer) + 1):
                    sub = self.buffer[i:]
                    full_match = False
                    for target in self.TARGET_PREFIXES:
                        if sub.startswith(target):
                            output.append(self.buffer[:i])
                            self.buffer = self.buffer[i + len(target):]
                            self.state = "SUPPRESSING"
                            full_match = True
                            found_action = True
                            break
                    if full_match:
                        break
                
                if found_action:
                    continue
                
                earliest_prefix_idx = None
                for i in range(len(self.buffer)):
                    sub = self.buffer[i:]
                    is_prefix = False
                    for target in self.TARGET_PREFIXES:
                        if target.startswith(sub):
                            is_prefix = True
                            break
                    if is_prefix:
                        earliest_prefix_idx = i
                        break
                
                if earliest_prefix_idx is not None:
                    output.append(self.buffer[:earliest_prefix_idx])
                    self.buffer = self.buffer[earliest_prefix_idx:]
                    break
                else:
                    output.append(self.buffer)
                    self.buffer = ""
                    break

            elif self.state == "SUPPRESSING":
                idx = self.buffer.find(":")
                if idx != -1:
                    self.buffer = self.buffer[idx + 1:]
                    self.state = "STRIPPING_WHITESPACE"
                else:
                    self.buffer = ""
                    break

            elif self.state == "STRIPPING_WHITESPACE":
                while self.buffer:
                    if self.buffer[0].isspace():
                        self.buffer = self.buffer[1:]
                    else:
                        self.state = "NORMAL"
                        break
                if self.state == "STRIPPING_WHITESPACE":
                    break

        return "".join(output)

class BaseChannel:
    async def send_message(self, session_id: str, message: Message):
        raise NotImplementedError

    async def send_stream_chunk(self, session_id: str, chunk: str):
        pass

    def filter_content(self, content: str) -> str:
        return content

    def filter_stream_chunk(self, session_id: str, chunk: str) -> str:
        return chunk

    def get_channel_tools(self, session_id: str) -> list[Any]:
        return []

class QueueChannel(BaseChannel):
    def __init__(self):
        self.queue = asyncio.Queue()

    async def send_stream_chunk(self, session_id: str, chunk: str):
        await self.queue.put({"event": "chunk", "data": chunk})

    async def send_message(self, session_id: str, message: Message):
        await self.queue.put({"event": "message", "data": message.model_dump()})
        await self.queue.put(None)

