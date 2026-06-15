import pytest
import os
from jarvis.memory.jsonl import JSONLMemoryEngine
from jarvis.models.base import Message
from jarvis.memory.base import SessionContext

@pytest.mark.asyncio
async def test_jsonl_memory(tmp_path):
    history_file = tmp_path / "sessions.jsonl"
    engine = JSONLMemoryEngine(file_path=str(history_file))
    ctx = SessionContext(session_id="session-1")
    await engine.save_history(ctx, [Message(role="user", content="Hi")])
    loaded = await engine.load_history(ctx)
    assert len(loaded) == 1
