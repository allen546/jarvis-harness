import json
import pytest
from pathlib import Path
from jarvis.hooks import JSONLHistoryHook
from jarvis.models.base import Message
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState
from jarvis.tools import ToolRegistry

@pytest.mark.asyncio
async def test_jsonl_history_hook(tmp_path: Path):
    storage_dir = tmp_path / "storage"
    hook = JSONLHistoryHook(storage_dir=str(storage_dir))
    state = SessionState(id="sess1")
    ctx = AgentContext(config=RuntimeConfig(), session=state, model=None, tools=ToolRegistry(), hooks=[])
    
    # 1. Test before_model loads empty history
    await hook.before_model(ctx, [])
    assert len(state.history) == 0

    # 2. Test after_turn writes history
    user_msg = Message(role="user", content="hello")
    assistant_msg = Message(role="assistant", content="hi there")
    state.history.extend([user_msg, assistant_msg])
    await hook.after_turn(ctx, assistant_msg)

    # Check file exists and contains correct lines
    file_path = storage_dir / "sessions" / "sess1" / "history.jsonl"
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["content"] == "hello"
    assert json.loads(lines[1])["content"] == "hi there"

    # 3. Test before_model loads from file
    new_state = SessionState(id="sess1")
    new_ctx = AgentContext(config=RuntimeConfig(), session=new_state, model=None, tools=ToolRegistry(), hooks=[])
    await hook.before_model(new_ctx, [])
    assert len(new_state.history) == 2
    assert new_state.history[0].content == "hello"
    assert new_state.history[1].content == "hi there"
