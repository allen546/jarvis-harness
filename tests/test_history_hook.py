import json
import pytest
from pathlib import Path
from typing import Any
from jarvis.hooks import JSONLHistoryHook
from jarvis.models.base import BaseModelClient, Message, ModelResponse, ToolCall
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState
from jarvis.tools import Tool, ToolRegistry
from jarvis.kernel import AgentKernel


class DummyModel(BaseModelClient):
    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        return ModelResponse()


@pytest.mark.asyncio
async def test_jsonl_history_hook(tmp_path: Path):
    storage_dir = tmp_path / "storage"
    hook = JSONLHistoryHook(storage_dir=str(storage_dir))
    state = SessionState(id="sess1")
    ctx = AgentContext(config=RuntimeConfig(), session=state, model=DummyModel(), tools=ToolRegistry(), hooks=[])

    # 1. Test before_model — sessions start fresh, no history loaded from disk
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

    # 3. Test before_model starts fresh — no history loaded from file
    new_state = SessionState(id="sess1")
    new_ctx = AgentContext(config=RuntimeConfig(), session=new_state, model=DummyModel(), tools=ToolRegistry(), hooks=[])
    await hook.before_model(new_ctx, [])
    assert len(new_state.history) == 0


class SequenceModel(BaseModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.calls: list[list[Message]] = []

    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        self.calls.append(list(messages))
        return self.responses.pop(0)


async def echo(args: dict[str, object]) -> str:
    return str(args["value"])


@pytest.mark.asyncio
async def test_history_hook_real_flow(tmp_path: Path):
    storage_dir = tmp_path / "storage"

    # Pre-populate history on disk (audit trail only — not replayed)
    session_id = "sess_real"
    file_path = storage_dir / "sessions" / session_id / "history.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(Message(role="user", content="past user msg").model_dump()) + "\n")
        f.write(json.dumps(Message(role="assistant", content="past assistant msg").model_dump()) + "\n")

    hook = JSONLHistoryHook(storage_dir=str(storage_dir))
    state = SessionState(id=session_id)
    model = SequenceModel([ModelResponse(content="current assistant response")])
    ctx = AgentContext(
        config=RuntimeConfig(system_prompt="system"),
        session=state,
        model=model,
        tools=ToolRegistry(),
        hooks=[hook],
    )

    kernel = AgentKernel()
    user_msg = Message(role="user", content="current user msg")

    events = [event async for event in kernel.run_turn(ctx, user_msg)]

    # Sessions start fresh: model receives system_prompt + current user only (no prior history)
    assert len(model.calls) == 1
    received = model.calls[0]
    roles_and_contents = [(m.role, m.content) for m in received]
    assert roles_and_contents == [
        ("system", "system"),
        ("user", "current user msg")
    ]

    # History file now contains the new turn's messages
    lines = file_path.read_text().splitlines()
    assert len(lines) == 4
    assert json.loads(lines[2])["content"] == "current user msg"
    assert json.loads(lines[3])["content"] == "current assistant response"


@pytest.mark.asyncio
async def test_history_hook_with_tool_calls(tmp_path: Path):
    storage_dir = tmp_path / "storage"
    hook = JSONLHistoryHook(storage_dir=str(storage_dir))
    state = SessionState(id="sess_tools")

    call = ToolCall(call_id="c1", tool_name="echo", arguments={"value": "ok"})
    model = SequenceModel([
        ModelResponse(content="using tool", tool_calls=[call]),
        ModelResponse(content="done")
    ])

    ctx = AgentContext(
        config=RuntimeConfig(system_prompt="system"),
        session=state,
        model=model,
        tools=ToolRegistry([Tool("echo", "Echo value.", {"type": "object"}, echo)]),
        hooks=[hook],
    )

    kernel = AgentKernel()
    user_msg = Message(role="user", content="run tool please")

    events = [event async for event in kernel.run_turn(ctx, user_msg)]

    # Verify history file has logged everything from the turn
    file_path = storage_dir / "sessions" / "sess_tools" / "history.jsonl"
    assert file_path.exists()
    lines = file_path.read_text().splitlines()

    # Should contain: user msg, tool call assistant msg, tool response, done assistant msg
    assert len(lines) == 4
    msgs = [json.loads(line) for line in lines]
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "run tool please"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "using tool"
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["content"] == "ok"
    assert msgs[3]["role"] == "assistant"
    assert msgs[3]["content"] == "done"
