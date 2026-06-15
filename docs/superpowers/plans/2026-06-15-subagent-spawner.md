# Subagent Spawner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the isolated subagent spawner tool manager `SubagentManager` that delegates subtasks to fresh harness/session instances.

**Architecture:** Create `jarvis/subagent.py` defining `SubagentManager` with constructor accepting `harness_factory`, `get_tool_definition()`, and `execute_subagent(...)`. Test the class in `tests/test_subagent.py` using TDD.

**Tech Stack:** Python, pytest, pydantic

---

### Task 1: Write TDD Tests

**Files:**
- Create: `tests/test_subagent.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from jarvis.subagent import SubagentManager
from jarvis.memory.base import SessionContext
from jarvis.models.base import Message, ModelResponse
from jarvis.harness import TurnResult

def test_subagent_manager_tool_definition():
    manager = SubagentManager(harness_factory=MagicMock())
    tool_def = manager.get_tool_definition()
    assert tool_def["name"] == "spawn_subagent"
    assert "properties" in tool_def["parameters"]
    assert "prompt" in tool_def["parameters"]["properties"]
    assert "task_name" in tool_def["parameters"]["properties"]

@pytest.mark.asyncio
async def test_execute_subagent():
    # Setup mocks
    harness_mock = MagicMock()
    turn_result = TurnResult(
        response=ModelResponse(content="Subagent completed the task successfully."),
        tool_results=[],
        has_more_actions=False
    )
    harness_mock.execute_turn = AsyncMock(return_value=turn_result)
    harness_factory = MagicMock(return_value=harness_mock)

    manager = SubagentManager(harness_factory=harness_factory)
    
    parent_ctx = SessionContext(session_id="parent-123", scope={"parent_key": "val"})
    channel_mock = MagicMock()
    
    result = await manager.execute_subagent(
        parent_ctx=parent_ctx,
        channel=channel_mock,
        prompt="Please run the task",
        task_name="subtask-alpha"
    )
    
    assert result == "Subagent completed the task successfully."
    harness_factory.assert_called_once()
    
    # Check that execute_turn was called with the correct SessionContext
    harness_mock.execute_turn.assert_called_once()
    called_ctx = harness_mock.execute_turn.call_args[0][0]
    
    assert isinstance(called_ctx, SessionContext)
    assert called_ctx.session_id != parent_ctx.session_id
    assert called_ctx.parent_session_id == parent_ctx.session_id
    assert called_ctx.scope == {"task_name": "subtask-alpha"}
    
    # Check that channel and message are forwarded
    called_channel = harness_mock.execute_turn.call_args[0][1]
    called_msg = harness_mock.execute_turn.call_args[0][2]
    
    assert called_channel == channel_mock
    assert isinstance(called_msg, Message)
    assert called_msg.role == "user"
    assert called_msg.content == "Please run the task"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/test_subagent.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'jarvis.subagent')

### Task 2: Implement SubagentManager

**Files:**
- Create: `jarvis/subagent.py`

- [ ] **Step 1: Write implementation**

```python
import uuid
from typing import Callable
from jarvis.memory.base import SessionContext
from jarvis.channels.base import BaseChannel
from jarvis.harness import AgentHarness
from jarvis.models.base import Message

class SubagentManager:
    def __init__(self, harness_factory: Callable[[], AgentHarness]):
        self.harness_factory = harness_factory

    def get_tool_definition(self) -> dict:
        return {
            "name": "spawn_subagent",
            "description": "Spawn a subagent to execute a subtask.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt or instruction for the subagent to execute."
                    },
                    "task_name": {
                        "type": "string",
                        "description": "The name or identifier of the subtask."
                    }
                },
                "required": ["prompt", "task_name"]
            }
        }

    async def execute_subagent(
        self,
        parent_ctx: SessionContext,
        channel: BaseChannel,
        prompt: str,
        task_name: str
    ) -> str:
        sub_ctx = SessionContext(
            session_id=str(uuid.uuid4()),
            parent_session_id=parent_ctx.session_id,
            scope={"task_name": task_name}
        )
        harness = self.harness_factory()
        user_message = Message(role="user", content=prompt)
        result = await harness.execute_turn(sub_ctx, channel, user_message)
        return result.response.content or ""
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/test_subagent.py -v`
Expected: PASS

- [ ] **Step 3: Commit all changes**

Run:
```bash
git add docs/superpowers/specs/2026-06-15-subagent-spawner-design.md docs/superpowers/plans/2026-06-15-subagent-spawner.md tests/test_subagent.py jarvis/subagent.py
git commit -m "feat: implement isolated subagent spawner tool"
```
