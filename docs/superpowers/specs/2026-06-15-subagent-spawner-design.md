# Design Spec: Subagent Spawner and Tool Delegation

This document specifies the design for the subagent spawner (`SubagentManager`) in the Jarvis agent harness.

## 1. Requirements

- Expose `SubagentManager` inside `jarvis/subagent.py`.
- `SubagentManager` takes a `harness_factory` callable in its constructor.
- `SubagentManager.get_tool_definition(self) -> dict` returns the JSON schema tool definition for the `spawn_subagent` tool.
- `SubagentManager.execute_subagent(self, parent_ctx: SessionContext, channel: BaseChannel, prompt: str, task_name: str) -> str` executes a turn on a fresh harness instance and returns the response string.
- Instantiates a fresh subagent `SessionContext` with:
  - Generated UUID `session_id`.
  - `parent_session_id` set to `parent_ctx.session_id`.
  - `scope` dictionary containing `{"task_name": task_name}`.

## 2. API Design

### 2.1 SubagentManager Class

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

## 3. Testing Plan

We will implement `tests/test_subagent.py` to:
1. Verify `get_tool_definition` returns the correct schema dictionary.
2. Verify `execute_subagent` correctly creates the new session context with the expected attributes (ID, parent linkage, and task name scope) and calls the harness factory, executes the turn, and returns the response string.
