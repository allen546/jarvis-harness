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
