from __future__ import annotations

import asyncio

from jarvis.events import ErrorEvent, MessageEvent, NativeActionEvent, TextDeltaEvent, ToolCallEvent, ToolResultEvent
from jarvis.models.base import Message
from jarvis.runtime import AgentSession


def render_cli_event(event: object) -> None:
    if isinstance(event, MessageEvent):
        print(event.message.content)
    elif isinstance(event, TextDeltaEvent):
        print(event.content, end="", flush=True)
    elif isinstance(event, ToolCallEvent):
        print(f"\n[tool_call] {event.tool_call.tool_name} {event.tool_call.arguments}")
    elif isinstance(event, ToolResultEvent):
        prefix = "[tool_error]" if event.is_error else "[tool_result]"
        print(f"\n{prefix} {event.tool_name}: {event.content}")
    elif isinstance(event, NativeActionEvent):
        print(f"[native_action {event.action.action_type}] {event.action.params}")
    elif isinstance(event, ErrorEvent):
        print(f"[error] {event.message}")


async def run_cli(session: AgentSession) -> None:
    while True:
        line = await asyncio.to_thread(input, "> ")
        if line.lower() in {"exit", "quit"}:
            return
        async for event in session.submit(Message(role="user", content=line)):
            render_cli_event(event)
