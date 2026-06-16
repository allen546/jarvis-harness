from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Union

from jarvis.models.base import Message, NativeAction, ToolCall


@dataclass(slots=True)
class TextDeltaEvent:
    session_id: str
    content: str
    event: Literal["text_delta"] = "text_delta"


@dataclass(slots=True)
class MessageEvent:
    session_id: str
    message: Message
    event: Literal["message"] = "message"


@dataclass(slots=True)
class ToolCallEvent:
    session_id: str
    tool_call: ToolCall
    event: Literal["tool_call"] = "tool_call"


@dataclass(slots=True)
class ToolResultEvent:
    session_id: str
    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    event: Literal["tool_result"] = "tool_result"


@dataclass(slots=True)
class NativeActionEvent:
    session_id: str
    action: NativeAction
    event: Literal["native_action"] = "native_action"


@dataclass(slots=True)
class ErrorEvent:
    session_id: str
    message: str
    event: Literal["error"] = "error"


AgentEvent = Union[
    TextDeltaEvent,
    MessageEvent,
    ToolCallEvent,
    ToolResultEvent,
    NativeActionEvent,
    ErrorEvent,
]


def event_to_dict(event: AgentEvent) -> dict[str, object]:
    data = asdict(event)
    if isinstance(event, MessageEvent):
        data["message"] = event.message.model_dump()
    elif isinstance(event, ToolCallEvent):
        data["tool_call"] = event.tool_call.model_dump()
    elif isinstance(event, NativeActionEvent):
        data["action"] = event.action.model_dump()
    return data
