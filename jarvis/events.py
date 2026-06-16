from dataclasses import dataclass
from typing import Any
from jarvis.models.base import Message, NativeAction, ToolCall

@dataclass(slots=True)
class TextDeltaEvent:
    session_id: str
    content: str

@dataclass(slots=True)
class MessageEvent:
    session_id: str
    message: Message

@dataclass(slots=True)
class ToolCallEvent:
    session_id: str
    tool_call: ToolCall

@dataclass(slots=True)
class ToolResultEvent:
    session_id: str
    call_id: str
    tool_name: str
    content: str
    is_error: bool

@dataclass(slots=True)
class NativeActionEvent:
    session_id: str
    action: NativeAction

@dataclass(slots=True)
class ErrorEvent:
    session_id: str
    message: str

_EVENT_TYPE_MAP = {
    TextDeltaEvent: "text_delta",
    MessageEvent: "message",
    ToolCallEvent: "tool_call",
    ToolResultEvent: "tool_result",
    NativeActionEvent: "native_action",
    ErrorEvent: "error",
}

def event_to_dict(event: Any) -> dict[str, Any]:
    event_type = _EVENT_TYPE_MAP.get(type(event))
    if not event_type:
        raise ValueError(f"Unknown event type: {type(event)}")
    
    result: dict[str, Any] = {"event": event_type}
    
    if isinstance(event, TextDeltaEvent):
        result["session_id"] = event.session_id
        result["content"] = event.content
    elif isinstance(event, MessageEvent):
        result["session_id"] = event.session_id
        result["message"] = event.message.model_dump()
    elif isinstance(event, ToolCallEvent):
        result["session_id"] = event.session_id
        result["tool_call"] = event.tool_call.model_dump()
    elif isinstance(event, ToolResultEvent):
        result["session_id"] = event.session_id
        result["call_id"] = event.call_id
        result["tool_name"] = event.tool_name
        result["content"] = event.content
        result["is_error"] = event.is_error
    elif isinstance(event, NativeActionEvent):
        result["session_id"] = event.session_id
        result["action"] = event.action.model_dump()
    elif isinstance(event, ErrorEvent):
        result["session_id"] = event.session_id
        result["message"] = event.message
        
    return result
