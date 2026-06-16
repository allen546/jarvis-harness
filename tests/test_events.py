from jarvis.events import (
    ErrorEvent,
    MessageEvent,
    NativeActionEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    event_to_dict,
)
from jarvis.models.base import Message, NativeAction, ToolCall


def test_text_delta_event_serializes() -> None:
    event = TextDeltaEvent(session_id="s1", content="hello")
    assert event_to_dict(event) == {
        "event": "text_delta",
        "session_id": "s1",
        "content": "hello",
    }


def test_message_event_serializes_message() -> None:
    msg = Message(role="assistant", content="done")
    event = MessageEvent(session_id="s1", message=msg)
    assert event_to_dict(event)["message"] == msg.model_dump()


def test_tool_events_serialize() -> None:
    call = ToolCall(call_id="c1", tool_name="read_file", arguments={"path": "x"})
    call_event = ToolCallEvent(session_id="s1", tool_call=call)
    result_event = ToolResultEvent(session_id="s1", call_id="c1", tool_name="read_file", content="ok", is_error=False)
    assert event_to_dict(call_event)["tool_call"] == call.model_dump()
    assert event_to_dict(result_event)["content"] == "ok"
    assert event_to_dict(result_event)["is_error"] is False


def test_native_action_and_error_events_serialize() -> None:
    action = NativeAction(action_type="reaction", params={"emoji": "thumbs_up"})
    native = NativeActionEvent(session_id="s1", action=action)
    error = ErrorEvent(session_id="s1", message="failed")
    assert event_to_dict(native)["action"] == action.model_dump()
    assert event_to_dict(error) == {
        "event": "error",
        "session_id": "s1",
        "message": "failed",
    }
