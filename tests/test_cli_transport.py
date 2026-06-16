import pytest

from jarvis.events import MessageEvent, NativeActionEvent
from jarvis.models.base import Message, NativeAction
from jarvis.transports.cli import render_cli_event


def test_render_cli_message_event(capsys: pytest.CaptureFixture[str]) -> None:
    render_cli_event(MessageEvent(session_id="s1", message=Message(role="assistant", content="hello")))
    assert capsys.readouterr().out == "hello\n"


def test_render_cli_native_action_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    action = NativeAction(action_type="reaction", params={"emoji": "thumbs_up"})
    render_cli_event(NativeActionEvent(session_id="s1", action=action))
    assert "native_action reaction" in capsys.readouterr().out
