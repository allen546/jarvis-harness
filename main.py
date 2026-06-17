from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from jarvis.config import SessionConfig, load_session_config
from jarvis.events import event_to_dict
from jarvis.kernel import AgentKernel
from jarvis.models.base import Message
from jarvis.runtime import AgentSession, context_from_config
from jarvis.tools import ToolRegistry, builtin_tools


app = FastAPI(title="Jarvis Gateway")
app.state.sessions: dict[str, AgentSession] = {}


class TurnRequest(BaseModel):
    content: str


class TurnResponse(BaseModel):
    session_id: str
    content: str
    tool_calls: list[dict] = []


def build_session(config: SessionConfig) -> AgentSession:
    ctx = context_from_config(config, tools=ToolRegistry(builtin_tools(Path.cwd())))
    return AgentSession(ctx=ctx, kernel=AgentKernel())


def get_or_create_session(session_id: str) -> AgentSession:
    sessions = app.state.sessions
    if session_id not in sessions:
        config = load_session_config(session_id)
        if config.session_id == "default":
            config.session_id = session_id
        sessions[session_id] = build_session(config)
    return sessions[session_id]


@app.post("/sessions/{session_id}/turns", response_model=TurnResponse)
async def execute_session_turn(session_id: str, request: TurnRequest) -> TurnResponse:
    try:
        session = get_or_create_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {exc}") from exc

    content = ""
    tool_calls = []
    async for event in session.submit(Message(role="user", content=request.content)):
        data = event_to_dict(event)
        if data.get("event") == "message":
            content = data["message"]["content"]
        elif data.get("event") == "tool_call":
            tool_calls.append(data["tool_call"])

    return TurnResponse(session_id=session_id, content=content, tool_calls=tool_calls)


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
