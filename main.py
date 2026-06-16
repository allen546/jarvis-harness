from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from jarvis.config import SessionConfig, load_session_config
from jarvis.events import event_to_dict
from jarvis.kernel import AgentKernel
from jarvis.models.base import Message, TurnRequest, get_model_class
from jarvis.runtime import AgentContext, AgentSession, RuntimeConfig, SessionState
from jarvis.tools import ToolRegistry, builtin_tools

app = FastAPI(title="Jarvis Gateway")
app.state.sessions = {}


def build_session(config: SessionConfig) -> AgentSession:
    model_cls = get_model_class(config.model.provider)
    ctx = AgentContext(
        config=RuntimeConfig(system_prompt=config.harness.system_prompt),
        session=SessionState(id=config.session_id),
        model=model_cls.from_cfg(config),
        tools=ToolRegistry(builtin_tools(Path.cwd())),
        hooks=[],
    )
    return AgentSession(ctx=ctx, kernel=AgentKernel())


def get_or_create_session(session_id: str) -> AgentSession:
    sessions: dict[str, AgentSession] = app.state.sessions
    if session_id not in sessions:
        config = load_session_config(session_id)
        if config.session_id == "default":
            config.session_id = session_id
        sessions[session_id] = build_session(config)
    return sessions[session_id]


def sse_line(event: dict[str, Any]) -> str:
    name = str(event.pop("event"))
    return f"event: {name}\ndata: {json.dumps(event)}\n\n"


@app.post("/sessions/{session_id}/turns")
async def execute_session_turn(session_id: str, request: TurnRequest) -> StreamingResponse:
    if request.channel.lower() != "sse":
        raise HTTPException(status_code=400, detail="Only sse channel is implemented in the microkernel gateway")
    try:
        session = get_or_create_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {exc}") from exc

    async def stream() -> AsyncGenerator[str, None]:
        async for event in session.submit(Message(role="user", content=request.content, metadata={"channel": request.channel})):
            yield sse_line(event_to_dict(event))

    return StreamingResponse(stream(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
