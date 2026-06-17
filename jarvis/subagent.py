from __future__ import annotations

import uuid
from typing import Any

from jarvis.events import MessageEvent
from jarvis.kernel import AgentKernel
from jarvis.models.base import Message
from jarvis.runtime import AgentContext, AgentSession, RuntimeConfig, SessionState
from jarvis.tools import ToolRegistry


async def spawn_subagent_tool(ctx: AgentContext, args: dict[str, Any]) -> dict[str, Any]:
    task_name = args.get("task_name", "")
    prompt_val = args.get("prompt") or args.get("prompts")
    if isinstance(prompt_val, list):
        prompt_str = "\n".join(prompt_val)
    else:
        prompt_str = str(prompt_val or "")

    if not prompt_str.strip():
        raise ValueError("Prompt cannot be empty or whitespace-only.")

    sub_session_id = f"sub_{uuid.uuid4()}"
    
    # Optional system override or default task prompt
    if "system_override" in args and args["system_override"] is not None:
        system_prompt = args["system_override"]
    else:
        system_prompt = f"You are a subagent assistant focusing on task: {task_name}. Help solve this specific subtask concisely."
    sub_config = RuntimeConfig(system_prompt=system_prompt)
    
    # Filter out subagent tools
    filtered_tools = [
        tool for tool in ctx.tools._tools.values()
        if tool.name not in {"task", "message", "close"}
    ]
    sub_tools = ToolRegistry(filtered_tools)
    
    sub_session_state = SessionState(id=sub_session_id, metadata={"task_name": task_name})
    
    sub_ctx = AgentContext(
        config=sub_config,
        session=sub_session_state,
        model=ctx.model,
        tools=sub_tools,
        hooks=list(ctx.hooks),
        emit_event=ctx.emit_event,
    )
    
    sub_session = AgentSession(ctx=sub_ctx, kernel=AgentKernel())
    
    if "subagent_sessions" not in ctx.session.metadata:
        ctx.session.metadata["subagent_sessions"] = {}
    ctx.session.metadata["subagent_sessions"][sub_session_id] = sub_session
    
    if "active_subagents" not in ctx.session.metadata:
        ctx.session.metadata["active_subagents"] = []
    ctx.session.metadata["active_subagents"].append(sub_session_id)
    
    msg = Message(role="user", content=prompt_str)
    response_text = ""
    async for event in sub_session.submit(msg):
        if isinstance(event, MessageEvent):
            response_text = event.message.content
            
    return {"sub_session_id": sub_session_id, "response": response_text}


async def send_subagent_message_tool(ctx: AgentContext, args: dict[str, Any]) -> dict[str, Any]:
    sub_session_id = args.get("sub_session_id", "")
    sub_session = ctx.session.metadata.get("subagent_sessions", {}).get(sub_session_id)
    if not sub_session:
        raise ValueError(f"No active subagent found with ID: {sub_session_id} in this session.")
        
    message_text = args.get("message") or args.get("prompt") or args.get("prompts")
    if isinstance(message_text, list):
        message_str = "\n".join(message_text)
    else:
        message_str = str(message_text or "")
        
    if not message_str.strip():
        raise ValueError("Message cannot be empty or whitespace-only.")
        
    msg = Message(role="user", content=message_str)
    response_text = ""
    async for event in sub_session.submit(msg):
        if isinstance(event, MessageEvent):
            response_text = event.message.content
            
    return {"response": response_text}


async def close_subagent_tool(ctx: AgentContext, args: dict[str, Any]) -> dict[str, Any]:
    sub_session_id = args.get("sub_session_id", "")
    sessions = ctx.session.metadata.get("subagent_sessions", {})
    if sub_session_id in sessions:
        del sessions[sub_session_id]
        
    active_list = ctx.session.metadata.get("active_subagents", [])
    if sub_session_id in active_list:
        active_list.remove(sub_session_id)
        
    return {"message": f"Subagent {sub_session_id} closed."}
