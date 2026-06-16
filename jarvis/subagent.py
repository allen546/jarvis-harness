from __future__ import annotations

import uuid
from typing import Any

from jarvis.events import MessageEvent
from jarvis.kernel import AgentKernel
from jarvis.models.base import Message
from jarvis.runtime import AgentContext, AgentSession, RuntimeConfig, SessionState
from jarvis.tools import ToolRegistry

active_subagents: dict[str, AgentSession] = {}


async def spawn_subagent_tool(ctx: AgentContext, args: dict[str, Any]) -> dict[str, Any]:
    task_name = args.get("task_name", "")
    prompt_val = args.get("prompt") or args.get("prompts")
    if isinstance(prompt_val, list):
        prompt_str = "\n".join(prompt_val)
    else:
        prompt_str = str(prompt_val or "")

    sub_session_id = f"sub_{uuid.uuid4()}"
    
    # Optional system override
    system_prompt = args.get("system_override") if "system_override" in args else ctx.config.system_prompt
    sub_config = RuntimeConfig(system_prompt=system_prompt)
    
    # Filter out subagent tools
    filtered_tools = [
        tool for tool in ctx.tools._tools.values()
        if tool.name not in {"spawn_subagent", "send_subagent_message", "close_subagent"}
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
    active_subagents[sub_session_id] = sub_session
    
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
    sub_session = active_subagents.get(sub_session_id)
    if not sub_session:
        raise ValueError(f"No active subagent found with ID: {sub_session_id}")
        
    message_text = args.get("message") or args.get("prompt") or args.get("prompts")
    if isinstance(message_text, list):
        message_str = "\n".join(message_text)
    else:
        message_str = str(message_text or "")
        
    msg = Message(role="user", content=message_str)
    response_text = ""
    async for event in sub_session.submit(msg):
        if isinstance(event, MessageEvent):
            response_text = event.message.content
            
    return {"response": response_text}


async def close_subagent_tool(ctx: AgentContext, args: dict[str, Any]) -> dict[str, Any]:
    sub_session_id = args.get("sub_session_id", "")
    if sub_session_id in active_subagents:
        del active_subagents[sub_session_id]
        
    active_list = ctx.session.metadata.get("active_subagents", [])
    if sub_session_id in active_list:
        active_list.remove(sub_session_id)
        
    return {"message": f"Subagent {sub_session_id} closed."}
