from __future__ import annotations

import asyncio
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from jarvis.config import SessionConfig
from jarvis.events import AgentEvent
from jarvis.hooks import BudgetGuardHook, ContextCompressionHook, JSONLHistoryHook, ToolApprovalHook, TurnHook
from jarvis.models.base import BaseModelClient, Message, ToolCall, get_model_class
from jarvis.tools import ToolRegistry

current_context: ContextVar[AgentContext | None] = ContextVar("current_context", default=None)


@dataclass(slots=True)
class RuntimeConfig:
    system_prompt: str | None = None
    max_consecutive_tools: int = 5
    require_tool_approval: bool = False
    allowed_skills: list[str] = field(default_factory=list)
    stream: bool = True


@dataclass(slots=True)
class SessionState:
    id: str
    history: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentContext:
    config: RuntimeConfig
    session: SessionState
    model: BaseModelClient
    tools: ToolRegistry
    hooks: list[TurnHook] = field(default_factory=list)
    emit_event: Callable[[AgentEvent], None] | None = None
    mcp_manager: Any | None = field(default=None, compare=False)
    approval_handler: Callable[[ToolCall], bool | Awaitable[bool]] | None = field(default=None, compare=False)


class AgentSession:
    def __init__(self, ctx: AgentContext, kernel: object) -> None:
        self.ctx = ctx
        self.kernel = kernel
        self._lock = asyncio.Lock()
        self._mcp_initialized = False
        self._mcp_tool_names: list[str] = []

    async def submit(self, message: Message) -> AsyncIterator[AgentEvent]:
        async with self._lock:
            if not self._mcp_initialized:
                from jarvis.skills import SkillManager, SkillInstructionsHook
                skill_mgr = SkillManager()
                skills = await skill_mgr.load_allowed_skills(self.ctx)
                if skills:
                    self.ctx.hooks.append(SkillInstructionsHook(skills))

                from jarvis.mcp import McpClientManager
                manager = McpClientManager()
                mcp_tools = await manager.initialize()
                for t in mcp_tools:
                    self.ctx.tools.register(t)
                    self._mcp_tool_names.append(t.name)
                self.ctx.mcp_manager = manager
                self._mcp_initialized = True
            token = current_context.set(self.ctx)
            event_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
            
            old_emitter = self.ctx.emit_event
            def queue_emitter(ev: AgentEvent) -> None:
                event_queue.put_nowait(ev)
                if old_emitter:
                    old_emitter(ev)
            self.ctx.emit_event = queue_emitter

            async def run() -> None:
                try:
                    async for event in self.kernel.run_turn(self.ctx, message):  # type: ignore[attr-defined]
                        if self.ctx.emit_event is not None:
                            self.ctx.emit_event(event)
                except Exception as exc:
                    from jarvis.events import ErrorEvent
                    err_event = ErrorEvent(session_id=self.ctx.session.id, message=str(exc))
                    if self.ctx.emit_event is not None:
                        self.ctx.emit_event(err_event)
                finally:
                    event_queue.put_nowait(None)

            task = asyncio.create_task(run())
            try:
                while True:
                     event = await event_queue.get()
                     if event is None:
                         break
                     yield event
            finally:
                self.ctx.emit_event = old_emitter
                try:
                    current_context.reset(token)
                except ValueError:
                    pass
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                else:
                    await task

    async def close(self) -> None:
        async with self._lock:
            if self.ctx.mcp_manager:
                await self.ctx.mcp_manager.close()
                self.ctx.mcp_manager = None
                self._mcp_initialized = False
            for name in self._mcp_tool_names:
                self.ctx.tools._tools.pop(name, None)
            self._mcp_tool_names.clear()


def _default_hooks() -> list[TurnHook]:
    hooks: list[TurnHook] = [
        JSONLHistoryHook(),
        ContextCompressionHook(),
        BudgetGuardHook(),
        ToolApprovalHook()
    ]
    embedding_url = os.environ.get("EMBEDDING_URL", "")
    if embedding_url:
        from jarvis.memory_store import SemanticMemoryHook
        hooks.append(SemanticMemoryHook(
            storage_dir="storage",
            embedding_url=embedding_url,
        ))
    return hooks


def context_from_config(config: SessionConfig, tools: ToolRegistry, hooks: list[TurnHook] | None = None) -> AgentContext:
    provider = config.model.provider.lower()
    model_cls = get_model_class(provider)
    return AgentContext(
        config=RuntimeConfig(
            system_prompt=config.harness.system_prompt,
            max_consecutive_tools=config.harness.max_consecutive_tools,
            require_tool_approval=config.harness.require_tool_approval,
            allowed_skills=config.harness.allowed_skills,
            stream=config.harness.stream,
        ),
        session=SessionState(id=config.session_id),
        model=model_cls.from_cfg(config),
        tools=tools,
        hooks=hooks if hooks is not None else _default_hooks(),
    )
