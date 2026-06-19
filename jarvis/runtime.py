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
    # safety hooks
    max_repeated_tool_calls: int = 3
    repeated_content_threshold: float = 0.8
    repeated_content_window: int = 3


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
    proxy_env: dict[str, str] = field(default_factory=dict)


class AgentSession:
    def __init__(self, ctx: AgentContext, kernel: object, proxy_env: dict[str, str] | None = None) -> None:
        self.ctx = ctx
        self.kernel = kernel
        self._lock = asyncio.Lock()
        self._mcp_initialized = False
        self._mcp_tool_names: list[str] = []
        self._skill_tool_names: list[str] = []
        if proxy_env:
            self.ctx.proxy_env = proxy_env

    async def submit(self, message: Message) -> AsyncIterator[AgentEvent]:
        async with self._lock:
            if not self._mcp_initialized:
                from jarvis.skills import SkillManager, SkillInstructionsHook
                skill_mgr = SkillManager()
                skills = await skill_mgr.load_allowed_skills(self.ctx)
                if skills:
                    self.ctx.hooks.append(SkillInstructionsHook(skills))
                    for s in skills:
                        for t_name, t_cfg in s.tools.items():
                            if isinstance(t_cfg, dict):
                                self._skill_tool_names.append(t_name)

                from jarvis.mcp import McpClientManager
                manager = McpClientManager(proxy_env=self.ctx.proxy_env)
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
            for name in self._skill_tool_names:
                self.ctx.tools._tools.pop(name, None)
            self._skill_tool_names.clear()
            
            from jarvis.skills import SkillInstructionsHook
            self.ctx.hooks = [h for h in self.ctx.hooks if not isinstance(h, SkillInstructionsHook)]


def _default_hooks(config: SessionConfig | None = None) -> list[TurnHook]:
    from jarvis.hooks import RepeatedLoopHook, RepeatedContentHook

    # Read safety params from config, fall back to defaults
    max_repeats = 3
    content_threshold = 0.8
    content_window = 3
    if config is not None:
        max_repeats = config.harness.max_repeated_tool_calls
        content_threshold = config.harness.repeated_content_threshold
        content_window = config.harness.repeated_content_window

    hooks: list[TurnHook] = [
        JSONLHistoryHook(),
        ContextCompressionHook(),
        BudgetGuardHook(),
        ToolApprovalHook(),
        RepeatedLoopHook(max_repeats=max_repeats),
        RepeatedContentHook(threshold=content_threshold, window=content_window),
    ]

    # Embedding: config URL > env var > disabled
    embedding_url = ""
    embedding_enabled = False
    embedding_dims = 256
    if config is not None:
        embedding_url = config.harness.embedding.url
        embedding_enabled = config.harness.embedding.enabled
        embedding_dims = config.harness.embedding.dimensions
    if not embedding_url:
        embedding_url = os.environ.get("EMBEDDING_URL", "")
    if embedding_url or embedding_enabled:
        from jarvis.memory_store import SemanticMemoryHook
        hooks.append(SemanticMemoryHook(
            storage_dir="storage",
            embedding_url=embedding_url or None,
            embedding_dimensions=embedding_dims,
        ))
    return hooks


def context_from_config(config: SessionConfig, tools: ToolRegistry, hooks: list[TurnHook] | None = None) -> AgentContext:
    provider = config.model.provider.lower()
    model_cls = get_model_class(provider)
    h = config.harness
    return AgentContext(
        config=RuntimeConfig(
            system_prompt=h.system_prompt,
            max_consecutive_tools=h.max_consecutive_tools,
            require_tool_approval=h.require_tool_approval,
            allowed_skills=h.allowed_skills,
            stream=h.stream,
            max_repeated_tool_calls=h.max_repeated_tool_calls,
            repeated_content_threshold=h.repeated_content_threshold,
            repeated_content_window=h.repeated_content_window,
        ),
        session=SessionState(id=config.session_id),
        model=model_cls.from_cfg(config),
        tools=tools,
        hooks=hooks if hooks is not None else _default_hooks(config),
    )
