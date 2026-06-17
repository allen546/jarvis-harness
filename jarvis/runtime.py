from __future__ import annotations

import asyncio
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from jarvis.config import SessionConfig
from jarvis.events import AgentEvent
from jarvis.hooks import ContextCompressionHook, JSONLHistoryHook, TurnHook
from jarvis.models.base import BaseModelClient, Message, get_model_class
from jarvis.tools import ToolRegistry

current_context: ContextVar[AgentContext | None] = ContextVar("current_context", default=None)


@dataclass(slots=True)
class RuntimeConfig:
    system_prompt: str | None = None


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


class AgentSession:
    def __init__(self, ctx: AgentContext, kernel: object) -> None:
        self.ctx = ctx
        self.kernel = kernel
        self._lock = asyncio.Lock()

    async def submit(self, message: Message) -> AsyncIterator[AgentEvent]:
        async with self._lock:
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


def _default_hooks() -> list[TurnHook]:
    hooks: list[TurnHook] = [JSONLHistoryHook(), ContextCompressionHook()]
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
        config=RuntimeConfig(system_prompt=config.harness.system_prompt),
        session=SessionState(id=config.session_id),
        model=model_cls.from_cfg(config),
        tools=tools,
        hooks=hooks if hooks is not None else _default_hooks(),
    )
