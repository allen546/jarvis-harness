from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from jarvis.config import SessionConfig
from jarvis.events import AgentEvent
from jarvis.hooks import TurnHook
from jarvis.models.base import BaseModelClient, Message, get_model_class
from jarvis.tools import ToolRegistry


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
            async for event in self.kernel.run_turn(self.ctx, message):  # type: ignore[attr-defined]
                if self.ctx.emit_event is not None:
                    self.ctx.emit_event(event)
                yield event


def context_from_config(config: SessionConfig, tools: ToolRegistry, hooks: list[TurnHook] | None = None) -> AgentContext:
    provider = config.model.provider.lower()
    model_cls = get_model_class(provider)
    return AgentContext(
        config=RuntimeConfig(system_prompt=config.harness.system_prompt),
        session=SessionState(id=config.session_id),
        model=model_cls.from_cfg(config),
        tools=tools,
        hooks=hooks or [],
    )
