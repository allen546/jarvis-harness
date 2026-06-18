from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator

from jarvis.config import SessionConfig, load_session_config
from jarvis.events import AgentEvent, ErrorEvent, MessageEvent
from jarvis.kernel import AgentKernel
from jarvis.models.base import Message
from jarvis.runtime import AgentSession, context_from_config
from jarvis.tools import ToolRegistry, builtin_tools

logger = logging.getLogger(__name__)


class SessionManager:
    """Owns all AgentSession instances. Shared by gateway, channels, and cron."""

    def __init__(self, workspace: str = ".", proxy_env: dict[str, str] | None = None) -> None:
        self.sessions: dict[str, AgentSession] = {}
        self.workspace = Path(workspace)
        self._proxy_env = proxy_env or {}

    def get_or_create(self, session_id: str) -> AgentSession:
        if session_id not in self.sessions:
            config = load_session_config(session_id)
            if config.session_id == "default":
                config.session_id = session_id
            tools = ToolRegistry(builtin_tools(self.workspace))
            ctx = context_from_config(config, tools)
            self.sessions[session_id] = AgentSession(ctx=ctx, kernel=AgentKernel(), proxy_env=self._proxy_env)
            logger.info("session: created %s (model=%s/%s)",
                        session_id, config.model.provider, config.model.model_name)
        return self.sessions[session_id]

    async def submit(self, session_id: str, message: Message) -> AsyncIterator[AgentEvent]:
        session = self.get_or_create(session_id)
        async for event in session.submit(message):
            yield event

    async def submit_and_collect(self, session_id: str, text: str) -> str:
        """Submit a message and return the final assistant text."""
        content = ""
        async for event in self.submit(session_id, Message(role="user", content=text)):
            if isinstance(event, ErrorEvent):
                raise RuntimeError(event.message)
            if isinstance(event, MessageEvent):
                content = event.message.content
        return content

    async def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session:
            await session.close()

    async def close_all(self) -> None:
        for session_id in list(self.sessions):
            await self.close(session_id)
