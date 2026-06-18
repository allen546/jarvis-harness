from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Awaitable, Callable

import botpy
from botpy.message import Message

logger = logging.getLogger(__name__)


class QQBot(botpy.Client):
    """Inbound QQ bot — handles C2C DMs only."""

    def __init__(
        self,
        *args: Any,
        on_message: Callable[[str, str], Awaitable[str]],
        allowed_senders: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._on_message = on_message
        self._allowed_senders = set(allowed_senders) if allowed_senders else None

    async def on_ready(self) -> None:
        logger.info("qq: bot ready as %s", self.robot.name)

    async def on_c2c_message_create(self, message: Message) -> None:
        """Handle C2C DM messages (QQ private messages)."""
        content = (message.content or "").strip()
        if not content:
            return
        openid = message.author.user_openid
        logger.info("qq: C2C DM from %s: %s", openid, content[:80])
        if self._allowed_senders is not None and openid not in self._allowed_senders:
            logger.warning("qq: rejected DM from unauthorized sender %s", openid)
            await message._api.post_c2c_message(
                openid=openid, msg_type=2,
                markdown={"content": "Unauthorized."}, msg_id=message.id,
            )
            return
        session_id = f"qq_c2c_{openid}"
        try:
            reply = await self._on_message(session_id, content)
            logger.info("qq: C2C reply to %s: %s", openid, reply[:120])
        except Exception as exc:
            logger.error("qq: C2C handler error: %s", exc)
            reply = f"[error] {exc}"
        await message._api.post_c2c_message(
            openid=openid, msg_type=2,
            markdown={"content": reply}, msg_id=message.id,
        )


class QQChannel:
    """Runs the QQ bot as an asyncio task."""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        intents: list[str] | None = None,
        on_message: Callable[[str, str], Awaitable[str]] | None = None,
        main_loop: asyncio.AbstractEventLoop | None = None,
        allowed_senders: list[str] | None = None,
    ) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.intents = intents or ["public_messages"]
        self._on_message = on_message
        self._main_loop = main_loop
        self._allowed_senders = allowed_senders
        self._bot: QQBot | None = None
        self._thread: threading.Thread | None = None

    def _build_intents(self) -> Any:
        """Convert string intent names to a botpy.Intents object."""
        from botpy import Intents
        return Intents(**{name: True for name in self.intents})

    async def run(self) -> None:
        """Start the bot in a dedicated thread with its own event loop.

        Messages from the bot thread are dispatched to the main event loop
        via ``run_coroutine_threadsafe`` so that shared ``SessionManager``
        locks/MCP contexts always execute on the loop that owns them.
        """
        logger.info("qq: starting bot (app_id=%s)", self.app_id)
        intents = self._build_intents()
        main_loop = self._main_loop or asyncio.get_running_loop()

        # This handler runs on the *bot thread's* event loop.
        # It bridges to the main loop where SessionManager lives.
        async def _bridge_handler(session_id: str, text: str) -> str:
            import concurrent.futures
            coro = self._on_message(session_id, text)
            try:
                fut = asyncio.run_coroutine_threadsafe(coro, main_loop)
            except RuntimeError:
                coro.close()
                return "[error] main loop shutting down"
            try:
                return await asyncio.wrap_future(fut, loop=asyncio.get_running_loop())
            except asyncio.CancelledError:
                fut.cancel()
                raise
            except concurrent.futures.CancelledError:
                fut.cancel()
                return "[cancelled]"
            except Exception as exc:
                fut.cancel()
                logger.error("qq: handler error: %s", exc)
                return f"[error] {exc}"

        def _run_bot() -> None:
            import asyncio as _aio
            loop = _aio.new_event_loop()
            _aio.set_event_loop(loop)
            try:
                bot = QQBot(
                    intents=intents,
                    on_message=_bridge_handler,
                    allowed_senders=self._allowed_senders,
                )
                self._bot = bot
                bot.run(appid=self.app_id, secret=self.app_secret)
            except Exception as exc:
                logger.error("qq: bot crashed: %s", exc)
            finally:
                loop.close()

        self._thread = threading.Thread(target=_run_bot, daemon=True, name="qq-bot")
        self._thread.start()
        logger.info("qq: bot thread started")
        # Keep this coroutine alive so the task isn't "done"
        try:
            while self._thread.is_alive():
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("qq: channel cancelled")
