from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Awaitable, Callable

import botpy
from botpy.message import Message as BotpyMessage
from jarvis.models.base import Message

logger = logging.getLogger(__name__)


class QQBot(botpy.Client):
    """Inbound QQ bot — handles C2C DMs only."""

    def __init__(
        self,
        *args: Any,
        on_message: Callable[[str, "Message"], Awaitable["Message"]],
        allowed_senders: list[str] | None = None,
        supported_media: list[str] | None = None,
        max_download_size_mb: int = 10,
        proxy_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._on_message = on_message
        self._allowed_senders = set(allowed_senders) if allowed_senders else None
        self._supported_media = supported_media or []
        self._max_download_bytes = max_download_size_mb * 1024 * 1024
        self._proxy_env = proxy_env or {}

    async def _download(self, url: str) -> bytes | None:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=60, proxy=self._proxy_env.get("https") or self._proxy_env.get("http") or None) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp.content
        except Exception as exc:
            logger.warning("qq: failed to download %s: %s", url, exc)
        return None

    def _media_supported(self, mime: str) -> bool:
        if not self._supported_media:
            return True
        for prefix in self._supported_media:
            if mime.startswith(prefix):
                return True
        return False

    async def on_ready(self) -> None:
        logger.info("qq: bot ready as %s", self.robot.name)

    async def on_c2c_message_create(self, message: BotpyMessage) -> None:
        """Handle C2C DM messages (QQ private messages)."""
        from jarvis.models.base import Attachment
        from jarvis.media import to_data_uri

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

        # Build attachments from botpy message attachments
        attachments: list[Attachment] = []
        botpy_attachments = getattr(message, "attachments", None) or []
        for att in botpy_attachments:
            att_url = getattr(att, "url", None)
            if att_url:
                mime = getattr(att, "content_type", None) or "application/octet-stream"
                if self._media_supported(mime):
                    data = await self._download(att_url)
                    if data and len(data) <= self._max_download_bytes:
                        attachments.append(Attachment(
                            mime_type=mime,
                            url=to_data_uri(data, mime),
                            description=getattr(att, "filename", None),
                        ))

        jarvis_msg = Message(role="user", content=content, attachments=attachments)

        try:
            reply_msg = await self._on_message(f"qq_c2c_{openid}", jarvis_msg)
            reply_text = reply_msg.content or ""
        except Exception as exc:
            logger.error("qq: C2C handler error: %s", exc)
            reply_text = f"[error] {exc}"

        await message._api.post_c2c_message(
            openid=openid, msg_type=2,
            markdown={"content": reply_text}, msg_id=message.id,
        )


class QQChannel:
    """Runs the QQ bot as an asyncio task."""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        intents: list[str] | None = None,
        on_message: Callable[[str, "Message"], Awaitable["Message"]] | None = None,
        main_loop: asyncio.AbstractEventLoop | None = None,
        allowed_senders: list[str] | None = None,
        supported_media: list[str] | None = None,
        max_download_size_mb: int = 10,
        proxy_env: dict[str, str] | None = None,
    ) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.intents = intents or ["public_messages"]
        self._on_message = on_message
        self._main_loop = main_loop
        self._allowed_senders = allowed_senders
        self._supported_media = supported_media or []
        self._max_download_size_mb = max_download_size_mb
        self._proxy_env = proxy_env or {}
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
        async def _bridge_handler(session_id: str, msg: Message) -> Message:
            import concurrent.futures
            coro = self._on_message(session_id, msg)
            try:
                fut = asyncio.run_coroutine_threadsafe(coro, self._main_loop)
            except RuntimeError:
                coro.close()
                return Message(role="assistant", content="[error] main loop shutting down")
            try:
                return await asyncio.wrap_future(fut, loop=asyncio.get_running_loop())
            except asyncio.CancelledError:
                fut.cancel()
                raise
            except concurrent.futures.CancelledError:
                fut.cancel()
                return Message(role="assistant", content="[cancelled]")
            except Exception as exc:
                fut.cancel()
                logger.error("qq: handler error: %s", exc)
                return Message(role="assistant", content=f"[error] {exc}")

        def _run_bot() -> None:
            import asyncio as _aio
            loop = _aio.new_event_loop()
            _aio.set_event_loop(loop)
            try:
                bot = QQBot(
                    intents=self._build_intents(),
                    on_message=_bridge_handler,
                    allowed_senders=self._allowed_senders,
                    supported_media=self._supported_media,
                    max_download_size_mb=self._max_download_size_mb,
                    proxy_env=self._proxy_env,
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
