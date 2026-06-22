from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Awaitable, Callable
from pathlib import Path

import botpy
from botpy.message import Message as BotpyMessage
from jarvis.models.base import Message

# Monkey-patch botpy Message._Attachments to preserve raw JSON dict
_old_attachments_init = botpy.message.Message._Attachments.__init__
def _new_attachments_init(self, data):
    _old_attachments_init(self, data)
    self._raw_data = data
botpy.message.Message._Attachments.__init__ = _new_attachments_init

logger = logging.getLogger(__name__)

_bot_api: Any = None  # botpy BotAPI, set on ready for send_file
_bot_loop: asyncio.AbstractEventLoop | None = None  # bot thread's event loop

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
        global _bot_api, _bot_loop
        _bot_api = self.api
        _bot_loop = asyncio.get_running_loop()
        logger.info("qq: bot ready as %s", self.robot.name)

    async def on_c2c_message_create(self, message: BotpyMessage) -> None:
        """Handle C2C DM messages (QQ private messages)."""
        from jarvis.models.base import Attachment
        from jarvis.media import to_data_uri

        content = (message.content or "").strip()
        openid = message.author.user_openid
        botpy_attachments = getattr(message, "attachments", None) or []
        if not content and not botpy_attachments:
            return
        logger.info("qq: C2C DM from %s (%d attachments): %s", openid, len(botpy_attachments), content[:80] or "[no text]")
        # Debug: dump full raw message for forwarded/card messages
        if not content or "卡片" in content or "转发" in content:
            import json as _json
            _raw = {}
            for k in dir(message):
                if k.startswith("_"):
                    continue
                try:
                    v = getattr(message, k, None)
                    _json.dumps(v)
                    _raw[k] = v
                except (TypeError, ValueError):
                    pass
            try:
                with open("qq_raw_debug.json", "w") as _f:
                    _json.dump(_raw, _f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.debug("qq: failed to write raw debug file: %s", exc)
        if self._allowed_senders is not None and openid not in self._allowed_senders:
            logger.warning("qq: rejected DM from unauthorized sender %s", openid)
            await message._api.post_c2c_message(
                openid=openid, msg_type=2,
                markdown={"content": "Unauthorized."}, msg_id=message.id,
            )
            return
        # Detect sticker/emoji: QQ encodes these as <faceType=N,...> in content
        is_sticker = content.startswith("<faceType")
        if is_sticker:
            # Strip the face tag from content — it's metadata, not user text
            content = ""

        # Build attachments from botpy message attachments
        attachments: list[Attachment] = []
        asr_text = None
        for att in botpy_attachments:
            att_url = getattr(att, "url", None)
            if att_url:
                mime = getattr(att, "content_type", None) or "application/octet-stream"
                if self._media_supported(mime):
                    # For voice, prefer voice_wav_url if present
                    raw_data = getattr(att, "_raw_data", None) or {}
                    is_wav = False
                    if mime == "voice" and isinstance(raw_data, dict):
                        voice_wav_url = raw_data.get("voice_wav_url")
                        if voice_wav_url:
                            att_url = voice_wav_url
                            is_wav = True
                    
                    data = await self._download(att_url)
                    if data and len(data) <= self._max_download_bytes:
                        filename = getattr(att, "filename", None) or "sticker.jpg"
                        if mime == "voice":
                            try:
                                if is_wav:
                                    from jarvis.media import transcode_wav_to_mp3
                                    mp3_data = transcode_wav_to_mp3(data)
                                else:
                                    from jarvis.media import transcode_amr_to_mp3
                                    mp3_data = transcode_amr_to_mp3(data)
                                url = to_data_uri(mp3_data, "audio/mpeg")
                                filename = Path(filename).with_suffix(".mp3").name
                            except Exception as exc:
                                logger.error("qq: voice transcoding failed: %s", exc)
                                url = to_data_uri(data, mime)
                            
                            # Extract platform ASR text from _raw_data key "asr_refer_text" (if any)
                            if isinstance(raw_data, dict):
                                asr_text = raw_data.get("asr_refer_text")
                        else:
                            url = to_data_uri(data, mime)

                        attachments.append(Attachment(
                            mime_type=mime,
                            url=url,
                            description="sticker" if is_sticker else filename,
                        ))
        if botpy_attachments:
            logger.info("qq: downloaded %d/%d attachments (sticker=%s)", len(attachments), len(botpy_attachments), is_sticker)

        jarvis_msg = Message(role="user", content=content, attachments=attachments, metadata={"asr_text": asr_text})

        try:
            reply_msg = await self._on_message(f"qq_c2c_{openid}", jarvis_msg)
            reply_text = reply_msg.content or ""
        except Exception as exc:
            logger.error("qq: C2C handler error: %s", exc)
            reply_text = f"[error] {exc}"

        try:
            await message._api.post_c2c_message(
                openid=openid, msg_type=2,
                markdown={"content": reply_text}, msg_id=message.id,
            )
        except botpy.errors.ServerError as exc:
            if "markdown" in str(exc).lower():
                logger.warning("qq: invalid markdown content, falling back to plaintext mode: %s", exc)
                await message._api.post_c2c_message(
                    openid=openid, msg_type=0,
                    content=reply_text, msg_id=message.id,
                )
            else:
                raise


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

async def qq_send_c2c_file(openid: str, file_data_b64: str, file_type: int) -> str:
    """Upload a file and send it via botpy's authenticated API.

    botpy's aiohttp session lives on the bot thread's event loop, so we
    dispatch the actual HTTP calls there via run_coroutine_threadsafe.
    """
    if _bot_api is None or _bot_loop is None:
        raise RuntimeError("QQ bot not ready")
    import concurrent.futures
    from botpy.http import Route

    async def _upload_and_send() -> str:
        route = Route("POST", "/v2/users/{openid}/files", openid=openid)
        upload_resp = await _bot_api._http.request(
            route, json={"file_data": file_data_b64, "file_type": file_type}
        )
        await _bot_api.post_c2c_message(
            openid=openid, msg_type=7, media=upload_resp,
        )
        return "ok"

    fut = asyncio.run_coroutine_threadsafe(_upload_and_send(), _bot_loop)
    return await asyncio.wrap_future(fut)
