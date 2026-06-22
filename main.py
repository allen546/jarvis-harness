from __future__ import annotations

import asyncio
import logging
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from jarvis.config import load_config
from jarvis.cron import CronScheduler, tasks_from_config
from jarvis.events import ErrorEvent, event_to_dict
from jarvis.models.base import Message
from jarvis.sessions import SessionManager
from jarvis.transports.qq import QQChannel

logger = logging.getLogger(__name__)

app = FastAPI(title="Jarvis Gateway")
_manager: SessionManager | None = None


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

class TurnRequest(BaseModel):
    content: str


class TurnResponse(BaseModel):
    session_id: str
    content: str
    tool_calls: list[dict] = []


@app.post("/sessions/{session_id}/turns", response_model=TurnResponse)
async def execute_session_turn(session_id: str, request: TurnRequest) -> TurnResponse:
    assert _manager is not None, "Gateway not initialised"
    try:
        content = ""
        tool_calls = []
        async for event in _manager.submit(session_id, Message(role="user", content=request.content)):
            if isinstance(event, ErrorEvent):
                raise HTTPException(status_code=500, detail=event.message)
            data = event_to_dict(event)
            if data.get("event") == "message":
                content = data["message"]["content"]
            elif data.get("event") == "tool_call":
                tool_calls.append(data["tool_call"])
        return TurnResponse(session_id=session_id, content=content, tool_calls=tool_calls)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Uncaught exception in execute_session_turn:")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Gateway runner (daemon thread)
# ---------------------------------------------------------------------------

def _run_gateway(host: str, port: int) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    global _manager

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root.addHandler(handler)
    root.setLevel(logging.INFO)

    config = load_config()
    proxy_env = {}
    p = config.proxy
    if p.http_proxy:
        proxy_env["HTTP_PROXY"] = p.http_proxy
        proxy_env["http_proxy"] = p.http_proxy
    if p.https_proxy:
        proxy_env["HTTPS_PROXY"] = p.https_proxy
        proxy_env["https_proxy"] = p.https_proxy
    # NOTE: all_proxy / ALL_PROXY deliberately omitted — no socksio installed.
    if p.no_proxy:
        proxy_env["NO_PROXY"] = p.no_proxy
        proxy_env["no_proxy"] = p.no_proxy
    _manager = SessionManager(proxy_env=proxy_env)

    # --- gateway daemon thread ---
    gw_thread = threading.Thread(
        target=_run_gateway,
        args=(config.gateway.host, config.gateway.port),
        daemon=True,
    )
    gw_thread.start()
    logger.info("gateway: listening on %s:%d", config.gateway.host, config.gateway.port)

    # --- cron scheduler ---
    cron = CronScheduler()
    for task in tasks_from_config(config.cron, manager=_manager):
        cron.register(task)
    cron_task = asyncio.create_task(cron.run())

    # --- QQ channel ---
    tasks: list[asyncio.Task[None]] = [cron_task]
    qq_channel: QQChannel | None = None
    if config.channels.qq.enabled:
        async def _qq_handler(session_id: str, message: Message) -> Message:
            # Check if there is a voice attachment
            has_voice = any(att.mime_type == "voice" for att in message.attachments)
            if not has_voice:
                return await _manager.submit_and_collect(session_id, message)

            session = _manager.get_or_create(session_id)

            # Tier 1: Native Audio
            logger.info("qq_handler: Tier 1: attempting native audio turn")
            try:
                return await _manager.submit_and_collect(session_id, message)
            except Exception as exc:
                logger.warning("qq_handler: Tier 1 failed: %s", exc)
                # Clean up turn history
                if session.ctx.session.history and session.ctx.session.history[-1] is message:
                    session.ctx.session.history.pop()

            # Tier 2: Platform ASR
            asr_text = message.metadata.get("asr_text") if message.metadata else None
            if asr_text:
                logger.info("qq_handler: Tier 2: retrying with platform ASR: %r", asr_text)
                asr_message = Message(role="user", content=asr_text, metadata=message.metadata)
                try:
                    return await _manager.submit_and_collect(session_id, asr_message)
                except Exception as exc:
                    logger.warning("qq_handler: Tier 2 failed: %s", exc)
                    # Clean up turn history
                    if session.ctx.session.history and session.ctx.session.history[-1] is asr_message:
                        session.ctx.session.history.pop()
            else:
                logger.info("qq_handler: Tier 2 skipped (no platform ASR text)")

            # Tier 3: Local Whisper
            logger.info("qq_handler: Tier 3: attempting local Whisper transcription")
            voice_att = next(att for att in message.attachments if att.mime_type == "voice")
            mp3_bytes = None
            if voice_att.url and "," in voice_att.url:
                try:
                    import base64
                    b64_data = voice_att.url.split(",", 1)[1]
                    mp3_bytes = base64.b64decode(b64_data)
                except Exception as exc:
                    logger.error("qq_handler: failed to decode voice attachment url: %s", exc)

            if mp3_bytes:
                try:
                    from jarvis.media import transcribe_locally
                    local_text = transcribe_locally(mp3_bytes)
                    logger.info("qq_handler: local transcription result: %r", local_text)
                    if local_text:
                        whisper_message = Message(role="user", content=local_text, metadata=message.metadata)
                        try:
                            return await _manager.submit_and_collect(session_id, whisper_message)
                        except Exception as exc:
                            logger.error("qq_handler: Tier 3 failed: %s", exc)
                            # Clean up turn history
                            if session.ctx.session.history and session.ctx.session.history[-1] is whisper_message:
                                session.ctx.session.history.pop()
                except Exception as exc:
                    logger.error("qq_handler: Tier 3 local Whisper failed: %s", exc)

            raise RuntimeError("QQ voice message turn failed across all tiers.")

        qq_channel = QQChannel(
            app_id=config.channels.qq.app_id,
            app_secret=config.channels.qq.app_secret,
            intents=config.channels.qq.intents,
            on_message=_qq_handler,
            main_loop=asyncio.get_running_loop(),
            allowed_senders=config.channels.qq.allowed_senders,
            supported_media=config.channels.qq.supported_media,
            max_download_size_mb=config.channels.qq.max_download_size_mb,
            proxy_env=proxy_env,
        )
        qq_task = asyncio.create_task(qq_channel.run())
        tasks.append(qq_task)
        logger.info("qq: channel enabled (app_id=%s)", config.channels.qq.app_id)

    # --- heartbeat ---
    heartbeat_mgr = None
    if config.harness.heartbeat.enabled:
        from jarvis.heartbeat import HeartbeatManager
        async def _heartbeat_submit(session_id: str, message: Message) -> Message:
            return await _manager.submit_and_collect(session_id, message)
        heartbeat_mgr = HeartbeatManager(
            workspace=config.harness.heartbeat.workspace,
            interval_secs=config.harness.heartbeat.interval_secs,
            submit_fn=_heartbeat_submit,
        )
        heartbeat_task = asyncio.create_task(heartbeat_mgr.run())
        tasks.append(heartbeat_task)
        logger.info("heartbeat: enabled (interval=%ds)", config.harness.heartbeat.interval_secs)

    # --- run forever ---
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        cron.stop()
        if heartbeat_mgr:
            heartbeat_mgr.stop()
        await _manager.close_all()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
