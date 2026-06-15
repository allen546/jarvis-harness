import os
import json
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from jarvis.config import load_session_config
from jarvis.memory.base import SessionContext
from jarvis.memory.jsonl import JSONLMemoryEngine
from jarvis.models.base import Message
from jarvis.harness import AgentHarness

from jarvis.models.openai import OpenAIClient
from jarvis.models.openai_compatible import OpenAICompatibleClient
from jarvis.models.anthropic import AnthropicClient
from jarvis.models.gemini import GeminiClient

from jarvis.channels.base import QueueChannel
from jarvis.channels.webhook import WebhookChannel
from jarvis.channels.discord import DiscordChannel
from jarvis.channels.qq import QQChannel

app = FastAPI(title="Jarvis Daemon Gateway")

class TurnRequest(BaseModel):
    content: str
    channel: str
    channel_params: Optional[Dict[str, Any]] = None

def instantiate_model_client(cfg):
    provider = cfg.model.provider.lower()
    extra_params = cfg.model.extra_params or {}
    
    if provider == "openai":
        return OpenAIClient(
            api_key=extra_params.get("api_key") or os.getenv("OPENAI_API_KEY", "mock-key"),
            model_name=cfg.model.model_name,
            base_url=extra_params.get("base_url"),
            max_tokens=cfg.model.max_tokens,
            temperature=cfg.model.temperature
        )
    elif provider == "openai_compatible":
        return OpenAICompatibleClient(
            api_key=extra_params.get("api_key") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_COMPATIBLE_API_KEY") or "mock-key",
            model_name=cfg.model.model_name,
            base_url=extra_params.get("base_url", "http://localhost:8000/v1"),
            max_tokens=cfg.model.max_tokens,
            temperature=cfg.model.temperature
        )
    elif provider == "anthropic":
        return AnthropicClient(
            api_key=extra_params.get("api_key") or os.getenv("ANTHROPIC_API_KEY", "mock-key"),
            model_name=cfg.model.model_name,
            base_url=extra_params.get("base_url"),
            max_tokens=cfg.model.max_tokens if cfg.model.max_tokens is not None else 1024,
            temperature=cfg.model.temperature
        )
    elif provider == "gemini":
        return GeminiClient(
            api_key=extra_params.get("api_key") or os.getenv("GEMINI_API_KEY", "mock-key"),
            model_name=cfg.model.model_name
        )
    else:
        raise ValueError(f"Unknown provider: {cfg.model.provider}")

def instantiate_channel(channel_name: str, params: Optional[Dict[str, Any]]):
    params = params or {}
    name = channel_name.lower()
    if name == "sse":
        return QueueChannel()
    elif name == "webhook":
        return WebhookChannel(callback_url=params.get("callback_url"))
    elif name == "discord":
        return DiscordChannel(bot_token=params.get("bot_token"), guild_id=params.get("guild_id"))
    elif name == "qq":
        return QQChannel(app_id=params.get("app_id"), app_secret=params.get("app_secret"))
    else:
        raise ValueError(f"Unknown channel: {channel_name}")

@app.post("/sessions/{session_id}/turns")
async def execute_session_turn(session_id: str, request: TurnRequest):
    # 1. Load session configuration
    try:
        session_cfg = load_session_config(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load session config: {str(e)}")

    # 2. Instantiate model client
    try:
        model_client = instantiate_model_client(session_cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to instantiate model client: {str(e)}")

    # 3. Instantiate JSONLMemoryEngine dynamically per session
    memory_engine = JSONLMemoryEngine(file_path=f"history_{session_id}.jsonl")

    # 4. Instantiate channel
    try:
        channel = instantiate_channel(request.channel, request.channel_params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 5. Prepare SessionContext and Message
    session_ctx = SessionContext(session_id=session_id)
    user_message = Message(role="user", content=request.content)

    # 6. Instantiate harness
    harness = AgentHarness(
        config=session_cfg.harness,
        model_client=model_client,
        memory_engine=memory_engine,
        mcp_manager=None,
        skills_manager=None
    )

    if request.channel.lower() == "sse":
        # Create background task for executing the turn
        task = asyncio.create_task(harness.execute_turn(session_ctx, channel, user_message))

        async def sse_generator():
            try:
                while True:
                    item = await channel.queue.get()
                    if item is None:
                        break
                    event = item["event"]
                    data = item["data"]
                    if isinstance(data, (dict, list)):
                        data_str = json.dumps(data)
                    else:
                        data_str = str(data)
                    yield f"event: {event}\ndata: {data_str}\n\n"
            except asyncio.CancelledError:
                # Cancel background task if client disconnects
                task.cancel()
                raise
            finally:
                try:
                    await task
                except Exception:
                    # Exception should already be handled/reported by harness,
                    # but we make sure we don't crash the generator's exit
                    pass

        return StreamingResponse(sse_generator(), media_type="text/event-stream")
    else:
        # Run in background for webhook/discord/qq/etc.
        asyncio.create_task(harness.execute_turn(session_ctx, channel, user_message))
        return {"status": "ok", "message": "Turn started in background"}

def main():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
