import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient
from jarvis.models.openai_compatible import OpenAICompatibleClient
from jarvis.models.base import ModelResponse
from main import app

@pytest.mark.asyncio
async def test_integration_sse_stream(monkeypatch):
    # Mock generate_stream of OpenAICompatibleClient
    async def mock_generate_stream(self, messages, tools):
        yield ModelResponse(content="Hello ", tool_calls=[], raw_response=None)
        yield ModelResponse(content="world!", tool_calls=[], raw_response=None)
        
    monkeypatch.setattr(OpenAICompatibleClient, "generate_stream", mock_generate_stream)

    # We also mock load_session_config to return an openai_compatible provider configuration
    from jarvis.config import SessionConfig, ModelConfig, HarnessConfig
    mock_config = SessionConfig(
        model=ModelConfig(
            provider="openai_compatible",
            model_name="local-llama",
            temperature=0.7,
            extra_params={"base_url": "http://localhost:8000/v1"}
        ),
        harness=HarnessConfig(system_prompt="Test system prompt")
    )
    monkeypatch.setattr("main.load_session_config", lambda session_id: mock_config)

    from httpx import ASGITransport
    # Make the HTTP request
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/sessions/test-session-sse/turns",
            json={
                "content": "Hi",
                "channel": "sse",
                "channel_params": {}
            }
        )
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        # Read the stream lines
        events = []
        async for line in response.aiter_lines():
            if line.strip():
                events.append(line)
        
        # Check that we received chunks and the final message event
        assert any("event: chunk" in e for e in events)
        assert any("event: message" in e for e in events)
        assert any("Hello world!" in e for e in events)
