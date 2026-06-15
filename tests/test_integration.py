import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from jarvis.models.openai_compatible import OpenAICompatibleClient
from jarvis.models.base import ModelResponse
from main import app

@pytest.mark.asyncio
async def test_integration_sse_stream(monkeypatch, tmp_path):
    # Avoid writing history to the workspace root
    monkeypatch.setenv("JARVIS_HISTORY_DIR", str(tmp_path))

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

@pytest.mark.asyncio
async def test_integration_sse_stream_error(monkeypatch, tmp_path):
    # Avoid writing history to the workspace root
    monkeypatch.setenv("JARVIS_HISTORY_DIR", str(tmp_path))

    # Mock generate_stream to raise an exception
    async def mock_generate_stream_error(self, messages, tools):
        raise ValueError("Simulated model error")
        yield  # Make it an async generator
        
    monkeypatch.setattr(OpenAICompatibleClient, "generate_stream", mock_generate_stream_error)

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

    # Make the HTTP request
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Awaiting response should complete (no hanging) even when an exception is raised
        async with ac.stream(
            "POST",
            "/sessions/test-session-sse-error/turns",
            json={
                "content": "Hi",
                "channel": "sse",
                "channel_params": {}
            }
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            
            # Read the stream lines
            events = []
            try:
                async for line in response.aiter_lines():
                    if line.strip():
                        events.append(line)
            except ValueError as e:
                assert "Simulated model error" in str(e)
            
            # Verify that error notification was sent to channel and the stream closed
            assert any("Error: Simulated model error" in e for e in events)


@pytest.mark.asyncio
async def test_integration_bad_channel_params(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_HISTORY_DIR", str(tmp_path))

    from jarvis.config import SessionConfig, ModelConfig
    mock_config = SessionConfig(
        model=ModelConfig(
            provider="openai_compatible",
            model_name="local-llama",
            extra_params={"base_url": "http://localhost:8000/v1"}
        )
    )
    monkeypatch.setattr("main.load_session_config", lambda session_id: mock_config)

    # Test invalid channel parameters (passing None/missing for expected arguments)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/sessions/test-session-bad-channel/turns",
            json={
                "content": "Hi",
                "channel": "discord",
                # discord channel requires bot_token (non-optional parameter),
                # so passing None or missing it should raise a TypeError/KeyError and return HTTP 400
                "channel_params": {"guild_id": "123"}
            }
        )
        assert response.status_code == 400
        assert "Invalid channel or parameters" in response.json()["detail"]
