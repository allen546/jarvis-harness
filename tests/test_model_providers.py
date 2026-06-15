import pytest
from jarvis.models.gemini import GeminiClient
from jarvis.models.openai_compatible import OpenAICompatibleClient

@pytest.mark.asyncio
async def test_gemini_stub():
    client = GeminiClient(api_key="fake-key", model_name="gemini-1.5-flash")
    with pytest.raises(NotImplementedError):
        await client.generate([], [])

def test_openai_compatible_client_init():
    client = OpenAICompatibleClient(api_key="fake-key", model_name="local-llama", base_url="http://localhost:8000/v1", max_tokens=2048, temperature=0.9)
    assert client.base_url == "http://localhost:8000/v1"
    assert client.max_tokens == 2048
    assert client.temperature == 0.9

@pytest.mark.asyncio
async def test_openai_compatible_client_lazyload(monkeypatch):
    from unittest.mock import MagicMock
    mock_openai_module = MagicMock()
    monkeypatch.setattr("importlib.import_module", lambda name: mock_openai_module if name == "openai" else MagicMock())

    client = OpenAICompatibleClient(api_key="fake-key", model_name="local-llama", base_url="http://localhost:8000/v1")
    # Call _get_client twice and assert it returns the same instance
    c1 = await client._get_client()
    c2 = await client._get_client()
    assert c1 is c2


