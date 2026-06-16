import pytest
from unittest.mock import MagicMock
from jarvis.models.gemini import GeminiClient
from jarvis.models.openai_compatible import OpenAICompatibleClient

@pytest.mark.asyncio
async def test_gemini_stub() -> None:
    client = GeminiClient(api_key="fake-key", model_name="gemini-1.5-flash")  # type: ignore[call-arg]
    with pytest.raises(NotImplementedError):
        await client.generate([], [])

def test_openai_compatible_client_init() -> None:
    client = OpenAICompatibleClient(api_key="fake-key", model_name="local-llama", base_url="http://localhost:8000/v1", max_tokens=2048, temperature=0.9)  # type: ignore[call-arg]
    assert client.base_url == "http://localhost:8000/v1"  # type: ignore[attr-defined]
    assert client.max_tokens == 2048  # type: ignore[attr-defined]
    assert client.temperature == 0.9  # type: ignore[attr-defined]

@pytest.mark.asyncio
async def test_openai_compatible_client_lazyload(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_openai_module = MagicMock()
    monkeypatch.setattr("importlib.import_module", lambda name: mock_openai_module if name == "openai" else MagicMock())

    client = OpenAICompatibleClient(api_key="fake-key", model_name="local-llama", base_url="http://localhost:8000/v1")  # type: ignore[call-arg]
    # Call _get_client twice and assert it returns the same instance
    c1 = await client._get_client()  # type: ignore[attr-defined]
    c2 = await client._get_client()  # type: ignore[attr-defined]
    assert c1 is c2
