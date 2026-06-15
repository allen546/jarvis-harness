import pytest
from jarvis.models.gemini import GeminiClient
from jarvis.models.openai_compatible import OpenAICompatibleClient

@pytest.mark.asyncio
async def test_gemini_stub():
    client = GeminiClient(api_key="fake-key", model_name="gemini-1.5-flash")
    with pytest.raises(NotImplementedError):
        await client.generate([], [])

def test_openai_compatible_client_init():
    client = OpenAICompatibleClient(api_key="fake-key", model_name="local-llama", base_url="http://localhost:8000/v1")
    assert client.base_url == "http://localhost:8000/v1"
