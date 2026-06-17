from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from jarvis.models.base import Message, ModelResponse
from main import app


class FakeModel:
    @classmethod
    def from_cfg(cls, cfg: object) -> "FakeModel":
        return cls()

    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        return ModelResponse(content="hello gateway")


@pytest.mark.asyncio
async def test_gateway_returns_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from jarvis.config import ModelConfig, SessionConfig

    monkeypatch.setattr("jarvis.runtime.get_model_class", lambda provider: FakeModel)
    monkeypatch.setattr("main.load_session_config", lambda session_id: SessionConfig(session_id=session_id, model=ModelConfig(provider="fake", model_name="fake")))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/sessions/s1/turns", json={"content": "hi"})

    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "hello gateway"
    assert data["session_id"] == "s1"
