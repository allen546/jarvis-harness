import os
from typing import TYPE_CHECKING, AsyncGenerator, Any
from jarvis.models.base import BaseModelClient, Message, ModelResponse, register_model

if TYPE_CHECKING:
    from jarvis.config import SessionConfig


@register_model("gemini")
class GeminiClient(BaseModelClient):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    @classmethod
    def from_cfg(cls, cfg: SessionConfig) -> GeminiClient:
        extra = cfg.model.extra_params or {}
        api_key = extra.get("api_key") or os.getenv("GEMINI_API_KEY", "mock-key")
        return cls(api_key=api_key, model_name=cfg.model.model_name)

    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        raise NotImplementedError("GeminiClient is currently stubbed and not implemented.")

    async def generate_stream(self, messages: list[Message], tools: list[Any]) -> AsyncGenerator[ModelResponse, None]:
        raise NotImplementedError("GeminiClient is currently stubbed and not implemented.")
        # Make this an async generator (unreachable but required for type)
        if False:  # pragma: no cover
            yield  # type: ignore[misc]
