from typing import AsyncGenerator, Any
from jarvis.models.base import BaseModelClient, Message, ModelResponse

class GeminiClient(BaseModelClient):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        raise NotImplementedError("GeminiClient is currently stubbed and not implemented.")

    async def generate_stream(self, messages: list[Message], tools: list[Any]) -> AsyncGenerator[ModelResponse, None]:
        raise NotImplementedError("GeminiClient is currently stubbed and not implemented.")
        yield
