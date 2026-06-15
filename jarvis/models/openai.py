import importlib
from typing import Any, AsyncGenerator, Optional
from jarvis.models.base import BaseModelClient, Message, ModelResponse, ToolCall

class OpenAIClient(BaseModelClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    async def _get_client(self):
        if self._client is None:
            openai = importlib.import_module("openai")
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        client = await self._get_client()
        openai_msgs = [{"role": m.role, "content": m.content} for m in messages]
        kwargs = {
            "model": self.model_name,
            "messages": openai_msgs,
            "temperature": self.temperature
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        tool_calls = []
        if choice.message.tool_calls:
            import json
            tool_calls = [ToolCall(call_id=tc.id, tool_name=tc.function.name, arguments=json.loads(tc.function.arguments)) for tc in choice.message.tool_calls]
        return ModelResponse(content=choice.message.content, tool_calls=tool_calls, raw_response=response)

    async def generate_stream(self, messages: list[Message], tools: list[Any]) -> AsyncGenerator[ModelResponse, None]:
        client = await self._get_client()
        openai_msgs = [{"role": m.role, "content": m.content} for m in messages]
        kwargs = {
            "model": self.model_name,
            "messages": openai_msgs,
            "temperature": self.temperature,
            "stream": True
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = await client.chat.completions.create(**kwargs)
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield ModelResponse(content=chunk.choices[0].delta.content, tool_calls=[], raw_response=chunk)
