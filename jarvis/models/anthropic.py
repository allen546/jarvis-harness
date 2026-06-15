import importlib
from typing import Any, AsyncGenerator, Optional
from jarvis.models.base import BaseModelClient, Message, ModelResponse, ToolCall

class AnthropicClient(BaseModelClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
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
            anthropic = importlib.import_module("anthropic")
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        client = await self._get_client()
        anthropic_msgs = []
        system_prompt = None
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                anthropic_msgs.append({"role": "assistant" if m.role == "assistant" else "user", "content": m.content})

        kwargs = {
            "model": self.model_name,
            "messages": anthropic_msgs,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = tools

        response = await client.messages.create(**kwargs)
        content_text = "".join([c.text for c in response.content if c.type == "text"])
        tool_calls = [ToolCall(call_id=c.id, tool_name=c.name, arguments=c.input) for c in response.content if c.type == "tool_use"]
        return ModelResponse(content=content_text, tool_calls=tool_calls, raw_response=response)

    async def generate_stream(self, messages: list[Message], tools: list[Any]) -> AsyncGenerator[ModelResponse, None]:
        client = await self._get_client()
        anthropic_msgs = [{"role": "assistant" if m.role == "assistant" else "user", "content": m.content} for m in messages if m.role != "system"]
        system_prompt = next((m.content for m in messages if m.role == "system"), None)
        
        kwargs = {
            "model": self.model_name,
            "messages": anthropic_msgs,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield ModelResponse(content=text, tool_calls=[], raw_response=None)
