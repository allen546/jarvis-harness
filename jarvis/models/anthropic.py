import importlib
import os
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional
from jarvis.models.base import BaseModelClient, Message, ModelResponse, ToolCall, register_model
from jarvis.retry import retry_with_backoff

if TYPE_CHECKING:
    from jarvis.config import SessionConfig

@register_model("anthropic")
class AnthropicClient(BaseModelClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    @classmethod
    def from_cfg(cls, cfg: SessionConfig) -> AnthropicClient:
        extra = cfg.model.extra_params or {}
        api_key = extra.get("api_key") or os.getenv("ANTHROPIC_API_KEY", "mock-key")
        return cls(
            api_key=api_key,
            model_name=cfg.model.model_name,
            base_url=extra.get("base_url"),
            max_tokens=cfg.model.max_tokens or 1024,
            temperature=cfg.model.temperature,
        )

    async def _get_client(self) -> Any:
        if self._client is None:
            anthropic = importlib.import_module("anthropic")
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        client = await self._get_client()
        anthropic_msgs: list[dict[str, Any]] = []
        system_prompt: Optional[str] = None
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                anthropic_msgs.append({"role": "assistant" if m.role == "assistant" else "user", "content": m.content})

        kwargs: dict[str, Any] = {
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
        import json
        client = await self._get_client()
        anthropic_msgs = [{"role": "assistant" if m.role == "assistant" else "user", "content": m.content} for m in messages if m.role != "system"]
        system_prompt = next((m.content for m in messages if m.role == "system"), None)
        
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_msgs,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = tools

        tool_calls_builder = {}
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        async def _enter_stream():
            mgr = client.messages.stream(**kwargs)
            return await mgr.__aenter__()

        stream = await _enter_stream()
        try:
            async for event in stream:
                if event.type == "content_block_start":
                    index = event.index
                    block = event.content_block
                    if block.type == "tool_use":
                        tool_calls_builder[index] = {
                            "id": block.id,
                            "name": block.name,
                            "arguments_str": ""
                        }
                elif event.type == "content_block_delta":
                    index = event.index
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield ModelResponse(content=delta.text, tool_calls=[], raw_response=event)
                    elif delta.type == "input_json_delta":
                        if index in tool_calls_builder:
                            tool_calls_builder[index]["arguments_str"] += delta.partial_json
        finally:
            await stream.__aexit__(None, None, None)

        tool_calls = []
        for index in sorted(tool_calls_builder.keys()):
            tc_data = tool_calls_builder[index]
            try:
                args = json.loads(tc_data["arguments_str"]) if tc_data["arguments_str"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                call_id=tc_data["id"],
                tool_name=tc_data["name"],
                arguments=args
            ))
        if tool_calls:
            yield ModelResponse(content=None, tool_calls=tool_calls, raw_response=None)
