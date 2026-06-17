import importlib
import json
import os
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional
from jarvis.models.base import BaseModelClient, Message, ModelResponse, ToolCall, register_model

if TYPE_CHECKING:
    from jarvis.config import SessionConfig

@register_model("openai")
class OpenAIClient(BaseModelClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        extra_params: Optional[dict[str, Any]] = None
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_params = extra_params or {}
        self._client = None

    @classmethod
    def from_cfg(cls, cfg: SessionConfig) -> OpenAIClient:
        extra = cfg.model.extra_params or {}
        api_key = extra.get("api_key") or os.getenv(f"{cfg.model.provider.upper()}_API_KEY", "mock-key")
        forward_params = {k: v for k, v in extra.items() if k not in ("api_key", "base_url")}
        return cls(
            api_key=api_key,
            model_name=cfg.model.model_name,
            base_url=extra.get("base_url"),
            max_tokens=cfg.model.max_tokens,
            temperature=cfg.model.temperature,
            extra_params=forward_params
        )

    async def _get_client(self) -> Any:
        if self._client is None:
            openai = importlib.import_module("openai")
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        client = await self._get_client()
        openai_msgs: list[dict[str, Any]] = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.role == "assistant" and m.metadata and "tool_calls" in m.metadata:
                msg["tool_calls"] = [
                    {"id": tc["call_id"], "type": "function", "function": {"name": tc["tool_name"], "arguments": json.dumps(tc["arguments"])}}
                    for tc in m.metadata["tool_calls"]
                ]
            elif m.role == "tool" and m.metadata:
                if "tool_call_id" in m.metadata:
                    msg["tool_call_id"] = m.metadata["tool_call_id"]
            openai_msgs.append(msg)
        
        is_thinking_enabled = False
        if "thinking" in self.extra_params:
            thinking_val = self.extra_params["thinking"]
            if thinking_val != "disabled" and not (isinstance(thinking_val, dict) and thinking_val.get("type") == "disabled"):
                is_thinking_enabled = True

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": openai_msgs,
        }
        if not is_thinking_enabled:
            kwargs["temperature"] = self.temperature
            
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        if self.extra_params:
            extra_body = {k: v for k, v in self.extra_params.items() if k not in ("temperature", "max_tokens", "tools", "model", "messages", "stream")}
            if extra_body:
                kwargs["extra_body"] = extra_body

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            tool_calls = [ToolCall(call_id=tc.id, tool_name=tc.function.name, arguments=json.loads(tc.function.arguments)) for tc in choice.message.tool_calls]
        return ModelResponse(content=choice.message.content, tool_calls=tool_calls, raw_response=response)

    async def generate_stream(self, messages: list[Message], tools: list[Any]) -> AsyncGenerator[ModelResponse, None]:
        client = await self._get_client()
        openai_msgs: list[dict[str, Any]] = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.role == "assistant" and m.metadata and "tool_calls" in m.metadata:
                msg["tool_calls"] = [
                    {"id": tc["call_id"], "type": "function", "function": {"name": tc["tool_name"], "arguments": json.dumps(tc["arguments"])}}
                    for tc in m.metadata["tool_calls"]
                ]
            elif m.role == "tool" and m.metadata:
                if "tool_call_id" in m.metadata:
                    msg["tool_call_id"] = m.metadata["tool_call_id"]
            openai_msgs.append(msg)
        
        is_thinking_enabled = False
        if "thinking" in self.extra_params:
            thinking_val = self.extra_params["thinking"]
            if thinking_val != "disabled" and not (isinstance(thinking_val, dict) and thinking_val.get("type") == "disabled"):
                is_thinking_enabled = True

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": openai_msgs,
            "stream": True
        }
        if not is_thinking_enabled:
            kwargs["temperature"] = self.temperature
            
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.extra_params:
            extra_body = {k: v for k, v in self.extra_params.items() if k not in ("temperature", "max_tokens", "tools", "model", "messages", "stream")}
            if extra_body:
                kwargs["extra_body"] = extra_body

        response = await client.chat.completions.create(**kwargs)
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield ModelResponse(content=chunk.choices[0].delta.content, tool_calls=[], raw_response=chunk)
