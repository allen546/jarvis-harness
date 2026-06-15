from pydantic import BaseModel, Field
from typing import Any, Optional, AsyncGenerator

class Attachment(BaseModel):
    file_path: str
    mime_type: str
    description: Optional[str] = None

class NativeAction(BaseModel):
    action_type: str
    params: dict[str, Any]

class Message(BaseModel):
    role: str
    content: str
    attachments: list[Attachment] = Field(default_factory=list)
    native_actions: list[NativeAction] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

class ToolCall(BaseModel):
    call_id: str
    tool_name: str
    arguments: dict[str, Any]

class ModelResponse(BaseModel):
    content: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    raw_response: Any = None

class BaseModelClient:
    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        raise NotImplementedError

    async def generate_stream(self, messages: list[Message], tools: list[Any]) -> AsyncGenerator[ModelResponse, None]:
        raise NotImplementedError
        yield
