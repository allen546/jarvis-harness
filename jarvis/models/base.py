from __future__ import annotations

from dataclasses import dataclass, field
from pydantic import BaseModel
from typing import TYPE_CHECKING, Any, Callable, Optional, AsyncGenerator, Dict, Type

if TYPE_CHECKING:
    from jarvis.config import SessionConfig

# Model Provider Registry
_MODEL_REGISTRY: Dict[str, Type['BaseModelClient']] = {}

def register_model(name: str) -> Callable[[Type['BaseModelClient']], Type['BaseModelClient']]:
    def decorator(cls: Type['BaseModelClient']) -> Type['BaseModelClient']:
        _MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def get_model_class(name: str) -> Type['BaseModelClient']:
    name_lower = name.lower()
    if name_lower not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown provider: {name}")
    return _MODEL_REGISTRY[name_lower]


@dataclass(slots=True)
class Attachment:
    mime_type: str
    url: str | None = None
    file_path: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.url and not self.file_path:
            raise ValueError("Attachment requires url or file_path")

    def model_dump(self) -> dict[str, Any]:
        return {
            "mime_type": self.mime_type,
            "url": self.url,
            "file_path": self.file_path,
            "description": self.description,
        }


@dataclass(slots=True)
class NativeAction:
    action_type: str
    params: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "params": self.params,
        }


@dataclass(slots=True)
class Message:
    role: str
    content: str
    attachments: list[Attachment] = field(default_factory=list)  # type: ignore[assignment]
    native_actions: list[NativeAction] = field(default_factory=list)  # type: ignore[assignment]
    metadata: dict[str, Any] = field(default_factory=dict)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Handle dict-based initialization for nested items (deserialization support)
        if self.attachments and isinstance(self.attachments[0], dict):
            self.attachments = [Attachment(**a) if isinstance(a, dict) else a for a in self.attachments]  # type: ignore[arg-type]
        if self.native_actions and isinstance(self.native_actions[0], dict):
            self.native_actions = [NativeAction(**a) if isinstance(a, dict) else a for a in self.native_actions]  # type: ignore[arg-type]

    def model_dump(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "attachments": [a.model_dump() for a in self.attachments],
            "native_actions": [na.model_dump() for na in self.native_actions],
            "metadata": self.metadata,
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> Message:
        return cls(**data)



@dataclass(slots=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        }


@dataclass(slots=True)
class ModelResponse:
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)  # type: ignore[assignment]
    raw_response: Any = None

    def __post_init__(self) -> None:
        if self.tool_calls and isinstance(self.tool_calls[0], dict):
            self.tool_calls = [ToolCall(**tc) if isinstance(tc, dict) else tc for tc in self.tool_calls]  # type: ignore[arg-type]

    def model_dump(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [tc.model_dump() for tc in self.tool_calls],
            "raw_response": self.raw_response,
        }


class BaseModelClient:
    @classmethod
    def from_cfg(cls, cfg: SessionConfig) -> BaseModelClient:
        raise NotImplementedError

    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        raise NotImplementedError

    async def generate_stream(self, messages: list[Message], tools: list[Any]) -> AsyncGenerator[ModelResponse, None]:
        raise NotImplementedError
