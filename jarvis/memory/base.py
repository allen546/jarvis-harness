from jarvis.models.base import Message
from pydantic import BaseModel, Field
from typing import Optional, Any

class SessionContext(BaseModel):
    session_id: str
    parent_session_id: Optional[str] = None
    scope: dict[str, Any] = Field(default_factory=dict)

class BaseMemoryEngine:
    async def load_history(self, context: SessionContext) -> list[Message]:
        raise NotImplementedError
    async def save_history(self, context: SessionContext, messages: list[Message]):
        raise NotImplementedError
