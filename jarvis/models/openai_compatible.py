import os
from typing import TYPE_CHECKING, Optional
from jarvis.models.openai import OpenAIClient
from jarvis.models.base import register_model

if TYPE_CHECKING:
    from jarvis.config import SessionConfig


@register_model("openai_compatible")
class OpenAICompatibleClient(OpenAIClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        extra_params: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_params=extra_params
        )

    @classmethod
    def from_cfg(cls, cfg: SessionConfig) -> OpenAICompatibleClient:
        extra = cfg.model.extra_params or {}
        api_key = extra.get("api_key") or os.getenv("OPENAI_COMPATIBLE_API_KEY", "mock-key")
        base_url = extra.get("base_url") or "http://localhost:8000/v1"
        forward_params = {k: v for k, v in extra.items() if k not in ("api_key", "base_url")}
        return cls(
            api_key=api_key,
            model_name=cfg.model.model_name,
            base_url=base_url,
            max_tokens=cfg.model.max_tokens,
            temperature=cfg.model.temperature,
            extra_params=forward_params
        )
