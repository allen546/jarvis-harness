from typing import Optional
from jarvis.models.openai import OpenAIClient

class OpenAICompatibleClient(OpenAIClient):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature
        )


