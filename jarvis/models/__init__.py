# Side-effect imports: each registers itself via @register_model.
from jarvis.models.base import register_model, get_model_class  # noqa: F401
from jarvis.models.openai import OpenAIClient  # noqa: F401
from jarvis.models.openai_compatible import OpenAICompatibleClient  # noqa: F401
from jarvis.models.anthropic import AnthropicClient  # noqa: F401
from jarvis.models.gemini import GeminiClient  # noqa: F401
