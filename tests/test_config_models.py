import json
from pathlib import Path
from jarvis.config import load_session_config
from jarvis.models.base import Attachment, NativeAction, Message

def test_configs_and_messages(tmp_path: Path) -> None:
    config_dir = tmp_path / "config" / "sessions"
    config_dir.mkdir(parents=True)
    session_file = config_dir / "session_123.json"
    session_file.write_text(json.dumps({
        "model": {
            "provider": "openai_compatible",
            "model_name": "local-llama",
            "temperature": 0.5,
            "extra_params": {"base_url": "http://localhost:11434/v1"}
        }
    }))
    
    cfg = load_session_config("123", config_dir=str(tmp_path / "config"))
    assert cfg.model.provider == "openai_compatible"
    assert cfg.model.extra_params["base_url"] == "http://localhost:11434/v1"

    attachment = Attachment(file_path="/tmp/test.jpg", mime_type="image/jpeg")
    action = NativeAction(action_type="react", params={"emoji": "\U0001f44d"})
    msg = Message(role="user", content="Hello", attachments=[attachment], native_actions=[action])
    assert len(msg.attachments) == 1


def test_runtime_config_extensions() -> None:
    from jarvis.runtime import RuntimeConfig
    config = RuntimeConfig(
        system_prompt="test",
        max_consecutive_tools=10,
        require_tool_approval=True,
        allowed_skills=["git"],
        stream=False
    )
    assert config.max_consecutive_tools == 10
    assert config.require_tool_approval is True
    assert config.allowed_skills == ["git"]
    assert config.stream is False

    # Assert default values
    config_default = RuntimeConfig()
    assert config_default.system_prompt is None
    assert config_default.max_consecutive_tools == 5
    assert config_default.require_tool_approval is False
    assert config_default.allowed_skills == []
    assert config_default.stream is True


def test_context_from_config_propagation() -> None:
    from jarvis.config import SessionConfig, ModelConfig, HarnessConfig
    from jarvis.runtime import context_from_config
    from jarvis.tools import ToolRegistry

    # 1. Default propagation from a minimal SessionConfig yields stream=True in ctx.config.
    config_default = SessionConfig(
        model=ModelConfig(provider="openai", model_name="gpt-4o")
    )
    ctx_default = context_from_config(config_default, ToolRegistry())
    assert ctx_default.config.stream is True
    assert ctx_default.config.max_consecutive_tools == 5
    assert ctx_default.config.require_tool_approval is False
    assert ctx_default.config.allowed_skills == []

    # 2. Explicit propagation (e.g. loading HarnessConfig(stream=False)) yields stream=False in ctx.config.
    config_explicit = SessionConfig(
        model=ModelConfig(provider="openai", model_name="gpt-4o"),
        harness=HarnessConfig(
            system_prompt="custom prompt",
            max_consecutive_tools=12,
            require_tool_approval=True,
            allowed_skills=["git", "cmd"],
            stream=False
        )
    )
    ctx_explicit = context_from_config(config_explicit, ToolRegistry())
    assert ctx_explicit.config.system_prompt == "custom prompt"
    assert ctx_explicit.config.max_consecutive_tools == 12
    assert ctx_explicit.config.require_tool_approval is True
    assert ctx_explicit.config.allowed_skills == ["git", "cmd"]
    assert ctx_explicit.config.stream is False

