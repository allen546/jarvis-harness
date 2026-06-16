import os
import json
from pydantic import BaseModel, Field
from typing import Optional, Any

class ModelConfig(BaseModel):
    provider: str
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    extra_params: dict[str, Any] = Field(default_factory=dict)

class HarnessConfig(BaseModel):
    system_prompt: Optional[str] = None
    max_consecutive_tools: int = 5
    require_tool_approval: bool = False
    allowed_skills: list[str] = Field(default_factory=list)

class SessionConfig(BaseModel):
    session_id: str = "default"
    model: ModelConfig
    harness: HarnessConfig = Field(default_factory=HarnessConfig)

def load_session_config(session_id: str, config_dir: str = "config") -> SessionConfig:
    session_file = os.path.join(config_dir, "sessions", f"session_{session_id}.json")
    global_file = os.path.join(config_dir, "global.json")
    
    data: dict[str, Any] = {}
    if os.path.exists(global_file):
        with open(global_file, "r") as f:
            data = json.load(f)
            
    if os.path.exists(session_file):
        with open(session_file, "r") as f:
            session_data: dict[str, Any] = json.load(f)
            for k, v in session_data.items():
                if k in data and isinstance(data[k], dict) and isinstance(v, dict):
                    data[k].update(v)
                else:
                    data[k] = v
    
    if not data:
        data = {
            "model": {"provider": "openai", "model_name": "gpt-4o"},
            "harness": {}
        }
    return SessionConfig(**data)
