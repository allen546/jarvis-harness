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

class EmbeddingConfig(BaseModel):
    enabled: bool = False
    url: str = ""                    # HTTP endpoint; empty = local char-ngram fallback
    model: str = "text-embedding-3-small"
    dimensions: int = 256            # only used by local fallback

class HeartbeatConfig(BaseModel):
    enabled: bool = False
    interval_secs: int = 300
    workspace: str = "."

class HarnessConfig(BaseModel):
    system_prompt: Optional[str] = None
    max_consecutive_tools: int = 5
    require_tool_approval: bool = False
    allowed_skills: list[str] = Field(default_factory=list)
    stream: bool = True
    # safety hooks
    max_repeated_tool_calls: int = 3
    repeated_content_threshold: float = 0.8
    repeated_content_window: int = 3
    # embedding
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    # heartbeat
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)

class QQChannelConfig(BaseModel):
    enabled: bool = False
    app_id: str = ""
    app_secret: str = ""
    intents: list[str] = Field(default_factory=lambda: ["public_messages"])
    allowed_senders: list[str] = Field(default_factory=list)

class CronTaskConfig(BaseModel):
    name: str
    schedule: str  # cron expression: "*/15 * * * *"
    prompt: str  # prompt to send to agent
    session_id: str = "cron"
    enabled: bool = True

class CronConfig(BaseModel):
    enabled: bool = False
    tasks: list[CronTaskConfig] = Field(default_factory=list)

class ChannelsConfig(BaseModel):
    qq: QQChannelConfig = Field(default_factory=QQChannelConfig)

class ProxyConfig(BaseModel):
    """HTTP/SOCKS proxy settings. Passed to all spawned MCP stdio servers."""
    http_proxy: str = ""      # e.g. "http://127.0.0.1:7890"
    https_proxy: str = ""     # e.g. "http://127.0.0.1:7890"
    all_proxy: str = ""       # e.g. "socks5h://127.0.0.1:7891"
    no_proxy: str = "localhost,127.0.0.1,::1"

class GatewayConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000

class JarvisConfig(BaseModel):
    model: ModelConfig
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    cron: CronConfig = Field(default_factory=CronConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)

class SessionConfig(BaseModel):
    session_id: str = "default"
    model: ModelConfig
    harness: HarnessConfig = Field(default_factory=HarnessConfig)

def load_config(config_dir: str = "config") -> JarvisConfig:
    global_file = os.path.join(config_dir, "global.json")
    data: dict[str, Any] = {}
    if os.path.exists(global_file):
        with open(global_file, "r") as f:
            data = json.load(f)
    return JarvisConfig(**data)

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
