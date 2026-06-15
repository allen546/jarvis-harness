# Design Specification: Jarvis Agent Harness

A lightweight, openclaw-style agent harness featuring a gateway-first daemon architecture. It separates components cleanly, allowing plug-and-play capability for models, memory, channels, skills, and MCP servers.

---

## 1. Core Architecture & Interfaces

The core of Jarvis uses a direct-component interface (Service Locator) to keep execution paths simple and debuggable. 

### 1.1 Rich Message, Streaming & Channels
Channels represent communication integrations (Discord, QQ, CLI, etc.). They handle streaming chunks and native behaviors without polluting the core loop.

```python
class BaseChannel:
    async def send_message(self, session_id: str, message: Message):
        """Sends a complete rich message back to the channel."""
        raise NotImplementedError

    async def send_stream_chunk(self, session_id: str, chunk: str):
        """Streams a text chunk back to the channel in real-time."""
        pass

    def get_channel_tools(self, session_id: str) -> list[Any]:
        """Dynamically registers channel-native tools (e.g. Discord emoji reactions)."""
        return []
```

### 1.2 Model Client & Streaming Support
The `BaseModelClient` supports both one-shot generation and real-time token streaming. Model SDKs (like `anthropic`, `openai`, or `google-genai`) are imported **dynamically** inside their respective modules to avoid startup bloat.

```python
class BaseModelClient:
    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        """One-shot generation returning the complete response."""
        raise NotImplementedError

    async def generate_stream(self, messages: list[Message], tools: list[Any]):
        """Async generator yielding ModelResponse chunks containing text deltas."""
        raise NotImplementedError
        yield
```

### 1.3 Memory Engine & JSON Session Configs
Configurations are loaded dynamically from session-specific JSON files: `config/sessions/session_<session_id>.json`. This allows running multiple different agent setups (different system prompts, models, keys) concurrently.

```python
class BaseMemoryEngine:
    async def load_history(self, context: SessionContext) -> list[Message]:
        raise NotImplementedError
    async def save_history(self, context: SessionContext, messages: list[Message]):
        raise NotImplementedError
```

### 1.4 Skills & MCP Registries
*   **Skills:** Parsers for directory-based markdown instructions (`SKILL.md`) containing local tools.
*   **MCPs:** Clients connecting to external tools via the Model Context Protocol. Exposes tools using Pydantic models for type-safe validation.

---

## 2. Core Execution Loop & Subagents

### 2.1 Streamed turn execution (`execute_turn`)
The core runner executes a single turn. It invokes the model's stream, forwards text chunks to the channel, aggregates the final response, and executes any generated tool calls before writing to history.

---

## 3. Project Directory Layout

All modules are flat sibling packages inside the main `jarvis` directory:

```text
jarvis/
├── config/
│   ├── global.json                 # Global configurations and defaults
│   └── sessions/                   # Session-specific configuration JSON files
├── main.py                         # Gateway daemon (FastAPI & SSE)
└── jarvis/
    ├── __init__.py
    ├── config.py                   # Config schemas (Pydantic)
    ├── harness.py                  # Core AgentHarness execution
    ├── subagent.py                 # Subagent factory & tool wrapper
    ├── models/                     # [Folder] Native SDKs & OpenAI-compatible providers
    ├── memory/                     # [Folder] Session history manager
    ├── channels/                   # [Folder] Discord, QQ, and generic Webhook channels
    ├── skills/                     # [Folder] SKILL.md parser
    └── mcp/                        # [Folder] MCP client integrations
```

### Dependency Stack
*   `httpx` (Standard async HTTP)
*   `pydantic` (Data schemas)
*   `fastapi` & `uvicorn` (Daemon interface)
*   `mcp` (Official MCP SDK)
*   `pyyaml` (YAML frontmatter)
