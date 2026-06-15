# Design Specification: Jarvis Agent Harness

A lightweight, openclaw-style agent harness featuring a gateway-first daemon architecture. It separates components cleanly, allowing plug-and-play capability for models, memory, channels, skills, and MCP servers.

---

## 1. Core Architecture & Interfaces

The core of Jarvis uses a direct-component interface (Service Locator) to keep execution paths simple and debuggable. 

### 1.1 Rich Message & Channels
Channels represent communication integrations (CLI, Discord, Slack, etc.) and handle rich media and native behaviors (reactions, spoilers) without polluting the core loop.

```python
class BaseChannel:
    async def send_message(self, session_id: str, message: Message):
        """Sends rich message (text, media, attachments) back to the channel."""
        raise NotImplementedError

    def get_channel_tools(self, session_id: str) -> list[Any]:
        """Dynamically registers channel-native tools (e.g. Discord reactions)."""
        return []
```

### 1.2 Model Client
The `ModelClient` handles all payload conversions, tool schema generation (via Pydantic), and response parsing. To avoid startup bloat, model SDKs are imported **dynamically** only when the provider is instantiated.

```python
class BaseModelClient:
    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        """Translates inputs, formats schemas, runs model, and parses response."""
        raise NotImplementedError
```

### 1.3 Memory Engine
Manages both short-term conversation histories and hierarchical scopes (for subagents). Long-term memory and distillation plans will be designed in a future phase.

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

### 2.1 Single-Inference Turn (`execute_turn`)
The core runner executes a single step of the agent's interaction—generating a prediction, executing its immediate tool calls, logging, and returning the step result. The outer driver/daemon coordinates multi-turn sequencing.

```python
class AgentHarness:
    async def execute_turn(
        self, 
        session_ctx: SessionContext, 
        channel: BaseChannel, 
        user_message: Optional[Message] = None
    ) -> TurnResult:
        # 1. Load context/history
        # 2. Apply pre-turn hooks
        # 3. Call Model Client (with active tools)
        # 4. Apply post-message hooks
        # 5. Execute any immediate tool calls
        # 6. Return TurnResult
```

### 2.2 Subagent Delegation
Subagents are treated as standard tool calls. When triggered, the harness builds a completely new, isolated `AgentHarness` and `SessionContext` instance, preventing state sharing or object reuse.

---

### 3.1 Project Directory Layout

All modules are flat sibling packages inside the main `jarvis` directory:

```text
jarvis/
├── config.yaml                     # Model parameters, ports, active MCPs
├── main.py                         # Gateway daemon (FastAPI & SSE)
└── jarvis/
    ├── __init__.py
    ├── config.py                   # Config schemas (Pydantic)
    ├── harness.py                  # Core AgentHarness execution
    ├── subagent.py                 # Subagent factory & tool wrapper
    ├── models/                     # [Folder] Dynamic-import model providers
    ├── memory/                     # [Folder] Context & history managers
    ├── channels/                   # [Folder] Discord/Slack/CLI channels
    ├── skills/                     # [Folder] SKILL.md parser
    └── mcp/                        # [Folder] MCP client integrations
```

### Dependency Stack
*   `httpx` (Standard async HTTP)
*   `pydantic` (Data schemas)
*   `fastapi` & `uvicorn` (Daemon interface)
*   `mcp` (Official MCP SDK)
*   `pyyaml` (YAML frontmatter)
