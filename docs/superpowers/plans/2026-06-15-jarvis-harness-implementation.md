# Jarvis Agent Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight, microkernel-style agent harness (`openclaw`-style) with dynamic-import model SDKs, a pluggable direct-component interface, custom hooks, and isolated subagent tool spawner.

**Architecture:** A thin main FastAPI gateway daemon manages sessions and coordinates model clients, memory, channels, skills parser, and MCP registries via a direct `execute_turn` runner step. Dynamic import of model SDKs inside provider classes avoids import overhead.

**Tech Stack:** Python, `httpx`, `pydantic`, `fastapi`, `uvicorn`, `mcp`, `pyyaml`

---

### Task 1: Core Config and Models Interface

**Files:**
- Create: `openclaw_lite/config.py`
- Create: `openclaw_lite/models/base.py`
- Create: `tests/test_config_models.py`

- [ ] **Step 1: Write the failing test for configuration and base models**
  Create `tests/test_config_models.py`:
  ```python
  import pytest
  from openclaw_lite.config import ModelConfig, HarnessConfig
  from openclaw_lite.models.base import Attachment, NativeAction, Message, ToolCall, ModelResponse

  def test_configs_and_messages():
      m_config = ModelConfig(provider="gemini", model_name="gemini-1.5-pro")
      h_config = HarnessConfig(system_prompt="Test prompt")
      assert m_config.provider == "gemini"
      assert h_config.system_prompt == "Test prompt"

      attachment = Attachment(file_path="/tmp/test.jpg", mime_type="image/jpeg")
      action = NativeAction(action_type="react", params={"emoji": "👍"})
      msg = Message(role="user", content="Hello", attachments=[attachment], native_actions=[action])
      assert len(msg.attachments) == 1
      assert msg.attachments[0].mime_type == "image/jpeg"
      assert msg.native_actions[0].params["emoji"] == "👍"

      tc = ToolCall(call_id="call-123", tool_name="get_weather", arguments={"location": "Paris"})
      resp = ModelResponse(content="Thinking", tool_calls=[tc], raw_response={})
      assert resp.tool_calls[0].tool_name == "get_weather"
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_config_models.py -v`
  Expected: FAIL (ModuleNotFoundError: No module named 'openclaw_lite')

- [ ] **Step 3: Write config and base models implementation**
  Create `openclaw_lite/config.py`:
  ```python
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
  ```

  Create `openclaw_lite/models/base.py`:
  ```python
  from pydantic import BaseModel, Field
  from typing import Any, Optional

  class Attachment(BaseModel):
      file_path: str
      mime_type: str
      description: Optional[str] = None

  class NativeAction(BaseModel):
      action_type: str
      params: dict[str, Any]

  class Message(BaseModel):
      role: str
      content: str
      attachments: list[Attachment] = Field(default_factory=list)
      native_actions: list[NativeAction] = Field(default_factory=list)
      metadata: dict[str, Any] = Field(default_factory=dict)

  class ToolCall(BaseModel):
      call_id: str
      tool_name: str
      arguments: dict[str, Any]

  class ModelResponse(BaseModel):
      content: Optional[str] = None
      tool_calls: list[ToolCall] = Field(default_factory=list)
      raw_response: Any = None

  class BaseModelClient:
      async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
          raise NotImplementedError
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_config_models.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add openclaw_lite/config.py openclaw_lite/models/base.py tests/test_config_models.py
  git commit -m "feat: add config schema and base models"
  ```

---

### Task 2: Dynamic-Import Model Providers

**Files:**
- Create: `openclaw_lite/models/gemini.py`
- Create: `openclaw_lite/models/anthropic.py`
- Create: `openclaw_lite/models/openai.py`
- Create: `tests/test_model_providers.py`

- [ ] **Step 1: Write mock tests for dynamic model imports**
  Create `tests/test_model_providers.py`:
  ```python
  import pytest
  from unittest.mock import MagicMock, patch
  from openclaw_lite.models.gemini import GeminiClient
  from openclaw_lite.models.anthropic import AnthropicClient
  from openclaw_lite.models.openai import OpenAIClient
  from openclaw_lite.models.base import Message

  @pytest.mark.asyncio
  @patch("httpx.AsyncClient.post")
  async def test_gemini_client(mock_post):
      mock_post.return_value = MagicMock(
          status_code=200,
          json=lambda: {"candidates": [{"content": {"parts": [{"text": "Hello Gemini"}]}}]}
      )
      client = GeminiClient(api_key="fake-key", model_name="gemini-1.5-flash")
      resp = await client.generate([Message(role="user", content="Hi")], [])
      assert resp.content == "Hello Gemini"

  @pytest.mark.asyncio
  async def test_anthropic_dynamic_import():
      # Ensure AnthropicClient initializes and only imports anthropic on generate
      with patch("importlib.import_module") as mock_import:
          client = AnthropicClient(api_key="fake-key", model_name="claude-3-5-sonnet")
          assert not mock_import.called
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_model_providers.py -v`
  Expected: FAIL (ModuleNotFoundError: No module named 'openclaw_lite.models.gemini')

- [ ] **Step 3: Implement dynamic-import model providers**
  Create `openclaw_lite/models/gemini.py`:
  ```python
  import httpx
  from typing import Any
  from openclaw_lite.models.base import BaseModelClient, Message, ModelResponse

  class GeminiClient(BaseModelClient):
      def __init__(self, api_key: str, model_name: str):
          self.api_key = api_key
          self.model_name = model_name

      async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
          # Use direct HTTP request via httpx to avoid heavy SDK
          url = f"https://generativelimitless.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
          contents = []
          for m in messages:
              contents.append({
                  "role": "model" if m.role == "assistant" else "user",
                  "parts": [{"text": m.content}]
              })
          payload = {"contents": contents}
          if tools:
              # Very basic tool conversion placeholder
              payload["tools"] = [{"functionDeclarations": tools}]
          
          async with httpx.AsyncClient() as client:
              r = await client.post(url, json=payload, timeout=30.0)
              r.raise_for_status()
              data = r.json()
              text = data["candidates"][0]["content"]["parts"][0]["text"]
              return ModelResponse(content=text, tool_calls=[], raw_response=data)
  ```

  Create `openclaw_lite/models/anthropic.py`:
  ```python
  import importlib
  from typing import Any
  from openclaw_lite.models.base import BaseModelClient, Message, ModelResponse, ToolCall

  class AnthropicClient(BaseModelClient):
      def __init__(self, api_key: str, model_name: str):
          self.api_key = api_key
          self.model_name = model_name

      async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
          # Dynamically import anthropic sdk to prevent startup import bloat
          anthropic = importlib.import_module("anthropic")
          client = anthropic.AsyncAnthropic(api_key=self.api_key)
          
          anthropic_msgs = []
          system_prompt = None
          for m in messages:
              if m.role == "system":
                  system_prompt = m.content
              else:
                  anthropic_msgs.append({
                      "role": "assistant" if m.role == "assistant" else "user",
                      "content": m.content
                  })

          kwargs = {
              "model": self.model_name,
              "messages": anthropic_msgs,
              "max_tokens": 1024,
          }
          if system_prompt:
              kwargs["system"] = system_prompt
          if tools:
              kwargs["tools"] = tools

          response = await client.messages.create(**kwargs)
          
          content_text = ""
          tool_calls = []
          for content_block in response.content:
              if content_block.type == "text":
                  content_text += content_block.text
              elif content_block.type == "tool_use":
                  tool_calls.append(ToolCall(
                      call_id=content_block.id,
                      tool_name=content_block.name,
                      arguments=content_block.input
                  ))

          return ModelResponse(content=content_text, tool_calls=tool_calls, raw_response=response)
  ```

  Create `openclaw_lite/models/openai.py`:
  ```python
  import importlib
  from typing import Any
  from openclaw_lite.models.base import BaseModelClient, Message, ModelResponse, ToolCall

  class OpenAIClient(BaseModelClient):
      def __init__(self, api_key: str, model_name: str):
          self.api_key = api_key
          self.model_name = model_name

      async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
          # Dynamically import openai sdk to prevent startup import bloat
          openai = importlib.import_module("openai")
          client = openai.AsyncOpenAI(api_key=self.api_key)
          
          openai_msgs = []
          for m in messages:
              openai_msgs.append({
                  "role": m.role,
                  "content": m.content
              })

          kwargs = {
              "model": self.model_name,
              "messages": openai_msgs,
          }
          if tools:
              kwargs["tools"] = [{"type": "function", "function": t} for t in tools]

          response = await client.chat.completions.create(**kwargs)
          choice = response.choices[0]
          
          tool_calls = []
          if choice.message.tool_calls:
              for tc in choice.message.tool_calls:
                  import json
                  tool_calls.append(ToolCall(
                      call_id=tc.id,
                      tool_name=tc.function.name,
                      arguments=json.loads(tc.function.arguments)
                  ))

          return ModelResponse(
              content=choice.message.content,
              tool_calls=tool_calls,
              raw_response=response
          )
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_model_providers.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add openclaw_lite/models/gemini.py openclaw_lite/models/anthropic.py openclaw_lite/models/openai.py tests/test_model_providers.py
  git commit -m "feat: implement dynamic model providers for gemini, claude, and openai"
  ```

---

### Task 3: Channels & Local Memory

**Files:**
- Create: `openclaw_lite/channels/base.py`
- Create: `openclaw_lite/channels/webhook.py`
- Create: `openclaw_lite/memory/base.py`
- Create: `openclaw_lite/memory/jsonl.py`
- Create: `tests/test_memory_channel.py`

- [ ] **Step 1: Write test for local memory and webhook channels**
  Create `tests/test_memory_channel.py`:
  ```python
  import pytest
  import os
  import json
  from openclaw_lite.memory.jsonl import JSONLMemoryEngine
  from openclaw_lite.models.base import Message
  from openclaw_lite.harness import SessionContext

  @pytest.mark.asyncio
  async def test_jsonl_memory(tmp_path):
      history_file = tmp_path / "sessions.jsonl"
      engine = JSONLMemoryEngine(file_path=str(history_file))
      ctx = SessionContext(session_id="session-1")
      
      msg1 = Message(role="user", content="Hi")
      msg2 = Message(role="assistant", content="Hello")
      await engine.save_history(ctx, [msg1, msg2])
      
      loaded = await engine.load_history(ctx)
      assert len(loaded) == 2
      assert loaded[0].content == "Hi"
      assert loaded[1].role == "assistant"
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_memory_channel.py -v`
  Expected: FAIL (ModuleNotFoundError: No module named 'openclaw_lite.memory.jsonl')

- [ ] **Step 3: Implement local memory and channels**
  Create `openclaw_lite/channels/base.py`:
  ```python
  from openclaw_lite.models.base import Message
  from typing import Any

  class BaseChannel:
      async def send_message(self, session_id: str, message: Message):
          raise NotImplementedError

      def get_channel_tools(self, session_id: str) -> list[Any]:
          return []
  ```

  Create `openclaw_lite/channels/webhook.py`:
  ```python
  from openclaw_lite.channels.base import BaseChannel
  from openclaw_lite.models.base import Message
  import httpx

  class WebhookChannel(BaseChannel):
      def __init__(self, callback_url: str):
          self.callback_url = callback_url

      async def send_message(self, session_id: str, message: Message):
          async with httpx.AsyncClient() as client:
              await client.post(self.callback_url, json={
                  "session_id": session_id,
                  "message": message.model_dump()
              })
  ```

  Create `openclaw_lite/memory/base.py`:
  ```python
  from openclaw_lite.models.base import Message
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
  ```

  Create `openclaw_lite/memory/jsonl.py`:
  ```python
  import json
  import os
  import aiofiles
  from openclaw_lite.memory.base import BaseMemoryEngine, SessionContext
  from openclaw_lite.models.base import Message

  class JSONLMemoryEngine(BaseMemoryEngine):
      def __init__(self, file_path: str = "history.jsonl"):
          self.file_path = file_path

      async def load_history(self, context: SessionContext) -> list[Message]:
          if not os.path.exists(self.file_path):
              return []
          messages = []
          async with aiofiles.open(self.file_path, mode="r") as f:
              async for line in f:
                  if not line.strip():
                      continue
                  data = json.loads(line)
                  if data.get("session_id") == context.session_id:
                      messages.append(Message(**data["message"]))
          return messages

      async def save_history(self, context: SessionContext, messages: list[Message]):
          async with aiofiles.open(self.file_path, mode="a") as f:
              for m in messages:
                  line = {
                      "session_id": context.session_id,
                      "message": m.model_dump()
                  }
                  await f.write(json.dumps(line) + "\n")
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_memory_channel.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add openclaw_lite/channels/base.py openclaw_lite/channels/webhook.py openclaw_lite/memory/base.py openclaw_lite/memory/jsonl.py tests/test_memory_channel.py
  git commit -m "feat: add channel and local memory JSONL engine"
  ```

---

### Task 4: Skills & MCP Registries

**Files:**
- Create: `openclaw_lite/skills/parser.py`
- Create: `openclaw_lite/mcp/client.py`
- Create: `tests/test_skills_mcp.py`

- [ ] **Step 1: Write test for SKILL.md parsing and triggers**
  Create `tests/test_skills_mcp.py`:
  ```python
  import pytest
  from openclaw_lite.skills.parser import parse_skill_file

  def test_skill_parsing(tmp_path):
      skill_dir = tmp_path / "my_skill"
      skill_dir.mkdir()
      skill_file = skill_dir / "SKILL.md"
      skill_file.write_text("""---
  name: greeting-skill
  description: Greets the user
  triggers: ["hello", "hi"]
  ---
  # Greeting Skill
  Always greet the user politely.""")

      skill = parse_skill_file(str(skill_file))
      assert skill.name == "greeting-skill"
      assert "greeting-skill" in skill.instructions
      assert "hello" in skill.triggers
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_skills_mcp.py -v`
  Expected: FAIL (ModuleNotFoundError: No module named 'openclaw_lite.skills.parser')

- [ ] **Step 3: Implement SKILL.md parser and MCP client**
  Create `openclaw_lite/skills/parser.py`:
  ```python
  import yaml
  import re
  from pydantic import BaseModel, Field
  from typing import list, Optional

  class Skill(BaseModel):
      name: str
      description: str
      instructions: str
      triggers: list[str] = Field(default_factory=list)
      local_script_path: Optional[str] = None

  def parse_skill_file(file_path: str) -> Skill:
      with open(file_path, "r", encoding="utf-8") as f:
          content = f.read()
      
      # Match yaml frontmatter
      match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
      if not match:
          raise ValueError(f"No valid frontmatter found in skill {file_path}")
      
      frontmatter_str, markdown_body = match.groups()
      meta = yaml.safe_load(frontmatter_str)
      
      # Inject skill name/details in instructions for LLM context compatibility
      instructions = f"Skill: {meta.get('name')}\nDescription: {meta.get('description')}\n{markdown_body}"
      
      return Skill(
          name=meta.get("name"),
          description=meta.get("description"),
          instructions=instructions,
          triggers=meta.get("triggers", []),
          local_script_path=meta.get("local_script_path")
      )
  ```

  Create `openclaw_lite/mcp/client.py`:
  ```python
  from mcp import ClientSession, StdioServerParameters
  from mcp.client.stdio import stdio_client
  from pydantic import BaseModel
  from typing import Any

  class MCPClientManager:
      def __init__(self, command: str, args: list[str]):
          self.server_params = StdioServerParameters(command=command, args=args)
          self.session = None
          self._client_context = None

      async def connect(self):
          self._client_context = stdio_client(self.server_params)
          read, write = await self._client_context.__aenter__()
          self.session = ClientSession(read, write)
          await self.session.__aenter__()
          await self.session.initialize()

      async def disconnect(self):
          if self.session:
              await self.session.__aexit__(None, None, None)
          if self._client_context:
              await self._client_context.__aexit__(None, None, None)

      async def list_tools(self) -> list[dict]:
          if not self.session:
              return []
          response = await self.session.list_tools()
          return [t.model_dump() for t in response.tools]

      async def call_tool(self, name: str, arguments: dict) -> str:
          if not self.session:
              raise RuntimeError("Not connected to MCP server")
          result = await self.session.call_tool(name, arguments)
          # Format output as string
          return "\n".join([c.text for c in result.content if hasattr(c, 'text')])
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_skills_mcp.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add openclaw_lite/skills/parser.py openclaw_lite/mcp/client.py tests/test_skills_mcp.py
  git commit -m "feat: implement skill markdown parser and mcp client session manager"
  ```

---

### Task 5: The Agent Harness Execution Step

**Files:**
- Create: `openclaw_lite/harness.py`
- Create: `tests/test_harness.py`

- [ ] **Step 1: Write mock tests for execute_turn and hooks**
  Create `tests/test_harness.py`:
  ```python
  import pytest
  from unittest.mock import AsyncMock, MagicMock
  from openclaw_lite.harness import AgentHarness, TurnResult, HarnessConfig, SessionContext
  from openclaw_lite.models.base import Message, ModelResponse

  @pytest.mark.asyncio
  async def test_execute_turn_without_tools():
      config = HarnessConfig(system_prompt="system instructions")
      model_client = MagicMock()
      model_client.generate = AsyncMock(return_value=ModelResponse(content="Response text"))
      
      memory = MagicMock()
      memory.load_history = AsyncMock(return_value=[])
      memory.save_history = AsyncMock()

      harness = AgentHarness(config, model_client, memory, MagicMock(), MagicMock())
      
      # Hook mock
      pre_hook_called = False
      async def dummy_pre_hook(ctx, history):
          nonlocal pre_hook_called
          pre_hook_called = True
          return history
      harness.pre_turn_hooks.append(dummy_pre_hook)

      ctx = SessionContext(session_id="session-test")
      channel = MagicMock()
      channel.send_message = AsyncMock()
      
      result = await harness.execute_turn(ctx, channel, Message(role="user", content="Hello"))
      
      assert result.response.content == "Response text"
      assert not result.has_more_actions
      assert pre_hook_called
      memory.save_history.assert_called()
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_harness.py -v`
  Expected: FAIL (ModuleNotFoundError: No module named 'openclaw_lite.harness')

- [ ] **Step 3: Implement execution harness step**
  Create `openclaw_lite/harness.py`:
  ```python
  from typing import Optional, Any, Callable, list
  from openclaw_lite.config import HarnessConfig
  from openclaw_lite.memory.base import SessionContext, BaseMemoryEngine
  from openclaw_lite.models.base import BaseModelClient, Message, ToolCall, ModelResponse
  from openclaw_lite.channels.base import BaseChannel
  from pydantic import BaseModel, Field

  class TurnResult(BaseModel):
      response: ModelResponse
      tool_results: list[tuple[ToolCall, str]] = Field(default_factory=list)
      has_more_actions: bool = False

  class AgentHarness:
      def __init__(
          self,
          config: HarnessConfig,
          model_client: BaseModelClient,
          memory_engine: BaseMemoryEngine,
          mcp_manager: Any,
          skills_manager: Any
      ):
          self.config = config
          self.model_client = model_client
          self.memory_engine = memory_engine
          self.mcp_manager = mcp_manager
          self.skills_manager = skills_manager
          
          self.pre_turn_hooks: list[Callable] = []
          self.post_message_hooks: list[Callable] = []

      async def execute_turn(
          self,
          session_ctx: SessionContext,
          channel: BaseChannel,
          user_message: Optional[Message] = None
      ) -> TurnResult:
          # 1. Load history
          history = await self.memory_engine.load_history(session_ctx)
          
          # Append system prompt if history is empty
          if not history and self.config.system_prompt:
              history.insert(0, Message(role="system", content=self.config.system_prompt))
          
          # 2. Append new user message if provided
          if user_message:
              history.append(user_message)
              await self.memory_engine.save_history(session_ctx, [user_message])

          # Apply pre-turn hooks
          for hook in self.pre_turn_hooks:
              history = await hook(session_ctx, history)

          # 3. Gather tools (aggregate MCP and skill instructions)
          tools = [] # Schema configurations
          
          # 4. Call Model Client
          response = await self.model_client.generate(history, tools=tools)

          # Apply post-message hooks
          for hook in self.post_message_hooks:
              await hook(session_ctx, response)

          # 5. Save assistant reply
          assistant_msg = Message(role="assistant", content=response.content or "")
          await self.memory_engine.save_history(session_ctx, [assistant_msg])

          # 6. Execute requested tool calls (if any)
          tool_results = []
          if response.tool_calls:
              for tc in response.tool_calls:
                  output = "Tool executed." # Placeholder tool execution
                  tool_results.append((tc, output))
              
              # Append tool responses to history
              tool_msgs = []
              for tc, output in tool_results:
                  tool_msgs.append(Message(
                      role="system",
                      content=f"Tool call result for {tc.tool_name}: {output}"
                  ))
              await self.memory_engine.save_history(session_ctx, tool_msgs)

          return TurnResult(
              response=response,
              tool_results=tool_results,
              has_more_actions=len(response.tool_calls) > 0
          )
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_harness.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add openclaw_lite/harness.py tests/test_harness.py
  git commit -m "feat: implement single turn execution harness and hooks"
  ```

---

### Task 6: Subagent Spawner and Tool Delegation

**Files:**
- Create: `openclaw_lite/subagent.py`
- Create: `tests/test_subagent.py`

- [ ] **Step 1: Write test for isolated subagent execution**
  Create `tests/test_subagent.py`:
  ```python
  import pytest
  from unittest.mock import AsyncMock, MagicMock
  from openclaw_lite.subagent import SubagentSpawner
  from openclaw_lite.memory.base import SessionContext
  from openclaw_lite.config import HarnessConfig
  from openclaw_lite.models.base import Message

  @pytest.mark.asyncio
  async def test_subagent_spawner():
      # Spawner builds new harness instances dynamically and loops execute_turn
      spawner = SubagentSpawner(factory_config={
          "model": lambda: MagicMock(),
          "memory": lambda: MagicMock()
      })
      
      parent_ctx = SessionContext(session_id="parent-session")
      channel = MagicMock()
      
      # Mock the builder methods on the spawner
      spawner._create_harness = MagicMock()
      mock_harness = MagicMock()
      mock_harness.execute_turn = AsyncMock(return_value=MagicMock(
          has_more_actions=False,
          response=MagicMock(content="Subagent final result")
      ))
      spawner._create_harness.return_value = mock_harness
      
      result = await spawner.spawn_and_run(
          parent_ctx=parent_ctx,
          channel=channel,
          task="Perform analysis",
          subagent_config=HarnessConfig()
      )
      
      assert result.content == "Subagent final result"
      spawner._create_harness.assert_called_once()
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_subagent.py -v`
  Expected: FAIL (ModuleNotFoundError: No module named 'openclaw_lite.subagent')

- [ ] **Step 3: Implement subagent spawner**
  Create `openclaw_lite/subagent.py`:
  ```python
  import uuid
  from typing import Any
  from openclaw_lite.memory.base import SessionContext
  from openclaw_lite.config import HarnessConfig
  from openclaw_lite.models.base import Message
  from openclaw_lite.channels.base import BaseChannel
  from openclaw_lite.harness import AgentHarness

  class SubagentSpawner:
      def __init__(self, factory_config: dict[str, Any]):
          self.factory_config = factory_config

      def _create_harness(self, config: HarnessConfig) -> AgentHarness:
          model_client = self.factory_config["model"]()
          memory_engine = self.factory_config["memory"]()
          # Reuse or setup empty registers
          return AgentHarness(
              config=config,
              model_client=model_client,
              memory_engine=memory_engine,
              mcp_manager=None,
              skills_manager=None
          )

      async def spawn_and_run(
          self,
          parent_ctx: SessionContext,
          channel: BaseChannel,
          task: str,
          subagent_config: HarnessConfig
      ) -> Message:
          # Create isolated context
          child_ctx = SessionContext(
              session_id=f"{parent_ctx.session_id}-sub-{uuid.uuid4().hex[:6]}",
              parent_session_id=parent_ctx.session_id,
              scope={"task": task}
          )

          # Build isolated harness
          harness = self._create_harness(subagent_config)

          # Send primary task
          initial_instruction = Message(
              role="system",
              content=f"Isolated subagent task: {task}. Return final answer."
          )

          current_msg = initial_instruction
          for _ in range(subagent_config.max_consecutive_tools):
              turn_result = await harness.execute_turn(
                  session_ctx=child_ctx,
                  channel=channel,
                  user_message=current_msg
              )
              current_msg = None
              
              if not turn_result.has_more_actions:
                  return Message(role="assistant", content=turn_result.response.content or "")

          raise RuntimeError("Subagent execution exceeded maximum tool steps.")
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_subagent.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add openclaw_lite/subagent.py tests/test_subagent.py
  git commit -m "feat: implement subagent spawner tool execution"
  ```

---

### Task 7: Main Daemon Entrypoint

**Files:**
- Create: `main.py`
- Create: `config.yaml`
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration tests**
  Create `tests/test_integration.py`:
  ```python
  import pytest
  from fastapi.testclient import TestClient
  from main import app

  client = TestClient(app)

  def test_webhook_endpoint():
      # Simple post verification
      response = client.post("/webhook", json={
          "session_id": "test-session",
          "message": {"role": "user", "content": "hello"}
      })
      assert response.status_code == 200
      assert response.json()["status"] == "queued"
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_integration.py -v`
  Expected: FAIL (ModuleNotFoundError: No module named 'main')

- [ ] **Step 3: Implement main FastAPI daemon**
  Create `main.py`:
  ```python
  from fastapi import FastAPI, BackgroundTasks
  from pydantic import BaseModel
  from typing import Any
  import uvicorn

  app = FastAPI(title="Jarvis Harness Gateway Daemon")

  class WebhookPayload(BaseModel):
      session_id: str
      message: dict[str, Any]

  async def process_async_turn(session_id: str, message: dict[str, Any]):
      # Place to instantiate harness and call execute_turn
      pass

  @app.post("/webhook")
  async def handle_webhook(payload: WebhookPayload, background_tasks: BackgroundTasks):
      background_tasks.add_task(process_async_turn, payload.session_id, payload.message)
      return {"status": "queued"}

  if __name__ == "__main__":
      uvicorn.run(app, host="127.0.0.1", port=8000)
  ```

  Create `config.yaml`:
  ```yaml
  model:
    provider: anthropic
    model_name: claude-3-5-sonnet
    temperature: 0.7
  gateway:
    host: 127.0.0.1
    port: 8000
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_integration.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add main.py config.yaml tests/test_integration.py
  git commit -m "feat: add main gateway daemon and config"
  ```
