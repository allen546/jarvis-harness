# Jarvis Agent Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight, microkernel-style agent harness (`openclaw`-style) written in Python, named `jarvis`. It supports pluggable models, memory, channels, skills, and MCP servers with minimal dependencies, QQ & Discord integrations, OpenAI-compatible clients, dynamic session JSON configs, streaming, channel-side filters, and a stubbed Gemini client.

**Architecture:** A thin main FastAPI gateway daemon manages sessions and coordinates model clients, memory, channels, skills parser, and MCP registries via a direct `execute_turn` runner step. Dynamic import of model SDKs inside provider classes avoids import overhead.

**Tech Stack:** Python, `httpx`, `pydantic`, `fastapi`, `uvicorn`, `mcp`, `pyyaml`

---

### Task 1: Core Config, Session JSON, and Models Interface

**Files:**
- Create: `jarvis/config.py`
- Create: `jarvis/models/base.py`
- Create: `tests/test_config_models.py`

- [ ] **Step 1: Write the failing test for configuration loading and streaming base models**
  Create `tests/test_config_models.py`:
  ```python
  import pytest
  import os
  import json
  from jarvis.config import load_session_config, ModelConfig
  from jarvis.models.base import Attachment, NativeAction, Message, ToolCall, ModelResponse

  def test_configs_and_messages(tmp_path):
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
      action = NativeAction(action_type="react", params={"emoji": "👍"})
      msg = Message(role="user", content="Hello", attachments=[attachment], native_actions=[action])
      assert len(msg.attachments) == 1
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_config_models.py -v`
  Expected: FAIL (ModuleNotFoundError: No module named 'jarvis')

- [ ] **Step 3: Write config and base models implementation**
  Create `jarvis/config.py`:
  ```python
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
      model: ModelConfig
      harness: HarnessConfig = Field(default_factory=HarnessConfig)

  def load_session_config(session_id: str, config_dir: str = "config") -> SessionConfig:
      session_file = os.path.join(config_dir, "sessions", f"session_{session_id}.json")
      global_file = os.path.join(config_dir, "global.json")
      
      data = {}
      if os.path.exists(global_file):
          with open(global_file, "r") as f:
              data = json.load(f)
              
      if os.path.exists(session_file):
          with open(session_file, "r") as f:
              session_data = json.load(f)
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
  ```

  Create `jarvis/models/base.py`:
  ```python
  from pydantic import BaseModel, Field
  from typing import Any, Optional, AsyncGenerator

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

      async def generate_stream(self, messages: list[Message], tools: list[Any]) -> AsyncGenerator[ModelResponse, None]:
          raise NotImplementedError
          yield
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_config_models.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add jarvis/config.py jarvis/models/base.py tests/test_config_models.py
  git commit -m "feat: implement JSON config loader and base streaming models"
  ```

---

### Task 2: Model Providers & OpenAI-Compatible (Native SDKs + Gemini Stub)

**Files:**
- Create: `jarvis/models/gemini.py`
- Create: `jarvis/models/anthropic.py`
- Create: `jarvis/models/openai.py`
- Create: `jarvis/models/openai_compatible.py`
- Create: `tests/test_model_providers.py`

- [ ] **Step 1: Write tests verifying Gemini client is stubbed and others compile**
  Create `tests/test_model_providers.py`:
  ```python
  import pytest
  from jarvis.models.gemini import GeminiClient
  from jarvis.models.openai_compatible import OpenAICompatibleClient

  @pytest.mark.asyncio
  async def test_gemini_stub():
      client = GeminiClient(api_key="fake-key", model_name="gemini-1.5-flash")
      with pytest.raises(NotImplementedError):
          await client.generate([], [])

  def test_openai_compatible_client_init():
      client = OpenAICompatibleClient(api_key="fake-key", model_name="local-llama", base_url="http://localhost:8000/v1")
      assert client.base_url == "http://localhost:8000/v1"
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_model_providers.py -v`
  Expected: FAIL (ModuleNotFoundError: No module named 'jarvis.models.gemini')

- [ ] **Step 3: Implement model providers (Gemini stubbed, others stream-enabled)**
  Create `jarvis/models/gemini.py`:
  ```python
  from typing import AsyncGenerator, Any
  from jarvis.models.base import BaseModelClient, Message, ModelResponse

  class GeminiClient(BaseModelClient):
      def __init__(self, api_key: str, model_name: str):
          self.api_key = api_key
          self.model_name = model_name

      async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
          raise NotImplementedError("GeminiClient is currently stubbed and not implemented.")

      async def generate_stream(self, messages: list[Message], tools: list[Any]) -> AsyncGenerator[ModelResponse, None]:
          raise NotImplementedError("GeminiClient is currently stubbed and not implemented.")
          yield
  ```

  Create `jarvis/models/anthropic.py`:
  ```python
  import importlib
  from typing import Any, AsyncGenerator, Optional
  from jarvis.models.base import BaseModelClient, Message, ModelResponse, ToolCall

  class AnthropicClient(BaseModelClient):
      def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
          self.api_key = api_key
          self.model_name = model_name
          self.base_url = base_url

      async def _get_client(self):
          anthropic = importlib.import_module("anthropic")
          kwargs = {"api_key": self.api_key}
          if self.base_url:
              kwargs["base_url"] = self.base_url
          return anthropic.AsyncAnthropic(**kwargs)

      async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
          client = await self._get_client()
          anthropic_msgs = []
          system_prompt = None
          for m in messages:
              if m.role == "system":
                  system_prompt = m.content
              else:
                  anthropic_msgs.append({"role": "assistant" if m.role == "assistant" else "user", "content": m.content})

          kwargs = {"model": self.model_name, "messages": anthropic_msgs, "max_tokens": 1024}
          if system_prompt:
              kwargs["system"] = system_prompt
          if tools:
              kwargs["tools"] = tools

          response = await client.messages.create(**kwargs)
          content_text = "".join([c.text for c in response.content if c.type == "text"])
          tool_calls = [ToolCall(call_id=c.id, tool_name=c.name, arguments=c.input) for c in response.content if c.type == "tool_use"]
          return ModelResponse(content=content_text, tool_calls=tool_calls, raw_response=response)

      async def generate_stream(self, messages: list[Message], tools: list[Any]) -> AsyncGenerator[ModelResponse, None]:
          client = await self._get_client()
          anthropic_msgs = [{"role": "assistant" if m.role == "assistant" else "user", "content": m.content} for m in messages if m.role != "system"]
          system_prompt = next((m.content for m in messages if m.role == "system"), None)
          
          kwargs = {"model": self.model_name, "messages": anthropic_msgs, "max_tokens": 1024}
          if system_prompt:
              kwargs["system"] = system_prompt

          async with client.messages.stream(**kwargs) as stream:
              async for text in stream.text_stream:
                  yield ModelResponse(content=text, tool_calls=[], raw_response=None)
  ```

  Create `jarvis/models/openai.py`:
  ```python
  import importlib
  from typing import Any, AsyncGenerator, Optional
  from jarvis.models.base import BaseModelClient, Message, ModelResponse, ToolCall

  class OpenAIClient(BaseModelClient):
      def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
          self.api_key = api_key
          self.model_name = model_name
          self.base_url = base_url

      async def _get_client(self):
          openai = importlib.import_module("openai")
          kwargs = {"api_key": self.api_key}
          if self.base_url:
              kwargs["base_url"] = self.base_url
          return openai.AsyncOpenAI(**kwargs)

      async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
          client = await self._get_client()
          openai_msgs = [{"role": m.role, "content": m.content} for m in messages]
          kwargs = {"model": self.model_name, "messages": openai_msgs}
          if tools:
              kwargs["tools"] = [{"type": "function", "function": t} for t in tools]

          response = await client.chat.completions.create(**kwargs)
          choice = response.choices[0]
          tool_calls = []
          if choice.message.tool_calls:
              import json
              tool_calls = [ToolCall(call_id=tc.id, tool_name=tc.function.name, arguments=json.loads(tc.function.arguments)) for tc in choice.message.tool_calls]
          return ModelResponse(content=choice.message.content, tool_calls=tool_calls, raw_response=response)

      async def generate_stream(self, messages: list[Message], tools: list[Any]) -> AsyncGenerator[ModelResponse, None]:
          client = await self._get_client()
          openai_msgs = [{"role": m.role, "content": m.content} for m in messages]
          response = await client.chat.completions.create(model=self.model_name, messages=openai_msgs, stream=True)
          async for chunk in response:
              if chunk.choices and chunk.choices[0].delta.content:
                  yield ModelResponse(content=chunk.choices[0].delta.content, tool_calls=[], raw_response=chunk)
  ```

  Create `jarvis/models/openai_compatible.py`:
  ```python
  from typing import Optional
  from jarvis.models.openai import OpenAIClient

  class OpenAICompatibleClient(OpenAIClient):
      def __init__(self, api_key: str, model_name: str, base_url: str):
          super().__init__(api_key=api_key, model_name=model_name, base_url=base_url)
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_model_providers.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add jarvis/models/gemini.py jarvis/models/anthropic.py jarvis/models/openai.py jarvis/models/openai_compatible.py tests/test_model_providers.py
  git commit -m "feat: add openai compatible client, dynamic sdk imports, and streaming endpoints"
  ```

---

### Task 3: Channels (with content filtering) & Local Memory

**Files:**
- Create: `jarvis/channels/base.py`
- Create: `jarvis/channels/webhook.py`
- Create: `jarvis/channels/discord.py`
- Create: `jarvis/channels/qq.py`
- Create: `tests/test_channels.py`

- [ ] **Step 1: Write test for new channels and stream chunks with filters**
  Create `tests/test_channels.py`:
  ```python
  import pytest
  from jarvis.channels.discord import DiscordChannel
  from jarvis.channels.qq import QQChannel
  from jarvis.models.base import Message

  def test_channels_initialization():
      discord = DiscordChannel(bot_token="token123")
      qq = QQChannel(app_id="app123", app_secret="sec123")
      assert discord.bot_token == "token123"
      assert qq.app_id == "app123"

      # Test content filtering on QQ channel
      assert qq.filter_content("Now let me read main.py: Hello!") == "Hello!"
      # Test content filtering on Discord (should preserve the thoughts)
      assert discord.filter_content("Now let me read main.py: Hello!") == "Now let me read main.py: Hello!"
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_channels.py -v`
  Expected: FAIL (ModuleNotFoundError: No module named 'jarvis.channels.discord')

- [ ] **Step 3: Implement Discord, QQ, and Webhook channels with filtering**
  Create `jarvis/channels/base.py`:
  ```python
  from jarvis.models.base import Message
  from typing import Any

  class BaseChannel:
      async def send_message(self, session_id: str, message: Message):
          raise NotImplementedError

      async def send_stream_chunk(self, session_id: str, chunk: str):
          pass

      def filter_content(self, content: str) -> str:
          return content

      def get_channel_tools(self, session_id: str) -> list[Any]:
          return []
  ```

  Create `jarvis/channels/webhook.py`:
  ```python
  from jarvis.channels.base import BaseChannel
  from jarvis.models.base import Message
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

      async def send_stream_chunk(self, session_id: str, chunk: str):
          async with httpx.AsyncClient() as client:
              await client.post(self.callback_url + "/stream", text=chunk)
  ```

  Create `jarvis/channels/discord.py`:
  ```python
  import importlib
  from typing import Optional, Any
  from jarvis.channels.base import BaseChannel
  from jarvis.models.base import Message

  class DiscordChannel(BaseChannel):
      def __init__(self, bot_token: str, guild_id: Optional[str] = None):
          self.bot_token = bot_token
          self.guild_id = guild_id

      async def send_message(self, session_id: str, message: Message):
          discord = importlib.import_module("discord")
          pass

      async def send_stream_chunk(self, session_id: str, chunk: str):
          pass
  ```

  Create `jarvis/channels/qq.py`:
  ```python
  import importlib
  import re
  from jarvis.channels.base import BaseChannel
  from jarvis.models.base import Message

  class QQChannel(BaseChannel):
      def __init__(self, app_id: str, app_secret: str):
          self.app_id = app_id
          self.app_secret = app_secret

      async def send_message(self, session_id: str, message: Message):
          botpy = importlib.import_module("botpy")
          pass

      async def send_stream_chunk(self, session_id: str, chunk: str):
          pass

      def filter_content(self, content: str) -> str:
          # Filter internal monologue thought logs (e.g. "Now let me read file:")
          pattern = r"(?:Now let me read|Reading file|Executing command|Calling tool).*?:\s*"
          return re.sub(pattern, "", content)
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_channels.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add jarvis/channels/base.py jarvis/channels/webhook.py jarvis/channels/discord.py jarvis/channels/qq.py tests/test_channels.py
  git commit -m "feat: add discord and qq channels with channel-side content filtering"
  ```

---

### Task 4: Memory & Local Storage

**Files:**
- Create: `jarvis/memory/base.py`
- Create: `jarvis/memory/jsonl.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write test for local JSONL memory**
  Create `tests/test_memory.py`:
  ```python
  import pytest
  import os
  from jarvis.memory.jsonl import JSONLMemoryEngine
  from jarvis.models.base import Message
  from jarvis.memory.base import SessionContext

  @pytest.mark.asyncio
  async def test_jsonl_memory(tmp_path):
      history_file = tmp_path / "sessions.jsonl"
      engine = JSONLMemoryEngine(file_path=str(history_file))
      ctx = SessionContext(session_id="session-1")
      await engine.save_history(ctx, [Message(role="user", content="Hi")])
      loaded = await engine.load_history(ctx)
      assert len(loaded) == 1
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_memory.py -v`
  Expected: FAIL (ModuleNotFoundError: No module named 'jarvis.memory')

- [ ] **Step 3: Implement Base and JSONL memory engines**
  Create `jarvis/memory/base.py`:
  ```python
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
  ```

  Create `jarvis/memory/jsonl.py`:
  ```python
  import json
  import os
  import aiofiles
  from jarvis.memory.base import BaseMemoryEngine, SessionContext
  from jarvis.models.base import Message

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
                  line = {"session_id": context.session_id, "message": m.model_dump()}
                  await f.write(json.dumps(line) + "\n")
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_memory.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add jarvis/memory/base.py jarvis/memory/jsonl.py tests/test_memory.py
  git commit -m "feat: implement JSONL local history storage"
  ```

---

### Task 5: Core Harness Loop (with streaming and content filtering)

**Files:**
- Create: `jarvis/harness.py`
- Create: `tests/test_harness.py`

- [ ] **Step 1: Write mock test for filtered streamed execute_turn**
  Create `tests/test_harness.py`:
  ```python
  import pytest
  from unittest.mock import AsyncMock, MagicMock
  from jarvis.harness import AgentHarness, HarnessConfig, SessionContext
  from jarvis.models.base import Message, ModelResponse

  @pytest.mark.asyncio
  async def test_execute_turn_streamed_filtered():
      config = HarnessConfig(system_prompt="system instructions")
      
      model_client = MagicMock()
      async def mock_stream(msgs, tools):
          yield ModelResponse(content="Reading file: chunk1", tool_calls=[], raw_response=None)
          yield ModelResponse(content="Reading file: chunk2", tool_calls=[], raw_response=None)
      model_client.generate_stream = mock_stream
      
      memory = MagicMock()
      memory.load_history = AsyncMock(return_value=[])
      memory.save_history = AsyncMock()

      harness = AgentHarness(config, model_client, memory, MagicMock(), MagicMock())
      
      ctx = SessionContext(session_id="session-stream")
      channel = MagicMock()
      channel.send_stream_chunk = AsyncMock()
      channel.send_message = AsyncMock()
      # Stub filter stripping thoughts
      channel.filter_content = lambda x: x.replace("Reading file: ", "")

      result = await harness.execute_turn(ctx, channel, Message(role="user", content="Hello"))
      assert result.response.content == "Reading file: chunk1Reading file: chunk2"
      
      # Verify that filtered content was sent to stream
      channel.send_stream_chunk.assert_any_call("session-stream", "chunk1")
      channel.send_stream_chunk.assert_any_call("session-stream", "chunk2")
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_harness.py -v`
  Expected: FAIL (ModuleNotFoundError: No module named 'jarvis.harness')

- [ ] **Step 3: Implement execute_turn streaming and filtering logic**
  Create `jarvis/harness.py`:
  ```python
  from typing import Optional, Any, Callable, list
  from jarvis.config import HarnessConfig
  from jarvis.memory.base import SessionContext, BaseMemoryEngine
  from jarvis.models.base import BaseModelClient, Message, ToolCall, ModelResponse
  from jarvis.channels.base import BaseChannel
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
          history = await self.memory_engine.load_history(session_ctx)
          
          if not history and self.config.system_prompt:
              history.insert(0, Message(role="system", content=self.config.system_prompt))
          
          if user_message:
              history.append(user_message)
              await self.memory_engine.save_history(session_ctx, [user_message])

          for hook in self.pre_turn_hooks:
              history = await hook(session_ctx, history)

          tools = []

          # Streaming generation
          accumulated_text = ""
          final_tool_calls = []
          async for response_chunk in self.model_client.generate_stream(history, tools=tools):
              if response_chunk.content:
                  accumulated_text += response_chunk.content
                  # Filter channel content before streaming
                  filtered_chunk = channel.filter_content(response_chunk.content)
                  if filtered_chunk:
                      await channel.send_stream_chunk(session_ctx.session_id, filtered_chunk)
              if response_chunk.tool_calls:
                  final_tool_calls.extend(response_chunk.tool_calls)

          final_response = ModelResponse(content=accumulated_text, tool_calls=final_tool_calls, raw_response=None)

          for hook in self.post_message_hooks:
              await hook(session_ctx, final_response)

          # Save full raw message history
          assistant_msg = Message(role="assistant", content=accumulated_text)
          await self.memory_engine.save_history(session_ctx, [assistant_msg])
          
          # Send filtered final message to channel
          filtered_message = Message(
              role="assistant",
              content=channel.filter_content(accumulated_text),
              attachments=assistant_msg.attachments,
              native_actions=assistant_msg.native_actions,
              metadata=assistant_msg.metadata
          )
          await channel.send_message(session_ctx.session_id, filtered_message)

          tool_results = []
          
          return TurnResult(
              response=final_response,
              tool_results=tool_results,
              has_more_actions=len(final_tool_calls) > 0
          )
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_harness.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add jarvis/harness.py tests/test_harness.py
  git commit -m "feat: implement streaming runner integration in execute_turn"
  ```
