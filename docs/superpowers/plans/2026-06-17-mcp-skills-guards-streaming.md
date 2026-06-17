# MCP, Skills, Guards, Streaming, and Error Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the MCP Client, Skill Loader, Tool Approval Hook, Budget Guard Hook, Model Streaming (Text Deltas), and Error Recovery in the Jarvis microkernel.

**Architecture:** We extend the Session Context and Hooks pattern. An `McpClientManager` manages external server connections using the `ClientSessionGroup` class. A `SkillManager` loads directory-based instructions and registers script tools. The approval and budget limits are enforced via `TurnHook` checkpoints. `AgentKernel` is updated to stream responses via `generate_stream()` and accumulate chunk states, while model calls are decorated with an exponential backoff wrapper.

**Tech Stack:** Python 3.14+, `mcp` SDK, `anyio`, `httpx`, `pydantic`, `pytest`

---

### Task 1: Runtime Config and Context Extensions

**Files:**
- Modify: `jarvis/runtime.py`
- Test: `tests/test_config_models.py`

- [ ] **Step 1: Write a failing test for context config extension**
  
  Add to `tests/test_config_models.py`:
  ```python
  def test_runtime_config_extensions() -> None:
      from jarvis.runtime import RuntimeConfig
      config = RuntimeConfig(
          system_prompt="test",
          max_consecutive_tools=10,
          require_tool_approval=True,
          allowed_skills=["git"]
      )
      assert config.max_consecutive_tools == 10
      assert config.require_tool_approval is True
      assert config.allowed_skills == ["git"]
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_config_models.py -k test_runtime_config_extensions`
  Expected: FAIL (AttributeError / TypeError due to missing slots/arguments)

- [ ] **Step 3: Update `RuntimeConfig` and `context_from_config` in `jarvis/runtime.py`**
  
  Modify `jarvis/runtime.py`:
  ```python
  # Target line 19-21:
  @dataclass(slots=True)
  class RuntimeConfig:
      system_prompt: str | None = None
      max_consecutive_tools: int = 5
      require_tool_approval: bool = False
      allowed_skills: list[str] = field(default_factory=list)
  ```
  And in `context_from_config` (line 106-115), pass these fields:
  ```python
  def context_from_config(config: SessionConfig, tools: ToolRegistry, hooks: list[TurnHook] | None = None) -> AgentContext:
      provider = config.model.provider.lower()
      model_cls = get_model_class(provider)
      return AgentContext(
          config=RuntimeConfig(
              system_prompt=config.harness.system_prompt,
              max_consecutive_tools=config.harness.max_consecutive_tools,
              require_tool_approval=config.harness.require_tool_approval,
              allowed_skills=config.harness.allowed_skills,
          ),
          session=SessionState(id=config.session_id),
          model=model_cls.from_cfg(config),
          tools=tools,
          hooks=hooks if hooks is not None else _default_hooks(),
      )
  ```

- [ ] **Step 4: Run test to verify it passes**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_config_models.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  
  Run:
  ```bash
  git add jarvis/runtime.py tests/test_config_models.py
  git commit -m "feat: extend RuntimeConfig with harness parameters"
  ```

---

### Task 2: Budget Guard and Tool Approval Hooks

**Files:**
- Modify: `jarvis/hooks.py`
- Create: `tests/test_hooks_extensions.py`

- [ ] **Step 1: Write failing tests for budget and approval hooks**
  
  Create `tests/test_hooks_extensions.py`:
  ```python
  import pytest
  from jarvis.models.base import ToolCall
  from jarvis.tools import ToolResult
  from jarvis.hooks import BudgetGuardHook, ToolApprovalHook, HookResult
  
  class MockContext:
      def __init__(self, max_consec=2, require_approval=True):
          from types import SimpleNamespace
          self.config = SimpleNamespace(
              max_consecutive_tools=max_consec,
              require_tool_approval=require_approval
          )
          self.approval_handler = None
  
  @pytest.mark.asyncio
  async def test_budget_guard_hook() -> None:
      hook = BudgetGuardHook()
      ctx = MockContext(max_consec=2)
      call = ToolCall(call_id="c1", tool_name="ls", arguments={})
      
      # First call should succeed
      r1 = await hook.before_tool(ctx, call)
      assert r1.stop is False
      
      # Simulate turn model loop by informing the hook of tool progress
      await hook.after_tool(ctx, call, ToolResult("c1", "ls", "ok"))
      
      # Second call should succeed
      r2 = await hook.before_tool(ctx, call)
      assert r2.stop is False
      await hook.after_tool(ctx, call, ToolResult("c2", "ls", "ok"))
      
      # Third call should hit budget limit
      r3 = await hook.before_tool(ctx, call)
      assert r3.stop is True
      assert "limit" in r3.reason.lower()
  
  @pytest.mark.asyncio
  async def test_tool_approval_hook() -> None:
      hook = ToolApprovalHook()
      ctx = MockContext(require_approval=True)
      call = ToolCall(call_id="c1", tool_name="Bash", arguments={"command": "rm -rf"})
      
      # Rejects when handler returns False
      ctx.approval_handler = lambda tc: False
      res_reject = await hook.before_tool(ctx, call)
      assert res_reject.skip_tool is True
      
      # Accepts when handler returns True
      ctx.approval_handler = lambda tc: True
      res_accept = await hook.before_tool(ctx, call)
      assert res_accept.skip_tool is False
  ```

- [ ] **Step 2: Run tests to verify they fail**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_hooks_extensions.py`
  Expected: FAIL (ImportError / NameError for missing Hooks)

- [ ] **Step 3: Implement `BudgetGuardHook` and `ToolApprovalHook` in `jarvis/hooks.py`**
  
  Add to `jarvis/hooks.py`:
  ```python
  class BudgetGuardHook(NoopTurnHook):
      __slots__ = ("_counts",)
      
      def __init__(self) -> None:
          self._counts: dict[int, int] = {}
          
      async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
          # Reset budget counter at the start of a turn (before model runs)
          session = getattr(ctx, "session")
          self._counts[id(session)] = 0
          return HookResult()
          
      async def before_tool(self, ctx: object, tool_call: ToolCall) -> HookResult:
          session = getattr(ctx, "session")
          config = getattr(ctx, "config")
          count = self._counts.get(id(session), 0)
          limit = getattr(config, "max_consecutive_tools", 5)
          if count >= limit:
              return HookResult(stop=True, reason=f"Tool execution budget exceeded: max {limit} consecutive calls")
          return HookResult()
          
      async def after_tool(self, ctx: object, tool_call: ToolCall, result: ToolResult) -> HookResult:
          session = getattr(ctx, "session")
          self._counts[id(session)] = self._counts.get(id(session), 0) + 1
          return HookResult()
  
  
  class ToolApprovalHook(NoopTurnHook):
      __slots__ = ()
      
      async def before_tool(self, ctx: object, tool_call: ToolCall) -> HookResult:
          config = getattr(ctx, "config")
          require_approval = getattr(config, "require_tool_approval", False)
          if not require_approval:
              return HookResult()
              
          handler = getattr(ctx, "approval_handler", None)
          if handler is None:
              # If approval is required but no handler is registered, fail safe (reject tool call)
              return HookResult(skip_tool=True, reason="Tool approval required but no handler registered")
              
          import inspect
          approved = handler(tool_call)
          if inspect.isawaitable(approved):
              approved = await approved
              
          if not approved:
              return HookResult(skip_tool=True, reason="Tool call rejected by user")
          return HookResult()
  ```
  Also update `_default_hooks()` in `jarvis/runtime.py` to include these hooks:
  ```python
  def _default_hooks() -> list[TurnHook]:
      hooks: list[TurnHook] = [
          JSONLHistoryHook(),
          ContextCompressionHook(),
          BudgetGuardHook(),
          ToolApprovalHook()
      ]
      embedding_url = os.environ.get("EMBEDDING_URL", "")
      # ...
  ```

- [ ] **Step 4: Run tests to verify they pass**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_hooks_extensions.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  
  Run:
  ```bash
  git add jarvis/hooks.py jarvis/runtime.py tests/test_hooks_extensions.py
  git commit -m "feat: implement budget guard and tool approval hooks"
  ```

---

### Task 3: MCP Client Integration

**Files:**
- Create: `jarvis/mcp.py`
- Modify: `jarvis/runtime.py`, `jarvis/tools.py`
- Create: `tests/test_mcp_client.py`

- [ ] **Step 1: Write failing tests for MCP client registration and routing**
  
  Create `tests/test_mcp_client.py`:
  ```python
  import pytest
  import json
  from pathlib import Path
  from jarvis.runtime import AgentSession
  from jarvis.tools import ToolRegistry, ToolCall
  
  # A simple mock MCP settings block to connect to a dummy command
  MOCK_MCP_SETTINGS = {
      "mcpServers": {
          "mock_server": {
              "command": "python",
              "args": ["-c", "import sys; print('mock tool init')"]
          }
      }
  }
  
  @pytest.mark.asyncio
  async def test_mcp_settings_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
      from jarvis.mcp import McpClientManager
      config_file = tmp_path / "mcp_settings.json"
      config_file.write_text(json.dumps(MOCK_MCP_SETTINGS))
      
      # Mock the connect call so we don't spawn subprocesses in this test
      class MockGroup:
          def __init__(self):
              from types import SimpleNamespace
              self.tools = {
                  "mcp_hello": SimpleNamespace(
                      name="mcp_hello",
                      description="test tool",
                      inputSchema={"type": "object", "properties": {}}
                  )
              }
          async def __aenter__(self): return self
          async def __aexit__(self, *args): pass
          async def connect_to_server(self, params): pass
          async def call_tool(self, name, args):
              from types import SimpleNamespace
              return SimpleNamespace(content=[SimpleNamespace(text="mcp response")], isError=False)
      
      monkeypatch.setattr("jarvis.mcp.ClientSessionGroup", MockGroup)
      
      manager = McpClientManager(config_path=str(config_file))
      tools = await manager.initialize()
      assert len(tools) == 1
      assert tools[0].name == "mcp_hello"
      
      res = await tools[0].handler({})
      assert res == "mcp response"
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_mcp_client.py`
  Expected: FAIL (ModuleNotFoundError / ImportError for `jarvis/mcp.py`)

- [ ] **Step 3: Create `jarvis/mcp.py`**
  
  Create `jarvis/mcp.py` containing the `McpClientManager`:
  ```python
  import json
  import os
  from typing import Any
  from mcp import StdioServerParameters
  from mcp.client.session_group import ClientSessionGroup, SseServerParameters
  from jarvis.tools import Tool
  
  class McpClientManager:
      def __init__(self, config_path: str = "config/mcp_settings.json") -> None:
          self.config_path = config_path
          self.group: ClientSessionGroup | None = None
          
      async def initialize(self) -> list[Tool]:
          if not os.path.exists(self.config_path):
              return []
              
          with open(self.config_path, "r", encoding="utf-8") as f:
              try:
                  data = json.load(f)
              except Exception:
                  return []
                  
          servers = data.get("mcpServers", {})
          if not servers:
              return []
              
          self.group = ClientSessionGroup()
          await self.group.__aenter__()
          
          for name, cfg in servers.items():
              try:
                  if "url" in cfg:
                      params = SseServerParameters(
                          url=cfg["url"],
                          headers=cfg.get("headers"),
                          timeout=cfg.get("timeout", 5.0),
                          sse_read_timeout=cfg.get("sse_read_timeout", 300.0)
                      )
                  else:
                      params = StdioServerParameters(
                          command=cfg["command"],
                          args=cfg.get("args", []),
                          env=cfg.get("env", None),
                          cwd=cfg.get("cwd", None)
                      )
                  await self.group.connect_to_server(params)
              except Exception as exc:
                  # Log failure to connect to this server, but continue with others
                  print(f"Failed to connect to MCP server {name}: {exc}")
                  
          jarvis_tools: list[Tool] = []
          for tool_name, mcp_tool in self.group.tools.items():
              jarvis_tools.append(Tool(
                  name=tool_name,
                  description=mcp_tool.description or "",
                  parameters=mcp_tool.inputSchema,
                  handler=lambda args, name=tool_name: self.execute_tool(name, args)
              ))
          return jarvis_tools
          
      async def execute_tool(self, name: str, args: dict[str, Any]) -> str:
          if not self.group:
              raise RuntimeError("MCP Client Group not initialized")
          res = await self.group.call_tool(name, args)
          
          parts = []
          for block in res.content:
              if hasattr(block, "text") and block.text:
                  parts.append(block.text)
              elif hasattr(block, "data") and block.data:
                  parts.append(block.data)
              else:
                  parts.append(str(block))
          text_result = "\n".join(parts)
          
          if res.isError:
              raise ValueError(text_result)
          return text_result
          
      async def close(self) -> None:
          if self.group:
              await self.group.__aexit__(None, None, None)
              self.group = None
  ```

- [ ] **Step 4: Update `AgentSession` in `jarvis/runtime.py` and `AgentContext`**
  
  Add properties to `AgentContext` and initialize `McpClientManager` in `AgentSession.submit`.
  In `jarvis/runtime.py`, add `mcp_manager` field to `AgentContext`:
  ```python
  # Target line 31-38:
  @dataclass(slots=True)
  class AgentContext:
      config: RuntimeConfig
      session: SessionState
      model: BaseModelClient
      tools: ToolRegistry
      hooks: list[TurnHook] = field(default_factory=list)
      emit_event: Callable[[AgentEvent], None] | None = None
      mcp_manager: Any | None = field(default=None, compare=False)
      approval_handler: Any | None = field(default=None, compare=False)
  ```
  In `AgentSession.__init__`:
  ```python
  class AgentSession:
      def __init__(self, ctx: AgentContext, kernel: object) -> None:
          self.ctx = ctx
          self.kernel = kernel
          self._lock = asyncio.Lock()
          self._mcp_initialized = False
  ```
  In `AgentSession.submit`:
  ```python
      async def submit(self, message: Message) -> AsyncIterator[AgentEvent]:
          async with self._lock:
              if not self._mcp_initialized:
                  from jarvis.mcp import McpClientManager
                  manager = McpClientManager()
                  mcp_tools = await manager.initialize()
                  for t in mcp_tools:
                      self.ctx.tools.register(t)
                  self.ctx.mcp_manager = manager
                  self._mcp_initialized = True
              # existing submit logic...
  ```
  Add `close` method to `AgentSession`:
  ```python
      async def close(self) -> None:
          async with self._lock:
              if self.ctx.mcp_manager:
                  await self.ctx.mcp_manager.close()
                  self.ctx.mcp_manager = None
                  self._mcp_initialized = False
  ```

- [ ] **Step 5: Run tests to verify they pass**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_mcp_client.py`
  Expected: PASS

- [ ] **Step 6: Commit**
  
  Run:
  ```bash
  git add jarvis/mcp.py jarvis/runtime.py tests/test_mcp_client.py
  git commit -m "feat: integrate MCP ClientSessionGroup into session runtime"
  ```

---

### Task 4: Skill Loader

**Files:**
- Create: `jarvis/skills.py`
- Modify: `jarvis/runtime.py`
- Create: `tests/test_skills.py`

- [ ] **Step 1: Write failing tests for parsing skills and executing script tools**
  
  Create `tests/test_skills.py`:
  ```python
  import pytest
  import shutil
  from pathlib import Path
  from jarvis.runtime import AgentContext, RuntimeConfig, SessionState
  from jarvis.tools import ToolRegistry
  from jarvis.hooks import HookResult
  from jarvis.models.base import Message
  
  MOCK_SKILL_MD = """---
  name: mock_git
  description: Mock git commands
  tools:
    mock_commit:
      description: commit files
      script: scripts/commit.sh
      parameters:
        type: object
        properties:
          message:
            type: string
        required: [message]
  ---
  Always commit files with clean messages.
  """
  
  @pytest.mark.asyncio
  async def test_skills_loader(tmp_path: Path) -> None:
      from jarvis.skills import SkillManager
      skill_dir = tmp_path / "skills" / "mock_git"
      skill_dir.mkdir(parents=True)
      (skill_dir / "SKILL.md").write_text(MOCK_SKILL_MD, encoding="utf-8")
      
      scripts_dir = skill_dir / "scripts"
      scripts_dir.mkdir()
      script_file = scripts_dir / "commit.sh"
      script_file.write_text("#!/bin/sh\necho \"committed: $message\"", encoding="utf-8")
      script_file.chmod(0o755)
      
      manager = SkillManager(skills_root=str(tmp_path / "skills"))
      
      # Mock context with allowed skill
      from types import SimpleNamespace
      ctx = SimpleNamespace(
          config=SimpleNamespace(allowed_skills=["mock_git"]),
          tools=ToolRegistry()
      )
      
      skills = await manager.load_allowed_skills(ctx)
      assert len(skills) == 1
      assert skills[0].name == "mock_git"
      assert "committed: $message" in skills[0].tools["mock_commit"]["script"]
      
      # Test prompt hook
      from jarvis.hooks import SkillInstructionsHook
      hook = SkillInstructionsHook(skills)
      msgs = [Message(role="system", content="System:")]
      res = await hook.before_model(ctx, msgs)
      assert "Always commit files with clean messages." in res.messages[0].content
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_skills.py`
  Expected: FAIL (ModuleNotFoundError / ImportErrors)

- [ ] **Step 3: Create `jarvis/skills.py`**
  
  Create `jarvis/skills.py` to parse skill folders:
  ```python
  import os
  import yaml
  import asyncio
  from dataclasses import dataclass
  from pathlib import Path
  from typing import Any
  from jarvis.tools import Tool
  from jarvis.hooks import NoopTurnHook, HookResult
  from jarvis.models.base import Message
  
  @dataclass
  class LoadedSkill:
      name: str
      instructions: str
      tools: dict[str, Any]
      dir_path: Path
  
  class SkillManager:
      def __init__(self, skills_root: str = "skills") -> None:
          self.skills_root = Path(skills_root)
          
      async def load_allowed_skills(self, ctx: Any) -> list[LoadedSkill]:
          allowed = getattr(ctx.config, "allowed_skills", [])
          if not allowed or not self.skills_root.exists():
              return []
              
          loaded_skills: list[LoadedSkill] = []
          for entry in self.skills_root.iterdir():
              if not entry.is_dir():
                  continue
              if entry.name not in allowed:
                  continue
                  
              skill_file = entry / "SKILL.md"
              if not skill_file.exists():
                  continue
                  
              content = skill_file.read_text(encoding="utf-8")
              parts = content.split("---", 2)
              if len(parts) < 3:
                  continue
                  
              metadata = yaml.safe_load(parts[1])
              instructions = parts[2].strip()
              name = metadata.get("name", entry.name)
              tools = metadata.get("tools", {})
              
              loaded = LoadedSkill(name=name, instructions=instructions, tools=tools, dir_path=entry)
              loaded_skills.append(loaded)
              
              # Register skill-backed tools
              for t_name, t_cfg in tools.items():
                  script_path = t_cfg.get("script")
                  desc = t_cfg.get("description", "")
                  params = t_cfg.get("parameters", {"type": "object", "properties": {}})
                  
                  tool = Tool(
                      name=t_name,
                      description=desc,
                      parameters=params,
                      handler=self._create_tool_handler(entry, script_path)
                  )
                  ctx.tools.register(tool)
                  
          return loaded_skills
          
      def _create_tool_handler(self, skill_dir: Path, script_rel: str) -> Any:
          async def handler(args: dict[str, Any]) -> str:
              script_path = (skill_dir / script_rel).resolve()
              
              # Pass arguments as environment variables
              env = os.environ.copy()
              for k, v in args.items():
                  env[str(k)] = str(v)
                  
              # Spawn script subprocess
              proc = await asyncio.create_subprocess_exec(
                  str(script_path),
                  stdout=asyncio.subprocess.PIPE,
                  stderr=asyncio.subprocess.STDOUT,
                  env=env,
                  cwd=str(skill_dir)
              )
              stdout, _ = await proc.communicate()
              return stdout.decode("utf-8", errors="replace")
          return handler
  
  class SkillInstructionsHook(NoopTurnHook):
      __slots__ = ("_skills",)
      
      def __init__(self, skills: list[LoadedSkill]) -> None:
          self._skills = skills
          
      async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
          if not self._skills:
              return HookResult()
              
          # Append skill instructions to system prompt
          skill_prompts = []
          for s in self._skills:
              skill_prompts.append(f"### Skill: {s.name}\n{s.instructions}")
          instructions_text = "\n\n".join(skill_prompts)
          
          new_msgs = list(messages)
          if new_msgs and new_msgs[0].role == "system":
              orig_sys = new_msgs[0].content
              new_msgs[0] = Message(role="system", content=f"{orig_sys}\n\n{instructions_text}")
          else:
              new_msgs.insert(0, Message(role="system", content=instructions_text))
              
          return HookResult(messages=new_msgs)
  ```

- [ ] **Step 4: Hook `SkillManager` into `AgentSession.submit`**
  
  In `jarvis/runtime.py`:
  ```python
  # Modify submit to load allowed skills on initialization
      async def submit(self, message: Message) -> AsyncIterator[AgentEvent]:
          async with self._lock:
              if not self._mcp_initialized:
                  # Initialize Skills
                  from jarvis.skills import SkillManager, SkillInstructionsHook
                  skill_mgr = SkillManager()
                  skills = await skill_mgr.load_allowed_skills(self.ctx)
                  if skills:
                      self.ctx.hooks.append(SkillInstructionsHook(skills))
                      
                  # Initialize MCP Client
                  from jarvis.mcp import McpClientManager
                  # ... (keep existing MCP initialization)
  ```

- [ ] **Step 5: Run tests to verify they pass**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_skills.py`
  Expected: PASS

- [ ] **Step 6: Commit**
  
  Run:
  ```bash
  git add jarvis/skills.py jarvis/runtime.py tests/test_skills.py
  git commit -m "feat: implement skill loader and script-backed tools"
  ```

---

### Task 5: Model Response Streaming and Delta Event Emission

**Files:**
- Modify: `jarvis/kernel.py`, `jarvis/models/openai.py`, `jarvis/models/anthropic.py`
- Create: `tests/test_kernel_streaming.py`

- [ ] **Step 1: Write failing tests for streaming text deltas**
  
  Create `tests/test_kernel_streaming.py`:
  ```python
  import pytest
  from typing import AsyncGenerator
  from jarvis.runtime import AgentContext, RuntimeConfig, SessionState
  from jarvis.tools import ToolRegistry
  from jarvis.kernel import AgentKernel
  from jarvis.events import TextDeltaEvent, MessageEvent
  from jarvis.models.base import BaseModelClient, Message, ModelResponse
  
  class MockStreamingModel(BaseModelClient):
      @classmethod
      def from_cfg(cls, cfg): return cls()
      
      async def generate(self, messages, tools):
          return ModelResponse(content="completed response")
          
      async def generate_stream(self, messages, tools) -> AsyncGenerator[ModelResponse, None]:
          yield ModelResponse(content="hello ")
          yield ModelResponse(content="streaming world")
  
  @pytest.mark.asyncio
  async def test_kernel_run_turn_streams_events() -> None:
      ctx = AgentContext(
          config=RuntimeConfig(),
          session=SessionState(id="s1"),
          model=MockStreamingModel(),
          tools=ToolRegistry()
      )
      kernel = AgentKernel()
      
      events = []
      async for event in kernel.run_turn(ctx, Message(role="user", content="hi")):
          events.append(event)
          
      # Verify text deltas were yielded
      deltas = [ev.content for ev in events if isinstance(ev, TextDeltaEvent)]
      assert deltas == ["hello ", "streaming world"]
      
      # Verify final message event has the full accumulated content
      msg_events = [ev.message.content for ev in events if isinstance(ev, MessageEvent)]
      assert msg_events == ["hello streaming world"]
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_kernel_streaming.py`
  Expected: FAIL (TextDeltaEvents not emitted/assert failures)

- [ ] **Step 3: Update `AgentKernel` inside `jarvis/kernel.py` to stream model generation**
  
  Modify `jarvis/kernel.py` around line 25:
  ```python
  # Replace: response = await ctx.model.generate(messages, ctx.tools.schemas())
  # With streaming:
  ```
  Let's replace `run_turn` block with:
  ```python
                  # Instead of one-shot ctx.model.generate, we stream:
                  accumulated_content = ""
                  accumulated_tool_calls = []
                  
                  # We call generate_stream:
                  async for chunk in ctx.model.generate_stream(messages, ctx.tools.schemas()):
                      if chunk.content:
                          accumulated_content += chunk.content
                          yield TextDeltaEvent(session_id=ctx.session.id, content=chunk.content)
                      if chunk.tool_calls:
                          accumulated_tool_calls.extend(chunk.tool_calls)
                          
                  response = ModelResponse(
                      content=accumulated_content if accumulated_content else None,
                      tool_calls=accumulated_tool_calls,
                      raw_response=None
                  )
  ```
  Wait! Let's verify that we import `TextDeltaEvent` at the top of `jarvis/kernel.py` (line 5 already has `TextDeltaEvent`). Yes!

- [ ] **Step 4: Implement `generate_stream` in OpenAI and Anthropic clients**
  
  Open `jarvis/models/openai.py`. Update `generate_stream`:
  ```python
      async def generate_stream(self, messages: list[Message], tools: list[Any]) -> AsyncGenerator[ModelResponse, None]:
          client = await self._get_client()
          openai_msgs: list[dict[str, Any]] = []
          for m in messages:
              msg: dict[str, Any] = {"role": m.role, "content": m.content}
              if m.role == "assistant" and m.metadata and "tool_calls" in m.metadata:
                  msg["tool_calls"] = [
                      {"id": tc["call_id"], "type": "function", "function": {"name": tc["tool_name"], "arguments": json.dumps(tc["arguments"])}}
                      for tc in m.metadata["tool_calls"]
                  ]
              elif m.role == "tool" and m.metadata:
                  if "tool_call_id" in m.metadata:
                      msg["tool_call_id"] = m.metadata["tool_call_id"]
              openai_msgs.append(msg)
          
          is_thinking_enabled = False
          if "thinking" in self.extra_params:
              thinking_val = self.extra_params["thinking"]
              if thinking_val != "disabled" and not (isinstance(thinking_val, dict) and thinking_val.get("type") == "disabled"):
                  is_thinking_enabled = True
  
          kwargs: dict[str, Any] = {
              "model": self.model_name,
              "messages": openai_msgs,
              "stream": True
          }
          if not is_thinking_enabled:
              kwargs["temperature"] = self.temperature
              
          if self.max_tokens is not None:
              kwargs["max_tokens"] = self.max_tokens
          if tools:
              kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
          if self.extra_params:
              extra_body = {k: v for k, v in self.extra_params.items() if k not in ("temperature", "max_tokens", "tools", "model", "messages", "stream")}
              if extra_body:
                  kwargs["extra_body"] = extra_body
  
          response = await client.chat.completions.create(**kwargs)
          
          # Accumulate tool calls over chunks
          # In OpenAI API, tool call arguments are streamed in fragments
          tool_calls_builder = {}
          
          async for chunk in response:
              if not chunk.choices:
                  continue
              delta = chunk.choices[0].delta
              
              content = delta.content or ""
              tool_calls = []
              
              if delta.tool_calls:
                  for tc_delta in delta.tool_calls:
                      idx = tc_delta.index
                      if idx not in tool_calls_builder:
                          tool_calls_builder[idx] = {
                              "call_id": tc_delta.id or "",
                              "tool_name": tc_delta.function.name or "",
                              "arguments": ""
                          }
                      else:
                          if tc_delta.id:
                              tool_calls_builder[idx]["call_id"] = tc_delta.id
                          if tc_delta.function and tc_delta.function.name:
                              tool_calls_builder[idx]["tool_name"] = tc_delta.function.name
                              
                      if tc_delta.function and tc_delta.function.arguments:
                          tool_calls_builder[idx]["arguments"] += tc_delta.function.arguments
                          
              if content or delta.tool_calls:
                  # Parse any completed tool call schemas if finished or return empty
                  yield ModelResponse(content=content if content else None, tool_calls=[], raw_response=chunk)
                  
          # Emit the accumulated tool calls at the end of the stream
          final_tool_calls = []
          for idx, tc in tool_calls_builder.items():
              try:
                  args = json.loads(tc["arguments"]) if tc["arguments"] else {}
              except json.JSONDecodeError:
                  args = {}
              final_tool_calls.append(ToolCall(call_id=tc["call_id"], tool_name=tc["tool_name"], arguments=args))
              
          if final_tool_calls:
              yield ModelResponse(content=None, tool_calls=final_tool_calls, raw_response=None)
  ```
  
  Now open `jarvis/models/anthropic.py`. Update `generate_stream`:
  ```python
      async def generate_stream(self, messages: list[Message], tools: list[Any]) -> AsyncGenerator[ModelResponse, None]:
          client = await self._get_client()
          anthropic_msgs = [{"role": "assistant" if m.role == "assistant" else "user", "content": m.content} for m in messages if m.role != "system"]
          system_prompt = next((m.content for m in messages if m.role == "system"), None)
          
          kwargs: dict[str, Any] = {
              "model": self.model_name,
              "messages": anthropic_msgs,
              "max_tokens": self.max_tokens,
              "temperature": self.temperature
          }
          if system_prompt:
              kwargs["system"] = system_prompt
          if tools:
              kwargs["tools"] = tools
  
          # Let's accumulate tool use events as well
          tool_calls_builder = {}
          
          async with client.messages.stream(**kwargs) as stream:
              async for event in stream:
                  # Anthropic client stream events can be parsed
                  # E.g. content block start, delta, etc.
                  # Standard stream.text_stream only yields text.
                  # To capture tools, we use raw event stream or helper
                  if event.type == "content_block_start" and event.content_block.type == "tool_use":
                      tb = event.content_block
                      tool_calls_builder[event.index] = {
                          "call_id": tb.id,
                          "tool_name": tb.name,
                          "arguments": ""
                      }
                  elif event.type == "content_block_delta" and event.delta.type == "input_json_delta":
                      tool_calls_builder[event.index]["arguments"] += event.delta.partial_json
                  elif event.type == "content_block_delta" and event.delta.type == "text_delta":
                      yield ModelResponse(content=event.delta.text, tool_calls=[], raw_response=None)
                      
          # Emit accumulated tool calls at the end
          final_tool_calls = []
          for idx, tc in tool_calls_builder.items():
              try:
                  args = json.loads(tc["arguments"]) if tc["arguments"] else {}
              except json.JSONDecodeError:
                  args = {}
              final_tool_calls.append(ToolCall(call_id=tc["call_id"], tool_name=tc["tool_name"], arguments=args))
              
          if final_tool_calls:
              yield ModelResponse(content=None, tool_calls=final_tool_calls, raw_response=None)
  ```

- [ ] **Step 5: Run tests to verify they pass**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_kernel_streaming.py`
  Expected: PASS

- [ ] **Step 6: Commit**
  
  Run:
  ```bash
  git add jarvis/kernel.py jarvis/models/openai.py jarvis/models/anthropic.py tests/test_kernel_streaming.py
  git commit -m "feat: enable kernel model response streaming and delta events"
  ```

---

### Task 6: Error Recovery with Backoff Retries

**Files:**
- Create: `jarvis/retry.py`
- Modify: `jarvis/models/openai.py`, `jarvis/models/anthropic.py`
- Create: `tests/test_error_recovery.py`

- [ ] **Step 1: Write failing tests for retry logic on transient errors**
  
  Create `tests/test_error_recovery.py`:
  ```python
  import pytest
  from jarvis.retry import retry_with_backoff
  
  counter = 0
  
  @pytest.mark.asyncio
  async def test_retry_eventual_success() -> None:
      global counter
      counter = 0
      
      @retry_with_backoff(max_retries=3, base_delay=0.01)
      async def unstable_api():
          global counter
          counter += 1
          if counter < 3:
              raise ValueError("Transient error")
          return "success"
          
      res = await unstable_api()
      assert res == "success"
      assert counter == 3
      
  @pytest.mark.asyncio
  async def test_retry_ultimate_failure() -> None:
      @retry_with_backoff(max_retries=2, base_delay=0.01)
      async def broken_api():
          raise ValueError("Fatal error")
          
      with pytest.raises(ValueError):
          await broken_api()
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_error_recovery.py`
  Expected: FAIL (ModuleNotFoundError / ImportErrors)

- [ ] **Step 3: Create `jarvis/retry.py`**
  
  Create `jarvis/retry.py`:
  ```python
  import asyncio
  import random
  import functools
  from typing import Callable, Any
  
  def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0) -> Callable:
      def decorator(func: Callable) -> Callable:
          @functools.wraps(func)
          async def wrapper(*args, **kwargs) -> Any:
              delay = base_delay
              last_exc = None
              for attempt in range(max_retries + 1):
                  try:
                      return await func(*args, **kwargs)
                  except Exception as exc:
                      last_exc = exc
                      # Identify transient errors to retry:
                      # We retry on:
                      # - openai.RateLimitError / openai.InternalServerError / openai.APIConnectionError
                      # - anthropic.RateLimitError / anthropic.InternalServerError / anthropic.APIConnectionError
                      # - ConnectionError / TimeoutError / httpx.HTTPError
                      # - generic 5xx / 429 errors from standard APIs
                      exc_name = type(exc).__name__
                      is_transient = (
                          "RateLimit" in exc_name or
                          "InternalServer" in exc_name or
                          "APIConnection" in exc_name or
                          isinstance(exc, (ConnectionError, TimeoutError)) or
                          "Timeout" in exc_name or
                          "HTTPError" in exc_name or
                          "HTTPStatusError" in exc_name or
                          getattr(exc, "status_code", 0) in (429, 500, 502, 503, 504)
                      )
                      if not is_transient or attempt == max_retries:
                          raise exc
                      
                      # Exponential backoff with jitter
                      jitter = random.uniform(0, 0.1 * delay)
                      await asyncio.sleep(delay + jitter)
                      delay *= 2
              raise last_exc
          return wrapper
      return decorator
  ```

- [ ] **Step 4: Apply retry decorator to OpenAI and Anthropic clients**
  
  In `jarvis/models/openai.py`:
  ```python
  # Decorate generate and generate_stream
  from jarvis.retry import retry_with_backoff
  
  # Update OpenAIClient.generate:
      @retry_with_backoff(max_retries=3, base_delay=1.0)
      async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
          # ... existing code
  ```
  Wait! Since `generate_stream` returns an async generator, we cannot apply a standard async function decorator to it directly because it returns a generator object instantly.
  Instead, we can wrap the generator's internal loop or apply a retry around the initialization of the stream.
  Let's wrap the streaming request:
  ```python
      # Inside generate_stream:
      @retry_with_backoff(max_retries=3, base_delay=1.0)
      async def _get_stream():
          return await client.chat.completions.create(**kwargs)
      
      response = await _get_stream()
  ```
  Apply the same pattern to `jarvis/models/anthropic.py` for both `generate` and `generate_stream`.

- [ ] **Step 5: Run tests to verify they pass**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_error_recovery.py`
  Expected: PASS

- [ ] **Step 6: Commit**
  
  Run:
  ```bash
  git add jarvis/retry.py jarvis/models/openai.py jarvis/models/anthropic.py tests/test_error_recovery.py
  git commit -m "feat: add backoff retries for transient model failures"
  ```
