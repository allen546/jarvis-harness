# MCP, Skills, Guards, Streaming, and Error Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the MCP Client, Skill Loader, Tool Approval Hook, Budget Guard Hook, Configurable Model Streaming (Text Deltas with Fallback), and SDK-safe Error Recovery in the Jarvis microkernel.

**Architecture:** We extend the Session Context and Hooks pattern. An `McpClientManager` manages external server connections using the `ClientSessionGroup` class. A `SkillManager` loads directory-based instructions and registers script tools. The approval and budget limits are enforced via `TurnHook` checkpoints. `AgentKernel` is updated to stream responses via `generate_stream()` if enabled and supported, falling back to `generate()` as needed, while model calls are decorated with a dynamic SDK-type-safe exponential backoff wrapper.

**Tech Stack:** Python 3.14+, `mcp` SDK, `anyio`, `httpx`, `pydantic`, `pytest`, `pyyaml` (added to runtime dependencies)

---

### Task 1: Runtime Config and Context Extensions

**Files:**
- Modify: `pyproject.toml` (add `pyyaml` to runtime dependencies)
- Modify: `jarvis/config.py` (add `stream` to `HarnessConfig`)
- Modify: `jarvis/runtime.py` (extend `RuntimeConfig` and context mapping)
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
          allowed_skills=["git"],
          stream=False
      )
      assert config.max_consecutive_tools == 10
      assert config.require_tool_approval is True
      assert config.allowed_skills == ["git"]
      assert config.stream is False
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_config_models.py -k test_runtime_config_extensions`
  Expected: FAIL (AttributeError / TypeError due to missing slots/arguments)

- [ ] **Step 3: Update dependencies in `pyproject.toml`**
  
  Add `"pyyaml>=6.0.3"` to `dependencies` array under `[project]` (line 7-12) so it is available at runtime.

- [ ] **Step 4: Update `HarnessConfig` in `jarvis/config.py`**
  
  Add `stream: bool = True` to `HarnessConfig` (line 13-18):
  ```python
  class HarnessConfig(BaseModel):
      system_prompt: Optional[str] = None
      max_consecutive_tools: int = 5
      require_tool_approval: bool = False
      allowed_skills: list[str] = Field(default_factory=list)
      stream: bool = True
  ```

- [ ] **Step 5: Update `RuntimeConfig` and `context_from_config` in `jarvis/runtime.py`**
  
  Modify `jarvis/runtime.py`:
  ```python
  @dataclass(slots=True)
  class RuntimeConfig:
      system_prompt: str | None = None
      max_consecutive_tools: int = 5
      require_tool_approval: bool = False
      allowed_skills: list[str] = field(default_factory=list)
      stream: bool = True
  ```
  And in `context_from_config`, pass these fields:
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
              stream=config.harness.stream,
          ),
          session=SessionState(id=config.session_id),
          model=model_cls.from_cfg(config),
          tools=tools,
          hooks=hooks if hooks is not None else _default_hooks(),
      )
  ```

- [ ] **Step 6: Run test to verify it passes**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_config_models.py`
  Expected: PASS

- [ ] **Step 7: Commit**
  
  Run:
  ```bash
  git add pyproject.toml jarvis/config.py jarvis/runtime.py tests/test_config_models.py
  git commit -m "feat: add pyyaml to runtime deps and extend configs with stream"
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
      
      # Reset counter for a mock session
      from types import SimpleNamespace
      session = SimpleNamespace(id="s1")
      ctx.session = session
      await hook.before_model(ctx, [])
      
      # First call should succeed
      r1 = await hook.before_tool(ctx, call)
      assert r1.stop is False
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
              return HookResult(skip_tool=True, reason="Tool approval required but no handler registered")
              
          import inspect
          approved = handler(tool_call)
          if inspect.isawaitable(approved):
              approved = await approved
              
          if not approved:
              return HookResult(skip_tool=True, reason="Tool call rejected by user")
          return HookResult()
  ```
  Update `_default_hooks()` in `jarvis/runtime.py`:
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
- Modify: `jarvis/runtime.py`
- Create: `tests/test_mcp_client.py`

*Note on gap-less lazy initialization:* The MCP client manager is initialized inside `AgentSession.submit` *before* the session executes `run_turn()`. This guarantees that MCP tools are dynamically retrieved and added to the `ToolRegistry` before the model's tool schema is generated for the first model call.

- [ ] **Step 1: Write failing tests for MCP client registration and error handling**
  
  Create `tests/test_mcp_client.py`:
  ```python
  import pytest
  import json
  from pathlib import Path
  from jarvis.runtime import AgentSession
  from jarvis.tools import ToolRegistry, ToolCall
  
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
              # Mock standard error block propagation
              if name == "mcp_error_tool":
                  return SimpleNamespace(content=[SimpleNamespace(text="mcp error msg")], isError=True)
              return SimpleNamespace(content=[SimpleNamespace(text="mcp response")], isError=False)
      
      monkeypatch.setattr("jarvis.mcp.ClientSessionGroup", MockGroup)
      
      manager = McpClientManager(config_path=str(config_file))
      tools = await manager.initialize()
      assert len(tools) == 1
      assert tools[0].name == "mcp_hello"
      
      res = await tools[0].handler({})
      assert res == "mcp response"
      
      # Mock the error path
      manager.group.tools["mcp_error_tool"] = manager.group.tools["mcp_hello"]
      with pytest.raises(ValueError, match="mcp error msg"):
          await manager.execute_tool("mcp_error_tool", {})
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_mcp_client.py`
  Expected: FAIL (ModuleNotFoundError for `jarvis/mcp.py`)

- [ ] **Step 3: Create `jarvis/mcp.py`**
  
  Create `jarvis/mcp.py`:
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
  
  In `jarvis/runtime.py`, add fields to `AgentContext` and initialize `McpClientManager` inside `submit()` before running the turn:
  ```python
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
  Update `AgentSession.__init__`:
  ```python
  class AgentSession:
      def __init__(self, ctx: AgentContext, kernel: object) -> None:
          self.ctx = ctx
          self.kernel = kernel
          self._lock = asyncio.Lock()
          self._mcp_initialized = False
  ```
  Update `AgentSession.submit`:
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
  from pathlib import Path
  from jarvis.tools import ToolRegistry
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
      # Verifies uppercase environment variable convention is active
      script_file.write_text("#!/bin/sh\necho \"committed: $MESSAGE\"", encoding="utf-8")
      script_file.chmod(0o755)
      
      manager = SkillManager(skills_root=str(tmp_path / "skills"))
      
      from types import SimpleNamespace
      ctx = SimpleNamespace(
          config=SimpleNamespace(allowed_skills=["mock_git"]),
          tools=ToolRegistry()
      )
      
      skills = await manager.load_allowed_skills(ctx)
      assert len(skills) == 1
      assert skills[0].name == "mock_git"
      
      # Run mock script tool
      committed_tool = ctx.tools._tools["mock_commit"]
      res = await committed_tool.handler({"message": "initial commit"})
      assert "committed: initial commit" in res
      
      # Test prompt hook
      from jarvis.hooks import SkillInstructionsHook
      hook = SkillInstructionsHook(skills)
      msgs = [Message(role="system", content="System:")]
      res = await hook.before_model(ctx, msgs)
      assert "Always commit files with clean messages." in res.messages[0].content
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_skills.py`
  Expected: FAIL (ModuleNotFoundError for `jarvis/skills.py`)

- [ ] **Step 3: Create `jarvis/skills.py`**
  
  Create `jarvis/skills.py` to parse skill folders and support dual case environment variable mapping:
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
              
              env = os.environ.copy()
              for k, v in args.items():
                  env[str(k).upper()] = str(v)
                  env[str(k).lower()] = str(v)
                  
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
      async def submit(self, message: Message) -> AsyncIterator[AgentEvent]:
          async with self._lock:
              if not self._mcp_initialized:
                  from jarvis.skills import SkillManager, SkillInstructionsHook
                  skill_mgr = SkillManager()
                  skills = await skill_mgr.load_allowed_skills(self.ctx)
                  if skills:
                      self.ctx.hooks.append(SkillInstructionsHook(skills))
                      
                  from jarvis.mcp import McpClientManager
                  manager = McpClientManager()
                  mcp_tools = await manager.initialize()
                  for t in mcp_tools:
                      self.ctx.tools.register(t)
                  self.ctx.mcp_manager = manager
                  self._mcp_initialized = True
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

### Task 5: Model Response Streaming and Delta Event Emission with Fallback

**Files:**
- Modify: `jarvis/kernel.py`, `jarvis/models/openai.py`, `jarvis/models/anthropic.py`
- Create: `tests/test_kernel_streaming.py`

- [ ] **Step 1: Write failing tests for streaming text deltas and tool call accumulation**
  
  Create `tests/test_kernel_streaming.py`:
  ```python
  import pytest
  from typing import AsyncGenerator
  from jarvis.runtime import AgentContext, RuntimeConfig, SessionState
  from jarvis.tools import ToolRegistry
  from jarvis.kernel import AgentKernel
  from jarvis.events import TextDeltaEvent, MessageEvent, ToolCallEvent
  from jarvis.models.base import BaseModelClient, Message, ModelResponse, ToolCall
  
  class MockStreamingModel(BaseModelClient):
      @classmethod
      def from_cfg(cls, cfg): return cls()
      
      async def generate(self, messages, tools):
          return ModelResponse(content="fallback completed response")
          
      async def generate_stream(self, messages, tools) -> AsyncGenerator[ModelResponse, None]:
          yield ModelResponse(content="hello ")
          yield ModelResponse(content="streaming world")
          yield ModelResponse(content=None, tool_calls=[ToolCall(call_id="c1", tool_name="ls", arguments={})])
  
  class MockNoStreamingModel(BaseModelClient):
      @classmethod
      def from_cfg(cls, cfg): return cls()
      
      async def generate(self, messages, tools):
          return ModelResponse(content="one-shot completion")
          
      async def generate_stream(self, messages, tools) -> AsyncGenerator[ModelResponse, None]:
          raise NotImplementedError("Streaming is not supported")
  
  @pytest.mark.asyncio
  async def test_kernel_run_turn_streams_events() -> None:
      ctx = AgentContext(
          config=RuntimeConfig(stream=True),
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
      
      # Verify tool call was accumulated and emitted
      tcalls = [ev.tool_call.tool_name for ev in events if isinstance(ev, ToolCallEvent)]
      assert tcalls == ["ls"]
  
  @pytest.mark.asyncio
  async def test_kernel_generate_stream_fallback() -> None:
      ctx = AgentContext(
          config=RuntimeConfig(stream=True),
          session=SessionState(id="s1"),
          model=MockNoStreamingModel(),
          tools=ToolRegistry()
      )
      kernel = AgentKernel()
      
      events = []
      async for event in kernel.run_turn(ctx, Message(role="user", content="hi")):
          events.append(event)
          
      # Fallback should call generate()
      msg_events = [ev.message.content for ev in events if isinstance(ev, MessageEvent)]
      assert msg_events == ["one-shot completion"]
  ```

- [ ] **Step 2: Run test to verify it fails**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_kernel_streaming.py`
  Expected: FAIL (TextDeltaEvents not emitted/assert failures)

- [ ] **Step 3: Update `AgentKernel` inside `jarvis/kernel.py` to stream model generation with fallback**
  
  Modify `jarvis/kernel.py` to support `stream=True` configuration, checking generator fallback on `NotImplementedError`:
  ```python
                  response = None
                  
                  # Check if streaming is enabled via config
                  if getattr(ctx.config, "stream", True):
                      try:
                          accumulated_content = ""
                          accumulated_tool_calls = []
                          
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
                      except NotImplementedError:
                          pass
                          
                  if response is None:
                      # One-shot generate() fallback path
                      response = await ctx.model.generate(messages, ctx.tools.schemas())
  ```

- [ ] **Step 4: Implement `generate_stream` in OpenAI and Anthropic clients**
  
  Modify `jarvis/models/openai.py` and `jarvis/models/anthropic.py` as detailed in the original design to parse and accumulate streamed chunk deltas (text and tool call blocks) and yield them correctly. (Refer to Task 5 in previous revision for exact implementation block).

- [ ] **Step 5: Run tests to verify they pass**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_kernel_streaming.py`
  Expected: PASS

- [ ] **Step 6: Commit**
  
  Run:
  ```bash
  git add jarvis/kernel.py jarvis/models/openai.py jarvis/models/anthropic.py tests/test_kernel_streaming.py
  git commit -m "feat: enable kernel model response streaming with fallback"
  ```

---

### Task 6: Error Recovery with Backoff Retries

**Files:**
- Create: `jarvis/retry.py`
- Modify: `jarvis/models/openai.py`, `jarvis/models/anthropic.py`
- Create: `tests/test_error_recovery.py`

- [ ] **Step 1: Write failing tests for retry logic on transient errors**
  
  Create `tests/test_error_recovery.py` testing SDK dynamic module `isinstance` checking:
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
              raise ConnectionError("Transient error")
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
  Expected: FAIL (ModuleNotFoundError for `jarvis/retry.py`)

- [ ] **Step 3: Create `jarvis/retry.py`**
  
  Create `jarvis/retry.py` with dynamic SDK module type checks:
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
                      is_transient = isinstance(exc, (ConnectionError, TimeoutError))
                      
                      if not is_transient:
                          # Dynamic check for openai
                          try:
                              import openai
                              if isinstance(exc, (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)):
                                  is_transient = True
                          except ImportError:
                              pass
                              
                      if not is_transient:
                          # Dynamic check for anthropic
                          try:
                              import anthropic
                              if isinstance(exc, (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError)):
                                  is_transient = True
                          except ImportError:
                              pass
                              
                      if not is_transient:
                          exc_name = type(exc).__name__
                          is_transient = (
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
  
  Apply `retry_with_backoff` decorator to OpenAI and Anthropic clients as specified.

- [ ] **Step 5: Run tests to verify they pass**
  
  Run: `PYTHONPATH=. .venv/bin/pytest tests/test_error_recovery.py`
  Expected: PASS

- [ ] **Step 6: Commit**
  
  Run:
  ```bash
  git add jarvis/retry.py jarvis/models/openai.py jarvis/models/anthropic.py tests/test_error_recovery.py
  git commit -m "feat: add backoff retries for transient model failures"
  ```
