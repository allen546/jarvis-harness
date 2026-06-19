from __future__ import annotations

import asyncio
import json
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from jarvis.config import SessionConfig
from jarvis.events import AgentEvent
from jarvis.hooks import BudgetGuardHook, SessionRefreshHook, JSONLHistoryHook, ToolApprovalHook, TurnHook
from jarvis.models.base import BaseModelClient, Message, ToolCall, get_model_class
from jarvis.tools import Tool, ToolRegistry

current_context: ContextVar[AgentContext | None] = ContextVar("current_context", default=None)

def render_system_prompt(template: str | None, root: str = ".", template_file: str | None = None) -> str | None:
    """Render a Jinja2 system prompt template with live context.

    If template_file is given, reads the template from that path (relative to root).
    Otherwise uses the inline template string.

    Available variables:
        skills_dirs  — list of configured skill directory names
        mcp_servers  — list of MCP server name strings from mcp_settings.json
    """
    from pathlib import Path as _P
    from jinja2 import Environment, BaseLoader
    base = _P(root)
    # Resolve template source
    if template_file:
        tpl_path = base / template_file
        if tpl_path.exists():
            template = tpl_path.read_text(encoding="utf-8")
    if not template:
        return template
    env = Environment(loader=BaseLoader(), autoescape=False)
    # Discover skill directories
    skills_dirs = [d for d in (base / "skills").iterdir() if d.is_dir()] if (base / "skills").exists() else []
    # Discover MCP servers from config
    mcp_servers: list[str] = []
    mcp_cfg = base / "config" / "mcp_settings.json"
    if mcp_cfg.exists():
        try:
            mcp_servers = list(json.loads(mcp_cfg.read_text()).get("mcpServers", {}).keys())
        except Exception:
            pass
    tmpl = env.from_string(template)
    return tmpl.render(skills_dirs=[d.name for d in skills_dirs], mcp_servers=mcp_servers)


@dataclass(slots=True)
class RuntimeMemoryConfig:
    enabled: bool = True
    storage_dir: str = "storage"
    scope: str = "global"
    session_ttl_seconds: int = 600
    refresh_threshold_messages: int = 24
    refresh_keep_messages: int = 12
    distill_interval_turns: int = 10
    inject_top_facts: int = 3
    inject_top_procedures: int = 2
    inject_min_score: float = 0.35
    auto_distill_skills: bool = True
    skill_min_observations: int = 3

@dataclass(slots=True)
class RuntimeConfig:
    system_prompt: str | None = None
    max_consecutive_tools: int = 5
    require_tool_approval: bool = False
    skills_dirs: list[str] = field(default_factory=lambda: ["skills/"])
    stream: bool = True
    # safety hooks
    max_repeated_tool_calls: int = 3
    repeated_content_threshold: float = 0.8
    repeated_content_window: int = 3
    memory: RuntimeMemoryConfig = field(default_factory=RuntimeMemoryConfig)


@dataclass(slots=True)
class SessionState:
    id: str
    history: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentContext:
    config: RuntimeConfig
    session: SessionState
    model: BaseModelClient
    tools: ToolRegistry
    hooks: list[TurnHook] = field(default_factory=list)
    emit_event: Callable[[AgentEvent], None] | None = None
    mcp_manager: Any | None = field(default=None, compare=False)
    approval_handler: Callable[[ToolCall], bool | Awaitable[bool]] | None = field(default=None, compare=False)
    proxy_env: dict[str, str] = field(default_factory=dict)


class AgentSession:
    def __init__(self, ctx: AgentContext, kernel: object, proxy_env: dict[str, str] | None = None) -> None:
        self.ctx = ctx
        self.kernel = kernel
        self._lock = asyncio.Lock()
        self._mcp_initialized = False
        self._mcp_tool_names: list[str] = []
        self.last_activity: float = time.time()
        if proxy_env:
            self.ctx.proxy_env = proxy_env
        # Register load_mcp tool so the model can trigger MCP loading on demand
        self.ctx.tools.register(Tool(
            name="load_mcp",
            description="Load MCP server tools. Call this if you need access to external services (maps, calendar, etc) that aren't yet available.",
            parameters={"type": "object", "properties": {}},
            handler=self._load_mcp_tool,
        ))
    async def _load_mcp_tool(self, args: dict[str, Any]) -> str:
        loaded = await self._ensure_mcp()
        if loaded:
            return f"Loaded {loaded} MCP tools: {', '.join(self._mcp_tool_names)}"
        return "No MCP servers configured or already loaded."


    async def _ensure_mcp(self) -> int:
        """Load MCP tools if not yet initialized. Returns count of newly loaded tools."""
        if self._mcp_initialized:
            return 0


        from jarvis.mcp import McpClientManager
        manager = McpClientManager(proxy_env=self.ctx.proxy_env)
        mcp_tools = await manager.initialize()
        for t in mcp_tools:
            self.ctx.tools.register(t)
            self._mcp_tool_names.append(t.name)
        self.ctx.mcp_manager = manager
        self._mcp_initialized = True
        return len(mcp_tools)

    async def _expire_session(self) -> None:
        """Expire idle session: distill remaining undistilled messages, then clear history."""
        import warnings
        session = self.ctx.session
        undistilled = [m for m in session.history if not m.metadata.get("distilled")]

        if undistilled and self.ctx.config.memory.enabled:
            try:
                from jarvis.memory_store import SemanticMemoryStore
                memory_config = self.ctx.config.memory
                store = SemanticMemoryStore(storage_dir=memory_config.storage_dir)
                model = self.ctx.model

                prompt_msgs = [
                    Message(role="system", content=(
                        "Extract durable memory from this conversation. "
                        "Return strict JSON only, no markdown.\n"
                        "Schema: {\"facts\":[{\"text\":str,\"tags\":[str],\"confidence\":float}],"
                        "\"procedures\":[{\"name\":str,\"trigger\":str,\"summary\":str,\"steps\":[str],"
                        "\"tools\":[str],\"confidence\":float}]}\n"
                        "Facts are stable truths. Procedures are reusable task patterns."
                    )),
                    *undistilled,
                ]
                response = await model.generate(prompt_msgs, [])
                parsed = json.loads(response.content or "{}")

                for fact in parsed.get("facts", []):
                    text = fact.get("text", "")
                    if text:
                        await store.add_memory(
                            session.id, text, fact.get("tags", ["truths"]),
                            kind="fact", scope=memory_config.scope,
                            metadata={"source": "session_expiry"},
                            confidence=fact.get("confidence", 1.0),
                        )
                for proc in parsed.get("procedures", []):
                    name = proc.get("name", "")
                    if name and proc.get("steps"):
                        await store.add_memory(
                            session.id, proc.get("summary", name),
                            ["procedure", *proc.get("tools", [])],
                            kind="procedure", scope=memory_config.scope,
                            metadata={
                                "name": name,
                                "trigger": proc.get("trigger", ""),
                                "steps": proc["steps"],
                                "tools": proc.get("tools", []),
                                "source": "session_expiry",
                            },
                            confidence=proc.get("confidence", 1.0),
                        )
            except Exception as exc:
                warnings.warn(f"Error during session expiry distillation: {exc}")

        # Mark all as distilled and clear
        for m in session.history:
            m.metadata["distilled"] = True
        session.history = []


    async def submit(self, message: Message) -> AsyncIterator[AgentEvent]:
        async with self._lock:
            # Session TTL check — expire idle sessions
            memory_config = self.ctx.config.memory
            if memory_config.session_ttl_seconds > 0:
                elapsed = time.time() - self.last_activity
                if elapsed >= memory_config.session_ttl_seconds:
                    await self._expire_session()
            self.last_activity = time.time()

            token = current_context.set(self.ctx)
            event_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
            
            old_emitter = self.ctx.emit_event
            def queue_emitter(ev: AgentEvent) -> None:
                event_queue.put_nowait(ev)
                if old_emitter:
                    old_emitter(ev)
            self.ctx.emit_event = queue_emitter

            async def run() -> None:
                try:
                    async for event in self.kernel.run_turn(self.ctx, message):  # type: ignore[attr-defined]
                        if self.ctx.emit_event is not None:
                            self.ctx.emit_event(event)
                except Exception as exc:
                    from jarvis.events import ErrorEvent
                    err_event = ErrorEvent(session_id=self.ctx.session.id, message=str(exc))
                    if self.ctx.emit_event is not None:
                        self.ctx.emit_event(err_event)
                finally:
                    event_queue.put_nowait(None)

            task = asyncio.create_task(run())
            try:
                while True:
                     event = await event_queue.get()
                     if event is None:
                         break
                     yield event
            finally:
                self.ctx.emit_event = old_emitter
                try:
                    current_context.reset(token)
                except ValueError:
                    pass
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                else:
                    await task

    async def close(self) -> None:
        async with self._lock:
            if self.ctx.mcp_manager:
                await self.ctx.mcp_manager.close()
                self.ctx.mcp_manager = None
                self._mcp_initialized = False
            for name in self._mcp_tool_names:
                self.ctx.tools._tools.pop(name, None)
            self._mcp_tool_names.clear()
            self.ctx.tools._tools.pop("load_mcp", None)


def _default_hooks(config: SessionConfig | None = None) -> list[TurnHook]:
    from jarvis.hooks import RepeatedLoopHook, RepeatedContentHook

    # Read safety params from config, fall back to defaults
    max_repeats = 3
    content_threshold = 0.8
    content_window = 3
    if config is not None:
        max_repeats = config.harness.max_repeated_tool_calls
        content_threshold = config.harness.repeated_content_threshold
        content_window = config.harness.repeated_content_window

    # Memory settings
    memory = config.harness.memory if config is not None else None
    memory_enabled = True if memory is None else memory.enabled
    memory_storage_dir = "storage" if memory is None else memory.storage_dir
    memory_scope = "global" if memory is None else memory.scope
    refresh_threshold = 24 if memory is None else memory.refresh_threshold_messages
    refresh_keep = 12 if memory is None else memory.refresh_keep_messages
    inject_top_facts = 3 if memory is None else memory.inject_top_facts
    inject_top_procedures = 2 if memory is None else memory.inject_top_procedures
    inject_min_score = 0.35 if memory is None else memory.inject_min_score
    distill_interval = 10 if memory is None else memory.distill_interval_turns
    auto_distill_skills = True if memory is None else memory.auto_distill_skills
    skill_min_observations = 3 if memory is None else memory.skill_min_observations

    if memory_scope not in ("global", "session"):
        raise ValueError(f"unsupported memory scope: {memory_scope}")

    # Embedding: config URL > env var > local fallback
    embedding_url = ""
    embedding_enabled = False
    embedding_dims = 256
    if config is not None:
        embedding_url = config.harness.embedding.url
        embedding_enabled = config.harness.embedding.enabled
        embedding_dims = config.harness.embedding.dimensions
    if not embedding_url:
        embedding_url = os.environ.get("EMBEDDING_URL", "")

    hooks: list[TurnHook] = [
        JSONLHistoryHook(storage_dir=memory_storage_dir),
    ]

    if memory_enabled:
        from jarvis.memory_store import MemoryInjectionHook
        hooks.append(MemoryInjectionHook(
            storage_dir=memory_storage_dir,
            embedding_url=embedding_url or None,
            embedding_dimensions=embedding_dims,
            top_facts=inject_top_facts,
            top_procedures=inject_top_procedures,
            min_score=inject_min_score,
            scope=memory_scope,
        ))

    hooks.extend([
        SessionRefreshHook(storage_dir=memory_storage_dir, threshold=refresh_threshold, keep_messages=refresh_keep),
        BudgetGuardHook(),
        ToolApprovalHook(),
        RepeatedLoopHook(max_repeats=max_repeats),
        RepeatedContentHook(threshold=content_threshold, window=content_window),
    ])

    if memory_enabled:
        from jarvis.memory_store import MemoryDistillationHook
        hooks.append(MemoryDistillationHook(
            storage_dir=memory_storage_dir,
            embedding_url=embedding_url or None,
            embedding_dimensions=embedding_dims,
            scope=memory_scope,
            auto_distill_skills=auto_distill_skills,
            skill_min_observations=skill_min_observations,
            distill_interval_turns=distill_interval,
        ))

    return hooks


def context_from_config(config: SessionConfig, tools: ToolRegistry, hooks: list[TurnHook] | None = None, root: str = ".") -> AgentContext:
    provider = config.model.provider.lower()
    model_cls = get_model_class(provider)
    h = config.harness
    return AgentContext(
        config=RuntimeConfig(
            system_prompt=render_system_prompt(h.system_prompt, root, template_file=h.system_prompt_file),
            max_consecutive_tools=h.max_consecutive_tools,
            require_tool_approval=h.require_tool_approval,
            skills_dirs=h.skills_dirs,
            stream=h.stream,
            max_repeated_tool_calls=h.max_repeated_tool_calls,
            repeated_content_threshold=h.repeated_content_threshold,
            repeated_content_window=h.repeated_content_window,
            memory=RuntimeMemoryConfig(
                enabled=h.memory.enabled,
                storage_dir=h.memory.storage_dir,
                scope=h.memory.scope,
                session_ttl_seconds=h.memory.session_ttl_seconds,
                refresh_threshold_messages=h.memory.refresh_threshold_messages,
                refresh_keep_messages=h.memory.refresh_keep_messages,
                distill_interval_turns=h.memory.distill_interval_turns,
                inject_top_facts=h.memory.inject_top_facts,
                inject_top_procedures=h.memory.inject_top_procedures,
                inject_min_score=h.memory.inject_min_score,
                auto_distill_skills=h.memory.auto_distill_skills,
                skill_min_observations=h.memory.skill_min_observations,
            ),
        ),
        session=SessionState(id=config.session_id),
        model=model_cls.from_cfg(config),
        tools=tools,
        hooks=hooks if hooks is not None else _default_hooks(config),
    )
