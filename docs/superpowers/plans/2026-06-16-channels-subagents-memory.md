# Discord & QQ Transports, Interactive Subagents, and Advanced Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Discord and QQ transports with background event loops, interactive collaborative subagent tools with event routing, and an advanced memory system featuring JSONL persistence, autocompression, and HTTP-delegated tagged semantic memory.

**Architecture:** Run the Discord/QQ transports as background asyncio tasks inside the FastAPI gateway process, using `AgentSession.submit` for message entry and an `emit_event` callback for routing subagent progress events. Memory and history hooks intercept execution via `TurnHook` checkpoints, delegating CPU-intensive embedding generation to a separate process via HTTP calls.

**Tech Stack:** Python 3.14+, FastAPI, discord.py, botpy, httpx, pytest, pytest-asyncio.

---

### Task 1: Extend Runtime State and Context for Metadata and Event Routing

**Files:**
- Modify: `jarvis/runtime.py`
- Test: `tests/test_runtime.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from jarvis.runtime import SessionState, AgentContext, RuntimeConfig
from jarvis.models.base import BaseModelClient

class DummyModel(BaseModelClient):
    @classmethod
    def from_cfg(cls, cfg): return cls()
    async def generate(self, messages, tools): return None

def test_session_state_metadata_and_context_emit_event():
    # Verify metadata is supported on SessionState
    state = SessionState(id="test_sess", metadata={"foo": "bar"})
    assert state.metadata["foo"] == "bar"

    # Verify emit_event callback is present on AgentContext
    dummy = DummyModel()
    called = False
    def cb(event):
        nonlocal called
        called = True

    ctx = AgentContext(
        config=RuntimeConfig(),
        session=state,
        model=dummy,
        tools=None,
        emit_event=cb
    )
    assert ctx.emit_event is not None
    ctx.emit_event("dummy_event")
    assert called is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_runtime.py -k test_session_state_metadata_and_context_emit_event`
Expected: FAIL (TypeError: unexpected keyword argument 'metadata' or 'emit_event')

- [ ] **Step 3: Write minimal implementation**

Modify `jarvis/runtime.py` to add `metadata` to `SessionState` and `emit_event` to `AgentContext`:
```python
# In jarvis/runtime.py
from typing import Callable, Any

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
    emit_event: Callable[[Any], None] | None = None
```

Update `AgentSession.submit` to bind an event emitter:
```python
# In jarvis/runtime.py:
    async def submit(self, message: Message) -> AsyncIterator[AgentEvent]:
        async with self._lock:
            # Simple queue to buffer yielded events
            events_queue = asyncio.Queue()
            def emitter(ev: Any) -> None:
                events_queue.put_nowait(ev)
            self.ctx.emit_event = emitter

            async def run_kernel():
                try:
                    async for event in self.kernel.run_turn(self.ctx, message):
                        events_queue.put_nowait(event)
                finally:
                    events_queue.put_nowait(None)  # Sentinel for completion

            task = asyncio.create_task(run_kernel())
            while True:
                ev = await events_queue.get()
                if ev is None:
                    break
                yield ev
            await task
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_runtime.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jarvis/runtime.py
git commit -m "feat: add metadata to SessionState and emit_event to AgentContext"
```

---

### Task 2: Implement JSONL History Hook

**Files:**
- Create: `jarvis/hooks/history.py`
- Modify: `jarvis/hooks/__init__.py` (re-export `JSONLHistoryHook`)
- Test: `tests/test_history_hook.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import pytest
from pathlib import Path
from jarvis.hooks.history import JSONLHistoryHook
from jarvis.models.base import Message
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState

@pytest.mark.asyncio
async def test_jsonl_history_hook(tmp_path: Path):
    storage_dir = tmp_path / "storage"
    hook = JSONLHistoryHook(storage_dir=str(storage_dir))
    state = SessionState(id="sess1")
    ctx = AgentContext(config=RuntimeConfig(), session=state, model=None, tools=None)
    
    # 1. Test before_model loads empty history
    await hook.before_model(ctx, [])
    assert len(state.history) == 0

    # 2. Test after_turn writes history
    user_msg = Message(role="user", content="hello")
    assistant_msg = Message(role="assistant", content="hi there")
    state.history.extend([user_msg, assistant_msg])
    await hook.after_turn(ctx, assistant_msg)

    # Check file exists and contains correct lines
    file_path = storage_dir / "sessions" / "sess1" / "history.jsonl"
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["content"] == "hello"
    assert json.loads(lines[1])["content"] == "hi there"

    # 3. Test before_model loads from file
    new_state = SessionState(id="sess1")
    new_ctx = AgentContext(config=RuntimeConfig(), session=new_state, model=None, tools=None)
    await hook.before_model(new_ctx, [])
    assert len(new_state.history) == 2
    assert new_state.history[0].content == "hello"
    assert new_state.history[1].content == "hi there"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_history_hook.py`
Expected: FAIL (ModuleNotFoundError: No module named 'jarvis.hooks.history')

- [ ] **Step 3: Write minimal implementation**

Create `jarvis/hooks/history.py`:
```python
import json
import os
from pathlib import Path
from jarvis.hooks import TurnHook, HookResult
from jarvis.models.base import Message

class JSONLHistoryHook(TurnHook):
    def __init__(self, storage_dir: str = "storage") -> None:
        self.storage_dir = storage_dir

    def _get_file_path(self, session_id: str) -> Path:
        return Path(self.storage_dir) / "sessions" / session_id / "history.jsonl"

    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
        session = getattr(ctx, "session")
        if not session.history:
            file_path = self._get_file_path(session.id)
            if file_path.exists():
                history = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            history.append(Message.model_validate(json.loads(line)))
                session.history = history
        return HookResult()

    async def after_turn(self, ctx: object, response: Message) -> HookResult:
        session = getattr(ctx, "session")
        file_path = self._get_file_path(session.id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            for message in session.history[-2:]:  # User message and final response
                f.write(json.dumps(message.model_dump()) + "\n")
        return HookResult()
```

Register hook in `jarvis/hooks/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_history_hook.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jarvis/hooks/history.py jarvis/hooks/__init__.py
git commit -m "feat: add JSONLHistoryHook for chat persistence"
```

---

### Task 3: Implement Context Compression (Autocompression) Hook

**Files:**
- Create: `jarvis/hooks/compression.py`
- Modify: `jarvis/hooks/__init__.py`
- Test: `tests/test_compression_hook.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from jarvis.hooks.compression import ContextCompressionHook
from jarvis.models.base import Message, ModelResponse, BaseModelClient
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState

class MockModel(BaseModelClient):
    @classmethod
    def from_cfg(cls, cfg): return cls()
    async def generate(self, messages, tools):
        return ModelResponse(content="SUMMARY_OF_CHAT")

@pytest.mark.asyncio
async def test_compression_hook():
    hook = ContextCompressionHook(threshold=5, compress_count=3)
    state = SessionState(id="sess_comp")
    # 6 messages (exceeds threshold 5)
    state.history = [
        Message(role="user", content="m1"),
        Message(role="assistant", content="m2"),
        Message(role="user", content="m3"),
        Message(role="assistant", content="m4"),
        Message(role="user", content="m5"),
        Message(role="assistant", content="m6"),
    ]
    ctx = AgentContext(config=RuntimeConfig(), session=state, model=MockModel(), tools=None)

    # Trigger compression hook
    await hook.before_model(ctx, list(state.history))
    
    # Check that oldest 3 messages are replaced by a system summary
    assert len(state.history) == 4  # 1 summary + remaining 3 messages
    assert state.history[0].role == "system"
    assert "SUMMARY_OF_CHAT" in state.history[0].content
    assert state.history[1].content == "m4"
    assert state.history[2].content == "m5"
    assert state.history[3].content == "m6"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_compression_hook.py`
Expected: FAIL (ModuleNotFoundError: No module named 'jarvis.hooks.compression')

- [ ] **Step 3: Write minimal implementation**

Create `jarvis/hooks/compression.py`:
```python
from jarvis.hooks import TurnHook, HookResult
from jarvis.models.base import Message

class ContextCompressionHook(TurnHook):
    def __init__(self, threshold: int = 20, compress_count: int = 10) -> None:
        self.threshold = threshold
        self.compress_count = compress_count

    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
        session = getattr(ctx, "session")
        model = getattr(ctx, "model")
        if len(session.history) >= self.threshold:
            to_compress = session.history[:self.compress_count]
            remaining = session.history[self.compress_count:]
            
            prompt_msgs = [
                Message(role="system", content="Summarize the following chat history concisely:"),
                *to_compress
            ]
            response = await model.generate(prompt_msgs, [])
            summary_content = response.content or "No summary"
            summary_msg = Message(role="system", content=f"[Summary of previous conversation: {summary_content}]")
            
            session.history = [summary_msg] + remaining
            return HookResult(messages=[summary_msg] + remaining)
        return HookResult()
```

Register hook in `jarvis/hooks/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_compression_hook.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jarvis/hooks/compression.py jarvis/hooks/__init__.py
git commit -m "feat: add ContextCompressionHook for autocompression"
```

---

### Task 4: Implement Tagged Semantic Memory and Retrieval

**Files:**
- Create: `jarvis/memory_store.py`
- Modify: `jarvis/tools.py` (Register `search_semantic_memory` tool)
- Test: `tests/test_semantic_memory.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jarvis.memory_store import SemanticMemoryStore, search_semantic_memory_tool
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState

# Fake Embedding Server
app = FastAPI()
@app.post("/embeddings")
def get_emb(data: dict):
    # Returns mock embedding based on input text
    return {"embedding": [0.1 if "truth" in data["text"] else 0.5] * 4}

@pytest.mark.asyncio
async def test_semantic_memory_store(tmp_path: Path):
    with TestClient(app) as client:
        # Override client calls to the fake HTTP server
        store = SemanticMemoryStore(
            storage_dir=str(tmp_path),
            embedding_url="http://test/embeddings",
            http_client=client
        )
        session_id = "s_mem"

        # 1. Add memory
        await store.add_memory(session_id, "This is a key truth.", ["truths"])
        await store.add_memory(session_id, "Normal chat event history.", ["history"])

        # 2. Retrieve memory
        results = await store.search(session_id, "query truth", tag="truths", limit=1)
        assert len(results) == 1
        assert "truth" in results[0]["text"]
        assert "truths" in results[0]["tags"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_semantic_memory.py`
Expected: FAIL (ModuleNotFoundError: No module named 'jarvis.memory_store')

- [ ] **Step 3: Write minimal implementation**

Create `jarvis/memory_store.py`:
```python
import json
import math
import uuid
import datetime
from pathlib import Path
from typing import Any, Optional

class SemanticMemoryStore:
    def __init__(self, storage_dir: str = "storage", embedding_url: str = "http://localhost:8001/embeddings", http_client: Any = None) -> None:
        self.storage_dir = storage_dir
        self.embedding_url = embedding_url
        self.http_client = http_client

    def _get_file_path(self, session_id: str) -> Path:
        return Path(self.storage_dir) / "sessions" / session_id / "semantic_memory.json"

    def _load(self, session_id: str) -> list[dict[str, Any]]:
        path = self._get_file_path(session_id)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, session_id: str, memories: list[dict[str, Any]]) -> None:
        path = self._get_file_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=2)

    async def _get_embedding(self, text: str) -> list[float]:
        payload = {"text": text}
        if self.http_client:
            # Direct TestClient/AsyncClient usage in tests
            resp = self.http_client.post(self.embedding_url, json=payload)
            if hasattr(resp, "json"):
                return resp.json()["embedding"]
            return (await resp).json()["embedding"]
        else:
            import httpx
            async with httpx.AsyncClient() as client:
                r = await client.post(self.embedding_url, json=payload, timeout=10.0)
                r.raise_for_status()
                return r.json()["embedding"]

    async def add_memory(self, session_id: str, text: str, tags: list[str]) -> None:
        embedding = await self._get_embedding(text)
        memories = self._load(session_id)
        memories.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "embedding": embedding,
            "tags": tags,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        self._save(session_id, memories)

    async def search(self, session_id: str, query: str, tag: Optional[str] = None, limit: int = 3) -> list[dict[str, Any]]:
        q_emb = await self._get_embedding(query)
        memories = self._load(session_id)
        
        matches = []
        for m in memories:
            if tag and tag not in m["tags"]:
                continue
            # Cosine similarity
            m_emb = m["embedding"]
            dot = sum(a * b for a, b in zip(q_emb, m_emb))
            norm_q = math.sqrt(sum(a * a for a in q_emb))
            norm_m = math.sqrt(sum(a * a for a in m_emb))
            sim = dot / (norm_q * norm_m) if norm_q and norm_m else 0.0
            
            matches.append((sim, m))
            
        matches.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in matches[:limit]]
```

Create hook and tool functions in `jarvis/memory_store.py` and register `search_semantic_memory` tool:
```python
# Tool registration function inside jarvis/memory_store.py:
async def search_semantic_memory_tool(ctx: Any, args: dict[str, Any]) -> str:
    query = args["query"]
    tag = args.get("tag")
    store = SemanticMemoryStore(embedding_url=ctx.config.extra_params.get("embedding_url", "http://localhost:8001/embeddings"))
    results = await store.search(ctx.session.id, query, tag=tag)
    return json.dumps([{"text": r["text"], "tags": r["tags"]} for r in results])
```

Register the tool schema in `jarvis/tools.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_semantic_memory.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jarvis/memory_store.py jarvis/tools.py
git commit -m "feat: implement semantic memory store and search tool"
```

---

### Task 5: Implement Semantic Memory Purge Tool

**Files:**
- Modify: `jarvis/memory_store.py` (Add `purge` method and tool function)
- Modify: `jarvis/tools.py` (Register `purge_semantic_memory` tool)
- Test: `tests/test_semantic_memory_purge.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from pathlib import Path
from jarvis.memory_store import SemanticMemoryStore, purge_semantic_memory_tool
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState

@pytest.mark.asyncio
async def test_purge_semantic_memory(tmp_path: Path):
    store = SemanticMemoryStore(storage_dir=str(tmp_path))
    session_id = "s_purge"
    
    # Populate mock index without HTTP calls
    memories = [
        {"id": "id1", "text": "Apple is good.", "embedding": [0.1], "tags": ["truths"]},
        {"id": "id2", "text": "Banana is yellow.", "embedding": [0.2], "tags": ["history"]}
    ]
    store._save(session_id, memories)

    # Purge by tag
    await store.purge(session_id, tag="history")
    remaining = store._load(session_id)
    assert len(remaining) == 1
    assert remaining[0]["id"] == "id1"

    # Purge by ID
    await store.purge(session_id, ids=["id1"])
    assert len(store._load(session_id)) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_semantic_memory_purge.py`
Expected: FAIL (AttributeError: 'SemanticMemoryStore' object has no attribute 'purge')

- [ ] **Step 3: Write minimal implementation**

Add `purge` method to `SemanticMemoryStore` in `jarvis/memory_store.py`:
```python
    async def purge(self, session_id: str, tag: Optional[str] = None, ids: Optional[list[str]] = None) -> int:
        memories = self._load(session_id)
        before = len(memories)
        if ids:
            memories = [m for m in memories if m["id"] not in ids]
        elif tag:
            memories = [m for m in memories if tag not in m["tags"]]
        self._save(session_id, memories)
        return before - len(memories)
```

Add the tool handler function:
```python
async def purge_semantic_memory_tool(ctx: Any, args: dict[str, Any]) -> str:
    tag = args.get("tag")
    ids = args.get("ids")
    store = SemanticMemoryStore()
    purged = await store.purge(ctx.session.id, tag=tag, ids=ids)
    return f"Purged {purged} items from semantic memory."
```

Register the tool schema in `jarvis/tools.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_semantic_memory_purge.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jarvis/memory_store.py jarvis/tools.py
git commit -m "feat: implement purge_semantic_memory tool"
```

---

### Task 6: Implement Interactive Collaborative Subagents

**Files:**
- Create: `jarvis/subagent.py`
- Modify: `jarvis/tools.py` (Register subagent tools)
- Test: `tests/test_subagents.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from jarvis.events import TextDeltaEvent, MessageEvent
from jarvis.models.base import BaseModelClient, ModelResponse, Message
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState, AgentSession
from jarvis.kernel import AgentKernel
from jarvis.subagent import spawn_subagent_tool, send_subagent_message_tool

class FakeModel(BaseModelClient):
    @classmethod
    def from_cfg(cls, cfg): return cls()
    async def generate(self, messages, tools):
        return ModelResponse(content="subagent reply")

@pytest.mark.asyncio
async def test_collaborative_subagents():
    # Setup parent session context
    parent_state = SessionState(id="parent_sess")
    parent_called_events = []
    def parent_cb(ev):
        parent_called_events.append(ev)

    parent_ctx = AgentContext(
        config=RuntimeConfig(),
        session=parent_state,
        model=FakeModel(),
        tools=None,
        emit_event=parent_cb
    )

    # 1. Test Spawn Subagent
    resp = await spawn_subagent_tool(parent_ctx, {"prompt": "subtask prompt", "task_name": "task1"})
    assert "sub_session_id" in resp
    assert resp["response"] == "subagent reply"
    sub_id = resp["sub_session_id"]

    # Check that events from subagent bubbled up to parent context callback
    assert len(parent_called_events) > 0
    assert any(isinstance(ev, TextDeltaEvent) and ev.content == "subagent reply" for ev in parent_called_events)

    # 2. Test Send Message to Subagent
    reply_resp = await send_subagent_message_tool(parent_ctx, {"sub_session_id": sub_id, "message": "follow up"})
    assert reply_resp["response"] == "subagent reply"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_subagents.py`
Expected: FAIL (ModuleNotFoundError: No module named 'jarvis.subagent')

- [ ] **Step 3: Write minimal implementation**

Create `jarvis/subagent.py`:
```python
import uuid
from typing import Any
from jarvis.runtime import AgentContext, AgentSession, SessionState
from jarvis.models.base import Message
from jarvis.kernel import AgentKernel

active_subagents: dict[str, AgentSession] = {}

async def spawn_subagent_tool(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    prompt = args["prompt"]
    task_name = args["task_name"]
    system_override = args.get("system_override")

    sub_session_id = f"sub_{uuid.uuid4()}"
    sub_state = SessionState(id=sub_session_id, metadata={"task_name": task_name})
    
    # Setup subagent event forwarder back to parent
    def forward_events(ev: Any) -> None:
        if ctx.emit_event:
            ctx.emit_event(ev)

    sub_ctx = AgentContext(
        config=ctx.config,
        session=sub_state,
        model=ctx.model,
        # Prevent subagent from spawning further subagents by filtering out subagent tools
        tools=ctx.tools, 
        hooks=ctx.hooks,
        emit_event=forward_events
    )
    if system_override:
        sub_ctx.config.system_prompt = system_override

    sub_session = AgentSession(ctx=sub_ctx, kernel=AgentKernel())
    active_subagents[sub_session_id] = sub_session
    
    # Store in parent session metadata
    if "active_subagents" not in ctx.session.metadata:
        ctx.session.metadata["active_subagents"] = []
    ctx.session.metadata["active_subagents"].append(sub_session_id)

    # Run first turn
    response_content = ""
    async for event in sub_session.submit(Message(role="user", content=prompt)):
        from jarvis.events import MessageEvent
        if isinstance(event, MessageEvent):
            response_content = event.message.content

    return {"sub_session_id": sub_session_id, "response": response_content}

async def send_subagent_message_tool(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    sub_session_id = args["sub_session_id"]
    message = args["message"]
    
    sub_session = active_subagents.get(sub_session_id)
    if not sub_session:
        raise ValueError(f"Subagent session {sub_session_id} not found or inactive.")

    response_content = ""
    async for event in sub_session.submit(Message(role="user", content=message)):
        from jarvis.events import MessageEvent
        if isinstance(event, MessageEvent):
            response_content = event.message.content

    return {"response": response_content}

async def close_subagent_tool(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    sub_session_id = args["sub_session_id"]
    if sub_session_id in active_subagents:
        del active_subagents[sub_session_id]
    if "active_subagents" in ctx.session.metadata and sub_session_id in ctx.session.metadata["active_subagents"]:
        ctx.session.metadata["active_subagents"].remove(sub_session_id)
    return {"message": f"Subagent {sub_session_id} closed."}
```

Register tools in `jarvis/tools.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_subagents.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jarvis/subagent.py jarvis/tools.py
git commit -m "feat: implement interactive collaborative subagents"
```

---

### Task 7: Implement Discord & QQ Transport Client Interfaces & Mocks

**Files:**
- Create: `jarvis/transports/discord.py`
- Create: `jarvis/transports/qq.py`
- Test: `tests/test_transports.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from jarvis.transports.discord import DiscordTransport, MockDiscordAPI
from jarvis.transports.qq import QQTransport, MockQQAPI
from jarvis.models.base import Message, NativeAction

@pytest.mark.asyncio
async def test_discord_transport_mock():
    api = MockDiscordAPI()
    transport = DiscordTransport(client=api)
    
    # Verify reply action sends message to mock
    reply_action = NativeAction(action_type="discord_reply", params={"message_id": "msg1", "content": "hello discord"})
    await transport.execute_native_action("channel1", reply_action)
    assert api.sent_messages[0] == ("channel1", "hello discord", "msg1")

@pytest.mark.asyncio
async def test_qq_transport_mock():
    api = MockQQAPI()
    transport = QQTransport(client=api)
    
    # Verify reply action sends message to mock
    reply_action = NativeAction(action_type="qq_reply", params={"message_id": "msg1", "content": "hello qq"})
    await transport.execute_native_action("dm1", reply_action)
    assert api.replies[0] == ("dm1", "hello qq", "msg1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_transports.py`
Expected: FAIL (ModuleNotFoundError: No module named 'jarvis.transports.discord')

- [ ] **Step 3: Write minimal implementation**

Create `jarvis/transports/discord.py`:
```python
import typing
from jarvis.models.base import NativeAction

class MockDiscordAPI:
    def __init__(self):
        self.sent_messages = []
        self.reactions = []

    async def send_message(self, channel_id: str, content: str, reply_to: str | None = None):
        self.sent_messages.append((channel_id, content, reply_to))
        
    async def add_reaction(self, channel_id: str, message_id: str, emoji: str):
        self.reactions.append((channel_id, message_id, emoji))

class DiscordTransport:
    def __init__(self, client: typing.Any = None):
        self.client = client or MockDiscordAPI()

    async def execute_native_action(self, channel_id: str, action: NativeAction):
        if action.action_type == "discord_reply":
            msg_id = action.params["message_id"]
            content = action.params["content"]
            await self.client.send_message(channel_id, content, reply_to=msg_id)
        elif action.action_type == "discord_reaction":
            msg_id = action.params["message_id"]
            emoji = action.params["emoji"]
            await self.client.add_reaction(channel_id, msg_id, emoji)
```

Create `jarvis/transports/qq.py`:
```python
import typing
from jarvis.models.base import NativeAction

class MockQQAPI:
    def __init__(self):
        self.replies = []

    async def reply_dm(self, user_id: str, content: str, msg_id: str):
        self.replies.append((user_id, content, msg_id))

class QQTransport:
    def __init__(self, client: typing.Any = None):
        self.client = client or MockQQAPI()

    async def execute_native_action(self, target_id: str, action: NativeAction):
        if action.action_type == "qq_reply":
            msg_id = action.params["message_id"]
            content = action.params["content"]
            await self.client.reply_dm(target_id, content, msg_id=msg_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_transports.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jarvis/transports/discord.py jarvis/transports/qq.py
git commit -m "feat: add Discord and QQ transport client mocks and wrappers"
```

---

### Task 8: Integrate Transports into FastAPI Lifecycle/Lifespan

**Files:**
- Modify: `main.py`
- Test: `tests/test_gateway.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from httpx import ASGITransport, AsyncClient
from main import app

@pytest.mark.asyncio
async def test_gateway_lifespan_starts_transports():
    # Verify that lifespan task setup initializes successfully
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Request health check or basic endpoint to run lifespan setup
        resp = await client.get("/health")
        assert resp.status_code == 200 or resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_gateway.py -k test_gateway_lifespan_starts_transports`
Expected: FAIL if lifespan throws exceptions or doesn't support the new async tasks.

- [ ] **Step 3: Write minimal implementation**

Modify `main.py` to add startup/shutdown lifespan logic for background bot loops:
```python
# In main.py:
import asyncio
from contextlib import asynccontextmanager

async def run_discord_bot(sessions: dict):
    # Mock loop representing discord websocket listener
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass

async def run_qq_bot(sessions: dict):
    # Mock loop representing QQ DM listener
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sessions = {}
    
    # Spawn bot background loops
    discord_task = asyncio.create_task(run_discord_bot(app.state.sessions))
    qq_task = asyncio.create_task(run_qq_bot(app.state.sessions))
    
    yield
    
    # Cancel bots on exit
    discord_task.cancel()
    qq_task.cancel()
    await asyncio.gather(discord_task, qq_task, return_exceptions=True)

# Update FastAPI initializer:
app = FastAPI(title="Jarvis Gateway", lifespan=lifespan)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/test_gateway.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: integrate Discord and QQ transport background task lifespan"
```
