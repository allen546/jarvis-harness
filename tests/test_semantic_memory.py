import pytest
import json
from pathlib import Path
from typing import Any
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jarvis.memory_store import SemanticMemoryStore, MemoryDistillationHook, MemoryInjectionHook
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState, current_context
from jarvis.tools import ToolRegistry
from jarvis.models.base import Message, BaseModelClient, ModelResponse

app = FastAPI()
@app.post("/embeddings")
def get_emb(data: dict):
    val = 0.1 if "truth" in data["text"] else 0.5
    return {"embedding": [val] * 4}


class SequenceModel(BaseModelClient):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.calls: list[list[Message]] = []

    async def generate(self, messages: list[Message], tools: list[Any]) -> ModelResponse:
        self.calls.append(list(messages))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_semantic_memory_store_and_distillation(tmp_path: Path):
    with TestClient(app) as client:
        store = SemanticMemoryStore(
            storage_dir=str(tmp_path),
            embedding_url="http://test/embeddings",
            http_client=client,
        )
        session_id = "s_mem"

        # 1. Direct store operations with new typed API
        await store.add_memory(session_id, "This is a key truth.", ["truths"], kind="fact", scope="global")
        await store.add_memory(session_id, "Normal chat event history.", ["history"], kind="fact", scope="global")

        results = await store.search(session_id, "query truth", tag="truths", limit=1, scope="global")
        assert len(results) == 1
        assert "truth" in results[0]["text"]
        assert "truths" in results[0]["tags"]
        assert results[0].get("kind") == "fact"
        assert "score" in results[0]

        # 2. Test MemoryDistillationHook
        hook = MemoryDistillationHook(
            storage_dir=str(tmp_path),
            embedding_url="http://test/embeddings",
            http_client=client,
            embedding_dimensions=4,
            scope="global",
            distill_interval_turns=0,  # disable periodic, we test distill_now directly
        )
        state = SessionState(id="sess_hook")
        model = SequenceModel([
            ModelResponse(content=json.dumps({
                "facts": [{"text": "User likes apples", "tags": ["truths"], "confidence": 0.9}],
                "procedures": [],
            }))
        ])
        ctx = AgentContext(
            config=RuntimeConfig(),
            session=state,
            model=model,
            tools=ToolRegistry(),
            hooks=[hook],
        )

        state.history = [
            Message(role="user", content="I like apples", metadata={}),
            Message(role="assistant", content="Noted!", metadata={}),
        ]

        # Use distill_now
        result = await hook.distill_now(ctx)
        assert "2 messages" in result

        # Verify fact was written to global memory
        global_memories = store._load("__global__")
        texts = [m["text"] for m in global_memories]
        assert any("apples" in t for t in texts)


@pytest.mark.asyncio
async def test_memory_injection_hook_adds_relevant_global_facts(tmp_path: Path):
    with TestClient(app) as client:
        store = SemanticMemoryStore(
            storage_dir=str(tmp_path),
            embedding_url="http://test/embeddings",
            http_client=client,
        )
        # Seed global memory with identity fact
        await store.add_memory("test", "User lives in Shanghai", ["identity"], kind="fact", scope="global")

        hook = MemoryInjectionHook(
            storage_dir=str(tmp_path),
            embedding_url="http://test/embeddings",
            http_client=client,
            embedding_dimensions=4,
            top_facts=3,
            min_score=0.1,
            scope="global",
        )
        state = SessionState(id="test_inject")
        ctx = AgentContext(
            config=RuntimeConfig(),
            session=state,
            model=SequenceModel([ModelResponse()]),
            tools=ToolRegistry(),
            hooks=[hook],
        )

        messages = [
            Message(role="system", content="You are Jarvis."),
            Message(role="user", content="Where do I live?", metadata={}),
        ]

        res = await hook.before_model(ctx, messages)
        assert res.messages is not None
        # Should have 3 messages: system, injection, user
        assert len(res.messages) == 3
        assert res.messages[0].role == "system"
        assert res.messages[0].content == "You are Jarvis."
        assert res.messages[1].role == "user"
        assert "Shanghai" in res.messages[1].content
        assert res.messages[1].metadata.get("memory_kind") == "long_term_memory_injection"
        assert res.messages[2].content == "Where do I live?"


@pytest.mark.asyncio
async def test_memory_injection_hook_skips_low_relevance(tmp_path: Path):
    with TestClient(app) as client:
        store = SemanticMemoryStore(
            storage_dir=str(tmp_path),
            embedding_url="http://test/embeddings",
            http_client=client,
        )
        # Seed with unrelated fact
        await store.add_memory("test", "Quantum physics is complex", ["science"], kind="fact", scope="global")

        hook = MemoryInjectionHook(
            storage_dir=str(tmp_path),
            embedding_url="http://test/embeddings",
            embedding_dimensions=4,
            min_score=0.9,  # very high threshold
            scope="global",
        )
        state = SessionState(id="test_skip")
        ctx = AgentContext(
            config=RuntimeConfig(),
            session=state,
            model=SequenceModel([ModelResponse()]),
            tools=ToolRegistry(),
            hooks=[hook],
        )

        messages = [
            Message(role="user", content="What's the weather?", metadata={}),
        ]

        res = await hook.before_model(ctx, messages)
        assert res.messages is None


@pytest.mark.asyncio
async def test_search_semantic_memory_tool(tmp_path: Path):
    with TestClient(app) as client:
        store = SemanticMemoryStore(
            storage_dir=str(tmp_path),
            embedding_url="http://test/embeddings",
            http_client=client,
        )
        await store.add_memory("s1", "Fact one", ["truths"], kind="fact", scope="global")
        await store.add_memory("s1", "Procedure for tasks", ["procedure"], kind="procedure", scope="global")

        from jarvis.memory_store import search_semantic_memory_tool
        token = current_context.set(AgentContext(
            config=RuntimeConfig(),
            session=SessionState(id="s1"),
            model=SequenceModel([ModelResponse()]),
            tools=ToolRegistry(),
            hooks=[MemoryInjectionHook(
                storage_dir=str(tmp_path),
                embedding_url="http://test/embeddings",
                http_client=client,
                embedding_dimensions=4,
            )],
        ))
        try:
            result = await search_semantic_memory_tool({"query": "fact", "scope": "global"})
            parsed = json.loads(result)
            assert len(parsed) >= 1
        finally:
            current_context.reset(token)


@pytest.mark.asyncio
async def test_memory_distillation_invalid_json_is_nonfatal(tmp_path: Path):
    hook = MemoryDistillationHook(
        storage_dir=str(tmp_path),
        embedding_url=None,
        scope="global",
        distill_interval_turns=0,
    )
    state = SessionState(id="bad_json")
    model = SequenceModel([ModelResponse(content="not json at all")])
    ctx = AgentContext(
        config=RuntimeConfig(),
        session=state,
        model=model,
        tools=ToolRegistry(),
        hooks=[hook],
    )
    state.history = [
        Message(role="user", content="hello", metadata={}),
        Message(role="assistant", content="hi", metadata={}),
    ]

    # Should not raise
    result = await hook.distill_now(ctx)
    assert "Distilled" in result

    # No facts should be stored
    store = SemanticMemoryStore(storage_dir=str(tmp_path))
    memories = store._load("__global__")
    assert len(memories) == 0
