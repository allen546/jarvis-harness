import pytest
import json
from pathlib import Path
from jarvis.memory_store import SemanticMemoryStore, purge_semantic_memory_tool
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState, current_context
from jarvis.tools import ToolRegistry


@pytest.mark.asyncio
async def test_purge_semantic_memory(tmp_path: Path):
    store = SemanticMemoryStore(storage_dir=str(tmp_path), embedding_url="http://test/embeddings")
    session_id = "s_purge"

    # Pre-populate index directly to avoid HTTP calls
    memories = [
        {"id": "id1", "text": "Apple is good.", "embedding": [0.1], "tags": ["truths"], "kind": "fact", "scope": "global"},
        {"id": "id2", "text": "Banana is yellow.", "embedding": [0.2], "tags": ["history"], "kind": "fact", "scope": "global"}
    ]
    store._save("__global__", memories)

    # 1. Test direct purge by tag
    await store.purge("__global__", tag="history")
    remaining = store._load("__global__")
    assert len(remaining) == 1
    assert remaining[0]["id"] == "id1"

    # 2. Test direct purge by ID
    await store.purge("__global__", ids=["id1"])
    assert len(store._load("__global__")) == 0

    # 3. Test Tool Callback execution
    # Populate again
    store._save("__global__", [
        {"id": "id3", "text": "Apple is good.", "embedding": [0.1], "tags": ["truths"], "kind": "fact", "scope": "global"}
    ])
    from jarvis.models.base import BaseModelClient
    class DummyModel(BaseModelClient):
        pass
    # Include a hook so _get_store_from_context picks up the right storage_dir
    from jarvis.memory_store import MemoryDistillationHook
    hook = MemoryDistillationHook(storage_dir=str(tmp_path))
    ctx = AgentContext(
        config=RuntimeConfig(),
        session=SessionState(id=session_id),
        model=DummyModel(),
        tools=ToolRegistry(),
        hooks=[hook],
    )
    token = current_context.set(ctx)
    try:
        res = await purge_semantic_memory_tool({"ids": ["id3"]})
        assert "1" in res
        assert len(store._load("__global__")) == 0
    finally:
        current_context.reset(token)


@pytest.mark.asyncio
async def test_purge_by_kind(tmp_path: Path):
    """Purge should support kind filter with new API."""
    store = SemanticMemoryStore(storage_dir=str(tmp_path), embedding_url="http://test/embeddings")

    # Seed mixed kinds
    memories = [
        {"id": "id1", "text": "A fact.", "kind": "fact", "scope": "global", "tags": ["truths"], "embedding": [0.1]},
        {"id": "id2", "text": "A procedure.", "kind": "procedure", "scope": "global", "tags": ["workflow"], "embedding": [0.2]},
        {"id": "id3", "text": "Another fact.", "kind": "fact", "scope": "global", "tags": ["truths"], "embedding": [0.3]},
    ]
    store._save("__global__", memories)

    # Purge by kind
    count = await store.purge("__global__", kind="procedure")
    assert count == 1
    remaining = store._load("__global__")
    assert len(remaining) == 2
    assert all(m["kind"] != "procedure" for m in remaining)


@pytest.mark.asyncio
async def test_purge_no_selector_returns_zero(tmp_path: Path):
    """Without any selector (ids, tag, kind), purge should return 0."""
    store = SemanticMemoryStore(storage_dir=str(tmp_path), embedding_url="http://test/embeddings")
    store._save("__global__", [
        {"id": "id1", "text": "Some fact.", "kind": "fact", "scope": "global", "tags": ["truths"], "embedding": [0.1]},
    ])
    count = await store.purge("__global__")
    assert count == 0
