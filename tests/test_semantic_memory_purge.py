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
        {"id": "id1", "text": "Apple is good.", "embedding": [0.1], "tags": ["truths"]},
        {"id": "id2", "text": "Banana is yellow.", "embedding": [0.2], "tags": ["history"]}
    ]
    store._save(session_id, memories)

    # 1. Test direct purge by tag
    await store.purge(session_id, tag="history")
    remaining = store._load(session_id)
    assert len(remaining) == 1
    assert remaining[0]["id"] == "id1"

    # 2. Test direct purge by ID
    await store.purge(session_id, ids=["id1"])
    assert len(store._load(session_id)) == 0

    # 3. Test Tool Callback execution
    # Populate again
    store._save(session_id, [
        {"id": "id3", "text": "Apple is good.", "embedding": [0.1], "tags": ["truths"]}
    ])
    from jarvis.memory_store import SemanticMemoryHook
    hook = SemanticMemoryHook(
        storage_dir=str(tmp_path),
        embedding_url="http://test/embeddings"
    )
    ctx = AgentContext(
        config=RuntimeConfig(),
        session=SessionState(id=session_id),
        model=None,
        tools=ToolRegistry(),
        hooks=[hook]
    )
    token = current_context.set(ctx)
    try:
        res = await purge_semantic_memory_tool({"ids": ["id3"]})
        assert "1" in res
        assert len(store._load(session_id)) == 0
    finally:
        current_context.reset(token)
