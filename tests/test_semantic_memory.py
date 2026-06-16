import pytest
import json
from pathlib import Path
from typing import Any
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jarvis.memory_store import SemanticMemoryStore, SemanticMemoryHook
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState
from jarvis.tools import ToolRegistry
from jarvis.models.base import Message, BaseModelClient, ModelResponse
from jarvis.kernel import AgentKernel

app = FastAPI()
@app.post("/embeddings")
def get_emb(data: dict):
    # Mock embedding return: vector of 4 elements
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
async def test_semantic_memory_store_and_hook(tmp_path: Path):
    with TestClient(app) as client:
        store = SemanticMemoryStore(
            storage_dir=str(tmp_path),
            embedding_url="http://test/embeddings",
            http_client=client
        )
        session_id = "s_mem"

        # 1. Direct store operations
        await store.add_memory(session_id, "This is a key truth.", ["truths"])
        await store.add_memory(session_id, "Normal chat event history.", ["history"])

        results = await store.search(session_id, "query truth", tag="truths", limit=1)
        assert len(results) == 1
        assert "truth" in results[0]["text"]
        assert "truths" in results[0]["tags"]

        # 2. Test Hook extraction and indexing
        hook = SemanticMemoryHook(
            storage_dir=str(tmp_path),
            embedding_url="http://test/embeddings",
            http_client=client
        )
        state = SessionState(id="sess_hook")
        # Model returns extracted facts when hook calls it in after_turn
        model = SequenceModel([
            ModelResponse(content="Fact: User likes apples.\nFact: User is a teacher.")
        ])
        ctx = AgentContext(
            config=RuntimeConfig(),
            session=state,
            model=model,
            tools=ToolRegistry(),
            hooks=[hook]
        )
        
        # Simulate messages in history to extract from
        state.history = [
            Message(role="user", content="I am a teacher and I like apples."),
            Message(role="assistant", content="That's nice to know.")
        ]
        
        await hook.after_turn(ctx, state.history[-1])
        
        # Verify facts were written to disk
        memories = store._load("sess_hook")
        assert len(memories) == 2
        texts = [m["text"] for m in memories]
        assert "User likes apples" in texts or "User likes apples." in texts
        assert "User is a teacher" in texts or "User is a teacher." in texts


@pytest.mark.asyncio
async def test_search_semantic_memory_tool(tmp_path: Path):
    with TestClient(app) as client:
        store = SemanticMemoryStore(
            storage_dir=str(tmp_path),
            embedding_url="http://test/embeddings",
            http_client=client
        )
        session_id = "s_tool"
        await store.add_memory(session_id, "Fact: User likes bananas.", ["truths"])
        
        hook = SemanticMemoryHook(
            storage_dir=str(tmp_path),
            embedding_url="http://test/embeddings",
            http_client=client
        )
        
        from jarvis.runtime import current_context
        from jarvis.memory_store import search_semantic_memory_tool
        
        ctx = AgentContext(
            config=RuntimeConfig(),
            session=SessionState(id=session_id),
            model=SequenceModel([]),
            tools=ToolRegistry(),
            hooks=[hook]
        )
        
        token = current_context.set(ctx)
        try:
            res = await search_semantic_memory_tool({"query": "bananas", "tag": "truths"})
            results = json.loads(res)
            assert len(results) == 1
            assert "bananas" in results[0]["text"]
        finally:
            current_context.reset(token)

