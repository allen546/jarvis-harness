"""Tests for write tool and CharNgramEncoder local embedding fallback."""

import pytest
from pathlib import Path
from jarvis.tools import builtin_tools, ToolRegistry


# ── Write tool ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_tool_creates_file(tmp_path: Path):
    tools = ToolRegistry(builtin_tools(tmp_path))
    result = await tools.execute(
        __import__("jarvis.models.base", fromlist=["ToolCall"]).ToolCall(
            call_id="1", tool_name="write", arguments={"path": "hello.txt", "content": "hello world"}
        )
    )
    assert not result.is_error
    assert "Wrote" in result.content
    assert (tmp_path / "hello.txt").read_text() == "hello world"


@pytest.mark.asyncio
async def test_write_tool_overwrites(tmp_path: Path):
    tools = ToolRegistry(builtin_tools(tmp_path))
    tc1 = __import__("jarvis.models.base", fromlist=["ToolCall"]).ToolCall(
        call_id="1", tool_name="write", arguments={"path": "f.txt", "content": "first"}
    )
    tc2 = __import__("jarvis.models.base", fromlist=["ToolCall"]).ToolCall(
        call_id="2", tool_name="write", arguments={"path": "f.txt", "content": "second"}
    )
    await tools.execute(tc1)
    result = await tools.execute(tc2)
    assert not result.is_error
    assert (tmp_path / "f.txt").read_text() == "second"


@pytest.mark.asyncio
async def test_write_tool_creates_parent_dirs(tmp_path: Path):
    tools = ToolRegistry(builtin_tools(tmp_path))
    result = await tools.execute(
        __import__("jarvis.models.base", fromlist=["ToolCall"]).ToolCall(
            call_id="1", tool_name="write", arguments={"path": "a/b/c.txt", "content": "nested"}
        )
    )
    assert not result.is_error
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "nested"


@pytest.mark.asyncio
async def test_write_tool_cannot_escape_workspace(tmp_path: Path):
    tools = ToolRegistry(builtin_tools(tmp_path))
    result = await tools.execute(
        __import__("jarvis.models.base", fromlist=["ToolCall"]).ToolCall(
            call_id="1", tool_name="write", arguments={"path": "../escape.txt", "content": "bad"}
        )
    )
    assert result.is_error

@pytest.mark.asyncio
async def test_write_tool_rejects_oversized_content(tmp_path: Path):
    tools = ToolRegistry(builtin_tools(tmp_path))
    huge_content = "x" * 1_048_577  # 1 byte over limit
    result = await tools.execute(
        __import__("jarvis.models.base", fromlist=["ToolCall"]).ToolCall(
            call_id="1", tool_name="write", arguments={"path": "big.txt", "content": huge_content}
        )
    )
    assert "exceeds" in result.content
    assert not (tmp_path / "big.txt").exists()


@pytest.mark.asyncio
async def test_write_tool_in_registry():
    tools = ToolRegistry(builtin_tools("."))
    tool_names = [t["name"] for t in tools.schemas()]
    assert "write" in tool_names


# ── CharNgramEncoder ────────────────────────────────────────────────────────


def test_char_ngram_encoder_deterministic():
    from jarvis.memory_store import _CharNgramEncoder
    enc = _CharNgramEncoder(dimensions=64)
    v1 = enc.encode("hello world")
    v2 = enc.encode("hello world")
    assert v1 == v2


def test_char_ngram_encoder_dimension():
    from jarvis.memory_store import _CharNgramEncoder
    enc = _CharNgramEncoder(dimensions=128)
    vec = enc.encode("test")
    assert len(vec) == 128


def test_char_ngram_encoder_normalized():
    from jarvis.memory_store import _CharNgramEncoder
    enc = _CharNgramEncoder(dimensions=64)
    vec = enc.encode("some text here")
    import math
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6


def test_char_ngram_encoder_empty():
    from jarvis.memory_store import _CharNgramEncoder
    enc = _CharNgramEncoder(dimensions=64)
    vec = enc.encode("")
    assert all(x == 0.0 for x in vec)


def test_char_ngram_encoder_similarity():
    from jarvis.memory_store import _CharNgramEncoder
    from jarvis.memory_store import cosine_similarity
    enc = _CharNgramEncoder(dimensions=256)
    v1 = enc.encode("the quick brown fox")
    v2 = enc.encode("the quick brown dog")
    v3 = enc.encode("completely different text about something else")
    sim_close = cosine_similarity(v1, v2)
    sim_far = cosine_similarity(v1, v3)
    assert sim_close > sim_far


@pytest.mark.asyncio
async def test_semantic_store_local_fallback(tmp_path: Path):
    """SemanticMemoryStore works with embedding_url=None (local encoder)."""
    from jarvis.memory_store import SemanticMemoryStore
    store = SemanticMemoryStore(
        storage_dir=str(tmp_path),
        embedding_url=None,
        embedding_dimensions=64,
    )
    assert store._local_encoder is not None
    await store.add_memory("s1", "Python is a programming language", ["truths"])
    await store.add_memory("s1", "The sky is blue", ["facts"])
    results = await store.search("s1", "coding language")
    assert len(results) >= 1
    assert "Python" in results[0]["text"]


@pytest.mark.asyncio
async def test_semantic_store_http_still_works(tmp_path: Path):
    """Existing HTTP path still works when embedding_url is provided."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from jarvis.memory_store import SemanticMemoryStore

    app = FastAPI()

    @app.post("/embeddings")
    def get_emb(data: dict):
        return {"embedding": [0.1] * 4}

    with TestClient(app) as client:
        store = SemanticMemoryStore(
            storage_dir=str(tmp_path),
            embedding_url="http://test/embeddings",
            http_client=client,
        )
        assert store._local_encoder is None
        await store.add_memory("s1", "test fact", ["truths"])
        results = await store.search("s1", "test query")
        assert len(results) == 1
