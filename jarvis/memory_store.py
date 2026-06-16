from __future__ import annotations

import os
import json
import math
import uuid
import inspect
import asyncio
from pathlib import Path
from typing import Any

from jarvis.hooks import NoopTurnHook, HookResult
from jarvis.models.base import Message


def get_context() -> Any | None:
    from jarvis.runtime import current_context
    return current_context.get()


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm_v1 = math.sqrt(sum(x * x for x in v1))
    norm_v2 = math.sqrt(sum(x * x for x in v2))
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


class SemanticMemoryStore:
    _locks: dict[str, asyncio.Lock] = {}

    @classmethod
    def _get_lock(cls, session_id: str) -> asyncio.Lock:
        if session_id not in cls._locks:
            cls._locks[session_id] = asyncio.Lock()
        return cls._locks[session_id]

    def __init__(self, storage_dir: str, embedding_url: str, http_client: Any | None = None) -> None:
        self.storage_dir = storage_dir
        self.embedding_url = embedding_url
        self.http_client = http_client

    def _get_file_path(self, session_id: str) -> Path:
        return Path(self.storage_dir) / "sessions" / session_id / "semantic_memory.json"

    def _load(self, session_id: str) -> list[dict[str, Any]]:
        path = self._get_file_path(session_id)
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self, session_id: str, memories: list[dict[str, Any]]) -> None:
        path = self._get_file_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=2)
        os.replace(tmp_path, path)

    async def _get_embedding(self, text: str) -> list[float]:
        json_data = {"text": text}
        if self.http_client is None:
            import httpx
            async with httpx.AsyncClient() as client:
                res = await client.post(self.embedding_url, json=json_data)
                data = res.json()
        else:
            res = self.http_client.post(self.embedding_url, json=json_data)
            if inspect.isawaitable(res):
                response = await res
            else:
                response = res
            data = response.json()
        return data["embedding"]

    async def add_memory(self, session_id: str, text: str, tags: list[str]) -> None:
        embedding = await self._get_embedding(text)
        async with self._get_lock(session_id):
            memories = self._load(session_id)
            memories.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "tags": list(tags),
                "embedding": embedding,
            })
            self._save(session_id, memories)

    async def search(self, session_id: str, query: str, tag: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        memories = self._load(session_id)
        if not memories:
            return []

        # Filter by tag if requested
        if tag is not None:
            filtered = [m for m in memories if tag in m.get("tags", [])]
        else:
            filtered = memories

        if not filtered:
            return []

        query_emb = await self._get_embedding(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for m in filtered:
            emb = m.get("embedding")
            if emb:
                sim = cosine_similarity(query_emb, emb)
                scored.append((sim, m))

        # Sort descending by similarity score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    async def purge(self, session_id: str, ids: list[str] | None = None, tag: str | None = None) -> int:
        async with self._get_lock(session_id):
            memories = self._load(session_id)
            original_len = len(memories)
            if ids is not None or tag is not None:
                new_memories = []
                for m in memories:
                    keep = True
                    if ids is not None and m.get("id") in ids:
                        keep = False
                    if tag is not None and tag in m.get("tags", []):
                        keep = False
                    if keep:
                        new_memories.append(m)
                memories = new_memories
            purged = original_len - len(memories)
            self._save(session_id, memories)
            return purged


class SemanticMemoryHook(NoopTurnHook):
    __slots__ = ("storage_dir", "embedding_url", "http_client")

    def __init__(self, storage_dir: str, embedding_url: str, http_client: Any | None = None) -> None:
        self.storage_dir = storage_dir
        self.embedding_url = embedding_url
        self.http_client = http_client

    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
        from jarvis.runtime import current_context
        current_context.set(ctx)
        return HookResult()

    async def after_turn(self, ctx: object, message: Message) -> HookResult:
        try:
            from jarvis.runtime import current_context
            current_context.set(ctx)
            
            session = getattr(ctx, "session", None)
            if not session or not session.history:
                return HookResult()

            # Compile latest user and assistant messages
            user_msg = None
            assistant_msg = None
            for msg in reversed(session.history):
                if msg.role == "user" and user_msg is None:
                    user_msg = msg
                elif msg.role == "assistant" and assistant_msg is None:
                    assistant_msg = msg
                if user_msg is not None and assistant_msg is not None:
                    break

            compiled_parts = []
            if user_msg:
                compiled_parts.append(f"User: {user_msg.content}")
            if assistant_msg:
                compiled_parts.append(f"Assistant: {assistant_msg.content}")

            if not compiled_parts:
                return HookResult()

            compiled_text = "\n".join(compiled_parts)

            # Call model to extract facts
            system_prompt = (
                "Extract key facts from the following messages. "
                "Return each fact on a new line starting with 'Fact: '."
            )
            system_message = Message(role="system", content=system_prompt)
            user_message = Message(role="user", content=compiled_text)

            model = getattr(ctx, "model", None)
            if not model:
                return HookResult()

            response = await model.generate([system_message, user_message], [])
            
            facts = []
            if response and response.content:
                for line in response.content.splitlines():
                    line = line.strip()
                    if line.startswith("Fact:"):
                        fact = line[len("Fact:"):].strip()
                        if fact:
                            facts.append(fact)

            if facts:
                store = SemanticMemoryStore(
                    storage_dir=self.storage_dir,
                    embedding_url=self.embedding_url,
                    http_client=self.http_client,
                )
                for fact in facts:
                    await store.add_memory(session.id, fact, ["truths"])
        except Exception as exc:
            import warnings
            warnings.warn(f"Error in SemanticMemoryHook.after_turn: {exc}")

        return HookResult()


async def search_semantic_memory_tool(args: dict[str, Any]) -> str:
    query = args["query"]
    tag = args.get("tag")

    ctx = get_context()
    session_id = ctx.session.id if ctx and hasattr(ctx, "session") else "default"

    # Default settings
    storage_dir = "storage"
    embedding_url = os.environ.get("EMBEDDING_URL", "http://localhost:8000/embeddings")
    http_client = None

    # Retrieve matching hook settings if present
    if ctx and hasattr(ctx, "hooks"):
        for hook in ctx.hooks:
            if type(hook).__name__ == "SemanticMemoryHook":
                storage_dir = getattr(hook, "storage_dir", storage_dir)
                embedding_url = getattr(hook, "embedding_url", embedding_url)
                http_client = getattr(hook, "http_client", http_client)
                break

    store = SemanticMemoryStore(
        storage_dir=storage_dir,
        embedding_url=embedding_url,
        http_client=http_client,
    )
    results = await store.search(session_id, query, tag=tag)
    return json.dumps(results)


async def purge_semantic_memory_tool(args: dict[str, Any]) -> str:
    tag = args.get("tag")
    ids = args.get("ids")

    ctx = get_context()
    session_id = ctx.session.id if ctx and hasattr(ctx, "session") else "default"

    # Default settings
    storage_dir = "storage"
    embedding_url = os.environ.get("EMBEDDING_URL", "http://localhost:8000/embeddings")
    http_client = None

    # Retrieve matching hook settings if present
    if ctx and hasattr(ctx, "hooks"):
        for hook in ctx.hooks:
            if type(hook).__name__ == "SemanticMemoryHook":
                storage_dir = getattr(hook, "storage_dir", storage_dir)
                embedding_url = getattr(hook, "embedding_url", embedding_url)
                http_client = getattr(hook, "http_client", http_client)
                break

    store = SemanticMemoryStore(
        storage_dir=storage_dir,
        embedding_url=embedding_url,
        http_client=http_client,
    )
    count = await store.purge(session_id, ids=ids, tag=tag)
    return f"Purged {count} items from semantic memory."
