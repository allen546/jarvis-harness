from __future__ import annotations

import os
import json
import math
import zlib
import uuid
import inspect
import asyncio
import warnings
import datetime
from pathlib import Path
from typing import Any

from jarvis.hooks import NoopTurnHook, HookResult
from jarvis.models.base import Message

GLOBAL_MEMORY_KEY = "__global__"
MEMORY_SCHEMA_VERSION = 1
LEGACY_MIGRATION_MARKER = ".legacy_semantic_memory_migrated_v1"


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


class _CharNgramEncoder:
    """Local char n-gram TF-IDF encoder. No external dependencies."""

    def __init__(self, dimensions: int = 256, n: int = 3) -> None:
        self.dimensions = dimensions
        self.n = n

    def encode(self, text: str) -> list[float]:
        lower = text.lower()
        counts: dict[str, int] = {}
        for i in range(max(0, len(lower) - self.n + 1)):
            gram = lower[i : i + self.n]
            counts[gram] = counts.get(gram, 0) + 1
        if not counts:
            return [0.0] * self.dimensions
        vec = [0.0] * self.dimensions
        total = sum(counts.values())
        for gram, count in counts.items():
            bucket = zlib.crc32(gram.encode()) % self.dimensions
            vec[bucket] += count / total
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


def _normalize_dedup_key(text: str, kind: str) -> str:
    return f"{' '.join(text.casefold().split())}|{kind}"


class SemanticMemoryStore:
    _locks: dict[str, asyncio.Lock] = {}
    _migration_done: bool = False

    @classmethod
    def _get_lock(cls, session_id: str) -> asyncio.Lock:
        if session_id not in cls._locks:
            cls._locks[session_id] = asyncio.Lock()
        return cls._locks[session_id]

    def __init__(
        self,
        storage_dir: str,
        embedding_url: str | None = None,
        http_client: Any | None = None,
        embedding_dimensions: int = 256,
    ) -> None:
        self.storage_dir = storage_dir
        self.embedding_url = embedding_url
        self.http_client = http_client
        self._local_encoder: _CharNgramEncoder | None = (
            _CharNgramEncoder(dimensions=embedding_dimensions) if not embedding_url else None
        )

    def _get_file_path(self, session_id: str) -> Path:
        if session_id == GLOBAL_MEMORY_KEY:
            return Path(self.storage_dir) / "memory" / "semantic_memory.json"
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
        if self._local_encoder is not None:
            return self._local_encoder.encode(text)
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

    def _migrate_legacy_session_memories(self) -> None:
        if SemanticMemoryStore._migration_done:
            return
        marker_path = Path(self.storage_dir) / "memory" / LEGACY_MIGRATION_MARKER
        if marker_path.exists():
            SemanticMemoryStore._migration_done = True
            return
        sessions_dir = Path(self.storage_dir) / "sessions"
        if not sessions_dir.exists():
            SemanticMemoryStore._migration_done = True
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text("migrated")
            return
        global_memories = self._load(GLOBAL_MEMORY_KEY)
        global_dedup = {_normalize_dedup_key(m["text"], m.get("kind", "fact")) for m in global_memories}
        for session_dir in sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            sm_file = session_dir / "semantic_memory.json"
            if not sm_file.exists():
                continue
            try:
                with open(sm_file, "r", encoding="utf-8") as f:
                    old_records = json.load(f)
            except Exception:
                continue
            sid = session_dir.name
            for old in old_records:
                text = old.get("text", "")
                kind = old.get("kind", "fact")
                dedup = _normalize_dedup_key(text, kind)
                if dedup in global_dedup:
                    continue
                new_record = {
                    "schema_version": MEMORY_SCHEMA_VERSION,
                    "id": old.get("id", str(uuid.uuid4())),
                    "kind": kind,
                    "text": text,
                    "tags": old.get("tags", ["truths"]),
                    "scope": "global",
                    "source_session_ids": [sid],
                    "confidence": 1.0,
                    "observations": 1,
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "metadata": {},
                    "embedding": old.get("embedding", []),
                }
                global_memories.append(new_record)
                global_dedup.add(dedup)
        self._save(GLOBAL_MEMORY_KEY, global_memories)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("migrated")
        SemanticMemoryStore._migration_done = True

    async def add_memory(
        self,
        session_id: str,
        text: str,
        tags: list[str],
        *,
        kind: str = "fact",
        scope: str = "global",
        metadata: dict[str, Any] | None = None,
        confidence: float = 1.0,
    ) -> str:
        if scope == "global":
            target = GLOBAL_MEMORY_KEY
        elif scope == "session":
            target = session_id
        else:
            raise ValueError(f"unsupported memory scope: {scope}")

        if scope in ("global", "both"):
            self._migrate_legacy_session_memories()

        embedding = await self._get_embedding(text)
        dedup_key = _normalize_dedup_key(text, kind)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        async with self._get_lock(target):
            memories = self._load(target)
            # Check for existing dedup
            for m in memories:
                existing_key = _normalize_dedup_key(m.get("text", ""), m.get("kind", "fact"))
                if existing_key == dedup_key:
                    m["observations"] = m.get("observations", 0) + 1
                    m["updated_at"] = now
                    m["tags"] = list(set(m.get("tags", []) + tags))
                    source_sids = m.get("source_session_ids", [])
                    if session_id not in source_sids:
                        source_sids.append(session_id)
                    m["source_session_ids"] = source_sids
                    if metadata:
                        m_meta = m.get("metadata", {})
                        m_meta.update(metadata)
                        m["metadata"] = m_meta
                    self._save(target, memories)
                    return m["id"]
            # New record
            record = {
                "schema_version": MEMORY_SCHEMA_VERSION,
                "id": str(uuid.uuid4()),
                "kind": kind,
                "text": text,
                "tags": list(tags),
                "scope": scope,
                "source_session_ids": [session_id],
                "confidence": confidence,
                "observations": 1,
                "created_at": now,
                "updated_at": now,
                "metadata": metadata or {},
                "embedding": embedding,
            }
            memories.append(record)
            self._save(target, memories)
            return record["id"]

    async def search(
        self,
        session_id: str,
        query: str,
        tag: str | None = None,
        limit: int = 5,
        *,
        kind: str | None = None,
        scope: str = "global",
    ) -> list[dict[str, Any]]:
        if scope in ("global", "both"):
            self._migrate_legacy_session_memories()

        if scope == "global":
            sources = [(GLOBAL_MEMORY_KEY, "global")]
        elif scope == "session":
            sources = [(session_id, "session")]
        elif scope == "both":
            sources = [(GLOBAL_MEMORY_KEY, "global"), (session_id, session_id)]
        else:
            raise ValueError(f"unsupported memory scope: {scope}")

        all_memories: list[tuple[str, dict[str, Any]]] = []
        for src_key, src_scope in sources:
            for m in self._load(src_key):
                all_memories.append((src_scope, m))

        if not all_memories:
            return []

        # Apply filters
        filtered: list[tuple[str, dict[str, Any]]] = []
        for src_scope, m in all_memories:
            if tag is not None and tag not in m.get("tags", []):
                continue
            if kind is not None and m.get("kind", "fact") != kind:
                continue
            filtered.append((src_scope, m))

        if not filtered:
            return []

        query_emb = await self._get_embedding(query)
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for src_scope, m in filtered:
            emb = m.get("embedding")
            if emb:
                sim = cosine_similarity(query_emb, emb)
                scored.append((sim, src_scope, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, Any]] = []
        for sim, src_scope, m in scored[:limit]:
            r = dict(m)
            r["score"] = sim
            r["source_scope"] = src_scope
            results.append(r)
        return results

    async def purge(
        self,
        session_id: str,
        ids: list[str] | None = None,
        tag: str | None = None,
        *,
        kind: str | None = None,
        scope: str = "global",
    ) -> int:
        if scope in ("global", "both"):
            self._migrate_legacy_session_memories()

        if scope == "global":
            sources = [GLOBAL_MEMORY_KEY]
        elif scope == "session":
            sources = [session_id]
        elif scope == "both":
            sources = [GLOBAL_MEMORY_KEY, session_id]
        else:
            raise ValueError(f"unsupported memory scope: {scope}")

        total_purged = 0
        for src_key in sources:
            async with self._get_lock(src_key):
                memories = self._load(src_key)
                original_len = len(memories)
                if ids is None and tag is None and kind is None:
                    continue
                new_memories = []
                for m in memories:
                    keep = True
                    if ids is not None and m.get("id") in ids:
                        keep = False
                    if tag is not None and tag in m.get("tags", []):
                        keep = False
                    if kind is not None and m.get("kind") == kind:
                        keep = False
                    if keep:
                        new_memories.append(m)
                purged = original_len - len(new_memories)
                if purged > 0:
                    self._save(src_key, new_memories)
                    total_purged += purged
        return total_purged

    async def update_record(
        self,
        record_id: str,
        *,
        text: str | None = None,
        tags: list[str] | None = None,
        confidence: float | None = None,
        scope: str = "global",
    ) -> bool:
        target = GLOBAL_MEMORY_KEY if scope == "global" else record_id
        if scope in ("global", "both"):
            self._migrate_legacy_session_memories()
        if scope == "both":
            for src in [GLOBAL_MEMORY_KEY, record_id]:
                async with self._get_lock(src):
                    memories = self._load(src)
                    for m in memories:
                        if m.get("id") == record_id:
                            if text is not None:
                                m["text"] = text
                            if tags is not None:
                                m["tags"] = list(tags)
                            if confidence is not None:
                                m["confidence"] = confidence
                            m["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                            self._save(src, memories)
                            return True
            return False

        async with self._get_lock(target):
            memories = self._load(target)
            for m in memories:
                if m.get("id") == record_id:
                    if text is not None:
                        m["text"] = text
                    if tags is not None:
                        m["tags"] = list(tags)
                    if confidence is not None:
                        m["confidence"] = confidence
                    m["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    self._save(target, memories)
                    return True
        return False

    async def merge_records(
        self,
        id_a: str,
        id_b: str,
        *,
        scope: str = "global",
    ) -> str | None:
        target = GLOBAL_MEMORY_KEY if scope == "global" else id_a
        if scope in ("global", "both"):
            self._migrate_legacy_session_memories()
        if scope == "both":
            target = GLOBAL_MEMORY_KEY

        async with self._get_lock(target):
            memories = self._load(target)
            rec_a = None
            rec_b = None
            for m in memories:
                if m.get("id") == id_a:
                    rec_a = m
                if m.get("id") == id_b:
                    rec_b = m
            if rec_a is None or rec_b is None:
                return None
            # Merge: keep rec_a, merge rec_b into it
            if len(rec_b.get("text", "")) > len(rec_a.get("text", "")):
                rec_a["text"] = rec_b["text"]
            rec_a["tags"] = list(set(rec_a.get("tags", []) + rec_b.get("tags", [])))
            rec_a["observations"] = rec_a.get("observations", 0) + rec_b.get("observations", 0)
            sids_a = set(rec_a.get("source_session_ids", []))
            sids_b = set(rec_b.get("source_session_ids", []))
            rec_a["source_session_ids"] = list(sids_a | sids_b)
            rec_a["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            memories = [m for m in memories if m.get("id") != id_b]
            self._save(target, memories)
            return rec_a["id"]


class MemoryInjectionHook(NoopTurnHook):
    __slots__ = (
        "storage_dir", "embedding_url", "http_client", "embedding_dimensions",
        "top_facts", "top_procedures", "min_score", "scope",
    )

    def __init__(
        self,
        storage_dir: str = "storage",
        embedding_url: str | None = None,
        http_client: Any | None = None,
        embedding_dimensions: int = 256,
        top_facts: int = 3,
        top_procedures: int = 2,
        min_score: float = 0.35,
        scope: str = "global",
    ) -> None:
        self.storage_dir = storage_dir
        self.embedding_url = embedding_url
        self.http_client = http_client
        self.embedding_dimensions = embedding_dimensions
        self.top_facts = top_facts
        self.top_procedures = top_procedures
        self.min_score = min_score
        self.scope = scope

    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
        try:
            # Find latest user message
            latest_user = None
            for m in reversed(messages):
                if m.role == "user" and m.metadata.get("memory_kind") != "long_term_memory_injection":
                    latest_user = m
                    break
            if latest_user is None or not (latest_user.content or "").strip():
                return HookResult()

            session = getattr(ctx, "session", None)
            if session is None:
                return HookResult()

            store = SemanticMemoryStore(
                storage_dir=self.storage_dir,
                embedding_url=self.embedding_url,
                http_client=self.http_client,
                embedding_dimensions=self.embedding_dimensions,
            )

            # Search facts
            fact_limit = max(self.top_facts * 3, self.top_facts)
            fact_results = await store.search(
                session.id, latest_user.content,
                kind="fact", scope=self.scope, limit=fact_limit,
            )
            # Search procedures
            proc_limit = max(self.top_procedures * 3, self.top_procedures)
            proc_results = await store.search(
                session.id, latest_user.content,
                kind="procedure", scope=self.scope, limit=proc_limit,
            )

            identity_tags = {"identity", "profile", "user", "preference"}

            # Filter facts
            filtered_facts = []
            for r in fact_results:
                score = r.get("score", 0.0)
                tags = set(r.get("tags", []))
                if score >= self.min_score:
                    filtered_facts.append(r)
                elif score >= 0.15 and tags & identity_tags:
                    filtered_facts.append(r)
            filtered_facts = filtered_facts[:self.top_facts]

            # Filter procedures
            filtered_procs = [r for r in proc_results if r.get("score", 0.0) >= self.min_score]
            filtered_procs = filtered_procs[:self.top_procedures]

            if not filtered_facts and not filtered_procs:
                return HookResult()

            # Build injection message
            sections = []
            if filtered_facts:
                sections.append("Facts:\n" + "\n".join(f"- {r['text']}" for r in filtered_facts))
            if filtered_procs:
                sections.append("Procedures:\n" + "\n".join(f"- {r['text']}" for r in filtered_procs))

            injection_content = (
                "[Relevant long-term memory for this turn]\n"
                + "\n".join(sections)
                + "\n[/Relevant long-term memory]"
            )

            injection_msg = Message(
                role="user",
                content=injection_content,
                metadata={"memory_kind": "long_term_memory_injection"},
            )

            # Remove any previous injection
            new_messages = [m for m in messages if m.metadata.get("memory_kind") != "long_term_memory_injection"]

            # Find insertion point: before the latest real user message
            insert_idx = len(new_messages)
            for i in range(len(new_messages) - 1, -1, -1):
                if new_messages[i].role == "user" and new_messages[i].metadata.get("memory_kind") != "long_term_memory_injection":
                    insert_idx = i
                    break

            new_messages.insert(insert_idx, injection_msg)
            return HookResult(messages=new_messages)
        except Exception as exc:
            warnings.warn(f"Error injecting long-term memory: {exc}")
            return HookResult()


class MemoryDistillationHook(NoopTurnHook):
    __slots__ = (
        "storage_dir", "embedding_url", "http_client", "embedding_dimensions",
        "scope", "auto_distill_skills", "skill_min_observations",
        "distill_interval_turns", "_turn_count",
    )

    def __init__(
        self,
        storage_dir: str = "storage",
        embedding_url: str | None = None,
        http_client: Any | None = None,
        embedding_dimensions: int = 256,
        scope: str = "global",
        auto_distill_skills: bool = True,
        skill_min_observations: int = 3,
        distill_interval_turns: int = 10,
    ) -> None:
        self.storage_dir = storage_dir
        self.embedding_url = embedding_url
        self.http_client = http_client
        self.embedding_dimensions = embedding_dimensions
        self.scope = scope
        self.auto_distill_skills = auto_distill_skills
        self.skill_min_observations = skill_min_observations
        self.distill_interval_turns = distill_interval_turns
        self._turn_count = 0

    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
        self._turn_count += 1
        if self.distill_interval_turns <= 0 or self._turn_count % self.distill_interval_turns != 0:
            return HookResult()

        session = getattr(ctx, "session", None)
        if session is None:
            return HookResult()

        undistilled = [m for m in session.history if not m.metadata.get("distilled")]
        if not undistilled:
            return HookResult()

        await self._distill_messages(ctx, session, undistilled)
        return HookResult()

    async def _distill_messages(self, ctx: object, session: Any, messages: list[Message]) -> None:
        model = getattr(ctx, "model", None)
        if model is None:
            return

        prompt_msgs = [
            Message(role="system", content=(
                "Extract durable memory from this Jarvis conversation segment. "
                "Return strict JSON only, no markdown.\n"
                "Schema: {\"facts\":[{\"text\":str,\"tags\":[str],\"confidence\":float}],"
                "\"procedures\":[{\"name\":str,\"trigger\":str,\"summary\":str,\"steps\":[str],"
                "\"tools\":[str],\"confidence\":float}]}\n"
                "Facts are stable user or environment truths. "
                "Procedures are reusable task patterns. Do not store transient chat."
            )),
            *messages,
        ]

        try:
            response = await model.generate(prompt_msgs, [])
            parsed = json.loads(response.content or "{}")
        except (json.JSONDecodeError, Exception) as exc:
            warnings.warn(f"Error in memory distillation: {exc}")
            return

        store = SemanticMemoryStore(
            storage_dir=self.storage_dir,
            embedding_url=self.embedding_url,
            http_client=self.http_client,
            embedding_dimensions=self.embedding_dimensions,
        )

        for fact in parsed.get("facts", []):
            text = fact.get("text", "")
            if text:
                await store.add_memory(
                    session.id, text, fact.get("tags", ["truths"]),
                    kind="fact", scope=self.scope,
                    metadata={"source": "distillation"},
                    confidence=fact.get("confidence", 1.0),
                )

        for proc in parsed.get("procedures", []):
            name = proc.get("name", "")
            steps = proc.get("steps", [])
            if name and steps:
                record_id = await store.add_memory(
                    session.id, proc.get("summary", name),
                    ["procedure", *proc.get("tools", [])],
                    kind="procedure", scope=self.scope,
                    metadata={
                        "name": name,
                        "trigger": proc.get("trigger", ""),
                        "steps": steps,
                        "tools": proc.get("tools", []),
                        "source": "distillation",
                    },
                    confidence=proc.get("confidence", 1.0),
                )
                # Auto skill distillation
                if self.auto_distill_skills:
                    await self._maybe_create_skill(ctx, store, record_id, proc)

        # Mark messages as distilled
        for m in messages:
            m.metadata["distilled"] = True

    async def _maybe_create_skill(self, ctx: object, store: SemanticMemoryStore, record_id: str, proc: dict) -> None:
        target = GLOBAL_MEMORY_KEY if self.scope == "global" else record_id
        async with store._get_lock(target):
            memories = store._load(target)
            record = None
            for m in memories:
                if m.get("id") == record_id:
                    record = m
                    break
        if record is None:
            return
        if record.get("observations", 0) < self.skill_min_observations:
            return

        from jarvis.skills import slugify_skill_name
        name = proc.get("name", "learned-procedure")
        slug = slugify_skill_name(name)
        skill_name = f"learned_{slug}"

        # Write to first configured skills directory
        config = getattr(ctx, "config", None)
        skills_dirs = ["skills/"]
        if config is not None:
            skills_dirs = getattr(config, "skills_dirs", skills_dirs)
        skill_dir = Path(skills_dirs[0]) / skill_name
        skill_file = skill_dir / "SKILL.md"
        try:
            skill_dir.mkdir(parents=True, exist_ok=True)
            steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(proc.get("steps", [])))
            content = (
                f"---\nname: {skill_name}\ndescription: Learned procedure: {name}\ntools: {{}}\n---\n"
                f"Use this learned procedure when: {proc.get('trigger', 'needed')}\n\n"
                f"Steps:\n{steps_text}\n\nSource: distilled from Jarvis procedural memory.\n"
            )
            tmp_path = skill_file.with_suffix(".tmp")
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(str(tmp_path), str(skill_file))
        except Exception as exc:
            warnings.warn(f"Error writing distilled skill {skill_name}: {exc}")

    async def distill_now(self, ctx: object) -> str:
        session = getattr(ctx, "session", None)
        if session is None:
            return "No active session."
        undistilled = [m for m in session.history if not m.metadata.get("distilled")]
        if not undistilled:
            return "Nothing to distill — all sections already processed."
        await self._distill_messages(ctx, session, undistilled)
        return f"Distilled {len(undistilled)} messages."




def _get_store_from_context(ctx: Any) -> SemanticMemoryStore:
    storage_dir = "storage"
    embedding_url: str | None = os.environ.get("EMBEDDING_URL") or None
    http_client = None
    embedding_dimensions = 256

    if ctx and hasattr(ctx, "hooks"):
        for hook in ctx.hooks:
            name = type(hook).__name__
            if name in ("MemoryDistillationHook", "MemoryInjectionHook"):
                storage_dir = getattr(hook, "storage_dir", storage_dir)
                embedding_url = getattr(hook, "embedding_url", embedding_url)
                http_client = getattr(hook, "http_client", http_client)
                embedding_dimensions = getattr(hook, "embedding_dimensions", embedding_dimensions)
                break

    return SemanticMemoryStore(
        storage_dir=storage_dir,
        embedding_url=embedding_url,
        http_client=http_client,
        embedding_dimensions=embedding_dimensions,
    )


def _get_scope_from_context(ctx: Any) -> str:
    if ctx and hasattr(ctx, "config"):
        mem = getattr(ctx.config, "memory", None)
        if mem is not None:
            return getattr(mem, "scope", "global")
    return "global"


async def search_semantic_memory_tool(args: dict[str, Any]) -> str:
    query = args["query"]
    tag = args.get("tag")
    kind = args.get("kind")
    scope = args.get("scope")
    limit = args.get("limit", 5)

    ctx = get_context()
    session_id = ctx.session.id if ctx and hasattr(ctx, "session") else "default"
    if scope is None:
        scope = _get_scope_from_context(ctx)
    store = _get_store_from_context(ctx)
    results = await store.search(session_id, query, tag=tag, limit=limit, kind=kind, scope=scope)
    return json.dumps(results)


async def purge_semantic_memory_tool(args: dict[str, Any]) -> str:
    tag = args.get("tag")
    ids = args.get("ids")
    kind = args.get("kind")
    scope = args.get("scope")

    ctx = get_context()
    session_id = ctx.session.id if ctx and hasattr(ctx, "session") else "default"
    if scope is None:
        scope = _get_scope_from_context(ctx)
    store = _get_store_from_context(ctx)
    count = await store.purge(session_id, ids=ids, tag=tag, kind=kind, scope=scope)
    return f"Purged {count} items from semantic memory."


async def store_semantic_memory_tool(args: dict[str, Any]) -> str:
    text = args["text"]
    tags = args.get("tags", ["truths"])
    kind = args.get("kind", "fact")
    scope = args.get("scope")
    metadata = args.get("metadata", {})
    confidence = args.get("confidence", 1.0)

    if kind == "history_summary":
        return "Error: history_summary can only be created by the system during session compression."

    ctx = get_context()
    session_id = ctx.session.id if ctx and hasattr(ctx, "session") else "default"
    if scope is None:
        scope = _get_scope_from_context(ctx)
    store = _get_store_from_context(ctx)
    record_id = await store.add_memory(session_id, text, tags, kind=kind, scope=scope, metadata=metadata, confidence=confidence)
    return f"Stored memory {record_id}: {text[:80]}"


async def check_redundancy_tool(args: dict[str, Any]) -> str:
    limit = args.get("limit", 10)
    ctx = get_context()
    scope = _get_scope_from_context(ctx)
    store = _get_store_from_context(ctx)
    session_id = ctx.session.id if ctx and hasattr(ctx, "session") else "default"

    all_records = await store.search(session_id, "memory record", scope=scope, limit=100)
    if len(all_records) < 2:
        return json.dumps([])

    pairs: list[dict[str, Any]] = []
    for i in range(len(all_records)):
        for j in range(i + 1, len(all_records)):
            a = all_records[i]
            b = all_records[j]
            emb_a = a.get("embedding", [])
            emb_b = b.get("embedding", [])
            if emb_a and emb_b:
                sim = cosine_similarity(emb_a, emb_b)
                if sim > 0.85:
                    pairs.append({
                        "id_a": a.get("id"),
                        "text_a": a.get("text", "")[:80],
                        "id_b": b.get("id"),
                        "text_b": b.get("text", "")[:80],
                        "similarity": round(sim, 3),
                    })
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return json.dumps(pairs[:limit])


async def distill_now_tool(args: dict[str, Any]) -> str:
    ctx = get_context()
    if ctx is None:
        return "No active context."
    for hook in getattr(ctx, "hooks", []):
        if isinstance(hook, MemoryDistillationHook):
            return await hook.distill_now(ctx)
    return "No MemoryDistillationHook found."


async def merge_memory_tool(args: dict[str, Any]) -> str:
    id_a = args["id_a"]
    id_b = args["id_b"]
    ctx = get_context()
    scope = _get_scope_from_context(ctx)
    store = _get_store_from_context(ctx)
    result = await store.merge_records(id_a, id_b, scope=scope)
    if result:
        return f"Merged into {result}."
    return "Could not find both records to merge."


async def update_memory_tool(args: dict[str, Any]) -> str:
    record_id = args["id"]
    text = args.get("text")
    tags = args.get("tags")
    confidence = args.get("confidence")
    ctx = get_context()
    scope = _get_scope_from_context(ctx)
    store = _get_store_from_context(ctx)
    found = await store.update_record(record_id, text=text, tags=tags, confidence=confidence, scope=scope)
    if found:
        return f"Updated record {record_id}."
    return f"Record {record_id} not found."
