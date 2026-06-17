# Design Spec: Discord & QQ Transports, Interactive Subagents, and Advanced Memory

## Goal

Extend the gateway-first microkernel harness to support Discord and QQ channel transports in the same process, implement interactive collaborative subagents via recursive session tool execution, and build a modular memory system with JSONL history, autocompression, and tagged semantic memory (delegating heavy embedding generation to a separate HTTP service).

---

## Architecture & Transport Integration

### 1. Unified Process Runner
The Discord and QQ transports run as asynchronous event loop tasks in the same Python process as the FastAPI gateway.
*   **FastAPI Lifespan**: We register lifespan tasks to start and stop the Discord/QQ bot daemons when the FastAPI server boots/shuts down.
*   **Routing via AgentSession**: Transports do not require custom event buses or routing wrappers. They identify or generate a `session_id` (e.g. `discord_channel_123`) and submit messages directly to the agent kernel by calling:
    ```python
    session = get_or_create_session(session_id)
    async for event in session.submit(message):
        # Render and send event back to Discord/QQ
    ```
*   **Concurrency**: The internal lock in `AgentSession.submit` serializes turns per session while permitting parallel turn execution across different sessions.

### 2. Client Mocking for Tests
To allow offline/offline testing without actual credentials:
*   We define abstract Client protocols/interfaces for Discord and QQ operations (e.g., sending text, files, and reactions).
*   We implement concrete wrappers using the real SDK libraries (`discord.py` and `botpy`).
*   We implement mock wrappers (`MockDiscordClient`, `MockQQClient`) for the unit/integration test suites.

---

## Platform-Specific Native Actions

The model can request native platform behaviors via `NativeAction` events. Transports process these actions according to platform rules:

### 1. Discord Native Actions
*   **`discord_reaction`**: Add/remove an emoji reaction to a message.
    *   Parameters: `{"message_id": str, "emoji": str, "action": Literal["add", "remove"]}`
*   **`discord_reply`**: Reply to a message with a specific message ID.
    *   Parameters: `{"message_id": str, "content": str}`
*   **`discord_send_embed`**: Send a rich embed card.
    *   Parameters: `{"title": str, "description": str, "fields": list[dict], "color": int}`
*   **`discord_create_thread`**: Create a public thread on a message.
    *   Parameters: `{"message_id": str, "name": str}`

### 2. QQ Native Actions
*   **`qq_reply`**: Reply to a specific user message using passive `msg_id` within the allowed QQ DM reply window.
    *   Parameters: `{"message_id": str, "content": str}`
*   **`qq_send_markdown`**: Send Markdown content in the DM.
    *   Parameters: `{"content": str, "template_id": str | None}`
*   **`qq_send_keyboard`**: Send custom interactive inline button keyboards along with Markdown in the DM.
    *   Parameters: `{"content": str, "keyboard_schema": dict}`

### 3. Message Routing Behavior
By default, the agent transport always sends responses back to the same channel, thread, or direct message (DM) conversation that the user's query originated from. This routing is maintained automatically by passing channel and message identifiers in the message metadata, unless a tool explicitly specifies an alternative routing target.

---

## Interactive Collaborative Subagents

We implement subagents as tools exposed to the parent agent using a multi-tool conversation pattern.

### 1. Context and Registry Changes
*   **SessionState**: Modified to include a `metadata: dict[str, Any]` field (where `slots=True` is preserved).
*   **Event Routing**: We add an `emit_event: Callable[[AgentEvent], None] | None` callback to `AgentContext`. When subagents submit messages, their events are forwarded back to the parent's `emit_event` callback, routing progress (such as text deltas) back to the parent session's active transport stream.
*   **Active Subagents**: We maintain a runtime-level active subagent session registry: `active_subagents: dict[str, AgentSession] = {}`.

### 2. Parent-to-Subagent Tools
*   **`spawn_subagent(prompt: str, task_name: str, system_override: str | None = None) -> dict`**:
    *   Generates a `sub_session_id`.
    *   Creates a child `AgentContext` and `SessionState`. The child context inherits the parent's model client and configuration, but receives a filtered tool registry (to avoid infinite subagent spawning loops).
    *   Runs the first turn with the initial prompt.
    *   Saves the subagent session in `active_subagents` and stores the association in the parent's metadata.
    *   Returns the subagent's response content and the `sub_session_id`.
*   **`send_subagent_message(sub_session_id: str, message: str) -> dict`**:
    *   Looks up the subagent session by ID.
    *   Submits the follow-up message to the subagent session.
    *   Returns the response content.
*   **`close_subagent(sub_session_id: str) -> dict`**:
    *   Terminates/closes the subagent session, removing it from active registries.

---

## Memory

Memory features are implemented via `TurnHook` lifecycles and dedicated tools. Chat history is append-only and is never deleted.

### 1. History Persistence (JSONL)
*   **`JSONLHistoryHook`**:
    *   In `after_turn`, it appends new messages as JSON-serialized lines to `storage/sessions/{session_id}/history.jsonl`.
    *   When an `AgentSession` is loaded, it pre-populates `session.history` from this file.

### 2. Autocompression
*   **`ContextCompressionHook`**:
    *   In `before_model`, it monitors the history length. If it exceeds a threshold (e.g. 20 messages), it compiles the oldest 10 messages.
    *   It calls the model client with a system summarization prompt.
    *   It replaces the oldest 10 messages with a single summary message in `ctx.session.history`.

### 3. Tagged Semantic Memory with HTTP Embedding Delegate
*   **HTTP Embedding Delegate**: Embedding generation is delegated to an external process via HTTP (e.g., calling a local FastAPI microservice or a cloud service).
*   **`SemanticMemoryHook`**:
    *   In `after_turn`, it extracts key facts/statements from new messages.
    *   It calls the embedding service via HTTP and saves the facts with tags (such as `"truths"` and `"history"`) to `storage/sessions/{session_id}/semantic_memory.json`.
*   **`SemanticMemoryTool`**:
    *   Exposes `search_semantic_memory(query: str, tag: str | None = None) -> list[dict]` to the model.
    *   Calculates cosine similarity in-thread using the embedding retrieved from the HTTP service.
*   **`PurgeSemanticMemoryTool`**:
    *   Exposes `purge_semantic_memory(query: str | None = None, tag: str | None = None, ids: list[str] | None = None) -> str` to allow the model to delete or correct facts.

---

## Verification Plan

### Automated Tests
1.  **Transports**:
    *   Unit tests checking mock client message sending and event rendering.
    *   Tests verifying native actions are parsed and correctly mapped to mock client calls (e.g. confirming Discord ignores QQ DMs/keyboards and vice-versa).
2.  **Subagents**:
    *   Tests verifying `spawn_subagent`, `send_subagent_message`, and `close_subagent` state progression.
    *   Tests verifying subagent events (such as `TextDeltaEvent`) are correctly bubbled and routed to the parent's event stream.
3.  **Memory**:
    *   Tests checking JSONL persistence and session loading.
    *   Tests checking autocompression triggers and content replacement in `before_model`.
    *   Mocked HTTP embedding server tests verifying semantic index calculations, filtering by `"truths"` or `"history"`, and fact purging.
