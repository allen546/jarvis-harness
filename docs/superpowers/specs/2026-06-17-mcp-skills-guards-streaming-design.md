# Design Spec: Jarvis Extensions (MCP, Skills, Guards, Streaming & Error Recovery)

This specification outlines the design and architecture for adding high-impact extensions, quality-of-life improvements, and resilience features to the Jarvis microkernel.

---

## 1. Goal

Enhance Jarvis with:
1. **MCP Client**: Dynamic connection to external MCP tool servers (both local stdio subprocesses and remote SSE/HTTP servers) using the official `mcp` SDK via a unified `ClientSessionGroup` manager, exposing their tools in `ToolRegistry`.
2. **Skill Loader**: Support for directory-based `SKILL.md` instruction sets and shell script tool definitions to extend capabilities without writing Python code.
3. **Tool Approval**: A hook-based confirmation step before executing tools (specifically `Bash` or any tool calls) supporting interactive TTY confirmation in the CLI and paused turns via HTTP endpoints in the Gateway.
4. **Budget Guards**: A turn-scoped hook checking consecutive tool call limits to prevent infinite tool loops.
5. **Streaming Text Deltas**: Updating the core kernel to stream text deltas token-by-token for a better UX.
6. **Error Recovery**: Automatic retries with exponential backoff for transient model client issues.

---

## 2. Architecture & Component Diagram

```text
+---------------------------------------------------------------------------------+
|                                 Session Context                                 |
|                                                                                 |
|  +------------------------+  +----------------------+  +---------------------+  |
|  |       McpManager       |  |     SkillManager     |  |   Approval Handler  |  |
|  |  (ClientSessionGroup)  |  |  (allowed_skills)    |  |  (CLI / Gateway)    |  |
|  +-----------+------------+  +----------+-----------+  +----------+----------+  |
|              |                          |                         |             |
|              v                          v                         |             |
|  +-----------+--------------------------+-----------+             |             |
|  |                     ToolRegistry                 |             |             |
|  +--------------------------+-----------------------+             |             |
|                             |                                     |             |
|                             v                                     v             |
|                   +---------+--------+                  +---------+--------+    |
|                   |  AgentKernel     | <--------------> | Turn Hooks       |    |
|                   |  (run_turn)      |                  | (before_tool,    |    |
|                   +---------+--------+                  |  before_model)   |    |
|                             |                           +------------------+    |
|                             v                                                   |
|                   +---------+--------+                                          |
|                   | BaseModelClient  |                                          |
|                   | (generate_stream)|                                          |
|                   +------------------+                                          |
+---------------------------------------------------------------------------------+
```

---

## 3. Component Details

### 3.1 MCP Client (`jarvis/mcp.py`)

A new module wrapping the `mcp` library client features.

*   **Manager Class**: `McpClientManager`
    *   Holds an instance of `mcp.client.session_group.ClientSessionGroup` which serves as the central connection hub.
    *   Loads server configurations from `config/mcp_settings.json`.
    *   **Initialization**: Instantiated under the session runtime and lazily connected when the first turn starts.
        *   If `url` is configured, it instantiates `mcp.client.session_group.SseServerParameters` and calls `connect_to_server`.
        *   If `command` is configured, it instantiates `mcp.StdioServerParameters` and calls `connect_to_server`.
    *   **Tool Registration**: Retrieves all aggregated tools from `.tools` property and registers them into the session's `ToolRegistry`.
    *   **Tool Execution**: Routes tool calls through `group.call_tool(name, arguments)`. It handles standardizing text output blocks and propagates error states correctly by throwing exceptions in the handler.
    *   **Cleanup**: Exits the group context `await group.__aexit__(None, None, None)` when the session is closed.

### 3.2 Skill Loader (`jarvis/skills.py`)

 A new module parsing directory-based skills.

*   **Structure**:
    *   A directory `skills/` contains subfolders representing individual skills.
    *   Each subfolder contains a `SKILL.md` file with a YAML frontmatter block defining parameters, metadata, and tools.
*   **YAML Frontmatter Schema**:
    ```yaml
    ---
    name: name_of_skill
    description: brief description
    tools:
      tool_name:
        description: tool description
        script: relative/path/to/executable
        parameters:
          type: object
          properties:
            arg1: { type: string }
          required: [arg1]
    ---
    Instructions block for system prompt injection.
    ```
*   **Logic**:
    *   If a skill matches the session config's `allowed_skills` list:
        1. It registers the instruction markdown (extracted below frontmatter) to be injected into the system prompt via a new turn hook (`SkillInstructionsHook`).
        2. It registers any defined tools. Executing these tools spawns the associated script in a subprocess with arguments passed as environment variables (or command arguments) and returns the standard output.

### 3.3 Turn Hooks (`jarvis/hooks.py`)

*   **ToolApprovalHook**:
    *   Intercepts tool calls in `before_tool`.
    *   If `ctx.config.require_tool_approval` is `True` and a tool call occurs:
        *   Awaits an async callback `ctx.approval_handler(tool_call)`.
        *   If the callback returns `True`, execution proceeds.
        *   If `False`, returns `HookResult(skip_tool=True, reason="Rejected by user")`.
*   **BudgetGuardHook**:
    *   Keeps a turn-scoped execution counter.
    *   Checks if the count exceeds `ctx.config.max_consecutive_tools`.
    *   If exceeded, stops execution with `HookResult(stop=True, reason="Consecutive tool call limit reached")`.
*   **SkillInstructionsHook**:
    *   In `before_model`, appends allowed skill instructions to the system prompt or inserts them as system messages.

### 3.4 Streaming Text Deltas (`jarvis/kernel.py` & models)

*   **Kernel Flow**:
    *   `AgentKernel.run_turn` is updated to call `ctx.model.generate_stream(...)`.
    *   Iterates through delta chunks yielded by the model client.
    *   If a chunk contains text, emits `TextDeltaEvent`.
    *   Accumulates all text and tool call deltas in memory.
    *   At the end of the stream, constructs the unified `ModelResponse` from the accumulated blocks, then executes any required tools exactly as in the original loop.
*   **Model Compatibility**:
    *   Implement/update `generate_stream` in `OpenAIClient` and `AnthropicClient`.
    *   Correctly parse streamed tool call arguments/IDs as they arrive in chunks and merge them into complete `ToolCall` structures.

### 3.5 Error Recovery (Model Clients)

*   **Transient Failures**: Catch rate limit errors, server timeout errors, connection errors, and status codes 429 / 5xx.
*   **Backoff Policy**: Wrap HTTP/API calls in a retry helper that performs up to 3 retries using exponential backoff (e.g., base delay of 1.0s, doubling each attempt with slight jitter).

---

## 4. Verification Plan

### 4.1 Automated Tests
*   `tests/test_mcp.py`: Test connection to mock stdio and mock SSE servers, tool registration, and execution routing.
*   `tests/test_skills.py`: Test parsing of `SKILL.md` files, instruction injection, and script-backed tool execution.
*   `tests/test_hooks.py`: Verify `ToolApprovalHook` halts/skips tools based on callback, and `BudgetGuardHook` halts turn when the tool call count exceeds limits.
*   `tests/test_streaming.py`: Test that text delta events are emitted incrementally during model generation.
*   `tests/test_error_recovery.py`: Mock transient failures in model clients and verify successful retry behavior.

### 4.2 Manual Verification
*   Start an interactive session using the CLI (`python run.py`) with `require_tool_approval` enabled to confirm stdin/stdout confirmation prompts work.
*   Configure a local stdio MCP server (e.g. standard file server) and run a turn verifying it executes correctly.
*   Add a custom skill folder under `skills/` and verify the agent utilizes its instructions and scripts.
