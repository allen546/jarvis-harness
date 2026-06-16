# Design Spec: Jarvis Gateway Microkernel

## Goal

Strip the current agent harness down to a gateway-first microkernel. The kernel should do the minimum work required to run one agent turn: assemble messages, call the model, execute tools, apply hooks, and emit structured events.

The gateway is the product boundary. The CLI is only the smallest local transport used to exercise the same kernel while the gateway loop is being rebuilt.

## Non-Goals

- Do not restore `AgentHarness` as a central abstraction.
- Do not restore memory as a first-class engine package.
- Do not make channels part of the kernel.
- Do not implement Discord, QQ, or subagents in the first pass.
- Do not build a full permissions system for tools beyond a minimal allow/disable switch for command execution.

## Architecture

The architecture has one kernel surrounded by ports:

```text
Transport -> Message -> AgentKernel -> AgentEvent stream -> Transport
                         |
                         +-> ModelClient
                         +-> ToolRegistry
                         +-> Hooks
                         +-> SessionState
```

The kernel is transport-agnostic. It must not import FastAPI, CLI code, MCP client implementations, skill loaders, or future Discord/QQ adapters. Those pieces construct dependencies and feed messages into the kernel.

### Modules

`jarvis/kernel.py`

- Defines `AgentKernel`.
- Exposes `run_turn(ctx: AgentContext, user_message: Message) -> AsyncIterator[AgentEvent]`.
- Owns the model/tool loop and event emission.
- Enforces `max_tool_rounds`.

`jarvis/runtime.py`

- Defines `AgentContext`, `SessionState`, and `RuntimeConfig`.
- Holds the session id, model client, tool registry, hooks, config, and message history.
- Keeps long-lived runtime state out of the kernel implementation.

`jarvis/events.py`

- Defines small event dataclasses:
  - `TextDeltaEvent`
  - `ToolCallEvent`
  - `ToolResultEvent`
  - `MessageEvent`
  - `ErrorEvent`
- Events are the only output surface the gateway and CLI need.

`jarvis/tools.py`

- Defines `Tool`, `ToolResult`, `ToolRegistry`, and optional `ToolSource`.
- Normalizes built-in tools, MCP tools, and skill-backed tools behind one interface.
- Provides tool schemas to model clients.
- Executes tool calls by name.

`jarvis/hooks.py`

- Defines hook protocols:
  - `PreMessageHook`
  - `PostMessageHook`
  - `PostToolHook`
- Memory behavior is implemented through hooks and optional tools.
- Hooks can load context, trim history, summarize, persist messages, index content, or audit tool usage.

`jarvis/transports/cli.py`

- Implements the minimal interactive CLI transport.
- Reads user input, creates `Message(role="user", content=line)`, calls the kernel, and renders events.
- Exists for local operation and debugging, not as the primary product boundary.

`main.py`

- Remains the gateway adapter.
- Converts HTTP requests into user messages.
- Converts kernel events into SSE responses.
- Owns web-specific request validation and response formatting.

## Channels And Transports

Channels are retained as a concept but moved outside the kernel. They should be treated as transports or adapters.

A transport can:

1. Receive user input.
2. Convert that input into a `Message`.
3. Call `AgentKernel.run_turn(...)`.
4. Render `AgentEvent`s back to the user or client.

The kernel never calls `channel.send_message()`, `channel.send_stream_chunk()`, or `channel.filter_content()`. Output filtering, presentation, buffering, and protocol-specific behavior belong in transports.

Current transport:

- CLI: minimal local adapter.

Primary transport:

- Gateway SSE: HTTP adapter exposed by `main.py`.

Future transports:

- Discord
- QQ
- Webhook
- Any other adapter that can translate between external events and kernel messages/events.

## Turn Flow

1. A transport receives input and creates a `Message` with role `user`.
2. `AgentKernel.run_turn(...)` appends the user message to `ctx.session.history`.
3. Pre-message hooks receive the context and working message list.
4. Hooks may return a modified message list.
5. The kernel calls `ctx.model.generate(messages, tool_schemas)`.
6. If the model returns content, the kernel emits text/message events.
7. If the model returns tool calls, the kernel emits tool call events and executes them through `ToolRegistry`.
8. Tool results are appended to the working messages and emitted as tool result events.
9. The kernel repeats model generation until there are no tool calls or `max_tool_rounds` is reached.
10. The final assistant message is appended to session history.
11. Post-message hooks run after the final assistant message.
12. The transport renders the event stream.

## Tool System

The first-pass tool system should be minimal and explicit.

Built-in tools:

- `list_files`
- `read_file`
- `search_text`
- `run_command`, disabled unless config allows it

Retained tool sources:

- MCP tools
- Skills

The registry is responsible for:

- Returning model-ready tool schemas.
- Finding tools by model call name.
- Executing tools.
- Converting exceptions into structured tool results.
- Returning a clean unknown-tool result when the model calls a missing tool.

The kernel should not know whether a tool came from a built-in, MCP server, or skill.

## Memory

Memory is not a central subsystem.

Memory-like behavior belongs in hooks and tools:

- A pre-message hook can load recent history or summaries.
- A pre-message hook can trim context.
- A post-message hook can persist messages.
- A post-message hook can update summaries or indexes.
- A tool can expose explicit recall/search behavior to the model.

This keeps the kernel small and allows multiple memory strategies without adding another core interface.

## Subagents

Subagents should be easy to add after the base loop works, but they are not part of the first implementation pass.

When added, a subagent should be a tool implementation:

1. `spawn_subagent` receives a prompt and optional scope.
2. The tool creates a child `AgentContext`.
3. The child context inherits selected model, tool, hook, and config dependencies.
4. The tool calls `AgentKernel.run_turn(...)` for the child.
5. The tool returns the child agent's final answer as a tool result.

This avoids a second harness hierarchy and keeps subagents as recursive kernel use.

## Error Handling

The kernel should emit `ErrorEvent` for model and tool failures before raising or returning, depending on the failure type.

Expected behavior:

- Unknown tool: append a tool result explaining the missing tool and continue the loop.
- Tool exception: emit a failed tool result and continue unless config says tool failures are fatal.
- Model exception: emit `ErrorEvent` and stop the turn.
- Max tool rounds exceeded: emit `ErrorEvent`, append a final assistant-facing failure message, and stop the turn.

The gateway decides how to serialize errors to SSE. The CLI decides how to print them.

## Testing

Replace stale harness/channel/memory tests with a small contract suite:

- Kernel emits a final message for a simple model response.
- Kernel executes one tool call, appends the tool result, and loops back to the model.
- Kernel stops at `max_tool_rounds`.
- Pre-message hooks can modify the working message list.
- Post-message hooks observe the final response.
- Tool registry resolves built-ins and reports unknown tools cleanly.
- Gateway SSE serializes kernel events.
- CLI transport can submit a message to the kernel and render a final event.

Tests should focus on contracts at the kernel boundary. Provider-specific tests should remain in the model adapter layer.

## Migration Plan

1. Add `events.py`, `runtime.py`, `tools.py`, `hooks.py`, and `kernel.py`.
2. Move the current incomplete loop from `agent.py` into `AgentKernel`.
3. Replace `Channel` usage with transport-specific code.
4. Rebuild CLI as a thin transport.
5. Rebuild gateway SSE on top of kernel events.
6. Delete or rewrite stale tests that import removed modules.
7. Add the minimal contract tests listed above.

## Acceptance Criteria

- The gateway can run a turn and stream events over SSE.
- The CLI can run the same turn loop without separate agent logic.
- The kernel has no transport imports.
- Memory is implemented only through hooks/tools, not a core memory engine.
- Built-in tools, MCP tools, and skills are exposed through one registry.
- Subagents can later be implemented as a tool without changing the kernel API.
