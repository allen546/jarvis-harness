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
Transport -> Message -> AgentSession -> AgentKernel -> AgentEvent stream -> Transport
                                      |
                                      +-> AgentContext
                                      +-> per-session turn serialization
                                      +-> cancellation/busy policy
```

The kernel is transport-agnostic. It must not import FastAPI, CLI code, MCP client implementations, skill loaders, or future Discord/QQ adapters. Those pieces construct dependencies and submit messages through `AgentSession`.

### Modules

`jarvis/kernel.py`

- Defines `AgentKernel`.
- Exposes `run_turn(ctx: AgentContext, user_message: Message) -> AsyncIterator[AgentEvent]`.
- Owns the model/tool loop and event emission.
- Calls hooks at lifecycle checkpoints and honors hook stop decisions.

`jarvis/runtime.py`

- Defines `AgentSession`, `AgentContext`, `SessionState`, and `RuntimeConfig`.
- `AgentSession.submit(message: Message) -> AsyncIterator[AgentEvent]` is the only public entrypoint transports use to run turns.
- Holds the session id, model client, tool registry, hooks, config, and message history.
- Serializes turns per session with a lock or queue so two messages cannot mutate the same session context concurrently.
- Keeps long-lived runtime state, cancellation behavior, and busy policy out of the kernel implementation.

`jarvis/events.py`

- Defines small event dataclasses:
  - `TextDeltaEvent`
  - `ToolCallEvent`
  - `ToolResultEvent`
  - `MessageEvent`
  - `NativeActionEvent`
  - `ErrorEvent`
- Events are the only output surface transports need.

`jarvis/tools.py`

- Defines `Tool`, `ToolResult`, `ToolRegistry`, and optional `ToolSource`.
- Normalizes built-in tools, MCP tools, and skill-backed tools behind one interface.
- Provides tool schemas to model clients.
- Executes tool calls by name.

`jarvis/hooks.py`

- Defines one hook protocol with lifecycle checkpoints:
  - `before_model`
  - `after_model`
  - `before_tool`
  - `after_tool`
  - `after_turn`
- Defines `HookResult`, which can update messages, skip a tool call, stop the turn, or attach a reason.
- Memory behavior is implemented through hooks and optional tools.
- Hooks can load context, trim history, summarize, persist messages, index content, audit tool usage, enforce budgets, or stop degenerate loops.

`jarvis/transports/cli.py`

- Implements the minimal interactive CLI transport.
- Reads user input, creates `Message(role="user", content=line)`, submits it to `AgentSession`, and renders events.
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
3. Submit the message to `AgentSession`.
4. Render `AgentEvent`s back to the user or client.

The kernel never calls `channel.send_message()`, `channel.send_stream_chunk()`, or `channel.filter_content()`. Output filtering, presentation, buffering, and protocol-specific behavior belong in transports.

Transports may still expose channel-specific capabilities. They do this by contributing transport-scoped tools, native actions, or renderers to the session runtime:

- Emoji reactions can be exposed as a channel-native tool such as `discord_add_reaction`.
- Native media replies can be represented as message attachments or `NativeActionEvent`s.
- Thread replies, mentions, quoting, embeds, and protocol-specific message references live in message metadata and transport renderers.
- A transport decides which native actions it supports and how to render unsupported ones.

This keeps quirks available without letting them leak into `AgentKernel`.

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
2. The transport submits the message to `AgentSession`.
3. `AgentSession` serializes the turn for that session and calls `AgentKernel.run_turn(...)`.
4. `AgentKernel.run_turn(...)` appends the user message to `ctx.session.history`.
5. `before_model` hooks receive the context and working message list.
6. Hooks may return modified messages or request that the turn stop.
7. The kernel calls `ctx.model.generate(messages, tool_schemas)`.
8. `after_model` hooks observe the response and may request that the turn stop.
9. If the model returns content, the kernel emits text/message events.
10. If the assistant message contains native actions, the kernel emits native action events.
11. If the model returns tool calls, the kernel emits tool call events.
12. `before_tool` hooks observe each tool call and may request that the call or turn stop.
13. Allowed tool calls execute through `ToolRegistry`.
14. Tool results are appended to the working messages and emitted as tool result events.
15. `after_tool` hooks observe each tool result and may request that the turn stop.
16. The kernel repeats model generation until there are no tool calls or a hook requests stop.
17. The final assistant message is appended to session history.
18. `after_turn` hooks run after the final assistant message.
19. `AgentSession` releases the session for the next queued turn.
20. The transport renders the event stream.

## Hook Checkpoints

Hooks are the single extension mechanism for memory, policy, budgets, and circuit breakers. The kernel does not have a separate guard system.

`HookResult` should stay small:

- `messages`: optional replacement for the working message list, allowed only at checkpoints that accept message mutation.
- `skip_tool`: whether the current tool call should be skipped, allowed only at `before_tool`.
- `stop`: whether the kernel should stop the current turn.
- `reason`: optional human-readable skip or stop reason.

The kernel is responsible for calling hooks at the right checkpoints and applying the allowed result fields. Hook implementations own policy.

Examples:

- Memory loading: `before_model` injects recent history or summaries.
- Context trimming: `before_model` replaces the working message list.
- Persistence: `after_turn` stores final messages.
- Tool budget: `before_tool` stops when a turn exceeds a configured tool count.
- Repeated tool loop detection: `before_tool` stops when the same tool and normalized arguments repeat too often.
- Repeated content detection: `after_model` stops when identical assistant content repeats too often.
- Risk control: `before_tool` skips or stops risky built-in tool calls.

The only behavior that remains hardcoded in the kernel is runtime mechanics required to operate the async loop, such as cancellation propagation and emitting an error event when an exception escapes a dependency.

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
- Transport-scoped native tools

The registry is responsible for:

- Returning model-ready tool schemas.
- Finding tools by model call name.
- Executing tools.
- Converting exceptions into structured tool results.
- Returning a clean unknown-tool result when the model calls a missing tool.

The kernel should not know whether a tool came from a built-in, MCP server, or skill.
The kernel also should not know whether a tool is transport-native. For example, `add_reaction` and `send_native_media` are tools registered by a Discord transport or gateway session, not kernel branches.

## Native Messages And Actions

The message model should preserve native channel information without making the kernel channel-aware.

Input messages may include:

- `attachments` for media and files.
- `native_actions` for protocol-specific incoming events.
- `metadata` for channel id, thread id, message id, author id, reply target, mentions, and other transport data.

Output can use:

- plain assistant `Message` content for portable text.
- attachments for media replies.
- `NativeActionEvent` for protocol-specific output that does not fit plain text, such as emoji reactions, quote replies, embeds, or platform-native cards.

Transports are responsible for rendering these outputs. CLI can print a readable fallback for native actions. Gateway SSE can serialize them as structured events. Discord/QQ can map them to platform APIs.

## Memory

Memory is not a central subsystem.

Memory-like behavior belongs in hooks and tools:

- A `before_model` hook can load recent history or summaries.
- A `before_model` hook can trim context.
- An `after_turn` hook can persist messages.
- An `after_turn` hook can update summaries or indexes.
- A tool can expose explicit recall/search behavior to the model.

This keeps the kernel small and allows multiple memory strategies without adding another core interface.

## Subagents

Subagents should be easy to add after the base loop works, but they are not part of the first implementation pass.

When added, a subagent should be a tool implementation:

1. `spawn_subagent` receives a prompt and optional scope.
2. The tool creates a child `AgentSession` with its own `AgentContext`.
3. The child context inherits selected model, tool, hook, transport-native capability, and config dependencies.
4. The tool submits the prompt to the child session.
5. The tool returns the child agent's final answer as a tool result.

This avoids a second harness hierarchy and keeps subagents as recursive kernel use.

## Error Handling

The kernel should emit `ErrorEvent` when a dependency failure escapes normal hook/tool handling. Policy decisions, including loop limits and budget stops, belong in hooks.

Expected behavior:

- Unknown tool: append a tool result explaining the missing tool and continue the loop.
- Tool exception: emit a failed tool result and continue unless a hook requests that the turn stop.
- Model exception: emit `ErrorEvent` and stop the turn.
- Hook stop decision: emit a stop/error event with the hook-provided reason and stop the turn.

The gateway decides how to serialize errors to SSE. The CLI decides how to print them.

## Testing

Replace stale harness/channel/memory tests with a small contract suite:

- Kernel emits a final message for a simple model response.
- Kernel executes one tool call, appends the tool result, and loops back to the model.
- `before_model` hooks can modify the working message list.
- Hook stop decisions stop the turn at model and tool checkpoints.
- `after_turn` hooks observe the final response.
- Tool registry resolves built-ins and reports unknown tools cleanly.
- Gateway SSE serializes kernel events.
- CLI transport can submit a message to an `AgentSession` and render a final event.
- Native action events can be rendered or safely ignored by transports that do not support them.

Tests should focus on contracts at the kernel boundary. Provider-specific tests should remain in the model adapter layer.

## Migration Plan

1. Add `events.py`, `runtime.py`, `tools.py`, `hooks.py`, and `kernel.py`.
2. Move the current incomplete loop from `agent.py` into `AgentKernel`.
3. Add `AgentSession` as the serialized runtime entrypoint for transports.
4. Rebuild CLI as a thin transport.
5. Rebuild gateway SSE on top of kernel events.
6. Replace `Channel` usage with transport-specific code and native capability registration.
7. Delete or rewrite stale tests that import removed modules.
8. Add the minimal contract tests listed above.

## Acceptance Criteria

- The gateway can run a turn and stream events over SSE.
- The CLI can run the same turn loop without separate agent logic.
- Transports submit turns through `AgentSession`, not directly to `AgentKernel`.
- Turns are serialized per session.
- The kernel has no transport imports.
- Hooks are the only policy/circuit-breaker extension mechanism.
- Memory is implemented only through hooks/tools, not a core memory engine.
- Built-in tools, MCP tools, and skills are exposed through one registry.
- Transport-native tools/actions preserve channel quirks without kernel imports.
- Subagents can later be implemented as a tool without changing the kernel API.
