from __future__ import annotations

from typing import AsyncIterator

from jarvis.events import ErrorEvent, MessageEvent, NativeActionEvent, TextDeltaEvent, ToolCallEvent, ToolResultEvent
from jarvis.hooks import HookResult
from jarvis.models.base import Message, ModelResponse
from jarvis.runtime import AgentContext
from jarvis.tools import ToolResult


class AgentKernel:
    async def run_turn(self, ctx: AgentContext, user_message: Message) -> AsyncIterator[object]:
        ctx.session.history.append(user_message)
        messages = self._with_system_prompt(ctx, list(ctx.session.history))
        try:
            while True:
                hook_result = await self._run_before_model(ctx, messages)
                if hook_result.messages is not None:
                    messages = hook_result.messages
                if hook_result.stop:
                    yield ErrorEvent(session_id=ctx.session.id, message=hook_result.reason or "turn stopped")
                    return

                response = await ctx.model.generate(messages, ctx.tools.schemas())
                after_model = await self._run_after_model(ctx, response)
                if after_model.stop:
                    yield ErrorEvent(session_id=ctx.session.id, message=after_model.reason or "turn stopped")
                    return

                assistant = Message(role="assistant", content=response.content or "")
                if assistant.content:
                    yield TextDeltaEvent(session_id=ctx.session.id, content=assistant.content)
                for action in user_message.native_actions + assistant.native_actions:
                    yield NativeActionEvent(session_id=ctx.session.id, action=action)

                if not response.tool_calls:
                    messages.append(assistant)
                    ctx.session.history = self._without_system_prompt(messages)
                    for hook in ctx.hooks:
                        result = await hook.after_turn(ctx, assistant)
                        if result.stop:
                            yield ErrorEvent(session_id=ctx.session.id, message=result.reason or "turn stopped")
                            return
                    yield MessageEvent(session_id=ctx.session.id, message=assistant)
                    return

                messages.append(Message(role="assistant", content=assistant.content))
                for tool_call in response.tool_calls:
                    yield ToolCallEvent(session_id=ctx.session.id, tool_call=tool_call)
                    before_tool = await self._run_before_tool(ctx, tool_call)
                    if before_tool.stop:
                        yield ErrorEvent(session_id=ctx.session.id, message=before_tool.reason or "turn stopped")
                        return
                    if before_tool.skip_tool:
                        result = ToolResult(tool_call.call_id, tool_call.tool_name, before_tool.reason or "tool skipped", True)
                    else:
                        result = await ctx.tools.execute(tool_call)
                    yield ToolResultEvent(
                        session_id=ctx.session.id,
                        call_id=result.call_id,
                        tool_name=result.tool_name,
                        content=result.content,
                        is_error=result.is_error,
                    )
                    messages.append(Message(role="tool", content=result.content, metadata={"tool_call_id": tool_call.call_id, "tool_name": tool_call.tool_name}))
                    after_tool = await self._run_after_tool(ctx, tool_call, result)
                    if after_tool.stop:
                        yield ErrorEvent(session_id=ctx.session.id, message=after_tool.reason or "turn stopped")
                        return
        except Exception as exc:
            yield ErrorEvent(session_id=ctx.session.id, message=f"{type(exc).__name__}: {exc}")
            return

    def _with_system_prompt(self, ctx: AgentContext, messages: list[Message]) -> list[Message]:
        if ctx.config.system_prompt and not any(message.role == "system" for message in messages):
            return [Message(role="system", content=ctx.config.system_prompt), *messages]
        return messages

    def _without_system_prompt(self, messages: list[Message]) -> list[Message]:
        return [message for message in messages if message.role != "system"]

    async def _run_before_model(self, ctx: AgentContext, messages: list[Message]) -> HookResult:
        current = messages
        for hook in ctx.hooks:
            result = await hook.before_model(ctx, current)
            if result.messages is not None:
                current = result.messages
            if result.stop:
                return HookResult(messages=current, stop=True, reason=result.reason)
        return HookResult(messages=current)

    async def _run_after_model(self, ctx: AgentContext, response: ModelResponse) -> HookResult:
        for hook in ctx.hooks:
            result = await hook.after_model(ctx, response)
            if result.stop:
                return result
        return HookResult()

    async def _run_before_tool(self, ctx: AgentContext, tool_call: object) -> HookResult:
        for hook in ctx.hooks:
            result = await hook.before_tool(ctx, tool_call)  # type: ignore[arg-type]
            if result.stop or result.skip_tool:
                return result
        return HookResult()

    async def _run_after_tool(self, ctx: AgentContext, tool_call: object, result: ToolResult) -> HookResult:
        for hook in ctx.hooks:
            hook_result = await hook.after_tool(ctx, tool_call, result)  # type: ignore[arg-type]
            if hook_result.stop:
                return hook_result
        return HookResult()
