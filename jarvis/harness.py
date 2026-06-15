from typing import Optional, Any, Callable
from jarvis.config import HarnessConfig
from jarvis.memory.base import SessionContext, BaseMemoryEngine
from jarvis.models.base import BaseModelClient, Message, ToolCall, ModelResponse
from jarvis.channels.base import BaseChannel
from pydantic import BaseModel, Field

class TurnResult(BaseModel):
    response: ModelResponse
    tool_results: list[tuple[ToolCall, str]] = Field(default_factory=list)
    has_more_actions: bool = False

class AgentHarness:
    def __init__(
        self,
        config: HarnessConfig,
        model_client: BaseModelClient,
        memory_engine: BaseMemoryEngine,
        mcp_manager: Any,
        skills_manager: Any
    ):
        self.config = config
        self.model_client = model_client
        self.memory_engine = memory_engine
        self.mcp_manager = mcp_manager
        self.skills_manager = skills_manager
        
        self.pre_turn_hooks: list[Callable] = []
        self.post_message_hooks: list[Callable] = []

    async def execute_turn(
        self,
        session_ctx: SessionContext,
        channel: BaseChannel,
        user_message: Optional[Message] = None
    ) -> TurnResult:
        history = await self.memory_engine.load_history(session_ctx)
        
        if not history and self.config.system_prompt:
            history.insert(0, Message(role="system", content=self.config.system_prompt))
        
        if user_message:
            history.append(user_message)
            await self.memory_engine.save_history(session_ctx, [user_message])

        for hook in self.pre_turn_hooks:
            history = await hook(session_ctx, history)

        tools = []

        # Streaming generation
        accumulated_text = ""
        final_tool_calls = []
        async for response_chunk in self.model_client.generate_stream(history, tools=tools):
            if response_chunk.content:
                accumulated_text += response_chunk.content
                # Filter channel content before streaming
                filtered_chunk = channel.filter_content(response_chunk.content)
                if filtered_chunk:
                    await channel.send_stream_chunk(session_ctx.session_id, filtered_chunk)
            if response_chunk.tool_calls:
                final_tool_calls.extend(response_chunk.tool_calls)

        final_response = ModelResponse(content=accumulated_text, tool_calls=final_tool_calls, raw_response=None)

        for hook in self.post_message_hooks:
            await hook(session_ctx, final_response)

        # Save full raw message history
        assistant_msg = Message(role="assistant", content=accumulated_text)
        await self.memory_engine.save_history(session_ctx, [assistant_msg])
        
        # Send filtered final message to channel
        filtered_message = Message(
            role="assistant",
            content=channel.filter_content(accumulated_text),
            attachments=assistant_msg.attachments,
            native_actions=assistant_msg.native_actions,
            metadata=assistant_msg.metadata
        )
        await channel.send_message(session_ctx.session_id, filtered_message)

        tool_results = []
        
        return TurnResult(
            response=final_response,
            tool_results=tool_results,
            has_more_actions=len(final_tool_calls) > 0
        )
