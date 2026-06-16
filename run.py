import asyncio
from pathlib import Path

from jarvis.config import ModelConfig, SessionConfig
from jarvis.kernel import AgentKernel
from jarvis.runtime import AgentSession, context_from_config
from jarvis.tools import ToolRegistry, builtin_tools
from jarvis.transports.cli import run_cli


async def main() -> None:
    config = SessionConfig(
        session_id="cli",
        model=ModelConfig(provider="openai", model_name="gpt-4o"),
    )
    ctx = context_from_config(config, tools=ToolRegistry(builtin_tools(Path.cwd())))
    session = AgentSession(ctx=ctx, kernel=AgentKernel())
    await run_cli(session)


if __name__ == "__main__":
    asyncio.run(main())
