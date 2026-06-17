import asyncio
import os
from pathlib import Path

from jarvis.config import load_session_config
from jarvis.kernel import AgentKernel
from jarvis.runtime import AgentSession, context_from_config
from jarvis.tools import ToolRegistry, builtin_tools
from jarvis.transports.cli import run_cli


async def main() -> None:
    config = load_session_config("cli")
    ctx = context_from_config(config, tools=ToolRegistry(builtin_tools(Path.cwd())))
    session = AgentSession(ctx=ctx, kernel=AgentKernel())
    await run_cli(session)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        os._exit(0)

