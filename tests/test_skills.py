import pytest
import json
from pathlib import Path
from jarvis.tools import ToolRegistry, builtin_tools
from jarvis.models.base import Message
from jarvis.runtime import AgentContext, RuntimeConfig, SessionState, current_context


MOCK_SKILL_MD = """---
name: mock_git
description: Mock git commands
tools: {}
---
Always commit files with clean messages.
"""




@pytest.mark.asyncio
async def test_read_skill_uri(tmp_path: Path) -> None:
    """read(skill://name) resolves to SKILL.md and returns content + path."""
    skills_dir = tmp_path / "skills" / "mock_git"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(MOCK_SKILL_MD)

    tools = ToolRegistry(builtin_tools(tmp_path))
    ctx = AgentContext(
        config=RuntimeConfig(skills_dirs=["skills/"]),
        session=SessionState(id="s1"),
        model=None,
        tools=tools,
        hooks=[],
    )
    token = current_context.set(ctx)
    try:
        handler = tools._tools["read"].handler
        result = handler({"path": "skill://mock_git"})
        assert "Always commit files with clean messages." in result
        assert "[skill path:" in result
        assert "mock_git" in result
    finally:
        current_context.reset(token)


@pytest.mark.asyncio
async def test_read_skill_not_found(tmp_path: Path) -> None:
    """read(skill://nonexistent) returns error."""
    tools = ToolRegistry(builtin_tools(tmp_path))
    ctx = AgentContext(
        config=RuntimeConfig(skills_dirs=["skills/"]),
        session=SessionState(id="s1"),
        model=None,
        tools=tools,
        hooks=[],
    )
    token = current_context.set(ctx)
    try:
        handler = tools._tools["read"].handler
        result = handler({"path": "skill://nonexistent"})
        assert "Error" in result
        assert "not found" in result
    finally:
        current_context.reset(token)




@pytest.mark.asyncio
async def test_read_skill_malformed_yaml(tmp_path: Path) -> None:
    """read(skill://name) still reads content even with bad YAML frontmatter."""
    skills_dir = tmp_path / "skills" / "bad_skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("not valid yaml frontmatter\n---\nSome instructions.")

    tools = ToolRegistry(builtin_tools(tmp_path))
    ctx = AgentContext(
        config=RuntimeConfig(skills_dirs=["skills/"]),
        session=SessionState(id="s1"),
        model=None,
        tools=tools,
        hooks=[],
    )
    token = current_context.set(ctx)
    try:
        handler = tools._tools["read"].handler
        result = handler({"path": "skill://bad_skill"})
        assert "Some instructions." in result
        assert "[skill path:" in result
    finally:
        current_context.reset(token)


@pytest.mark.asyncio
async def test_slugify_skill_name() -> None:
    from jarvis.skills import slugify_skill_name
    assert slugify_skill_name("Deploy to Pi") == "deploy-to-pi"
    assert slugify_skill_name("Hello World!") == "hello-world"
    assert slugify_skill_name("") == "learned-procedure"
    assert slugify_skill_name("A" * 60)[:48] == "a" * 48
