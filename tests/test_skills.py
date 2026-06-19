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
async def test_list_skills(tmp_path: Path) -> None:
    """list_skills returns skill:// URIs for all skills in configured dirs."""
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
        handler = tools._tools["list_skills"].handler
        result = handler({})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "mock_git"
        assert parsed[0]["uri"] == "skill://mock_git"
        assert "Mock git" in parsed[0]["description"]
    finally:
        current_context.reset(token)


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
async def test_list_skills_multiple_dirs(tmp_path: Path) -> None:
    """list_skills scans multiple configured directories."""
    for d in [".claude/skills/alpha", ".codex/skills/beta"]:
        dir_path = tmp_path / d
        dir_path.mkdir(parents=True)
        (dir_path / "SKILL.md").write_text(f"---\nname: {d.split('/')[-2]}_{d.split('/')[-1]}\ndescription: Test skill\ntools: {{}}\n---\nBody.")

    tools = ToolRegistry(builtin_tools(tmp_path))
    ctx = AgentContext(
        config=RuntimeConfig(skills_dirs=[".claude/skills/", ".codex/skills/"]),
        session=SessionState(id="s1"),
        model=None,
        tools=tools,
        hooks=[],
    )
    token = current_context.set(ctx)
    try:
        handler = tools._tools["list_skills"].handler
        result = handler({})
        parsed = json.loads(result)
        assert len(parsed) == 2
        names = {s["name"] for s in parsed}
        assert "alpha" in names
        assert "beta" in names
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
