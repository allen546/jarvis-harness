import pytest
from pathlib import Path
from jarvis.tools import ToolRegistry
from jarvis.models.base import Message

MOCK_SKILL_MD = """---
name: mock_git
description: Mock git commands
tools:
  mock_commit:
    description: commit files
    script: scripts/commit.sh
    parameters:
      type: object
      properties:
        message:
          type: string
      required: [message]
---
Always commit files with clean messages.
"""

@pytest.mark.asyncio
async def test_skills_loader(tmp_path: Path) -> None:
    from jarvis.skills import SkillManager
    skill_dir = tmp_path / "skills" / "mock_git"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(MOCK_SKILL_MD, encoding="utf-8")
    
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    script_file = scripts_dir / "commit.sh"
    # Verifies uppercase environment variable convention is active
    script_file.write_text("#!/bin/sh\necho \"committed: $MESSAGE\"", encoding="utf-8")
    script_file.chmod(0o755)
    
    manager = SkillManager(skills_root=str(tmp_path / "skills"))
    
    from types import SimpleNamespace
    ctx = SimpleNamespace(
        config=SimpleNamespace(allowed_skills=["mock_git"]),
        tools=ToolRegistry()
    )
    
    skills = await manager.load_allowed_skills(ctx)
    assert len(skills) == 1
    assert skills[0].name == "mock_git"
    
    # Run mock script tool
    committed_tool = ctx.tools._tools["mock_commit"]
    res = await committed_tool.handler({"message": "initial commit"})
    assert "committed: initial commit" in res
    
    # Test prompt hook
    from jarvis.hooks import SkillInstructionsHook
    hook = SkillInstructionsHook(skills)
    msgs = [Message(role="system", content="System:")]
    res = await hook.before_model(ctx, msgs)
    assert "Always commit files with clean messages." in res.messages[0].content
