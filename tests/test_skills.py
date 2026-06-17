import pytest
import json
from pathlib import Path
from jarvis.tools import ToolRegistry
from jarvis.models.base import Message
from types import SimpleNamespace

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
    
    # Test prompt duplication prevention in subsequent model invocations
    res2 = await hook.before_model(ctx, res.messages)
    assert res2.messages[0].content.count("Always commit files with clean messages.") == 1


@pytest.mark.asyncio
async def test_skills_loader_malformed_yaml(tmp_path: Path) -> None:
    from jarvis.skills import SkillManager
    
    # 1. Mock valid skill
    skill_dir = tmp_path / "skills" / "mock_git"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(MOCK_SKILL_MD, encoding="utf-8")
    
    # 2. Mock malformed skill
    skill_dir_malformed = tmp_path / "skills" / "malformed_skill"
    skill_dir_malformed.mkdir(parents=True)
    (skill_dir_malformed / "SKILL.md").write_text("---\nmalformed: :\n---\nHello", encoding="utf-8")
    
    manager = SkillManager(skills_root=str(tmp_path / "skills"))
    
    ctx = SimpleNamespace(
        config=SimpleNamespace(allowed_skills=["mock_git", "malformed_skill"]),
        tools=ToolRegistry()
    )
    
    skills = await manager.load_allowed_skills(ctx)
    # Should skip the malformed skill and only load mock_git
    assert len(skills) == 1
    assert skills[0].name == "mock_git"


@pytest.mark.asyncio
async def test_skills_path_traversal(tmp_path: Path) -> None:
    from jarvis.skills import SkillManager
    skill_dir = tmp_path / "skills" / "mock_git"
    skill_dir.mkdir(parents=True)
    
    manager = SkillManager(skills_root=str(tmp_path / "skills"))
    
    handler = manager._create_tool_handler(skill_dir, "../outside.sh")
    res = await handler({})
    assert "Path escape detected" in res


@pytest.mark.asyncio
async def test_skills_exit_code_failure(tmp_path: Path) -> None:
    from jarvis.skills import SkillManager
    skill_dir = tmp_path / "skills" / "mock_git"
    skill_dir.mkdir(parents=True)
    
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    script_file_fail = scripts_dir / "fail.sh"
    script_file_fail.write_text("#!/bin/sh\necho \"some error output\"\nexit 42", encoding="utf-8")
    script_file_fail.chmod(0o755)
    
    manager = SkillManager(skills_root=str(tmp_path / "skills"))
    
    handler_fail = manager._create_tool_handler(skill_dir, "scripts/fail.sh")
    res = await handler_fail({})
    assert "failed with exit code 42" in res
    assert "some error output" in res


@pytest.mark.asyncio
async def test_skills_complex_serialization(tmp_path: Path) -> None:
    from jarvis.skills import SkillManager
    skill_dir = tmp_path / "skills" / "mock_git"
    skill_dir.mkdir(parents=True)
    
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    script_file_complex = scripts_dir / "complex.sh"
    script_file_complex.write_text("#!/bin/sh\necho \"data: $DATA\"", encoding="utf-8")
    script_file_complex.chmod(0o755)
    
    manager = SkillManager(skills_root=str(tmp_path / "skills"))
    
    handler_complex = manager._create_tool_handler(skill_dir, "scripts/complex.sh")
    res = await handler_complex({"data": {"key": "val", "list": [1, 2]}})
    assert "data: " in res
    # Parse the json printed by the shell script output to verify it was serialized correctly
    parsed = json.loads(res.split("data: ")[1].strip())
    assert parsed == {"key": "val", "list": [1, 2]}
