import os
import yaml
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from jarvis.tools import Tool
from jarvis.hooks import NoopTurnHook, HookResult
from jarvis.models.base import Message

@dataclass
class LoadedSkill:
    name: str
    instructions: str
    tools: dict[str, Any]
    dir_path: Path

class SkillManager:
    def __init__(self, skills_root: str = "skills") -> None:
        self.skills_root = Path(skills_root)
        
    async def load_allowed_skills(self, ctx: Any) -> list[LoadedSkill]:
        allowed = getattr(ctx.config, "allowed_skills", [])
        if not allowed or not self.skills_root.exists():
            return []
            
        loaded_skills: list[LoadedSkill] = []
        for entry in self.skills_root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name not in allowed:
                continue
                
            skill_file = entry / "SKILL.md"
            if not skill_file.exists():
                continue
                
            content = skill_file.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            if len(parts) < 3:
                continue
                
            metadata = yaml.safe_load(parts[1])
            instructions = parts[2].strip()
            name = metadata.get("name", entry.name)
            tools = metadata.get("tools", {})
            
            loaded = LoadedSkill(name=name, instructions=instructions, tools=tools, dir_path=entry)
            loaded_skills.append(loaded)
            
            for t_name, t_cfg in tools.items():
                script_path = t_cfg.get("script")
                desc = t_cfg.get("description", "")
                params = t_cfg.get("parameters", {"type": "object", "properties": {}})
                
                tool = Tool(
                    name=t_name,
                    description=desc,
                    parameters=params,
                    handler=self._create_tool_handler(entry, script_path)
                )
                ctx.tools.register(tool)
                
        return loaded_skills
        
    def _create_tool_handler(self, skill_dir: Path, script_rel: str) -> Any:
        async def handler(args: dict[str, Any]) -> str:
            script_path = (skill_dir / script_rel).resolve()
            
            env = os.environ.copy()
            for k, v in args.items():
                env[str(k).upper()] = str(v)
                env[str(k).lower()] = str(v)
                
            proc = await asyncio.create_subprocess_exec(
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                cwd=str(skill_dir)
            )
            stdout, _ = await proc.communicate()
            return stdout.decode("utf-8", errors="replace")
        return handler

class SkillInstructionsHook(NoopTurnHook):
    __slots__ = ("_skills",)
    
    def __init__(self, skills: list[LoadedSkill]) -> None:
        self._skills = skills
        
    async def before_model(self, ctx: object, messages: list[Message]) -> HookResult:
        if not self._skills:
            return HookResult()
            
        skill_prompts = []
        for s in self._skills:
            skill_prompts.append(f"### Skill: {s.name}\n{s.instructions}")
        instructions_text = "\n\n".join(skill_prompts)
        
        new_msgs = list(messages)
        if new_msgs and new_msgs[0].role == "system":
            orig_sys = new_msgs[0].content
            new_msgs[0] = Message(role="system", content=f"{orig_sys}\n\n{instructions_text}")
        else:
            new_msgs.insert(0, Message(role="system", content=instructions_text))
            
        return HookResult(messages=new_msgs)
