import os
import sys
import yaml
import json
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
                
            try:
                metadata = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError as exc:
                print(f"Error parsing metadata for skill {entry.name}: {exc}", file=sys.stderr)
                continue
                
            instructions = parts[2].strip()
            name = metadata.get("name", entry.name)
            tools = metadata.get("tools", {})
            if not isinstance(tools, dict):
                tools = {}
            
            loaded = LoadedSkill(name=name, instructions=instructions, tools=tools, dir_path=entry)
            loaded_skills.append(loaded)
            
            for t_name, t_cfg in tools.items():
                if not isinstance(t_cfg, dict):
                    continue
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
            try:
                script_path = (skill_dir / script_rel).resolve()
                try:
                    script_path.relative_to(skill_dir.resolve())
                except ValueError:
                    raise ValueError(f"Path escape detected: {script_rel} escapes skill directory")
                
                env = os.environ.copy()
                for k, v in args.items():
                    val_str = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    env[str(k).upper()] = val_str
                    env[str(k).lower()] = val_str
                    
                proc = await asyncio.create_subprocess_exec(
                    str(script_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                    cwd=str(skill_dir)
                )
                try:
                    stdout, _ = await proc.communicate()
                except asyncio.CancelledError:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                    raise
                    
                stdout_str = stdout.decode("utf-8", errors="replace")
                if proc.returncode != 0:
                    return f"Tool script failed with exit code {proc.returncode}.\nOutput:\n{stdout_str}"
                return stdout_str
            except Exception as exc:
                return f"Error executing tool script: {exc}"
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
        if not any(instructions_text in m.content for m in new_msgs if m.role == "system"):
            if new_msgs and new_msgs[0].role == "system":
                orig_sys = new_msgs[0].content
                new_msgs[0] = Message(role="system", content=f"{orig_sys}\n\n{instructions_text}")
            else:
                new_msgs.insert(0, Message(role="system", content=instructions_text))
                
        return HookResult(messages=new_msgs)
