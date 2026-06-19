import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoadedSkill:
    name: str
    instructions: str
    tools: dict[str, Any]
    dir_path: Path


def slugify_skill_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if len(slug) > 48:
        slug = slug[:48].rstrip("-")
    return slug or "learned-procedure"
