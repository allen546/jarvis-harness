from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from jarvis.media import (
    detect_media_type, chunk_count, get_media_info,
    extract_pdf_pages, extract_media_chunk, to_data_uri,
    enforce_size_limit, get_mime_type,
)
from jarvis.models.base import Attachment, ToolCall

@dataclass(slots=True)
class ToolResult:
    call_id: str
    tool_name: str
    content: str
    attachments: list[Attachment] = field(default_factory=list)
    is_error: bool = False

ToolHandler = Callable[[dict[str, Any]], Awaitable[str | ToolResult] | str | ToolResult]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.tool_name)
        if tool is None:
            return ToolResult(call_id=call.call_id, tool_name=call.tool_name, content=f"Unknown tool: {call.tool_name}", is_error=True)
        try:
            value = tool.handler(call.arguments)
            if not isinstance(value, ToolResult):
                value = await value if asyncio.iscoroutine(value) else value
            if not isinstance(value, ToolResult):
                return ToolResult(call_id=call.call_id, tool_name=call.tool_name, content=str(value))
            value.call_id = call.call_id
            value.tool_name = call.tool_name
            return value
        except Exception as exc:
            return ToolResult(call_id=call.call_id, tool_name=call.tool_name, content=f"{type(exc).__name__}: {exc}", is_error=True)


def _resolve(root: Path, raw_path: str) -> Path:
    candidate = (root / raw_path).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"path escapes root: {raw_path}")
    return candidate

def _resolve_skill(skill_name: str, base: Path) -> Path | None:
    """Resolve a skill:// URI to a SKILL.md path by scanning skills_dirs."""
    from jarvis.runtime import current_context
    ctx = current_context.get()
    skills_dirs = ["skills/"]
    if ctx and hasattr(ctx, "config"):
        skills_dirs = getattr(ctx.config, "skills_dirs", skills_dirs)
    for d in skills_dirs:
        skill_dir = base / d / skill_name
        skill_file = skill_dir / "SKILL.md"
        if skill_file.exists():
            return skill_file
    return None


def builtin_tools(root: Path | str = ".") -> list[Tool]:
    base = Path(root)

    def list_files(args: dict[str, Any]) -> str:
        raw_path = str(args.get("path", "."))
        target = _resolve(base, raw_path)
        if target.is_file():
            return target.name
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        return "\n".join(
            (f"{p.name}/" if p.is_dir() else p.name)
            for p in entries
            if not p.name.startswith(".")
        )

    def read_file(args: dict[str, Any]) -> str | ToolResult:
        raw_path = str(args["path"])
        # Resolve skill:/// URIs
        if raw_path.startswith("skill://"):
            skill_name = raw_path[len("skill://"):]
            resolved = _resolve_skill(skill_name, base)
            if resolved is None:
                return f"Error: skill not found: {skill_name}"
            # Read the SKILL.md and return content + resolved path
            content = resolved.read_text(encoding="utf-8", errors="replace")
            return f"[skill path: {resolved.parent}]\n\n{content}"
        target = _resolve(base, raw_path)
        chunk_idx = args.get("chunk", 0)
        if not target.exists():
            return f"Error: file not found: {target}"
        if target.is_dir():
            return f"Error: path is a directory: {target}"
        try:
            media_type = detect_media_type(target)
        except Exception:
            media_type = "text"
        size = target.stat().st_size
        if media_type == "text":
            text = target.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            total = len(lines)
            if total <= 200:
                return text
            chunks = chunk_count(total, 200)
            if chunk_idx < 0 or chunk_idx >= chunks:
                return f"Error: chunk {chunk_idx} out of range (total {chunks})"
            start = chunk_idx * 200
            end = min(start + 200, total)
            chunk_lines = lines[start:end]
            header = f"[lines {start+1}-{end} of {total} | chunk {chunk_idx}/{chunks-1}]\n"
            footer = f"\n---\nread(path=\"{target.name}\", chunk={chunk_idx + 1}) → next chunk" if chunk_idx + 1 < chunks else ""
            return header + "\n".join(chunk_lines) + footer
        elif media_type == "document":
            try:
                total_pages = get_media_info(target).get("total_pages", 0)
                if total_pages <= 2:
                    return extract_pdf_pages(target, 0, total_pages)
                chunks = chunk_count(total_pages, 2)
                if chunk_idx < 0 or chunk_idx >= chunks:
                    return f"Error: chunk {chunk_idx} out of range (total {chunks})"
                start_page = chunk_idx * 2
                end_page = min(start_page + 2, total_pages)
                page_text = extract_pdf_pages(target, start_page, end_page)
                header = f"[pages {start_page+1}-{end_page} of {total_pages} | chunk {chunk_idx}/{chunks-1}]\n"
                footer = f"\n---\nread(path=\"{target.name}\", chunk={chunk_idx + 1}) → next chunk" if chunk_idx + 1 < chunks else ""
                return header + page_text + footer
            except Exception as exc:
                return f"Error reading PDF: {exc}"
        elif media_type in ("video", "audio"):
            try:
                info = get_media_info(target)
                duration = info.get("duration_secs", 0)
                if not duration:
                    return f"Error: could not determine duration of {target}"
                chunk_secs = 10
                chunks = chunk_count(duration, chunk_secs)
                if chunk_idx >= chunks:
                    return f"Error: chunk {chunk_idx} out of range (total {chunks})"
                raw, mime = extract_media_chunk(target, chunk_idx * chunk_secs, chunk_secs)
                enforce_size_limit(len(raw), 10)
                data_uri = to_data_uri(raw, mime)
                start_s = chunk_idx * chunk_secs
                end_s = min(start_s + chunk_secs, int(duration))
                size_kb = len(raw) / 1024
                tag = f"<{media_type}-{chunk_idx + 1}:{target.name}:{size_kb:.1f}KB {start_s:.1f}s-{end_s:.1f}s>"
                footer = f"\n---\nread(path=\"{target.name}\", chunk={chunk_idx + 1}) → next chunk" if chunk_idx + 1 < chunks else ""
                return ToolResult(
                    call_id="", tool_name="read",
                    content=tag + footer,
                    attachments=[Attachment(mime_type=mime, url=data_uri, description=target.name)],
                )
            except Exception as exc:
                return f"Error extracting {media_type} chunk: {exc}"
        elif media_type == "image":
            try:
                raw = target.read_bytes()
                enforce_size_limit(len(raw), 10)
                mime = get_mime_type("image", target)
                data_uri = to_data_uri(raw, mime)
                size_kb = len(raw) / 1024
                tag = f"<image-1:{target.name}:{size_kb:.1f}KB>"
                return ToolResult(
                    call_id="", tool_name="read",
                    content=tag,
                    attachments=[Attachment(mime_type=mime, url=data_uri, description=target.name)],
                )
            except Exception as exc:
                return f"Error reading image: {exc}"
        else:
            return target.read_text(encoding="utf-8", errors="replace")
    def write_file(args: dict[str, Any]) -> str:
        target = _resolve(base, str(args["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        content = str(args["content"])
        max_chars = 1_048_576  # 1 MB text limit
        if len(content) > max_chars:
            return f"Error: content exceeds {max_chars} char limit ({len(content)} chars)"
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {target.relative_to(base.resolve())}"

    def search_text(args: dict[str, Any]) -> str:
        query = str(args["query"])
        raw_path = str(args.get("path", "."))
        target = _resolve(base, raw_path)
        files = [target] if target.is_file() else [path for path in target.rglob("*") if path.is_file()]
        matches: list[str] = []
        for path in files:
            try:
                for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                    if query in line:
                        matches.append(f"{path.relative_to(base.resolve())}:{line_no}:{line}")
            except UnicodeDecodeError:
                continue
        return "\n".join(matches)

    async def run_command(args: dict[str, Any]) -> str:
        command = str(args["command"])
        proc = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output, _ = await proc.communicate()
        return output.decode("utf-8", errors="replace")

    object_params = {"type": "object", "properties": {}, "additionalProperties": True}
    write_params = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative workspace path to write."},
            "content": {"type": "string", "description": "File content to write."},
        },
        "required": ["path", "content"],
    }
    read_params = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace."},
            "chunk": {"type": "integer", "description": "Chunk index (0-based). Omit for first chunk of large files."}
        },
        "required": ["path"],
    }
    from jarvis.memory_store import (
        search_semantic_memory_tool, purge_semantic_memory_tool, store_semantic_memory_tool,
        check_redundancy_tool, distill_now_tool, merge_memory_tool, update_memory_tool,
    )
    search_params = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "tag": {"type": "string", "description": "Optional tag filter (e.g. 'truths', 'history')."},
            "kind": {"type": "string", "enum": ["fact", "procedure", "history_summary"], "description": "Optional kind filter."},
            "scope": {"type": "string", "enum": ["global", "session", "both"], "description": "Memory scope to search."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Max results (default 5)."},
        },
        "required": ["query"]
    }
    purge_params = {
        "type": "object",
        "properties": {
            "tag": {"type": "string", "description": "Tag to purge memories by."},
            "ids": {"type": "array", "items": {"type": "string"}, "description": "Memory item IDs to purge."},
            "kind": {"type": "string", "enum": ["fact", "procedure", "history_summary"], "description": "Kind to purge."},
            "scope": {"type": "string", "enum": ["global", "session", "both"], "description": "Memory scope to purge from."},
        }
    }
    store_params = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The fact or procedure to store."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags (default: ['truths'])."},
            "kind": {"type": "string", "enum": ["fact", "procedure"], "description": "Memory kind (default: 'fact')."},
            "scope": {"type": "string", "enum": ["global", "session"], "description": "Memory scope."},
            "metadata": {"type": "object", "description": "Arbitrary metadata dict."},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Confidence 0-1 (default 1.0)."},
        },
        "required": ["text"]
    }
    check_redundancy_params = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Max pairs to return (default 10)."},
        }
    }
    distill_now_params = {"type": "object", "properties": {}}
    merge_params = {
        "type": "object",
        "properties": {
            "id_a": {"type": "string", "description": "First record ID."},
            "id_b": {"type": "string", "description": "Second record ID."},
        },
        "required": ["id_a", "id_b"]
    }
    update_params = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Record ID to update."},
            "text": {"type": "string", "description": "New text value."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "New tags list."},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "New confidence value."},
        },
        "required": ["id"]
    }


    async def spawn_subagent_handler(args: dict[str, Any]) -> str:
        from jarvis.runtime import current_context
        from jarvis.subagent import spawn_subagent_tool
        import json
        ctx = current_context.get()
        if ctx is None:
            raise RuntimeError("No active agent context")
        res = await spawn_subagent_tool(ctx, args)
        return json.dumps(res)

    async def send_subagent_message_handler(args: dict[str, Any]) -> str:
        from jarvis.runtime import current_context
        from jarvis.subagent import send_subagent_message_tool
        import json
        ctx = current_context.get()
        if ctx is None:
            raise RuntimeError("No active agent context")
        res = await send_subagent_message_tool(ctx, args)
        return json.dumps(res)

    async def close_subagent_handler(args: dict[str, Any]) -> str:
        from jarvis.runtime import current_context
        from jarvis.subagent import close_subagent_tool
        import json
        ctx = current_context.get()
        if ctx is None:
            raise RuntimeError("No active agent context")
        res = await close_subagent_tool(ctx, args)
        return json.dumps(res)

    spawn_subagent_params = {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of prompts/messages for the subagent to process."
            },
            "prompt": {
                "type": "string",
                "description": "Alternative single prompt for the subagent to process."
            },
            "task_name": {
                "type": "string",
                "description": "Name or description of the subtask."
            },
            "system_override": {
                "type": "string",
                "description": "Optional custom system prompt override for the subagent."
            }
        },
        "required": ["task_name"]
    }

    send_subagent_message_params = {
        "type": "object",
        "properties": {
            "sub_session_id": {
                "type": "string",
                "description": "The session ID of the target subagent."
            },
            "message": {
                "type": "string",
                "description": "The follow-up message to send."
            }
        },
        "required": ["sub_session_id", "message"]
    }

    close_subagent_params = {
        "type": "object",
        "properties": {
            "sub_session_id": {
                "type": "string",
                "description": "The session ID of the subagent to close."
            }
        },
        "required": ["sub_session_id"]
    }
    send_file_params = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Local file path relative to workspace."},
            "file_type": {"type": "integer", "description": "QQ file type: 1=image, 2=video, 3=audio. Auto-detected if omitted."}
        },
        "required": ["path"],
    }

    async def send_file_handler(args: dict[str, Any]) -> str | ToolResult:
        target = _resolve(base, str(args["path"]))
        if not target.exists():
            return f"Error: file not found: {target}"
        file_type = args.get("file_type")
        media_type = detect_media_type(target)
        mime = get_mime_type(media_type, target)
        if file_type is None:
            type_map = {"image": 1, "video": 2, "audio": 3, "document": 1, "text": 1}
            file_type = type_map.get(media_type, 1)
        raw = target.read_bytes()
        enforce_size_limit(len(raw), 10)
        import base64
        b64 = base64.b64encode(raw).decode("ascii")
        from jarvis.runtime import current_context
        ctx = current_context.get()
        if ctx is None:
            return "Error: no active agent context (send_file requires a running session)"
        session_id = ctx.session.id
        if not session_id.startswith("qq_c2c_"):
            return "Error: send_file can only be used in QQ C2C sessions"
        openid = session_id[len("qq_c2c_"):]
        import httpx
        api_base = "https://api.senvr.net"
        async with httpx.AsyncClient() as client:
            files_resp = await client.post(
                f"{api_base}/openapi/v2/users/{openid}/files",
                json={"file_data": b64, "file_type": file_type},
            )
            if files_resp.status_code != 200:
                return f"Error uploading file: {files_resp.status_code} {files_resp.text}"
            file_info = files_resp.json().get("data", {})
            media_resp = await client.post(
                f"{api_base}/openapi/v2/users/{openid}/messages",
                json={
                    "msg_type": 7,
                    "markdown": {},
                    "media": file_info,
                },
            )
            if media_resp.status_code != 200:
                return f"Error sending media: {media_resp.status_code} {media_resp.text}"
        return f"Sent file: {target.name} ({len(raw)} bytes)"

    return [
        Tool("ls", "List files under a workspace path.", object_params, list_files),
        Tool("read", "Read a UTF-8 text file under the workspace.", read_params, read_file),
        Tool("write", "Create or overwrite a UTF-8 file under the workspace.", write_params, write_file),
        Tool("grep", "Search for text under the workspace.", object_params, search_text),
        Tool("Bash", "Run a shell command. Policy hooks decide whether it is allowed.", object_params, run_command),
        Tool("memory_search", "Search semantic memory for previously stored facts and history.", search_params, search_semantic_memory_tool),
        Tool("memory_purge", "Purge specific items or tags from semantic memory.", purge_params, purge_semantic_memory_tool),
        Tool("memory_store", "Store a fact or procedure in semantic memory.", store_params, store_semantic_memory_tool),
        Tool("memory_check_redundancy", "Find duplicate or near-duplicate memories via embedding similarity.", check_redundancy_params, check_redundancy_tool),
        Tool("memory_distill_now", "Force-distill undistilled session messages into semantic memory.", distill_now_params, distill_now_tool),
        Tool("memory_merge", "Merge two similar memory records into one.", merge_params, merge_memory_tool),
        Tool("memory_update", "Edit text, tags, or confidence of an existing memory record.", update_params, update_memory_tool),
        Tool("task", "Spawn a collaborative subagent to handle a specific task.", spawn_subagent_params, spawn_subagent_handler),
        Tool("message", "Send a message to an active subagent.", send_subagent_message_params, send_subagent_message_handler),
        Tool("close", "Close an active subagent and clean up resources.", close_subagent_params, close_subagent_handler),
        Tool("send_file", "Send a file to the current QQ user. Accepts a local file path.", send_file_params, send_file_handler),
    ]
