# Media Handling & Universal Chunking

## Overview

Add media support end-to-end: inbound attachments from QQ, outbound file sending, and universal chunking for large files (text, PDF, video, audio). The model receives media as vision/audio content blocks, can request more chunks on demand, and can send files back to the user.

## Key Design Decisions

- **One `read` tool** handles all file access with a `chunk` parameter for progressive disclosure
- **QQ C2C file sending** uses `file_data` (base64 in JSON body), not URLs — no tunnel/server needed
- **Lazy chunking**: first chunk sent eagerly, model requests more by index
- **`send_file` tool** accepts local file paths — reads, base64-encodes, sends via QQ API
- **Multimodal content blocks** in OpenAI client built from `Message.attachments`

---

## Changes by File

### 1. `jarvis/config.py` — Media config on `ModelConfig`

```python
class ModelConfig(BaseModel):
    # ... existing fields ...
    supported_media: list[str] = []        # MIME prefixes: "image", "audio", "video", "document"
    max_download_size_mb: int = 10         # per-attachment size limit
    video_chunk_secs: int = 10             # video chunk duration
    audio_chunk_secs: int = 60             # audio chunk duration
    text_chunk_lines: int = 200            # text/code chunk size (lines)
    pdf_chunk_pages: int = 2               # PDF chunk size (pages)
```

### 2. `jarvis/models/base.py` — Attachment & ToolResult

**Attachment** — add `url` field for data URIs:

```python
@dataclass(slots=True)
class Attachment:
    mime_type: str
    url: str | None = None          # data:image/jpeg;base64,... or HTTP URL
    file_path: str | None = None    # local file path
    description: str | None = None

    def __post_init__(self):
        if not self.url and not self.file_path:
            raise ValueError("Attachment requires url or file_path")

    def model_dump(self) -> dict[str, Any]:
        return {
            "mime_type": self.mime_type,
            "url": self.url,
            "file_path": self.file_path,
            "description": self.description,
        }
```

**ToolResult** — add `attachments` field:

```python
@dataclass(slots=True)
class ToolResult:
    call_id: str
    tool_name: str
    content: str
    attachments: list[Attachment] = field(default_factory=list)
    is_error: bool = False
```

### 3. `jarvis/media.py` — New module

Shared helpers for media processing:

```python
@dataclass
class MediaInfo:
    media_type: str          # "video", "audio", "image", "document"
    mime_type: str           # "video/mp4", "audio/mpeg", etc.
    duration_secs: float | None = None
    width: int | None = None
    height: int | None = None
    total_lines: int | None = None
    total_pages: int | None = None

def get_media_info(path: Path) -> MediaInfo
    # ffprobe for video/audio, PyPDF2 for PDF, line count for text

def extract_lines(path: Path, start: int, count: int) -> str
    # Read lines [start:start+count]

def extract_pdf_pages(path: Path, start: int, count: int) -> str
    # PyPDF2: extract text from pages [start:start+count]

def extract_media_chunk(path: Path, start_secs: float, duration_secs: float) -> Attachment
    # ffmpeg: extract segment, base64-encode, return Attachment with data URI

def to_data_uri(data: bytes, mime_type: str) -> str
    # f"data:{mime_type};base64,{b64encode(data).decode()}"

def chunk_count(total: float, chunk_size: float) -> int
    # ceil(total / chunk_size)

def detect_media_type(path: Path) -> str
    # Extension-based: video, audio, image, document, text
```

All functions enforce `max_download_size_mb` from config. ffmpeg and ffprobe called via `subprocess` (no Python binding — standard on Raspberry Pi).

### 4. `jarvis/tools.py` — `read` upgrade + `send_file` + type change

**ToolHandler return type:**

```python
ToolHandler = Callable[[dict[str, Any]], Awaitable[str | ToolResult] | str | ToolResult]
```

**ToolRegistry.execute** — check return type:

```python
result = await tool.handler(call.arguments)
if isinstance(result, ToolResult):
    result.call_id = call.call_id
    result.tool_name = call.tool_name
    return result
return ToolResult(call_id=call.call_id, tool_name=call.tool_name, content=result)
```

**`read` tool** — new `chunk` parameter:

```json
{
  "type": "object",
  "properties": {
    "path": {"type": "string", "description": "File path relative to workspace."},
    "chunk": {"type": "integer", "description": "Chunk index (0-based). Omit for first chunk of large files."}
  },
  "required": ["path"]
}
```

Behavior by file type:

| Type | Under threshold | Over threshold |
|---|---|---|
| Text/code | Return full content (str) | Return chunk N of M (200 lines each), header with metadata |
| PDF | Return full text (str) | Return chunk N of M (2 pages each), header with metadata |
| Image | Return metadata (str) + data URI attachment | N/A (never chunked) |
| Video | Return first chunk + metadata + data URI attachment | Same (always chunked by duration) |
| Audio | Return first chunk + metadata + data URI attachment | Same (always chunked by duration) |

Content format for chunked files:

```
[lines 1-200 of 1000 | chunk 0/5]
<content>
---
read(path="file.py", chunk=1) → next chunk
```

**`send_file` tool** — new:

```json
{
  "name": "send_file",
  "description": "Send a file to the current QQ user. Accepts a local file path.",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "description": "Local file path relative to workspace."},
      "file_type": {"type": "integer", "description": "QQ file type: 1=image, 2=video, 3=audio. Auto-detected if omitted."}
    },
    "required": ["path"]
  }
}
```

Implementation:
1. Extract OpenID from session ID (`qq_c2c_{openid}` prefix → openid)
2. If no `file_type`, auto-detect from MIME type
3. Read file bytes, enforce size limit
4. Base64-encode
5. POST to QQ API `/v2/users/{openid}/files` with `file_data` field (not `url`)
6. Use returned `file_info` to send media message via `/v2/users/{openid}/messages` with `msg_type=7`

This follows ZeroClaw's proven pattern — base64 upload, no URL hosting needed.

### 5. `jarvis/transports/qq.py` — Message-based callback

**Callback signature** changes from `(session_id, str) -> str` to `(session_id, Message) -> Message`.

**`on_c2c_message_create`:**

```python
async def on_c2c_message_create(self, message: Message) -> None:
    content = (message.content or "").strip()
    openid = message.author.user_openid

    # ... allowed_senders check ...

    # Build attachments from botpy message attachments
    attachments = []
    for att in (message.attachments or []):
        if att.url:
            mime = att.content_type or "application/octet-stream"
            # Check if model supports this media type
            if _media_supported(mime, self._supported_media):
                # Download and create data URI
                data = await _download(att.url, self._proxy_env)
                if len(data) <= self._max_download_bytes:
                    attachments.append(Attachment(
                        mime_type=mime,
                        url=to_data_uri(data, mime),
                        description=att.filename,
                    ))

    jarvis_msg = Message(role="user", content=content, attachments=attachments)

    try:
        reply_msg = await self._on_message(f"qq_c2c_{openid}", jarvis_msg)
        reply_text = reply_msg.content
    except Exception as exc:
        reply_text = f"[error] {exc}"

    await message._api.post_c2c_message(
        openid=openid, msg_type=2,
        markdown={"content": reply_text}, msg_id=message.id,
    )
```

**`QQChannel.__init__`** gains `supported_media` and `max_download_size_mb` params (from config).

### 6. `jarvis/sessions.py` — Message in, Message out

```python
async def submit_and_collect(self, session_id: str, message: Message) -> Message:
    """Submit a message and return the final assistant Message."""
    result = Message(role="assistant", content="")
    async for event in self.submit(session_id, message):
        if isinstance(event, ErrorEvent):
            raise RuntimeError(event.message)
        if isinstance(event, MessageEvent):
            result = event.message
    return result
```

### 7. `jarvis/models/openai.py` — Multimodal content blocks

In `generate()` and `generate_stream()`, when building `openai_msgs`:

```python
for m in messages:
    if m.attachments:
        # Build multimodal content blocks
        content_blocks: list[dict[str, Any]] = []
        if m.content:
            content_blocks.append({"type": "text", "text": m.content})
        for att in m.attachments:
            block = _attachment_to_content_block(att)
            if block:
                content_blocks.append(block)
        msg = {"role": m.role, "content": content_blocks}
    else:
        msg = {"role": m.role, "content": m.content}
    # ... tool_calls handling unchanged ...
```

`_attachment_to_content_block`:

```python
def _attachment_to_content_block(att: Attachment) -> dict[str, Any] | None:
    mime = att.mime_type
    url = att.url
    if not url:
        return None
    if mime.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": url}}
    if mime.startswith("audio/"):
        # base64 data from data URI
        b64 = url.split(",", 1)[1] if "," in url else ""
        fmt = mime.split("/", 1)[1].split(";")[0]  # "mp3", "wav", etc.
        return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}
    if mime.startswith("video/"):
        return {"type": "video_url", "video_url": {"url": url}}
    return None
```

Filter by `supported_media` from config (only include blocks for supported MIME prefixes).

### 8. `main.py` — Callback wiring

```python
async def _qq_handler(session_id: str, message: Message) -> Message:
    result_msg = Message(role="assistant", content="")
    async for event in _manager.submit(session_id, message):
        if isinstance(event, ErrorEvent):
            raise RuntimeError(event.message)
        if isinstance(event, MessageEvent):
            result_msg = event.message
    return result_msg
```

Pass `supported_media` and `max_download_size_mb` from config to `QQChannel`.

### 9. `pyproject.toml` — New dependency

```
"PyPDF2>=3.0.0",
```

### 10. Remote deployment

- `apt install ffmpeg` on Raspberry Pi (if not already installed)
- Update `config/global.json` with media config fields
- `pip install PyPDF2` in the venv

### 11. Tests

| Test | What it covers |
|---|---|
| `test_attachment_url_field` | Attachment with `url`, with `file_path`, validation error if neither |
| `test_tool_result_attachments` | ToolResult carries attachments through `execute()` |
| `test_read_text_chunking` | `read(path, chunk=1)` returns correct line range |
| `test_read_pdf_chunking` | `read(path, chunk=0)` returns first N pages |
| `test_read_small_file_no_chunk` | Small files return full content, no chunk metadata |
| `test_send_file_tool` | `send_file` calls QQ API with `file_data` field |
| `test_multimodal_content_blocks` | OpenAI client builds correct blocks for image/audio/video attachments |
| `test_supported_media_filter` | Unsupported media types excluded from content blocks |
| `test_media_info_detection` | `detect_media_type` returns correct type for each extension |

---

## Data Flow

### Inbound (QQ → model)

```
QQ botpy message (has .attachments with CDN URLs)
  → QQBot.on_c2c_message_create
    → Download attachment from CDN URL (check size limit, check supported_media)
    → base64-encode → data:image/jpeg;base64,... URI
    → Attachment(mime_type="image/jpeg", url="data:...")
    → Message(role="user", content="...", attachments=[Attachment])
      → SessionManager.submit
        → kernel.run_turn
          → OpenAI client builds multimodal content blocks
            → API call with image/audio/video
```

### Outbound (model → QQ)

```
Model calls send_file(path="photo.jpg")
  → Tool handler reads file, enforces size limit
  → base64-encode
  → POST /v2/users/{openid}/files with file_data field
  → Get file_info back
  → POST /v2/users/{openid}/messages with msg_type=7, media={file_info}
  → QQ delivers file to user
```

### Chunked file read

```
Model calls read(path="big.py")
  → Detect 800 lines → chunk into 4 × 200
  → Return lines 1-200 with metadata: "[lines 1-200 of 800 | chunk 0/4]"
  → Footer: "read(path=\"big.py\", chunk=1) → next chunk"

Model calls read(path="big.py", chunk=2)
  → Return lines 401-600 with metadata
```

### Video/audio chunking

```
Model calls read(path="video.mp4")
  → ffprobe: 45s total → 5 chunks (10s each)
  → ffmpeg: extract 0-10s → base64 data URI
  → Return Attachment + metadata
  → Footer: "read(path=\"video.mp4\", chunk=1) → next chunk"
```

---

## What stays out of scope

- DOCX/XLSX text extraction (additional heavy dependencies)
- Image/audio/video content generation by the model
- Per-media-type size limits (one global limit for now)
- Upload caching (ZeroClaw caches `file_info` with TTL — can add later)
- Streaming video download (download full file, then chunk locally)
