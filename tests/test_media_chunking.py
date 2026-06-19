"""Tests for media handling and universal chunking."""

import pytest
from pathlib import Path
from jarvis.models.base import Attachment, Message, ToolCall
from jarvis.tools import Tool, ToolRegistry, ToolResult
from jarvis.media import (
    detect_media_type, to_data_uri, chunk_count,
    enforce_size_limit, get_mime_type,
)


# --- Attachment ---

def test_attachment_url_field():
    a = Attachment(mime_type="image/jpeg", url="data:image/jpeg;base64,abc")
    assert a.url == "data:image/jpeg;base64,abc"
    assert a.model_dump()["url"] == "data:image/jpeg;base64,abc"


def test_attachment_file_path():
    a = Attachment(mime_type="text/plain", file_path="/tmp/test.txt")
    assert a.file_path == "/tmp/test.txt"
    assert a.url is None


def test_attachment_validation_error_if_neither():
    with pytest.raises(ValueError, match="Attachment requires url or file_path"):
        Attachment(mime_type="text/plain")


# --- ToolResult attachments ---

@pytest.mark.asyncio
async def test_tool_result_attachments():
    registry = ToolRegistry()
    result = await registry.execute(ToolCall(call_id="c1", tool_name="unknown", arguments={}))
    assert result.attachments == []


def test_tool_result_has_attachments():
    r = ToolResult(call_id="id", tool_name="name", content="c")
    assert r.attachments == []
    r.attachments.append(Attachment(mime_type="image/png", url="data:"))
    assert len(r.attachments) == 1


# --- read chunking ---

@pytest.mark.asyncio
async def test_read_text_chunking(tmp_path: Path):
    # Create a file with >200 lines
    lines = [f"line {i}" for i in range(500)]
    test_file = tmp_path / "big.py"
    test_file.write_text("\n".join(lines))

    from jarvis.tools import builtin_tools
    tools = builtin_tools(tmp_path)
    read_tool = next(t for t in tools if t.name == "read")
    handler = read_tool.handler

    # First chunk: lines 0-199 (displayed as 1-200)
    result = handler({"path": "big.py", "chunk": 0})
    assert isinstance(result, str)
    assert "line 0" in result
    assert "line 199" in result
    assert "chunk 0/2" in result

    # Second chunk: lines 200-399 (displayed as 201-400)
    result = handler({"path": "big.py", "chunk": 1})
    assert "line 200" in result
    assert "line 399" in result

    # Third chunk: lines 400-499 (displayed as 401-500)
    result = handler({"path": "big.py", "chunk": 2})
    assert "line 400" in result
    assert "line 499" in result


@pytest.mark.asyncio
async def test_read_small_file_no_chunk(tmp_path: Path):
    test_file = tmp_path / "small.py"
    test_file.write_text("hello\nworld\n")

    from jarvis.tools import builtin_tools
    tools = builtin_tools(tmp_path)
    read_tool = next(t for t in tools if t.name == "read")
    handler = read_tool.handler

    result = handler({"path": "small.py"})
    assert "hello" in result
    assert "world" in result
    assert "chunk" not in result


# --- send_file ---

@pytest.mark.asyncio
async def test_send_file_not_available_without_qq(tmp_path: Path):
    from jarvis.tools import builtin_tools
    tools = builtin_tools(tmp_path)
    sf = next(t for t in tools if t.name == "send_file")
    # Without QQ context, should return error
    result = await sf.handler({"path": "nonexistent.txt"})
    assert "Error" in result or "error" in result


# --- multimodal content blocks ---

def test_multimodal_content_blocks():
    from jarvis.models.openai import _attachment_to_content_block

    img = Attachment(mime_type="image/jpeg", url="data:image/jpeg;base64,abc")
    block = _attachment_to_content_block(img)
    assert block == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}}

    audio = Attachment(mime_type="audio/mpeg", url="data:audio/mpeg;base64,xyz")
    block = _attachment_to_content_block(audio)
    assert block["type"] == "input_audio"
    assert block["input_audio"]["data"] == "xyz"
    assert block["input_audio"]["format"] == "mpeg"

    video = Attachment(mime_type="video/mp4", url="http://example.com/v.mp4")
    block = _attachment_to_content_block(video)
    assert block["type"] == "video_url"
    assert block["video_url"]["url"] == "http://example.com/v.mp4"

    # No url → None
    no_url = Attachment(mime_type="image/png", file_path="/tmp/img.png")
    assert _attachment_to_content_block(no_url) is None


def test_supported_media_filter():
    from jarvis.models.openai import _attachment_to_content_block

    # Unsupported mime (no prefix match)
    att = Attachment(mime_type="application/zip", url="data:application/zip;base64,x")
    assert _attachment_to_content_block(att) is None


# --- media info detection ---

def test_media_info_detection():
    assert detect_media_type(Path("video.mp4")) == "video"
    assert detect_media_type(Path("audio.mp3")) == "audio"
    assert detect_media_type(Path("photo.jpg")) == "image"
    assert detect_media_type(Path("doc.pdf")) == "document"
    assert detect_media_type(Path("script.py")) == "text"


def test_to_data_uri():
    uri = to_data_uri(b"hello", "text/plain")
    assert uri == "data:text/plain;base64,aGVsbG8="


def test_chunk_count():
    assert chunk_count(10.0, 3.0) == 4
    assert chunk_count(9.0, 3.0) == 3
    assert chunk_count(0, 5.0) == 0


def test_enforce_size_limit():
    enforce_size_limit(5 * 1024 * 1024, 10)  # OK
    with pytest.raises(ValueError, match="exceeds limit"):
        enforce_size_limit(11 * 1024 * 1024, 10)


def test_get_mime_type():
    assert get_mime_type("image", Path("photo.jpg")) == "image/jpeg"
    assert get_mime_type("image", Path("photo.png")) == "image/png"
    assert get_mime_type("audio", Path("song.mp3")) == "audio/mpeg"
    assert get_mime_type("video", Path("clip.mp4")) == "video/mp4"
