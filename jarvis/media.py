from __future__ import annotations

import base64
import math
import subprocess
from pathlib import Path


MEDIA_EXTENSIONS: dict[str, str] = {
    ".mp4": "video",
    ".mov": "video",
    ".avi": "video",
    ".mkv": "video",
    ".webm": "video",
    ".mp3": "audio",
    ".wav": "audio",
    ".ogg": "audio",
    ".flac": "audio",
    ".aac": "audio",
    ".m4a": "audio",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".gif": "image",
    ".webp": "image",
    ".bmp": "image",
    ".pdf": "document",
    ".txt": "text",
    ".py": "text",
    ".md": "text",
    ".js": "text",
    ".ts": "text",
    ".html": "text",
    ".css": "text",
    ".json": "text",
    ".yaml": "text",
    ".yml": "text",
    ".xml": "text",
    ".csv": "text",
    ".log": "text",
    ".sh": "text",
    ".rb": "text",
    ".go": "text",
    ".rs": "text",
    ".c": "text",
    ".cpp": "text",
    ".java": "text",
    ".kt": "text",
    ".swift": "text",
    ".php": "text",
    ".sql": "text",
    ".toml": "text",
    ".ini": "text",
    ".cfg": "text",
    ".conf": "text",
}

MIME_MAP: dict[str, str] = {
    "video": "video/mp4",
    "audio": "audio/mpeg",
    "image": "image/jpeg",
    "document": "application/pdf",
    "text": "text/plain",
}


def detect_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    return MEDIA_EXTENSIONS.get(ext, "text")


def get_mime_type(media_type: str, path: Path | None = None) -> str:
    if path:
        ext = path.suffix.lower()
        if ext == ".jpg":
            return "image/jpeg"
        if ext == ".jpeg":
            return "image/jpeg"
        if ext == ".png":
            return "image/png"
        if ext == ".gif":
            return "image/gif"
        if ext == ".webp":
            return "image/webp"
        if ext == ".mp3":
            return "audio/mpeg"
        if ext == ".wav":
            return "audio/wav"
        if ext == ".ogg":
            return "audio/ogg"
        if ext == ".flac":
            return "audio/flac"
        if ext == ".mp4":
            return "video/mp4"
        if ext == ".mov":
            return "video/quicktime"
        if ext == ".pdf":
            return "application/pdf"
    return MIME_MAP.get(media_type, "application/octet-stream")


def enforce_size_limit(size_bytes: int, max_mb: int) -> None:
    limit = max_mb * 1024 * 1024
    if size_bytes > limit:
        raise ValueError(
            f"File size {size_bytes} bytes exceeds limit of {max_mb} MB"
        )


def to_data_uri(data: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def chunk_count(total: float, chunk_size: float) -> int:
    if chunk_size <= 0:
        return 0
    return math.ceil(total / chunk_size)


def extract_lines(path: Path, start: int, count: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start : start + count])


def extract_pdf_pages(path: Path, start: int, count: int) -> str:
    import PyPDF2

    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = reader.pages
        end = min(start + count, len(pages))
        texts = []
        for i in range(start, end):
            texts.append(pages[i].extract_text() or "")
    return "\n\n--- PAGE BREAK ---\n\n".join(texts)


def extract_media_chunk(path: Path, start_secs: float, duration_secs: float) -> tuple[bytes, str]:
    media_type = detect_media_type(path)
    if media_type == "audio":
        out_mime = "audio/mp4"
        fmt = "mp4"
    else:
        out_mime = get_mime_type(media_type, path)
        fmt = "mp4"
    cmd = [
        "ffmpeg",
        "-i", str(path),
        "-ss", str(start_secs),
        "-t", str(duration_secs),
        "-c", "copy",
        "-f", fmt,
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed: {result.stderr.decode('utf-8', errors='replace')}"
        )
    return result.stdout, out_mime


def get_media_info(path: Path) -> dict:
    """Return media metadata dict."""
    media_type = detect_media_type(path)
    mime = get_mime_type(media_type, path)
    info: dict = {"media_type": media_type, "mime_type": mime}

    if media_type == "text":
        text = path.read_text(encoding="utf-8", errors="replace")
        info["total_lines"] = len(text.splitlines())
        info["size_bytes"] = path.stat().st_size

    elif media_type == "document":
        import PyPDF2
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                info["total_pages"] = len(reader.pages)
        except Exception:
            info["total_pages"] = 0

    elif media_type in ("video", "audio"):
        try:
            probe_cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                probe = __import__("json").loads(result.stdout)
                fmt = probe.get("format", {})
                duration = float(fmt.get("duration", 0))
                info["duration_secs"] = duration
                streams = probe.get("streams", [])
                for stream in streams:
                    if stream.get("codec_type") == "video":
                        info["width"] = int(stream.get("width", 0))
                        info["height"] = int(stream.get("height", 0))
                        break
        except Exception:
            pass

    else:
        info["size_bytes"] = path.stat().st_size

    return info
