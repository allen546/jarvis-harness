from __future__ import annotations

import base64
import math
import subprocess
from pathlib import Path


# ponytail: only extensions that drive non-text behavior need entries here.
# Everything else falls through to "text" via the default.
_NON_TEXT_EXTENSIONS: dict[str, str] = {
    ".mp4": "video", ".mov": "video", ".avi": "video", ".mkv": "video", ".webm": "video",
    ".mp3": "audio", ".wav": "audio", ".ogg": "audio", ".flac": "audio", ".aac": "audio", ".m4a": "audio",
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image", ".webp": "image", ".bmp": "image",
    ".pdf": "document",
}

# ponytail: extension → MIME, used by get_mime_type when a path is given.
EXT_MIME: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif",
    ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".flac": "audio/flac",
    ".aac": "audio/aac", ".m4a": "audio/mp4",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska", ".webm": "video/webm",
    ".pdf": "application/pdf",
}

_MEDIA_MIME: dict[str, str] = {
    "video": "video/mp4", "audio": "audio/mpeg", "image": "image/jpeg",
    "document": "application/pdf", "text": "text/plain",
}


def detect_media_type(path: Path) -> str:
    return _NON_TEXT_EXTENSIONS.get(path.suffix.lower(), "text")


def get_mime_type(media_type: str, path: Path | None = None) -> str:
    if path:
        return EXT_MIME.get(path.suffix.lower(), _MEDIA_MIME.get(media_type, "application/octet-stream"))
    return _MEDIA_MIME.get(media_type, "application/octet-stream")


def enforce_size_limit(size_bytes: int, max_mb: int) -> None:
    limit = max_mb * 1024 * 1024
    if size_bytes > limit:
        raise ValueError(f"File size {size_bytes} bytes exceeds limit of {max_mb} MB")


def to_data_uri(data: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def chunk_count(total: float, chunk_size: float) -> int:
    if chunk_size <= 0:
        return 0
    return math.ceil(total / chunk_size)


def extract_pdf_pages(path: Path, start: int, count: int) -> str:
    import PyPDF2
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = reader.pages
        end = min(start + count, len(pages))
        texts = [pages[i].extract_text() or "" for i in range(start, end)]
    return "\n\n--- PAGE BREAK ---\n\n".join(texts)


def extract_media_chunk(path: Path, start_secs: float, duration_secs: float) -> tuple[bytes, str]:
    media_type = detect_media_type(path)
    # ponytail: always mux as mp4 — it's the ffmpeg default container for copy mode.
    # audio gets audio/mp4, video keeps original MIME.
    out_mime = "audio/mp4" if media_type == "audio" else get_mime_type(media_type, path)
    cmd = [
        "ffmpeg", "-i", str(path),
        "-ss", str(start_secs), "-t", str(duration_secs),
        "-c", "copy", "-f", "mp4", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode('utf-8', errors='replace')}")
    return result.stdout, out_mime


def get_media_info(path: Path) -> dict:
    """Return media metadata dict."""
    media_type = detect_media_type(path)
    mime = get_mime_type(media_type, path)
    info: dict = {"media_type": media_type, "mime_type": mime}

    if media_type == "text":
        info["total_lines"] = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        info["size_bytes"] = path.stat().st_size
    elif media_type == "document":
        import PyPDF2
        try:
            with open(path, "rb") as f:
                info["total_pages"] = len(PyPDF2.PdfReader(f).pages)
        except Exception:
            info["total_pages"] = 0
    elif media_type in ("video", "audio"):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                probe = __import__("json").loads(result.stdout)
                info["duration_secs"] = float(probe.get("format", {}).get("duration", 0))
                for stream in probe.get("streams", []):
                    if stream.get("codec_type") == "video":
                        info["width"] = int(stream.get("width", 0))
                        info["height"] = int(stream.get("height", 0))
                        break
        except Exception:
            pass
    else:
        info["size_bytes"] = path.stat().st_size

    return info

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


def transcode_amr_to_mp3(data: bytes) -> bytes:
    import subprocess
    cmd = ["ffmpeg", "-y", "-f", "amr", "-i", "pipe:0", "-f", "mp3", "pipe:1"]
    result = subprocess.run(cmd, input=data, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode('utf-8', errors='replace')}")
    return result.stdout

def transcode_wav_to_mp3(data: bytes) -> bytes:
    import subprocess
    cmd = ["ffmpeg", "-y", "-f", "wav", "-i", "pipe:0", "-f", "mp3", "pipe:1"]
    result = subprocess.run(cmd, input=data, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode('utf-8', errors='replace')}")
    return result.stdout

def transcribe_locally(audio_bytes: bytes) -> str:
    import tempfile
    from pathlib import Path
    from faster_whisper import WhisperModel
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        tf.write(audio_bytes)
        temp_path = Path(tf.name)
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(temp_path), beam_size=5)
        return "".join(segment.text for segment in segments)
    finally:
        if temp_path.exists():
            temp_path.unlink()
