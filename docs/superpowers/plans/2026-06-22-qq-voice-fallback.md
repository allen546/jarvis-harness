# QQ Voice Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a robust multi-tier fallback chain for QQ voice messages, including local transcoding (AMR to MP3), QQ native ASR extraction, and local speech-to-text (faster-whisper) fallback.

**Architecture:** We transcode QQ's `.amr` voice message to `.mp3` locally and set the MIME type to `"voice"`. The model layer only generates `input_audio` blocks for `"voice"` attachments. If the LLM call fails, the QQ handler intercepts the exception and falls back to: 1) QQ's native `asr_refer_text`, or 2) local transcription via `faster-whisper`.

**Tech Stack:** Python 3.14, botpy, faster-whisper, ffmpeg, pytest.

---

### Task 1: Dependencies Setup

**Files:**
- Modify: [pyproject.toml](file:///Users/allen/Desktop/jarvis/pyproject.toml)

- [ ] **Step 1: Add dependency to `pyproject.toml`**
  Modify [pyproject.toml](file:///Users/allen/Desktop/jarvis/pyproject.toml) to add `faster-whisper` dependency.
  ```toml
  dependencies = [
      ...
      "faster-whisper>=1.0.3",
  ]
  ```

- [ ] **Step 2: Sync dependencies**
  Run: `uv sync`
  Expected: Success, installs `faster-whisper` and its sub-dependencies.

- [ ] **Step 3: Verify import**
  Run: `.venv/bin/python3 -c "import faster_whisper; print('ok')"`
  Expected: `ok`

- [ ] **Step 4: Commit**
  ```bash
  git add pyproject.toml uv.lock
  git commit -m "chore: add faster-whisper dependency"
  ```

---

### Task 2: Media Transcoding & Local Transcription Helpers

**Files:**
- Modify: [media.py](file:///Users/allen/Desktop/jarvis/jarvis/media.py)
- Modify: [test_media_chunking.py](file:///Users/allen/Desktop/jarvis/tests/test_media_chunking.py)

- [ ] **Step 1: Write the failing tests**
  Add tests to [test_media_chunking.py](file:///Users/allen/Desktop/jarvis/tests/test_media_chunking.py) to check local transcoding and transcription functions.
  ```python
  def test_transcode_amr_to_mp3(monkeypatch):
      import subprocess
      class MockCompletedProcess:
          returncode = 0
          stdout = b"mock-mp3-bytes"
      monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: MockCompletedProcess())
      from jarvis.media import transcode_amr_to_mp3
      res = transcode_amr_to_mp3(b"mock-amr-bytes")
      assert res == b"mock-mp3-bytes"

  def test_transcribe_locally_mocked(monkeypatch):
      class MockSegment:
          text = "hello world"
      class MockWhisperModel:
          def __init__(self, *args, **kwargs):
              pass
          def transcribe(self, *args, **kwargs):
              return [MockSegment()], None
      
      import sys
      from types import ModuleType
      fw = ModuleType("faster_whisper")
      fw.WhisperModel = MockWhisperModel
      sys.modules["faster_whisper"] = fw
      
      from jarvis.media import transcribe_locally
      res = transcribe_locally(b"mock-audio-bytes")
      assert res == "hello world"
  ```

- [ ] **Step 2: Run tests to verify they fail**
  Run: `.venv/bin/pytest tests/test_media_chunking.py -k "transcode or transcribe"`
  Expected: FAIL (cannot import `transcode_amr_to_mp3` / `transcribe_locally`)

- [ ] **Step 3: Implement helper functions in `media.py`**
  Add the following functions to [media.py](file:///Users/allen/Desktop/jarvis/jarvis/media.py):
  ```python
  def transcode_amr_to_mp3(data: bytes) -> bytes:
      import subprocess
      cmd = ["ffmpeg", "-y", "-f", "amr", "-i", "pipe:0", "-f", "mp3", "pipe:1"]
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
  ```

- [ ] **Step 4: Run tests to verify they pass**
  Run: `.venv/bin/pytest tests/test_media_chunking.py -k "transcode or transcribe"`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add jarvis/media.py tests/test_media_chunking.py
  git commit -m "feat: implement media transcoding and local transcription helpers"
  ```

---

### Task 3: Transport Layer ASR & Transcoding Integration

**Files:**
- Modify: [qq.py](file:///Users/allen/Desktop/jarvis/jarvis/transports/qq.py)
- Modify: [test_transports.py](file:///Users/allen/Desktop/jarvis/tests/test_transports.py)

- [ ] **Step 1: Write failing test**
  Add a test to [test_transports.py](file:///Users/allen/Desktop/jarvis/tests/test_transports.py) ensuring that the transport correctly parses `asr_refer_text`, transcodes AMR to MP3, and labels the attachment MIME type as `"voice"`.
  ```python
  @pytest.mark.asyncio
  async def test_qq_voice_attachment_processing(monkeypatch):
      from unittest.mock import AsyncMock, MagicMock
      from jarvis.transports.qq import QQBot
      from jarvis.models.base import Message
      
      monkeypatch.setattr("jarvis.media.transcode_amr_to_mp3", lambda d: b"mock-mp3-bytes")
      
      received_msg = None
      async def mock_on_message(session_id, msg):
          nonlocal received_msg
          received_msg = msg
          return Message(role="assistant", content="reply")

      from botpy import Intents
      bot = QQBot(
          intents=Intents(public_messages=True),
          on_message=mock_on_message,
          allowed_senders=["user123"],
      )
      
      mock_message = MagicMock()
      mock_message.content = ""
      mock_message.author.user_openid = "user123"
      mock_message.id = "msg123"
      
      mock_att = MagicMock()
      mock_att.url = "http://qq.com/voice.amr"
      mock_att.content_type = "voice"
      mock_att.filename = "test.amr"
      
      # Simulate payload with raw dictionary fields
      mock_att._raw_data = {
          "content_type": "voice",
          "url": "http://qq.com/voice.amr",
          "asr_refer_text": "hello from QQ ASR"
      }
      mock_message.attachments = [mock_att]
      
      mock_api = AsyncMock()
      mock_message._api = mock_api
      
      monkeypatch.setattr(bot, "_download", AsyncMock(return_value=b"mock-amr-bytes"))
      
      await bot.on_c2c_message_create(mock_message)
      
      assert received_msg is not None
      assert len(received_msg.attachments) == 1
      att = received_msg.attachments[0]
      assert att.mime_type == "voice"
      assert "audio/mpeg" not in att.mime_type
      assert "base64," in att.url
      assert received_msg.metadata.get("asr_text") == "hello from QQ ASR"
  ```

- [ ] **Step 2: Run tests to verify they fail**
  Run: `.venv/bin/pytest tests/test_transports.py -k "test_qq_voice_attachment_processing"`
  Expected: FAIL

- [ ] **Step 3: Modify `qq.py` to support `asr_refer_text` and voice transcoding**
  In [qq.py](file:///Users/allen/Desktop/jarvis/jarvis/transports/qq.py):
  1. Add monkey-patching of `Message._Attachments` at the top of the file:
     ```python
     # Monkey-patch botpy Message._Attachments to preserve raw JSON dict
     _old_attachments_init = botpy.message.Message._Attachments.__init__
     def _new_attachments_init(self, data):
         _old_attachments_init(self, data)
         self._raw_data = data
     botpy.message.Message._Attachments.__init__ = _new_attachments_init
     ```
  2. Modify `on_c2c_message_create`:
     - If `att.content_type == "voice"`, download, run `transcode_amr_to_mp3`, build Data URI.
     - Extract `asr_text = getattr(att, "_raw_data", {}).get("asr_refer_text", "").strip()` (or from `att.asr_refer_text` if it gets added, but fallback to `_raw_data`).
     - Set the attachment's `mime_type` to `"voice"` and description / filename to end in `.mp3`.
     - Pass `metadata={"asr_text": asr_text}` to the `Message` constructor.
     - Ensure the code looks like:
     ```python
             for att in botpy_attachments:
                 att_url = getattr(att, "url", None)
                 if att_url:
                     mime = getattr(att, "content_type", None) or "application/octet-stream"
                     if self._media_supported(mime):
                         data = await self._download(att_url)
                         if data and len(data) <= self._max_download_bytes:
                             filename = getattr(att, "filename", None) or "sticker.jpg"
                             asr_text = ""
                             if mime == "voice":
                                 # Transcode AMR to MP3 bytes
                                 from jarvis.media import transcode_amr_to_mp3
                                 try:
                                     data = transcode_amr_to_mp3(data)
                                     mime = "voice"
                                     filename = Path(filename).with_suffix(".mp3").name
                                 except Exception as exc:
                                     logger.error("qq: failed to transcode AMR to MP3: %s", exc)
                                 
                                 # Extract platform ASR text from raw data
                                 raw_data = getattr(att, "_raw_data", {}) or {}
                                 asr_text = raw_data.get("asr_refer_text", "").strip()

                             attachments.append(Attachment(
                                 mime_type=mime,
                                 url=to_data_uri(data, mime),
                                 description="sticker" if is_sticker else filename,
                             ))
     ```
     Pass `metadata={"asr_text": asr_text}` to `Message`:
     ```python
             jarvis_msg = Message(role="user", content=content, attachments=attachments, metadata={"asr_text": asr_text if mime == "voice" else ""})
     ```

- [ ] **Step 4: Run tests to verify they pass**
  Run: `.venv/bin/pytest tests/test_transports.py -k "test_qq_voice_attachment_processing"`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add jarvis/transports/qq.py tests/test_transports.py
  git commit -m "feat: parse asr_refer_text and transcode voice to voice/mp3 in QQ bot"
  ```

---

### Task 4: LLM Client voice MIME Filter

**Files:**
- Modify: [openai.py](file:///Users/allen/Desktop/jarvis/jarvis/models/openai.py)
- Modify: [test_model_providers.py](file:///Users/allen/Desktop/jarvis/tests/test_model_providers.py)

- [ ] **Step 1: Write failing test**
  Add a test to [test_model_providers.py](file:///Users/allen/Desktop/jarvis/tests/test_model_providers.py) that asserts `_attachment_to_content_block` returns `input_audio` block ONLY for MIME type `"voice"`, and returns `None` for standard `"audio/*"` types.
  ```python
  def test_attachment_to_content_block_types():
      from jarvis.models.openai import _attachment_to_content_block
      from jarvis.models.base import Attachment
      
      # Voice MIME should map to input_audio
      voice_att = Attachment(mime_type="voice", url="data:voice;base64,YWJj")
      block1 = _attachment_to_content_block(voice_att)
      assert block1 is not None
      assert block1["type"] == "input_audio"
      assert block1["input_audio"]["format"] == "mp3"
      
      # Standard audio MIME should NOT map to input_audio (returns None)
      audio_att = Attachment(mime_type="audio/mpeg", url="data:audio/mpeg;base64,YWJj")
      block2 = _attachment_to_content_block(audio_att)
      assert block2 is None
  ```

- [ ] **Step 2: Run tests to verify they fail**
  Run: `.venv/bin/pytest tests/test_model_providers.py -k "test_attachment_to_content_block_types"`
  Expected: FAIL

- [ ] **Step 3: Modify `_attachment_to_content_block` in `openai.py`**
  Modify `_attachment_to_content_block` in [openai.py](file:///Users/allen/Desktop/jarvis/jarvis/models/openai.py):
  ```python
  def _attachment_to_content_block(att: Attachment) -> dict[str, Any] | None:
      mime = att.mime_type
      url = att.url
      if not url:
          return None
      if mime.startswith("image/"):
          return {"type": "image_url", "image_url": {"url": url}}
      if mime == "voice":
          b64 = url.split(",", 1)[1] if "," in url else ""
          return {"type": "input_audio", "input_audio": {"data": b64, "format": "mp3"}}
      if mime.startswith("video/"):
          return {"type": "video_url", "video_url": {"url": url}}
      return None
  ```

- [ ] **Step 4: Run tests to verify they pass**
  Run: `.venv/bin/pytest tests/test_model_providers.py -k "test_attachment_to_content_block_types"`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add jarvis/models/openai.py tests/test_model_providers.py
  git commit -m "feat: restrict input_audio blocks to voice MIME type only"
  ```

---

### Task 5: Orchestration and Fallback Implementation

**Files:**
- Modify: [main.py](file:///Users/allen/Desktop/jarvis/main.py)
- Create: `tests/test_voice_fallback.py`

- [ ] **Step 1: Write failing test for fallback orchestration**
  Create a new test file `tests/test_voice_fallback.py` to test the multi-tier fallback chain.
  ```python
  import pytest
  from unittest.mock import AsyncMock, MagicMock
  from jarvis.models.base import Message, Attachment
  
  @pytest.mark.asyncio
  async def test_qq_handler_fallback_chain(monkeypatch):
      # Mock the session history cleanup
      mock_session = MagicMock()
      mock_session.ctx.session.history = []
      
      mock_manager = MagicMock()
      mock_manager.get_or_create.return_value = mock_session
      
      # Mock transcribe_locally
      monkeypatch.setattr("jarvis.media.transcribe_locally", lambda b: "whisper transcript")
      
      # Simulate a call to _qq_handler
      from main import main
      # Set up mock flows inside the test
  ```
  We will write a comprehensive unit test in the new file `tests/test_voice_fallback.py` representing:
  - Step 1 fails with exception -> fallback to ASR text -> succeeds.
  - Step 1 fails with exception -> ASR text empty -> fallback to Whisper transcription -> succeeds.

- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv/bin/pytest tests/test_voice_fallback.py`
  Expected: FAIL / ModuleNotFoundError

- [ ] **Step 3: Modify `_qq_handler` inside `main.py`**
  Modify [main.py](file:///Users/allen/Desktop/jarvis/main.py)'s `_qq_handler` implementation (around line 118):
  ```python
      if config.channels.qq.enabled:
          async def _qq_handler(session_id: str, message: Message) -> Message:
              # Check if message contains a voice attachment
              voice_att = None
              for att in message.attachments:
                  if att.mime_type == "voice":
                      voice_att = att
                      break
              
              if not voice_att:
                  return await _manager.submit_and_collect(session_id, message)
              
              # Tier 1: Try native audio submit
              try:
                  return await _manager.submit_and_collect(session_id, message)
              except Exception as exc:
                  logger.warning("qq_handler: Tier 1 audio completion failed: %s. Initiating fallback.", exc)
                  session = _manager.get_or_create(session_id)
                  if session.ctx.session.history and session.ctx.session.history[-1] == message:
                      session.ctx.session.history.pop()
                  
                  # Tier 2: Platform ASR
                  asr_text = message.metadata.get("asr_text") if message.metadata else None
                  if asr_text:
                      logger.info("qq_handler: Tier 2 falling back to platform ASR: %s", asr_text)
                      asr_message = Message(
                          role="user",
                          content=f"<VOICE_TRANSCRIPTION>{asr_text}</VOICE_TRANSCRIPTION>",
                          metadata=message.metadata
                      )
                      try:
                          return await _manager.submit_and_collect(session_id, asr_message)
                      except Exception as asr_exc:
                          logger.warning("qq_handler: Tier 2 ASR completion failed: %s. Cleaning history.", asr_exc)
                          if session.ctx.session.history and session.ctx.session.history[-1] == asr_message:
                              session.ctx.session.history.pop()
                  
                  # Tier 3: Local Whisper transcription
                  logger.info("qq_handler: Tier 3 falling back to local transcription")
                  url = voice_att.url
                  b64_data = url.split(",", 1)[1] if url and "," in url else ""
                  import base64
                  audio_bytes = base64.b64decode(b64_data)
                  
                  from jarvis.media import transcribe_locally
                  try:
                      transcript = transcribe_locally(audio_bytes)
                      logger.info("qq_handler: local transcription success: %s", transcript)
                      whisper_message = Message(
                          role="user",
                          content=f"<VOICE_TRANSCRIPTION>{transcript}</VOICE_TRANSCRIPTION>",
                          metadata=message.metadata
                      )
                      return await _manager.submit_and_collect(session_id, whisper_message)
                  except Exception as whisper_exc:
                      logger.error("qq_handler: local transcription fallback failed: %s", whisper_exc)
                      raise
  ```

- [ ] **Step 4: Run tests to verify they pass**
  Run: `.venv/bin/pytest tests/test_voice_fallback.py`
  Expected: PASS

- [ ] **Step 5: Run all test suites**
  Run: `.venv/bin/pytest`
  Expected: 121 passed (all old + new tests)

- [ ] **Step 6: Commit**
  ```bash
  git add main.py tests/test_voice_fallback.py
  git commit -m "feat: implement multi-tier voice fallback orchestration in qq_handler"
  ```
