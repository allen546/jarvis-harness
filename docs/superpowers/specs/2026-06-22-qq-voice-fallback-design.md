# Spec: Multi-Tier QQ Voice Fallback System

## Goal Description
Currently, when a user sends a voice message on the QQ channel, it is downloaded as an `.amr` file with the MIME type `"voice"`. This file is not sent to the LLM as audio because the OpenAI client only supports standard MIME types starting with `"audio/"` (like `audio/mpeg` or `audio/wav`), and standard APIs do not support the AMR format. 

This design implements a robust, multi-tier fallback chain to ensure voice messages are processed gracefully. 

Additionally, as a key design constraint, we do **not** send generic audio types (e.g., `audio/mpeg` or `audio/mp4` returned by tools like `read_file`) to the model directly. Instead, we **only** send attachments with the specific `"voice"` MIME type as `input_audio` blocks to the LLM, distinguishing voice speech from generic audio files by their MIME type (and not by file postfix).

## Proposed Changes

### 1. Dependencies (`pyproject.toml`)
We add `faster-whisper` to the project dependencies to enable local CPU-optimized speech-to-text transcription on the Raspberry Pi.

```toml
dependencies = [
    ...
    "faster-whisper>=1.0.3",
]
```

---

### 2. Media Helper Layer (`jarvis/media.py`)
We implement two helper functions:
1. `transcode_amr_to_mp3(data: bytes) -> bytes`: Runs a local `ffmpeg` process using stdin/stdout streams to convert AMR bytes to MP3 bytes.
2. `transcribe_locally(audio_bytes: bytes) -> str`: Loads the `faster-whisper` model (using the `"tiny"` model on CPU with `int8` quantization for optimal memory and speed on the Pi) and returns the transcribed text.

---

### 3. Transport Layer (`jarvis/transports/qq.py`)
* **Monkey-patching `botpy`:** At module load time, we monkey-patch `botpy.message.Message._Attachments.__init__` to retain the raw dictionary payload (`self._raw_data = data`), which contains platform-specific fields.
* **ASR Extraction:** In `on_c2c_message_create`, when a voice attachment is received, we extract `att._raw_data.get("asr_refer_text")` and attach it to the `Message` object's metadata dictionary as `metadata={"asr_text": asr_refer_text}`.
* **Transcoding:** We download the voice attachment, run `transcode_amr_to_mp3`, and set the attachment's MIME type to `"voice"`. This explicitly designates the transcoded MP3 bytes as a voice message.

---

### 4. LLM Client Layer (`jarvis/models/openai.py`)
We modify `_attachment_to_content_block(att: Attachment)` to only generate `input_audio` content blocks for `"voice"` MIME type attachments:
```python
if mime == "voice":
    b64 = url.split(",", 1)[1] if "," in url else ""
    return {"type": "input_audio", "input_audio": {"data": b64, "format": "mp3"}}
```
Generic audio attachments (e.g. `audio/mpeg` or `audio/mp4`) returned by the `read_file` tool will not be wrapped in `input_audio` blocks, preventing the model from receiving non-voice audio directly as API payloads.

---

### 5. Channel Handler/Orchestration Layer (`main.py`)
We update `_qq_handler` to orchestrate the fallback loop:
1. **Tier 1 (Native Audio):** Call `_manager.submit_and_collect` with the message containing the `"voice"` attachment. If it succeeds, return the LLM response.
2. **Tier 2 (Platform ASR Fallback):** If Tier 1 raises an error:
   - Remove the failed message from the session history.
   - Extract `asr_text` from the metadata. If present, submit a new text-only message containing `<VOICE_TRANSCRIPTION>{asr_text}</VOICE_TRANSCRIPTION>`.
3. **Tier 3 (Local Whisper Fallback):** If Tier 2 fails or if the ASR text is not available:
   - Run `transcribe_locally` on the MP3 bytes.
   - Remove the failed message from the session history.
   - Submit a new text-only message containing `<VOICE_TRANSCRIPTION>{local_whisper_text}</VOICE_TRANSCRIPTION>`.

---

## Verification Plan

### Automated Tests
* Create unit tests to verify:
  1. `transcode_amr_to_mp3` correctly converts sample AMR files to MP3.
  2. `transcribe_locally` returns expected transcriptions for sample audio.
  3. `_qq_handler` executes the fallback flow step-by-step when simulated model turns fail.
  4. OpenAI payload construction only includes `input_audio` blocks for `"voice"` attachments and excludes generic `"audio/*"` attachments.

### Manual Verification
* Send a voice message via QQ to the running bot:
  1. Verify the bot receives the `.amr` file, transcodes it, and sends it to the LLM.
  2. Temporarily disable audio support in `global.json` and verify the bot falls back to native QQ ASR.
  3. Clear the native ASR metadata field and verify the bot falls back to local `faster-whisper` transcription.
* Use the `read_file` tool to read a paged audio/video file and verify the tags are properly rendered without raising API payload exceptions.
