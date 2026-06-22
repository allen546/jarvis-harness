import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from jarvis.models.base import Message, Attachment
import asyncio

@pytest.mark.asyncio
async def test_qq_voice_fallback_tier1_success(monkeypatch):
    # Mock SessionManager and its session objects
    from jarvis.sessions import SessionManager
    from jarvis.runtime import AgentSession
    
    mock_session = MagicMock()
    mock_session.ctx.session.history = []
    
    manager = MagicMock(spec=SessionManager)
    manager.get_or_create.return_value = mock_session
    
    # Tier 1 returns successfully
    expected_reply = Message(role="assistant", content="Tier 1 reply")
    manager.submit_and_collect = AsyncMock(return_value=expected_reply)
    
    # Import the local _qq_handler from main.py via patch/injecting manager
    with patch("main._manager", manager):
        from main import main
        pass

@pytest.mark.asyncio
async def test_qq_voice_fallback_tiers(monkeypatch):
    from jarvis.sessions import SessionManager
    
    mock_history = []
    
    mock_session = MagicMock()
    mock_session.ctx.session.history = mock_history
    
    manager = MagicMock(spec=SessionManager)
    manager.get_or_create.return_value = mock_session
    
    captured_handler = None
    class MockQQChannel:
        def __init__(self, **kwargs):
            nonlocal captured_handler
            captured_handler = kwargs.get("on_message")
        async def run(self):
            pass
            
    monkeypatch.setattr("main.QQChannel", MockQQChannel)
    
    # Mock config
    mock_config = MagicMock()
    mock_config.channels.qq.enabled = True
    mock_config.channels.qq.app_id = "test_app_id"
    mock_config.channels.qq.app_secret = "test_secret"
    mock_config.channels.qq.intents = None
    mock_config.channels.qq.allowed_senders = []
    mock_config.channels.qq.supported_media = []
    mock_config.channels.qq.max_download_size_mb = 10
    mock_config.cron = MagicMock()
    mock_config.cron.tasks = []
    mock_config.harness.heartbeat.enabled = False
    mock_config.gateway.host = "127.0.0.1"
    mock_config.gateway.port = 9999
    
    monkeypatch.setattr("main.load_config", lambda: mock_config)
    
    # Mock _run_gateway to do nothing
    monkeypatch.setattr("main._run_gateway", lambda host, port: None)
    
    # Mock SessionManager
    manager = MagicMock(spec=SessionManager)
    manager.get_or_create.return_value = mock_session
    monkeypatch.setattr("main.SessionManager", lambda **kwargs: manager)
    
    # Mock CronScheduler run
    class MockCronScheduler:
        def register(self, task):
            pass
        async def run(self):
            pass
        def stop(self):
            pass
    monkeypatch.setattr("main.CronScheduler", MockCronScheduler)
    
    # Start main() as a task and let it run to initialize the channel, then cancel/stop it
    from main import main
    main_task = asyncio.create_task(main())
    await asyncio.sleep(0.1) # yield to let main run
    
    try:
        assert captured_handler is not None
        
        # Now test captured_handler (which is _qq_handler)
        
        # 1. Non-voice message passes through
        text_message = Message(role="user", content="hello text")
        manager.submit_and_collect.return_value = Message(role="assistant", content="hello reply")
        reply = await captured_handler("session123", text_message)
        assert reply.content == "hello reply"
        manager.submit_and_collect.assert_called_with("session123", text_message)
        
        # Prepare voice message
        voice_att = Attachment(mime_type="voice", url="data:audio/mpeg;base64,bW9jay1tcDM=")
        voice_message = Message(role="user", content="", attachments=[voice_att], metadata={"asr_text": "platform ASR text"})
        
        # Case A: Tier 1 succeeds
        manager.submit_and_collect.reset_mock()
        manager.submit_and_collect.return_value = Message(role="assistant", content="Tier 1 reply")
        reply = await captured_handler("session123", voice_message)
        assert reply.content == "Tier 1 reply"
        manager.submit_and_collect.assert_called_once_with("session123", voice_message)
        
        # Case B: Tier 1 fails, Tier 2 succeeds
        manager.submit_and_collect.reset_mock()
        
        # Simulate history appending during submission
        def submit_side_effect(session_id, msg):
            if msg is voice_message:
                mock_history.append(msg)
                raise RuntimeError("Tier 1 error")
            elif msg.content == "platform ASR text":
                mock_history.append(msg)
                return Message(role="assistant", content="Tier 2 reply")
            raise RuntimeError("unexpected message")
            
        manager.submit_and_collect.side_effect = submit_side_effect
        mock_history.clear()
        
        reply = await captured_handler("session123", voice_message)
        assert reply.content == "Tier 2 reply"
        # Check history was cleaned up: voice_message popped, so only tier 2 message or nothing is left
        assert voice_message not in mock_history
        
        # Case C: Tier 1 and Tier 2 fail, Tier 3 succeeds
        manager.submit_and_collect.reset_mock()
        mock_history.clear()
        
        def transcribe_mock(mp3_bytes):
            assert mp3_bytes == b"mock-mp3"
            return "whisper transcribed text"
            
        monkeypatch.setattr("jarvis.media.transcribe_locally", transcribe_mock)
        
        call_count = 0
        def submit_side_effect_t3(session_id, msg):
            nonlocal call_count
            call_count += 1
            mock_history.append(msg)
            if msg is voice_message:
                raise RuntimeError("Tier 1 error")
            elif msg.content == "platform ASR text":
                raise RuntimeError("Tier 2 error")
            elif msg.content == "whisper transcribed text":
                return Message(role="assistant", content="Tier 3 reply")
            raise RuntimeError("unexpected message")
            
        manager.submit_and_collect.side_effect = submit_side_effect_t3
        
        # Create a new voice message with different base64 bytes for test
        import base64
        url = "data:audio/mpeg;base64," + base64.b64encode(b"mock-mp3").decode("ascii")
        voice_message_t3 = Message(
            role="user",
            content="",
            attachments=[Attachment(mime_type="voice", url=url)],
            metadata={"asr_text": "platform ASR text"}
        )
        
        reply = await captured_handler("session123", voice_message_t3)
        assert reply.content == "Tier 3 reply"
        assert voice_message_t3 not in mock_history
        assert not any(m.content == "platform ASR text" for m in mock_history)
        
        # Case D: All fail
        manager.submit_and_collect.reset_mock()
        mock_history.clear()
        
        def submit_side_effect_all_fail(session_id, msg):
            mock_history.append(msg)
            raise RuntimeError("always error")
            
        manager.submit_and_collect.side_effect = submit_side_effect_all_fail
        
        with pytest.raises(RuntimeError, match="QQ voice message turn failed across all tiers"):
            await captured_handler("session123", voice_message_t3)
            
        assert not mock_history
        
    finally:
        main_task.cancel()
        try:
            await main_task
        except asyncio.CancelledError:
            pass
