"""Tests for RepeatedLoopHook and RepeatedContentHook."""

import pytest
from jarvis.hooks import (
    RepeatedLoopHook,
    RepeatedContentHook,
    _jaccard_ngrams,
    _char_ngrams,
)
from jarvis.models.base import Message, ToolCall
from jarvis.tools import ToolResult


class FakeSession:
    def __init__(self, sid: str = "safety") -> None:
        self.id = sid
        self.history: list[Message] = []


class FakeCtx:
    def __init__(self, session: FakeSession | None = None) -> None:
        self.session = session or FakeSession()


# ── RepeatedLoopHook ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_repeated_loop_detects_identical_calls():
    hook = RepeatedLoopHook(max_repeats=3)
    ctx = FakeCtx()
    tc = ToolCall(call_id="1", tool_name="ls", arguments={"path": "."})

    # First two calls: no stop
    r1 = await hook.after_tool(ctx, tc, ToolResult("1", "ls", "ok"))
    assert not r1.stop
    r2 = await hook.after_tool(ctx, tc, ToolResult("2", "ls", "ok"))
    assert not r2.stop
    r3 = await hook.after_tool(ctx, tc, ToolResult("3", "ls", "ok"))
    assert r3.stop
    assert "Repeated tool loop" in r3.reason


@pytest.mark.asyncio
async def test_repeated_loop_resets_on_different_tool():
    hook = RepeatedLoopHook(max_repeats=3)
    ctx = FakeCtx()
    tc1 = ToolCall(call_id="1", tool_name="ls", arguments={"path": "."})
    tc2 = ToolCall(call_id="2", tool_name="grep", arguments={"query": "foo"})

    await hook.after_tool(ctx, tc1, ToolResult("1", "ls", "ok"))
    await hook.after_tool(ctx, tc1, ToolResult("2", "ls", "ok"))
    # Different tool resets counter
    r = await hook.after_tool(ctx, tc2, ToolResult("3", "grep", "ok"))
    assert not r.stop
    # Same tool again starts fresh
    r2 = await hook.after_tool(ctx, tc1, ToolResult("4", "ls", "ok"))
    assert not r2.stop


@pytest.mark.asyncio
async def test_repeated_loop_resets_on_different_args():
    hook = RepeatedLoopHook(max_repeats=3)
    ctx = FakeCtx()
    tc1 = ToolCall(call_id="1", tool_name="ls", arguments={"path": "."})
    tc2 = ToolCall(call_id="2", tool_name="ls", arguments={"path": "/tmp"})

    await hook.after_tool(ctx, tc1, ToolResult("1", "ls", "ok"))
    await hook.after_tool(ctx, tc1, ToolResult("2", "ls", "ok"))
    r = await hook.after_tool(ctx, tc2, ToolResult("3", "ls", "ok"))
    assert not r.stop


@pytest.mark.asyncio
async def test_repeated_loop_resets_after_turn():
    hook = RepeatedLoopHook(max_repeats=3)
    ctx = FakeCtx()
    tc = ToolCall(call_id="1", tool_name="ls", arguments={"path": "."})

    await hook.after_tool(ctx, tc, ToolResult("1", "ls", "ok"))
    await hook.after_tool(ctx, tc, ToolResult("2", "ls", "ok"))
    await hook.after_turn(ctx, Message(role="assistant", content="done"))
    # Counter should be reset
    r = await hook.after_tool(ctx, tc, ToolResult("3", "ls", "ok"))
    assert not r.stop


@pytest.mark.asyncio
async def test_repeated_loop_different_sessions_independent():
    hook = RepeatedLoopHook(max_repeats=3)
    ctx1 = FakeCtx(FakeSession("s1"))
    ctx2 = FakeCtx(FakeSession("s2"))
    tc = ToolCall(call_id="1", tool_name="ls", arguments={"path": "."})

    # 3 identical calls on s1 → stop
    await hook.after_tool(ctx1, tc, ToolResult("1", "ls", "ok"))
    await hook.after_tool(ctx1, tc, ToolResult("2", "ls", "ok"))
    r1 = await hook.after_tool(ctx1, tc, ToolResult("3", "ls", "ok"))
    assert r1.stop
    # s2 should not be affected
    r2 = await hook.after_tool(ctx2, tc, ToolResult("4", "ls", "ok"))
    assert not r2.stop


# ── RepeatedContentHook ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_repeated_content_detects_similar():
    hook = RepeatedContentHook(threshold=0.8, window=3)
    ctx = FakeCtx()

    msg1 = Message(role="assistant", content="The answer is 42. It is always 42.")
    msg2 = Message(role="assistant", content="The answer is 42. It is always 42.")
    msg3 = Message(role="assistant", content="The answer is 42. It is always 42.")

    r1 = await hook.after_turn(ctx, msg1)
    assert not r1.stop
    r2 = await hook.after_turn(ctx, msg2)
    assert r2.stop
    assert "Repeated content" in r2.reason


@pytest.mark.asyncio
async def test_repeated_content_allows_different():
    hook = RepeatedContentHook(threshold=0.8, window=3)
    ctx = FakeCtx()

    r1 = await hook.after_turn(ctx, Message(role="assistant", content="Hello, how can I help?"))
    assert not r1.stop
    r2 = await hook.after_turn(ctx, Message(role="assistant", content="Sure, let me calculate that for you."))
    assert not r2.stop
    r3 = await hook.after_turn(ctx, Message(role="assistant", content="The result is 137."))
    assert not r3.stop


@pytest.mark.asyncio
async def test_repeated_content_empty_ignored():
    hook = RepeatedContentHook(threshold=0.8, window=3)
    ctx = FakeCtx()

    r1 = await hook.after_turn(ctx, Message(role="assistant", content=""))
    assert not r1.stop
    r2 = await hook.after_turn(ctx, Message(role="assistant", content=""))
    assert not r2.stop

@pytest.mark.asyncio
async def test_repeated_content_short_messages_detected():
    hook = RepeatedContentHook(threshold=0.8, window=3)
    ctx = FakeCtx()

    r1 = await hook.after_turn(ctx, Message(role="assistant", content="y"))
    assert not r1.stop
    r2 = await hook.after_turn(ctx, Message(role="assistant", content="y"))
    assert r2.stop
    assert "Repeated content" in r2.reason


@pytest.mark.asyncio
async def test_repeated_content_short_different_not_triggered():
    hook = RepeatedContentHook(threshold=0.8, window=3)
    ctx = FakeCtx()

    r1 = await hook.after_turn(ctx, Message(role="assistant", content="y"))
    assert not r1.stop
    r2 = await hook.after_turn(ctx, Message(role="assistant", content="n"))
    assert not r2.stop


@pytest.mark.asyncio
async def test_repeated_content_window_slides():
    hook = RepeatedContentHook(threshold=0.8, window=2)
    ctx = FakeCtx()

    msg_a = Message(role="assistant", content="AAAA BBBB CCCC DDDD")
    msg_b = Message(role="assistant", content="EEEE FFFF GGGG HHHH")
    msg_c = Message(role="assistant", content="AAAA BBBB CCCC DDDD")

    await hook.after_turn(ctx, msg_a)
    await hook.after_turn(ctx, msg_b)
    # msg_a is now out of the window (window=2, only msg_b is recent)
    r = await hook.after_turn(ctx, msg_c)
    assert not r.stop


# ── _jaccard_ngrams ─────────────────────────────────────────────────────────


def test_jaccard_identical():
    assert _jaccard_ngrams("hello world", "hello world") == 1.0


def test_jaccard_disjoint():
    assert _jaccard_ngrams("aaa", "bbb") == 0.0


def test_jaccard_partial():
    score = _jaccard_ngrams("hello", "jello")
    assert 0.0 < score < 1.0


def test_char_ngrams_empty():
    assert _char_ngrams("") == {}
