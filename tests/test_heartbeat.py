"""Tests for HeartbeatManager."""

import pytest
import json
import asyncio
from pathlib import Path
from jarvis.heartbeat import HeartbeatManager


def test_read_tasks_empty(tmp_path: Path):
    mgr = HeartbeatManager(workspace=str(tmp_path))
    assert mgr._read_tasks() == []


def test_read_tasks_with_file(tmp_path: Path):
    (tmp_path / "HEARTBEAT.md").write_text(
        "# Comment\nCheck email\nReview calendar\n\n# Another comment\nRun git status\n"
    )
    mgr = HeartbeatManager(workspace=str(tmp_path))
    tasks = mgr._read_tasks()
    assert tasks == ["Check email", "Review calendar", "Run git status"]


def test_read_tasks_all_comments(tmp_path: Path):
    (tmp_path / "HEARTBEAT.md").write_text("# only comments\n# nothing else\n")
    mgr = HeartbeatManager(workspace=str(tmp_path))
    assert mgr._read_tasks() == []


@pytest.mark.asyncio
async def test_tick_executes_tasks(tmp_path: Path):
    (tmp_path / "HEARTBEAT.md").write_text("Task A\nTask B\n")
    executed: list[str] = []

    async def fake_submit(session_id: str, text: str) -> str:
        executed.append(text)
        return f"done: {text}"

    mgr = HeartbeatManager(
        workspace=str(tmp_path),
        submit_fn=fake_submit,
        storage_dir=str(tmp_path / "hb"),
    )
    await mgr.tick()
    assert executed == ["Task A", "Task B"]
    # Check results persisted
    log_files = list((tmp_path / "hb").glob("*.jsonl"))
    assert len(log_files) == 1
    entry = json.loads(log_files[0].read_text().strip())
    assert len(entry["results"]) == 2
    assert entry["results"][0]["status"] == "ok"


@pytest.mark.asyncio
async def test_tick_no_tasks_noop(tmp_path: Path):
    executed: list[str] = []

    async def fake_submit(session_id: str, text: str) -> str:
        executed.append(text)
        return "ok"

    mgr = HeartbeatManager(
        workspace=str(tmp_path),
        submit_fn=fake_submit,
        storage_dir=str(tmp_path / "hb"),
    )
    await mgr.tick()
    assert executed == []


@pytest.mark.asyncio
async def test_tick_error_handling(tmp_path: Path):
    (tmp_path / "HEARTBEAT.md").write_text("failing task\n")

    async def bad_submit(session_id: str, text: str) -> str:
        raise RuntimeError("boom")

    mgr = HeartbeatManager(
        workspace=str(tmp_path),
        submit_fn=bad_submit,
        storage_dir=str(tmp_path / "hb"),
    )
    await mgr.tick()
    log_files = list((tmp_path / "hb").glob("*.jsonl"))
    entry = json.loads(log_files[0].read_text().strip())
    assert entry["results"][0]["status"] == "error"
    assert "boom" in entry["results"][0]["error"]


@pytest.mark.asyncio
async def test_tick_no_submit_fn(tmp_path: Path):
    (tmp_path / "HEARTBEAT.md").write_text("some task\n")
    mgr = HeartbeatManager(workspace=str(tmp_path))
    # Should not raise
    await mgr.tick()


@pytest.mark.asyncio
async def test_run_cancels_cleanly(tmp_path: Path):
    (tmp_path / "HEARTBEAT.md").write_text("task\n")

    async def fake_submit(session_id: str, text: str) -> str:
        return "ok"

    mgr = HeartbeatManager(
        workspace=str(tmp_path),
        interval_secs=1,
        submit_fn=fake_submit,
        storage_dir=str(tmp_path / "hb"),
    )
    task = asyncio.create_task(mgr.run())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert task.done()
