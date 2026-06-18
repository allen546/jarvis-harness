"""Heartbeat system — periodic autonomous checks from HEARTBEAT.md."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class HeartbeatManager:
    """Reads HEARTBEAT.md and submits tasks to an agent session on an interval."""

    def __init__(
        self,
        workspace: str = ".",
        interval_secs: int = 300,
        submit_fn: Callable[[str, str], Awaitable[str]] | None = None,
        storage_dir: str = "storage/heartbeat",
    ) -> None:
        self.workspace = Path(workspace)
        self.interval_secs = interval_secs
        self.submit_fn = submit_fn
        self.storage_dir = Path(storage_dir)
        self._running = False

    def _read_tasks(self) -> list[str]:
        """Parse HEARTBEAT.md — one task per non-empty, non-comment line."""
        path = self.workspace / "HEARTBEAT.md"
        if not path.exists():
            return []
        tasks: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                tasks.append(stripped)
        return tasks

    async def tick(self) -> None:
        """Execute one heartbeat cycle."""
        tasks = self._read_tasks()
        if not tasks:
            return
        if self.submit_fn is None:
            logger.warning("heartbeat: no submit_fn configured, skipping")
            return

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        results: list[dict[str, Any]] = []

        for task in tasks:
            try:
                result = await self.submit_fn("heartbeat", task)
                results.append({"task": task, "status": "ok", "result": result[:500]})
            except Exception as exc:
                results.append({"task": task, "status": "error", "error": str(exc)})

        # Persist results
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        date_str = time.strftime("%Y-%m-%d")
        log_path = self.storage_dir / f"{date_str}.jsonl"
        entry = {"timestamp": timestamp, "results": results}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def run(self) -> None:
        """Run the heartbeat loop until cancelled."""
        self._running = True
        try:
            while self._running:
                await self.tick()
                await asyncio.sleep(self.interval_secs)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
