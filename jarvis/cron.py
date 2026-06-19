from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Awaitable

from croniter import croniter

logger = logging.getLogger(__name__)


@dataclass
class CronTask:
    name: str
    schedule: str  # cron expression
    handler: Callable[[], Awaitable[str]]
    enabled: bool = True
    last_run: float | None = None
    next_run: float | None = None
    last_result: str | None = None
    consecutive_errors: int = 0

    def __post_init__(self) -> None:
        if self.next_run is None:
            self._compute_next_run()

    def _compute_next_run(self, base: datetime | None = None) -> None:
        base = base or datetime.now()
        cron = croniter(self.schedule, base)
        self.next_run = cron.get_next(float)

    def should_run(self) -> bool:
        if not self.enabled or self.next_run is None:
            return False
        return time.time() >= self.next_run

    def mark_run(self, result: str | None = None) -> None:
        self.last_run = time.time()
        self.last_result = result
        self.consecutive_errors = 0
        self._compute_next_run()

    def mark_error(self, error: str) -> None:
        self.last_run = time.time()
        self.last_result = f"ERROR: {error}"
        self.consecutive_errors += 1
        self._compute_next_run()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schedule": self.schedule,
            "enabled": self.enabled,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "last_result": self.last_result,
            "consecutive_errors": self.consecutive_errors,
        }




class CronScheduler:
    def __init__(self, state_file: str = "storage/cron_state.json") -> None:
        self.tasks: dict[str, CronTask] = {}
        self.state_file = Path(state_file)
        self._running = False
        self._load_state()

    def register(self, task: CronTask) -> None:
        """Register a task, restoring state if available."""
        saved = self.tasks.get(task.name)
        if saved:
            task.last_run = saved.last_run
            task.next_run = saved.next_run
            task.consecutive_errors = saved.consecutive_errors
        self.tasks[task.name] = task
        logger.info("cron: registered %s [%s], next run at %s",
                     task.name, task.schedule,
                     datetime.fromtimestamp(task.next_run).isoformat() if task.next_run else "unknown")

    async def run_task(self, task: CronTask) -> str:
        """Execute a single task and return its result."""
        logger.info("cron: running %s", task.name)
        try:
            result = await task.handler()
            task.mark_run(result)
            logger.info("cron: %s completed: %s", task.name, result[:100] if result else "ok")
            return result
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            task.mark_error(error_msg)
            logger.error("cron: %s failed: %s", task.name, error_msg)
            return error_msg

    async def tick(self) -> list[str]:
        """Run all due tasks. Returns list of results."""
        results: list[str] = []
        for task in self.tasks.values():
            if task.should_run():
                result = await self.run_task(task)
                results.append(f"{task.name}: {result}")
        if results:
            self._save_state()
        return results

    async def run(self, tick_interval: float = 10.0) -> None:
        """Main loop — ticks every `tick_interval` seconds."""
        self._running = True
        logger.info("cron: scheduler started with %d tasks, tick interval %.1fs",
                     len(self.tasks), tick_interval)
        try:
            while self._running:
                await self.tick()
                await asyncio.sleep(tick_interval)
        except asyncio.CancelledError:
            logger.info("cron: scheduler cancelled")
        finally:
            self._running = False
            self._save_state()

    def stop(self) -> None:
        self._running = False

    def _save_state(self) -> None:
        state = {name: task.to_dict() for name, task in self.tasks.items()}
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text())
            # State is loaded but tasks aren't created here —
            # they get registered via register() which restores state.
            # We store the raw dicts so register() can pick them up.
            self._saved_state = data
        except Exception:
            self._saved_state = {}

    def get_saved_state(self, task_name: str) -> dict[str, Any] | None:
        return getattr(self, "_saved_state", {}).get(task_name)
def create_agent_task(
    name: str,
    schedule: str,
    prompt: str,
    session_id: str = "cron",
    manager: Any | None = None,
) -> CronTask:
    """Create a CronTask that submits a prompt to an agent session."""
    if manager is None:
        from jarvis.sessions import SessionManager
        manager = SessionManager()

    async def handler() -> str:
        from jarvis.models.base import Message
        result = await manager.submit_and_collect(session_id, Message(role="user", content=prompt))
        return result.content or ""

    return CronTask(name=name, schedule=schedule, handler=handler)

def tasks_from_config(config_cron: Any, manager: Any | None = None) -> list[CronTask]:
    """Build CronTask list from CronConfig."""
    tasks: list[CronTask] = []
    for task_cfg in config_cron.tasks:
        if not task_cfg.enabled:
            continue
        tasks.append(create_agent_task(
            name=task_cfg.name,
            schedule=task_cfg.schedule,
            prompt=task_cfg.prompt,
            session_id=task_cfg.session_id,
            manager=manager,
        ))
    return tasks
