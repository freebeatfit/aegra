"""In-memory broker for dev mode: each thread's event log lives in this process."""

import asyncio
import contextlib
import time
from collections import deque
from typing import Any

import structlog

from aegra_api.core.active_runs import active_runs
from aegra_api.models.enums import RunCancellationAction
from aegra_api.services.base_broker import (
    READ_BATCH,
    ROOT_NAMESPACE,
    SNAPSHOT_MARKER,
    BaseBrokerManager,
    Snapshot,
    ThreadEntry,
    entry_key,
)
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Sweep often enough that an expired log is freed close to its deadline.
_SWEEP_INTERVAL_SECONDS = 60
# A read waits this long for a write, like the Redis backend's XREAD BLOCK.
_READ_WAIT_SECONDS = 5.0


class _ThreadLog:
    def __init__(self) -> None:
        self.entries: deque[ThreadEntry] = deque()
        self.snapshots: dict[str, Snapshot] = {}
        self.last_key = (0, 0)
        self.last_write = time.monotonic()
        # Replaced on every write, so a reader waits on the one current when it looked.
        self.written = asyncio.Event()

    def after(self, key: tuple[int, int]) -> list[ThreadEntry]:
        fresh: list[ThreadEntry] = []
        for entry in reversed(self.entries):
            if entry_key(entry.entry_id) <= key:
                break
            fresh.append(entry)
        fresh.reverse()
        return fresh[:READ_BATCH]

    def write(self, run_id: str, write_id: str, event: Any) -> str:
        """Trim like the Redis backend's ``XADD MINID``, append, and wake readers. Returns the entry id."""
        ms = time.time_ns() // 1_000_000
        oldest = ms - settings.event_streaming.RESUMABLE_STREAM_TTL_SECONDS * 1000
        if root := self.snapshots.get(ROOT_NAMESPACE):
            oldest = min(oldest, entry_key(root.marker_id)[0])
        while self.entries and entry_key(self.entries[0].entry_id)[0] < oldest:
            self.entries.popleft()
        if self.entries:
            first = entry_key(self.entries[0].entry_id)
            self.snapshots = {
                namespace: snapshot
                for namespace, snapshot in self.snapshots.items()
                if namespace == ROOT_NAMESPACE or entry_key(snapshot.marker_id) >= first
            }
        last_ms, last_n = self.last_key
        self.last_key = (ms, 0) if ms > last_ms else (last_ms, last_n + 1)
        entry_id = f"{self.last_key[0]}-{self.last_key[1]}"
        self.entries.append(ThreadEntry(entry_id, run_id, write_id, event))
        self.last_write = time.monotonic()
        written, self.written = self.written, asyncio.Event()
        written.set()
        return entry_id


class BrokerManager(BaseBrokerManager):
    """Thread logs in memory with periodic cleanup, and local-task cancel."""

    def __init__(self) -> None:
        super().__init__()
        self._logs: dict[str, _ThreadLog] = {}
        self._cleanup_task: asyncio.Task[None] | None = None

    def _log(self, thread_id: str) -> _ThreadLog:
        log = self._logs.get(thread_id)
        if log is None:
            log = self._logs[thread_id] = _ThreadLog()
        return log

    async def append(self, thread_id: str, run_id: str, write_id: str, event: Any) -> bool:
        self._log(thread_id).write(run_id, write_id, event)
        return True

    async def put_snapshot(
        self, thread_id: str, run_id: str, write_id: str, namespace: str, record: dict[str, Any]
    ) -> bool:
        log = self._log(thread_id)
        log.snapshots[namespace] = Snapshot(log.write(run_id, write_id, (SNAPSHOT_MARKER, namespace)), record)
        return True

    async def read_thread(self, thread_id: str, after: str) -> list[ThreadEntry]:
        key = entry_key(after)
        log = self._log(thread_id)
        written = log.written
        if entries := log.after(key):
            return entries
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(written.wait(), _READ_WAIT_SECONDS)
        return self._log(thread_id).after(key)

    async def read_snapshot_ids(self, thread_id: str) -> dict[str, str]:
        return {namespace: snapshot.marker_id for namespace, snapshot in self._log(thread_id).snapshots.items()}

    async def read_snapshots(self, thread_id: str, namespaces: list[str]) -> dict[str, Snapshot]:
        snapshots = self._log(thread_id).snapshots
        return {namespace: snapshots[namespace] for namespace in namespaces if namespace in snapshots}

    async def start(self) -> None:
        """Start the periodic cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_old_logs())

    async def stop(self) -> None:
        """Stop the periodic cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

    async def request_cancel(
        self,
        run_id: str,
        action: RunCancellationAction = "cancel",
        *,
        emit_end_event: bool = True,
    ) -> None:
        """Cancel a run locally by cancelling its asyncio task."""
        task = active_runs.get(run_id)
        if task is None or task.done():
            return

        logger.info(f"Executing {action} for run {run_id} (local task)")
        task.cancel()
        if emit_end_event:
            await self.end_cancelled_run(run_id)

    def sweep_expired_logs(self, retention_seconds: float | None = None) -> list[str]:
        """Drop thread logs not written to within the replay window. Returns their thread ids."""
        if retention_seconds is None:
            retention_seconds = settings.event_streaming.RESUMABLE_STREAM_TTL_SECONDS
        now = time.monotonic()
        expired = [thread_id for thread_id, log in self._logs.items() if now - log.last_write > retention_seconds]
        for thread_id in expired:
            del self._logs[thread_id]
        return expired

    async def _cleanup_old_logs(self) -> None:
        while True:
            try:
                await asyncio.sleep(_SWEEP_INTERVAL_SECONDS)
                self.sweep_expired_logs()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in broker cleanup task")


def _create_broker_manager() -> BaseBrokerManager:
    """Select broker backend based on settings.

    Returns RedisBrokerManager when REDIS_BROKER_ENABLED=true,
    otherwise the default in-memory BrokerManager.
    """
    if settings.redis.REDIS_BROKER_ENABLED:
        # Conditional import: redis is only required when the Redis broker is enabled
        from aegra_api.services.redis_broker import RedisBrokerManager

        logger.info("Using Redis broker for SSE streaming")
        return RedisBrokerManager()
    logger.info("Using in-memory broker for SSE streaming")
    return BrokerManager()


# Global broker manager instance
broker_manager: BaseBrokerManager = _create_broker_manager()
