"""Per-thread event log and run cancellation, shared by both broker backends.

Every event of every run on a thread goes to one log per thread, in producer
order (a Redis Stream when ``REDIS_BROKER_ENABLED=true``, a list in dev mode).
A thread stream reads the log from its cursor, the entry id, with a bounded
blocking read, so a run started after the stream opened shows up on it.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

import structlog

from aegra_api.models.enums import RunCancellationAction

logger = structlog.getLogger(__name__)

# A thread's log expires this long after its last write. Reconnects happen in
# seconds; this is the documented window in docs/guides/streaming.mdx.
REPLAY_RETENTION_SECONDS = 600
# Entries kept per thread log (approximate in Redis).
REPLAY_MAX_EVENTS = 10_000
# Entries returned per read.
READ_BATCH = 200

# (entry_id, run_id, write_id, event). entry_id is "<ms>-<n>" and rises with
# every write on the thread; a retried write keeps its write_id.
ThreadEntry = tuple[str, str, str, Any]


def is_end_event(event: Any) -> bool:
    return isinstance(event, (tuple, list)) and len(event) >= 1 and event[0] == "end"


def entry_key(entry_id: str) -> tuple[int, int]:
    """Sort key of a ``<ms>-<n>`` entry id."""
    ms, _, n = entry_id.partition("-")
    return int(ms), int(n or 0)


class RunBroker:
    """Writes one run's events to its thread's log and remembers that the run ended."""

    def __init__(self, manager: "BaseBrokerManager", run_id: str, thread_id: str) -> None:
        self._manager = manager
        self.run_id = run_id
        self.thread_id = thread_id
        self._finished = False
        # Keeps this run's writes in order when a signal races a retrying write.
        self._write_lock = asyncio.Lock()

    async def put(self, event: Any) -> None:
        """Append *event* (a ``(method, payload)`` pair) unless the run already ended."""
        if self._finished:
            logger.warning("Dropped an event of an ended run", run_id=self.run_id)
            return
        if is_end_event(event):
            self._finished = True
        async with self._write_lock:
            await self._manager.append(self.thread_id, self.run_id, uuid4().hex, event)

    def mark_finished(self) -> None:
        self._finished = True

    def is_finished(self) -> bool:
        return self._finished


class BaseBrokerManager(ABC):
    """Run writers, the thread logs they write to, and cross-instance cancel."""

    def __init__(self) -> None:
        self._brokers: dict[str, RunBroker] = {}

    def get_or_create_broker(self, run_id: str, thread_id: str) -> RunBroker:
        broker = self._brokers.get(run_id)
        if broker is None:
            broker = self._brokers[run_id] = RunBroker(self, run_id, thread_id)
        return broker

    def get_broker(self, run_id: str) -> RunBroker | None:
        return self._brokers.get(run_id)

    def cleanup_broker(self, run_id: str) -> None:
        broker = self._brokers.pop(run_id, None)
        if broker is not None:
            broker.mark_finished()

    async def end_cancelled_run(self, run_id: str) -> None:
        """Write the interrupted end of a run cancelled on this instance, once."""
        broker = self.get_broker(run_id)
        if broker is not None and not broker.is_finished():
            # Stays registered: execute_run's cancel path then sees the end and skips a second one.
            await broker.put(("end", {"status": "interrupted"}))

    @abstractmethod
    async def append(self, thread_id: str, run_id: str, write_id: str, event: Any) -> None:
        """Append one event to the thread's log. A write that keeps failing is logged, not raised."""

    @abstractmethod
    async def read_thread(self, thread_id: str, after: str) -> list[ThreadEntry]:
        """Up to ``READ_BATCH`` entries after entry id *after*, waiting a bounded time for one.

        An empty list means none arrived in time; ``"0-0"`` reads from the oldest kept entry.
        """

    @abstractmethod
    async def start(self) -> None:
        """Start background tasks (cleanup loop, cancel listener, etc.)"""

    @abstractmethod
    async def stop(self) -> None:
        """Stop background tasks and release resources"""

    @abstractmethod
    async def request_cancel(
        self,
        run_id: str,
        action: RunCancellationAction = "cancel",
        *,
        emit_end_event: bool = True,
    ) -> None:
        """Request cancellation of a run.

        In single-instance mode, cancels the local asyncio task directly.
        In multi-instance mode, broadcasts via Redis pub/sub so the owning
        instance can act on it. ``emit_end_event`` is disabled when the API
        has already reconciled the database and owns terminal signaling.
        """
