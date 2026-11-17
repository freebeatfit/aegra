"""Per-thread event log and run cancellation, shared by both broker backends.

Every event of every run on a thread goes to one log per thread, in producer
order (a Redis Stream when ``REDIS_BROKER_ENABLED=true``, a list in dev mode).
A thread stream reads the log from its cursor, the entry id, with a bounded
blocking read, so a run started after the stream opened shows up on it.

Full state is kept once per namespace, not in the log: every ``values`` event,
interrupts included, replaces the namespace's snapshot (with its checkpoint
envelope) and appends a small marker entry in the same atomic write. Each write
trims log entries older than ``RESUMABLE_STREAM_TTL_SECONDS``, but never the root
snapshot's marker or anything after it, and the thread's keys expire one window
after its last write.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

import structlog

from aegra_api.models.enums import RunCancellationAction
from aegra_api.observability.metrics import STREAMING_DATA_LOSS

logger = structlog.getLogger(__name__)

# Entries returned per read.
READ_BATCH = 200

# (entry_id, run_id, write_id, event). entry_id is "<ms>-<n>" and rises with
# every write on the thread; a retried write keeps its write_id. A marker's event
# is (SNAPSHOT_MARKER, namespace key): its snapshot lives in the thread's snapshots.
ThreadEntry = tuple[str, str, str, Any]
SNAPSHOT_MARKER = "snapshot"
# (marker entry id, record): the record is {"run_id", "checkpoint": envelope event or None, "values": event}.
Snapshot = tuple[str, dict[str, Any]]
ROOT_NAMESPACE = "[]"


def namespace_key(namespace: list[str]) -> str:
    return json.dumps(namespace, separators=(",", ":"))


def is_end_event(event: Any) -> bool:
    return isinstance(event, (tuple, list)) and len(event) >= 1 and event[0] == "end"


def entry_key(entry_id: str) -> tuple[int, int]:
    """Sort key of a ``<ms>-<n>`` entry id."""
    ms, _, n = entry_id.partition("-")
    return int(ms), int(n or 0)


def entry_before(entry_id: str) -> str:
    """The cursor a read starts from to include *entry_id*."""
    ms, n = entry_key(entry_id)
    return f"{ms}-{n - 1}" if n else f"{ms - 1}-{2**64 - 1}"


def _method(event: Any) -> Any:
    return event[0] if isinstance(event, (tuple, list)) and event else None


def _params(event: Any) -> dict[str, Any]:
    payload = event[1] if isinstance(event, (tuple, list)) and len(event) == 2 else None
    params = payload.get("params") if isinstance(payload, dict) else None
    return params if isinstance(params, dict) else {}


def _namespace(event: Any) -> list[str]:
    namespace = _params(event).get("namespace")
    return namespace if isinstance(namespace, list) else []


class RunBroker:
    """Writes one run's events to its thread's log and remembers that the run ended."""

    def __init__(self, manager: "BaseBrokerManager", run_id: str, thread_id: str) -> None:
        self._manager = manager
        self.run_id = run_id
        self.thread_id = thread_id
        self._finished = False
        # Keeps this run's writes in order when a signal races a retrying write.
        self._write_lock = asyncio.Lock()
        # A checkpoints event waits for the values it precedes, to be stored with it.
        self._held: Any = None
        # Each namespace's last stored envelope: values that restate a step (an interrupt) get no new one.
        self._envelopes: dict[str, Any] = {}
        # Set once a write failed after its retries: later events are dropped, not retried.
        self._lost = False

    async def put(self, event: Any) -> None:
        """Append *event* (a ``(method, payload)`` pair) unless the run already ended."""
        if self._finished:
            logger.warning("Dropped an event of an ended run", run_id=self.run_id)
            return
        if is_end_event(event):
            self._finished = True
        async with self._write_lock:
            held, self._held = self._held, None
            if _method(event) == "values" and held is not None and _namespace(held) == _namespace(event):
                self._envelopes[namespace_key(_namespace(event))] = held
            elif held is not None:
                await self._append(held)
            if _method(event) == "values":
                await self._snapshot(event)
            elif _method(event) == "checkpoints":
                self._held = event
            else:
                await self._append(event)

    async def _append(self, event: Any) -> None:
        if self._skip(terminal=_method(event) in ("end", "error")):
            return
        self._count(await self._manager.append(self.thread_id, self.run_id, uuid4().hex, event))

    async def _snapshot(self, values: Any) -> None:
        if self._skip(terminal=False):
            return
        namespace = namespace_key(_namespace(values))
        checkpoint = self._envelopes.get(namespace)
        record = {
            "run_id": self.run_id,
            "checkpoint": None if checkpoint is None else checkpoint[1],
            "values": values[1],
        }
        self._count(await self._manager.put_snapshot(self.thread_id, self.run_id, uuid4().hex, namespace, record))

    def _skip(self, *, terminal: bool) -> bool:
        """After a lost write, drop the run's later events (as the Agent Server does) but still try its terminal."""
        if self._lost and not terminal:
            STREAMING_DATA_LOSS.inc()
            return True
        return False

    def _count(self, written: bool) -> None:
        if written:
            return
        STREAMING_DATA_LOSS.inc()
        if not self._lost:
            logger.error(
                "Stopped publishing the run's events after a failed write; the run continues", run_id=self.run_id
            )
        self._lost = True

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
    async def append(self, thread_id: str, run_id: str, write_id: str, event: Any) -> bool:
        """Append one event to the thread's log. False when the write failed after bounded retries."""

    @abstractmethod
    async def put_snapshot(
        self, thread_id: str, run_id: str, write_id: str, namespace: str, record: dict[str, Any]
    ) -> bool:
        """Replace *namespace*'s snapshot with *record* and append its marker, in one atomic write.

        False when the write failed after bounded retries.
        """

    @abstractmethod
    async def read_thread(self, thread_id: str, after: str) -> list[ThreadEntry]:
        """Up to ``READ_BATCH`` entries after entry id *after*, waiting a bounded time for one.

        An empty list means none arrived in time; ``"0-0"`` reads from the oldest kept entry.
        """

    @abstractmethod
    async def read_snapshot_ids(self, thread_id: str) -> dict[str, str]:
        """The marker entry id of each namespace's snapshot."""

    @abstractmethod
    async def read_snapshots(self, thread_id: str, namespaces: list[str]) -> dict[str, Snapshot]:
        """The current snapshots of *namespaces* that exist."""

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
