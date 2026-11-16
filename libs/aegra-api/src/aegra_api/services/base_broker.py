"""Abstract base classes for the broker system"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any

from aegra_api.models.enums import RunCancellationAction

# How long a finished run's events stay replayable, both backends. Reconnects
# happen in seconds; this is the documented window in docs/guides/streaming.mdx.
REPLAY_RETENTION_SECONDS = 600


class BaseRunBroker(ABC):
    """Abstract base class for a run-specific event broker.

    Handles both live event broadcast and replay storage for SSE reconnection.
    """

    @abstractmethod
    async def put(self, event_id: str, payload: Any, *, resumable: bool = True) -> None:
        """Publish an event to live subscribers and optionally store for replay.

        Args:
            event_id: Unique event identifier (format: {run_id}_event_{seq}).
            payload: The event data (typically a tuple like ("values", {...})).
            resumable: If True, store the event for replay on reconnect.
        """

    @abstractmethod
    def aiter(self) -> AsyncIterator[tuple[str, Any]]:
        """Async iterator yielding (event_id, payload) pairs."""
        ...

    @abstractmethod
    async def replay(self, last_event_id: str | None) -> list[tuple[str, Any]]:
        """Return stored events for replay on reconnect.

        Args:
            last_event_id: If provided, return events after this ID.
                           If None, return all stored events.

        Returns:
            List of (event_id, payload) tuples in order.
        """

    async def replay_iter(self, last_event_id: str | None) -> AsyncIterator[tuple[str, Any]]:
        """Like :meth:`replay`, but may yield in batches instead of loading all at once."""
        for item in await self.replay(last_event_id):
            yield item

    async def replay_settled(self, last_event_id: str | None, *, tail: int = 64) -> list[tuple[str, Any]]:
        """A settled run's compact replay: its last state and how it ended.

        The latest values snapshots plus the last ``tail`` stored events that
        :func:`~aegra_api.services.replay_policy.kept_when_settled` keeps.
        """
        from aegra_api.services.replay_policy import kept_when_settled, snapshot_slot

        replayed = await self.replay(last_event_id)
        kept = [item for item in replayed if snapshot_slot(item[1]) is not None or kept_when_settled(item[1])]
        snapshots = [item for item in kept if snapshot_slot(item[1]) is not None]
        rest = [item for item in kept if snapshot_slot(item[1]) is None][-tail:]
        from aegra_api.services.replay_policy import merge_replay

        return merge_replay(snapshots, rest)

    @abstractmethod
    def mark_finished(self) -> None:
        """Mark this broker as finished."""

    @abstractmethod
    def is_finished(self) -> bool:
        """Check if this broker is finished."""


class ThreadWaiter:
    """Wakes a thread stream when a run is created on its thread.

    Register before reading the thread's runs, ``clear()`` before each read and
    ``wait()`` after it: a notification that lands in between stays set, so
    no new run is missed.
    """

    def __init__(self, manager: "BaseBrokerManager", thread_id: str) -> None:
        self._manager = manager
        self.thread_id = thread_id
        self._event = asyncio.Event()

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    async def wait(self, timeout: float) -> bool:
        """True when notified, False when *timeout* elapsed first."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
        except TimeoutError:
            return False
        return True

    async def __aenter__(self) -> "ThreadWaiter":
        self._manager._thread_waiters.setdefault(self.thread_id, set()).add(self)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        waiters = self._manager._thread_waiters.get(self.thread_id)
        if waiters is not None:
            waiters.discard(self)
            if not waiters:
                self._manager._thread_waiters.pop(self.thread_id, None)


class BaseBrokerManager(ABC):
    """Abstract base class for managing multiple RunBroker instances"""

    @property
    def _thread_waiters(self) -> dict[str, set[ThreadWaiter]]:
        """thread_id -> waiters on this process, created on first use."""
        waiters: dict[str, set[ThreadWaiter]] | None = self.__dict__.get("_thread_waiter_map")
        if waiters is None:
            waiters = self.__dict__["_thread_waiter_map"] = {}
        return waiters

    def thread_waiter(self, thread_id: str) -> ThreadWaiter:
        """A waiter woken when a run is created on *thread_id* (use as ``async with``)."""
        return ThreadWaiter(self, thread_id)

    def _wake_thread(self, thread_id: str) -> None:
        for waiter in list(self._thread_waiters.get(thread_id, ())):
            waiter.set()

    async def notify_thread(self, thread_id: str) -> None:
        """Tell the thread's stream sessions that a run was created.

        Single-process default; the Redis manager broadcasts to every instance.
        """
        self._wake_thread(thread_id)

    @abstractmethod
    def get_or_create_broker(self, run_id: str) -> BaseRunBroker:
        """Get or create a broker for a run"""

    @abstractmethod
    def get_broker(self, run_id: str) -> BaseRunBroker | None:
        """Get an existing broker or None"""

    @abstractmethod
    def cleanup_broker(self, run_id: str) -> None:
        """Clean up a broker for a run"""

    @abstractmethod
    def remove_broker(self, run_id: str) -> None:
        """Remove a broker completely"""

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

    @abstractmethod
    async def get_event_sequence(self, run_id: str) -> int:
        """Return the current (highest) event sequence number for a run.

        Used by signal methods (cancel, error) to generate the next event ID
        when they need to inject events into a run's stream.
        """

    async def allocate_event_id(self, run_id: str) -> str:
        """Atomically allocate the next event sequence number and return event_id.

        Default implementation uses get_event_sequence + 1 (not atomic).
        Redis backend overrides with atomic INCR.
        """
        from aegra_api.utils import generate_event_id

        counter = await self.get_event_sequence(run_id) + 1
        return generate_event_id(run_id, counter)
