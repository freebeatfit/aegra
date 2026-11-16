"""Redis-backed broker for multi-instance SSE streaming.

Uses Redis pub/sub for live event broadcast and a Redis Stream plus a hash
for replay storage, both with a TTL so they expire without a cleanup loop.

The stream holds every resumable event except a v2 run's full-state
``values``, keyed ``<ordinal>-0`` by the event's per-run ordinal, so a
reconnect XRANGEs from its cursor in bounded batches instead of loading the
whole buffer. The hash keeps only the newest ``values`` per namespace
(``replay_policy.snapshot_slot``): the baseline a new subscriber needs.

Event sequence counters are stored as Redis INCR keys for O(1) cross-instance
access (instead of deriving from the replay buffer with O(N) LRANGE).
"""

import asyncio
import contextlib
import json
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import structlog
from redis import RedisError

from aegra_api.core.active_runs import active_runs, explicit_run_cancellations
from aegra_api.core.redis_manager import redis_manager
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.models.enums import RunCancellationAction
from aegra_api.services.base_broker import REPLAY_RETENTION_SECONDS, BaseBrokerManager, BaseRunBroker
from aegra_api.services.replay_policy import is_after, merge_replay, ordered_ordinal, snapshot_slot
from aegra_api.settings import settings
from aegra_api.utils import generate_event_id

logger = structlog.getLogger(__name__)

_serializer = GeneralSerializer()

# Every write refreshes the TTL, so the buffer expires this long after the last
# event, matching the in-memory post-completion window.
_REPLAY_TTL_SECONDS = REPLAY_RETENTION_SECONDS
# Max events in the replay stream (prevents unbounded growth)
_REPLAY_MAX_EVENTS = 10_000
# Events read per XRANGE call while replaying.
_REPLAY_BATCH = 200

# Reconnect backoff for Redis pub/sub listeners
_BACKOFF_BASE = 0.5
_BACKOFF_MAX = 30.0
_BACKOFF_FACTOR = 2.0

# Bounded retry for the write path (put). The read path (aiter) retries
# RedisError indefinitely; the write path must be bounded so a producer can't
# block forever, but it should still retry a transient blip rather than drop the
# event — a dropped event is a permanently lost SSE token.
_PUT_MAX_ATTEMPTS = 3


def _serialize_payload(payload: Any) -> str:
    """Serialize an event payload to a JSON string for Redis transport."""
    return json.dumps(payload, default=_serializer.serialize)


def _deserialize_payload(raw: Any) -> Any:
    """Convert JSON-deserialized data back to expected Python types.

    Event payloads are tuples like ("values", {...}). JSON has no tuple type,
    so they arrive as lists. We convert top-level lists with a string first
    element back to tuples.
    """
    if isinstance(raw, list) and len(raw) >= 1 and isinstance(raw[0], str):
        return tuple(raw)
    return raw


def _backoff_delay(attempt: int) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = min(_BACKOFF_BASE * (_BACKOFF_FACTOR**attempt), _BACKOFF_MAX)
    jitter = random.uniform(0, delay * 0.25)  # noqa: S311  # nosec B311
    return delay + jitter


class RedisRunBroker(BaseRunBroker):
    """Broker for a single run backed by Redis pub/sub + a Redis Stream.

    Producer: XADD to the stream (or HSET a values snapshot) + PUBLISH (live).
    Consumer: XRANGE from the cursor for replay, then SUBSCRIBE for live events.
    """

    def __init__(self, run_id: str, channel: str, cache_key: str, counter_key: str) -> None:
        self.run_id = run_id
        self._channel = channel
        self._cache_key = cache_key
        self._snapshot_key = f"{cache_key}:snapshots"
        self._counter_key = counter_key
        # Last stream ordinal this producer wrote, so an event whose id has no
        # usable ordinal still lands after it (``<last>-*``), not at the end.
        self._last_ordinal = 0
        self._finished = False
        # Serializes writes for this run so concurrent put() calls (e.g. a
        # cancel/error signal racing a stream write) keep cache-then-publish
        # ordering — without it, a backoff sleep on one put could let a later
        # event's publish overtake an earlier one.
        self._write_lock = asyncio.Lock()

    async def put(self, event_id: str, payload: Any, *, resumable: bool = True) -> None:
        """Append an event to the replay buffer and publish it to live subscribers.

        Writes are retried with bounded backoff (see ``_write_with_retry``), so
        delivery is **at-least-once**: in the rare case a write times out *after*
        Redis already applied it, a retry may re-deliver the event. ``event_id``
        is stable (allocated once upstream), so consumers de-duplicate on
        ``Last-Event-ID`` during replay.
        """
        if self._finished:
            logger.warning(f"Attempted to put event {event_id} into finished broker for run {self.run_id}")
            return

        message = json.dumps(
            {
                "event_id": event_id,
                "payload": json.loads(_serialize_payload(payload)),
            }
        )
        slot = snapshot_slot(payload)

        is_end = isinstance(payload, tuple) and len(payload) >= 1 and payload[0] == "end"

        async with self._write_lock:
            # Cache and publish are retried independently so a publish failure
            # never re-runs the cache pipeline (which would double-write the
            # replay buffer and double-increment the sequence counter).
            operation = "cache"
            try:
                if resumable:
                    if slot is not None:
                        await self._write_with_retry(lambda m: self._cache_snapshot(slot, m), message)
                    else:
                        await self._write_with_retry(lambda m: self._cache_event(event_id, m), message)

                operation = "publish"
                await self._write_with_retry(self._publish_event, message)

                if is_end:
                    self._finished = True
            except RedisError as e:
                logger.error(
                    f"Redis {operation} write failed for run {self.run_id} after {_PUT_MAX_ATTEMPTS} attempts: {e}"
                )
                # Even if Redis fails, mark finished for end events so aiter() can exit
                # rather than looping forever waiting for an end event that won't arrive.
                if is_end:
                    self._finished = True

    async def _cache_event(self, event_id: str, message: str) -> None:
        """Append the event to the replay stream under its ordinal.

        The entry id is ``<ordinal>-0``, so XRANGE can start right after a
        cursor. A retried write that Redis already applied is rejected as a
        non-increasing id, which is the duplicate we want to drop.
        """
        ordinal = ordered_ordinal(event_id)
        entry_id = f"{ordinal}-0" if ordinal is not None else f"{self._last_ordinal}-*"
        client = redis_manager.get_client()
        pipe = client.pipeline()
        pipe.xadd(self._cache_key, {"m": message}, id=entry_id, maxlen=_REPLAY_MAX_EVENTS, approximate=True)
        pipe.expire(self._cache_key, _REPLAY_TTL_SECONDS)
        pipe.expire(self._snapshot_key, _REPLAY_TTL_SECONDS)
        pipe.expire(self._counter_key, _REPLAY_TTL_SECONDS)
        try:
            await pipe.execute()
        except RedisError as e:
            if "equal or smaller" in str(e):
                logger.debug("Dropped a replayed stream write", run_id=self.run_id, event_id=event_id)
                return
            raise
        if ordinal is not None:
            self._last_ordinal = max(self._last_ordinal, ordinal)

    async def _cache_snapshot(self, slot: str, message: str) -> None:
        """Overwrite the namespace's latest full-state values snapshot."""
        client = redis_manager.get_client()
        pipe = client.pipeline()
        pipe.hset(self._snapshot_key, slot, message)
        pipe.expire(self._snapshot_key, _REPLAY_TTL_SECONDS)
        pipe.expire(self._cache_key, _REPLAY_TTL_SECONDS)
        pipe.expire(self._counter_key, _REPLAY_TTL_SECONDS)
        await pipe.execute()

    async def _publish_event(self, message: str) -> None:
        """Broadcast the event to live subscribers."""
        client = redis_manager.get_client()
        await client.publish(self._channel, message)

    async def _write_with_retry(self, op: Callable[[str], Awaitable[None]], message: str) -> None:
        """Run a Redis write with bounded exponential backoff.

        Mirrors the retry/backoff the read path (aiter) already applies, so a
        transient client-side blip (e.g. socket timeout) is retried instead of
        dropping the event. Bounded by ``_PUT_MAX_ATTEMPTS`` so a producer never
        blocks indefinitely; the final ``RedisError`` propagates to ``put()``.
        """
        attempt = 0
        while True:
            try:
                await op(message)
                return
            except RedisError as e:
                attempt += 1
                if attempt >= _PUT_MAX_ATTEMPTS:
                    raise
                delay = _backoff_delay(attempt)
                logger.warning(
                    f"Redis write failed for run {self.run_id}, retrying in {delay:.1f}s: {e}",
                    attempt=attempt,
                )
                await asyncio.sleep(delay)

    async def aiter(self) -> AsyncIterator[tuple[str, Any]]:
        attempt = 0
        while not self._finished:
            try:
                async for event_id, payload in self._subscribe_and_listen():
                    attempt = 0
                    yield event_id, payload
                # Clean exit from _subscribe_and_listen (end event or finished)
                break
            except RedisError as e:
                attempt += 1
                delay = _backoff_delay(attempt)
                logger.warning(
                    f"Redis pub/sub error for run {self.run_id}, retrying in {delay:.1f}s: {e}",
                    attempt=attempt,
                )
                await asyncio.sleep(delay)

    async def _subscribe_and_listen(self) -> AsyncIterator[tuple[str, Any]]:
        """Subscribe to the channel and yield events until end or disconnect."""
        client = redis_manager.get_client()
        pubsub = client.pubsub()
        await pubsub.subscribe(self._channel)

        # After subscribing, check if the run already ended (closes the race where
        # the end event was published before we subscribed on this instance).
        end_already_in_buffer = await self._check_end_in_buffer()

        last_yielded_event_id: str | None = None
        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=0.5,
                )
                if message is None:
                    if self._finished or end_already_in_buffer:
                        break
                    continue

                if message["type"] != "message":
                    continue

                data = json.loads(message["data"])
                event_id: str = data["event_id"]

                # Drop an at-least-once republish (put() retries a transient
                # publish failure). event_ids are unique per event and the write
                # lock keeps a retry adjacent, so skipping a repeat of the last
                # id de-duplicates without dropping a distinct event.
                if event_id == last_yielded_event_id:
                    continue
                last_yielded_event_id = event_id

                payload = _deserialize_payload(data["payload"])

                yield event_id, payload

                if isinstance(payload, tuple) and len(payload) >= 1 and payload[0] == "end":
                    self._finished = True
                    break
        finally:
            await pubsub.unsubscribe(self._channel)
            await pubsub.aclose()

    async def _check_end_in_buffer(self) -> bool:
        """Check if an 'end' event is already in the replay buffer.

        Used after subscribing to detect runs that finished before we subscribed,
        preventing infinite loops in cross-instance consumer scenarios.
        """
        try:
            client = redis_manager.get_client()
            entries = await client.xrevrange(self._cache_key, count=1)
            if entries:
                data = json.loads(entries[0][1]["m"])
                payload = _deserialize_payload(data["payload"])
                if isinstance(payload, tuple) and len(payload) >= 1 and payload[0] == "end":
                    self._finished = True
                    return True
        except RedisError as e:
            logger.warning(f"Failed checking replay buffer for end event for run {self.run_id}: {e}")
        return False

    async def replay(self, last_event_id: str | None) -> list[tuple[str, Any]]:
        """Events after *last_event_id*: the latest values snapshots, then the log.

        Snapshots older than the cursor are skipped, so a reconnect that has
        already seen the newest state does not get it again.
        """
        replayed: list[tuple[str, Any]] = []
        async for event_id, payload in self.replay_iter(last_event_id):
            replayed.append((event_id, payload))
        return replayed

    async def replay_iter(self, last_event_id: str | None) -> AsyncIterator[tuple[str, Any]]:
        """Stream the replay in XRANGE batches instead of loading it at once.

        The latest values snapshot of each namespace is merged in at its own
        position, so the consumer sees the same order the producer wrote.
        """
        try:
            client = redis_manager.get_client()
            raw_snapshots = await client.hvals(self._snapshot_key)  # type: ignore[invalid-await]
        except RedisError as e:
            logger.error(f"Redis replay failed for run {self.run_id}: {e}")
            return
        snapshots: list[tuple[str, Any]] = []
        for raw in raw_snapshots:
            data = json.loads(raw)
            if is_after(data["event_id"], last_event_id):
                snapshots.append((data["event_id"], _deserialize_payload(data["payload"])))
        snapshots.sort(key=lambda item: ordered_ordinal(item[0]) or 0)

        start = "-"
        if last_event_id is not None and (cursor := ordered_ordinal(last_event_id)) is not None:
            # Exclusive of the cursor and of any fallback-id entries written as <cursor>-N.
            start = f"({cursor}-18446744073709551615"
        prev_event_id: str | None = None
        while True:
            try:
                entries = await client.xrange(self._cache_key, min=start, max="+", count=_REPLAY_BATCH)
            except RedisError as e:
                logger.error(f"Redis replay failed for run {self.run_id}: {e}")
                return
            batch: list[tuple[str, Any]] = []
            for _entry_id, fields in entries:
                data = json.loads(fields["m"])
                event_id: str = data["event_id"]
                if event_id == prev_event_id:
                    continue
                prev_event_id = event_id
                batch.append((event_id, _deserialize_payload(data["payload"])))
            last_ordinal = next((o for e, _ in reversed(batch) if (o := ordered_ordinal(e)) is not None), None)
            due = [
                snap
                for snap in snapshots
                if last_ordinal is not None and (ordered_ordinal(snap[0]) or 0) < last_ordinal
            ]
            snapshots = snapshots[len(due) :]
            for event_id, payload in merge_replay(due, batch):
                yield event_id, payload
            if len(entries) < _REPLAY_BATCH:
                break
            start = f"({entries[-1][0]}"
        for event_id, payload in snapshots:
            yield event_id, payload

    def mark_finished(self) -> None:
        self._finished = True
        logger.debug(f"Redis broker for run {self.run_id} marked as finished")

    def is_finished(self) -> bool:
        return self._finished


class RedisBrokerManager(BaseBrokerManager):
    """Manages RedisRunBroker instances with cross-instance cancel support.

    The local _brokers dict is a cache for status checks. Redis is the source
    of truth for event data. Cancel commands are broadcast via a dedicated
    pub/sub channel so any instance can cancel a run.
    """

    def __init__(self) -> None:
        self._brokers: dict[str, RedisRunBroker] = {}
        self._channel_prefix = settings.redis.REDIS_CHANNEL_PREFIX
        self._cache_prefix = f"{self._channel_prefix}cache:"
        self._counter_prefix = f"{self._channel_prefix}counter:"
        self._cancel_channel = f"{self._channel_prefix}cancel"
        self._thread_channel = f"{self._channel_prefix}thread-runs"
        self._listener_task: asyncio.Task[None] | None = None
        self._running: bool = False

    def _make_broker(self, run_id: str) -> RedisRunBroker:
        """Create a RedisRunBroker for a run_id."""
        channel = f"{self._channel_prefix}{run_id}"
        cache_key = f"{self._cache_prefix}{run_id}"
        counter_key = f"{self._counter_prefix}{run_id}"
        return RedisRunBroker(run_id, channel, cache_key, counter_key)

    def get_or_create_broker(self, run_id: str) -> RedisRunBroker:
        if run_id not in self._brokers:
            self._brokers[run_id] = self._make_broker(run_id)
            logger.debug(f"Created Redis broker for run {run_id}")
        return self._brokers[run_id]

    def get_broker(self, run_id: str) -> RedisRunBroker | None:
        """Get an existing broker from the local cache, or None."""
        return self._brokers.get(run_id)

    def cleanup_broker(self, run_id: str) -> None:
        broker = self._brokers.pop(run_id, None)
        if broker:
            broker.mark_finished()
            logger.debug(f"Cleaned up Redis broker for run {run_id}")

    def remove_broker(self, run_id: str) -> None:
        broker = self._brokers.pop(run_id, None)
        if broker:
            broker.mark_finished()
            logger.debug(f"Removed Redis broker for run {run_id}")

    async def start(self) -> None:
        """Start the cancel command listener."""
        self._running = True
        self._listener_task = asyncio.create_task(self._listen_for_cancel_commands())
        logger.info("Redis broker manager started", cancel_channel=self._cancel_channel)

    async def stop(self) -> None:
        """Stop the cancel command listener."""
        self._running = False
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task
            self._listener_task = None
        logger.debug("Redis broker manager stopped")

    async def request_cancel(
        self,
        run_id: str,
        action: RunCancellationAction = "cancel",
        *,
        emit_end_event: bool = True,
    ) -> None:
        """Broadcast a cancel command via Redis pub/sub."""
        message = json.dumps(
            {
                "run_id": run_id,
                "action": action,
                "emit_end_event": emit_end_event,
            }
        )
        try:
            client = redis_manager.get_client()
            await client.publish(self._cancel_channel, message)
            logger.info(f"Published {action} command for run {run_id}")
        except RedisError as e:
            logger.error(f"Failed to publish {action} for run {run_id}: {e}")
            # Fall back to local execution - if the task is on this instance,
            # we can still cancel it even if Redis publish fails.
            await self._execute_cancel(run_id, emit_end_event=emit_end_event)

    async def notify_thread(self, thread_id: str) -> None:
        """Broadcast that a run was created on *thread_id* to every instance."""
        self._wake_thread(thread_id)
        try:
            client = redis_manager.get_client()
            await client.publish(self._thread_channel, thread_id)
        except RedisError as e:
            # Stream sessions still find the run on their periodic fallback read.
            logger.warning(f"Failed to publish thread run notice for {thread_id}: {e}")

    async def get_event_sequence(self, run_id: str) -> int:
        """Read the current event sequence counter from Redis (O(1) GET)."""
        try:
            client = redis_manager.get_client()
            value = await client.get(f"{self._counter_prefix}{run_id}")
            if value is not None:
                return int(value)
        except (RedisError, ValueError) as e:
            logger.warning(f"Failed to read event counter for run {run_id}: {e}")
        return 0

    async def allocate_event_id(self, run_id: str) -> str:
        """Atomically allocate the next event sequence number and return the event_id.

        Uses Redis INCR which is atomic — two concurrent callers will always
        get different sequence numbers. This prevents the race condition where
        get_event_sequence + 1 could return the same value to two callers.
        """
        counter_key = f"{self._counter_prefix}{run_id}"
        try:
            client = redis_manager.get_client()
            seq = await client.incr(counter_key)
            await client.expire(counter_key, _REPLAY_TTL_SECONDS)
            return generate_event_id(run_id, int(seq))
        except RedisError as e:
            logger.warning(f"Failed to allocate event_id for run {run_id}: {e}")
            # Fallback: use timestamp-based ID (unique but not sequential)
            return generate_event_id(run_id, int(time.time() * 1000))

    async def _listen_for_cancel_commands(self) -> None:
        """Background task: subscribe to cancel channel with reconnect backoff."""
        attempt = 0
        while self._running:
            try:
                await self._subscribe_and_handle_cancels()
                attempt = 0
            except asyncio.CancelledError:
                break
            except RedisError as e:
                attempt += 1
                delay = _backoff_delay(attempt)
                logger.error(
                    f"Cancel listener disconnected, retrying in {delay:.1f}s: {e}",
                    attempt=attempt,
                )
                await asyncio.sleep(delay)

    async def _subscribe_and_handle_cancels(self) -> None:
        """Subscribe to cancel channel and process commands until stopped."""
        client = redis_manager.get_client()
        pubsub = client.pubsub()
        await pubsub.subscribe(self._cancel_channel, self._thread_channel)

        try:
            while self._running:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=0.5,
                )
                if message is None:
                    continue
                if message["type"] != "message":
                    continue
                if message.get("channel") == self._thread_channel:
                    self._wake_thread(str(message["data"]))
                    continue

                try:
                    data = json.loads(message["data"])
                    emit_end_event = data.get("emit_end_event", True)
                    if not isinstance(emit_end_event, bool):
                        raise ValueError("emit_end_event must be a boolean")
                    await self._execute_cancel(data["run_id"], emit_end_event=emit_end_event)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Invalid cancel message: {e}")
        finally:
            await pubsub.unsubscribe(self._cancel_channel, self._thread_channel)
            await pubsub.aclose()

    async def _execute_cancel(self, run_id: str, *, emit_end_event: bool = True) -> None:
        """Cancel a locally-owned task and signal the broker."""
        task = active_runs.get(run_id)
        if task is None or task.done():
            return

        logger.info(f"Cancelling run {run_id} (local task found)")
        explicit_run_cancellations.add(run_id)
        task.cancel()

        broker = self.get_or_create_broker(run_id)
        if emit_end_event and not broker.is_finished():
            event_id = await self.allocate_event_id(run_id)
            await broker.put(event_id, ("end", {"status": "interrupted"}))
            # Do NOT call cleanup_broker here - execute_run's finally block
            # owns cleanup.  Leaving the finished broker in _brokers lets
            # signal_run_cancelled see is_finished() → True and skip the
            # duplicate end event.

    async def start_cleanup_task(self) -> None:
        """No-op. Redis TTL handles replay buffer expiry."""

    async def stop_cleanup_task(self) -> None:
        """No-op."""
