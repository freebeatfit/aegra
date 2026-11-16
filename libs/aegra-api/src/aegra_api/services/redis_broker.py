"""Redis-backed broker for multi-instance streaming.

Each thread's events live in one Redis Stream, ``<prefix>thread:{<thread_id>}:events``
(the braces are a Cluster hash tag, so a thread's keys share a slot). Every run
on the thread appends to it; a thread stream reads it with ``XREAD BLOCK`` from
its cursor on a connection from a dedicated pool. Writes trim the stream to
about ``REPLAY_MAX_EVENTS`` entries and expire it ``REPLAY_RETENTION_SECONDS``
after the last one. Cancel commands still go over pub/sub: they are signals,
not data.
"""

import asyncio
import contextlib
import json
import random
from collections.abc import Awaitable
from typing import Any

import anyio
import structlog
from redis import RedisError

from aegra_api.core.active_runs import active_runs, explicit_run_cancellations
from aegra_api.core.redis_manager import redis_manager
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.models.enums import RunCancellationAction
from aegra_api.services.base_broker import (
    READ_BATCH,
    REPLAY_MAX_EVENTS,
    REPLAY_RETENTION_SECONDS,
    BaseBrokerManager,
    ThreadEntry,
)
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

_serializer = GeneralSerializer()

# Reconnect backoff for the Redis cancel listener and write retries
_BACKOFF_BASE = 0.5
_BACKOFF_MAX = 30.0
_BACKOFF_FACTOR = 2.0

# Bounded retry for writes: a producer must never block forever, but a transient
# blip is retried rather than dropping the event. Readers drop a retried copy by write id.
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


async def _release_on_cancel(read: Awaitable[Any]) -> Any:
    """Await *read* in its own task, so cancelling the reader still returns its connection.

    anyio re-cancels every await of a cancelled scope, which can interrupt redis-py
    releasing the connection to the pool and leak the slot.
    """
    task = asyncio.ensure_future(read)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.cancel()
        with anyio.CancelScope(shield=True), contextlib.suppress(asyncio.CancelledError, RedisError):
            await task
        raise


class RedisBrokerManager(BaseBrokerManager):
    """Per-thread Redis Streams for events; pub/sub only for cross-instance cancel."""

    def __init__(self) -> None:
        super().__init__()
        self._channel_prefix = settings.redis.REDIS_CHANNEL_PREFIX
        self._cancel_channel = f"{self._channel_prefix}cancel"
        self._listener_task: asyncio.Task[None] | None = None
        self._running: bool = False

    def stream_key(self, thread_id: str) -> str:
        return f"{self._channel_prefix}thread:{{{thread_id}}}:events"

    async def append(self, thread_id: str, run_id: str, write_id: str, event: Any) -> None:
        fields = {"r": run_id, "w": write_id, "e": _serialize_payload(event)}
        key = self.stream_key(thread_id)
        attempt = 0
        while True:
            try:
                pipe = redis_manager.get_client().pipeline()
                pipe.xadd(key, fields, maxlen=REPLAY_MAX_EVENTS, approximate=True)
                pipe.expire(key, REPLAY_RETENTION_SECONDS)
                await pipe.execute()
                return
            except RedisError as e:
                attempt += 1
                if attempt >= _PUT_MAX_ATTEMPTS:
                    logger.error(f"Redis write failed for run {run_id} after {attempt} attempts: {e}")
                    return
                delay = _backoff_delay(attempt)
                logger.warning(f"Redis write failed for run {run_id}, retrying in {delay:.1f}s: {e}", attempt=attempt)
                await asyncio.sleep(delay)

    async def read_thread(self, thread_id: str, after: str) -> list[ThreadEntry]:
        client = redis_manager.get_stream_read_client()
        key = self.stream_key(thread_id)
        reply = await _release_on_cancel(
            client.xread({key: after}, count=READ_BATCH, block=redis_manager.stream_read_block_ms)
        )
        if not reply:
            return []
        # RESP2 answers [[key, entries]], RESP3 {key: [entries]}.
        entries = reply[key][0] if isinstance(reply, dict) else reply[0][1]
        return [
            (entry_id, fields["r"], fields["w"], _deserialize_payload(json.loads(fields["e"])))
            for entry_id, fields in entries
        ]

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
        await pubsub.subscribe(self._cancel_channel)

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

                try:
                    data = json.loads(message["data"])
                    emit_end_event = data.get("emit_end_event", True)
                    if not isinstance(emit_end_event, bool):
                        raise ValueError("emit_end_event must be a boolean")
                    await self._execute_cancel(data["run_id"], emit_end_event=emit_end_event)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Invalid cancel message: {e}")
        finally:
            await pubsub.unsubscribe(self._cancel_channel)
            await pubsub.aclose()

    async def _execute_cancel(self, run_id: str, *, emit_end_event: bool = True) -> None:
        """Cancel a locally-owned task and end its stream."""
        task = active_runs.get(run_id)
        if task is None or task.done():
            return

        logger.info(f"Cancelling run {run_id} (local task found)")
        explicit_run_cancellations.add(run_id)
        task.cancel()
        if emit_end_event:
            await self.end_cancelled_run(run_id)
