"""Redis-backed broker for multi-instance streaming.

Each thread's events live in one Redis Stream, ``<prefix>thread:{<thread_id>}:events``,
and its latest full state per namespace in two hashes, ``...:snapshots`` (namespace ->
record) and ``...:snapshot-ids`` (namespace -> "<marker entry id> <seq>"); ``...:seq``
holds the thread's last seq and outlives the others by a day. The braces are a
Cluster hash tag, so a thread's keys share a slot. Every run on the thread appends
to the stream; a thread stream reads it with ``XREAD BLOCK`` from its cursor on a
connection from a dedicated pool. Cancel commands still go over pub/sub: they are
signals, not data.
"""

import asyncio
import contextlib
import json
import random
from collections.abc import Awaitable
from typing import Any, NamedTuple

import anyio
import structlog
from redis import RedisError

from aegra_api.core.active_runs import active_runs, explicit_run_cancellations
from aegra_api.core.redis_manager import redis_manager
from aegra_api.core.serializers import GeneralSerializer
from aegra_api.models.enums import RunCancellationAction
from aegra_api.services.base_broker import (
    READ_BATCH,
    ROOT_NAMESPACE,
    SEQ_STRIDE,
    SNAPSHOT_MARKER,
    BaseBrokerManager,
    Snapshot,
    ThreadEntry,
    entry_key,
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

# The seq counter outlives the stream window by far, so seq never goes back after the stream expires.
_SEQ_TTL_SECONDS = 24 * 3600

# One atomic write per event on keys sharing the thread's hash tag; a script, not MULTI, because seq, the
# marker id and the trim bound each depend on what Redis holds. ARGV: see _write.
_WRITE_SCRIPT = f"""
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local oldest = now_ms - tonumber(ARGV[1])
local root = redis.call('HGET', KEYS[2], '{ROOT_NAMESPACE}')
if root then oldest = math.min(oldest, tonumber(string.match(root, '^%d+'))) end
local minid = string.format('%.0f', oldest)
local seq = string.format('%.0f', math.max(tonumber(redis.call('GET', KEYS[4]) or 0) + {SEQ_STRIDE}, now_ms * {SEQ_STRIDE}))
redis.call('SET', KEYS[4], seq, 'EX', ARGV[7])
local id
if ARGV[5] == '' then
  id = redis.call('XADD', KEYS[1], 'MINID', '~', minid, '*', 's', seq, 'r', ARGV[3], 'w', ARGV[4], 'e', ARGV[6])
else
  id = redis.call('XADD', KEYS[1], 'MINID', '~', minid, '*', 's', seq, 'r', ARGV[3], 'w', ARGV[4], 'm', ARGV[5])
  redis.call('HSET', KEYS[2], ARGV[5], id .. ' ' .. seq)
  redis.call('HSET', KEYS[3], ARGV[5], ARGV[6])
end
local first = redis.call('XRANGE', KEYS[1], '-', '+', 'COUNT', 1)[1][1]
local fms, fn = string.match(first, '^(%d+)-(%d+)$')
local ids = redis.call('HGETALL', KEYS[2])
for i = 1, #ids, 2 do
  local ms, n = string.match(ids[i + 1], '^(%d+)-(%d+) ')
  if ids[i] ~= '{ROOT_NAMESPACE}' and (tonumber(ms) < tonumber(fms) or (ms == fms and tonumber(n) < tonumber(fn))) then
    redis.call('HDEL', KEYS[2], ids[i])
    redis.call('HDEL', KEYS[3], ids[i])
  end
end
for i = 1, 3 do redis.call('EXPIRE', KEYS[i], ARGV[2]) end
return id
"""


class _ThreadKeys(NamedTuple):
    events: str
    snapshot_ids: str
    snapshots: str
    seq: str


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


def _entry_event(fields: dict[str, str]) -> Any:
    """A stream entry's event: the stored one, or a snapshot marker's (SNAPSHOT_MARKER, namespace)."""
    if "m" in fields:
        return (SNAPSHOT_MARKER, fields["m"])
    return _deserialize_payload(json.loads(fields["e"]))


def _stored_seq(entry_id: str, seq: str | None) -> int:
    """The entry's stored seq; an entry fb.6 wrote has none, so number it as fb.6 did."""
    # ponytail: rolling-upgrade shim, delete once no fb.6 keys can be left (one stream window after fb.7 is live).
    if seq is not None:
        return int(seq)
    ms, n = entry_key(entry_id)
    return ((ms << 8) + n) << 3


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
        return self._keys(thread_id).events

    def _keys(self, thread_id: str) -> _ThreadKeys:
        base = f"{self._channel_prefix}thread:{{{thread_id}}}"
        return _ThreadKeys(f"{base}:events", f"{base}:snapshot-ids", f"{base}:snapshots", f"{base}:seq")

    async def append(self, thread_id: str, run_id: str, write_id: str, event: Any) -> bool:
        return await self._write(
            thread_id, run_id=run_id, write_id=write_id, namespace="", payload=_serialize_payload(event)
        )

    async def put_snapshot(
        self, thread_id: str, *, run_id: str, write_id: str, namespace: str, record: dict[str, Any]
    ) -> bool:
        return await self._write(
            thread_id, run_id=run_id, write_id=write_id, namespace=namespace, payload=_serialize_payload(record)
        )

    async def _write(self, thread_id: str, *, run_id: str, write_id: str, namespace: str, payload: str) -> bool:
        window = settings.event_streaming.RESUMABLE_STREAM_TTL_SECONDS
        # namespace "" is a plain event; otherwise payload is that namespace's snapshot record.
        args = [window * 1000, window, run_id, write_id, namespace, payload, _SEQ_TTL_SECONDS]
        attempt = 0
        while True:
            try:
                # EVALSHA, loading the script on a NOSCRIPT reply.
                await redis_manager.get_client().register_script(_WRITE_SCRIPT)(
                    keys=list(self._keys(thread_id)), args=args
                )
                return True
            except RedisError as e:
                attempt += 1
                if attempt >= _PUT_MAX_ATTEMPTS:
                    logger.error(f"Redis write failed for run {run_id} after {attempt} attempts: {e}")
                    return False
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
            ThreadEntry(
                entry_id, _stored_seq(entry_id, fields.get("s")), fields["r"], fields["w"], _entry_event(fields)
            )
            for entry_id, fields in entries
        ]

    async def read_snapshot_ids(self, thread_id: str) -> dict[str, str]:
        ids = await redis_manager.get_client().hgetall(self._keys(thread_id).snapshot_ids)  # type: ignore[invalid-await]  # redis.asyncio stubs
        return {namespace: marker.partition(" ")[0] for namespace, marker in ids.items()}

    async def read_snapshots(self, thread_id: str, namespaces: list[str]) -> dict[str, Snapshot]:
        keys = self._keys(thread_id)
        # One transaction, so each id belongs to the record read with it.
        pipe = redis_manager.get_client().pipeline(transaction=True)
        pipe.hmget(keys.snapshot_ids, namespaces)
        pipe.hmget(keys.snapshots, namespaces)
        markers, records = await pipe.execute()
        snapshots: dict[str, Snapshot] = {}
        for namespace, marker, record in zip(namespaces, markers, records, strict=True):
            if marker is None or record is None:
                continue
            entry_id, _, seq = marker.partition(" ")
            snapshots[namespace] = Snapshot(entry_id, _stored_seq(entry_id, seq or None), json.loads(record))
        return snapshots

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
