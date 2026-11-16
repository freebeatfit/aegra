"""RedisRunBroker replay storage against a real Redis (FB patch P2).

Set ``AEGRA_TEST_REDIS_URL`` (e.g. ``redis://localhost:6379/15``) to run;
skipped otherwise. Each test uses its own run ids and deletes its keys.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import redis.asyncio as aioredis

from aegra_api.services import redis_broker as redis_broker_module
from aegra_api.services.redis_broker import RedisRunBroker

REDIS_URL = os.environ.get("AEGRA_TEST_REDIS_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not REDIS_URL, reason="AEGRA_TEST_REDIS_URL not set"),
]


class _Manager:
    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    def get_client(self) -> aioredis.Redis:
        return self._client


@pytest.fixture
async def redis_client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[aioredis.Redis]:
    client = aioredis.from_url(REDIS_URL or "", decode_responses=True)
    monkeypatch.setattr(redis_broker_module, "redis_manager", _Manager(client))
    yield client
    await client.aclose()


@pytest.fixture
async def broker(redis_client: aioredis.Redis) -> AsyncIterator[RedisRunBroker]:
    run_id = f"test-{uuid.uuid4().hex}"
    instance = RedisRunBroker(run_id, f"t:{run_id}", f"t:cache:{run_id}", f"t:counter:{run_id}")
    yield instance
    await redis_client.delete(f"t:cache:{run_id}", f"t:cache:{run_id}:snapshots", f"t:counter:{run_id}")


def _values(data: dict[str, Any], *, namespace: list[str] | None = None, **params: Any) -> tuple[str, dict[str, Any]]:
    return (
        "values",
        {"type": "event", "method": "values", "params": {"namespace": namespace or [], "data": data, **params}},
    )


def _event(method: str, data: Any) -> tuple[str, dict[str, Any]]:
    return (method, {"type": "event", "method": method, "params": {"namespace": [], "data": data}})


def _eid(broker: RedisRunBroker, n: int) -> str:
    return f"{broker.run_id}_event_{n}"


async def test_values_keep_only_the_newest_snapshot(broker: RedisRunBroker, redis_client: aioredis.Redis) -> None:
    big = "x" * 10_000
    for n in range(1, 6):
        await broker.put(_eid(broker, n), _values({"messages": [big] * n}))
    assert await redis_client.xlen(broker._cache_key) == 0
    snapshots = await redis_client.hgetall(broker._snapshot_key)
    assert list(snapshots) == [""]
    assert json.loads(snapshots[""])["event_id"] == _eid(broker, 5)

    replayed = await broker.replay(None)
    assert [eid for eid, _ in replayed] == [_eid(broker, 5)]


async def test_snapshot_keeps_its_place_among_deltas(broker: RedisRunBroker) -> None:
    await broker.put(_eid(broker, 1), _event("checkpoints", {"id": "c0", "step": 0}))
    await broker.put(_eid(broker, 2), _values({"messages": ["a"]}))
    await broker.put(_eid(broker, 3), _event("messages", {"event": "message-start"}))
    await broker.put(_eid(broker, 4), _event("checkpoints", {"id": "c1", "step": 1}))
    await broker.put(_eid(broker, 5), _values({"messages": ["a", "b"]}))
    await broker.put(_eid(broker, 6), _event("messages", {"event": "message-finish"}))
    await broker.put(_eid(broker, 7), ("end", {"status": "success"}))

    replayed = [eid for eid, _ in await broker.replay(None)]
    n = [int(eid.rsplit("_", 1)[1]) for eid in replayed]
    # Superseded values (2) are gone; the newest (5) sits between its envelope and the next delta.
    assert n == [1, 3, 4, 5, 6, 7]


async def test_replay_from_a_cursor_reads_only_later_events(broker: RedisRunBroker) -> None:
    for n in range(1, 501):
        await broker.put(_eid(broker, n), _event("messages", {"i": n}))
    await broker.put(_eid(broker, 501), _values({"messages": ["final"]}))

    replayed = await broker.replay(_eid(broker, 480))
    assert [int(eid.rsplit("_", 1)[1]) for eid, _ in replayed] == list(range(481, 502))


async def test_cursor_past_the_snapshot_skips_it(broker: RedisRunBroker) -> None:
    await broker.put(_eid(broker, 1), _values({"messages": ["a"]}))
    await broker.put(_eid(broker, 2), _event("messages", {"event": "message-start"}))
    replayed = await broker.replay(_eid(broker, 1))
    assert [eid for eid, _ in replayed] == [_eid(broker, 2)]


async def test_replay_streams_in_batches(broker: RedisRunBroker, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redis_broker_module, "_REPLAY_BATCH", 7)
    for n in range(1, 31):
        await broker.put(_eid(broker, n), _event("messages", {"i": n}))
    await broker.put(_eid(broker, 31), _values({"messages": ["v"]}))
    await broker.put(_eid(broker, 32), ("end", {"status": "success"}))
    seen = [eid async for eid, _ in broker.replay_iter(None)]
    assert [int(eid.rsplit("_", 1)[1]) for eid in seen] == list(range(1, 33))


async def test_interrupt_values_stay_in_the_log(broker: RedisRunBroker, redis_client: aioredis.Redis) -> None:
    await broker.put(_eid(broker, 1), _values({"messages": []}))
    await broker.put(_eid(broker, 2), _values({"messages": []}, interrupts=[{"id": "i1", "value": {"q": 1}}]))
    await broker.put(_eid(broker, 3), ("end", {"status": "interrupted"}))
    assert await redis_client.xlen(broker._cache_key) == 2
    replayed = [eid for eid, _ in await broker.replay(None)]
    assert replayed == [_eid(broker, 1), _eid(broker, 2), _eid(broker, 3)]


async def test_retried_write_is_stored_once(broker: RedisRunBroker, redis_client: aioredis.Redis) -> None:
    message = json.dumps({"event_id": _eid(broker, 1), "payload": ["messages", {}]})
    await broker._cache_event(_eid(broker, 1), message)
    await broker._cache_event(_eid(broker, 1), message)  # a retry Redis already applied
    assert await redis_client.xlen(broker._cache_key) == 1


async def test_fallback_event_id_lands_in_write_order(broker: RedisRunBroker) -> None:
    await broker.put(_eid(broker, 1), _event("messages", {"i": 1}))
    await broker.put(f"{broker.run_id}_event_1794574673377", _event("messages", {"i": 2}))
    await broker.put(_eid(broker, 3), ("end", {"status": "success"}))
    replayed = [payload for _, payload in await broker.replay(None)]
    assert [p[0] for p in replayed] == ["messages", "messages", "end"]


async def test_end_in_buffer_is_detected(broker: RedisRunBroker) -> None:
    await broker.put(_eid(broker, 1), _event("messages", {"i": 1}))
    assert await broker._check_end_in_buffer() is False
    broker._finished = False
    await broker.put(_eid(broker, 2), ("end", {"status": "success"}))
    broker._finished = False
    assert await broker._check_end_in_buffer() is True


async def test_v1_values_are_appended_unchanged(broker: RedisRunBroker, redis_client: aioredis.Redis) -> None:
    await broker.put(_eid(broker, 1), ("values", {"messages": ["a"]}))
    await broker.put(_eid(broker, 2), ("values", {"messages": ["a", "b"]}))
    assert await redis_client.xlen(broker._cache_key) == 2
    assert [p for _, p in await broker.replay(None)] == [
        ("values", {"messages": ["a"]}),
        ("values", {"messages": ["a", "b"]}),
    ]
