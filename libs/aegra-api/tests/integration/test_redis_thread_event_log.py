"""The per-thread event log against a real Redis (FB patches P2/P3).

Set ``AEGRA_TEST_REDIS_URL`` (e.g. ``redis://localhost:6379/15``) to run;
skipped otherwise. Each test uses its own key prefix and deletes its keys.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator

import pytest
import redis.asyncio as aioredis

from aegra_api.core.redis_manager import RedisManager
from aegra_api.services import redis_broker as redis_broker_module
from aegra_api.services.base_broker import READ_BATCH, REPLAY_RETENTION_SECONDS, entry_key
from aegra_api.services.redis_broker import RedisBrokerManager
from aegra_api.settings import settings

REDIS_URL = os.environ.get("AEGRA_TEST_REDIS_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not REDIS_URL, reason="AEGRA_TEST_REDIS_URL not set"),
]


@pytest.fixture
async def redis_client() -> AsyncIterator[aioredis.Redis]:
    client = aioredis.from_url(REDIS_URL or "", decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
async def manager(monkeypatch: pytest.MonkeyPatch, redis_client: aioredis.Redis) -> AsyncIterator[RedisBrokerManager]:
    prefix = f"test-{uuid.uuid4().hex}:"
    monkeypatch.setattr(settings.redis, "REDIS_URL", REDIS_URL or "")
    monkeypatch.setattr(settings.redis, "REDIS_CHANNEL_PREFIX", prefix)
    redis = RedisManager()
    redis.stream_read_block_ms = 200
    await redis.initialize()
    monkeypatch.setattr(redis_broker_module, "redis_manager", redis)
    yield RedisBrokerManager()
    await redis.close()
    keys = [key async for key in redis_client.scan_iter(f"{prefix}*")]
    if keys:
        await redis_client.delete(*keys)


async def test_runs_of_a_thread_share_one_hash_tagged_stream_in_write_order(
    manager: RedisBrokerManager, redis_client: aioredis.Redis
) -> None:
    await manager.append("t-1", "run-1", "w1", ("values", {"a": 1}))
    await manager.append("t-1", "run-2", "w2", ("messages", {"event": "message-start"}))
    await manager.append("t-1", "run-1", "w3", ("end", {"status": "success"}))

    entries = await manager.read_thread("t-1", "0-0")

    key = manager.stream_key("t-1")
    assert key.endswith("thread:{t-1}:events")
    assert await redis_client.xlen(key) == 3
    assert 0 < await redis_client.ttl(key) <= REPLAY_RETENTION_SECONDS
    assert [(run_id, write_id, event) for _, run_id, write_id, event in entries] == [
        ("run-1", "w1", ("values", {"a": 1})),
        ("run-2", "w2", ("messages", {"event": "message-start"})),
        ("run-1", "w3", ("end", {"status": "success"})),
    ]
    keys = [entry_key(entry_id) for entry_id, *_ in entries]
    assert keys == sorted(set(keys))


async def test_read_from_a_cursor_returns_later_entries_in_batches(manager: RedisBrokerManager) -> None:
    for i in range(READ_BATCH + 50):
        await manager.append("t-1", "run-1", f"w{i}", ("messages", {"i": i}))

    first = await manager.read_thread("t-1", "0-0")
    rest = await manager.read_thread("t-1", first[-1][0])

    assert len(first) == READ_BATCH
    assert [event[1]["i"] for *_, event in first + rest] == list(range(READ_BATCH + 50))


async def test_blocked_read_returns_when_a_write_lands(
    manager: RedisBrokerManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(redis_broker_module.redis_manager, "stream_read_block_ms", 5_000)
    reader = asyncio.create_task(manager.read_thread("t-1", "0-0"))
    await asyncio.sleep(0.1)
    assert not reader.done()

    await manager.append("t-1", "run-1", "w1", ("values", {"a": 1}))

    entries = await asyncio.wait_for(reader, timeout=2.0)
    assert [event for *_, event in entries] == [("values", {"a": 1})]


async def test_idle_read_returns_nothing_after_its_block(manager: RedisBrokerManager) -> None:
    assert await manager.read_thread("t-1", "0-0") == []


async def test_writes_trim_the_stream(
    manager: RedisBrokerManager, redis_client: aioredis.Redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(redis_broker_module, "REPLAY_MAX_EVENTS", 10)
    for i in range(300):
        await manager.append("t-1", "run-1", f"w{i}", ("messages", {"i": i}))

    # Approximate trimming drops whole stream nodes (100 entries by default).
    assert await redis_client.xlen(manager.stream_key("t-1")) < 300
