"""One event stream per thread, over HTTP, against real Redis and PostgreSQL (FB patches P2/P3).

A uvicorn server on a random port serves the v2 routes with the Redis broker
and the local executor; runs execute a small echo graph. Clients drive
``POST /threads/{t}/commands`` and ``POST /threads/{t}/stream/events`` as the
SDK does. Needs a migrated PostgreSQL (``settings.db``) and
``AEGRA_TEST_REDIS_URL``; skipped otherwise.
"""

from __future__ import annotations

import asyncio
import json
import operator
import os
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict
from uuid import uuid4

import httpx
import psycopg
import pytest
import redis
import uvicorn
from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from aegra_api.api import event_streaming as es_module
from aegra_api.core import orm
from aegra_api.core.auth_deps import get_current_user, require_auth
from aegra_api.core.redis_manager import STREAM_READ_CLIENT_NAME, RedisManager
from aegra_api.models.auth import User
from aegra_api.services import redis_broker as redis_broker_module
from aegra_api.services import streaming_service as streaming_service_module
from aegra_api.services.event_streaming import capabilities as caps
from aegra_api.services.event_streaming import session as session_module
from aegra_api.services.redis_broker import RedisBrokerManager
from aegra_api.settings import settings
from tests.fixtures.streams import is_root_terminal

REDIS_URL = os.environ.get("AEGRA_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="AEGRA_TEST_REDIS_URL not set")

_GRAPH_ID = "thread_stream_echo"
# Short XREAD BLOCK so idle-time assertions cover many read cycles quickly.
_BLOCK_MS = 300


class _EchoState(TypedDict):
    items: Annotated[list[str], operator.add]


async def _echo(state: _EchoState) -> dict[str, list[str]]:
    last = state["items"][-1]
    if last.startswith("big:"):
        return {"items": ["x" * int(last.removeprefix("big:"))]}
    return {"items": [f"echo:{last}"]}


class _GraphService:
    """Serves one compiled graph in place of the aegra.json registry."""

    def __init__(self) -> None:
        builder = StateGraph(_EchoState)
        builder.add_node("echo", _echo)
        builder.add_edge(START, "echo")
        builder.add_edge("echo", END)
        self._graph = builder.compile(checkpointer=InMemorySaver())

    def list_graphs(self) -> dict[str, str]:
        return {_GRAPH_ID: "in-memory"}

    @asynccontextmanager
    async def get_graph(self, _graph_id: str, **_kwargs: Any) -> AsyncIterator[Any]:
        yield self._graph


@dataclass
class _Server:
    base_url: str
    port: int
    assistant_id: str
    broker: RedisBrokerManager
    redis: redis.Redis
    statements: list[str] = field(default_factory=list)

    async def start_run(self, http: httpx.AsyncClient, thread_id: str, text: str) -> str:
        params = {"assistant_id": self.assistant_id, "input": {"items": [text]}}
        resp = await http.post(
            f"/threads/{thread_id}/commands", json={"id": 1, "method": "run.start", "params": params}
        )
        body = resp.json()
        assert body["type"] == "success", body
        return body["result"]["run_id"]

    def reader_ids(self) -> list[str]:
        db = self.redis.connection_pool.connection_kwargs.get("db", 0)
        return [
            str(client["id"])
            for client in self.redis.client_list()
            if client.get("name") == STREAM_READ_CLIENT_NAME and str(client.get("db")) == str(db)
        ]


class _Stream:
    """An open thread stream: frames read so far, and whether the server ended it."""

    def __init__(self, resp: httpx.Response) -> None:
        self._lines = resp.aiter_lines()
        self.events: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.ended = False

    async def read_until(self, stop: Callable[[_Stream], bool], timeout: float = 15.0) -> None:
        name = None
        async with asyncio.timeout(timeout):
            async for line in self._lines:
                if line.startswith("event: "):
                    name = line.removeprefix("event: ")
                elif line.startswith("data: "):
                    data = json.loads(line.removeprefix("data: "))
                    (self.errors if name == "error" else self.events).append(data)
                    if stop(self):
                        return
            self.ended = True

    def terminals(self) -> int:
        return sum(is_root_terminal(event) for event in self.events)


@asynccontextmanager
async def _open(http: httpx.AsyncClient, thread_id: str, channels: list[str]) -> AsyncIterator[_Stream]:
    async with http.stream("POST", f"/threads/{thread_id}/stream/events", json={"channels": channels}) as resp:
        assert resp.status_code == 200
        yield _Stream(resp)


def _runs_ended(count: int) -> Callable[[_Stream], bool]:
    return lambda stream: stream.terminals() >= count


def _skip_without_migrated_database(url: str) -> None:
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            row = conn.execute("SELECT to_regclass('public.runs')").fetchone()
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL test database is unavailable: {exc}")
    if row is None or row[0] is None:
        pytest.skip("runs table is unavailable; run Alembic migrations before this DB test")


@pytest.fixture
def stream_server(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Server]:
    """A uvicorn server on the Redis broker; ``request.param`` may set ``pool`` (stream-read connections)."""
    url = settings.db.database_url_sync
    _skip_without_migrated_database(url)
    options: dict[str, Any] = getattr(request, "param", {})
    user = User(identity=f"thread-stream-{uuid4().hex}")
    assistant_id = str(uuid4())
    with psycopg.connect(url) as conn:
        conn.execute(
            "INSERT INTO assistant (assistant_id, name, graph_id, user_id) VALUES (%s, %s, %s, %s)",
            (assistant_id, "thread-stream", _GRAPH_ID, user.identity),
        )

    service = _GraphService()
    for module in ("run_preparation", "run_executor", "langgraph_service"):
        monkeypatch.setattr(f"aegra_api.services.{module}.get_langgraph_service", lambda: service)
    # NullPool: the server runs on its own loop; asyncpg connections can't cross loops.
    engine = create_async_engine(settings.db.database_url, poolclass=NullPool)
    statements: list[str] = []
    event.listen(engine.sync_engine, "before_cursor_execute", lambda *args: statements.append(args[2]))
    monkeypatch.setattr(orm, "async_session_maker", async_sessionmaker(engine, expire_on_commit=False))
    monkeypatch.setattr(caps.settings.event_streaming, "FF_V2_EVENT_STREAMING", True)
    caps._probe_runtime_symbols.cache_clear()

    prefix = f"test-{uuid4().hex[:12]}:"
    monkeypatch.setattr(settings.redis, "REDIS_URL", REDIS_URL or "")
    monkeypatch.setattr(settings.redis, "REDIS_CHANNEL_PREFIX", prefix)
    monkeypatch.setattr(settings.redis, "REDIS_STREAM_READ_MAX_CONNECTIONS", options.get("pool", 50))
    redis_manager = RedisManager()
    redis_manager.stream_read_block_ms = _BLOCK_MS
    broker = RedisBrokerManager()
    monkeypatch.setattr(redis_broker_module, "redis_manager", redis_manager)
    monkeypatch.setattr(session_module, "broker_manager", broker)
    monkeypatch.setattr(streaming_service_module, "broker_manager", broker)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await redis_manager.initialize()
        await broker.start()
        yield
        await broker.stop()
        await redis_manager.close()

    app = FastAPI(lifespan=lifespan)
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_current_user] = lambda: user
    app.include_router(es_module.router)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", timeout_graceful_shutdown=2)
    )
    thread = threading.Thread(target=asyncio.run, args=(server.serve(),), daemon=True)
    thread.start()
    sync_redis = redis.Redis.from_url(REDIS_URL or "", decode_responses=True)
    try:
        for _ in range(250):
            if server.started and server.servers:
                break
            threading.Event().wait(0.02)
        else:
            raise RuntimeError("uvicorn did not start")
        port = server.servers[0].sockets[0].getsockname()[1]
        yield _Server(f"http://127.0.0.1:{port}", port, assistant_id, broker, sync_redis, statements)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        keys = list(sync_redis.scan_iter(f"{prefix}*"))
        if keys:
            sync_redis.delete(*keys)
        sync_redis.close()
        with psycopg.connect(url) as conn:
            conn.execute("DELETE FROM runs WHERE user_id = %s", (user.identity,))
            conn.execute("DELETE FROM thread WHERE user_id = %s", (user.identity,))
            conn.execute("DELETE FROM assistant WHERE assistant_id = %s", (assistant_id,))
        if thread.is_alive():
            raise TimeoutError("uvicorn did not shut down")


def _root_lifecycle(stream: _Stream) -> list[tuple[str, str]]:
    return [
        (event["params"]["data"]["event"], event["event_id"])
        for event in stream.events
        if event["method"] == "lifecycle" and not event["params"]["namespace"]
    ]


@pytest.mark.asyncio
async def test_run_started_after_the_stream_opened_is_delivered(stream_server: _Server) -> None:
    thread_id = str(uuid4())
    async with (
        httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http,
        _open(http, thread_id, ["values", "lifecycle"]) as stream,
    ):
        await asyncio.sleep(0.5)  # the stream is now blocked on an empty thread
        run_id = await stream_server.start_run(http, thread_id, "hi")
        await stream.read_until(_runs_ended(1))

    assert [status for status, _ in _root_lifecycle(stream)] == ["running", "completed"]
    assert all(event_id.startswith(f"synth:{run_id}:") for _, event_id in _root_lifecycle(stream))
    assert [e["params"]["data"] for e in stream.events if e["method"] == "values"][-1] == {"items": ["hi", "echo:hi"]}


@pytest.mark.asyncio
async def test_seq_rises_across_runs_and_is_the_same_on_every_connection(stream_server: _Server) -> None:
    thread_id = str(uuid4())
    async with httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http:
        async with _open(http, thread_id, ["values", "lifecycle"]) as first:
            run_ids = [await stream_server.start_run(http, thread_id, "one")]
            await first.read_until(_runs_ended(1))
            run_ids.append(await stream_server.start_run(http, thread_id, "two"))
            await first.read_until(_runs_ended(2))
        async with _open(http, thread_id, ["values", "lifecycle"]) as second:
            await second.read_until(_runs_ended(2))

    seqs = [event["seq"] for event in first.events]
    assert seqs == sorted(set(seqs))
    assert [event_id.split(":")[1] for _, event_id in _root_lifecycle(first)] == [run_ids[0]] * 2 + [run_ids[1]] * 2
    assert [(e["event_id"], e["seq"]) for e in second.events] == [(e["event_id"], e["seq"]) for e in first.events]


@pytest.mark.asyncio
async def test_idle_streams_query_no_database(stream_server: _Server) -> None:
    thread_id = str(uuid4())
    async with (
        httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http,
        _open(http, thread_id, ["lifecycle"]) as first,
        _open(http, thread_id, ["values"]) as second,
    ):
        await stream_server.start_run(http, thread_id, "hi")
        await first.read_until(_runs_ended(1))
        await second.read_until(lambda stream: len(stream.events) >= 2)
        async with _open(http, thread_id, ["lifecycle"]) as third:
            await third.read_until(_runs_ended(1))
            before = len(stream_server.statements)

            await asyncio.sleep(2.0)  # about 6 empty XREAD BLOCK cycles per stream

            idle_statements = stream_server.statements[before:]
    assert idle_statements == []


@pytest.mark.asyncio
async def test_reader_resumes_after_client_kill_without_gap_or_duplicate(stream_server: _Server) -> None:
    thread_id = str(uuid4())
    async with httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http:
        async with _open(http, thread_id, ["values", "lifecycle"]) as killed:
            await stream_server.start_run(http, thread_id, "one")
            await killed.read_until(_runs_ended(1))
            readers = stream_server.reader_ids()
            assert readers
            for client_id in readers:
                stream_server.redis.execute_command("CLIENT", "KILL", "ID", client_id)
            await stream_server.start_run(http, thread_id, "two")
            await killed.read_until(_runs_ended(2))
            reconnected = stream_server.reader_ids()
        async with _open(http, thread_id, ["values", "lifecycle"]) as fresh:
            await fresh.read_until(_runs_ended(2))

    assert reconnected and not set(reconnected) & set(readers)
    event_ids = [event["event_id"] for event in killed.events]
    assert len(event_ids) == len(set(event_ids))
    assert [(e["event_id"], e["seq"]) for e in killed.events] == [(e["event_id"], e["seq"]) for e in fresh.events]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_server", [{"pool": 1}], indirect=True)
async def test_one_connection_pool_serves_reopened_and_concurrent_streams(stream_server: _Server) -> None:
    """Closed streams give their connection back, and a full pool waits instead of failing."""
    thread_id = str(uuid4())
    async with httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http:
        for _ in range(3):
            async with _open(http, thread_id, ["lifecycle"]):
                await asyncio.sleep(0.4)  # the reader is blocked in XREAD when the client leaves
        async with _open(http, thread_id, ["lifecycle"]) as first, _open(http, thread_id, ["lifecycle"]) as second:
            await stream_server.start_run(http, thread_id, "hi")
            await asyncio.gather(first.read_until(_runs_ended(1)), second.read_until(_runs_ended(1)))

    assert [status for status, _ in _root_lifecycle(first)] == ["running", "completed"]
    assert [status for status, _ in _root_lifecycle(second)] == ["running", "completed"]
