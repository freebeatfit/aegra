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
import socket
import threading
import time
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
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.stream.transformers import CustomTransformer
from langgraph.types import interrupt
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from aegra_api.api import event_streaming as es_module
from aegra_api.core import orm
from aegra_api.core import sse as sse_module
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


class _EchoState(TypedDict, total=False):
    items: Annotated[list[str], operator.add]
    blob: str
    step: int


class _SubState(TypedDict):
    items: Annotated[list[str], operator.add]


# The input's last item picks the behavior: "big:<bytes>", "loop:<steps>:<blob bytes>[:<ms per step>]",
# "sub:<x>" (a subgraph), "ask:<x>" (an interrupt), "tick:<count>:<ms apart>" (custom events), else echo.
def _route(state: _EchoState) -> str:
    name = state["items"][-1].split(":", 1)[0]
    return name if name in ("sub", "ask", "tick") else "echo"


async def _echo(state: _EchoState) -> dict[str, Any]:
    last = state["items"][-1]
    if last.startswith("big:"):
        return {"items": ["x" * int(last.removeprefix("big:"))]}
    if last.startswith("loop:"):
        _, _, size, *delay = last.split(":")
        await asyncio.sleep(int(delay[0]) / 1000 if delay else 0)
        step = state.get("step", 0) + 1
        return {"blob": f"{step}:" + "x" * int(size), "step": step}
    return {"items": [f"echo:{last}"]}


def _loops(state: _EchoState) -> str:
    last = state["items"][-1]
    return "echo" if last.startswith("loop:") and state["step"] < int(last.split(":")[1]) else END


async def _inner(_state: _SubState) -> dict[str, list[str]]:
    return {"items": ["inner"]}


async def _ask(_state: _EchoState) -> dict[str, list[str]]:
    return {"items": [interrupt("question?")]}


async def _tick(state: _EchoState) -> dict[str, list[str]]:
    _, count, delay = state["items"][-1].split(":")
    write = get_stream_writer()
    for i in range(int(count)):
        write({"tick": i})
        await asyncio.sleep(int(delay) / 1000)
    return {"items": ["ticked"]}


class _GraphService:
    """Serves one compiled graph in place of the aegra.json registry."""

    def __init__(self) -> None:
        sub = StateGraph(_SubState)
        sub.add_node("inner", _inner)
        sub.add_edge(START, "inner")
        sub.add_edge("inner", END)
        builder = StateGraph(_EchoState)
        builder.add_node("echo", _echo)
        builder.add_node("sub", sub.compile())
        builder.add_node("ask", _ask)
        builder.add_node("tick", _tick)
        builder.add_conditional_edges(START, _route)
        builder.add_conditional_edges("echo", _loops)
        for node in ("sub", "ask", "tick"):
            builder.add_edge(node, END)
        self._graph = builder.compile(checkpointer=InMemorySaver(), transformers=[CustomTransformer])

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
    prefix: str
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
        self.bytes = 0

    async def read_until(self, stop: Callable[[_Stream], bool], timeout: float = 15.0) -> None:
        name = None
        async with asyncio.timeout(timeout):
            async for line in self._lines:
                self.bytes += len(line.encode()) + 1
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
        yield _Server(f"http://127.0.0.1:{port}", port, assistant_id, broker, sync_redis, prefix, statements)
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


def _entry_events(stream: _Stream) -> list[tuple[str, int]]:
    """(event_id, seq) of every event but the running seed, whose seq depends on where a replay started."""
    return [(e["event_id"], e["seq"]) for e in stream.events if not e["event_id"].endswith(":running:0")]


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
            await second.read_until(_runs_ended(1))

    seqs = [event["seq"] for event in first.events]
    assert seqs == sorted(set(seqs))
    assert [event_id.split(":")[1] for _, event_id in _root_lifecycle(first)] == [run_ids[0]] * 2 + [run_ids[1]] * 2
    # A new connection replays the latest state on: the same events keep their event id and seq.
    assert _entry_events(second) and set(_entry_events(second)) <= set(_entry_events(first))


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
        await second.read_until(lambda stream: stream.events[-1]["params"]["data"] == {"items": ["hi", "echo:hi"]})
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
            await fresh.read_until(_runs_ended(1))

    assert reconnected and not set(reconnected) & set(readers)
    event_ids = [event["event_id"] for event in killed.events]
    assert len(event_ids) == len(set(event_ids))
    assert [status for status, _ in _root_lifecycle(killed)] == ["running", "completed"] * 2
    assert _entry_events(fresh) and set(_entry_events(fresh)) <= set(_entry_events(killed))


def _read_to_eof(sock: socket.socket, timeout: float) -> bytes:
    """Everything until the server closes; heartbeats would keep a per-recv timeout from firing."""
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        sock.settimeout(max(deadline - time.monotonic(), 0.01))
        chunk = sock.recv(65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
    raise TimeoutError("the server kept the stalled connection open")


@pytest.mark.asyncio
async def test_stalled_client_is_closed_after_the_send_timeout(
    stream_server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sse_module, "SEND_TIMEOUT_SECS", 1.0)
    thread_id = str(uuid4())
    async with (
        httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http,
        _open(http, thread_id, ["lifecycle"]) as stream,
    ):
        await stream_server.start_run(http, thread_id, "big:4000000")  # a 4 MB values event
        await stream.read_until(_runs_ended(1))

    body = json.dumps({"channels": ["values", "lifecycle"]}).encode()
    request = (
        f"POST /threads/{thread_id}/stream/events HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode() + body
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    sock.connect(("127.0.0.1", stream_server.port))
    try:
        sock.sendall(request)
        await asyncio.sleep(3)  # not reading: the server's send blocks, then times out
        received = await asyncio.to_thread(_read_to_eof, sock, 15)
    finally:
        sock.close()

    # The server closed the connection (EOF) without sending the frames after the stalled one.
    assert received.startswith(b"HTTP/1.1 200")
    assert b"event: values" in received
    assert b'"event":"completed"' not in received


@pytest.mark.asyncio
async def test_read_failure_sends_the_error_event_and_closes(stream_server: _Server) -> None:
    thread_id = str(uuid4())
    async with (
        httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http,
        _open(http, thread_id, ["lifecycle"]) as stream,
    ):
        await asyncio.sleep(0.5)
        stream_server.redis.set(stream_server.broker.stream_key(thread_id), "not a stream")
        await stream.read_until(lambda _stream: False)

    assert stream.ended
    assert stream.errors == [
        {"type": "error", "id": None, "error": "unknown_error", "message": "Thread event stream failed."}
    ]


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


# --- Latest-state replay (FB patch P2/P3, fb-super-agent#152) ---

_STATE_CHANNELS = ["values", "checkpoints", "lifecycle"]


def _wire(event: dict[str, Any]) -> tuple[Any, ...]:
    """What a client keys on; a root terminal's timestamp is the time it was sent, so it is left out."""
    params = event["params"]
    timestamp = params["timestamp"] if event["method"] in ("values", "checkpoints") else None
    return event["event_id"], event["seq"], event["method"], params["namespace"], params["data"], timestamp


async def _run_to_end(server: _Server, http: httpx.AsyncClient, thread_id: str, text: str) -> _Stream:
    async with _open(http, thread_id, _STATE_CHANNELS) as live:
        run_id = await server.start_run(http, thread_id, text)
        await live.read_until(
            lambda stream: any(
                is_root_terminal(e) and e["event_id"].startswith(f"synth:{run_id}:") for e in stream.events
            )
        )
    return live


async def _reconnect(http: httpx.AsyncClient, thread_id: str, channels: list[str] = _STATE_CHANNELS) -> _Stream:
    async with _open(http, thread_id, channels) as fresh:
        await fresh.read_until(_runs_ended(1))
    return fresh


def _thread_values(server: _Server) -> list[str]:
    """Every value Redis holds for the server's threads: stream entry fields and hash fields."""
    held: list[str] = []
    for key in server.redis.scan_iter(f"{server.prefix}thread:*"):
        kind = server.redis.type(key)
        if kind == "stream":
            held += [value for _, fields in server.redis.xrange(key) for value in fields.values()]
        elif kind == "hash":
            held += server.redis.hvals(key)
    return held


@pytest.mark.asyncio
async def test_replay_is_each_namespace_latest_snapshot_then_the_later_deltas(stream_server: _Server) -> None:
    thread_id = str(uuid4())
    async with httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http:
        live = await _run_to_end(stream_server, http, thread_id, "sub:x")
        fresh = await _reconnect(http, thread_id)

    subagent = next(e["params"]["namespace"] for e in live.events if e["params"]["namespace"])
    snapshots = [
        i for i, e in enumerate(live.events) if e["method"] == "values" and e["params"]["namespace"] == subagent
    ]
    # The subagent's latest snapshot is older than the root's: replay starts at its envelope.
    expected = live.events[snapshots[-1] - 1 :]
    assert expected[0]["method"] == "checkpoints" and expected[0]["params"]["namespace"] == subagent
    assert [e["method"] for e in expected[-3:]] == ["checkpoints", "values", "lifecycle"]
    assert fresh.events[0]["event_id"].endswith(":running:0")
    assert [_wire(e) for e in fresh.events[1:]] == [_wire(e) for e in expected]
    assert [e["params"]["data"] for e in fresh.events if e["method"] == "values"] == [
        {"items": ["sub:x", "inner"]},
        {"items": ["sub:x", "sub:x", "inner"]},
    ]


@pytest.mark.asyncio
async def test_a_marker_whose_snapshot_was_replaced_is_skipped(
    stream_server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_snapshots = stream_server.broker.read_snapshots

    async def slow_read(thread_id: str, namespaces: list[str]) -> Any:
        await asyncio.sleep(0.3)  # the run writes newer snapshots meanwhile
        return await read_snapshots(thread_id, namespaces)

    monkeypatch.setattr(stream_server.broker, "read_snapshots", slow_read)
    thread_id = str(uuid4())
    async with httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http:
        live = await _run_to_end(stream_server, http, thread_id, "loop:20:1000:50")

    root = [e for e in live.events if not e["params"]["namespace"] and e["method"] in ("values", "checkpoints")]
    pairs = [(root[i - 1], e) for i, e in enumerate(root) if e["method"] == "values" and i > 0]
    assert pairs and len(pairs) == sum(e["method"] == "values" for e in root)
    for envelope, values in pairs:
        assert envelope["method"] == "checkpoints"
        assert envelope["event_id"].rsplit(":", 1)[0] == values["event_id"].rsplit(":", 1)[0]
        assert envelope["params"]["data"]["step"] == values["params"]["data"].get("step", 0)
    steps = [values["params"]["data"].get("step", 0) for _, values in pairs]
    assert steps == sorted(set(steps))
    assert steps[-1] == 20
    assert len(steps) < 21  # some markers were skipped: their snapshot had been replaced


@pytest.mark.asyncio
async def test_retention_keeps_the_snapshot_that_later_deltas_need(
    stream_server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.event_streaming, "RESUMABLE_STREAM_TTL_SECONDS", 1)
    thread_id = str(uuid4())
    async with httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http:
        # One step streams 300 custom events over about 3 s, three times the retention window.
        await stream_server.start_run(http, thread_id, "tick:300:10")
        await asyncio.sleep(2.5)
        fresh = await _reconnect(http, thread_id, ["values", "custom", "lifecycle"])

    methods = [e["method"] for e in fresh.events]
    first_tick = methods.index("custom")
    assert "values" in methods[:first_tick]
    assert fresh.events[methods.index("values")]["params"]["data"] == {"items": ["tick:300:10"]}
    ticks = [e["params"]["data"]["payload"]["tick"] for e in fresh.events if e["method"] == "custom"]
    assert ticks == list(range(300))


@pytest.mark.asyncio
async def test_retention_trims_by_time_and_expires_the_keys_after_the_last_write(
    stream_server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings.event_streaming, "RESUMABLE_STREAM_TTL_SECONDS", 1)
    thread_id = str(uuid4())
    stream_key = stream_server.broker.stream_key(thread_id)
    async with httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http:
        await _run_to_end(stream_server, http, thread_id, "tick:250:0")
        oldest = stream_server.redis.xrange(stream_key, count=1)[0][0]
        await asyncio.sleep(0.6)
        await _run_to_end(stream_server, http, thread_id, "tick:300:5")
        keys = list(stream_server.redis.scan_iter(f"{stream_server.prefix}thread:{{{thread_id}}}:*"))
        kept = stream_server.redis.xrange(stream_key, count=1)[0][0]
        ttls = [stream_server.redis.pttl(key) for key in keys]
        await asyncio.sleep(1.2)
        left = list(stream_server.redis.scan_iter(f"{stream_server.prefix}thread:{{{thread_id}}}:*"))

    assert kept != oldest  # the first run's events outlived the window and were trimmed
    assert len(keys) == 3 and all(0 < ttl <= 1000 for ttl in ttls)
    assert left == []


@pytest.mark.asyncio
async def test_redis_holds_one_full_state_per_namespace(stream_server: _Server) -> None:
    thread_id = str(uuid4())
    async with httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http:
        await _run_to_end(stream_server, http, thread_id, "loop:8:20000")
        await _run_to_end(stream_server, http, thread_id, "sub:x")

    held = _thread_values(stream_server)
    assert sum("x" * 20000 in value for value in held) == 1  # the root's latest values only
    assert sum('"inner"' in value for value in held) == 2  # the subagent's latest values, and the root's


@pytest.mark.asyncio
async def test_reconnect_bytes_do_not_grow_with_the_number_of_steps(stream_server: _Server) -> None:
    sizes: dict[int, int] = {}
    async with httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http:
        for steps in (5, 10):
            thread_id = str(uuid4())
            await _run_to_end(stream_server, http, thread_id, f"loop:{steps}:20000")
            sizes[steps] = (await _reconnect(http, thread_id)).bytes

    assert sizes[5] < 2 * 20000  # one state, not one per step
    assert sizes[10] < sizes[5] * 1.1


@pytest.mark.asyncio
async def test_interrupt_reaches_a_client_that_reconnects(stream_server: _Server) -> None:
    thread_id = str(uuid4())
    async with httpx.AsyncClient(base_url=stream_server.base_url, timeout=None) as http:
        await _run_to_end(stream_server, http, thread_id, "ask:x")
        fresh = await _reconnect(http, thread_id, ["values", "input", "lifecycle"])

    requests = [e["params"]["data"] for e in fresh.events if e["method"] == "input.requested"]
    assert [request["payload"] for request in requests] == ["question?"]
    assert requests[0]["interrupt_id"]
    assert [status for status, _ in _root_lifecycle(fresh)] == ["running", "interrupted"]
