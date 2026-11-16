"""Streaming is Agent Protocol v2 only (FB patch P8).

The v1 run-stream endpoints answer 410 with a pointer to v2. Runs created
through REST keep working and execute on the v2 event path, so the thread
stream carries their events. Those tests drive a real graph through the local
executor against a real PostgreSQL and skip when no migrated database is reachable.
"""

from __future__ import annotations

import asyncio
import json
import operator
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, TypedDict
from uuid import uuid4

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from aegra_api.api import event_streaming as es_module
from aegra_api.api import runs as runs_module
from aegra_api.core import orm
from aegra_api.core.auth_deps import get_current_user, require_auth
from aegra_api.models.auth import User
from aegra_api.services.event_streaming import capabilities as caps
from aegra_api.settings import settings
from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.streams import SessionUntilRunEnds

_V2_POINTER = {
    "commands": "POST /threads/{thread_id}/commands",
    "stream": "POST /threads/{thread_id}/stream/events",
}


class TestV1RunStreamsAreGone:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("POST", "/threads/t1/runs/stream"),
            ("GET", "/threads/t1/runs/r1/stream"),
            ("POST", "/runs/stream"),
        ],
    )
    def test_answers_410_with_v2_pointer(self, method: str, path: str) -> None:
        client = make_client(create_test_app(include_runs=True, include_threads=False))
        body = {"assistant_id": "agent", "input": {"messages": []}} if method == "POST" else None

        resp = client.request(method, path, json=body)

        assert resp.status_code == 410
        payload = resp.json()
        assert payload["error"] == "gone"
        assert payload["details"] == _V2_POINTER


_GRAPH_ID = "v2_only_echo"


class _EchoState(TypedDict):
    items: Annotated[list[str], operator.add]


async def _echo(state: _EchoState) -> dict[str, list[str]]:
    last = state["items"][-1]
    if last == "slow":
        await asyncio.sleep(30)
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


def _skip_without_migrated_database(url: str) -> None:
    try:
        with psycopg.connect(url, connect_timeout=3) as conn:
            row = conn.execute("SELECT to_regclass('public.runs')").fetchone()
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL test database is unavailable: {exc}")
    if row is None or row[0] is None:
        pytest.skip("runs table is unavailable; run Alembic migrations before this DB test")


@pytest.fixture
def rest(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, str]]:
    """A client for the REST run and v2 routes, and an assistant on the echo graph."""
    url = settings.db.database_url_sync
    _skip_without_migrated_database(url)
    user = User(identity=f"v2-only-{uuid4().hex}")
    assistant_id = str(uuid4())
    with psycopg.connect(url) as conn:
        conn.execute(
            "INSERT INTO assistant (assistant_id, name, graph_id, user_id) VALUES (%s, %s, %s, %s)",
            (assistant_id, "v2-only", _GRAPH_ID, user.identity),
        )

    service = _GraphService()
    for module in ("run_preparation", "run_executor", "langgraph_service"):
        monkeypatch.setattr(f"aegra_api.services.{module}.get_langgraph_service", lambda: service)
    # NullPool: TestClient runs the app on its own loop; asyncpg connections can't cross loops.
    engine = create_async_engine(settings.db.database_url, poolclass=NullPool)
    monkeypatch.setattr(orm, "async_session_maker", async_sessionmaker(engine, expire_on_commit=False))
    monkeypatch.setattr(caps.settings.event_streaming, "FF_V2_EVENT_STREAMING", True)
    # The thread stream never ends on its own; TestClient buffers the whole body.
    monkeypatch.setattr(es_module, "ThreadEventSession", SessionUntilRunEnds)

    app = FastAPI()
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_current_user] = lambda: user
    app.include_router(runs_module.router)
    app.include_router(es_module.router)
    try:
        with TestClient(app) as client:
            yield client, assistant_id
    finally:
        with psycopg.connect(url) as conn:
            conn.execute("DELETE FROM runs WHERE user_id = %s", (user.identity,))
            conn.execute("DELETE FROM thread WHERE user_id = %s", (user.identity,))
            conn.execute("DELETE FROM assistant WHERE assistant_id = %s", (assistant_id,))


def _thread_events(client: TestClient, thread_id: str) -> list[dict[str, Any]]:
    with client.stream(
        "POST", f"/threads/{thread_id}/stream/events", json={"channels": ["values", "lifecycle"]}
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    return [
        json.loads(line.removeprefix("data: "))
        for frame in body.split("\n\n")
        for line in frame.splitlines()
        if line.startswith("data: ")
    ]


class TestRestRunsOnV2EventPath:
    def test_rest_run_emits_v2_events_on_thread_stream(self, rest: tuple[TestClient, str]) -> None:
        client, assistant_id = rest
        thread_id = str(uuid4())

        created = client.post(
            f"/threads/{thread_id}/runs", json={"assistant_id": assistant_id, "input": {"items": ["hi"]}}
        )
        assert created.status_code == 200
        run_id = created.json()["run_id"]
        joined = client.get(f"/threads/{thread_id}/runs/{run_id}/join")
        events = _thread_events(client, thread_id)

        assert json.loads(joined.text) == {"items": ["hi", "echo:hi"]}
        assert all(event["type"] == "event" for event in events)
        values = [event["params"]["data"] for event in events if event["method"] == "values"]
        assert values[-1] == {"items": ["hi", "echo:hi"]}
        root_lifecycle = [
            (event["params"]["data"]["event"], event["event_id"])
            for event in events
            if event["method"] == "lifecycle" and event["params"]["namespace"] == []
        ]
        assert [status for status, _ in root_lifecycle] == ["running", "completed"]
        assert all(event_id.startswith(f"synth:{run_id}:") for _, event_id in root_lifecycle)

    def test_rest_run_operations_keep_working(self, rest: tuple[TestClient, str]) -> None:
        client, assistant_id = rest
        thread_id = str(uuid4())

        waited = client.post(
            f"/threads/{thread_id}/runs/wait", json={"assistant_id": assistant_id, "input": {"items": ["a"]}}
        )
        listed = client.get(f"/threads/{thread_id}/runs")
        slow = client.post(
            f"/threads/{thread_id}/runs", json={"assistant_id": assistant_id, "input": {"items": ["slow"]}}
        )
        slow_id = slow.json()["run_id"]
        cancelled = client.post(f"/threads/{thread_id}/runs/{slow_id}/cancel", params={"wait": 1})
        deleted = client.delete(f"/threads/{thread_id}/runs/{slow_id}")

        assert json.loads(waited.text) == {"items": ["a", "echo:a"]}
        assert [run["status"] for run in listed.json()] == ["success"]
        done = client.get(f"/threads/{thread_id}/runs/{listed.json()[0]['run_id']}").json()
        assert done["status"] == "success"
        assert done["output"] is None  # v2 runs keep their state in the checkpoint only
        assert cancelled.json()["status"] == "interrupted"
        assert deleted.status_code == 204
        assert client.get(f"/threads/{thread_id}/runs/{slow_id}").status_code == 404

    def test_rest_interrupt_before_pauses_the_run(self, rest: tuple[TestClient, str]) -> None:
        client, assistant_id = rest
        thread_id = str(uuid4())

        waited = client.post(
            f"/threads/{thread_id}/runs/wait",
            json={"assistant_id": assistant_id, "input": {"items": ["x"]}, "interrupt_before": ["echo"]},
        )
        run = client.get(f"/threads/{thread_id}/runs").json()[0]

        assert json.loads(waited.text) == {"items": ["x"]}
        assert run["status"] == "interrupted"
