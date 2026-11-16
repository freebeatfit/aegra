"""Integration test for SSE keepalive on the thread event stream.

Regression cover for a production bug where agents that opened upstream
WebSockets (e.g. Sberdevices voice gateway) and held them silently for
~60s were softly interrupted by Aegra: the SSE response had no keepalive,
so an idle proxy dropped the HTTP connection. ``sse_starlette``'s periodic
``: heartbeat`` comment keeps it open.

This exercises the wiring the v2 route uses: ``ThreadEventSession`` →
``_frame_events`` → ``sse_to_bytes`` → ``EventSourceResponse``.
"""

import asyncio
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from sse_starlette import EventSourceResponse

from aegra_api.api.event_streaming import _frame_events
from aegra_api.core.sse import get_sse_headers, heartbeat_factory, sse_to_bytes
from aegra_api.services.broker import BrokerManager, RunBroker
from aegra_api.services.event_streaming import session as session_module
from aegra_api.services.event_streaming.session import ThreadEventSession


@pytest.fixture
def run_id() -> str:
    return str(uuid4())


@pytest.fixture
def local_broker_manager(monkeypatch: pytest.MonkeyPatch) -> BrokerManager:
    """Fresh BrokerManager patched into the thread session module."""
    manager = BrokerManager()
    monkeypatch.setattr(session_module, "broker_manager", manager)
    return manager


@pytest.mark.asyncio
async def test_keepalive_pings_during_silent_run(run_id: str, local_broker_manager: BrokerManager) -> None:
    """A silent run must still produce ping bytes so idle proxies don't drop us.

    Without sse-starlette's ``_ping`` task, a proxy with a 60s idle timeout
    closes the HTTP connection when the graph node blocks on an upstream
    WebSocket without emitting events — the exact prod scenario this fix
    addresses.
    """
    broker = RunBroker(run_id)
    local_broker_manager._brokers[run_id] = broker

    async def _list_runs() -> list[tuple[str, str, str, None]]:
        return [(run_id, "running", "graph", None)]

    session = ThreadEventSession("thread-1", channels={"values"}, list_run_ids=_list_runs, max_idle_seconds=0.1)
    app = FastAPI()

    @app.get("/stream")
    async def _stream() -> EventSourceResponse:
        return EventSourceResponse(
            sse_to_bytes(_frame_events(session)),
            ping=1,  # fast ping for test speed
            ping_message_factory=heartbeat_factory,
            headers=get_sse_headers(),
        )

    collected = bytearray()

    async def _collect() -> None:
        transport = httpx.ASGITransport(app=app)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://test") as client,
            client.stream("GET", "/stream", timeout=12.0) as resp,
        ):
            assert resp.status_code == 200
            async for chunk in resp.aiter_bytes():
                collected.extend(chunk)

    async def _drive_broker() -> None:
        # Stay silent long enough that at least two ping intervals fire even
        # when ASGI startup eats a noticeable slice at the beginning (CI can
        # be slow). ping=1, sleep=4.0 → ~3 expected heartbeats, asserting >=2
        # leaves ~2s of safety margin.
        await asyncio.sleep(4.0)
        await broker.put(f"{run_id}_event_1", ("end", {"status": "success"}))

    await asyncio.gather(_collect(), _drive_broker())

    body = bytes(collected)
    heartbeat_hits = body.count(b": heartbeat")
    assert heartbeat_hits >= 2, f"Expected >= 2 heartbeat comments during silence, got {heartbeat_hits}. Body={body!r}"
