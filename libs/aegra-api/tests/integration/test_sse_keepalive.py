"""SSE keepalive on the thread event stream.

Regression cover for a production bug where agents that opened upstream
WebSockets (e.g. Sberdevices voice gateway) and held them silently for
~60s were softly interrupted by Aegra: the SSE response had no keepalive,
so an idle proxy dropped the HTTP connection. ``sse_starlette``'s periodic
``: heartbeat`` comment keeps it open.

This exercises the wiring the v2 route uses: ``ThreadEventSession`` →
``_frame_events`` → ``sse_to_bytes`` → ``make_sse_response``.
"""

import asyncio
from collections.abc import Iterator
from uuid import uuid4

import pytest
from sse_starlette import EventSourceResponse

from aegra_api.api.event_streaming import _frame_events
from aegra_api.core.sse import get_sse_headers, make_sse_response, sse_to_bytes
from aegra_api.services.broker import BrokerManager
from aegra_api.services.event_streaming import session as session_module
from aegra_api.services.event_streaming.session import ThreadEventSession
from aegra_api.settings import settings
from tests.fixtures.streams import drive_sse


@pytest.fixture
def local_broker_manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[BrokerManager]:
    """Fresh BrokerManager patched into the thread session module, with a 1 s heartbeat."""
    manager = BrokerManager()
    monkeypatch.setattr(session_module, "broker_manager", manager)
    monkeypatch.setattr(settings.app, "KEEPALIVE_INTERVAL_SECS", 1)
    yield manager


async def _visible(_run_id: str) -> str:
    return "graph"


def _response(thread_id: str) -> EventSourceResponse:
    session = ThreadEventSession(thread_id, channels={"lifecycle"}, lookup_run=_visible)
    return make_sse_response(sse_to_bytes(_frame_events(session)), headers=get_sse_headers())


@pytest.mark.asyncio
async def test_keepalive_pings_during_silent_run(local_broker_manager: BrokerManager) -> None:
    """A silent run must still produce ping bytes so idle proxies don't drop us."""
    thread_id = str(uuid4())

    async def end_run_after_silence() -> None:
        # ping=1, sleep=3.5 → ~3 heartbeats; asserting >= 2 leaves margin for a slow CI.
        await asyncio.sleep(3.5)
        await local_broker_manager.append(thread_id, "run-1", "w1", ("end", {"status": "success"}))

    writer = asyncio.create_task(end_run_after_silence())
    body = await drive_sse(_response(thread_id), stop=lambda b: b'"completed"' in b, timeout=10)
    await writer

    heartbeat_hits = body.count(b": heartbeat")
    assert heartbeat_hits >= 2, f"Expected >= 2 heartbeat comments during silence, got {heartbeat_hits}. Body={body!r}"
