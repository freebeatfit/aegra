"""SSE lifetime of the thread event stream: heartbeats, the send timeout, the error event.

Regression cover for a production bug where agents that opened upstream
WebSockets (e.g. Sberdevices voice gateway) and held them silently for
~60s were softly interrupted by Aegra: the SSE response had no keepalive,
so an idle proxy dropped the HTTP connection. ``sse_starlette``'s periodic
``: heartbeat`` comment keeps it open.

This exercises the wiring the v2 route uses: ``ThreadEventSession`` →
``_frame_events`` → ``sse_to_bytes`` → ``make_sse_response``.
"""

import asyncio
import json
from collections.abc import Iterator
from uuid import uuid4

import pytest
from sse_starlette import EventSourceResponse
from starlette.types import Message

from aegra_api.api.event_streaming import _frame_events
from aegra_api.core import sse as sse_module
from aegra_api.core.sse import get_sse_headers, make_sse_response, sse_to_bytes
from aegra_api.services.base_broker import ThreadEntry
from aegra_api.services.broker import BrokerManager
from aegra_api.services.event_streaming import session as session_module
from aegra_api.services.event_streaming.session import ThreadEventSession, VisibleRun
from aegra_api.settings import settings
from tests.fixtures.streams import drive_sse


@pytest.fixture
def local_broker_manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[BrokerManager]:
    """Fresh BrokerManager patched into the thread session module, with a 1 s heartbeat."""
    manager = BrokerManager()
    monkeypatch.setattr(session_module, "broker_manager", manager)
    monkeypatch.setattr(settings.app, "KEEPALIVE_INTERVAL_SECS", 1)
    yield manager


async def _visible(_run_id: str) -> VisibleRun:
    return VisibleRun(graph="graph")


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


@pytest.mark.asyncio
async def test_idle_stream_whose_client_stopped_reading_ends_at_the_send_timeout(
    local_broker_manager: BrokerManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On an idle stream the heartbeat is the only send, so it must time out too."""
    monkeypatch.setattr(sse_module, "SEND_TIMEOUT_SECS", 0.5)

    def stalled(message: Message) -> float:
        return 3600 if message["type"] == "http.response.body" else 0

    body = await drive_sse(_response(str(uuid4())), stop=lambda _b: False, timeout=5, send_delay=stalled)

    assert body == b""


@pytest.mark.asyncio
async def test_failed_read_sends_the_error_event_and_ends(
    local_broker_manager: BrokerManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_read(_thread_id: str, _after: str) -> list[ThreadEntry]:
        raise ConnectionError("Redis went away")

    monkeypatch.setattr(local_broker_manager, "read_thread", failing_read)

    body = await drive_sse(_response(str(uuid4())), stop=lambda _b: False, timeout=5)

    frames = [frame for frame in body.decode().split("\n\n") if frame.startswith("event: ")]
    assert len(frames) == 1
    assert frames[0].startswith("event: error\n")
    data = json.loads(frames[0].split("data: ", 1)[1])
    # Fixed text: backend errors never reach the client.
    assert data == {"type": "error", "id": None, "error": "unknown_error", "message": "Thread event stream failed."}
