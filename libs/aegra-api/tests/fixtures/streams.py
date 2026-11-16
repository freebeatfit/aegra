"""Helpers for tests of the thread event stream, which never ends on its own."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

from starlette.responses import Response
from starlette.types import Message

from aegra_api.services.event_streaming.session import ThreadEventSession


def is_root_terminal(envelope: dict[str, Any]) -> bool:
    """True for a run's root lifecycle terminal (completed, failed, interrupted)."""
    return (
        envelope["method"] == "lifecycle"
        and not envelope["params"]["namespace"]
        and envelope["params"]["data"].get("event") != "running"
    )


class SessionUntilRunEnds(ThreadEventSession):
    """A thread session that stops after the first run's root terminal, so a buffered client returns."""

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        async for envelope in super().stream():
            yield envelope
            if is_root_terminal(envelope):
                return


async def drive_sse(
    response: Response,
    *,
    stop: Callable[[bytes], bool],
    timeout: float,
    send_delay: Callable[[Message], float] | None = None,
) -> bytes:
    """Run an SSE response as an ASGI app; the client disconnects once ``stop(body)`` holds.

    ``send_delay`` makes a send wait, as a client that stopped reading does.
    """
    body = bytearray()
    disconnected = asyncio.Event()

    async def receive() -> Message:
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if send_delay is not None:
            await asyncio.sleep(send_delay(message))
        if message["type"] == "http.response.body":
            body.extend(message.get("body", b""))
            if stop(bytes(body)):
                disconnected.set()

    scope = {"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""}
    await asyncio.wait_for(response(scope, receive, send), timeout)
    return bytes(body)
