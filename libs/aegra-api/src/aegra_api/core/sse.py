"""Server-Sent Events utilities and formatting"""

import contextlib
import json
import re
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, MutableMapping
from typing import Any

from sse_starlette import EventSourceResponse, ServerSentEvent

from aegra_api.core.serializers import GeneralSerializer
from aegra_api.settings import settings

# Global serializer instance
_serializer = GeneralSerializer()

# Cached SSE keepalive payload: ``: heartbeat\r\n\r\n`` (15 bytes).
# Matches langgraph-api's wire-format so tcpdump/logs line up with LangGraph
# Platform, and avoids per-tick datetime formatting that sse-starlette's
# default ping does.
_HEARTBEAT_EVENT = ServerSentEvent(comment="heartbeat")


def heartbeat_factory() -> ServerSentEvent:
    """Ping factory for ``EventSourceResponse(ping_message_factory=...)``.

    Returns the same cached ``ServerSentEvent`` on every call — the encoded
    payload is identical on every tick, so there's no reason to allocate.
    """
    return _HEARTBEAT_EVENT


# Some LLMs stream tool_call_chunks.args with literal \uXXXX sequences
# instead of actual Unicode characters. After json.dumps these become \\uXXXX (double-escaped).
# We decode them back in two passes: surrogate pairs first to avoid lone surrogates that
# cannot be encoded to UTF-8, then remaining non-ASCII, non-surrogate code points.
# ASCII control characters (< 0x80) are left intact to preserve JSON validity.
_SURROGATE_PAIR_RE = re.compile(
    r"\\\\u(D[89AB][0-9a-fA-F]{2})\\\\u(D[C-F][0-9a-fA-F]{2})",
    re.IGNORECASE,
)
_NONASCII_ESCAPE_RE = re.compile(r"\\\\u([0-9a-fA-F]{4})")


def _decode_literal_unicode_escapes(data_str: str) -> str:
    """Decode double-escaped \\uXXXX sequences in an already-JSON-encoded string."""
    if "\\u" not in data_str:
        return data_str
    # First pass: combine surrogate pairs (e.g. \\uD83D\\uDE00 → 😀)
    data_str = _SURROGATE_PAIR_RE.sub(
        lambda m: chr(((int(m.group(1), 16) - 0xD800) << 10) + (int(m.group(2), 16) - 0xDC00) + 0x10000),
        data_str,
    )
    # Second pass: decode remaining non-ASCII, non-surrogate escapes
    return _NONASCII_ESCAPE_RE.sub(
        lambda m: chr(cp) if (cp := int(m.group(1), 16)) >= 0x80 and not (0xD800 <= cp <= 0xDFFF) else m.group(0),
        data_str,
    )


def get_sse_headers() -> dict[str, str]:
    """Get standard SSE headers"""
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Last-Event-ID",
    }


async def sse_to_bytes(inner: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
    """Adapt an ``AsyncGenerator[str]`` of pre-formatted SSE messages to bytes.

    ``sse_starlette.EventSourceResponse`` wraps plain strings as *new* SSE
    events (double-encoding them). Pass bytes through its iterator so our
    already-formatted messages reach the wire untouched.

    Wraps ``inner`` in ``contextlib.aclosing`` so its ``finally`` blocks
    (broker cleanup, replay-buffer release) fire deterministically even
    when sse-starlette aborts us mid-stream on cancel.
    """
    async with contextlib.aclosing(inner) as managed:
        async for chunk in managed:
            yield chunk.encode("utf-8")


def make_sse_response(
    body: AsyncIterator[bytes],
    *,
    headers: Mapping[str, str],
    close_handler: Callable[[MutableMapping[str, Any]], Awaitable[None]] | None = None,
    status_code: int = 200,
) -> EventSourceResponse:
    """Construct an ``EventSourceResponse`` with our shared SSE defaults.

    Centralizes ping interval + heartbeat factory so every SSE endpoint
    emits identical wire-format keepalives. ``settings`` is read at call
    time so live overrides (e.g. tests, env reloads) take effect.
    """
    return EventSourceResponse(
        body,
        status_code=status_code,
        ping=settings.app.sse_ping_interval_secs,
        ping_message_factory=heartbeat_factory,
        client_close_handler_callable=close_handler,
        headers=dict(headers),
    )


def format_sse_message(
    event: str,
    data: Any,
    event_id: str | None = None,
    serializer: Callable[[Any], Any] | None = None,
) -> str:
    """Format a message as Server-Sent Event following SSE standard

    Args:
        event: SSE event type
        data: Data to serialize and send
        event_id: Optional event ID
        serializer: Optional custom serializer function
    """
    lines = []

    lines.append(f"event: {event}")

    # Convert data to JSON string
    if data is None:
        data_str = ""
    else:
        # Use our general serializer by default to handle complex objects
        default_serializer = serializer or _serializer.serialize
        data_str = json.dumps(data, default=default_serializer, separators=(",", ":"), ensure_ascii=False)
        data_str = _decode_literal_unicode_escapes(data_str)

    lines.append(f"data: {data_str}")

    if event_id:
        lines.append(f"id: {event_id}")

    lines.append("")  # Empty line to end the event

    return "\n".join(lines) + "\n"
