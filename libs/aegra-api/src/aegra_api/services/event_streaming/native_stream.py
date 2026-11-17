"""Drive a run through langgraph's native v3 stream for Agent Protocol v2.

``astream_events(version="v3")`` emits events already in protocol shape —
``{type:"event", method, params:{data, namespace, ...}}`` — including the
content-block message lifecycle, tool-call blocks, token usage, and
``params.interrupts``. We forward those into the broker verbatim; the
session restamps seq/event_id and filters by channel. Every run streams
through here, whichever route created it.

The only reshaping here: ``messages`` events arrive as ``[event, metadata]``
tuples, so we unwrap element 0 into ``params.data``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from aegra_api.models.runs import Durability


@lru_cache(maxsize=1)
def _extra_transformers() -> list[Any]:
    """Transformers Aegra adds to every v2 run: only the checkpoint envelope.

    The LangGraph Agent Server adds no transformer of its own; Updates, Custom,
    Tasks and Debug are opt-in because their stream modes are expensive
    (langgraph#7640). A graph that wants them declares them itself
    (``compile(transformers=...)`` or ``AgentMiddleware.transformers``).

    ``checkpoints`` is the one protocol channel the SDK reads for timeline
    order, and the protocol defines it as a lightweight envelope, so Aegra
    supplies that envelope rather than LangGraph's full-state snapshot.
    """
    # Optional-dep guard: a too-old langgraph should fail the capability probe,
    # but if it's bypassed surface a clean error, not a raw ImportError mid-run.
    try:
        from langgraph.stream.transformers import StreamTransformer
    except ImportError as exc:
        raise RuntimeError("langgraph.stream.transformers unavailable; run the v2 capability probe first.") from exc

    class CheckpointEnvelopeTransformer(StreamTransformer):
        """Rewrite each ``checkpoints`` event to the protocol envelope in place.

        LangGraph's ``checkpoints`` stream mode yields a ``get_state()``-shaped
        snapshot. The protocol's ``checkpoints`` channel carries only
        ``{id, parent_id?, step, source}``; full state is the ``values``
        channel's job. Rewriting in ``process`` runs before the event reaches
        the main log, so the snapshot never reaches the broker.
        """

        required_stream_modes = ("checkpoints",)
        before_builtins = True

        def init(self) -> dict[str, Any]:
            return {}

        def process(self, event: Any) -> bool:
            if event.get("method") != "checkpoints":
                return True
            params = event.get("params")
            if isinstance(params, dict):
                params["data"] = checkpoint_envelope(params.get("data"))
            return True

    return [CheckpointEnvelopeTransformer]


def checkpoint_envelope(snapshot: Any) -> dict[str, Any]:
    """The protocol ``checkpoints`` envelope for one checkpoint snapshot.

    ``snapshot`` is LangGraph's ``checkpoints`` payload
    (``{config, parent_config, metadata, values, ...}``); an already-reduced
    envelope passes through unchanged.
    """
    if not isinstance(snapshot, dict):
        return {}
    if "config" not in snapshot and isinstance(snapshot.get("id"), str):
        return snapshot
    config = snapshot.get("config") or {}
    parent = snapshot.get("parent_config") or {}
    metadata = snapshot.get("metadata") or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    parent_configurable = parent.get("configurable", {}) if isinstance(parent, dict) else {}
    envelope: dict[str, Any] = {"id": configurable.get("checkpoint_id")}
    if parent_id := parent_configurable.get("checkpoint_id"):
        envelope["parent_id"] = parent_id
    if isinstance(metadata, dict):
        if isinstance(metadata.get("step"), int):
            envelope["step"] = metadata["step"]
        if isinstance(metadata.get("source"), str):
            envelope["source"] = metadata["source"]
    return envelope


def unwrap_message_event(data: Any) -> dict[str, Any] | None:
    """Return the content-block event dict from a v3 ``messages`` payload.

    v3 ``messages`` data is ``[event_dict, metadata]`` where the event dict
    carries the ``event`` discriminator (``message-start`` etc.). Accepts a
    bare event dict too. Returns ``None`` for anything not v2-message-shaped.
    """
    if isinstance(data, dict) and isinstance(data.get("event"), str):
        return data
    if isinstance(data, (list, tuple)) and len(data) == 2:
        head = data[0]
        if isinstance(head, dict) and isinstance(head.get("event"), str):
            return head
    return None


async def stream_native_v3_events(
    *,
    graph: Any,
    input_data: Any,
    config: dict[str, Any],
    context: dict[str, Any] | None = None,
    durability: Durability | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Yield ``(method, protocol_event)`` pairs from a native v3 run.

    Each pair goes into the broker as one raw event; the session re-envelopes
    it. ``messages`` payloads are unwrapped to their event dict; a message
    event we can't reconstruct is dropped rather than forwarded malformed.
    """
    # v3 forwards extra kwargs to astream; omitted when unset to keep LangGraph's default.
    durability_kwargs: dict[str, Durability] = {"durability": durability} if durability is not None else {}
    config, breakpoints = _split_breakpoints(config)
    run_stream = await graph.astream_events(
        input_data,
        config,
        version="v3",
        context=context,
        transformers=_extra_transformers(),
        **breakpoints,
        **durability_kwargs,
    )
    # LangGraph Python emits values then their checkpoint; the protocol and SDK expect the
    # envelope first, so hold one values event until we see whether its checkpoint follows.
    held: tuple[str, dict[str, Any]] | None = None
    async with run_stream as stream:
        async for event in stream:
            if not isinstance(event, dict) or event.get("type") != "event":
                continue
            method = event.get("method")
            if not isinstance(method, str):
                continue

            if method == "messages":
                unwrapped = unwrap_message_event(event.get("params", {}).get("data"))
                if unwrapped is None:
                    continue
                event = {**event, "params": {**event["params"], "data": unwrapped}}

            if held is not None:
                if method == "checkpoints" and _namespace(event) == _namespace(held[1]):
                    yield method, event
                    yield held
                    held = None
                    continue
                yield held
                held = None
            if method == "values":
                held = (method, event)
                continue
            yield method, event
    if held is not None:
        yield held


def _split_breakpoints(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move ``interrupt_before``/``interrupt_after`` out of the config into stream kwargs.

    LangGraph reads breakpoints only as arguments; in the config they are ignored.
    """
    config = dict(config)
    breakpoints: dict[str, Any] = {}
    for key in ("interrupt_before", "interrupt_after"):
        value = config.pop(key, None)
        if value is not None:
            breakpoints[key] = "*" if value == ["*"] else value
    return config, breakpoints


def _namespace(event: dict[str, Any]) -> list[Any]:
    params = event.get("params")
    namespace = params.get("namespace") if isinstance(params, dict) else None
    return namespace if isinstance(namespace, list) else []
