"""Thread-scoped session that forwards a thread's native v3 events as v2 events.

A v2 stream is scoped to a *thread*, not a run (matching the LangGraph SDK):
the client opens the stream, issues ``run.start``, and the events of whatever
run(s) execute on the thread flow through. The run's broker already holds
native protocol events (the v3 producer wrote them) as ``(method, event)``
pairs, plus a terminal ``("end"|"error", payload)``. The session re-envelopes
each under one thread-monotonic ``seq``: it restamps seq/event_id, filters by
channel, splits interrupts onto the ``input`` channel, normalizes state
message payloads, and derives the terminal lifecycle event.

``seq`` is the reconnect cursor: a client resends the last ``seq`` it saw as
``since`` and the session skips anything at or below it. ``event_id`` is a
distinct unique-per-event string used by the client for dedup.

What a connection replays (aligned with the LangGraph Agent Server, which
replays the run in flight and leaves settled history to ``getState()``):

* an active run (``pending``/``running``): its latest values snapshot and
  every delta after it, then live events;
* settled runs: only the newest one, compactly (its last state and how it
  ended), so a client that subscribed late still sees the terminal and the
  state it closed on; older settled runs are not read at all;
* a ``since`` cursor at or past a run's last event skips that run's broker.

The connection stays open with heartbeats until the client leaves. New runs
are found from a thread notice (``broker_manager.notify_thread``) with a slow
fallback read, not by polling the database.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from aegra_api.services.broker import broker_manager
from aegra_api.services.event_streaming.channels import is_supported_channel
from aegra_api.services.event_streaming.normalizers import (
    lifecycle_status,
    normalize_input_requested,
    normalize_state_payload,
    normalize_updates,
    strip_interrupts,
)
from aegra_api.services.event_streaming.protocol import build_event
from aegra_api.services.replay_policy import is_after
from aegra_api.utils import extract_event_sequence

__all__ = ["RunLister", "ThreadEventSession", "validate_channels"]

# Fallback re-read of the thread's runs when no notice arrives (a lost publish,
# a run created by an older instance). Notices make new runs visible at once.
_RUN_DISCOVERY_FALLBACK_SECONDS = 5.0
# Settled runs that still have buffered events, newest first, that a
# connection replays compactly. One: the run the client may have just missed.
_SETTLED_RUNS_REPLAYED = 1
_ACTIVE_RUN_STATUSES = frozenset({"pending", "running"})

# Async callable returning the thread's (run_id, status, graph_name, created_at) rows, oldest
# first. Status backstops expired runs (see _drain_run); created_at anchors the run's seq.
RunLister = Callable[[], Awaitable[list[tuple[str, str | None, str | None, datetime | None]]]]

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
# seq slots per broker event: one broker event can project to several wire events.
_SEQ_SLOTS_PER_EVENT = 16
# Larger id suffixes are allocate_event_id's epoch-ms fallback, not a per-run count.
_MAX_EVENT_ORDINAL = 1_000_000_000

_TERMINAL_RUN_STATUSES = frozenset({"success", "error", "timeout", "interrupted"})

# A projected channel event: (wire method, params.data, namespace).
_ChannelEvent = tuple[str, dict[str, Any], list[str]]


class ThreadEventSession:
    """Forwards native v3 events for all runs on a thread, filtered to channels."""

    def __init__(
        self,
        thread_id: str,
        *,
        channels: set[str],
        list_run_ids: RunLister,
        since: int | None = None,
        namespaces: list[list[str]] | None = None,
        depth: int | None = None,
        discovery_fallback_seconds: float = _RUN_DISCOVERY_FALLBACK_SECONDS,
        max_idle_seconds: float | None = None,
    ) -> None:
        self._thread_id = thread_id
        self._channels = channels
        self._list_run_ids = list_run_ids
        self._since = since
        self._namespaces = [tuple(prefix) for prefix in namespaces] if namespaces else None
        self._depth = depth
        self._discovery_fallback = discovery_fallback_seconds
        # None: stay open until the client disconnects (the server default).
        self._max_idle = max_idle_seconds
        self._seq = 0
        self._drained: set[str] = set()
        # One input.requested per interrupt per session — the same interrupt can
        # surface via updates (__interrupt__ node) AND the next values snapshot.
        self._sent_interrupts: set[str] = set()
        # Per-run lifecycle bookkeeping, reset by _drain_run: the run's root
        # graph name and the subgraph namespaces still open (started, no
        # completed/failed yet) so the terminal can cascade-close them.
        self._current_run: str | None = None
        self._run_seq_base: int | None = None
        self._current_graph: str | None = None
        self._open_namespaces: dict[str, list[str]] = {}

    @property
    def applied_through_seq(self) -> int:
        """The highest seq assigned so far (the SDK's initial cursor)."""
        return self._seq

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Yield v2 envelopes for the thread's runs until the client disconnects.

        The stream stays open across the gap between runs so a HITL resume
        (``input.respond`` starts a fresh run on the same thread) is delivered
        on the same SSE connection the SDK holds open; the SSE layer sends
        heartbeats meanwhile. A new run is picked up when its creation notice
        arrives, or at the latest after the fallback interval.
        """
        loop = asyncio.get_running_loop()
        idle_since = loop.time()
        first = True
        async with broker_manager.thread_waiter(self._thread_id) as waiter:
            while True:
                waiter.clear()
                rows = await self._list_run_ids()
                plan = self._plan(rows, first=first)
                first = False
                progressed = False
                for run_id, run_status, graph_name, created_at, settled in plan:
                    async for envelope in self._drain_run(run_id, run_status, graph_name, created_at, settled=settled):
                        progressed = True
                        yield envelope
                    self._drained.add(run_id)
                for row in rows:
                    if row[1] not in _ACTIVE_RUN_STATUSES:
                        self._drained.add(row[0])

                if progressed:
                    idle_since = loop.time()
                    continue
                if self._max_idle is not None and loop.time() - idle_since >= self._max_idle:
                    return
                await waiter.wait(self._discovery_fallback)

    def _plan(
        self, rows: list[tuple[str, str | None, str | None, datetime | None]], *, first: bool
    ) -> list[tuple[str, str | None, str | None, datetime | None, bool]]:
        """Which runs to drain now, oldest first, and whether each is settled.

        Active runs are always drained in full. On the connection's first read,
        the newest settled runs are replayed compactly; settled runs that show
        up later (a run that finished between two reads) are drained in full,
        since this connection never saw them start.
        """
        fresh = [row for row in rows if row[0] not in self._drained]
        settled = [row for row in fresh if row[1] not in _ACTIVE_RUN_STATUSES]
        keep_settled = {row[0] for row in settled[-_SETTLED_RUNS_REPLAYED:]} if first else {row[0] for row in settled}
        plan: list[tuple[str, str | None, str | None, datetime | None, bool]] = []
        for run_id, status, graph_name, created_at in fresh:
            if status in _ACTIVE_RUN_STATUSES:
                plan.append((run_id, status, graph_name, created_at, False))
            elif run_id in keep_settled:
                plan.append((run_id, status, graph_name, created_at, first))
        return plan

    def _cursor_for(self, created_at: datetime | None) -> str | None:
        """The broker event id the client's ``since`` already covers in this run.

        ``seq`` is ``run_base + ordinal * slots (+ index)``, so ``since`` maps
        back to the last ordinal it saw. None replays the run from its start.
        """
        if self._since is None or created_at is None:
            return None
        base = _run_seq_base(created_at)
        if self._since < base:
            return None
        ordinal = (self._since - base) // _SEQ_SLOTS_PER_EVENT
        if ordinal <= 0:
            return None
        return f"{self._current_run}_event_{ordinal}"

    async def _drain_run(
        self,
        run_id: str,
        run_status: str | None,
        graph_name: str | None,
        created_at: datetime | None,
        *,
        settled: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Replay then tail one run's broker, re-enveloping each native event.

        Each run's lifecycle tree is self-contained on the stream: a root
        ``running`` seed opens it, per-subgraph events flow through, and the
        terminal closes any still-open subgraph namespaces before the root
        status (a cancel mid-subgraph otherwise leaves them started forever).

        Replay starts at the client's ``since`` cursor and reads the broker in
        batches. A settled run replays compactly (``replay_settled``).

        The persisted run status backstops runs whose broker events expired
        (replay TTL / cleanup): without it, ``aiter`` on a recreated empty
        broker waits forever for an end event nobody will publish — wedging
        the whole thread stream on its first historical run. A terminal run
        with no surviving events drains silently; one whose events survived
        but whose end frame was lost gets a synthesized terminal.
        """
        broker = broker_manager.get_or_create_broker(run_id)
        seen: set[str] = set()
        self._sent_interrupts = set()
        self._current_run = run_id
        self._current_graph = graph_name
        self._run_seq_base = _run_seq_base(created_at) if created_at is not None else None
        if self._run_seq_base is not None:
            self._seq = max(self._seq, self._run_seq_base)
        self._open_namespaces = {}
        cursor = self._cursor_for(created_at)

        if settled:
            replay: AsyncIterator[tuple[str, Any]] = _aiter_list(await broker.replay_settled(cursor))
        else:
            replay = broker.replay_iter(cursor)

        started = False
        async for event_id, raw_event in replay:
            if not started:
                started = True
                for envelope in self._emit(
                    [("lifecycle", self._root_lifecycle("running"), [])], f"synth:{run_id}:running"
                ):
                    yield envelope
            seen.add(event_id)
            for envelope in self._project(event_id, raw_event):
                yield envelope
            if _is_terminal(raw_event):
                return

        if run_status in _TERMINAL_RUN_STATUSES:
            if not started:
                return
            for envelope in self._project(f"{run_id}:status-end", ("end", {"status": run_status})):
                yield envelope
            return

        if not started:
            for envelope in self._emit([("lifecycle", self._root_lifecycle("running"), [])], f"synth:{run_id}:running"):
                yield envelope

        async for event_id, raw_event in broker.aiter():
            if event_id in seen:
                continue
            if cursor is not None and not is_after(event_id, cursor):
                continue
            for envelope in self._project(event_id, raw_event):
                yield envelope
            if _is_terminal(raw_event):
                return

    def _project(self, event_id: str, raw_event: Any) -> list[dict[str, Any]]:
        """Re-envelope one raw broker event into filtered, seq'd envelopes."""
        method, payload = _unwrap(raw_event)
        if method is None:
            return []
        ordinal = extract_event_sequence(event_id)
        if self._run_seq_base is not None and 0 < ordinal < _MAX_EVENT_ORDINAL:
            # Number from the broker's stored ordinal, so a replay buffer trimmed by Redis keeps each event's seq.
            self._seq = max(self._seq, self._run_seq_base + ordinal * _SEQ_SLOTS_PER_EVENT)
        if method in ("end", "error"):
            # The SDK reads the run of a root lifecycle event from a synth:<run_id>: id.
            event_id = f"synth:{self._current_run}:{method}"
        params = payload.get("params") if isinstance(payload, dict) else None
        timestamp = params.get("timestamp") if isinstance(params, dict) else None
        return self._emit(self._channel_events(method, payload), event_id, timestamp=timestamp)

    def _emit(
        self, channel_events: list[_ChannelEvent], event_id: str, *, timestamp: int | None = None
    ) -> list[dict[str, Any]]:
        """Filter + seq a batch of channel events into wire envelopes."""
        envelopes: list[dict[str, Any]] = []
        for index, (channel, data, namespace) in enumerate(channel_events):
            # seq counts before channel filtering — absolute cursor, so a
            # reconnect with a different channel set still resumes correctly.
            self._seq += 1
            if not self._wants(channel, data):
                continue
            if not self._wants_namespace(namespace):
                continue
            if self._since is not None and self._seq <= self._since:
                continue
            envelopes.append(
                build_event(
                    channel,
                    data,
                    namespace=namespace,
                    seq=self._seq,
                    event_id=f"{event_id}:{index}",
                    timestamp=timestamp,
                )
            )
        return envelopes

    def _channel_events(self, method: str, payload: Any) -> list[_ChannelEvent]:
        """Map one raw broker event to zero or more (channel, data, namespace)."""
        if method in ("end", "error"):
            return self._lifecycle(method, payload)

        # Native producer events: payload is a ProtocolEvent dict.
        params = payload.get("params", {}) if isinstance(payload, dict) else {}
        data = params.get("data")
        namespace = params.get("namespace") or []

        if method == "values":
            return self._values_events(data, params.get("interrupts"), namespace)
        if method == "updates":
            return self._updates_events(data, namespace)
        if method == "lifecycle":
            return self._native_lifecycle_events(data, namespace)
        if method == "custom" or method.startswith("custom:"):
            return _custom_events(method, data, namespace)
        return [(method, data if isinstance(data, dict) else {"value": data}, namespace)]

    def _values_events(self, data: Any, interrupts: Any, namespace: list[str]) -> list[_ChannelEvent]:
        """Split interrupts onto the input channel; forward cleaned, normalized values."""
        requests, cleaned = strip_interrupts(data)
        if isinstance(interrupts, (list, tuple)) and interrupts:
            requests = _dedupe_requests(requests + normalize_input_requested(_coerce_interrupts(interrupts)))
        events = self._input_request_events(requests, namespace)
        if _has_state(cleaned):
            events.append(("values", normalize_state_payload(cleaned), namespace))
        return events

    def _updates_events(self, data: Any, namespace: list[str]) -> list[_ChannelEvent]:
        """Forward updates, splitting any embedded interrupt onto the input channel.

        v3 emits updates as raw ``{node: values}``; an interrupt arrives as the
        ``__interrupt__`` node whose values are the interrupt array. A sibling
        node's update in the same chunk (parallel branches) must still forward,
        and an interrupt nested inside a node's values must still surface.
        """
        events: list[_ChannelEvent] = []
        if isinstance(data, dict) and "__interrupt__" in data:
            events.extend(self._input_request_events(normalize_input_requested(data["__interrupt__"]), namespace))
            data = {key: value for key, value in data.items() if key != "__interrupt__"}
            if not data:
                return events
        if isinstance(data, dict):
            for node_values in data.values():
                if isinstance(node_values, dict) and "__interrupt__" in node_values:
                    events.extend(
                        self._input_request_events(normalize_input_requested(node_values["__interrupt__"]), namespace)
                    )
        normalized = normalize_updates(data)
        normalized["values"] = normalize_state_payload(normalized["values"])
        events.append(("updates", normalized, namespace))
        return events

    def _input_request_events(self, requests: list[dict[str, Any]], namespace: list[str]) -> list[_ChannelEvent]:
        """input.requested events for *requests*, at most once per interrupt id."""
        events: list[_ChannelEvent] = []
        for request in requests:
            interrupt_id = request.get("interrupt_id")
            if isinstance(interrupt_id, str):
                if interrupt_id in self._sent_interrupts:
                    continue
                self._sent_interrupts.add(interrupt_id)
            events.append(("input.requested", request, namespace))
        return events

    def _native_lifecycle_events(self, data: Any, namespace: list[str]) -> list[_ChannelEvent]:
        """Forward a per-subgraph lifecycle event from the native producer.

        The producer's ``LifecycleTransformer`` emits one flat lifecycle stream
        at the root scope (``params.namespace == []``) while the announced
        subgraph identity lives in ``data.namespace``. Promote that deeper
        namespace onto the wire so ``started``/``completed``/``failed`` land on
        the subgraph, not the root. Root-scoped lifecycle is owned by the
        terminal ``_lifecycle`` (it resolves interrupt/failure), so drop it here.
        """
        if not isinstance(data, dict) or not isinstance(data.get("event"), str):
            return [("lifecycle", data if isinstance(data, dict) else {"value": data}, namespace)]

        data_ns = data.get("namespace")
        if isinstance(data_ns, list) and all(isinstance(seg, str) for seg in data_ns) and len(data_ns) > len(namespace):
            namespace = list(data_ns)

        if not namespace:
            return []

        event = data["event"]
        key = "\0".join(namespace)
        if event == "started":
            self._open_namespaces[key] = list(namespace)
        elif event in ("completed", "failed"):
            self._open_namespaces.pop(key, None)

        forwarded: dict[str, Any] = {"event": event}
        for field in ("graph_name", "trigger_call_id", "error", "cause"):
            if field in data:
                forwarded[field] = data[field]
        return [("lifecycle", forwarded, namespace)]

    def _lifecycle(self, method: str, payload: Any) -> list[_ChannelEvent]:
        """Build the run's terminal lifecycle from a terminal broker payload.

        Cascade-closes any still-open subgraph namespaces (deepest first) before
        the root status — a cancel or crash mid-subgraph never sends the
        producer's ``completed``, and clients must not see subgraphs stuck in
        ``started`` after the run ended.

        The ``error`` broker event carries ``{error, message}`` with no status —
        it is terminal and drains the run before the trailing ``end`` event, so
        the failed status comes from the method, not a status key.
        """
        status = payload.get("status") if isinstance(payload, dict) else None
        if method == "error":
            status = "error"

        events: list[_ChannelEvent] = []
        for namespace in sorted(self._open_namespaces.values(), key=_namespace_depth, reverse=True):
            events.append(("lifecycle", {"event": "completed"}, namespace))
        self._open_namespaces = {}

        data = self._root_lifecycle(lifecycle_status(status or ""))
        if isinstance(payload, dict) and (message := payload.get("message")):
            data["error"] = message
        events.append(("lifecycle", data, []))
        return events

    def _root_lifecycle(self, status: str) -> dict[str, Any]:
        """Root-namespace lifecycle data, carrying the run's graph name when known."""
        data: dict[str, Any] = {"event": status}
        if self._current_graph:
            data["graph_name"] = self._current_graph
        return data

    def _wants(self, channel: str, data: dict[str, Any] | None = None) -> bool:
        """True if the client subscribed to this channel.

        The ``input.requested`` wire method maps to the ``input`` channel.
        Custom events: a plain ``custom`` subscription receives everything; a
        ``custom:<name>`` subscription receives only events whose ``data.name``
        matches.
        """
        if channel == "input.requested":
            return "input" in self._channels
        if channel == "custom":
            if "custom" in self._channels:
                return True
            name = data.get("name") if isinstance(data, dict) else None
            return isinstance(name, str) and f"custom:{name}" in self._channels
        return channel in self._channels

    def _wants_namespace(self, namespace: list[str]) -> bool:
        """True if the event's namespace passes the subgraph and depth filters.

        Thread-level events (empty namespace) always pass — lifecycle and other
        terminal signals are not subgraph-scoped. ``namespaces`` is a prefix
        include-list; ``depth`` caps subgraph nesting.
        """
        if not namespace:
            return True
        if self._depth is not None and len(namespace) > self._depth:
            return False
        if self._namespaces is None:
            return True
        return any(_prefix_matches(namespace, prefix) for prefix in self._namespaces)


async def _aiter_list(items: list[tuple[str, Any]]) -> AsyncIterator[tuple[str, Any]]:
    for item in items:
        yield item


def _run_seq_base(created_at: datetime) -> int:
    """Microseconds since the epoch: the run's events number up from here on every connection."""
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return (created_at - _EPOCH) // timedelta(microseconds=1)


def _namespace_depth(namespace: list[str]) -> int:
    return len(namespace)


def _custom_events(method: str, data: Any, namespace: list[str]) -> list[_ChannelEvent]:
    """Wrap a custom payload in the wire CustomEvent shape: ``{name?, payload}``.

    Named user stream channels arrive as ``custom:<name>`` source methods; the
    wire method is always ``custom`` with the name inside the data.
    """
    wrapped: dict[str, Any] = {"payload": data}
    if method.startswith("custom:"):
        wrapped = {"name": method[len("custom:") :], "payload": data}
    return [("custom", wrapped, namespace)]


def _prefix_matches(namespace: list[str], prefix: tuple[str, ...]) -> bool:
    """True if *namespace* starts with *prefix*, ignoring dynamic ``:task_id`` suffixes.

    Subgraph namespace segments carry a runtime task id (``node:<uuid>``); a
    client's include-list uses the clean node name (``node``). Compare literally
    first, then against the suffix-stripped segment when the prefix has no ``:``.
    """
    if len(prefix) > len(namespace):
        return False
    for i, segment in enumerate(prefix):
        candidate = namespace[i]
        if candidate == segment:
            continue
        if ":" in segment:
            return False
        if candidate.split(":", 1)[0] != segment:
            return False
    return True


def _is_terminal(raw_event: Any) -> bool:
    """True for a run's final ``end`` / ``error`` broker event."""
    method, _ = _unwrap(raw_event)
    return method in ("end", "error")


def _unwrap(raw_event: Any) -> tuple[str | None, Any]:
    """Pull ``(method, payload)`` out of a broker event; ``(None, None)`` if unknown."""
    if isinstance(raw_event, (tuple, list)) and len(raw_event) == 2:
        return raw_event[0], raw_event[1]
    return None, None


def _has_state(value: Any) -> bool:
    """True if a values/updates payload carries forwardable state."""
    if isinstance(value, dict):
        return bool(value)
    return value is not None


def _coerce_interrupts(interrupts: Any) -> list[dict[str, Any]]:
    """Coerce native Interrupt objects/dicts into ``{id, value}`` entries."""
    out: list[dict[str, Any]] = []
    for item in interrupts:
        if isinstance(item, dict):
            out.append(item)
            continue
        interrupt_id = getattr(item, "id", None)
        if isinstance(interrupt_id, str):
            out.append({"id": interrupt_id, "value": getattr(item, "value", None)})
    return out


def _dedupe_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate input requests by interrupt_id, preserving order."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for request in requests:
        interrupt_id = request.get("interrupt_id")
        if interrupt_id in seen:
            continue
        if isinstance(interrupt_id, str):
            seen.add(interrupt_id)
        out.append(request)
    return out


def validate_channels(channels: Any) -> tuple[set[str], list[str]]:
    """Split a requested channel list into (valid set, invalid names)."""
    if not isinstance(channels, list) or not channels:
        return set(), ["channels must be a non-empty array"]
    valid: set[str] = set()
    invalid: list[str] = []
    for channel in channels:
        if isinstance(channel, str) and is_supported_channel(channel):
            valid.add(channel)
        else:
            invalid.append(str(channel))
    return valid, invalid
