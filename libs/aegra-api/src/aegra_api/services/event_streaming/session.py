"""Thread-scoped session that forwards a thread's native v3 events as v2 events.

A v2 stream is scoped to a *thread*, not a run (matching the LangGraph SDK):
the client opens the stream, issues ``run.start``, and the events of whatever
run(s) execute on the thread flow through. Every run on the thread appends its
native protocol events (the v3 producer wrote them) as ``(method, event)``
pairs, plus a terminal ``("end"|"error", payload)``, to one event log per
thread; a ``values`` event, interrupts included, instead replaces its namespace's
snapshot (stored with its checkpoint envelope) and appends a marker. The session
replays, per requested namespace, the latest snapshot, then the log from the
earliest of those snapshots' markers, then reads on with bounded blocking reads
from its cursor, so a run started after the stream opened arrives on it with no
database polling. A marker emits its snapshot only while the snapshot is still
the one it marks; otherwise a newer marker follows. The session restamps
seq/event_id, filters by channel, splits interrupts onto the ``input``
channel, normalizes state message payloads, and derives each run's root
lifecycle.

``seq`` numbers up from the seq stored with each log entry, so it rises across
runs and is the same on every connection. ``event_id`` is ``<entry id>:<index>`` (a snapshot's
is its marker's; root lifecycle: ``synth:<run_id>:...``), stable across replays
for client dedup. ``since`` filters what was already seen.

The first entry of a run id the session has not seen costs one ownership-
filtered run lookup; runs the caller can't see are skipped.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from aegra_api.services.base_broker import (
    INTERRUPT_MARKER,
    SNAPSHOT_MARKER,
    Snapshot,
    ThreadEntry,
    entry_before,
    entry_key,
)
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

__all__ = ["RunLookup", "ThreadEventSession", "VisibleRun", "validate_channels"]


@dataclasses.dataclass(frozen=True)
class VisibleRun:
    """A run the caller can see; ``graph`` is None when its assistant is gone."""

    graph: str | None


# Async callable: the run, or None when the caller can't see it.
RunLookup = Callable[[str], Awaitable[VisibleRun | None]]

# A projected channel event: (wire method, params.data, namespace).
_ChannelEvent = tuple[str, dict[str, Any], list[str]]


@dataclasses.dataclass
class _RunView:
    """What the session tracks for one visible run until its terminal."""

    graph: str | None
    # Subgraph namespaces started but not yet completed, closed by the terminal.
    open_namespaces: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    # One input.requested per interrupt: it can surface via updates and the next values.
    sent_interrupts: set[str] = dataclasses.field(default_factory=set)
    last_write: str | None = None


class ThreadEventSession:
    """Forwards native v3 events for all runs on a thread, filtered to channels."""

    def __init__(
        self,
        thread_id: str,
        *,
        channels: set[str],
        lookup_run: RunLookup,
        since: int | None = None,
        namespaces: list[list[str]] | None = None,
        depth: int | None = None,
    ) -> None:
        self._thread_id = thread_id
        self._channels = channels
        self._lookup_run = lookup_run
        self._since = since
        self._namespaces = [tuple(prefix) for prefix in namespaces] if namespaces else None
        self._depth = depth
        self._seq = 0
        # run_id -> the run, or None when the caller can't see it.
        self._visible: dict[str, VisibleRun | None] = {}
        # run_id -> its view, or None once it ended or when the caller can't see it.
        self._runs: dict[str, _RunView | None] = {}
        self._run_id = ""
        self._entry_ms: int | None = None
        self._view = _RunView(graph=None)
        self._wants_state = bool({"values", "checkpoints"} & channels)
        # Interrupts ride on snapshots, so an input-only stream reads the snapshots that carry them.
        self._wants_interrupts = "input" in channels
        # Snapshots read but not yet emitted at their marker, by namespace key.
        self._pending: dict[str, Snapshot] = {}

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Yield v2 envelopes for the thread's runs until the client disconnects.

        Never ends on its own: an idle thread stream blocks on the log while
        the SSE layer sends heartbeats. A failed read raises.
        """
        cursor = await self._replay_start()
        first_read = True
        while True:
            entries = await broker_manager.read_thread(self._thread_id, cursor)
            if first_read:
                first_read = False
                for envelope in await self._trimmed_snapshots(entries):
                    yield envelope
            await self._read_snapshots(entries)
            for entry in entries:
                cursor = entry.entry_id
                for envelope in await self._ingest(entry):
                    yield envelope
            # A snapshot whose marker passed unmatched waits for a newer marker, not here.
            self._pending = {
                ns: snap for ns, snap in self._pending.items() if entry_key(snap.marker_id) > entry_key(cursor)
            }

    async def _replay_start(self) -> str:
        """The cursor to replay from: before the earliest marker of a requested namespace's snapshot."""
        ids = await broker_manager.read_snapshot_ids(self._thread_id)
        wanted = [namespace for namespace in ids if self._wants_namespace(json.loads(namespace))]
        if not wanted:
            return "0-0"
        if self._wants_state or self._wants_interrupts:
            # ponytail: an input-only stream also reads snapshots without interrupts here (once per connect);
            # record the marker kind in snapshot-ids if that read shows up in Redis bandwidth.
            self._pending = await broker_manager.read_snapshots(self._thread_id, wanted)
        return entry_before(min((ids[namespace] for namespace in wanted), key=entry_key))

    async def _trimmed_snapshots(self, entries: list[ThreadEntry]) -> list[dict[str, Any]]:
        """Snapshots whose marker the log no longer holds, oldest first, ahead of the first entry read."""
        first = entry_key(entries[0].entry_id) if entries else None
        envelopes: list[dict[str, Any]] = []
        for namespace, (entry_id, seq, record) in sorted(
            self._pending.items(), key=lambda item: entry_key(item[1].marker_id)
        ):
            if first is not None and entry_key(entry_id) >= first:
                continue
            del self._pending[namespace]
            run_id = record.get("run_id")
            if not isinstance(run_id, str) or await self._lookup(run_id) is None:
                continue
            # Past the running seed's slot, as at the marker.
            self._seq = max(self._seq, seq) + 1
            self._entry_ms = entry_key(entry_id)[0]
            envelopes += self._snapshot(entry_id, record)
        return envelopes

    async def _read_snapshots(self, entries: list[ThreadEntry]) -> None:
        """Read the snapshot of each wanted namespace's last marker in *entries*, unless already read."""
        latest: dict[str, str] = {}
        for entry in entries:
            namespace = _marker_namespace(entry.event)
            if namespace is None or not self._reads_marker(entry.event):
                continue
            if self._wants_namespace(json.loads(namespace)):
                latest[namespace] = entry.entry_id
        stale = [
            namespace
            for namespace, entry_id in latest.items()
            if namespace not in self._pending or entry_key(self._pending[namespace].marker_id) < entry_key(entry_id)
        ]
        if stale:
            self._pending.update(await broker_manager.read_snapshots(self._thread_id, stale))

    def _reads_marker(self, raw_event: Any) -> bool:
        method, _ = _unwrap(raw_event)
        return self._wants_state or (self._wants_interrupts and method == INTERRUPT_MARKER)

    async def _lookup(self, run_id: str) -> VisibleRun | None:
        if run_id not in self._visible:
            self._visible[run_id] = await self._lookup_run(run_id)
        return self._visible[run_id]

    async def _ingest(self, entry: ThreadEntry) -> list[dict[str, Any]]:
        """Project one log entry; the first entry of a run also opens its root lifecycle."""
        entry_id, seq, run_id, write_id, raw_event = entry
        first = run_id not in self._runs
        if first:
            run = await self._lookup(run_id)
            self._runs[run_id] = None if run is None else _RunView(graph=run.graph)
        view = self._runs[run_id]
        marker = _marker_namespace(raw_event)
        # A retried write lands twice with the same write id; a retried marker can match only once.
        if view is None or (marker is None and write_id == view.last_write):
            return []
        view.last_write = write_id
        self._run_id, self._view = run_id, view
        # max() keeps seq rising should an entry ever project to more than SEQ_STRIDE events.
        self._seq = max(self._seq, seq)

        # Events the producer didn't stamp (root lifecycle) take the entry's write time, the same on every read.
        self._entry_ms = entry_key(entry_id)[0]
        envelopes: list[dict[str, Any]] = []
        if first:
            envelopes = self._emit(
                [("lifecycle", self._root_lifecycle("running"), [])],
                f"synth:{run_id}:running",
                timestamp=self._entry_ms,
            )
        else:
            # The running seed's slot, so an entry's seqs don't depend on where the replay started.
            self._seq += 1
        if marker is not None:
            snapshot = self._pending.get(marker)
            if snapshot is not None and snapshot.marker_id == entry_id:
                del self._pending[marker]
                envelopes += self._snapshot(entry_id, snapshot.record)
            return envelopes
        envelopes += self._project(entry_id, raw_event)
        if _is_terminal(raw_event):
            # The run ended: drop its state and ignore anything it writes after.
            self._runs[run_id] = None
        return envelopes

    def _snapshot(self, entry_id: str, record: dict[str, Any]) -> list[dict[str, Any]]:
        """A snapshot's checkpoint envelope then its values, with their producer timestamps."""
        checkpoint = record.get("checkpoint")
        if checkpoint is None:
            self._seq += 1
            envelopes = []
        else:
            envelopes = self._project(entry_id, ("checkpoints", checkpoint))
        return envelopes + self._project(entry_id, ("values", record.get("values")), first_index=1)

    def _project(self, event_id: str, raw_event: Any, *, first_index: int = 0) -> list[dict[str, Any]]:
        """Re-envelope one raw log event into filtered, seq'd envelopes."""
        method, payload = _unwrap(raw_event)
        if method is None:
            return []
        if method in ("end", "error"):
            # The SDK reads the run of a root lifecycle event from a synth:<run_id>: id.
            event_id = f"synth:{self._run_id}:{method}"
        params = payload.get("params") if isinstance(payload, dict) else None
        timestamp = params.get("timestamp") if isinstance(params, dict) else None
        if timestamp is None:
            timestamp = self._entry_ms
        return self._emit(self._channel_events(method, payload), event_id, timestamp=timestamp, first_index=first_index)

    def _emit(
        self,
        channel_events: list[_ChannelEvent],
        event_id: str,
        *,
        timestamp: int | None = None,
        first_index: int = 0,
    ) -> list[dict[str, Any]]:
        """Filter + seq a batch of channel events into wire envelopes."""
        envelopes: list[dict[str, Any]] = []
        for index, (channel, data, namespace) in enumerate(channel_events, start=first_index):
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
                if interrupt_id in self._view.sent_interrupts:
                    continue
                self._view.sent_interrupts.add(interrupt_id)
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
            self._view.open_namespaces[key] = list(namespace)
        elif event in ("completed", "failed"):
            self._view.open_namespaces.pop(key, None)

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
        for namespace in sorted(self._view.open_namespaces.values(), key=_namespace_depth, reverse=True):
            events.append(("lifecycle", {"event": "completed"}, namespace))
        self._view.open_namespaces.clear()

        data = self._root_lifecycle(lifecycle_status(status or ""))
        if isinstance(payload, dict) and (message := payload.get("message")):
            data["error"] = message
        events.append(("lifecycle", data, []))
        return events

    def _root_lifecycle(self, status: str) -> dict[str, Any]:
        """Root-namespace lifecycle data, carrying the run's graph name when known."""
        data: dict[str, Any] = {"event": status}
        if self._view.graph:
            data["graph_name"] = self._view.graph
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


def _marker_namespace(raw_event: Any) -> str | None:
    """The namespace key of a snapshot marker entry, else None."""
    method, payload = _unwrap(raw_event)
    return payload if method in (SNAPSHOT_MARKER, INTERRUPT_MARKER) and isinstance(payload, str) else None


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
