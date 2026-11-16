"""Which broker events a run's replay buffer keeps, and how much of each.

A v2 run's ``values`` event is the full graph state after a step. The
protocol makes the first ``values`` a new subscriber sees the current state
(its baseline) and everything after it a delta; nobody needs the earlier
snapshots of a run. So a broker keeps only the newest ``values`` per
namespace as a *snapshot slot*, overwritten in place, and appends every
other event to the replay log. Replay hands a new subscriber the slots
first, then the log after the cursor, so a reconnect costs the deltas it
missed plus one state, not every state the run ever had.

v1 events (``values`` with stream_mode suffixes, ``updates`` and the rest)
are untouched: only the native v2 ``("values", ProtocolEvent)`` shape is
recognized.
"""

from __future__ import annotations

from typing import Any

from aegra_api.utils import extract_event_sequence

# Hash fields are "<namespace>", joined with this; "" is the root namespace.
_NS_SEP = "|"
# Larger id suffixes are allocate_event_id's epoch-ms fallback, not a per-run count.
MAX_EVENT_ORDINAL = 1_000_000_000


def snapshot_slot(payload: Any) -> str | None:
    """The snapshot slot a v2 ``values`` event overwrites, or None to append it.

    Only native v2 events carry a ``{"type": "event", "method": "values"}``
    protocol envelope as their payload; a v1 ``("values", state)`` pair is
    appended as before.
    """
    if not (isinstance(payload, (tuple, list)) and len(payload) == 2 and payload[0] == "values"):
        return None
    event = payload[1]
    if not isinstance(event, dict) or event.get("type") != "event" or event.get("method") != "values":
        return None
    params = event.get("params")
    if not isinstance(params, dict):
        return None
    # A values event that carries interrupts is how the run announces them
    # (the session splits them onto the input channel); keep it in the log.
    if params.get("interrupts"):
        return None
    data = params.get("data")
    if isinstance(data, dict) and "__interrupt__" in data:
        return None
    namespace = params.get("namespace")
    if not isinstance(namespace, list):
        namespace = []
    return _NS_SEP.join(str(segment) for segment in namespace)


def event_ordinal(event_id: str) -> int:
    """The per-run ordinal allocated with an event id (``<run>_event_<n>``)."""
    return extract_event_sequence(event_id)


def ordered_ordinal(event_id: str) -> int | None:
    """The event's per-run ordinal, or None when it carries no usable order.

    None for ids without an ordinal and for the allocator's epoch-ms fallback,
    which is not comparable with real per-run counts.
    """
    ordinal = event_ordinal(event_id)
    return ordinal if 0 < ordinal < MAX_EVENT_ORDINAL else None


def is_after(event_id: str, cursor: str | None) -> bool:
    """True if *event_id* was allocated after the cursor event id.

    A cursor or event without a usable ordinal counts as "after", so the
    caller replays it and the client deduplicates by event id.
    """
    if cursor is None:
        return True
    event, mark = ordered_ordinal(event_id), ordered_ordinal(cursor)
    if event is None or mark is None:
        return True
    return event > mark


def merge_replay(
    snapshots: list[tuple[str, Any]],
    log: list[tuple[str, Any]],
) -> list[tuple[str, Any]]:
    """Snapshots and log entries in allocation order.

    Each snapshot sits where it was produced, so a ``values`` that followed a
    ``checkpoints`` envelope still follows it and one that preceded an
    ``input`` event still precedes it.
    """
    pending = sorted(snapshots, key=lambda item: ordered_ordinal(item[0]) or 0)
    merged: list[tuple[str, Any]] = []
    # The log is already in write order; never reorder it. Insert each snapshot
    # before the first log entry allocated after it.
    for event_id, payload in log:
        entry = ordered_ordinal(event_id)
        if entry is not None:
            while pending and (snap := ordered_ordinal(pending[0][0])) is not None and snap < entry:
                merged.append(pending.pop(0))
        if merged and merged[-1][0] == event_id:
            continue
        merged.append((event_id, payload))
    merged.extend(pending)
    return merged


# Methods a settled run's compact replay keeps: its state, timeline and
# lifecycle. Message, tool and custom deltas are already folded into the
# final values snapshot, and a delta without its start would confuse the
# client's message assembly.
_SETTLED_METHODS = frozenset({"values", "checkpoints", "lifecycle", "updates", "end", "error"})


def kept_when_settled(payload: Any) -> bool:
    """True for an event a settled run's compact replay still sends."""
    if isinstance(payload, (tuple, list)) and len(payload) == 2 and isinstance(payload[0], str):
        return payload[0] in _SETTLED_METHODS
    return False
