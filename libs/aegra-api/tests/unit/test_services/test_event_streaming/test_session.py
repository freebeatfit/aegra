"""Tests for ThreadEventSession: native-event forwarding, seq, filter, since, HITL, lifecycle."""

import asyncio
import contextlib
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest

from aegra_api.services import broker as broker_module
from aegra_api.services.base_broker import ThreadEntry, entry_key
from aegra_api.services.broker import BrokerManager
from aegra_api.services.event_streaming import session as session_module
from aegra_api.services.event_streaming.session import (
    RunLookup,
    ThreadEventSession,
    VisibleRun,
    validate_channels,
)


class _Drained(Exception):
    """The thread's log has nothing more to read."""


class _DrainingBrokerManager(BrokerManager):
    """In-memory broker whose read raises once the log is drained, so a session ends."""

    async def read_thread(self, thread_id: str, after: str) -> list[ThreadEntry]:
        entries = self._log(thread_id).after(entry_key(after))
        if not entries:
            raise _Drained
        return entries


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[BrokerManager]:
    """Swap in a fresh in-memory broker manager for the session under test."""
    mgr = _DrainingBrokerManager()
    monkeypatch.setattr(session_module, "broker_manager", mgr)
    yield mgr


@pytest.fixture
def live_manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[BrokerManager]:
    """An in-memory broker whose reads block briefly, like a live thread stream."""
    mgr = BrokerManager()
    monkeypatch.setattr(session_module, "broker_manager", mgr)
    monkeypatch.setattr(broker_module, "_READ_WAIT_SECONDS", 0.05)
    yield mgr


def _lookup(graph: str | None = None, *, hidden: tuple[str, ...] = ()) -> RunLookup:
    """Run lookup: every run is visible with *graph* except the *hidden* ones."""

    async def lookup(run_id: str) -> VisibleRun | None:
        return None if run_id in hidden else VisibleRun(graph=graph)

    return lookup


def _protocol_event(method: str, data: Any, *, namespace: list[str] | None = None, **params: Any) -> dict[str, Any]:
    """A native v3 ProtocolEvent as a v2 run writes it to the thread's log."""
    return {
        "type": "event",
        "method": method,
        "params": {"namespace": namespace or [], "data": data, **params},
    }


async def _seed(manager: BrokerManager, run_id: str, events: list[tuple[str, Any]], *, thread_id: str = "t1") -> None:
    for raw in events:
        await manager.append(thread_id, run_id, uuid4().hex, raw)


def _msg_event(event_kind: str, **extra: Any) -> tuple[str, dict[str, Any]]:
    return ("messages", _protocol_event("messages", {"event": event_kind, **extra}))


def _make_session(
    thread_id: str,
    *,
    channels: set[str],
    since: int | None = None,
    namespaces: list[list[str]] | None = None,
    depth: int | None = None,
    graph: str | None = None,
    lookup_run: RunLookup | None = None,
) -> ThreadEventSession:
    return ThreadEventSession(
        thread_id,
        channels=channels,
        lookup_run=lookup_run or _lookup(graph),
        since=since,
        namespaces=namespaces,
        depth=depth,
    )


async def _collect(session: ThreadEventSession) -> list[dict[str, Any]]:
    """Every envelope the session yields until the log is drained."""
    events: list[dict[str, Any]] = []
    with contextlib.suppress(_Drained):
        async for event in session.stream():
            events.append(event)
    return events


def _root_ended(event: dict[str, Any]) -> bool:
    return (
        event["method"] == "lifecycle"
        and not event["params"]["namespace"]
        and event["params"]["data"].get("event") != "running"
    )


async def _read_until_runs_end(session: ThreadEventSession, runs: int) -> list[dict[str, Any]]:
    """Read a live session until *runs* root terminals arrived."""
    events: list[dict[str, Any]] = []
    stream = session.stream()
    async with asyncio.timeout(2.0):
        async for event in stream:
            events.append(event)
            if sum(_root_ended(e) for e in events) == runs:
                break
    await stream.aclose()
    return events


class TestForwarding:
    async def test_message_lifecycle_forwarded_verbatim(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                _msg_event("message-start", role="ai", id="m1"),
                _msg_event("content-block-delta", index=0, delta={"type": "text-delta", "text": "hi"}),
                _msg_event("message-finish", usage={"total_tokens": 5}),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"messages", "lifecycle"}))
        kinds = [(e["method"], e["params"]["data"].get("event")) for e in events]
        assert kinds == [
            ("lifecycle", "running"),
            ("messages", "message-start"),
            ("messages", "content-block-delta"),
            ("messages", "message-finish"),
            ("lifecycle", "completed"),
        ]

    async def test_producer_timestamp_is_kept(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1}, timestamp=1_700_000_000_000)),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"values", "lifecycle"}))
        assert [e["method"] for e in events] == ["lifecycle", "values", "lifecycle"]
        assert events[1]["params"]["timestamp"] == 1_700_000_000_000
        assert events[0]["params"]["timestamp"] > 1_700_000_000_000

    async def test_values_payload_in_params_data(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"values"}))
        assert events[0]["params"]["data"] == {"a": 1}
        assert events[0]["params"]["namespace"] == []

    async def test_namespace_preserved(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("values", _protocol_event("values", {"a": 1}, namespace=["sub:1"])), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"values"}))
        assert events[0]["params"]["namespace"] == ["sub:1"]

    async def test_tools_channel_forwarded(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("tools", _protocol_event("tools", {"event": "tool-started", "tool_call_id": "c1"})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"tools"}))
        assert events[0]["params"]["data"] == {"event": "tool-started", "tool_call_id": "c1"}

    async def test_raw_v3_updates_normalized_to_node_values(self, manager: BrokerManager) -> None:
        """v3 emits updates as raw {node: values}; the wire form is {node, values}."""
        await _seed(
            manager,
            "run-1",
            [("updates", _protocol_event("updates", {"worker": {"count": 1}})), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"updates"}))
        assert events[0]["params"]["data"] == {"node": "worker", "values": {"count": 1}}

    async def test_interrupt_node_in_updates_routes_to_input(self, manager: BrokerManager) -> None:
        """An __interrupt__ update surfaces on the input channel, not as an updates event."""
        await _seed(
            manager,
            "run-1",
            [
                ("updates", _protocol_event("updates", {"__interrupt__": [{"id": "int-9", "value": {"q": "ok?"}}]})),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"input", "updates"}))
        input_events = [e for e in events if e["method"] == "input.requested"]
        assert input_events[0]["params"]["data"] == {
            "interrupt_id": "int-9",
            "value": {"q": "ok?"},
            "payload": {"q": "ok?"},
        }
        assert not [e for e in events if e["method"] == "updates"]


class TestSeqAndFilter:
    async def test_seq_numbers_up_from_the_log_entry_id(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})])
        entry_id = manager._log("t1").entries[0][0]
        ms, n = entry_key(entry_id)

        events = await _collect(_make_session("t1", channels={"values", "lifecycle"}))

        # The run's running seed takes the first slot of its first entry, the values the next.
        assert [e["seq"] for e in events[:2]] == [((ms << 8) + n << 3) + 1, ((ms << 8) + n << 3) + 2]
        assert events[1]["event_id"] == f"{entry_id}:0"

    async def test_seq_rises_across_runs(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})])
        await _seed(manager, "run-2", [("values", _protocol_event("values", {"a": 2})), ("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"values", "lifecycle"}))
        seqs = [e["seq"] for e in events]
        assert len(events) == 6
        assert seqs == sorted(set(seqs))

    async def test_channel_filter_drops_unsubscribed(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1})),
                ("updates", _protocol_event("updates", {"n": {"b": 2}})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"values"}))
        assert {e["method"] for e in events} == {"values"}

    async def test_seq_absolute_not_filter_relative(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1})),
                ("updates", _protocol_event("updates", {"n": {"b": 2}})),
                ("end", {"status": "success"}),
            ],
        )
        everything = await _collect(_make_session("t1", channels={"values", "updates", "lifecycle"}))
        lifecycle = await _collect(_make_session("t1", channels={"lifecycle"}))
        assert [(e["event_id"], e["seq"]) for e in lifecycle] == [
            (e["event_id"], e["seq"]) for e in everything if e["method"] == "lifecycle"
        ]

    async def test_since_skips_already_seen(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1})),
                ("values", _protocol_event("values", {"a": 2})),
                ("end", {"status": "success"}),
            ],
        )
        full = await _collect(_make_session("t1", channels={"values", "lifecycle"}))
        resumed = await _collect(_make_session("t1", channels={"values", "lifecycle"}, since=full[1]["seq"]))
        assert [(e["seq"], e["event_id"]) for e in resumed] == [(e["seq"], e["event_id"]) for e in full[2:]]
        assert [e["params"]["data"] for e in resumed if e["method"] == "values"] == [{"a": 2}]

    async def test_seq_is_the_same_on_every_connection(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})])

        first = await _collect(_make_session("t1", channels={"values", "lifecycle"}))
        second = await _collect(_make_session("t1", channels={"values", "lifecycle"}))

        assert [(e["seq"], e["event_id"]) for e in first] == [(e["seq"], e["event_id"]) for e in second]

    async def test_seq_survives_a_trimmed_log(self, manager: BrokerManager) -> None:
        """A retained entry keeps the seq and event id it had before older entries were trimmed."""
        events = [("updates", _protocol_event("updates", {"node": {"i": i}})) for i in range(1, 6)] + [
            ("end", {"status": "success"})
        ]
        await _seed(manager, "run-1", events)
        full = await _collect(_make_session("t1", channels={"updates", "lifecycle"}))
        for _ in range(3):
            manager._log("t1").entries.popleft()

        trimmed = await _collect(_make_session("t1", channels={"updates", "lifecycle"}))

        seq_by_id = {e["event_id"]: e["seq"] for e in full}
        assert [e["event_id"] for e in trimmed[1:]] == [e["event_id"] for e in full[4:]]
        assert [e["seq"] for e in trimmed[1:]] == [seq_by_id[e["event_id"]] for e in trimmed[1:]]

    async def test_seq_keeps_rising_after_earlier_runs_expire(self, manager: BrokerManager) -> None:
        """#751: a new connection must not number a new run below seqs the client already saw."""
        await _seed(manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})])
        first = await _collect(_make_session("t1", channels={"values", "lifecycle"}))
        cursor = first[-1]["seq"]
        manager.sweep_expired_logs(retention_seconds=-1)
        await asyncio.sleep(0.002)  # a log recreated 10 min later starts in a later millisecond
        await _seed(manager, "run-2", [("values", _protocol_event("values", {"a": 2})), ("end", {"status": "success"})])

        fresh = await _collect(_make_session("t1", channels={"values", "lifecycle"}))
        resumed = await _collect(_make_session("t1", channels={"values", "lifecycle"}, since=cursor))

        assert len(fresh) == 3
        assert fresh[0]["seq"] > cursor
        assert [(e["seq"], e["event_id"]) for e in resumed] == [(e["seq"], e["event_id"]) for e in fresh]

    async def test_retried_write_is_emitted_once(self, manager: BrokerManager) -> None:
        """A write retried after Redis applied it lands twice with the same write id."""
        event = ("values", _protocol_event("values", {"a": 1}))
        await manager.append("t1", "run-1", "w1", event)
        await manager.append("t1", "run-1", "w1", event)
        await manager.append("t1", "run-1", "w2", ("end", {"status": "success"}))

        events = await _collect(_make_session("t1", channels={"values", "lifecycle"}))

        assert [e["method"] for e in events] == ["lifecycle", "values", "lifecycle"]


class TestInterruptsToInputChannel:
    async def test_interrupt_on_values_becomes_input_requested(self, manager: BrokerManager) -> None:
        """An interrupt riding on a values event's params.interrupts surfaces on input."""
        interrupt_payload = {"question": "Approve?"}
        await _seed(
            manager,
            "run-1",
            [
                (
                    "values",
                    _protocol_event(
                        "values",
                        {"messages": []},
                        interrupts=[{"id": "int-1", "value": interrupt_payload}],
                    ),
                ),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"input", "values"}))
        input_events = [e for e in events if e["method"] == "input.requested"]
        assert input_events
        assert input_events[0]["params"]["data"] == {
            "interrupt_id": "int-1",
            "value": interrupt_payload,
            "payload": interrupt_payload,
        }

    async def test_interrupt_stripped_from_values_payload(self, manager: BrokerManager) -> None:
        """__interrupt__ never leaks into the forwarded values data."""
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"messages": [], "__interrupt__": [{"id": "int-1", "value": 1}]})),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"values"}))
        values_events = [e for e in events if e["method"] == "values"]
        assert "__interrupt__" not in values_events[0]["params"]["data"]


class TestLifecycle:
    async def test_running_seed_opens_each_run(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}))
        assert events[0]["params"]["data"] == {"event": "running"}
        assert events[0]["params"]["namespace"] == []

    async def test_root_lifecycle_carries_graph_name_when_known(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}, graph="my_graph"))
        assert events[0]["params"]["data"] == {"event": "running", "graph_name": "my_graph"}
        assert events[-1]["params"]["data"] == {"event": "completed", "graph_name": "my_graph"}

    async def test_root_lifecycle_event_ids_name_the_run(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "success"})])
        await _seed(manager, "run-2", [("error", {"error": "RuntimeError", "message": "boom"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}))
        assert [e["event_id"].split(":")[:2] for e in events] == [
            ["synth", "run-1"],
            ["synth", "run-1"],
            ["synth", "run-2"],
            ["synth", "run-2"],
        ]

    async def test_completed(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}))
        assert events[-1]["params"]["data"] == {"event": "completed"}

    async def test_interrupted(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "interrupted"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}))
        assert events[-1]["params"]["data"] == {"event": "interrupted"}

    async def test_failed_carries_error(self, manager: BrokerManager) -> None:
        # The real broker error event is {error, message} with NO status; the
        # failed status must come from the method, not a status key.
        await _seed(manager, "run-1", [("error", {"error": "RuntimeError", "message": "boom"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}))
        assert events[-1]["params"]["data"] == {"event": "failed", "error": "boom"}


class TestMisc:
    async def test_custom_payload_wrapped_in_wire_shape(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("custom", _protocol_event("custom", {"hello": "world"})), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"custom"}))
        custom = [e for e in events if e["method"] == "custom"]
        assert custom and custom[0]["params"]["data"] == {"payload": {"hello": "world"}}

    async def test_named_custom_source_becomes_custom_with_name(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("custom:my_event", _protocol_event("custom:my_event", {"x": 1})), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"custom:my_event"}))
        custom = [e for e in events if e["method"] == "custom"]
        assert custom and custom[0]["params"]["data"] == {"name": "my_event", "payload": {"x": 1}}

    async def test_named_subscription_filters_other_custom_events(self, manager: BrokerManager) -> None:
        """custom:foo subscribers get only name==foo; unnamed events need a plain custom subscription."""
        await _seed(
            manager,
            "run-1",
            [
                ("custom:other", _protocol_event("custom:other", {"x": 1})),
                ("custom", _protocol_event("custom", {"y": 2})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"custom:my_event"}))
        assert not [e for e in events if e["method"] == "custom"]

    async def test_plain_custom_subscription_receives_named_events(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [("custom:my_event", _protocol_event("custom:my_event", {"x": 1})), ("end", {"status": "success"})],
        )
        events = await _collect(_make_session("t1", channels={"custom"}))
        assert [e for e in events if e["method"] == "custom"]

    async def test_empty_thread_yields_nothing(self, manager: BrokerManager) -> None:
        events = await _collect(_make_session("t1", channels={"lifecycle"}))
        assert events == []


class TestResumeAcrossRuns:
    """A HITL resume starts a fresh run on the thread; its events must reach the
    same open stream, not be dropped when the interrupted run drains first."""

    async def test_repeated_interrupt_id_is_emitted_for_each_run(
        self: "TestResumeAcrossRuns", manager: BrokerManager
    ) -> None:
        """A resumed run may reuse its predecessor's interrupt ID.

        Both runs are replayed from the thread's log; each emits its own input.requested.
        """
        for run_id, question in (("run-a", "First question?"), ("run-b", "Second question?")):
            await _seed(
                manager,
                run_id,
                [
                    (
                        "values",
                        _protocol_event(
                            "values",
                            {"messages": []},
                            interrupts=[{"id": "same-id", "value": {"question": question}}],
                        ),
                    ),
                    ("end", {"status": "interrupted"}),
                ],
            )

        events = await _collect(
            _make_session(
                "t1",
                channels={"input", "values", "lifecycle"},
            )
        )

        requests = [event["params"]["data"] for event in events if event["method"] == "input.requested"]
        assert [request["value"] for request in requests] == [
            {"question": "First question?"},
            {"question": "Second question?"},
        ]

    async def test_followup_run_after_interrupt_streams_on_same_session(self, live_manager: BrokerManager) -> None:
        await _seed(
            live_manager,
            "run-a",
            [
                (
                    "values",
                    _protocol_event("values", {"messages": []}, interrupts=[{"id": "int-1", "value": {"q": "ok?"}}]),
                ),
                ("end", {"status": "interrupted"}),
            ],
        )

        async def resume_as_run_b() -> None:
            # run-b starts a beat after run-a ended, as the resume round-trip lands.
            await asyncio.sleep(0.1)
            await _seed(
                live_manager,
                "run-b",
                [
                    ("values", _protocol_event("values", {"messages": [{"type": "ai", "content": "done"}]})),
                    ("end", {"status": "success"}),
                ],
            )

        seeder = asyncio.create_task(resume_as_run_b())
        events = await _read_until_runs_end(_make_session("t1", channels={"input", "values", "lifecycle"}), 2)
        await seeder

        methods = [(e["method"], e["params"]["data"]) for e in events]
        assert (
            "input.requested",
            {"interrupt_id": "int-1", "value": {"q": "ok?"}, "payload": {"q": "ok?"}},
        ) in methods
        # run-b's completion proves the stream stayed open across the run gap.
        assert events[-1]["params"]["data"] == {"event": "completed"}


class TestNamespaceFilter:
    """namespaces (prefix include-list) and depth (nesting cap) filter subgraph events."""

    async def test_namespaces_include_only_matching_prefix(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1}, namespace=["sub_a"])),
                ("values", _protocol_event("values", {"b": 2}, namespace=["sub_b"])),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"values"}, namespaces=[["sub_a"]]))
        assert [e["params"]["data"] for e in events] == [{"a": 1}]

    async def test_depth_caps_nesting(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"top": 1}, namespace=["sub_a"])),
                ("values", _protocol_event("values", {"deep": 2}, namespace=["sub_a", "sub_b"])),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"values"}, depth=1))
        assert [e["params"]["data"] for e in events] == [{"top": 1}]

    async def test_thread_level_events_always_pass_the_filter(self, manager: BrokerManager) -> None:
        """Lifecycle (empty namespace) is not subgraph-scoped; a namespace filter must not drop it."""
        await _seed(manager, "run-1", [("end", {"status": "success"})])
        events = await _collect(_make_session("t1", channels={"lifecycle"}, namespaces=[["sub_a"]]))
        assert events[-1]["params"]["data"] == {"event": "completed"}

    async def test_namespace_filter_matches_dynamic_task_id_suffix(self, manager: BrokerManager) -> None:
        """Real subgraph namespaces are node:<task_id>; a clean-name prefix must match."""
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1}, namespace=["sub_a:9f3c-1"])),
                ("values", _protocol_event("values", {"b": 2}, namespace=["sub_b:7e1a-2"])),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"values"}, namespaces=[["sub_a"]]))
        assert [e["params"]["data"] for e in events] == [{"a": 1}]

    async def test_seq_stays_absolute_under_namespace_filter(self, manager: BrokerManager) -> None:
        """Filtered-out events still advance seq, so reconnect cursors stay stable."""
        await _seed(
            manager,
            "run-1",
            [
                ("values", _protocol_event("values", {"a": 1}, namespace=["sub_b"])),  # filtered
                ("values", _protocol_event("values", {"b": 2}, namespace=["sub_a"])),
                ("end", {"status": "success"}),
            ],
        )
        everything = await _collect(_make_session("t1", channels={"values"}))
        filtered = await _collect(_make_session("t1", channels={"values"}, namespaces=[["sub_a"]]))
        assert [(e["params"]["data"], e["seq"]) for e in filtered] == [({"b": 2}, everything[1]["seq"])]


class TestUpdatesInterruptHandling:
    async def test_sibling_node_update_survives_interrupt_in_same_chunk(self, manager: BrokerManager) -> None:
        """A parallel branch's update arriving alongside __interrupt__ must not vanish."""
        await _seed(
            manager,
            "run-1",
            [
                (
                    "updates",
                    _protocol_event(
                        "updates",
                        {"__interrupt__": [{"id": "int-1", "value": {"q": "ok?"}}], "worker": {"count": 2}},
                    ),
                ),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"input", "updates"}))
        assert [e for e in events if e["method"] == "input.requested"]
        updates = [e for e in events if e["method"] == "updates"]
        assert updates and updates[0]["params"]["data"] == {"node": "worker", "values": {"count": 2}}

    async def test_interrupt_nested_in_node_values_surfaces_on_input(self, manager: BrokerManager) -> None:
        """An interrupt riding inside a node's values must emit input.requested, not be silently stripped."""
        await _seed(
            manager,
            "run-1",
            [
                (
                    "updates",
                    _protocol_event(
                        "updates",
                        {"gate": {"messages": [], "__interrupt__": [{"id": "int-2", "value": {"q": "sure?"}}]}},
                    ),
                ),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"input", "updates"}))
        input_events = [e for e in events if e["method"] == "input.requested"]
        assert input_events and input_events[0]["params"]["data"]["interrupt_id"] == "int-2"
        updates = [e for e in events if e["method"] == "updates"]
        assert "__interrupt__" not in updates[0]["params"]["data"]["values"]

    async def test_same_interrupt_via_updates_then_values_emits_once(self, manager: BrokerManager) -> None:
        """The interrupt often rides an updates chunk AND the following values snapshot."""
        await _seed(
            manager,
            "run-1",
            [
                ("updates", _protocol_event("updates", {"__interrupt__": [{"id": "int-3", "value": {"q": "go?"}}]})),
                (
                    "values",
                    _protocol_event("values", {"messages": []}, interrupts=[{"id": "int-3", "value": {"q": "go?"}}]),
                ),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"input", "values"}))
        input_events = [e for e in events if e["method"] == "input.requested"]
        assert len(input_events) == 1
        assert input_events[0]["params"]["data"]["interrupt_id"] == "int-3"


class TestRunVisibility:
    """The first entry of a run costs one ownership-filtered lookup; hidden runs are skipped."""

    async def test_run_the_caller_cannot_see_is_skipped(self, manager: BrokerManager) -> None:
        await _seed(
            manager, "theirs", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})]
        )
        await _seed(manager, "mine", [("values", _protocol_event("values", {"a": 2})), ("end", {"status": "success"})])

        events = await _collect(
            _make_session("t1", channels={"values", "lifecycle"}, lookup_run=_lookup(hidden=("theirs",)))
        )

        assert [e["event_id"].split(":")[:2] for e in events if e["method"] == "lifecycle"] == [["synth", "mine"]] * 2
        assert [e["params"]["data"] for e in events if e["method"] == "values"] == [{"a": 2}]

    async def test_each_run_is_looked_up_once(self, manager: BrokerManager) -> None:
        for run_id in ("run-1", "run-2"):
            await _seed(
                manager,
                run_id,
                [
                    ("values", _protocol_event("values", {"a": 1})),
                    ("values", _protocol_event("values", {"a": 2})),
                    ("end", {"status": "success"}),
                ],
            )
        looked_up: list[str] = []

        async def lookup(run_id: str) -> VisibleRun | None:
            looked_up.append(run_id)
            return VisibleRun(graph=None)

        await _collect(_make_session("t1", channels={"values"}, lookup_run=lookup))

        assert looked_up == ["run-1", "run-2"]

    async def test_events_after_a_runs_terminal_are_ignored(self, manager: BrokerManager) -> None:
        """An error is terminal; the end written after it must not add a second terminal."""
        await _seed(
            manager, "run-1", [("error", {"error": "RuntimeError", "message": "boom"}), ("end", {"status": "error"})]
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}))
        assert [e["params"]["data"]["event"] for e in events] == ["running", "failed"]


class TestSubgraphLifecycle:
    """Native producer emits per-subgraph lifecycle at root scope with the child
    namespace in data.namespace; the session promotes it onto the wire so nested
    agents get started/completed/failed frames on their own namespace."""

    async def test_subgraph_started_completed_promoted_to_child_namespace(self, manager: BrokerManager) -> None:
        ns = ["subgraph_agent:abc-123"]
        await _seed(
            manager,
            "run-1",
            [
                (
                    "lifecycle",
                    _protocol_event(
                        "lifecycle",
                        {
                            "event": "started",
                            "namespace": ns,
                            "graph_name": "subgraph_agent",
                            "trigger_call_id": "abc-123",
                        },
                    ),
                ),
                ("lifecycle", _protocol_event("lifecycle", {"event": "completed", "namespace": ns})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}))
        lc = [(e["params"]["data"].get("event"), e["params"]["namespace"]) for e in events]
        assert lc == [("running", []), ("started", ns), ("completed", ns), ("completed", [])]
        started = next(e for e in events if e["params"]["data"].get("event") == "started")
        assert started["params"]["data"]["graph_name"] == "subgraph_agent"

    async def test_subgraph_failed_carries_error_on_child_namespace(self, manager: BrokerManager) -> None:
        ns = ["child_node:xyz"]
        await _seed(
            manager,
            "run-1",
            [
                (
                    "lifecycle",
                    _protocol_event(
                        "lifecycle",
                        {"event": "started", "namespace": ns, "graph_name": "child_graph", "trigger_call_id": "xyz"},
                    ),
                ),
                ("lifecycle", _protocol_event("lifecycle", {"event": "failed", "namespace": ns, "error": "boom"})),
                ("error", {"error": "RuntimeError", "message": "boom"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}))
        failed = next(e for e in events if e["params"]["data"].get("event") == "failed")
        assert failed["params"]["namespace"] == ns
        assert failed["params"]["data"]["error"] == "boom"

    async def test_root_scoped_native_lifecycle_dropped(self, manager: BrokerManager) -> None:
        """A native lifecycle whose data.namespace is empty is root-scoped; the
        terminal _lifecycle owns root, so the native one must not double-emit."""
        await _seed(
            manager,
            "run-1",
            [
                ("lifecycle", _protocol_event("lifecycle", {"event": "running", "namespace": []})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}))
        assert [e["params"]["data"] for e in events] == [{"event": "running"}, {"event": "completed"}]

    async def test_terminal_cascades_still_open_subgraph_namespaces(self, manager: BrokerManager) -> None:
        """A cancel mid-subgraph never gets the producer's completed — the terminal
        must close the open namespace so clients don't see it started forever."""
        ns = ["worker:abc"]
        await _seed(
            manager,
            "run-1",
            [
                (
                    "lifecycle",
                    _protocol_event("lifecycle", {"event": "started", "namespace": ns, "graph_name": "worker"}),
                ),
                ("end", {"status": "interrupted"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}))
        lc = [(e["params"]["data"].get("event"), e["params"]["namespace"]) for e in events]
        assert lc == [("running", []), ("started", ns), ("completed", ns), ("interrupted", [])]

    async def test_subgraph_lifecycle_respects_namespace_filter(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                (
                    "lifecycle",
                    _protocol_event("lifecycle", {"event": "started", "namespace": ["sub_a:1"], "graph_name": "a"}),
                ),
                (
                    "lifecycle",
                    _protocol_event("lifecycle", {"event": "started", "namespace": ["sub_b:1"], "graph_name": "b"}),
                ),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"lifecycle"}, namespaces=[["sub_a"]]))
        started = [e["params"]["namespace"] for e in events if e["params"]["data"].get("event") == "started"]
        assert started == [["sub_a:1"]]


class TestValidateChannels:
    def test_valid_channels(self) -> None:
        valid, invalid = validate_channels(["messages", "values", "custom:foo"])
        assert valid == {"messages", "values", "custom:foo"}
        assert invalid == []

    def test_invalid_channels_collected(self) -> None:
        valid, invalid = validate_channels(["messages", "bogus"])
        assert valid == {"messages"}
        assert invalid == ["bogus"]

    def test_empty_list_is_error(self) -> None:
        valid, invalid = validate_channels([])
        assert valid == set()
        assert invalid

    def test_non_list_is_error(self) -> None:
        valid, invalid = validate_channels("messages")
        assert invalid


class TestReplayScope:
    """A connection replays every run the thread's log still holds, as the LangGraph Agent Server does."""

    async def test_every_retained_run_is_replayed(self, manager: BrokerManager) -> None:
        for run_id in ("run-1", "run-2", "run-3"):
            await _seed(
                manager, run_id, [("values", _protocol_event("values", {"r": run_id})), ("end", {"status": "success"})]
            )
        events = await _collect(_make_session("t1", channels={"values", "lifecycle"}))
        assert [e["params"]["data"] for e in events if e["method"] == "values"] == [
            {"r": "run-1"},
            {"r": "run-2"},
            {"r": "run-3"},
        ]

    async def test_message_deltas_of_settled_runs_are_replayed(self, manager: BrokerManager) -> None:
        await _seed(
            manager,
            "run-1",
            [
                _msg_event("message-start", role="ai", id="m1"),
                ("values", _protocol_event("values", {"messages": ["done"]})),
                ("end", {"status": "success"}),
            ],
        )
        events = await _collect(_make_session("t1", channels={"messages", "values", "lifecycle"}))
        assert [e["method"] for e in events] == ["lifecycle", "messages", "values", "lifecycle"]

    async def test_other_threads_are_not_read(self, manager: BrokerManager) -> None:
        await _seed(manager, "run-1", [("end", {"status": "success"})], thread_id="t2")
        events = await _collect(_make_session("t1", channels={"lifecycle"}))
        assert events == []


class TestStaysOpen:
    """The stream stays open and receives new runs from the thread's log, without polling."""

    async def test_run_started_after_the_stream_opened_is_delivered(self, live_manager: BrokerManager) -> None:
        looked_up: list[str] = []

        async def lookup(run_id: str) -> VisibleRun | None:
            looked_up.append(run_id)
            return VisibleRun(graph=None)

        session = _make_session("t1", channels={"values", "lifecycle"}, lookup_run=lookup)
        reader = asyncio.create_task(_read_until_runs_end(session, 1))
        await asyncio.sleep(0.2)  # several empty reads: idle costs no lookups
        assert looked_up == []
        await _seed(
            live_manager, "run-1", [("values", _protocol_event("values", {"a": 1})), ("end", {"status": "success"})]
        )

        events = await reader

        assert looked_up == ["run-1"]
        assert [e["method"] for e in events] == ["lifecycle", "values", "lifecycle"]

    async def test_idle_stream_does_not_close_by_itself(self, live_manager: BrokerManager) -> None:
        stream = _make_session("t1", channels={"lifecycle"}).stream()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(stream.__anext__(), timeout=0.4)
        await stream.aclose()


async def _publish(
    manager: BrokerManager, run_id: str, events: list[tuple[str, Any]], *, thread_id: str = "t1"
) -> None:
    """Write events the way a run does, so values become namespace snapshots."""
    broker = manager.get_or_create_broker(run_id, thread_id)
    for raw in events:
        await broker.put(raw)


def _step(step: int, data: Any, namespace: list[str] | None = None) -> list[tuple[str, Any]]:
    """A superstep's checkpoint envelope and the values it precedes."""
    envelope = {"id": f"c{step}", "step": step}
    return [
        ("checkpoints", _protocol_event("checkpoints", envelope, namespace=namespace, timestamp=step)),
        ("values", _protocol_event("values", data, namespace=namespace, timestamp=step)),
    ]


def _shape(events: list[dict[str, Any]]) -> list[tuple[str, list[str], Any]]:
    return [(e["method"], e["params"]["namespace"], e["params"]["data"]) for e in events]


_STATE = {"values", "checkpoints", "updates", "lifecycle"}


class TestLatestStateReplay:
    """A connection replays each namespace's latest snapshot, then the log after the earliest one."""

    async def test_replay_is_the_latest_snapshot_then_the_later_deltas(self, manager: BrokerManager) -> None:
        update = ("updates", _protocol_event("updates", {"n": {"b": 1}}))
        await _publish(
            manager, "run-1", [*_step(0, {"a": 0}), update, *_step(1, {"a": 1}), update, ("end", {"status": "success"})]
        )

        events = await _collect(_make_session("t1", channels=_STATE))

        assert _shape(events)[1:] == [
            ("checkpoints", [], {"id": "c1", "step": 1}),
            ("values", [], {"a": 1}),
            ("updates", [], {"node": "n", "values": {"b": 1}}),
            ("lifecycle", [], {"event": "completed"}),
        ]
        assert events[0]["params"]["data"] == {"event": "running"}
        assert [e["params"]["timestamp"] for e in events[1:3]] == [1, 1]

    async def test_a_nested_snapshot_older_than_the_roots_starts_the_replay(self, manager: BrokerManager) -> None:
        await _publish(manager, "run-1", [*_step(0, {"r": 0}), *_step(0, {"s": 0}, ["sub:1"]), *_step(1, {"r": 1})])

        events = await _collect(_make_session("t1", channels=_STATE))

        assert [(method, ns) for method, ns, _ in _shape(events)] == [
            ("lifecycle", []),
            ("checkpoints", ["sub:1"]),
            ("values", ["sub:1"]),
            ("checkpoints", []),
            ("values", []),
        ]
        assert _shape(events)[-1] == ("values", [], {"r": 1})

    async def test_a_snapshot_whose_marker_was_trimmed_comes_first(self, manager: BrokerManager) -> None:
        await _publish(
            manager, "run-1", [*_step(0, {"s": 0}, ["sub:1"]), *_step(1, {"r": 1}), ("end", {"status": "success"})]
        )
        manager._log("t1").entries.popleft()  # the subgraph's marker

        events = await _collect(_make_session("t1", channels=_STATE))

        assert [(method, ns) for method, ns, _ in _shape(events)] == [
            ("checkpoints", ["sub:1"]),
            ("values", ["sub:1"]),
            ("lifecycle", []),
            ("checkpoints", []),
            ("values", []),
            ("lifecycle", []),
        ]
        seqs = [e["seq"] for e in events]
        assert seqs == sorted(set(seqs))

    async def test_snapshot_ids_and_seqs_are_their_markers_on_every_connection(self, manager: BrokerManager) -> None:
        await _publish(manager, "run-1", [*_step(0, {"a": 0}), ("end", {"status": "success"})])
        marker = manager._log("t1").entries[0][0]

        first = await _collect(_make_session("t1", channels=_STATE))
        second = await _collect(_make_session("t1", channels=_STATE))

        assert [e["event_id"] for e in first[1:3]] == [f"{marker}:0", f"{marker}:1"]
        assert [(e["event_id"], e["seq"]) for e in first] == [(e["event_id"], e["seq"]) for e in second]

    async def test_a_marker_whose_snapshot_was_replaced_is_skipped(
        self, manager: BrokerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _publish(manager, "run-1", _step(0, {"a": 0}))
        read_snapshots = manager.read_snapshots
        reads: list[list[str]] = []

        async def racing_read(thread_id: str, namespaces: list[str]) -> Any:
            if not reads:  # the run replaces the snapshot after the stream saw the first marker's id
                await _publish(manager, "run-1", _step(1, {"a": 1}))
            reads.append(namespaces)
            return await read_snapshots(thread_id, namespaces)

        monkeypatch.setattr(manager, "read_snapshots", racing_read)

        events = await _collect(_make_session("t1", channels=_STATE))

        assert _shape(events)[1:] == [("checkpoints", [], {"id": "c1", "step": 1}), ("values", [], {"a": 1})]
        assert len(reads) == 1  # the newer snapshot, read once, waited for its own marker

    async def test_a_stream_without_state_channels_reads_no_snapshot(
        self, manager: BrokerManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _publish(manager, "run-1", [*_step(0, {"a": 0}), ("end", {"status": "success"})])

        async def no_read(thread_id: str, namespaces: list[str]) -> Any:
            raise AssertionError("read a snapshot for a stream that wants none")

        monkeypatch.setattr(manager, "read_snapshots", no_read)

        events = await _collect(_make_session("t1", channels={"lifecycle"}))

        assert [e["params"]["data"]["event"] for e in events] == ["running", "completed"]

    async def test_namespace_filter_leaves_out_other_snapshots(self, manager: BrokerManager) -> None:
        await _publish(manager, "run-1", [*_step(0, {"s": 0}, ["sub:1"]), *_step(1, {"r": 1})])

        events = await _collect(_make_session("t1", channels=_STATE, depth=0))

        assert [(method, ns) for method, ns, _ in _shape(events)] == [
            ("lifecycle", []),
            ("checkpoints", []),
            ("values", []),
        ]

    async def test_interrupt_values_are_the_snapshot_and_surface_on_reconnect(self, manager: BrokerManager) -> None:
        interrupted = ("values", _protocol_event("values", {"a": 0}, interrupts=[{"id": "i-1", "value": "q"}]))
        await _publish(manager, "run-1", [*_step(0, {"a": 0}), interrupted, ("end", {"status": "interrupted"})])

        events = await _collect(_make_session("t1", channels={"values", "checkpoints", "input", "lifecycle"}))

        assert _shape(events)[1:] == [
            ("checkpoints", [], {"id": "c0", "step": 0}),
            ("input.requested", [], {"interrupt_id": "i-1", "value": "q", "payload": "q"}),
            ("values", [], {"a": 0}),
            ("lifecycle", [], {"event": "interrupted"}),
        ]

    async def test_a_live_reader_gets_the_interrupt_from_the_snapshot(self, live_manager: BrokerManager) -> None:
        interrupted = ("values", _protocol_event("values", {"a": 0}, interrupts=[{"id": "i-1", "value": "q"}]))
        session = _make_session("t1", channels={"values", "input", "lifecycle"})
        reader = asyncio.create_task(_read_until_runs_end(session, 1))
        await asyncio.sleep(0.1)

        await _publish(live_manager, "run-1", [*_step(0, {"a": 0}), interrupted, ("end", {"status": "interrupted"})])
        events = await reader

        requested = [e["params"]["data"]["interrupt_id"] for e in events if e["method"] == "input.requested"]
        assert requested == ["i-1"]
        assert [e["params"]["data"] for e in events if e["method"] == "values"][-1] == {"a": 0}
