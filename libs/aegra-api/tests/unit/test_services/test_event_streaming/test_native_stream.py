"""Tests for the native v3 stream producer (forwarding + message-tuple unwrap)."""

from __future__ import annotations

from typing import Any

import pytest

from aegra_api.models.runs import Durability
from aegra_api.services.event_streaming.native_stream import (
    stream_native_v3_events,
    unwrap_message_event,
)


class _FakeRunStream:
    """Mimics langgraph's AsyncGraphRunStream: async-iterable + async context."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeRunStream:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    def __aiter__(self) -> _FakeRunStream:
        self._it = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


class _FakeGraph:
    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.calls: list[dict[str, Any]] = []

    async def astream_events(self, input_data: Any, config: Any, **kwargs: Any) -> _FakeRunStream:
        self.calls.append({"input": input_data, "config": config, **kwargs})
        return _FakeRunStream(self._events)


def _event(method: str, data: Any, *, namespace: list[str] | None = None) -> dict[str, Any]:
    return {"type": "event", "method": method, "params": {"namespace": namespace or [], "data": data}}


class TestUnwrapMessageEvent:
    def test_unwraps_event_meta_tuple(self) -> None:
        """v3 message data arrives as [event_dict, metadata]; element 0 is the event."""
        data = [{"event": "content-block-delta", "index": 0, "delta": {"type": "text-delta", "text": "hi"}}, {"m": 1}]
        assert unwrap_message_event(data) == {
            "event": "content-block-delta",
            "index": 0,
            "delta": {"type": "text-delta", "text": "hi"},
        }

    def test_bare_event_dict_passes_through(self) -> None:
        data = {"event": "message-start", "id": "m1"}
        assert unwrap_message_event(data) == data

    def test_non_event_tuple_returns_none(self) -> None:
        assert unwrap_message_event([{"no_event_key": 1}, {}]) is None

    def test_non_message_shape_returns_none(self) -> None:
        assert unwrap_message_event("nonsense") is None


@pytest.mark.asyncio
class TestStreamNativeV3Events:
    async def test_drives_v3_and_forwards_method_event_pairs(self) -> None:
        events = [
            _event("values", {"messages": []}),
            _event("messages", [{"event": "message-start", "id": "m1"}, {}]),
        ]
        graph = _FakeGraph(events)

        out: list[tuple[str, Any]] = []
        async for method, payload in stream_native_v3_events(graph=graph, input_data={"x": 1}, config={"c": 2}):
            out.append((method, payload))

        # v3 is requested with version="v3"
        assert graph.calls[0]["version"] == "v3"
        # values forwarded as-is
        assert out[0][0] == "values"
        assert out[0][1]["params"]["data"] == {"messages": []}
        # messages: tuple unwrapped so params.data is the event dict
        assert out[1][0] == "messages"
        assert out[1][1]["params"]["data"] == {"event": "message-start", "id": "m1"}

    async def test_skips_non_event_dicts(self) -> None:
        graph = _FakeGraph([{"not": "an event"}, _event("values", {"a": 1})])
        out = [pair async for pair in stream_native_v3_events(graph=graph, input_data={}, config={})]
        assert len(out) == 1
        assert out[0][0] == "values"

    async def test_drops_unreconstructable_message_events(self) -> None:
        """A messages event whose data isn't an event tuple/dict is dropped, not forwarded raw."""
        graph = _FakeGraph([_event("messages", [{"no_event": 1}, {}])])
        out = [pair async for pair in stream_native_v3_events(graph=graph, input_data={}, config={})]
        assert out == []

    @pytest.mark.parametrize("mode", ["sync", "async", "exit"])
    async def test_forwards_durability_to_the_v3_run(self, mode: Durability) -> None:
        """v3 hands extra kwargs to astream, which is where LangGraph reads durability."""
        graph = _FakeGraph([_event("values", {"a": 1})])
        out = [pair async for pair in stream_native_v3_events(graph=graph, input_data={}, config={}, durability=mode)]
        assert out
        assert graph.calls[0]["durability"] == mode

    async def test_omits_durability_when_unset(self) -> None:
        graph = _FakeGraph([_event("values", {"a": 1})])
        out = [pair async for pair in stream_native_v3_events(graph=graph, input_data={}, config={})]
        assert out
        assert "durability" not in graph.calls[0]

    async def test_passes_breakpoints_as_arguments_not_config(self) -> None:
        """LangGraph ignores interrupt_before/after in the config; they must be stream arguments."""
        graph = _FakeGraph([_event("values", {"a": 1})])
        config = {"configurable": {}, "interrupt_before": ["*"], "interrupt_after": ["review"]}

        out = [pair async for pair in stream_native_v3_events(graph=graph, input_data={}, config=config)]

        assert out
        assert graph.calls[0]["interrupt_before"] == "*"
        assert graph.calls[0]["interrupt_after"] == ["review"]
        assert graph.calls[0]["config"] == {"configurable": {}}


def _real_graph() -> Any:
    """A two-node graph with a checkpointer, so v3 emits values + checkpoints."""
    from typing import TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        messages: list[str]

    def first(state: State) -> State:
        return {"messages": [*state["messages"], "x" * 1000]}

    def second(state: State) -> State:
        return {"messages": [*state["messages"], "y" * 1000]}

    builder = StateGraph(State)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    return builder.compile(checkpointer=InMemorySaver())


@pytest.mark.asyncio
class TestOnlyEnvelopeTransformer:
    async def test_adds_no_updates_custom_or_tasks_channels(self) -> None:
        """Aegra adds only the checkpoint envelope; other channels are graph opt-in."""
        graph = _real_graph()
        config = {"configurable": {"thread_id": "t-optin"}}
        methods = {m async for m, _ in stream_native_v3_events(graph=graph, input_data={"messages": []}, config=config)}
        assert "updates" not in methods
        assert "tasks" not in methods
        assert "custom" not in methods
        assert {"values", "checkpoints"} <= methods

    async def test_checkpoints_are_lightweight_envelopes(self) -> None:
        """No full state on the checkpoints channel: only id/parent_id/step/source."""
        graph = _real_graph()
        config = {"configurable": {"thread_id": "t-envelope"}}
        checkpoints = [
            event["params"]["data"]
            async for method, event in stream_native_v3_events(graph=graph, input_data={"messages": []}, config=config)
            if method == "checkpoints"
        ]
        assert checkpoints
        for envelope in checkpoints:
            assert set(envelope) <= {"id", "parent_id", "step", "source"}
            assert isinstance(envelope["id"], str)
            assert isinstance(envelope["step"], int)
        assert all("x" * 1000 not in repr(envelope) for envelope in checkpoints)
        # Each envelope links to the previous checkpoint.
        assert [e.get("parent_id") for e in checkpoints[1:]] == [e["id"] for e in checkpoints[:-1]]

    async def test_values_still_carry_full_state(self) -> None:
        graph = _real_graph()
        config = {"configurable": {"thread_id": "t-values"}}
        values = [
            event["params"]["data"]
            async for method, event in stream_native_v3_events(graph=graph, input_data={"messages": []}, config=config)
            if method == "values"
        ]
        assert values[-1]["messages"] == ["x" * 1000, "y" * 1000]


class TestCheckpointEnvelope:
    def test_reduces_a_snapshot(self) -> None:
        from aegra_api.services.event_streaming.native_stream import checkpoint_envelope

        snapshot = {
            "config": {"configurable": {"checkpoint_id": "c2", "checkpoint_ns": ""}},
            "parent_config": {"configurable": {"checkpoint_id": "c1"}},
            "metadata": {"step": 3, "source": "loop", "parents": {}},
            "values": {"messages": ["big"]},
            "next": ["model"],
            "tasks": [],
        }
        assert checkpoint_envelope(snapshot) == {"id": "c2", "parent_id": "c1", "step": 3, "source": "loop"}

    def test_root_checkpoint_has_no_parent(self) -> None:
        from aegra_api.services.event_streaming.native_stream import checkpoint_envelope

        snapshot = {"config": {"configurable": {"checkpoint_id": "c1"}}, "metadata": {"step": -1, "source": "input"}}
        assert checkpoint_envelope(snapshot) == {"id": "c1", "step": -1, "source": "input"}

    def test_envelope_passes_through(self) -> None:
        from aegra_api.services.event_streaming.native_stream import checkpoint_envelope

        envelope = {"id": "c1", "step": 0, "source": "loop"}
        assert checkpoint_envelope(envelope) == envelope


@pytest.mark.asyncio
class TestEnvelopeOrder:
    async def test_each_values_follows_its_checkpoint_envelope(self) -> None:
        """The SDK pairs a buffered envelope with the next values on the namespace."""
        graph = _real_graph()
        config = {"configurable": {"thread_id": "t-order"}}
        seq = [
            (method, event["params"]["data"])
            async for method, event in stream_native_v3_events(graph=graph, input_data={"messages": []}, config=config)
            if method in ("values", "checkpoints")
        ]
        for index, (method, data) in enumerate(seq):
            if method != "values":
                continue
            assert index > 0 and seq[index - 1][0] == "checkpoints"
            # The envelope paired with values carrying N messages is step N - 1 (loop steps start at 0).
            assert seq[index - 1][1]["step"] == len(data["messages"])

    async def test_a_lone_values_event_is_not_lost(self) -> None:
        graph = _FakeGraph([_event("values", {"a": 1}), _event("messages", [{"event": "message-start"}, {}])])
        out = [method async for method, _ in stream_native_v3_events(graph=graph, input_data={}, config={})]
        assert out == ["values", "messages"]

    async def test_a_trailing_values_event_is_flushed(self) -> None:
        graph = _FakeGraph([_event("values", {"a": 1})])
        out = [method async for method, _ in stream_native_v3_events(graph=graph, input_data={}, config={})]
        assert out == ["values"]

    async def test_checkpoint_from_another_namespace_does_not_pair(self) -> None:
        graph = _FakeGraph(
            [
                _event("values", {"a": 1}),
                _event("checkpoints", {"id": "c", "step": 0}, namespace=["sub:1"]),
            ]
        )
        out = [method async for method, _ in stream_native_v3_events(graph=graph, input_data={}, config={})]
        assert out == ["values", "checkpoints"]
