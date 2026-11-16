"""Unit tests for the in-memory broker: per-thread logs, run writers, cancel."""

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aegra_api.services import broker as broker_module
from aegra_api.services.base_broker import SNAPSHOT_MARKER, entry_key
from aegra_api.services.broker import BrokerManager
from aegra_api.settings import settings

RETENTION = settings.event_streaming.RESUMABLE_STREAM_TTL_SECONDS


def _event(method: str, data: Any, namespace: list[str] | None = None, **params: Any) -> tuple[str, dict[str, Any]]:
    return method, {"type": "event", "method": method, "params": {"namespace": namespace or [], "data": data, **params}}


class TestRunBroker:
    @pytest.mark.asyncio
    async def test_put_appends_to_its_threads_log(self) -> None:
        manager = BrokerManager()
        broker = manager.get_or_create_broker("run-1", "t1")

        await broker.put(("updates", {"a": 1}))

        [(entry_id, run_id, _write_id, event)] = await manager.read_thread("t1", "0-0")
        assert (run_id, event) == ("run-1", ("updates", {"a": 1}))
        assert entry_key(entry_id) > (0, 0)

    @pytest.mark.asyncio
    async def test_values_replace_the_namespace_snapshot_with_its_envelope_and_append_a_marker(self) -> None:
        manager = BrokerManager()
        broker = manager.get_or_create_broker("run-1", "t1")
        checkpoint, values = _event("checkpoints", {"id": "c1", "step": 1}), _event("values", {"a": 1}, timestamp=7)

        await broker.put(checkpoint)
        await broker.put(values)
        await broker.put(_event("values", {"a": 2}))

        entries = await manager.read_thread("t1", "0-0")
        assert [event for *_rest, event in entries] == [(SNAPSHOT_MARKER, "[]"), (SNAPSHOT_MARKER, "[]")]
        assert await manager.read_snapshot_ids("t1") == {"[]": entries[1][0]}
        snapshots = await manager.read_snapshots("t1", ["[]", '["sub"]'])
        assert snapshots == {
            "[]": (entries[1][0], {"run_id": "run-1", "checkpoint": None, "values": _event("values", {"a": 2})[1]})
        }

    @pytest.mark.asyncio
    async def test_snapshot_keeps_the_envelope_written_just_before_it(self) -> None:
        manager = BrokerManager()
        broker = manager.get_or_create_broker("run-1", "t1")
        checkpoint, values = _event("checkpoints", {"id": "c1"}, ["sub"]), _event("values", {"a": 1}, ["sub"])

        await broker.put(checkpoint)
        await broker.put(values)

        [(entry_id, record)] = (await manager.read_snapshots("t1", ['["sub"]'])).values()
        assert record == {"run_id": "run-1", "checkpoint": checkpoint[1], "values": values[1]}

    @pytest.mark.asyncio
    async def test_values_with_interrupts_and_unpaired_envelopes_stay_in_the_log(self) -> None:
        manager = BrokerManager()
        broker = manager.get_or_create_broker("run-1", "t1")
        first, second = _event("checkpoints", {"id": "c1"}), _event("checkpoints", {"id": "c2"}, ["sub"])
        interrupted = _event("values", {"a": 1}, interrupts=[{"id": "i1", "value": "q"}])

        for event in (first, second, interrupted, ("end", {"status": "interrupted"})):
            await broker.put(event)

        entries = await manager.read_thread("t1", "0-0")
        assert [event for *_rest, event in entries] == [first, second, interrupted, ("end", {"status": "interrupted"})]
        assert await manager.read_snapshot_ids("t1") == {}

    @pytest.mark.asyncio
    async def test_end_event_marks_finished_and_later_events_are_dropped(self) -> None:
        manager = BrokerManager()
        broker = manager.get_or_create_broker("run-1", "t1")

        await broker.put(("end", {"status": "success"}))
        await broker.put(("values", {"late": True}))

        assert broker.is_finished()
        assert [event for *_rest, event in await manager.read_thread("t1", "0-0")] == [("end", {"status": "success"})]


class TestThreadLog:
    @pytest.mark.asyncio
    async def test_entry_ids_rise_with_every_write_across_runs(self) -> None:
        manager = BrokerManager()
        for i in range(5):
            await manager.append("t1", f"run-{i % 2}", f"w{i}", ("updates", {"i": i}))

        entries = await manager.read_thread("t1", "0-0")

        keys = [entry_key(entry_id) for entry_id, *_ in entries]
        assert keys == sorted(set(keys))
        assert [run_id for _, run_id, _, _ in entries] == ["run-0", "run-1", "run-0", "run-1", "run-0"]

    @pytest.mark.asyncio
    async def test_read_returns_only_entries_after_the_cursor(self) -> None:
        manager = BrokerManager()
        for i in range(3):
            await manager.append("t1", "run-1", f"w{i}", ("updates", {"i": i}))
        first, *_ = await manager.read_thread("t1", "0-0")

        rest = await manager.read_thread("t1", first[0])

        assert [event for *_rest, event in rest] == [("updates", {"i": 1}), ("updates", {"i": 2})]

    @pytest.mark.asyncio
    async def test_threads_are_separate(self) -> None:
        manager = BrokerManager()
        await manager.append("t2", "run-1", "w1", ("end", {}))

        with patch.object(broker_module, "_READ_WAIT_SECONDS", 0.01):
            assert await manager.read_thread("t1", "0-0") == []

    @pytest.mark.asyncio
    async def test_blocked_read_returns_when_a_write_lands(self) -> None:
        manager = BrokerManager()
        reader = asyncio.create_task(manager.read_thread("t1", "0-0"))
        await asyncio.sleep(0.05)
        assert not reader.done()

        await manager.append("t1", "run-1", "w1", ("values", {"a": 1}))

        entries = await asyncio.wait_for(reader, timeout=1.0)
        assert [event for *_rest, event in entries] == [("values", {"a": 1})]

    @pytest.mark.asyncio
    async def test_idle_read_returns_empty_after_its_wait(self) -> None:
        manager = BrokerManager()
        with patch.object(broker_module, "_READ_WAIT_SECONDS", 0.01):
            assert await manager.read_thread("t1", "0-0") == []


class TestBrokerManager:
    def test_get_or_create_broker_returns_the_same_writer(self) -> None:
        manager = BrokerManager()

        first = manager.get_or_create_broker("run-1", "t1")

        assert manager.get_or_create_broker("run-1", "t1") is first
        assert manager.get_broker("run-1") is first
        assert first.thread_id == "t1"

    def test_get_nonexistent_broker(self) -> None:
        assert BrokerManager().get_broker("nonexistent") is None

    def test_cleanup_broker_finishes_and_forgets_it(self) -> None:
        manager = BrokerManager()
        broker = manager.get_or_create_broker("run-1", "t1")

        manager.cleanup_broker("run-1")
        manager.cleanup_broker("nonexistent")

        assert broker.is_finished()
        assert manager.get_broker("run-1") is None

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        manager = BrokerManager()

        await manager.start()
        assert manager._cleanup_task is not None
        assert not manager._cleanup_task.done()

        await manager.stop()
        assert manager._cleanup_task.cancelled() or manager._cleanup_task.done()

    @pytest.mark.asyncio
    async def test_request_cancel_cancels_the_local_task_and_ends_its_stream(self) -> None:
        manager = BrokerManager()
        await manager.get_or_create_broker("run-1", "t1").put(("values", {"a": 1}))
        task = MagicMock()
        task.done.return_value = False

        with patch.dict(broker_module.active_runs, {"run-1": task}, clear=True):
            await manager.request_cancel("run-1")

        task.cancel.assert_called_once()
        broker = manager.get_broker("run-1")
        assert broker is not None and broker.is_finished()
        assert (await manager.read_thread("t1", "0-0"))[-1][3] == ("end", {"status": "interrupted"})

    @pytest.mark.asyncio
    async def test_request_cancel_can_leave_terminal_signaling_to_the_api(self) -> None:
        manager = BrokerManager()
        broker = manager.get_or_create_broker("run-1", "t1")
        task = MagicMock()
        task.done.return_value = False

        with patch.dict(broker_module.active_runs, {"run-1": task}, clear=True):
            await manager.request_cancel("run-1", emit_end_event=False)

        task.cancel.assert_called_once()
        assert not broker.is_finished()


class TestReplayRetention:
    @pytest.mark.asyncio
    async def test_log_survives_inside_the_window(self) -> None:
        manager = BrokerManager()
        await manager.append("t1", "run-1", "w1", ("end", {}))
        manager._logs["t1"].last_write -= RETENTION - 60

        assert manager.sweep_expired_logs() == []

    @pytest.mark.asyncio
    async def test_log_is_swept_past_the_window_after_its_last_write(self) -> None:
        manager = BrokerManager()
        await manager.append("t1", "run-1", "w1", ("end", {}))
        manager._logs["t1"].last_write -= RETENTION + 60

        assert manager.sweep_expired_logs() == ["t1"]
        with patch.object(broker_module, "_READ_WAIT_SECONDS", 0.01):
            assert await manager.read_thread("t1", "0-0") == []

    @pytest.mark.asyncio
    async def test_writes_trim_entries_older_than_the_window_but_not_from_the_root_snapshot_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = BrokerManager()
        broker = manager.get_or_create_broker("run-1", "t1")
        clock = [1_000_000_000_000_000_000]
        monkeypatch.setattr(broker_module.time, "time_ns", lambda: clock[0])
        await manager.append("t1", "run-1", "w0", ("updates", {"i": 0}))
        clock[0] += 1_000_000_000
        await broker.put(_event("values", {"a": 1}))
        await manager.append("t1", "run-1", "w1", ("updates", {"i": 1}))

        clock[0] += (RETENTION + 1) * 1_000_000_000
        await manager.append("t1", "run-1", "w2", ("updates", {"i": 2}))

        events = [event for *_rest, event in await manager.read_thread("t1", "0-0")]
        assert events == [(SNAPSHOT_MARKER, "[]"), ("updates", {"i": 1}), ("updates", {"i": 2})]
