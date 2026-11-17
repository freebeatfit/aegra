"""Unit tests for RedisBrokerManager: per-thread streams and cross-instance cancel."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from aegra_api.services.base_broker import READ_BATCH, SNAPSHOT_MARKER
from aegra_api.services.redis_broker import (
    _PUT_MAX_ATTEMPTS,
    _SEQ_TTL_SECONDS,
    _WRITE_SCRIPT,
    RedisBrokerManager,
    _deserialize_payload,
    _serialize_payload,
)
from aegra_api.settings import settings


class TestSerializationHelpers:
    """Test serialization/deserialization helpers"""

    def test_serialize_payload_simple_tuple(self) -> None:
        result = _serialize_payload(("values", {"key": "value"}))
        parsed = json.loads(result)
        assert parsed == ["values", {"key": "value"}]

    def test_serialize_payload_end_event(self) -> None:
        result = _serialize_payload(("end", {"status": "success"}))
        parsed = json.loads(result)
        assert parsed == ["end", {"status": "success"}]

    def test_serialize_payload_dict(self) -> None:
        result = _serialize_payload({"data": "test"})
        parsed = json.loads(result)
        assert parsed == {"data": "test"}

    def test_deserialize_payload_converts_list_to_tuple(self) -> None:
        result = _deserialize_payload(["values", {"key": "value"}])
        assert result == ("values", {"key": "value"})
        assert isinstance(result, tuple)

    def test_deserialize_payload_preserves_dict(self) -> None:
        result = _deserialize_payload({"data": "test"})
        assert result == {"data": "test"}

    def test_deserialize_payload_preserves_non_event_list(self) -> None:
        result = _deserialize_payload([1, 2, 3])
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_deserialize_payload_preserves_empty_list(self) -> None:
        result = _deserialize_payload([])
        assert result == []

    def test_deserialize_end_event(self) -> None:
        result = _deserialize_payload(["end", {"status": "success"}])
        assert result == ("end", {"status": "success"})
        assert isinstance(result, tuple)


def _scripted(script: AsyncMock) -> MagicMock:
    client = MagicMock()
    client.register_script.return_value = script
    return client


_KEYS = [
    "aegra:run:thread:{t-1}:events",
    "aegra:run:thread:{t-1}:snapshot-ids",
    "aegra:run:thread:{t-1}:snapshots",
    "aegra:run:thread:{t-1}:seq",
]


class TestAppend:
    @pytest.mark.asyncio
    async def test_appends_with_one_script_call_on_the_threads_hash_tagged_keys(self) -> None:
        manager = RedisBrokerManager()
        client = _scripted(AsyncMock())

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = client
            await manager.append("t-1", "run-1", "w1", ("updates", {"a": 1}))

        window = settings.event_streaming.RESUMABLE_STREAM_TTL_SECONDS
        client.register_script.assert_called_once_with(_WRITE_SCRIPT)
        client.register_script.return_value.assert_awaited_once_with(
            keys=_KEYS,
            args=[window * 1000, window, "run-1", "w1", "", json.dumps(["updates", {"a": 1}]), _SEQ_TTL_SECONDS, "m"],
        )

    @pytest.mark.asyncio
    async def test_a_snapshot_names_its_namespace_in_the_same_script_call(self) -> None:
        manager = RedisBrokerManager()
        client = _scripted(AsyncMock())

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = client
            await manager.put_snapshot(
                "t-1", run_id="run-1", write_id="w1", namespace="[]", record={"values": {"a": 1}}
            )

        _, kwargs = client.register_script.return_value.call_args
        assert kwargs["keys"] == _KEYS
        assert kwargs["args"][2:] == ["run-1", "w1", "[]", json.dumps({"values": {"a": 1}}), _SEQ_TTL_SECONDS, "m"]

    @pytest.mark.asyncio
    async def test_retries_a_transient_failure_with_the_same_write_id(self) -> None:
        manager = RedisBrokerManager()
        client = _scripted(AsyncMock(side_effect=[RedisConnectionError("blip"), None]))

        with (
            patch("aegra_api.services.redis_broker.redis_manager") as mock_rm,
            patch("aegra_api.services.redis_broker.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_rm.get_client.return_value = client
            await manager.append("t-1", "run-1", "w1", ("updates", {"a": 1}))

        writes = client.register_script.return_value.call_args_list
        assert len(writes) == 2
        assert writes[0] == writes[1]
        mock_sleep.assert_awaited()

    @pytest.mark.asyncio
    async def test_gives_up_after_bounded_attempts_without_raising(self) -> None:
        manager = RedisBrokerManager()
        client = _scripted(AsyncMock(side_effect=RedisConnectionError("Redis down")))

        with (
            patch("aegra_api.services.redis_broker.redis_manager") as mock_rm,
            patch("aegra_api.services.redis_broker.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_rm.get_client.return_value = client
            written = await manager.append("t-1", "run-1", "w1", ("updates", {"a": 1}))

        assert written is False
        assert client.register_script.return_value.await_count == _PUT_MAX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_end_event_marks_the_run_finished_even_when_its_write_fails(self) -> None:
        manager = RedisBrokerManager()
        client = _scripted(AsyncMock(side_effect=RedisConnectionError("down")))

        with (
            patch("aegra_api.services.redis_broker.redis_manager") as mock_rm,
            patch("aegra_api.services.redis_broker.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_rm.get_client.return_value = client
            broker = manager.get_or_create_broker("run-1", "t-1")
            await broker.put(("end", {"status": "success"}))

        assert broker.is_finished()

    @pytest.mark.asyncio
    async def test_run_writes_do_not_interleave(self) -> None:
        """A signal racing a retrying write lands after it, so readers never see an event after the end."""
        manager = RedisBrokerManager()
        order: list[str] = []
        gate = asyncio.Event()

        async def append(thread_id: str, run_id: str, write_id: str, event: Any) -> bool:
            order.append(f"{event[0]}-start")
            if event[0] == "updates":
                await gate.wait()
            order.append(f"{event[0]}-end")
            return True

        manager.append = append  # type: ignore[method-assign]
        broker = manager.get_or_create_broker("run-1", "t-1")
        first = asyncio.create_task(broker.put(("updates", {})))
        await asyncio.sleep(0)
        second = asyncio.create_task(broker.put(("end", {})))
        await asyncio.sleep(0)
        assert order == ["updates-start"]

        gate.set()
        await asyncio.gather(first, second)

        assert order == ["updates-start", "updates-end", "end-start", "end-end"]


class TestReadThread:
    @staticmethod
    def _fields(run_id: str, write_id: str, event: Any) -> dict[str, str]:
        return {"s": "14336", "r": run_id, "w": write_id, "e": json.dumps(event)}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("resp3", [False, True])
    async def test_reads_entries_after_the_cursor_with_a_bounded_block(self, resp3: bool) -> None:
        manager = RedisBrokerManager()
        key = "aegra:run:thread:{t-1}:events"
        entries = [("5-0", self._fields("run-1", "w1", ["values", {"a": 1}]))]
        client = MagicMock()
        client.xread = AsyncMock(return_value={key: [entries]} if resp3 else [[key, entries]])

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_stream_read_client.return_value = client
            mock_rm.stream_read_block_ms = 5000
            result = await manager.read_thread("t-1", "4-0")

        assert result == [("5-0", 14336, "run-1", "w1", ("values", {"a": 1}))]
        client.xread.assert_awaited_once_with({key: "4-0"}, count=READ_BATCH, block=5000)

    @pytest.mark.asyncio
    async def test_a_marker_entry_reads_as_its_namespace(self) -> None:
        manager = RedisBrokerManager()
        key = "aegra:run:thread:{t-1}:events"
        client = MagicMock()
        client.xread = AsyncMock(
            return_value=[[key, [("5-0", {"s": "14336", "r": "run-1", "w": "w1", "m": '["sub"]'})]]]
        )

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_stream_read_client.return_value = client
            result = await manager.read_thread("t-1", "4-0")

        assert result == [("5-0", 14336, "run-1", "w1", (SNAPSHOT_MARKER, '["sub"]'))]

    @pytest.mark.asyncio
    async def test_an_entry_written_by_fb6_reads_with_the_seq_fb6_gave_it(self) -> None:
        manager = RedisBrokerManager()
        key = "aegra:run:thread:{t-1}:events"
        client = MagicMock()
        client.xread = AsyncMock(return_value=[[key, [("5-2", {"r": "run-1", "w": "w1", "e": '["updates", {}]'})]]])

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_stream_read_client.return_value = client
            [entry] = await manager.read_thread("t-1", "4-0")

        assert entry.seq == ((5 << 8) + 2) << 3

    @pytest.mark.asyncio
    async def test_block_timeout_returns_no_entries(self) -> None:
        manager = RedisBrokerManager()
        client = MagicMock()
        client.xread = AsyncMock(return_value=[])

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_stream_read_client.return_value = client
            assert await manager.read_thread("t-1", "0-0") == []

    @pytest.mark.asyncio
    async def test_a_failed_read_raises(self) -> None:
        manager = RedisBrokerManager()
        client = MagicMock()
        client.xread = AsyncMock(side_effect=RedisConnectionError("down"))

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_stream_read_client.return_value = client
            with pytest.raises(RedisConnectionError):
                await manager.read_thread("t-1", "0-0")

    @pytest.mark.asyncio
    async def test_cancelled_reader_still_releases_its_connection(self) -> None:
        """anyio re-cancels every await of a cancelled scope; the read's cleanup must still finish."""
        manager = RedisBrokerManager()
        released = asyncio.Event()

        async def blocking_xread(*_args: Any, **_kwargs: Any) -> list[Any]:
            try:
                await asyncio.sleep(60)
            finally:
                await asyncio.sleep(0.01)  # an await in cleanup, like redis-py's pool release
                released.set()
            return []

        client = MagicMock()
        client.xread = blocking_xread

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_stream_read_client.return_value = client
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(manager.read_thread, "t-1", "0-0")
                await asyncio.sleep(0.01)
                task_group.cancel_scope.cancel()

        assert released.is_set()


class TestRedisBrokerManager:
    """Test RedisBrokerManager class"""

    def _make_manager(self) -> RedisBrokerManager:
        return RedisBrokerManager()

    def test_stream_key_carries_the_thread_hash_tag(self) -> None:
        assert self._make_manager().stream_key("t-1") == "aegra:run:thread:{t-1}:events"

    def test_cleanup_broker(self) -> None:
        manager = self._make_manager()
        broker = manager.get_or_create_broker("run-123", "t-1")
        manager.cleanup_broker("run-123")

        assert broker.is_finished()
        assert manager.get_broker("run-123") is None

    @pytest.mark.asyncio
    async def test_request_cancel_publishes_to_redis(self) -> None:
        """Test that request_cancel publishes cancel command to Redis"""
        manager = self._make_manager()
        mock_client = MagicMock()
        mock_client.publish = AsyncMock()

        with patch("aegra_api.services.redis_broker.redis_manager") as mock_rm:
            mock_rm.get_client.return_value = mock_client

            await manager.request_cancel("run-123", "cancel")

            mock_client.publish.assert_awaited_once()
            channel, message = mock_client.publish.call_args[0]
            assert channel == "aegra:run:cancel"
            payload = json.loads(message)
            assert payload == {"run_id": "run-123", "action": "cancel", "emit_end_event": True}

    @pytest.mark.asyncio
    async def test_request_cancel_falls_back_on_redis_error(self) -> None:
        """Test that request_cancel falls back to local execution on Redis error"""
        manager = self._make_manager()
        mock_client = MagicMock()
        mock_client.publish = AsyncMock(side_effect=RedisConnectionError("Redis down"))

        with (
            patch("aegra_api.services.redis_broker.redis_manager") as mock_rm,
            patch.object(manager, "_execute_cancel", new_callable=AsyncMock) as mock_exec,
        ):
            mock_rm.get_client.return_value = mock_client

            await manager.request_cancel("run-123", "cancel")

            mock_exec.assert_awaited_once_with("run-123", emit_end_event=True)

    @pytest.mark.asyncio
    async def test_execute_cancel_cancels_local_task(self) -> None:
        """Test that _execute_cancel cancels a locally-owned task"""
        manager = self._make_manager()
        mock_task = MagicMock()
        mock_task.done.return_value = False

        manager.append = AsyncMock()  # type: ignore[method-assign]
        broker = manager.get_or_create_broker("run-123", "t-1")

        with (
            patch.dict("aegra_api.core.active_runs.active_runs", {"run-123": mock_task}, clear=True),
            patch("aegra_api.services.redis_broker.explicit_run_cancellations", set()) as cancellations,
        ):
            await manager._execute_cancel("run-123")

        mock_task.cancel.assert_called_once()
        _thread_id, run_id, _write_id, event = manager.append.await_args.args
        assert (run_id, event) == ("run-123", ("end", {"status": "interrupted"}))
        # Kept registered and finished, so execute_run's cancel path skips a second end.
        assert broker.is_finished() and manager.get_broker("run-123") is broker
        assert cancellations == {"run-123"}

    @pytest.mark.asyncio
    async def test_execute_cancel_can_leave_terminal_signaling_to_api(self) -> None:
        manager = self._make_manager()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        manager.append = AsyncMock()  # type: ignore[method-assign]
        manager.get_or_create_broker("run-123", "t-1")

        with (
            patch.dict("aegra_api.core.active_runs.active_runs", {"run-123": mock_task}, clear=True),
            patch("aegra_api.services.redis_broker.explicit_run_cancellations", set()) as cancellations,
        ):
            await manager._execute_cancel("run-123", emit_end_event=False)

        mock_task.cancel.assert_called_once()
        manager.append.assert_not_awaited()
        assert cancellations == {"run-123"}

    @pytest.mark.asyncio
    async def test_execute_cancel_ignores_missing_task(self) -> None:
        """Test that _execute_cancel does nothing when task not found locally"""
        manager = self._make_manager()

        with patch.dict("aegra_api.core.active_runs.active_runs", {}, clear=True):
            await manager._execute_cancel("run-123")
            # Should not raise

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        """Test start creates listener task and stop cancels it"""
        manager = self._make_manager()

        with patch.object(manager, "_listen_for_cancel_commands", new_callable=AsyncMock):
            await manager.start()
            assert manager._running is True
            assert manager._listener_task is not None

            await manager.stop()
            assert manager._running is False
