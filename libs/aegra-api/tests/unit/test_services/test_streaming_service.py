"""Unit tests for streaming_service: run events and signals land in the run's thread log."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.services import streaming_service as streaming_service_module
from aegra_api.services.broker import BrokerManager
from aegra_api.services.streaming_service import StreamingService


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[BrokerManager]:
    mgr = BrokerManager()
    monkeypatch.setattr(streaming_service_module, "broker_manager", mgr)
    yield mgr


def _logged(manager: BrokerManager, thread_id: str = "t1") -> list[tuple[str, Any]]:
    return [(entry.run_id, entry.event) for entry in manager._log(thread_id).entries]


@pytest.mark.asyncio
class TestStreamingService:
    async def test_put_to_broker_appends_to_the_threads_log(self, manager: BrokerManager) -> None:
        await StreamingService().put_to_broker("run-1", "t1", ("updates", {"a": 1}))

        assert _logged(manager) == [("run-1", ("updates", {"a": 1}))]

    async def test_signal_run_cancelled_ends_the_run_once(self, manager: BrokerManager) -> None:
        service = StreamingService()
        await service.put_to_broker("run-1", "t1", ("values", {"a": 1}))

        await service.signal_run_cancelled("run-1", "t1")

        assert _logged(manager)[1:] == [("run-1", ("end", {"status": "interrupted"}))]
        assert manager.get_broker("run-1") is None

    async def test_signal_run_cancelled_skips_an_ended_run(self, manager: BrokerManager) -> None:
        manager.get_or_create_broker("run-1", "t1").mark_finished()

        await StreamingService().signal_run_cancelled("run-1", "t1")

        assert _logged(manager) == []

    async def test_signal_run_error_sends_error_then_end(self, manager: BrokerManager) -> None:
        await StreamingService().signal_run_error("run-1", "t1", "Something went wrong", "ValueError")

        assert _logged(manager) == [
            ("run-1", ("error", {"error": "ValueError", "message": "Something went wrong"})),
            ("run-1", ("end", {"status": "error"})),
        ]
        assert manager.get_broker("run-1") is None

    async def test_signal_run_error_default_type(self, manager: BrokerManager) -> None:
        await StreamingService().signal_run_error("run-1", "t1", "Generic error")

        assert _logged(manager)[0] == ("run-1", ("error", {"error": "Error", "message": "Generic error"}))

    async def test_signal_run_error_skips_an_ended_run(self, manager: BrokerManager) -> None:
        manager.get_or_create_broker("run-1", "t1").mark_finished()

        await StreamingService().signal_run_error("run-1", "t1", "error msg")

        assert _logged(manager) == []

    async def test_events_after_the_end_are_dropped(self, manager: BrokerManager) -> None:
        service = StreamingService()
        await service.put_to_broker("run-1", "t1", ("end", {"status": "success"}))

        await service.put_to_broker("run-1", "t1", ("values", {"late": True}))

        assert _logged(manager) == [("run-1", ("end", {"status": "success"}))]

    async def test_signal_run_end_ends_a_run_that_streamed_here_once(self, manager: BrokerManager) -> None:
        service = StreamingService()
        await service.put_to_broker("run-1", "t1", ("values", {"a": 1}))

        await service.signal_run_end("run-1", "success")
        await service.signal_run_end("run-1", "success")

        assert _logged(manager)[1:] == [("run-1", ("end", {"status": "success"}))]

    async def test_signal_run_end_skips_a_run_without_events_here(self, manager: BrokerManager) -> None:
        await StreamingService().signal_run_end("run-1", "success")

        assert manager.get_broker("run-1") is None

    async def test_interrupt_run_delegates_to_broker_manager(self) -> None:
        with patch("aegra_api.services.streaming_service.broker_manager") as mock_bm:
            mock_bm.request_cancel = AsyncMock()
            success = await StreamingService().interrupt_run("run-123")

        assert success is True
        mock_bm.request_cancel.assert_awaited_once_with("run-123", "interrupt", emit_end_event=True)

    async def test_cancel_run_delegates_to_broker_manager(self) -> None:
        with patch("aegra_api.services.streaming_service.broker_manager") as mock_bm:
            mock_bm.request_cancel = AsyncMock()
            success = await StreamingService().cancel_run("run-123")

        assert success is True
        mock_bm.request_cancel.assert_awaited_once_with("run-123", "cancel", emit_end_event=True)

    async def test_is_run_streaming(self) -> None:
        service = StreamingService()
        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_broker.return_value = mock_broker
            assert service.is_run_streaming("run-1") is True

            mock_broker.is_finished.return_value = True
            assert service.is_run_streaming("run-1") is False

            mock_manager.get_broker.return_value = None
            assert service.is_run_streaming("run-1") is False

    async def test_cleanup_run_forgets_the_run(self, manager: BrokerManager) -> None:
        broker = manager.get_or_create_broker("run-1", "t1")

        await StreamingService().cleanup_run("run-1")

        assert broker.is_finished()
        assert manager.get_broker("run-1") is None
