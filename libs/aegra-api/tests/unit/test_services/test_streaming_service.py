"""Unit tests for streaming_service module"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.services.streaming_service import StreamingService


@pytest.mark.asyncio
class TestStreamingService:
    """Test StreamingService class"""

    async def test_put_to_broker(self) -> None:
        """Test putting event to broker"""
        service = StreamingService()
        run_id = "run-123"
        event_id = "run-123_event_1"
        raw_event = {"data": "test"}

        mock_broker = AsyncMock()

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker

            await service.put_to_broker(run_id, event_id, raw_event)

            mock_manager.get_or_create_broker.assert_called_with(run_id)
            mock_broker.put.assert_awaited_with(event_id, raw_event)

    async def test_signal_run_cancelled(self) -> None:
        """Test signalling run cancellation"""
        service = StreamingService()
        run_id = "run-123"

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False
        mock_broker.put = AsyncMock()

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker
            mock_manager.allocate_event_id = AsyncMock(return_value="run-123_event_1")

            await service.signal_run_cancelled(run_id)

            # Should put end event
            mock_broker.put.assert_awaited()
            args = mock_broker.put.call_args
            assert args[0][1] == ("end", {"status": "interrupted"})

            # Should cleanup broker
            mock_manager.cleanup_broker.assert_called_with(run_id)

    async def test_signal_run_cancelled_skips_when_finished(self) -> None:
        """Test that signal_run_cancelled is a no-op when broker is finished"""
        service = StreamingService()

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = True

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker

            await service.signal_run_cancelled("run-123")

            mock_manager.cleanup_broker.assert_not_called()

    async def test_signal_run_error(self) -> None:
        """Test signalling run error sends error event then end event"""
        service = StreamingService()
        run_id = "run-123"
        error_msg = "Something went wrong"
        error_type = "ValueError"

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False
        mock_broker.put = AsyncMock()

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker
            mock_manager.allocate_event_id = AsyncMock(side_effect=["run-123_event_1", "run-123_event_2"])

            await service.signal_run_error(run_id, error_msg, error_type)

            # Should put error event first, then end event
            assert mock_broker.put.await_count == 2
            call_args_list = mock_broker.put.call_args_list

            # First call should be error event
            first_call = call_args_list[0]
            event_type_val, event_data = first_call[0][1]
            assert event_type_val == "error"
            assert event_data["error"] == error_type
            assert event_data["message"] == error_msg

            # Second call should be end event
            second_call = call_args_list[1]
            end_type, end_data = second_call[0][1]
            assert end_type == "end"
            assert end_data["status"] == "error"

            # Should cleanup broker
            mock_manager.cleanup_broker.assert_called_with(run_id)

    async def test_signal_run_error_skips_when_finished(self) -> None:
        """Test that signal_run_error is a no-op when broker is finished"""
        service = StreamingService()

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = True

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker

            await service.signal_run_error("run-123", "error msg")

            mock_manager.cleanup_broker.assert_not_called()

    async def test_signal_run_error_default_type(self) -> None:
        """Test signal_run_error with default error type"""
        service = StreamingService()
        run_id = "run-123"
        error_msg = "Generic error"

        mock_broker = MagicMock()
        mock_broker.is_finished.return_value = False
        mock_broker.put = AsyncMock()

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            mock_manager.get_or_create_broker.return_value = mock_broker
            mock_manager.allocate_event_id = AsyncMock(side_effect=["run-123_event_1", "run-123_event_2"])

            await service.signal_run_error(run_id, error_msg)

            call_args = mock_broker.put.call_args_list[0]
            event_type_val, event_data = call_args[0][1]
            assert event_type_val == "error"
            assert event_data["error"] == "Error"
            assert event_data["message"] == error_msg

    async def test_interrupt_run_delegates_to_broker_manager(self) -> None:
        """Test run interruption delegates to broker_manager"""
        service = StreamingService()
        run_id = "run-123"

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_bm:
            mock_bm.request_cancel = AsyncMock()
            success = await service.interrupt_run(run_id)

            assert success is True
            mock_bm.request_cancel.assert_awaited_once_with(run_id, "interrupt", emit_end_event=True)

    async def test_cancel_run_delegates_to_broker_manager(self) -> None:
        """Test run cancellation delegates to broker_manager"""
        service = StreamingService()
        run_id = "run-123"

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_bm:
            mock_bm.request_cancel = AsyncMock()
            success = await service.cancel_run(run_id)

            assert success is True
            mock_bm.request_cancel.assert_awaited_once_with(run_id, "cancel", emit_end_event=True)

    async def test_is_run_streaming(self) -> None:
        """Test fetching if run is streaming"""
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

    @pytest.mark.asyncio
    async def test_cleanup_run(self) -> None:
        """Test run cleanup"""
        service = StreamingService()
        run_id = "run-123"

        with patch("aegra_api.services.streaming_service.broker_manager") as mock_manager:
            await service.cleanup_run(run_id)
            mock_manager.cleanup_broker.assert_called_with(run_id)
