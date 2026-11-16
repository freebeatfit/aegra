"""Publishes run events and run signals to the run's thread event log."""

from typing import Any

import structlog

from aegra_api.services.broker import broker_manager

logger = structlog.getLogger(__name__)


class StreamingService:
    """Writes a run's events and terminal signals to its thread's event log.

    The Agent Protocol v2 thread stream (``ThreadEventSession``) reads that log.
    """

    async def put_to_broker(self, run_id: str, thread_id: str, raw_event: Any) -> None:
        """Append one ``(method, event)`` pair of the run to its thread's log."""
        await broker_manager.get_or_create_broker(run_id, thread_id).put(raw_event)

    async def signal_run_cancelled(self, run_id: str, thread_id: str) -> None:
        """End the run's stream as interrupted, unless it already ended (any instance may call this)."""
        broker = broker_manager.get_or_create_broker(run_id, thread_id)
        if broker.is_finished():
            return
        await broker.put(("end", {"status": "interrupted"}))
        broker_manager.cleanup_broker(run_id)

    async def signal_run_error(
        self, run_id: str, thread_id: str, error_message: str, error_type: str = "Error"
    ) -> None:
        """Send an ``error`` event followed by an ``end`` event, unless the run already ended."""
        broker = broker_manager.get_or_create_broker(run_id, thread_id)
        if broker.is_finished():
            return
        await broker.put(("error", {"error": error_type, "message": error_message}))
        await broker.put(("end", {"status": "error"}))
        broker_manager.cleanup_broker(run_id)

    async def signal_run_end(self, run_id: str, status: str) -> None:
        """End the stream of a run that produced events here, so readers see its terminal."""
        broker = broker_manager.get_broker(run_id)
        if broker is None or broker.is_finished():
            return
        await broker.put(("end", {"status": status}))

    async def interrupt_run(self, run_id: str, *, emit_end_event: bool = True) -> bool:
        """Interrupt a running execution.

        Delegates to broker_manager for cross-instance support via Redis pub/sub.
        """
        try:
            await broker_manager.request_cancel(run_id, "interrupt", emit_end_event=emit_end_event)
            return True
        except Exception as e:
            logger.error(f"Error interrupting run {run_id}: {e}")
            return False

    async def cancel_run(self, run_id: str, *, emit_end_event: bool = True) -> bool:
        """Cancel a pending or running execution.

        Delegates to broker_manager for cross-instance support via Redis pub/sub.
        """
        try:
            await broker_manager.request_cancel(run_id, "cancel", emit_end_event=emit_end_event)
            return True
        except Exception as e:
            logger.error(f"Error cancelling run {run_id}: {e}")
            return False

    def is_run_streaming(self, run_id: str) -> bool:
        """Check if run is currently active (has a broker)."""
        broker = broker_manager.get_broker(run_id)
        return broker is not None and not broker.is_finished()

    async def cleanup_run(self, run_id: str) -> None:
        """Clean up streaming resources for a run."""
        broker_manager.cleanup_broker(run_id)


# Global streaming service instance
streaming_service = StreamingService()
