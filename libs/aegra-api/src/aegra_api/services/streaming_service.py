"""Publishes run events and run signals to the per-run broker."""

from typing import Any

import structlog

from aegra_api.services.broker import broker_manager

logger = structlog.getLogger(__name__)


class StreamingService:
    """Writes a run's events and terminal signals to its broker.

    The broker handles both live broadcast and replay storage; the Agent
    Protocol v2 thread stream (``ThreadEventSession``) reads it.
    """

    async def put_to_broker(
        self,
        run_id: str,
        event_id: str,
        raw_event: Any,
    ) -> None:
        """Put an event into the run's broker for live consumers and replay storage."""
        broker = broker_manager.get_or_create_broker(run_id)
        await broker.put(event_id, raw_event)

    async def signal_run_cancelled(self, run_id: str) -> None:
        """Signal that a run was cancelled.

        Uses get_or_create_broker so this works cross-instance: any instance
        can signal cancellation since the broker is backed by Redis.
        """
        broker = broker_manager.get_or_create_broker(run_id)
        if broker.is_finished():
            return

        event_id = await broker_manager.allocate_event_id(run_id)
        await broker.put(event_id, ("end", {"status": "interrupted"}))
        broker_manager.cleanup_broker(run_id)

    async def signal_run_error(self, run_id: str, error_message: str, error_type: str = "Error") -> None:
        """Signal that a run encountered an error.

        Sends a proper 'error' event to the broker followed by an 'end' event.
        Uses get_or_create_broker so this works cross-instance.
        """
        broker = broker_manager.get_or_create_broker(run_id)
        if broker.is_finished():
            return

        error_event_id = await broker_manager.allocate_event_id(run_id)
        error_payload = {"error": error_type, "message": error_message}
        await broker.put(error_event_id, ("error", error_payload))

        end_event_id = await broker_manager.allocate_event_id(run_id)
        await broker.put(end_event_id, ("end", {"status": "error"}))
        broker_manager.cleanup_broker(run_id)

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
