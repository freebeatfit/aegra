"""Stateless (thread-free) run endpoints.

These endpoints accept POST /runs/wait and /runs without a thread_id. They
generate an ephemeral thread, delegate to the existing threaded endpoint
functions, and clean up the thread afterward (unless the caller explicitly sets
``on_completion="keep"``). POST /runs/stream is retired (410): streaming is
Agent Protocol v2 only.
"""

from collections.abc import AsyncIterator
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.api.runs import V1_STREAMING_GONE, create_run, wait_for_run
from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.orm import get_session
from aegra_api.models import Run, RunCreate, User
from aegra_api.models.errors import CONFLICT, NOT_FOUND, AgentProtocolError
from aegra_api.services.run_cleanup import (
    _CLEANUP_ERRORS,
    delete_thread_by_id,
    schedule_background_cleanup,
)

router = APIRouter(tags=["Stateless Runs"], dependencies=auth_dependency)
logger = structlog.getLogger(__name__)


async def _delete_thread_with_log(thread_id: str, user_id: str, *, reason: str) -> None:
    """Delete an ephemeral thread, logging infra failures."""
    try:
        await delete_thread_by_id(thread_id, user_id)
    except _CLEANUP_ERRORS:
        logger.exception(reason, thread_id=thread_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/runs/wait", responses={**NOT_FOUND, **CONFLICT})
async def stateless_wait_for_run(
    request: RunCreate,
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Create a stateless run and wait for completion.

    Generates an ephemeral thread, delegates to the threaded ``wait_for_run``
    endpoint, and deletes the thread after the response finishes streaming
    (unless ``on_completion="keep"``).
    """
    thread_id = str(uuid4())
    should_delete = request.on_completion != "keep"

    try:
        response = await wait_for_run(thread_id, request, user)
    except Exception:
        if should_delete:
            try:
                await delete_thread_by_id(thread_id, user.identity)
            except _CLEANUP_ERRORS:
                logger.exception(
                    "Failed to delete ephemeral thread after wait error",
                    thread_id=thread_id,
                )
        raise

    if not should_delete:
        return response

    # Wrap the body_iterator so cleanup happens after the stream ends.
    original_iterator = response.body_iterator

    async def _wrapped_iterator() -> AsyncIterator[bytes]:
        completed = False
        try:
            async for chunk in original_iterator:
                yield chunk
            completed = True
        finally:
            aclose = getattr(original_iterator, "aclose", None)
            if aclose is not None:
                await aclose()
            if completed:
                await _delete_thread_with_log(
                    thread_id, user.identity, reason="Failed to delete ephemeral thread after wait"
                )
            else:
                logger.info(
                    "Client disconnected before stream completed, keeping ephemeral thread",
                    thread_id=thread_id,
                )

    return StreamingResponse(
        _wrapped_iterator(),
        status_code=response.status_code,
        media_type=response.media_type,
        headers=dict(response.headers),
    )


@router.post("/runs/stream", status_code=410, response_model=AgentProtocolError, deprecated=True)
async def stateless_stream_run() -> AgentProtocolError:
    """Retired: use Agent Protocol v2 (`POST /threads/{thread_id}/commands`, then `/stream/events`)."""
    return V1_STREAMING_GONE


@router.post("/runs", response_model=Run, responses={**NOT_FOUND, **CONFLICT})
async def stateless_create_run(
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Create a stateless background run.

    Generates an ephemeral thread, delegates to the threaded ``create_run``
    endpoint, and schedules cleanup as a background task (unless
    ``on_completion="keep"``).
    """
    thread_id = str(uuid4())
    should_delete = request.on_completion != "keep"

    try:
        result = await create_run(thread_id, request, user, session)
    except Exception:
        # create_run may have auto-created the thread via
        # update_thread_metadata before raising; clean up to avoid orphans.
        if should_delete:
            try:
                await delete_thread_by_id(thread_id, user.identity)
            except _CLEANUP_ERRORS:
                logger.exception(
                    "Failed to delete ephemeral thread after create error",
                    thread_id=thread_id,
                )
        raise

    if should_delete:
        schedule_background_cleanup(result.run_id, thread_id, user.identity)

    return result
