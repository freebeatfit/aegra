"""Unit tests for stateless (thread-free) run endpoints."""

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.responses import StreamingResponse
from psycopg import Error as PsycopgError
from redis import RedisError

from aegra_api.api.stateless_runs import stateless_create_run, stateless_wait_for_run
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models import Run, RunCreate, User
from aegra_api.services.run_cleanup import (
    cleanup_after_background_run as _cleanup_after_background_run,
)
from aegra_api.services.run_cleanup import (
    delete_thread_by_id as _delete_thread_by_id,
)


class _RaisingIter:
    """Iterator that raises a configured exception on first ``__next__``.

    Used as the ``__await__`` return value for ``_FailingTask``. Plain
    iterator class (not a generator) so there's no unreachable ``yield``
    statement for static analyzers (CodeQL) to flag.
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def __iter__(self) -> "_RaisingIter":
        return self

    def __next__(self) -> None:
        raise self._exc


class _FailingTask:
    """Minimal awaitable stand-in for an asyncio.Task that raises on await.

    Mimics ``done()``/``cancel()`` so ``_delete_thread_by_id`` enters the
    ``await task`` branch, then raises the configured exception when
    awaited. ``asyncio.Future.set_exception`` flips ``done()`` to True up
    front, which would skip the await branch entirely — hence this stub.
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def done(self) -> bool:
        return False

    def cancel(self) -> None:
        return None

    def __await__(self) -> Iterator[None]:
        return _RaisingIter(self._exc)


class TestDeleteThreadById:
    """Tests for the _delete_thread_by_id helper."""

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        session = AsyncMock()
        session.delete = AsyncMock()
        return session

    @pytest.fixture
    def mock_checkpointer(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def mock_db_manager(self, mock_checkpointer: AsyncMock) -> MagicMock:
        db = MagicMock()
        db.get_checkpointer.return_value = mock_checkpointer
        return db

    @pytest.mark.asyncio
    async def test_deletes_thread_with_cascade(
        self, mock_session: AsyncMock, mock_db_manager: MagicMock, mock_checkpointer: AsyncMock
    ) -> None:
        """Thread, its runs (via cascade), and its checkpoints are deleted."""
        thread_id = str(uuid4())
        user_id = "test-user"

        thread_orm = ThreadORM(
            thread_id=thread_id,
            user_id=user_id,
            status="idle",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        # No active runs
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_session.scalars.return_value = mock_scalars

        # Thread lookup returns the thread
        mock_session.scalar.return_value = thread_orm

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("aegra_api.services.run_cleanup._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_cleanup.db_manager", mock_db_manager),
        ):
            await _delete_thread_by_id(thread_id, user_id)

        mock_checkpointer.adelete_thread.assert_awaited_once_with(thread_id)
        mock_session.delete.assert_called_once_with(thread_orm)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_deletes_checkpoints_before_thread_commit(
        self, mock_session: AsyncMock, mock_db_manager: MagicMock, mock_checkpointer: AsyncMock
    ) -> None:
        """Checkpoints are deleted before the thread-row commit.

        The reverse order would recreate the orphaned-checkpoints bug: a
        committed row delete followed by a failed adelete_thread leaves
        checkpoint rows with no thread handle to retry through.
        """
        thread_id = str(uuid4())
        user_id = "test-user"

        thread_orm = ThreadORM(
            thread_id=thread_id,
            user_id=user_id,
            status="idle",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_session.scalars.return_value = mock_scalars
        mock_session.scalar.return_value = thread_orm

        order: list[str] = []
        mock_checkpointer.adelete_thread.side_effect = lambda _tid: order.append("checkpoints")
        mock_session.commit.side_effect = lambda: order.append("commit")

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("aegra_api.services.run_cleanup._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_cleanup.db_manager", mock_db_manager),
        ):
            await _delete_thread_by_id(thread_id, user_id)

        assert order == ["checkpoints", "commit"]

    @pytest.mark.asyncio
    async def test_checkpoint_delete_failure_leaves_thread_row(
        self, mock_session: AsyncMock, mock_db_manager: MagicMock, mock_checkpointer: AsyncMock
    ) -> None:
        """A failed checkpoint delete propagates and keeps the thread row.

        The surviving row makes the failure retryable; deleting it anyway
        would orphan whatever checkpoints the failed call left behind.
        """
        thread_id = str(uuid4())
        user_id = "test-user"

        thread_orm = ThreadORM(
            thread_id=thread_id,
            user_id=user_id,
            status="idle",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_session.scalars.return_value = mock_scalars
        mock_session.scalar.return_value = thread_orm

        mock_checkpointer.adelete_thread.side_effect = PsycopgError("checkpoint backend down")

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("aegra_api.services.run_cleanup._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_cleanup.db_manager", mock_db_manager),
            pytest.raises(PsycopgError, match="checkpoint backend down"),
        ):
            await _delete_thread_by_id(thread_id, user_id)

        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_checkpoint_delete_when_thread_not_owned(
        self, mock_session: AsyncMock, mock_db_manager: MagicMock, mock_checkpointer: AsyncMock
    ) -> None:
        """No checkpoint delete without an ownership-verified thread row.

        adelete_thread takes only a thread_id — calling it when the
        user-scoped lookup found nothing could wipe another user's data.
        """
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_session.scalars.return_value = mock_scalars
        mock_session.scalar.return_value = None

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("aegra_api.services.run_cleanup._get_session_maker", return_value=mock_maker),
            patch("aegra_api.services.run_cleanup.db_manager", mock_db_manager),
        ):
            await _delete_thread_by_id("nonexistent", "user")

        mock_checkpointer.adelete_thread.assert_not_called()
        mock_session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancels_active_runs_before_delete(self, mock_session: AsyncMock) -> None:
        """Active runs are cancelled before thread deletion."""
        thread_id = str(uuid4())
        user_id = "test-user"
        run_id = str(uuid4())

        active_run = RunORM(
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            status="running",
            input={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [active_run]
        mock_session.scalars.return_value = mock_scalars
        mock_session.scalar.return_value = None  # Thread already gone

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_task = MagicMock()
        mock_task.done.return_value = True

        with (
            patch("aegra_api.services.run_cleanup._get_session_maker", return_value=mock_maker),
            patch(
                "aegra_api.services.run_cleanup.streaming_service.cancel_run",
                new_callable=AsyncMock,
            ) as mock_cancel,
            patch("aegra_api.services.run_cleanup.active_runs", {run_id: mock_task}),
        ):
            await _delete_thread_by_id(thread_id, user_id)

        mock_cancel.assert_called_once_with(run_id)

    @pytest.mark.asyncio
    async def test_noop_when_thread_not_found(self, mock_session: AsyncMock) -> None:
        """No error when thread doesn't exist (idempotent)."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_session.scalars.return_value = mock_scalars
        mock_session.scalar.return_value = None

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("aegra_api.services.run_cleanup._get_session_maker", return_value=mock_maker):
            await _delete_thread_by_id("nonexistent", "user")

        mock_session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_cancelled_error_on_task_await(self, mock_session: AsyncMock) -> None:
        """CancelledError from awaiting a cancelled task is silently absorbed."""
        thread_id = str(uuid4())
        user_id = "test-user"
        run_id = str(uuid4())

        active_run = RunORM(
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            status="running",
            input={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [active_run]
        mock_session.scalars.return_value = mock_scalars
        mock_session.scalar.return_value = None

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        # asyncio.Future is a real awaitable; the source calls .cancel()
        # then awaits — the await raises CancelledError.
        fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()

        with (
            patch("aegra_api.services.run_cleanup._get_session_maker", return_value=mock_maker),
            patch(
                "aegra_api.services.run_cleanup.streaming_service.cancel_run",
                new_callable=AsyncMock,
            ),
            patch("aegra_api.services.run_cleanup.active_runs", {run_id: fut}),
        ):
            # Should not raise — CancelledError is caught
            await _delete_thread_by_id(thread_id, user_id)

    @pytest.mark.asyncio
    async def test_logs_infra_error_on_task_await(self, mock_session: AsyncMock) -> None:
        """Defensive: infra-class errors on ``await task`` are logged, not re-raised.

        Real ``asyncio.Task`` always raises ``CancelledError`` after
        ``task.cancel()``, so this RedisError flow is reachable only if
        ``active_runs`` ever holds a non-Task awaitable (e.g. a custom
        wrapper or a future). Test guards the narrow ``(RedisError,
        SQLAlchemyError, OSError)`` tuple from accidental widening —
        complements ``test_propagates_programmer_error`` which asserts
        non-infra exceptions still propagate.
        """
        thread_id = str(uuid4())
        user_id = "test-user"
        run_id = str(uuid4())

        active_run = RunORM(
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            status="running",
            input={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [active_run]
        mock_session.scalars.return_value = mock_scalars
        mock_session.scalar.return_value = None

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("aegra_api.services.run_cleanup._get_session_maker", return_value=mock_maker),
            patch(
                "aegra_api.services.run_cleanup.streaming_service.cancel_run",
                new_callable=AsyncMock,
            ),
            patch(
                "aegra_api.services.run_cleanup.active_runs",
                {run_id: _FailingTask(RedisError("redis hiccup"))},
            ),
        ):
            # Should not raise — RedisError is in the narrow cleanup tuple
            await _delete_thread_by_id(thread_id, user_id)

    @pytest.mark.asyncio
    async def test_propagates_programmer_error(self, mock_session: AsyncMock) -> None:
        """Programmer errors (RuntimeError, AttributeError, ...) on ``await task`` propagate.

        Regression for the narrow-tuple cleanup: per CLAUDE.md and the
        review comment, broker/cancel paths must not swallow random
        programmer errors — those signal real bugs and need to surface.
        """
        thread_id = str(uuid4())
        user_id = "test-user"
        run_id = str(uuid4())

        active_run = RunORM(
            run_id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            status="running",
            input={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [active_run]
        mock_session.scalars.return_value = mock_scalars

        mock_maker = MagicMock()
        mock_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

        with (  # noqa: SIM117
            patch("aegra_api.services.run_cleanup._get_session_maker", return_value=mock_maker),
            patch(
                "aegra_api.services.run_cleanup.streaming_service.cancel_run",
                new_callable=AsyncMock,
            ),
            patch(
                "aegra_api.services.run_cleanup.active_runs",
                {run_id: _FailingTask(RuntimeError("task exploded"))},
            ),
        ):
            with pytest.raises(RuntimeError, match="task exploded"):
                await _delete_thread_by_id(thread_id, user_id)


class TestCleanupAfterBackgroundRun:
    """Tests for the _cleanup_after_background_run helper."""

    @pytest.mark.asyncio
    async def test_awaits_task_then_deletes(self) -> None:
        """Waits for the background task to finish, then deletes the thread."""
        run_id = str(uuid4())
        thread_id = str(uuid4())
        user_id = "test-user"

        task_awaited = False

        async def _fake_task() -> None:
            nonlocal task_awaited
            task_awaited = True

        # Create a real asyncio.Task so `await task` works
        task = asyncio.create_task(_fake_task())
        await task  # let it finish before test to avoid timing issues

        with (
            patch("aegra_api.services.run_cleanup.active_runs", {run_id: task}),
            patch(
                "aegra_api.services.run_cleanup.delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            await _cleanup_after_background_run(run_id, thread_id, user_id)

        assert task_awaited
        mock_delete.assert_called_once_with(thread_id, user_id)

    @pytest.mark.asyncio
    async def test_deletes_thread_when_no_task_in_active_runs(self) -> None:
        """Cleanup proceeds directly to thread deletion when run_id is not in active_runs."""
        run_id = str(uuid4())
        thread_id = str(uuid4())
        user_id = "test-user"

        with (
            patch("aegra_api.services.run_cleanup.active_runs", {}),
            patch(
                "aegra_api.services.run_cleanup.delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            await _cleanup_after_background_run(run_id, thread_id, user_id)

        mock_delete.assert_called_once_with(thread_id, user_id)

    @pytest.mark.asyncio
    async def test_swallows_psycopg_error_from_thread_delete(self) -> None:
        """psycopg failures from cleanup are logged, not propagated.

        Checkpoint deletion runs on the psycopg pool, so PsycopgError must
        be in _CLEANUP_ERRORS — otherwise a transient backend hiccup would
        crash the fire-and-forget cleanup task.
        """
        run_id = str(uuid4())
        thread_id = str(uuid4())
        user_id = "test-user"

        with (
            patch("aegra_api.services.run_cleanup.active_runs", {}),
            patch(
                "aegra_api.services.run_cleanup.delete_thread_by_id",
                new_callable=AsyncMock,
                side_effect=PsycopgError("checkpoint backend down"),
            ) as mock_delete,
        ):
            # Should not raise — PsycopgError is in the cleanup tuple
            await _cleanup_after_background_run(run_id, thread_id, user_id)

        mock_delete.assert_awaited_once_with(thread_id, user_id)

    @pytest.mark.asyncio
    async def test_deletes_thread_when_wait_times_out(self) -> None:
        """A run exceeding the wait cap still gets its ephemeral thread deleted."""
        run_id = str(uuid4())
        thread_id = str(uuid4())
        user_id = "test-user"

        with (
            patch(
                "aegra_api.services.run_cleanup.executor.wait_for_completion",
                new_callable=AsyncMock,
                side_effect=TimeoutError,
            ),
            patch(
                "aegra_api.services.run_cleanup.delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            await _cleanup_after_background_run(run_id, thread_id, user_id)

        mock_delete.assert_awaited_once_with(thread_id, user_id)

    @pytest.mark.asyncio
    async def test_deletes_thread_when_wait_fails_with_infra_error(self) -> None:
        """A tolerated infra failure while waiting is logged, not fatal to cleanup."""
        run_id = str(uuid4())
        thread_id = str(uuid4())
        user_id = "test-user"

        with (
            patch(
                "aegra_api.services.run_cleanup.executor.wait_for_completion",
                new_callable=AsyncMock,
                side_effect=PsycopgError("broker connection lost"),
            ),
            patch(
                "aegra_api.services.run_cleanup.delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            await _cleanup_after_background_run(run_id, thread_id, user_id)

        mock_delete.assert_awaited_once_with(thread_id, user_id)


class TestStatelessWaitForRun:
    """Tests for POST /runs/wait."""

    @pytest.fixture
    def mock_user(self) -> User:
        return User(identity="test-user", scopes=[])

    @pytest.mark.asyncio
    async def test_delegates_and_deletes_thread(self, mock_user: User) -> None:
        """Delegates to wait_for_run and deletes ephemeral thread after stream."""
        import json

        expected_output = {"result": "done"}
        request = RunCreate(assistant_id="agent", input={"msg": "hi"})

        # wait_for_run now returns a StreamingResponse, so mock it accordingly
        mock_response = StreamingResponse(
            iter([json.dumps(expected_output).encode()]),
            media_type="application/json",
        )

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-1"),
            patch(
                "aegra_api.api.stateless_runs.wait_for_run",
                new_callable=AsyncMock,
                return_value=mock_response,
            ) as mock_wait,
            patch(
                "aegra_api.api.stateless_runs.delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            result = await stateless_wait_for_run(request, mock_user)

            # Result is a StreamingResponse; consume to trigger cleanup
            assert isinstance(result, StreamingResponse)
            body = b""
            async for chunk in result.body_iterator:
                body += chunk if isinstance(chunk, bytes) else chunk.encode()

            assert json.loads(body) == expected_output
            mock_wait.assert_called_once_with("eph-thread-1", request, mock_user)
            mock_delete.assert_called_once_with("eph-thread-1", mock_user.identity)

    @pytest.mark.asyncio
    async def test_keeps_thread_when_requested(self, mock_user: User) -> None:
        """Thread is preserved when on_completion='keep'."""
        import json

        request = RunCreate(assistant_id="agent", input={"msg": "hi"}, on_completion="keep")

        mock_response = StreamingResponse(
            iter([json.dumps({}).encode()]),
            media_type="application/json",
        )

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-2"),
            patch(
                "aegra_api.api.stateless_runs.wait_for_run",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch(
                "aegra_api.api.stateless_runs.delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
        ):
            result = await stateless_wait_for_run(request, mock_user)

        # Returns original response unchanged (no wrapper)
        assert result is mock_response
        mock_delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleans_up_on_error(self, mock_user: User) -> None:
        """Thread is deleted even when wait_for_run raises."""
        request = RunCreate(assistant_id="agent", input={"msg": "hi"})

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-3"),
            patch(
                "aegra_api.api.stateless_runs.wait_for_run",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "aegra_api.api.stateless_runs.delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
            pytest.raises(RuntimeError, match="boom"),
        ):
            await stateless_wait_for_run(request, mock_user)

        mock_delete.assert_called_once_with("eph-thread-3", mock_user.identity)

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_mask_original_error(self, mock_user: User) -> None:
        """If _delete_thread_by_id raises during cleanup, the original error propagates."""
        request = RunCreate(assistant_id="agent", input={"msg": "hi"})

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-err"),
            patch(
                "aegra_api.api.stateless_runs.wait_for_run",
                new_callable=AsyncMock,
                side_effect=RuntimeError("original"),
            ),
            patch(
                "aegra_api.api.stateless_runs.delete_thread_by_id",
                new_callable=AsyncMock,
                side_effect=OSError("cleanup failed"),
            ),
            pytest.raises(RuntimeError, match="original"),
        ):
            await stateless_wait_for_run(request, mock_user)


class TestStatelessCreateRun:
    """Tests for POST /runs."""

    @pytest.fixture
    def mock_user(self) -> User:
        return User(identity="test-user", scopes=[])

    @pytest.fixture
    def mock_session(self) -> AsyncMock:
        session = AsyncMock()
        session.refresh = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.mark.asyncio
    async def test_delegates_and_schedules_cleanup(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Delegates to create_run and schedules background cleanup."""
        run_id = str(uuid4())
        request = RunCreate(assistant_id="agent", input={"msg": "hi"})

        mock_run = Run(
            run_id=run_id,
            thread_id="eph-thread-6",
            assistant_id="agent",
            status="pending",
            input={"msg": "hi"},
            user_id=mock_user.identity,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-6"),
            patch(
                "aegra_api.api.stateless_runs.create_run",
                new_callable=AsyncMock,
                return_value=mock_run,
            ) as mock_create,
            patch("aegra_api.services.run_cleanup.asyncio.create_task") as mock_create_task,
        ):
            result = await stateless_create_run(request, mock_user, mock_session)

        assert result.run_id == run_id
        mock_create.assert_called_once_with("eph-thread-6", request, mock_user, mock_session)
        mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_cleanup_when_keep(self, mock_user: User, mock_session: AsyncMock) -> None:
        """No background cleanup task when on_completion='keep'."""
        run_id = str(uuid4())
        request = RunCreate(assistant_id="agent", input={"msg": "hi"}, on_completion="keep")

        mock_run = Run(
            run_id=run_id,
            thread_id="eph-thread-7",
            assistant_id="agent",
            status="pending",
            input={"msg": "hi"},
            user_id=mock_user.identity,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-7"),
            patch(
                "aegra_api.api.stateless_runs.create_run",
                new_callable=AsyncMock,
                return_value=mock_run,
            ),
            patch("aegra_api.services.run_cleanup.asyncio.create_task") as mock_create_task,
        ):
            result = await stateless_create_run(request, mock_user, mock_session)

        assert result.run_id == run_id
        mock_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleans_up_thread_when_delegation_raises(self, mock_user: User, mock_session: AsyncMock) -> None:
        """Thread is deleted if create_run raises after auto-creating the thread."""
        request = RunCreate(assistant_id="agent", input={"msg": "hi"})

        with (
            patch("aegra_api.api.stateless_runs.uuid4", return_value="eph-thread-err"),
            patch(
                "aegra_api.api.stateless_runs.create_run",
                new_callable=AsyncMock,
                side_effect=RuntimeError("create failed"),
            ),
            patch(
                "aegra_api.api.stateless_runs.delete_thread_by_id",
                new_callable=AsyncMock,
            ) as mock_delete,
            pytest.raises(RuntimeError, match="create failed"),
        ):
            await stateless_create_run(request, mock_user, mock_session)

        mock_delete.assert_called_once_with("eph-thread-err", mock_user.identity)
