"""Busy-thread refusal and pending-interrupt checks against a real PostgreSQL.

Drives ``POST /threads/{id}/commands`` and ``POST /threads/{id}/runs`` with real
sessions and a real checkpointer, and races two run creations on separate
connections. Skips when no database is reachable.
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any, TypedDict
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from aegra_api.api import event_streaming as es_module
from aegra_api.api import runs as runs_module
from aegra_api.core.auth_deps import get_current_user, require_auth
from aegra_api.core.database import db_manager
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import get_session
from aegra_api.models import RunCreate, User
from aegra_api.services import run_preparation
from aegra_api.services.event_streaming import capabilities as caps
from aegra_api.services.run_preparation import _prepare_run
from aegra_api.settings import settings
from aegra_api.utils.assistants import resolve_assistant_id

_USER_ID = "command-semantics-test-user"
_GRAPH_ID = "command_semantics_graph"
_ASSISTANT_ID = resolve_assistant_id(_GRAPH_ID, {_GRAPH_ID: None})
_WAIT_TIMEOUT = 10.0

Maker = async_sessionmaker[AsyncSession]


class _State(TypedDict, total=False):
    first: str
    second: str


def _ask_first(_state: _State) -> _State:
    return {"first": interrupt("first?")}


def _ask_second(_state: _State) -> _State:
    return {"second": interrupt("second?")}


def _two_question_graph(saver: AsyncPostgresSaver) -> CompiledStateGraph:
    builder = StateGraph(_State)
    builder.add_node("ask_first", _ask_first)
    builder.add_node("ask_second", _ask_second)
    builder.add_edge(START, "ask_first")
    builder.add_edge("ask_first", "ask_second")
    return builder.compile(checkpointer=saver)


@pytest.fixture(autouse=True)
def _v2_enabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(caps.settings.event_streaming, "FF_V2_EVENT_STREAMING", True)
    caps._probe_runtime_symbols.cache_clear()
    yield
    caps._probe_runtime_symbols.cache_clear()


@pytest.fixture
async def maker(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Maker]:
    """Real sessions for every code path, a registered graph, and no executor; or skip."""
    engine = create_async_engine(settings.db.database_url)
    try:
        try:
            async with engine.begin() as conn:
                thread_table = await conn.scalar(text("SELECT to_regclass('public.thread')"))
        except Exception as exc:  # noqa: BLE001 - any connect failure means "no database here"
            pytest.skip(f"PostgreSQL test database is unavailable: {exc}")
        if thread_table is None:
            pytest.skip("thread table is unavailable; run Alembic migrations before this DB test")

        session_maker = async_sessionmaker(engine, expire_on_commit=False)
        monkeypatch.setattr(es_module, "_get_session_maker", lambda: session_maker)
        monkeypatch.setattr(run_preparation, "_get_session_maker", lambda: session_maker)
        monkeypatch.setattr(
            run_preparation, "get_langgraph_service", lambda: SimpleNamespace(list_graphs=lambda: {_GRAPH_ID: None})
        )
        monkeypatch.setattr(run_preparation, "executor", SimpleNamespace(submit=AsyncMock()))
        async with session_maker() as session:
            await session.execute(delete(AssistantORM).where(AssistantORM.user_id == _USER_ID))
            session.add(
                AssistantORM(assistant_id=_ASSISTANT_ID, name="semantics", graph_id=_GRAPH_ID, user_id=_USER_ID)
            )
            await session.commit()
        try:
            yield session_maker
        finally:
            async with session_maker() as cleanup:
                await cleanup.execute(delete(ThreadORM).where(ThreadORM.user_id == _USER_ID))
                await cleanup.execute(delete(AssistantORM).where(AssistantORM.user_id == _USER_ID))
                await cleanup.commit()
    finally:
        await engine.dispose()


@pytest.fixture
async def saver(monkeypatch: pytest.MonkeyPatch, maker: Maker) -> AsyncIterator[AsyncPostgresSaver]:
    async with AsyncPostgresSaver.from_conn_string(settings.db.database_url_sync) as checkpointer:
        await checkpointer.setup()
        monkeypatch.setattr(db_manager, "_checkpointer", checkpointer)
        yield checkpointer


@pytest.fixture
async def client(maker: Maker) -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    user = User(identity=_USER_ID)
    app.dependency_overrides[require_auth] = lambda: user
    app.dependency_overrides[get_current_user] = lambda: user

    async def _session() -> AsyncIterator[AsyncSession]:
        async with maker() as session:
            yield session

    app.dependency_overrides[get_session] = _session
    app.include_router(es_module.router)
    app.include_router(runs_module.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        yield http


async def _seed_thread(maker: Maker, *, status: str, run_status: str | None = None) -> str:
    thread_id = str(uuid4())
    async with maker() as session:
        session.add(
            ThreadORM(thread_id=thread_id, status=status, metadata_json={"graph_id": _GRAPH_ID}, user_id=_USER_ID)
        )
        await session.flush()
        if run_status is not None:
            session.add(
                RunORM(
                    run_id=str(uuid4()),
                    thread_id=thread_id,
                    assistant_id=_ASSISTANT_ID,
                    status=run_status,
                    user_id=_USER_ID,
                )
            )
        await session.commit()
    return thread_id


async def _set_thread_status(maker: Maker, thread_id: str, status: str) -> None:
    async with maker() as session:
        await session.execute(update(ThreadORM).where(ThreadORM.thread_id == thread_id).values(status=status))
        await session.commit()


async def _run_count(maker: Maker, thread_id: str) -> int:
    async with maker() as session:
        count = await session.scalar(select(func.count()).select_from(RunORM).where(RunORM.thread_id == thread_id))
    return int(count or 0)


async def _command(client: httpx.AsyncClient, thread_id: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
    resp = await client.post(f"/threads/{thread_id}/commands", json={"id": 1, "method": method, "params": params})
    assert resp.status_code == 200
    return resp.json()


def _reject_request() -> RunCreate:
    return RunCreate(assistant_id=_GRAPH_ID, input={"messages": []}, multitask_strategy="reject")


class TestRejectOnBusyThread:
    async def test_run_start_with_reject_on_busy_thread_is_unknown_error(
        self, maker: Maker, client: httpx.AsyncClient
    ) -> None:
        thread_id = await _seed_thread(maker, status="busy", run_status="running")

        envelope = await _command(
            client,
            thread_id,
            "run.start",
            {"assistant_id": _GRAPH_ID, "input": {"messages": []}, "multitaskStrategy": "reject"},
        )

        assert envelope["type"] == "error"
        assert envelope["error"] == "unknown_error"
        assert "already running" in envelope["message"]
        assert await _run_count(maker, thread_id) == 1

    async def test_rest_run_create_with_reject_on_busy_thread_is_409(
        self, maker: Maker, client: httpx.AsyncClient
    ) -> None:
        thread_id = await _seed_thread(maker, status="busy", run_status="pending")

        resp = await client.post(
            f"/threads/{thread_id}/runs",
            json={"assistant_id": _GRAPH_ID, "input": {"messages": []}, "multitask_strategy": "reject"},
        )

        assert resp.status_code == 409
        assert await _run_count(maker, thread_id) == 1

    async def test_run_start_with_reject_after_the_last_run_settled_starts_a_run(
        self, maker: Maker, client: httpx.AsyncClient
    ) -> None:
        thread_id = await _seed_thread(maker, status="idle", run_status="success")

        envelope = await _command(
            client,
            thread_id,
            "run.start",
            {"assistant_id": _GRAPH_ID, "input": {"messages": []}, "multitaskStrategy": "reject"},
        )

        assert envelope["type"] == "success"
        assert await _run_count(maker, thread_id) == 2

    async def test_run_start_without_reject_on_busy_thread_still_starts_a_run(
        self, maker: Maker, client: httpx.AsyncClient
    ) -> None:
        thread_id = await _seed_thread(maker, status="busy", run_status="running")

        envelope = await _command(
            client, thread_id, "run.start", {"assistant_id": _GRAPH_ID, "input": {"messages": []}}
        )

        assert envelope["type"] == "success"
        assert await _run_count(maker, thread_id) == 2

    async def test_concurrent_reject_creations_start_exactly_one_run(self, maker: Maker) -> None:
        """Two replicas creating a run on one idle thread: the second sees the first's run."""
        thread_id = await _seed_thread(maker, status="idle")
        committing = asyncio.Event()
        release = asyncio.Event()

        class _HeldCommit:
            def __init__(self, session: AsyncSession) -> None:
                self._session = session

            def __getattr__(self, name: str) -> Any:
                return getattr(self._session, name)

            async def commit(self) -> None:
                committing.set()
                await release.wait()
                await self._session.commit()

        user = User(identity=_USER_ID)
        async with maker() as first_session, maker() as second_session:
            first = asyncio.create_task(
                _prepare_run(_HeldCommit(first_session), thread_id, _reject_request(), user, initial_status="pending")
            )
            await asyncio.wait_for(committing.wait(), timeout=_WAIT_TIMEOUT)
            second = asyncio.create_task(
                _prepare_run(second_session, thread_id, _reject_request(), user, initial_status="pending")
            )
            # Without the row lock the second creation finishes here, before the first commits.
            await asyncio.sleep(0.3)
            release.set()
            results = await asyncio.wait_for(
                asyncio.gather(first, second, return_exceptions=True), timeout=_WAIT_TIMEOUT
            )

        assert not isinstance(results[0], BaseException)
        assert isinstance(results[1], HTTPException)
        assert results[1].status_code == 409
        assert await _run_count(maker, thread_id) == 1


class TestRespondToPendingInterruptsOnly:
    @pytest.fixture
    async def answered_thread(self, maker: Maker, saver: AsyncPostgresSaver) -> AsyncIterator[tuple[str, str, str]]:
        """A thread whose first interrupt was answered and whose second is pending."""
        thread_id = await _seed_thread(maker, status="busy", run_status="interrupted")
        graph = _two_question_graph(saver)
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        await graph.ainvoke({}, config)
        first_id = (await graph.aget_state(config)).interrupts[0].id
        await graph.ainvoke(Command(resume={first_id: "yes"}), config)
        second_id = (await graph.aget_state(config)).interrupts[0].id
        await _set_thread_status(maker, thread_id, "interrupted")
        try:
            yield thread_id, first_id, second_id
        finally:
            await saver.adelete_thread(thread_id)

    async def test_answered_interrupt_is_no_such_interrupt(
        self, maker: Maker, client: httpx.AsyncClient, answered_thread: tuple[str, str, str]
    ) -> None:
        thread_id, first_id, _second_id = answered_thread

        envelope = await _command(
            client, thread_id, "input.respond", {"interrupt_id": first_id, "namespace": [], "response": "again"}
        )

        assert envelope["type"] == "error"
        assert envelope["error"] == "no_such_interrupt"
        assert await _run_count(maker, thread_id) == 1

    async def test_batch_with_an_answered_interrupt_is_no_such_interrupt(
        self, maker: Maker, client: httpx.AsyncClient, answered_thread: tuple[str, str, str]
    ) -> None:
        thread_id, first_id, second_id = answered_thread

        envelope = await _command(
            client,
            thread_id,
            "input.respond",
            {
                "responses": [
                    {"interrupt_id": second_id, "namespace": [], "response": "yes"},
                    {"interrupt_id": first_id, "namespace": [], "response": "again"},
                ]
            },
        )

        assert envelope["error"] == "no_such_interrupt"
        assert await _run_count(maker, thread_id) == 1

    async def test_pending_interrupt_starts_a_resume_run(
        self, maker: Maker, client: httpx.AsyncClient, answered_thread: tuple[str, str, str]
    ) -> None:
        thread_id, _first_id, second_id = answered_thread

        envelope = await _command(
            client, thread_id, "input.respond", {"interrupt_id": second_id, "namespace": [], "response": "yes"}
        )

        assert envelope["type"] == "success"
        assert await _run_count(maker, thread_id) == 2

    async def test_interrupt_answered_by_a_finished_run_is_no_such_interrupt(
        self, maker: Maker, saver: AsyncPostgresSaver, client: httpx.AsyncClient, answered_thread: tuple[str, str, str]
    ) -> None:
        thread_id, _first_id, second_id = answered_thread
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        await _two_question_graph(saver).ainvoke(Command(resume={second_id: "yes"}), config)
        await _set_thread_status(maker, thread_id, "idle")

        envelope = await _command(
            client, thread_id, "input.respond", {"interrupt_id": second_id, "namespace": [], "response": "again"}
        )

        assert envelope["error"] == "no_such_interrupt"
        assert await _run_count(maker, thread_id) == 1
