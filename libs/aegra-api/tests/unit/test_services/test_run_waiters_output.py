"""FB patch P5: join/wait answer a v2 run from the thread's values."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from aegra_api.services import run_waiters
from aegra_api.services.run_waiters import run_result_body, thread_values_for


def _run(**fields: Any) -> Any:
    base = {"run_id": "r1", "thread_id": "t1", "assistant_id": "a1", "status": "success", "output": None}
    return SimpleNamespace(**{**base, **fields})


class TestRunResultBody:
    def test_stored_output_wins(self) -> None:
        assert run_result_body(_run(output={"a": 1}), "r1", thread_values={"b": 2}) == {"a": 1}

    def test_v2_run_answers_with_thread_values(self) -> None:
        assert run_result_body(_run(), "r1", thread_values={"messages": ["m"]}) == {"messages": ["m"]}

    def test_no_values_is_an_empty_body(self) -> None:
        assert run_result_body(_run(), "r1") == {}

    def test_error_run_still_reports_its_error(self) -> None:
        body = run_result_body(_run(status="error", error_message="boom"), "r1", thread_values={"x": 1})
        assert body == {"__error__": {"error": "Error", "message": "boom"}}


@pytest.mark.asyncio
class TestThreadValuesFor:
    async def test_skips_runs_with_output_or_errors(self) -> None:
        assert await thread_values_for(_run(output={"a": 1}), MagicMock()) is None
        assert await thread_values_for(_run(status="error"), MagicMock()) is None
        assert await thread_values_for(None, MagicMock()) is None

    async def test_reads_the_current_checkpoint(self) -> None:
        graph = MagicMock()
        graph.aget_state = AsyncMock(return_value=SimpleNamespace(values={"messages": ["m"]}))
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=graph)
        cm.__aexit__ = AsyncMock(return_value=False)
        service = MagicMock()
        service.get_graph = MagicMock(return_value=cm)
        session = MagicMock()
        session.scalar = AsyncMock(return_value="graph-1")
        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock(return_value=session)
        session_cm.__aexit__ = AsyncMock(return_value=False)
        with (
            patch.object(run_waiters, "_get_session_maker", return_value=lambda: session_cm),
            patch("aegra_api.services.langgraph_service.get_langgraph_service", return_value=service),
        ):
            values = await thread_values_for(_run(), SimpleNamespace(identity="u1", permissions=[]))
        assert values == {"messages": ["m"]}
        assert service.get_graph.call_args.args[0] == "graph-1"

    async def test_a_failed_read_falls_back_to_none(self) -> None:
        with patch.object(run_waiters, "_get_session_maker", side_effect=RuntimeError("db down")):
            assert await thread_values_for(_run(), MagicMock()) is None


class TestEncodeOutput:
    def test_thread_values_keep_message_structure(self) -> None:
        """Thread values hold message objects; join/wait must send them as dicts, as /state does."""
        body = json.loads(run_waiters.encode_output({"messages": [AIMessage(content="hi", id="m1")], "n": 1}))

        assert body["n"] == 1
        assert body["messages"][0]["type"] == "ai"
        assert body["messages"][0]["content"] == "hi"
        assert body["messages"][0]["id"] == "m1"
