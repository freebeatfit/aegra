"""Shared fixtures for the v2 event streaming tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.types import Interrupt

from aegra_api.core.database import db_manager


@pytest.fixture(autouse=True)
def checkpointer(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """The thread's latest checkpoint has interrupts a*32 and b*32 pending, as LangGraph saves them."""
    interrupts = (Interrupt(value="?", id="a" * 32), Interrupt(value="?", id="b" * 32))
    saver = MagicMock()
    saver.aget_tuple = AsyncMock(
        return_value=SimpleNamespace(
            pending_writes=[("task-1", "__interrupt__", interrupts), ("task-1", "messages", [])]
        )
    )
    monkeypatch.setattr(db_manager, "_checkpointer", saver)
    return saver
