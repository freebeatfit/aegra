"""Unit tests for run_preparation helpers."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from langgraph.types import Interrupt

from aegra_api.config import CheckpointerConfig
from aegra_api.core.database import db_manager
from aegra_api.models.runs import RunCreate
from aegra_api.services import run_preparation as mod
from aegra_api.services.run_preparation import (
    InterruptNotPendingError,
    _refuse_if_interrupt_answered,
    _resolve_checkpoint,
    _resolve_durability,
    _validate_resume_command,
    get_default_durability,
)


@pytest.fixture(autouse=True)
def _fast_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the resume-settle backoff so reject paths don't wait."""
    monkeypatch.setattr(mod, "_RESUME_SETTLE_INTERVAL_SECONDS", 0)


def _thread(status: str) -> SimpleNamespace:
    return SimpleNamespace(status=status)


def _session_returning(thread: object) -> AsyncMock:
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=thread)
    return session


def _patch_fresh_sessions(monkeypatch: pytest.MonkeyPatch, *threads: object) -> None:
    """Make run_preparation's fresh-session poll yield the given threads in order."""
    seq = list(threads)

    async def scalar(_stmt: object) -> object:
        return seq.pop(0) if len(seq) > 1 else (seq[0] if seq else None)

    fresh = AsyncMock()
    fresh.scalar = AsyncMock(side_effect=scalar)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=fresh)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(mod, "_get_session_maker", lambda: MagicMock(return_value=ctx))


class TestRefuseIfInterruptAnswered:
    """Under the thread row lock, a resume of an answered interrupt is refused (REST 409)."""

    @pytest.fixture
    def checkpointer(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        saver = MagicMock()
        pending = (Interrupt(value="?", id="a" * 32),)
        saver.aget_tuple = AsyncMock(return_value=SimpleNamespace(pending_writes=[("t", "__interrupt__", pending)]))
        monkeypatch.setattr(db_manager, "_checkpointer", saver)
        return saver

    async def test_thread_no_longer_interrupted_is_refused_with_409(self, checkpointer: MagicMock) -> None:
        with pytest.raises(InterruptNotPendingError) as refused:
            await _refuse_if_interrupt_answered(_session_returning("busy"), "t1", {"a" * 32: "yes"})
        assert refused.value.status_code == 409

    async def test_targeted_interrupt_no_longer_pending_is_refused(self, checkpointer: MagicMock) -> None:
        with pytest.raises(InterruptNotPendingError, match="c" * 32):
            await _refuse_if_interrupt_answered(_session_returning("interrupted"), "t1", {"c" * 32: "yes"})

    async def test_pending_targeted_interrupt_passes(self, checkpointer: MagicMock) -> None:
        await _refuse_if_interrupt_answered(_session_returning("interrupted"), "t1", {"a" * 32: "yes"})

    async def test_untargeted_resume_needs_only_the_interrupted_status(self, checkpointer: MagicMock) -> None:
        await _refuse_if_interrupt_answered(_session_returning("interrupted"), "t1", {"approved": True})
        checkpointer.aget_tuple.assert_not_awaited()


class TestValidateResumeCommand:
    async def test_resume_on_interrupted_thread_passes(self) -> None:
        session = _session_returning(_thread("interrupted"))
        await _validate_resume_command(session, "t1", {"resume": "yes"})

    async def test_resume_none_on_non_interrupted_thread_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """{"resume": None} must still be guarded — None is a valid resume payload."""
        _patch_fresh_sessions(monkeypatch, _thread("idle"))
        session = _session_returning(_thread("idle"))
        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "t1", {"resume": None})
        assert exc.value.status_code == 400

    async def test_resume_settles_when_status_flips_to_interrupted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The interrupt reaches the client before finalize commits 'interrupted';
        the guard polls a fresh session and accepts once the status settles."""
        _patch_fresh_sessions(monkeypatch, _thread("busy"), _thread("interrupted"))
        session = _session_returning(_thread("busy"))  # first (request-session) read is stale
        await _validate_resume_command(session, "t1", {"resume": "yes"})

    async def test_resume_on_missing_thread_is_404(self) -> None:
        session = _session_returning(None)
        with pytest.raises(HTTPException) as exc:
            await _validate_resume_command(session, "t1", {"resume": None})
        assert exc.value.status_code == 404

    async def test_non_resume_command_skips_check(self) -> None:
        session = _session_returning(_thread("idle"))
        await _validate_resume_command(session, "t1", {"goto": "node"})
        session.scalar.assert_not_awaited()

    async def test_none_command_skips_check(self) -> None:
        session = _session_returning(_thread("idle"))
        await _validate_resume_command(session, "t1", None)
        session.scalar.assert_not_awaited()


_CHECKPOINT_ID = "1ef4f797-8335-6428-8001-8a1503f9b875"
_OTHER_CHECKPOINT_ID = "1ef4f797-8335-6428-8001-8a1503f9b876"


class TestResolveCheckpoint:
    def test_returns_checkpoint_unchanged_without_checkpoint_id(self) -> None:
        request = RunCreate(assistant_id="agent", checkpoint={"checkpoint_id": "chk-1", "checkpoint_ns": ""})
        assert _resolve_checkpoint(request) == {"checkpoint_id": "chk-1", "checkpoint_ns": ""}

    def test_returns_none_when_neither_is_set(self) -> None:
        assert _resolve_checkpoint(RunCreate(assistant_id="agent", input={"x": 1})) is None

    def test_builds_checkpoint_from_top_level_checkpoint_id(self) -> None:
        request = RunCreate(assistant_id="agent", checkpoint_id=_CHECKPOINT_ID)
        assert _resolve_checkpoint(request) == {"checkpoint_id": _CHECKPOINT_ID}

    def test_keeps_other_checkpoint_keys(self) -> None:
        request = RunCreate(assistant_id="agent", checkpoint_id=_CHECKPOINT_ID, checkpoint={"checkpoint_ns": "sub"})
        assert _resolve_checkpoint(request) == {"checkpoint_id": _CHECKPOINT_ID, "checkpoint_ns": "sub"}

    def test_checkpoint_dict_wins_over_top_level_checkpoint_id(self) -> None:
        request = RunCreate(
            assistant_id="agent",
            checkpoint_id=_CHECKPOINT_ID,
            checkpoint={"checkpoint_id": _OTHER_CHECKPOINT_ID},
        )
        assert _resolve_checkpoint(request) == {"checkpoint_id": _OTHER_CHECKPOINT_ID}


def _durability_sources(
    monkeypatch: pytest.MonkeyPatch, *, env: str | None, checkpointer: CheckpointerConfig | None
) -> None:
    monkeypatch.setattr(mod.settings.checkpointer, "AEGRA_CHECKPOINT_DURABILITY", env)
    monkeypatch.setattr(mod, "load_checkpointer_config", lambda: checkpointer)


@pytest.fixture
def _fresh_default_durability() -> Iterator[None]:
    get_default_durability.cache_clear()
    yield
    get_default_durability.cache_clear()


@pytest.mark.usefixtures("_fresh_default_durability")
class TestGetDefaultDurability:
    def test_none_when_nothing_is_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _durability_sources(monkeypatch, env=None, checkpointer=None)
        assert get_default_durability() is None

    def test_none_when_checkpointer_block_has_no_durability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _durability_sources(monkeypatch, env=None, checkpointer={"ttl": None})
        assert get_default_durability() is None

    def test_reads_aegra_json_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _durability_sources(monkeypatch, env=None, checkpointer={"durability": "exit"})
        assert get_default_durability() == "exit"

    def test_env_var_wins_over_aegra_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _durability_sources(monkeypatch, env="sync", checkpointer={"durability": "exit"})
        assert get_default_durability() == "sync"

    def test_blank_env_var_falls_through_to_aegra_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _durability_sources(monkeypatch, env="  ", checkpointer={"durability": "async"})
        assert get_default_durability() == "async"

    def test_env_var_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _durability_sources(monkeypatch, env=" exit\n", checkpointer=None)
        assert get_default_durability() == "exit"

    @pytest.mark.parametrize("value", ["EXIT", "never", "0"])
    def test_invalid_env_var_raises_naming_the_variable(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        _durability_sources(monkeypatch, env=value, checkpointer=None)
        with pytest.raises(ValueError, match="AEGRA_CHECKPOINT_DURABILITY"):
            get_default_durability()

    @pytest.mark.parametrize("value", ["eventually", True, 1, {"mode": "exit"}])
    def test_invalid_aegra_json_value_raises_naming_the_key(self, monkeypatch: pytest.MonkeyPatch, value: Any) -> None:
        _durability_sources(monkeypatch, env=None, checkpointer=cast(CheckpointerConfig, {"durability": value}))
        with pytest.raises(ValueError, match=r"checkpointer\.durability"):
            get_default_durability()


class TestResolveDurability:
    @pytest.fixture(autouse=True)
    def _server_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "get_default_durability", lambda: "sync")

    def test_run_durability_wins_over_alias_and_server_default(self) -> None:
        request = RunCreate(assistant_id="agent", input={"x": 1}, durability="async", checkpoint_during=False)
        assert _resolve_durability(request) == "async"

    @pytest.mark.parametrize(("checkpoint_during", "expected"), [(True, "async"), (False, "exit")])
    def test_checkpoint_during_maps_like_langgraph(self, checkpoint_during: bool, expected: str) -> None:
        request = RunCreate(assistant_id="agent", input={"x": 1}, checkpoint_during=checkpoint_during)
        assert _resolve_durability(request) == expected

    def test_falls_back_to_server_default(self) -> None:
        assert _resolve_durability(RunCreate(assistant_id="agent", input={"x": 1})) == "sync"

    def test_none_without_run_value_or_server_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "get_default_durability", lambda: None)
        assert _resolve_durability(RunCreate(assistant_id="agent", input={"x": 1})) is None
