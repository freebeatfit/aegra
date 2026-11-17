"""Tests for RunCreate model validation."""

from typing import Any
from uuid import UUID

import pytest
from langgraph_sdk import get_client
from pydantic import ValidationError

from aegra_api.models.runs import RunCreate

_CHECKPOINT_ID = "1ef4f797-8335-6428-8001-8a1503f9b875"


class TestRunCreateValidation:
    """Tests for RunCreate input/command validation."""

    def test_checkpoint_only_payload_preserves_none_input(self):
        """Checkpoint-only payloads must keep input as None so LangGraph resumes
        from the checkpoint instead of restarting the graph from __start__.

        Regression test: previously the validator coerced input to ``{}`` which
        LangGraph Pregel treats as "new input" and re-enters __start__, ignoring
        the checkpoint's ``next=[...]``.
        """
        run_create = RunCreate(
            assistant_id="agent",
            checkpoint={"checkpoint_id": "chk-1", "checkpoint_ns": ""},
        )

        assert run_create.input is None
        assert run_create.command is None
        assert run_create.checkpoint == {"checkpoint_id": "chk-1", "checkpoint_ns": ""}

    def test_rejects_payload_without_input_command_or_checkpoint(self):
        """Ensure payloads with no input, command, or checkpoint are rejected."""
        with pytest.raises(ValueError, match="Must specify at least one of 'input', 'command', or 'checkpoint'"):
            RunCreate(assistant_id="agent")

    def test_top_level_checkpoint_id_counts_as_a_checkpoint(self) -> None:
        run_create = RunCreate(assistant_id="agent", checkpoint_id=_CHECKPOINT_ID)

        assert run_create.checkpoint_id == UUID(_CHECKPOINT_ID)
        assert run_create.input is None

    @pytest.mark.parametrize("checkpoint_id", ["", "not-a-uuid", 123, ["a"]])
    def test_rejects_malformed_checkpoint_id(self, checkpoint_id: Any) -> None:
        with pytest.raises(ValidationError):
            RunCreate(assistant_id="agent", input={"x": 1}, checkpoint_id=checkpoint_id)


class TestRunCreateDurability:
    """``durability`` and its deprecated ``checkpoint_during`` alias are declared, not dropped."""

    def test_defaults_leave_both_unset(self) -> None:
        run_create = RunCreate(assistant_id="agent", input={"x": 1})

        assert run_create.durability is None
        assert run_create.checkpoint_during is None

    @pytest.mark.parametrize("mode", ["sync", "async", "exit"])
    def test_accepts_every_langgraph_mode(self, mode: str) -> None:
        run_create = RunCreate(assistant_id="agent", input={"x": 1}, durability=mode)

        assert run_create.durability == mode

    @pytest.mark.parametrize("mode", ["", "SYNC", "eventual", 1, True])
    def test_rejects_unknown_mode(self, mode: Any) -> None:
        with pytest.raises(ValidationError, match="durability"):
            RunCreate(assistant_id="agent", input={"x": 1}, durability=mode)

    @pytest.mark.parametrize("value", [True, False])
    def test_accepts_checkpoint_during_bool(self, value: bool) -> None:
        run_create = RunCreate(assistant_id="agent", input={"x": 1}, checkpoint_during=value)

        assert run_create.checkpoint_during is value

    def test_rejects_non_bool_checkpoint_during(self) -> None:
        with pytest.raises(ValidationError, match="checkpoint_during"):
            RunCreate(assistant_id="agent", input={"x": 1}, checkpoint_during="sometimes")


class TestRunCreateMetadataValidation:
    """Tests for ``RunCreate.metadata`` shape enforcement.

    The schema constrains user-supplied metadata to OTEL-attribute
    primitives at request time so the contract is honest in the OpenAPI
    schema.  Previously ``dict[str, Any]`` accepted nested values that
    were silently dropped downstream by ``merge_run_metadata``; surfacing
    the rejection as a 422 means clients can fix the payload upstream
    rather than wonder why their metadata never reaches Langfuse.
    """

    def _payload(self, **overrides):
        base = {"assistant_id": "agent", "input": {"x": 1}}
        base.update(overrides)
        return base

    def test_none_metadata_accepted(self):
        run_create = RunCreate(**self._payload(metadata=None))
        assert run_create.metadata is None

    def test_empty_dict_metadata_accepted(self):
        run_create = RunCreate(**self._payload(metadata={}))
        assert run_create.metadata == {}

    def test_all_primitive_types_accepted(self):
        run_create = RunCreate(**self._payload(metadata={"tenant": "acme", "retries": 3, "ratio": 0.5, "flag": True}))
        assert run_create.metadata == {
            "tenant": "acme",
            "retries": 3,
            "ratio": 0.5,
            "flag": True,
        }

    def test_nested_dict_value_rejected(self):
        with pytest.raises(ValidationError):
            RunCreate(**self._payload(metadata={"k": {"nested": 1}}))

    def test_list_value_rejected(self):
        with pytest.raises(ValidationError):
            RunCreate(**self._payload(metadata={"k": [1, 2, 3]}))

    def test_too_many_keys_rejected(self):
        with pytest.raises(ValidationError, match="exceeds 32 keys"):
            RunCreate(**self._payload(metadata={f"k{i}": i for i in range(33)}))

    def test_dotted_key_rejected(self):
        """Reject keys containing ``.`` so users cannot land bare attributes
        like ``langfuse.user.id`` next to the system ones via the
        ``langfuse.trace.metadata.<key>`` channel."""
        with pytest.raises(ValidationError, match="must match"):
            RunCreate(**self._payload(metadata={"langfuse.user.id": "spoof"}))

    def test_empty_key_rejected(self):
        with pytest.raises(ValidationError, match="must match"):
            RunCreate(**self._payload(metadata={"": "v"}))

    def test_too_long_key_rejected(self):
        with pytest.raises(ValidationError, match="must match"):
            RunCreate(**self._payload(metadata={"k" * 65: "v"}))

    def test_non_ascii_key_rejected(self):
        """Reject non-ASCII keys to prevent visual-spoof of join keys
        (e.g. ``run_id`` vs ``run_id​`` zero-width-space variant)."""
        with pytest.raises(ValidationError, match="must match"):
            RunCreate(**self._payload(metadata={"run_id​": "v"}))

    def test_too_long_string_value_rejected(self):
        with pytest.raises(ValidationError, match="exceeds 512 characters"):
            RunCreate(**self._payload(metadata={"k": "v" * 513}))

    def test_max_keys_exactly_32_accepted(self):
        """Boundary: exactly 32 keys is allowed."""
        run_create = RunCreate(**self._payload(metadata={f"k{i}": i for i in range(32)}))
        assert len(run_create.metadata) == 32

    def test_max_value_length_exactly_512_accepted(self):
        """Boundary: exactly 512 chars is allowed."""
        run_create = RunCreate(**self._payload(metadata={"k": "v" * 512}))
        assert len(run_create.metadata["k"]) == 512


async def _sdk_default_bodies() -> list[dict[str, Any]]:
    """The bodies ``langgraph_sdk`` sends for ``runs.create`` and ``runs.wait`` with no options."""
    client = get_client(url="http://sdk.invalid")
    bodies: list[dict[str, Any]] = []

    async def capture(*_args: Any, json: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        bodies.append(json)
        return {}

    client.runs.http.post = capture  # type: ignore[method-assign]
    client.runs.http.request_reconnect = capture  # type: ignore[method-assign]
    await client.runs.create("t1", "agent", input={"x": 1})
    await client.runs.wait("t1", "agent", input={"x": 1})
    return bodies


class TestV1StreamOptions:
    """Streaming is Agent Protocol v2 only: a v1 stream option is refused unless it is the SDK's no-op default."""

    async def test_sdk_default_bodies_are_accepted(self) -> None:
        bodies = await _sdk_default_bodies()

        assert bodies[0]["stream_mode"] == "values"  # the SDK sends its stream defaults on every create
        for body in bodies:
            RunCreate.model_validate(body)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("stream", False),
            ("stream_mode", None),
            ("stream_mode", "values"),
            ("stream_mode", ["values"]),
            ("stream_subgraphs", None),
            ("stream_subgraphs", False),
            ("stream_resumable", False),
            ("on_disconnect", None),
            ("on_disconnect", "continue"),
        ],
    )
    def test_no_op_values_are_accepted(self, field: str, value: Any) -> None:
        RunCreate.model_validate({"assistant_id": "agent", "input": {"x": 1}, field: value})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("stream", True),
            ("stream_mode", "updates"),
            ("stream_mode", ["values", "messages"]),
            ("stream_subgraphs", True),
            ("stream_resumable", True),
            ("on_disconnect", "cancel"),
        ],
    )
    def test_a_value_that_asks_for_a_v1_stream_is_refused(self, field: str, value: Any) -> None:
        with pytest.raises(ValidationError) as exc_info:
            RunCreate.model_validate({"assistant_id": "agent", "input": {"x": 1}, field: value})

        error = exc_info.value.errors()[0]
        assert error["loc"] == (field,)
        assert field in error["msg"]
        assert "Agent Protocol v2" in error["msg"]
