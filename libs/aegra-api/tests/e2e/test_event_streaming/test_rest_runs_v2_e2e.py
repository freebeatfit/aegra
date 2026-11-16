"""E2E: streaming is Agent Protocol v2 only, and REST runs ride the v2 event path.

The v1 run-stream endpoints answer 410 with a pointer to v2. A run created
through the REST routes streams the same v2 envelopes on the thread stream
as one started with ``run.start``, and the REST run operations keep working.
Uses the ``stress_test`` graph (no LLM); skipped when the server has v2 off.
"""

import json
from typing import Any

import httpx
import pytest
from langgraph_sdk import get_client

from aegra_api.settings import settings
from tests.e2e._utils import await_terminal_run, elog

_V2_POINTER = {
    "commands": "POST /threads/{thread_id}/commands",
    "stream": "POST /threads/{thread_id}/stream/events",
}


def _base_url() -> str:
    url = settings.app.SERVER_URL
    assert url is not None
    return url


def _input(**config: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": json.dumps({"delay": 0.1, "steps": 1, **config})}]}


async def _setup() -> tuple[Any, str, str]:
    """An SDK client, a stress_test assistant and a fresh thread; skips when v2 is off."""
    client = get_client(url=_base_url())
    thread = await client.threads.create()
    async with httpx.AsyncClient(base_url=_base_url(), timeout=10.0) as http:
        probe = await http.post(f"/threads/{thread['thread_id']}/commands", json={"id": 0, "method": "", "params": {}})
    if probe.status_code == 503:
        pytest.skip("FF_V2_EVENT_STREAMING is disabled on the server under test")
    assistant = await client.assistants.create(graph_id="stress_test", if_exists="do_nothing")
    return client, assistant["assistant_id"], thread["thread_id"]


async def _thread_events_until_root_terminal(thread_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async with (
        httpx.AsyncClient(base_url=_base_url(), timeout=30.0) as http,
        http.stream("POST", f"/threads/{thread_id}/stream/events", json={"channels": ["values", "lifecycle"]}) as resp,
    ):
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line.removeprefix("data: "))
            events.append(event)
            root = event["method"] == "lifecycle" and not event["params"]["namespace"]
            if root and event["params"]["data"]["event"] != "running":
                return events
    return events


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path"),
    [("POST", "/threads/t1/runs/stream"), ("GET", "/threads/t1/runs/r1/stream"), ("POST", "/runs/stream")],
)
async def test_v1_run_streams_answer_410_with_v2_pointer(method: str, path: str) -> None:
    body = {"assistant_id": "stress_test", "input": _input()} if method == "POST" else None
    async with httpx.AsyncClient(base_url=_base_url(), timeout=10.0) as http:
        resp = await http.request(method, path, json=body)

    assert resp.status_code == 410
    assert resp.json()["error"] == "gone"
    assert resp.json()["details"] == _V2_POINTER


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_rest_run_emits_v2_events_on_thread_stream() -> None:
    client, assistant_id, thread_id = await _setup()

    run = await client.runs.create(thread_id, assistant_id, input=_input())
    events = await _thread_events_until_root_terminal(thread_id)
    elog("thread stream", [(e["method"], e["event_id"]) for e in events])

    assert all(event["type"] == "event" for event in events)
    values = [event["params"]["data"] for event in events if event["method"] == "values"]
    assert values and values[-1]["step_count"] == 1
    root = [event for event in events if event["method"] == "lifecycle" and not event["params"]["namespace"]]
    assert [event["params"]["data"]["event"] for event in root] == ["running", "completed"]
    assert all(event["event_id"].startswith(f"synth:{run['run_id']}:") for event in root)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_rest_run_operations_keep_working() -> None:
    client, assistant_id, thread_id = await _setup()

    waited = await client.runs.wait(thread_id, assistant_id, input=_input())
    state = await client.threads.get_state(thread_id)
    assert waited == state["values"]

    run = await client.runs.create(thread_id, assistant_id, input=_input())
    joined = await client.runs.join(thread_id, run["run_id"])
    assert joined == (await client.threads.get_state(thread_id))["values"]
    got = await client.runs.get(thread_id, run["run_id"])
    assert got["status"] == "success"
    assert got["output"] is None
    assert run["run_id"] in [listed["run_id"] for listed in await client.runs.list(thread_id)]

    slow = await client.runs.create(thread_id, assistant_id, input=_input(delay=1.0, steps=10))
    await client.runs.cancel(thread_id, slow["run_id"])
    assert (await await_terminal_run(client, thread_id, slow["run_id"]))["status"] == "interrupted"
    await client.runs.delete(thread_id, slow["run_id"])
    async with httpx.AsyncClient(base_url=_base_url(), timeout=10.0) as http:
        gone = await http.get(f"/threads/{thread_id}/runs/{slow['run_id']}")
    assert gone.status_code == 404
