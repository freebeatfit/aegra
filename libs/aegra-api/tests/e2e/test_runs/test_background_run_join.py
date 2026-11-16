import asyncio
import json
from typing import Any

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog, get_e2e_client


def _is_root_terminal(event: dict[str, Any]) -> bool:
    return (
        event["method"] == "lifecycle"
        and not event["params"]["namespace"]
        and event["params"]["data"]["event"] != "running"
    )


async def _read_thread_stream(thread_id: str, *, since: int | None, max_events: int | None) -> list[dict[str, Any]]:
    """Read v2 thread-stream events until *max_events* (a simulated drop) or the run's root lifecycle ends."""
    body: dict[str, Any] = {"channels": ["messages", "values", "lifecycle"]}
    if since is not None:
        body["since"] = since
    events: list[dict[str, Any]] = []
    async with (
        httpx.AsyncClient(base_url=settings.app.SERVER_URL or "", timeout=60.0) as http,
        http.stream("POST", f"/threads/{thread_id}/stream/events", json=body) as resp,
    ):
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            events.append(json.loads(line.removeprefix("data: ")))
            if _is_root_terminal(events[-1]) or (max_events is not None and len(events) >= max_events):
                return events
    return events


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_background_run_and_join_e2e() -> None:
    """
    End-to-end test that:
      1) Creates a background run (non-streaming)
      2) Follows it on the v2 thread stream while it's running
      3) Simulates a network drop after a few events
      4) Reconnects with ``since`` the last seq seen and reads to the end
      5) Joins for final state and validates thread history growth
    """
    client = get_e2e_client()

    assistant = await client.assistants.create(
        graph_id="stress_test",
        config={"tags": ["background"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create", assistant)
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    initial_count = len(await client.threads.get_history(thread_id))

    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant["assistant_id"],
        input={"messages": [{"role": "user", "content": json.dumps({"delay": 0.3, "steps": 4})}]},
    )
    elog("Runs.create (background)", run)
    run_id = run["run_id"]

    first = await _read_thread_stream(thread_id, since=None, max_events=3)
    elog("First session", [(e["seq"], e["method"]) for e in first])
    assert first, "Expected some events before the simulated network drop"

    await asyncio.sleep(0.25)
    last_seq = first[-1]["seq"]
    second = (
        [] if _is_root_terminal(first[-1]) else await _read_thread_stream(thread_id, since=last_seq, max_events=None)
    )
    elog("Rejoin session", [(e["seq"], e["method"]) for e in second])
    assert all(event["seq"] > last_seq for event in second), "Reconnect replayed events already seen"
    assert _is_root_terminal((first + second)[-1]), "Run did not reach its terminal lifecycle"

    final_state = await client.runs.join(thread_id, run_id)
    elog("Runs.join final_state", final_state)
    assert isinstance(final_state, dict)
    assert len(await client.threads.get_history(thread_id)) >= initial_count + 1
