"""E2E tests for the worker executor architecture.

These tests verify that runs execute correctly through the Redis job
queue → worker coroutine → broker pipeline, not just via in-process
asyncio tasks.  They exercise:

  - Basic run execution via worker (create → join → verify output)
  - Concurrent runs processed by multiple workers
  - Wait endpoint returning results from worker execution
  - Cancel propagation via Redis pub/sub to worker
  - Stateless run execution via worker
  - The v2 thread stream carrying a worker-executed run's events
"""

import asyncio
import json

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import await_terminal_run, check_and_skip_if_geo_blocked, elog, get_e2e_client


async def _run_and_check_geo(client, thread_id: str, run_id: str) -> dict:
    """Join a run and skip if geo-blocked."""
    output = await client.runs.join(thread_id=thread_id, run_id=run_id)
    run = await client.runs.get(thread_id=thread_id, run_id=run_id)
    check_and_skip_if_geo_blocked(run)
    return output


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.asyncio
async def test_worker_executes_run() -> None:
    """A run submitted to the queue is picked up by a worker and produces output."""
    client = get_e2e_client()
    # geo-block check happens after run completes

    thread = await client.threads.create()
    elog("Thread", thread)

    run = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Say 'worker-ok' and nothing else."}]},
    )
    elog("Run created", run)
    assert run["status"] in ("pending", "running")

    output = await _run_and_check_geo(client, thread["thread_id"], run["run_id"])
    elog("Join output", output)
    assert output, "Expected non-empty output from worker execution"


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.asyncio
async def test_worker_concurrent_runs() -> None:
    """Multiple runs execute concurrently across workers."""
    client = get_e2e_client()
    # geo-block check happens after run completes

    threads_and_runs: list[tuple[str, str]] = []

    for i in range(3):
        thread = await client.threads.create()
        run = await client.runs.create(
            thread_id=thread["thread_id"],
            assistant_id="agent",
            input={"messages": [{"role": "user", "content": f"Say 'concurrent-{i}' and nothing else."}]},
        )
        threads_and_runs.append((thread["thread_id"], run["run_id"]))
        elog(f"Run {i} created", run)

    results = await asyncio.gather(*[client.runs.join(thread_id=tid, run_id=rid) for tid, rid in threads_and_runs])

    for i, result in enumerate(results):
        elog(f"Run {i} result", result)
        assert result, f"Run {i} returned empty output"

    elog("All concurrent runs completed", {"count": len(results)})


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.asyncio
async def test_worker_wait_endpoint() -> None:
    """The /runs/wait endpoint returns output after worker execution."""
    client = get_e2e_client()
    # geo-block check happens after run completes

    thread = await client.threads.create()
    elog("Thread", thread)

    output = await client.runs.wait(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Say 'wait-ok' and nothing else."}]},
    )
    elog("Wait output", output)
    assert output, "Expected non-empty output from wait endpoint"


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.asyncio
async def test_worker_cancel_via_redis() -> None:
    """Cancel propagates through Redis pub/sub to the executing worker."""
    client = get_e2e_client()
    # geo-block check happens after run completes

    thread = await client.threads.create()

    run = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Write a very long essay about the history of computing."}]},
    )
    elog("Run created", run)

    await asyncio.sleep(1)

    cancelled = await client.runs.cancel(
        thread_id=thread["thread_id"],
        run_id=run["run_id"],
    )
    elog("Cancel response", cancelled)

    # The worker owns this run, so it finalizes asynchronously after the cancel.
    final_run = await await_terminal_run(client, thread["thread_id"], run["run_id"])
    check_and_skip_if_geo_blocked(final_run)
    elog("Final run state", final_run)
    assert final_run["status"] == "interrupted", f"Expected interrupted, got {final_run['status']}"


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.asyncio
async def test_worker_stateless_run() -> None:
    """Stateless /runs/wait works through the worker pipeline."""
    client = get_e2e_client()

    # Stateless runs still need a thread via SDK — use an ephemeral one
    thread = await client.threads.create()
    output = await client.runs.wait(
        thread_id=thread["thread_id"],
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Say 'stateless-ok' and nothing else."}]},
    )
    elog("Stateless wait output", output)
    assert output, "Expected non-empty output from stateless wait"


@pytest.mark.e2e
@pytest.mark.prod_only
@pytest.mark.asyncio
async def test_worker_stream_produces_events() -> None:
    """A run executed by a worker streams its v2 events on the thread stream."""
    client = get_e2e_client()
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    elog("Thread", thread)
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id="agent",
        input={"messages": [{"role": "user", "content": "Say 'stream-ok' and nothing else."}]},
    )

    methods: list[str] = []
    async with (
        httpx.AsyncClient(base_url=settings.app.SERVER_URL or "", timeout=60.0) as http,
        http.stream("POST", f"/threads/{thread_id}/stream/events", json={"channels": ["values", "lifecycle"]}) as resp,
    ):
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line.removeprefix("data: "))
            methods.append(event["method"])
            root = event["method"] == "lifecycle" and not event["params"]["namespace"]
            if root and event["params"]["data"]["event"] != "running":
                break

    check_and_skip_if_geo_blocked(await client.runs.get(thread_id=thread_id, run_id=run["run_id"]))
    elog("Stream events", methods)
    assert "lifecycle" in methods, "Expected lifecycle events in stream"
    assert "values" in methods, "Expected values events in stream"
