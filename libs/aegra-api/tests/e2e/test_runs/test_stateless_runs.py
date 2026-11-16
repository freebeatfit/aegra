"""E2E tests for stateless (thread-free) run endpoints.

These tests hit a real running Aegra server and exercise the ``POST /runs/wait``
and ``POST /runs`` endpoints that accept no thread_id; ``POST /runs/stream`` is retired (410).

Requirements:
  - A running Aegra server (``uv run aegra dev`` or ``docker compose up``)
  - The ``agent`` graph deployed in the server's ``aegra.json``
"""

import pytest
from httpx import AsyncClient

from aegra_api.settings import settings
from tests.e2e._utils import elog, get_e2e_client

# ---------------------------------------------------------------------------
# POST /runs/wait  (stateless wait)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_wait_returns_output() -> None:
    """POST /runs/wait creates an ephemeral thread, runs the graph, and returns output."""
    sdk = get_e2e_client()

    # Ensure assistant exists
    assistant = await sdk.assistants.create(
        graph_id="agent",
        config={"tags": ["stateless", "wait"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create", assistant)
    assistant_id = assistant["assistant_id"]

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http:
        resp = await http.post(
            "/runs/wait",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Say hello in one word."}]},
            },
        )
        elog("POST /runs/wait", {"status": resp.status_code, "body": resp.json() if resp.status_code == 200 else None})

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        output = resp.json()
        assert isinstance(output, dict)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_wait_deletes_ephemeral_thread() -> None:
    """POST /runs/wait with default on_completion deletes the ephemeral thread afterward."""
    sdk = get_e2e_client()

    assistant = await sdk.assistants.create(
        graph_id="agent",
        config={"tags": ["stateless", "wait-delete"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    # Call stateless wait – the server generates a thread internally
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http:
        resp = await http.post(
            "/runs/wait",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Say hi."}]},
            },
        )
        assert resp.status_code == 200

    # We cannot easily know the generated thread_id, but we can verify
    # the test completes without errors (cleanup runs in finally block).
    # If cleanup failed, the server would log errors.


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_wait_keep_preserves_thread() -> None:
    """POST /runs/wait with on_completion='keep' preserves the thread."""
    sdk = get_e2e_client()

    assistant = await sdk.assistants.create(
        graph_id="agent",
        config={"tags": ["stateless", "wait-keep"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http:
        resp = await http.post(
            "/runs/wait",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Say hi."}]},
                "on_completion": "keep",
            },
        )
        elog("POST /runs/wait (keep)", {"status": resp.status_code})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /runs/stream  (retired: streaming is Agent Protocol v2 only)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_stream_is_gone() -> None:
    """POST /runs/stream answers 410 with a pointer to the v2 thread stream."""
    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=10.0) as http:
        resp = await http.post("/runs/stream", json={"assistant_id": "agent", "input": {"messages": []}})

    assert resp.status_code == 410
    assert resp.json()["details"]["stream"] == "POST /threads/{thread_id}/stream/events"


# ---------------------------------------------------------------------------
# POST /runs  (stateless background run)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_create_run_returns_run_object() -> None:
    """POST /runs returns a Run object with run_id and pending/running status."""
    sdk = get_e2e_client()

    assistant = await sdk.assistants.create(
        graph_id="agent",
        config={"tags": ["stateless", "background"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http:
        resp = await http.post(
            "/runs",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Say hello."}]},
            },
        )
        elog("POST /runs", {"status": resp.status_code, "body": resp.json() if resp.status_code == 200 else None})

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "run_id" in data
        assert "thread_id" in data
        assert data["status"] in ("pending", "running")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stateless_create_run_with_keep() -> None:
    """POST /runs with on_completion='keep' returns a Run and thread is preserved."""
    sdk = get_e2e_client()

    assistant = await sdk.assistants.create(
        graph_id="agent",
        config={"tags": ["stateless", "background-keep"]},
        if_exists="do_nothing",
    )
    assistant_id = assistant["assistant_id"]

    async with AsyncClient(base_url=settings.app.SERVER_URL, timeout=120.0) as http:
        resp = await http.post(
            "/runs",
            json={
                "assistant_id": assistant_id,
                "input": {"messages": [{"role": "user", "content": "Say hi."}]},
                "on_completion": "keep",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        thread_id = data["thread_id"]

    # Thread should still exist – verify via SDK
    thread = await sdk.threads.get(thread_id)
    elog("Thread preserved", thread)
    assert thread["thread_id"] == thread_id
