import pytest

from tests.e2e._utils import elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_run_join_returns_actual_output():
    """
    Test that the join endpoint returns the run's result instead of an empty dict.

    A run stores no copy of its final state (runs.output is null, as in the
    LangGraph Agent Server); join answers from the thread's current values.
    """
    client = get_e2e_client()

    # Create assistant
    assistant = await client.assistants.create(
        graph_id="stress_test",
        config={"tags": ["join-output-test"]},
        if_exists="do_nothing",
    )
    elog("Assistant.create", assistant)
    assistant_id = assistant["assistant_id"]

    # Create thread
    thread = await client.threads.create()
    elog("Threads.create", thread)
    thread_id = thread["thread_id"]

    # Create run that should produce meaningful output
    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input={"messages": [{"role": "user", "content": '{"delay": 0.1, "steps": 1}'}]},
    )
    elog("Runs.create", run)
    run_id = run["run_id"]

    # Join the run and get final output
    final_output = await client.runs.join(thread_id, run_id)
    elog("Runs.join final_output", final_output)

    # Verify output is not empty (the main fix)
    assert final_output is not None, "Final output should not be None"
    assert final_output != {}, "Final output should not be empty dict - this was the bug!"

    # The output should be a dict containing the final state
    assert isinstance(final_output, dict), f"Expected dict output, got {type(final_output)}"

    # Verify the run details also show the correct output
    run_details = await client.runs.get(thread_id, run_id)
    elog("Runs.get details", run_details)

    assert run_details["status"] == "success", f"Expected success status, got {run_details['status']}"
    assert run_details["output"] is None, "The run should store no copy of the final state"

    # Join answers from the thread's current values
    state = await client.threads.get_state(thread_id)
    assert final_output == state["values"], "Join output should match the thread's values"

    elog(
        "Join returned the thread values",
        {
            "output_type": type(final_output).__name__,
            "output_empty": final_output == {},
            "output_keys": list(final_output.keys()) if isinstance(final_output, dict) else "not_dict",
        },
    )
