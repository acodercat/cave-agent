"""Integration tests for context compaction with real API calls."""

import pytest
from cave_agent import CaveAgent
from cave_agent.runtime import IPythonRuntime, Function, Variable


def analyze(data: list) -> dict:
    """Compute basic statistics for a list of numbers."""
    return {
        "count": len(data),
        "sum": sum(data),
        "mean": sum(data) / len(data) if data else 0,
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }


@pytest.fixture
def compaction_agent(model):
    runtime = IPythonRuntime(
        functions=[Function(analyze)],
        variables=[
            Variable(
                name="sales",
                value=[120, 340, 250, 890, 670, 430, 510, 780, 360, 290, 150, 920],
                description="Monthly sales data for the year",
            ),
            Variable(name="result", description="Store analysis results here"),
        ],
    )
    return CaveAgent(
        model,
        runtime=runtime,
        context_window=5_000,
        display=False,
    )


@pytest.mark.asyncio
async def test_compaction_triggers_on_long_conversation(compaction_agent):
    """After several turns with a small context_window, compaction should fire
    and the agent should still be able to answer correctly."""
    await compaction_agent.run("Analyze the sales data using the analyze function and store in result")
    result = await compaction_agent.runtime.retrieve("result")
    assert result is not None
    assert "count" in result

    await compaction_agent.run("Which month had the highest sales?")
    await compaction_agent.run("Calculate the average sales per quarter")

    # After 3 turns with context_window=5000, compaction should have fired.
    # The agent should still work correctly with compacted context.
    response = await compaction_agent.run("What is the total annual sales?")
    assert response is not None
    assert response.content  # non-empty response


@pytest.mark.asyncio
async def test_compaction_preserves_runtime_state(compaction_agent):
    """Compaction should not affect runtime variables — only message history."""
    await compaction_agent.run("Use analyze on the sales data and store in result")
    result_before = await compaction_agent.runtime.retrieve("result")

    # Force more turns to trigger compaction
    await compaction_agent.run("What is the mean value?")
    await compaction_agent.run("List the months above average")

    # Runtime state should be intact even after compaction
    result_after = await compaction_agent.runtime.retrieve("result")
    assert result_after == result_before
