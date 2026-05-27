"""Context compaction demo — observe compaction in action.

Uses a small context_window (35k) so that a few rounds of code execution
push the token count over the compaction threshold, triggering
"● Compacting conversation..." in the terminal.

Usage:
    uv run python examples/compaction_demo.py
"""

import asyncio
import os

from cave_agent import CaveAgent
from cave_agent.models import OpenAIServerModel
from cave_agent.runtime import IPythonRuntime, Function, Variable

model = OpenAIServerModel(
    model_id=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


def analyze(data: list) -> dict:
    """Compute basic statistics for a list of numbers."""
    return {
        "count": len(data),
        "sum": sum(data),
        "mean": sum(data) / len(data) if data else 0,
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }


async def main():
    runtime = IPythonRuntime(
        functions=[Function(analyze)],
        variables=[
            Variable(
                name="sales",
                value=[120, 340, 250, 890, 670, 430, 510, 780, 360, 290, 150, 920],
                description="Monthly sales data for the year",
            ),
            Variable(name="report", description="Store the final analysis report here"),
        ],
    )

    # Small context_window → compaction triggers after a few rounds
    agent = CaveAgent(
        model,
        runtime=runtime,
        context_window=3_000,
        display=True,
    )

    # Each turn generates code + execution result, filling up context quickly.
    # After a few turns the compaction threshold is exceeded.
    await agent.run("Analyze the sales data using the analyze function")
    await agent.run("Find which months had above-average sales")
    await agent.run("Calculate the total revenue and the quarter-by-quarter breakdown")
    await agent.run("What's the month-over-month growth rate?")
    await agent.run("Summarize everything into a report variable")

    report = await runtime.retrieve("report")
    print("\n=== Final Report ===")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
