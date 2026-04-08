"""API retry and output recovery demo.

Demonstrates:
1. Automatic retry with exponential backoff on API errors
2. Output truncation recovery (finish_reason="length")

Usage:
    uv run python examples/retry_demo.py
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


def generate_data(n: int) -> list[dict]:
    """Generate n sample records with id, name, and score."""
    import random
    random.seed(42)
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank"]
    return [
        {"id": i, "name": random.choice(names), "score": random.randint(50, 100)}
        for i in range(n)
    ]


async def main():
    print("=== Retry & Recovery Demo ===\n")
    print("This demo uses a real API call. Retry is transparent — if a 429/5xx")
    print("error occurs, you'll see retry logs. Output truncation recovery is")
    print("triggered when the model's response exceeds max_tokens.\n")

    runtime = IPythonRuntime(
        functions=[Function(generate_data)],
        variables=[
            Variable(name="summary", description="Store the final summary here"),
        ],
    )

    agent = CaveAgent(
        model,
        runtime=runtime,
        display=True,
    )

    # This prompt asks for a long, detailed output that may trigger
    # finish_reason="length" if the model's max_tokens is low.
    await agent.run(
        "Generate 50 sample records using generate_data, then produce a detailed "
        "statistical analysis: group by name, compute per-person average/min/max "
        "scores, rank them, and store a formatted summary in the 'summary' variable."
    )

    summary = await runtime.retrieve("summary")
    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
