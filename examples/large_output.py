"""Oversize execution output stays reachable.

When a code block prints more than ``max_exec_output`` characters, the full text
is not discarded — it is bound to a runtime variable and the model receives a
marker naming it plus a leading preview. Because the runtime *is* Python, the
model can then slice, filter or re-parse that variable on the next turn instead
of re-running a query that may be slow, expensive, or not reproducible at all.

Run:
    LLM_MODEL_ID=... LLM_API_KEY=... LLM_BASE_URL=... python -m examples.large_output
"""

import asyncio
import os

from cave_agent import CaveAgent
from cave_agent.models import OpenAIModel
from cave_agent.renderers import render_run
from cave_agent.runtime import Function, IPythonRuntime


def fetch_log() -> str:
    """Return a large application log. Slow and non-reproducible — call once."""
    lines = []
    for i in range(4000):
        level = "ERROR" if i % 500 == 0 else "INFO"
        lines.append(
            f"2026-07-28 10:{i % 60:02d}:00 {level} worker-{i % 8} request={i} latency={i % 300}ms"
        )
    return "\n".join(lines)


async def main():
    model = OpenAIModel(
        model_id=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    runtime = IPythonRuntime(functions=[Function(fetch_log)])
    agent = CaveAgent(
        model,
        runtime=runtime,
        # Deliberately small so the marker path is easy to observe. The default
        # is 5000; the full text is kept regardless of how low this goes.
        max_exec_output=1500,
    )

    try:
        # Turn 1 — the output is far larger than the budget, so the model gets a
        # <persisted-output> marker instead of the text.
        await render_run(agent, "Call fetch_log() and print everything it returns.")

        # The full output is in the runtime, untouched by the truncation. Each
        # oversize output gets its own name, so this is `_output_1` — the second
        # would be `_output_2`, leaving the first marker still pointing at the
        # text it actually described.
        stashed = await runtime.get_from_namespace("_output_1")
        print(f"\n[host] _output_1 holds {len(stashed):,} characters\n")

        # Turn 2 — the model works from that variable. No second fetch_log() call.
        await render_run(
            agent,
            "How many ERROR lines were there, and what is the highest latency? "
            "Use the output you already have.",
        )
    finally:
        await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
