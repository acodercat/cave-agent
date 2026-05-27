"""Timeout demo — test execution timeout with both runtimes.

Usage:
    cd cave-agent && uv run python examples/timeout_demo.py
"""

import asyncio
import os

from rich.console import Console
from rich.rule import Rule

from cave_agent import CaveAgent, Function, IPythonRuntime
from cave_agent.runtime import IPyKernelRuntime
from cave_agent.models.openai import OpenAIServerModel
from cave_agent.display import render_user_prompt

console = Console()

model = OpenAIServerModel(
    model_id=os.getenv("LLM_MODEL_ID", "qwen3.5-397b-a17b"),
    api_key=os.getenv("LLM_API_KEY", "sk-8898a87b8e7b4e399d6fc5ee804c4666"),
    base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)


def slow_compute(n: int) -> str:
    """Simulate a slow computation that takes n seconds."""
    import time
    time.sleep(n)
    return f"Computed after {n}s"


query = "Call slow_compute(10) — it takes 10 seconds but we have a 5 second timeout."


async def main():
    # -- IPythonRuntime (in-process) ---------------------------------------
    console.print()
    console.print(Rule("[bold]IPythonRuntime (in-process)[/]", style="dim"))

    agent = CaveAgent(
        model=model,
        runtime=IPythonRuntime(functions=[Function(slow_compute)]),
        max_exec_timeout=5,
        max_steps=3,
    )
    render_user_prompt(query)
    async for event in agent.stream_events(query):
        pass

    # -- IPyKernelRuntime (isolated process) -------------------------------
    console.print()
    console.print(Rule("[bold]IPyKernelRuntime (isolated process)[/]", style="dim"))

    async with IPyKernelRuntime(functions=[Function(slow_compute)]) as runtime:
        agent = CaveAgent(
            model=model,
            runtime=runtime,
            max_exec_timeout=5,
            max_steps=3,
        )
        render_user_prompt(query)
        async for event in agent.stream_events(query):
            pass


if __name__ == "__main__":
    asyncio.run(main())
