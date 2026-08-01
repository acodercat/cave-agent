"""Execution timeouts — and why they need a preemptible runtime.

``max_exec_timeout`` is only offered by backends that can actually stop running
code. ``IPyKernelRuntime`` can: the kernel is a separate process, so an
interrupt reaches the cell. ``IPythonRuntime`` cannot — generated code runs
inline in this process, where a deadline can abandon the code but never stop
it, and abandoning it leaves the shell's stdout capture unwound out of order.

So the combination is refused at construction rather than approximated. This
example shows both halves: the refusal, then a timeout that really fires.

Run:
    LLM_MODEL_ID=... LLM_API_KEY=... LLM_BASE_URL=... python -m examples.timeout_demo
"""

import asyncio
import os

from rich.console import Console
from rich.rule import Rule

from cave_agent import CaveAgent, Function, IPythonRuntime, PreemptibleRuntime
from cave_agent.models.openai import OpenAIModel
from cave_agent.renderers import render_run
from cave_agent.runtime import IPyKernelRuntime

console = Console()


def slow_compute(n: int) -> str:
    """Simulate a slow computation that takes n seconds."""
    import time

    time.sleep(n)
    return f"Computed after {n}s"


QUERY = "Call slow_compute(10) — it takes 10 seconds but we have a 5 second timeout."


async def main():
    model = OpenAIModel(
        model_id=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    # -- IPythonRuntime: not preemptible, so the timeout is refused ---------
    console.print()
    console.print(Rule("[bold]IPythonRuntime (in-process)[/]", style="dim"))

    in_process = IPythonRuntime(functions=[Function(slow_compute)])
    console.print(
        f"  isinstance(runtime, PreemptibleRuntime) = {isinstance(in_process, PreemptibleRuntime)}"
    )
    try:
        CaveAgent(model=model, runtime=in_process, max_exec_timeout=5)
    except ValueError as error:
        console.print(f"  [yellow]refused:[/] {error}")

    # -- IPyKernelRuntime: separate process, so interrupt() lands ----------
    console.print()
    console.print(Rule("[bold]IPyKernelRuntime (isolated process)[/]", style="dim"))

    try:
        async with IPyKernelRuntime(functions=[Function(slow_compute)]) as runtime:
            agent = CaveAgent(
                model=model,
                runtime=runtime,
                max_exec_timeout=5,
                max_steps=3,
            )
            await render_run(agent, QUERY)
    finally:
        await model.aclose()


if __name__ == "__main__":
    asyncio.run(main())
