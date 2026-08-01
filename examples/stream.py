import asyncio
import os

from rich.console import Console
from rich.text import Text

from cave_agent import (
    CaveAgent,
    CodeEvent,
    ExecutionResultEvent,
    FinalResponseEvent,
    StoppedEvent,
    TextEvent,
)
from cave_agent.models import OpenAIModel
from cave_agent.runtime import IPythonRuntime, Type, Variable

# Initialize LLM engine
model = OpenAIModel(
    model_id=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


# Define tools and context
class DataProcessor:
    """A data processor object that can sort lists of numbers"""

    def process(self, data: list) -> list:
        """Sort a list of numbers"""
        return sorted(data)


async def main():
    processor = DataProcessor()
    numbers = [3, 1, 4, 1, 5, 9]

    # Create Variable objects
    processor_var = Variable(
        name="processor",
        value=processor,
        description="A data processor object that can sort lists of numbers\nusage: result = processor.process([3, 1, 4])",
    )

    numbers_var = Variable(
        name="numbers",
        value=numbers,
        description="Input list of numbers to be processed\nusage: print(numbers)  # Access the list directly",
    )

    result_var = Variable(
        name="result",
        description="Store the result of the processing in this variable.\nusage: result = processor.process([3, 1, 4])",
    )

    # Create runtime
    runtime = IPythonRuntime(
        variables=[processor_var, numbers_var, result_var], types=[Type(DataProcessor)]
    )

    # Create agent
    agent = CaveAgent(
        model,
        runtime=runtime,
    )

    console = Console()

    def label(title: str, body: str, style: str) -> None:
        console.print(Text.assemble((f"{title}: ", f"bold {style}"), (body, style)))

    label("User Prompt", "Use processor to sort the numbers", "yellow")

    # Events are frozen dataclasses — match on the type, read named fields.
    async for event in agent.stream_events("Use processor to sort the numbers"):
        match event:
            case TextEvent():
                text = Text(event.content)
                text.stylize("cyan")
                console.print(text, end="", highlight=True)

            case CodeEvent():
                label("Executing code", event.code, "cyan")

            case ExecutionResultEvent() if event.success:
                label("Execution output", event.output, "yellow")

            case ExecutionResultEvent():
                label("Execution error", event.output, "red")

            case FinalResponseEvent():
                label("Final response", event.content, "green")

            case StoppedEvent():
                label(
                    "Stopped",
                    f"{event.stop_reason} after {event.steps} steps, "
                    f"{event.usage.total_tokens} tokens in {event.elapsed:.1f}s",
                    "magenta",
                )

    print("\n")
    label(
        "Runtime State",
        str(
            {
                "processor": await agent.runtime.retrieve("processor"),
                "numbers": await agent.runtime.retrieve("numbers"),
                "result": await agent.runtime.retrieve("result"),
            }
        ),
        "yellow",
    )


if __name__ == "__main__":
    asyncio.run(main())
