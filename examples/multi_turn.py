from cave_agent import CaveAgent
from cave_agent.models import OpenAIServerModel
from cave_agent.python_runtime import PythonRuntime, Variable
import os
import asyncio

model = OpenAIServerModel(
    model_id=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

# Define data processing class
class DataAnalyzer:
    """A data analyzer that provides statistical analysis for numerical data.
    
    This analyzer calculates basic descriptive statistics including:
    - Minimum value
    - Maximum value
    - Average (mean)
    - Length of data
    
    Example:
        >>> analyzer = DataAnalyzer()
        >>> stats = analyzer.analyze([1, 2, 3, 4, 5])
        >>> print(stats)
        {'min': 1, 'max': 5, 'avg': 3.0, 'len': 5}
    """
    
    def analyze(self, data: list) -> dict:
        """Calculate basic statistical measures for a list of numbers."""
        return {
            'min': min(data),
            'max': max(data),
            'avg': sum(data) / len(data),
            'len': len(data)
        }

async def main():
    # Setup context
    analyzer = DataAnalyzer()
    numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5]

    # Create Variable objects
    analyzer_var = Variable(
        name="analyzer",
        value=analyzer,
        description="Tool for analyzing numerical data\nusage: stats = analyzer.analyze(numbers)"
    )
    
    numbers_var = Variable(
        name="numbers",
        value=numbers,
        description="Input data to analyze\nusage: print(numbers)"
    )
    
    stats_var = Variable(
        name="stats",
        description="Store analysis results here\nusage: stats = analyzer.analyze(numbers)"
    )

    # Create runtime with variables
    runtime = PythonRuntime(
        variables=[analyzer_var, numbers_var, stats_var]
    )

    # Create agent
    agent = CaveAgent(model, runtime=runtime)

    # Multi-turn conversation
    print("Starting analysis conversation...")

    # First turn - get basic stats
    await agent.run("Analyze the numbers and store the results in 'stats'")
    stats = agent.runtime.get_variable_value('stats')
    print("\nBasic stats:", stats)

    # Second turn - ask about specific stat
    response = await agent.run("What is the average value in the stats?")
    print("\nAverage value:", response.content)

    # Third turn - ask for interpretation
    response = await agent.run("Is the maximum value (9) significantly higher than the average?")
    print("\nInterpretation:", response.content)

    # Alternative approach with streaming
    print("\n\n=== Streaming Example ===")
    
    # Create a new agent for streaming demo
    runtime_streaming = PythonRuntime(
        variables=[analyzer_var, numbers_var, stats_var]
    )
    agent_streaming = CaveAgent(model, runtime=runtime_streaming)
    
    async for event in agent_streaming.stream_events(
        "Analyze the numbers, calculate the range (max - min), and tell me if the data is spread out"
    ):
        print(f"[{event.type.value}] {event.content}")

if __name__ == "__main__":
    asyncio.run(main())