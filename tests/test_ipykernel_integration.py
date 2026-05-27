"""Integration tests for CaveAgent + IPyKernelRuntime with a real LLM."""

import pytest
import pytest_asyncio
from dataclasses import dataclass

from cave_agent import CaveAgent
from cave_agent.runtime import IPyKernelRuntime, Function, Variable


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------


class DataAnalyzer:
    """A data analyzer that provides statistical analysis for numerical data."""

    def analyze(self, data: list) -> dict:
        """Calculate basic statistical measures for a list of numbers."""
        return {
            "min": min(data),
            "max": max(data),
            "avg": sum(data) / len(data),
            "len": len(data),
        }


@dataclass
class DataProcessor:
    """A utility class for processing and filtering data collections.

    Example:
        >>> processor = DataProcessor()
        >>> processor.process_list([3, 1, 2, 1, 3])
        [1, 2, 3]
        >>> processor.filter_numbers([1, 5, 3, 8, 2], 4)
        [5, 8]
    """

    def process_list(self, data: list) -> list:
        """Sort a list and remove duplicates"""
        return sorted(set(data))

    def filter_numbers(self, data: list, threshold: int) -> list:
        """Filter numbers greater than threshold"""
        return [x for x in data if x > threshold]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUMBERS = [3, 1, 4, 1, 5, 9, 2, 6, 5]


@pytest_asyncio.fixture
async def multi_turn_agent(model):
    """Agent with DataAnalyzer for multi-turn analysis."""
    runtime = IPyKernelRuntime(
        variables=[
            Variable("analyzer", DataAnalyzer(), "Tool for analyzing numerical data\nusage: stats = analyzer.analyze(numbers)"),
            Variable("numbers", NUMBERS, "Input data to analyze\nusage: print(numbers)"),
            Variable("stats", description="Store analysis results here\nusage: stats = analyzer.analyze(numbers)"),
        ],
    )
    await runtime.start()
    agent = CaveAgent(model, runtime=runtime)
    yield agent
    await runtime.stop()


@pytest_asyncio.fixture
async def object_agent(model):
    """Agent with DataProcessor for object method calls."""
    runtime = IPyKernelRuntime(
        variables=[
            Variable("processor", DataProcessor(), "Data processing tool\nusage: processed_data = processor.process_list(numbers)"),
            Variable("numbers", NUMBERS, "Input list of numbers\nusage: filtered_data = processor.filter_numbers(numbers, 5)"),
            Variable("processed_data", description="Store processed data here\nusage: processed_data = processor.process_list(numbers)"),
            Variable("filtered_data", description="Store filtered data here\nusage: filtered_data = processor.filter_numbers(numbers, 5)"),
        ],
    )
    await runtime.start()
    agent = CaveAgent(model, runtime=runtime)
    yield agent
    await runtime.stop()


@pytest_asyncio.fixture
async def calc_agent(model):
    """Agent with injected functions."""

    def add(a: int, b: int) -> int:
        """Add two numbers together"""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together"""
        return a * b

    runtime = IPyKernelRuntime(functions=[Function(add), Function(multiply)])
    await runtime.start()
    agent = CaveAgent(model, runtime=runtime)
    yield agent
    await runtime.stop()


# ---------------------------------------------------------------------------
# Multi-turn conversation (mirrors test_multi_turn.py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_analysis(multi_turn_agent):
    await multi_turn_agent.run("Analyze the numbers and store results in 'stats'")
    stats = await multi_turn_agent.runtime.retrieve("stats")

    assert stats["min"] == 1
    assert stats["max"] == 9
    assert stats["avg"] == 4.0
    assert stats["len"] == 9


@pytest.mark.asyncio
async def test_multi_turn_conversation(multi_turn_agent):
    # Turn 1: analyze
    await multi_turn_agent.run("Analyze the numbers and store results in 'stats'")
    stats = await multi_turn_agent.runtime.retrieve("stats")
    assert stats is not None
    assert all(k in stats for k in ["min", "max", "avg", "len"])

    # Turn 2: query previous result
    response = await multi_turn_agent.run("What is the average value in the stats?")
    assert "4" in response.content.lower() or "four" in response.content.lower()

    # Turn 3: reasoning about data
    response = await multi_turn_agent.run("Is the maximum value (9) significantly higher than the average?")
    assert any(word in response.content.lower() for word in ["yes", "higher", "greater", "more"])


# ---------------------------------------------------------------------------
# Object method calls (mirrors test_object_methods.py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_and_deduplicate(object_agent):
    await object_agent.run("Use processor to sort and deduplicate numbers")
    processed_data = await object_agent.runtime.retrieve("processed_data")
    expected = [1, 2, 3, 4, 5, 6, 9]
    assert sorted(set(processed_data)) == sorted(set(expected))


@pytest.mark.asyncio
async def test_filter_numbers(object_agent):
    await object_agent.run("Filter numbers greater than 4")
    filtered_data = await object_agent.runtime.retrieve("filtered_data")
    expected = [5, 6, 9]
    assert sorted(set(filtered_data)) == sorted(set(expected))


# ---------------------------------------------------------------------------
# Function calling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_function_call(calc_agent):
    response = await calc_agent.run("Calculate 5 plus 3")
    assert "8" in response.content


@pytest.mark.asyncio
async def test_multi_turn_with_functions(calc_agent):
    """State persists across turns with kernel runtime."""
    await calc_agent.run("Set x = add(10, 20)")
    response = await calc_agent.run("Now multiply x by 3")
    assert "90" in response.content


# ---------------------------------------------------------------------------
# Kernel isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kernel_survives_error(model):
    """Kernel stays alive after an error, next turn works."""
    runtime = IPyKernelRuntime(
        variables=[Variable("result", description="Store result here")],
    )
    await runtime.start()
    try:
        agent = CaveAgent(model, runtime=runtime, instructions="Execute exactly what the user asks.")
        # First turn: trigger an error
        await agent.run("Calculate 1/0 and catch the error, then set result = 'recovered'")
        result = await runtime.retrieve("result")
        assert result == "recovered"
    finally:
        await runtime.stop()
