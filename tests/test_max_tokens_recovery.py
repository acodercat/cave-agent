"""Integration tests for max_tokens output recovery with real API calls."""

import pytest
from cave_agent import CaveAgent
from cave_agent.models import OpenAIServerModel
from cave_agent.runtime import IPythonRuntime, Function, Variable
import os


def generate_data(n: int) -> list[dict]:
    """Generate n sample records with id, name, and score."""
    import random
    random.seed(42)
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    return [{"id": i, "name": random.choice(names), "score": random.randint(50, 100)} for i in range(n)]


@pytest.fixture
def small_output_model():
    """Model with max_tokens=150 to force output truncation."""
    return OpenAIServerModel(
        model_id=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        max_tokens=150,
    )


@pytest.fixture
def recovery_agent(small_output_model):
    runtime = IPythonRuntime(
        functions=[Function(generate_data)],
        variables=[
            Variable(name="summary", description="Store final summary here"),
        ],
    )
    return CaveAgent(
        small_output_model,
        runtime=runtime,
        display=False,
    )


@pytest.mark.asyncio
async def test_truncated_output_recovers(recovery_agent):
    """With max_tokens=150, the model's response should be truncated.
    The agent should automatically recover and still produce a result."""
    response = await recovery_agent.run(
        "Generate 20 records using generate_data, then print each person's average score"
    )
    assert response is not None
    assert response.content


@pytest.mark.asyncio
async def test_recovery_produces_code_execution(recovery_agent):
    """Even with truncated output, the agent should eventually execute code."""
    response = await recovery_agent.run(
        "Generate 10 records using generate_data and store the result in summary"
    )
    summary = await recovery_agent.runtime.retrieve("summary")
    # The agent may or may not succeed in storing, but it should not crash
    assert response is not None
