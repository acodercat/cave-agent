"""Integration tests for max_tokens output recovery with real API calls."""

import os

import pytest
import pytest_asyncio

from cave_agent import CaveAgent
from cave_agent.models import OpenAIModel
from cave_agent.runtime import Function, IPythonRuntime, Variable


def generate_data(n: int) -> list[dict]:
    """Generate n sample records with id, name, and score."""
    import random

    random.seed(42)
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    return [
        {"id": i, "name": random.choice(names), "score": random.randint(50, 100)} for i in range(n)
    ]


@pytest_asyncio.fixture
async def small_output_model(live_llm_env):
    """Model with max_tokens=150 to force output truncation.

    Closed like the shared ``model`` fixture: ``OpenAIModel`` builds its
    ``AsyncOpenAI`` eagerly, so returning one without closing leaks a
    connection pool and an unclosed-client warning at loop teardown.
    """
    engine = OpenAIModel(
        model_id=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        max_tokens=150,
    )
    yield engine
    await engine.aclose()


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
    await recovery_agent.runtime.retrieve("summary")
    assert response is not None
