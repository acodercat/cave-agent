import os
import pytest_asyncio
from cave_agent.models import OpenAIModel


@pytest_asyncio.fixture
async def model():
    """Provide a real LLM engine for testing, closed after the test."""
    engine = OpenAIModel(
        model_id=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )
    yield engine
    await engine.aclose()
