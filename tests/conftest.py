import pytest
import os
from cave_agent.models import OpenAIServerModel

@pytest.fixture
def model():
    """Provide a real LLM engine for testing."""
    return OpenAIServerModel(
        model_id=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL")
    ) 