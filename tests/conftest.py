"""Shared fixtures.

The :func:`model` fixture reaches a **real** LLM, so every test that asks for
one costs tokens and needs an endpoint. Those tests skip themselves when none
is configured, which is what lets a fresh clone run ``pytest`` and read the
result — rather than a wall of credential errors from tests it was never
going to be able to run.
"""

import os

import pytest
import pytest_asyncio

from cave_agent.models import OpenAIModel

#: What a live run needs. All three, because a partial set is a
#: misconfiguration rather than an opt-out — and because requiring the model
#: id and base URL as well as the key makes it unlikely that an ``LLM_API_KEY``
#: exported for something else quietly starts spending on this suite.
_LIVE_ENV = ("LLM_MODEL_ID", "LLM_API_KEY", "LLM_BASE_URL")


@pytest.fixture
def live_llm_env():
    """Skip unless a live endpoint is configured.

    A fixture rather than a helper so any model fixture — here or in a test
    module with its own tuning — shares one gate by depending on it, instead
    of each repeating the check and drifting from the others.
    """
    missing = [name for name in _LIVE_ENV if not os.getenv(name)]
    if missing:
        pytest.skip(f"needs a live LLM endpoint; set {', '.join(missing)} to run it")


@pytest_asyncio.fixture
async def model(live_llm_env):
    """A real LLM engine, closed after the test."""
    engine = OpenAIModel(
        model_id=os.getenv("LLM_MODEL_ID"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )
    yield engine
    await engine.aclose()
