"""Tests for execution timeout — no real LLM or runtime needed."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cave_agent.agent import CaveAgent, EventType, _ExecutionContext
from cave_agent.runtime.executor import ExecutionResult


class MockRuntime:
    """Runtime that simulates slow execution."""

    def __init__(self, delay: float = 0):
        self._delay = delay
        self.interrupted = False

    async def execute(self, code: str) -> ExecutionResult:
        await asyncio.sleep(self._delay)
        return ExecutionResult(stdout=f"OK after {self._delay}s")

    async def interrupt(self) -> None:
        self.interrupted = True

    def describe_functions(self) -> str:
        return "No functions"

    def describe_variables(self) -> str:
        return "No variables"

    def describe_types(self) -> str:
        return "No types"


class MockModel:
    """Model that returns a code block response."""

    def __init__(self, code: str = "print('hello')"):
        self._code = code
        self.last_stream_usage = MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    async def call(self, messages):
        return MagicMock(
            content=f"```python\n{self._code}\n```",
            token_usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def stream(self, messages):
        return MockStreamResponse(self._code)


class MockStreamResponse:
    def __init__(self, code: str):
        self._chunks = list(f"```python\n{code}\n```")
        self._index = 0
        self.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


def _make_agent(runtime_delay: float, timeout: float | None) -> CaveAgent:
    """Create an agent with mocked runtime and model."""
    agent = CaveAgent.__new__(CaveAgent)
    agent.model = MockModel("import time; time.sleep(999)")
    agent.runtime = MockRuntime(delay=runtime_delay)
    agent.max_steps = 3
    agent.max_history = 20
    agent.max_exec_output = 5000
    agent.max_exec_timeout = timeout
    agent.display = False
    agent.instructions = ""
    agent.system_instructions = ""
    agent.system_prompt_template = "{instructions}{system_instructions}{functions}{variables}{types}{skills}"
    agent.python_block_identifier = "python"
    agent.messages = []
    agent._init_skills(None)
    return agent


@pytest.mark.asyncio
async def test_timeout_triggers():
    """Execution slower than timeout should return timeout error."""
    agent = _make_agent(runtime_delay=3.0, timeout=1.0)
    context = _ExecutionContext(max_steps=5)

    result = await agent._execute_code("slow code", context)

    assert result.event_type == EventType.EXECUTION_ERROR
    assert "timed out" in result.event_content


@pytest.mark.asyncio
async def test_no_timeout_when_fast():
    """Execution faster than timeout should succeed normally."""
    agent = _make_agent(runtime_delay=0.1, timeout=5.0)
    context = _ExecutionContext(max_steps=5)

    result = await agent._execute_code("fast code", context)

    assert result.event_type == EventType.EXECUTION_OUTPUT
    assert "OK" in result.event_content


@pytest.mark.asyncio
async def test_no_timeout_when_none():
    """No timeout configured should execute without limit."""
    agent = _make_agent(runtime_delay=0.1, timeout=None)
    context = _ExecutionContext(max_steps=5)

    result = await agent._execute_code("any code", context)

    assert result.event_type == EventType.EXECUTION_OUTPUT


@pytest.mark.asyncio
async def test_interrupt_called_on_timeout():
    """Runtime.interrupt() should be called when execution times out."""
    agent = _make_agent(runtime_delay=3.0, timeout=1.0)
    context = _ExecutionContext(max_steps=5)

    await agent._execute_code("slow code", context)

    assert agent.runtime.interrupted is True


@pytest.mark.asyncio
async def test_interrupt_not_called_on_success():
    """Runtime.interrupt() should not be called when execution succeeds."""
    agent = _make_agent(runtime_delay=0.1, timeout=5.0)
    context = _ExecutionContext(max_steps=5)

    await agent._execute_code("fast code", context)

    assert agent.runtime.interrupted is False


@pytest.mark.asyncio
async def test_timeout_message_includes_duration():
    """Timeout error message should include the configured timeout value."""
    agent = _make_agent(runtime_delay=3.0, timeout=2.0)
    context = _ExecutionContext(max_steps=5)

    result = await agent._execute_code("slow code", context)

    assert "2.0" in result.event_content or "2" in result.event_content


@pytest.mark.asyncio
async def test_timeout_prompt_guides_llm():
    """Timeout next_prompt should tell LLM to simplify."""
    agent = _make_agent(runtime_delay=3.0, timeout=1.0)
    context = _ExecutionContext(max_steps=5)

    result = await agent._execute_code("slow code", context)

    assert "simplify" in result.next_prompt.lower() or "smaller" in result.next_prompt.lower()


@pytest.mark.asyncio
async def test_system_prompt_includes_timeout():
    """System prompt should mention timeout when configured."""
    agent = _make_agent(runtime_delay=0, timeout=30.0)

    prompt = agent.build_system_prompt()

    assert "30" in prompt
    assert "timeout" in prompt.lower()


@pytest.mark.asyncio
async def test_system_prompt_no_timeout_when_none():
    """System prompt should not mention timeout when not configured."""
    agent = _make_agent(runtime_delay=0, timeout=None)

    prompt = agent.build_system_prompt()

    assert "timeout" not in prompt.lower()
