"""Tests for execution timeout — no real LLM or runtime needed."""

import asyncio

from cave_agent import IPythonRuntime
from cave_agent.agent import CaveAgent
from cave_agent.events import (
    ExecutionResultEvent,
    ExecutionTimeoutEvent,
    StopReason,
)
from cave_agent.runtime.executor import ExecutionResult, RuntimeStateLostError

from .fakes import FakeModel


class SlowRuntime:
    """Slow-execution fake satisfying PreemptibleRuntime.

    Defining ``interrupt`` is what makes it preemptible — do not delete it to
    "simplify", or every test here starts failing at construction.
    """

    def __init__(self, delay: float = 0, output: str | None = None):
        self._delay = delay
        self._output = output
        self.interrupted = False
        self.namespace: dict = {}

    async def execute(self, code: str) -> ExecutionResult:
        try:
            await asyncio.sleep(self._delay)
        except asyncio.CancelledError:
            # Runtime.execute owns request-scoped cleanup. The agent cannot send
            # an untargeted interrupt after this coroutine releases its request.
            await self.interrupt()
            raise
        return ExecutionResult(stdout=self._output or f"OK after {self._delay}s")

    async def interrupt(self) -> None:
        self.interrupted = True

    async def reset(self) -> None:
        self.namespace.clear()

    async def retrieve(self, name: str):
        return self.namespace[name]

    async def get_from_namespace(self, name: str):
        return self.namespace.get(name)

    def inject_into_namespace(self, name: str, value) -> None:
        self.namespace[name] = value

    async def bind_unique(self, prefix: str, value, *, start: int = 1) -> str:
        index = start
        while f"{prefix}_{index}" in self.namespace:
            index += 1
        name = f"{prefix}_{index}"
        self.namespace[name] = value
        return name

    def inject_function(self, function) -> None:
        self.namespace[function.name] = function.func

    def inject_variable(self, variable) -> None:
        self.namespace[variable.name] = variable.value

    def inject_type(self, type_obj) -> None:
        self.namespace[type_obj.name] = type_obj.value

    def inject_resources(
        self,
        *,
        functions=None,
        variables=None,
        types=None,
        bindings=None,
    ) -> None:
        additions = [
            *((function.name, function.func) for function in functions or []),
            *((variable.name, variable.value) for variable in variables or []),
            *((type_obj.name, type_obj.value) for type_obj in types or []),
            *(bindings or []),
        ]
        if len({name for name, _ in additions}) != len(additions):
            raise ValueError("duplicate resource name")
        if any(name in self.namespace for name, _ in additions):
            raise ValueError("resource name already exists")
        self.namespace.update(additions)

    def describe_functions(self) -> str:
        return "No functions"

    def describe_variables(self) -> str:
        return "No variables"

    def describe_types(self) -> str:
        return "No types"


class StateLosingRuntime(SlowRuntime):
    """Reports that cancellation terminated its backing process."""

    async def execute(self, code: str) -> ExecutionResult:
        try:
            return await super().execute(code)
        except asyncio.CancelledError as error:
            raise RuntimeStateLostError from error


def _make_agent(runtime_delay: float, timeout: float | None) -> CaveAgent:
    return CaveAgent(
        model=FakeModel(["```python\nimport time; time.sleep(999)\n```"]),
        runtime=SlowRuntime(delay=runtime_delay),
        max_steps=3,
        max_exec_timeout=timeout,
    )


async def test_timeout_triggers():
    """Execution slower than the timeout yields a timeout event."""
    agent = _make_agent(runtime_delay=3.0, timeout=1.0)

    event, prompt = await agent._execute("slow code")

    assert isinstance(event, ExecutionTimeoutEvent)
    assert event.timeout == 1.0
    assert "timed out" in prompt


async def test_no_timeout_when_fast():
    """Execution faster than the timeout succeeds normally."""
    agent = _make_agent(runtime_delay=0.1, timeout=5.0)

    event, prompt = await agent._execute("fast code")

    assert isinstance(event, ExecutionResultEvent)
    assert event.success
    assert "OK" in prompt


async def test_no_timeout_when_none():
    """No timeout configured executes without a limit."""
    agent = _make_agent(runtime_delay=0.1, timeout=None)

    event, _ = await agent._execute("any code")

    assert isinstance(event, ExecutionResultEvent)
    assert event.success


async def test_runtime_interrupts_its_owned_request_on_timeout():
    """The runtime, not the agent, interrupts the request it admitted."""
    agent = _make_agent(runtime_delay=3.0, timeout=1.0)

    await agent._execute("slow code")

    assert agent.runtime.interrupted is True


async def test_interrupt_not_called_on_success():
    """Runtime.interrupt() is not called when execution succeeds."""
    agent = _make_agent(runtime_delay=0.1, timeout=5.0)

    await agent._execute("fast code")

    assert agent.runtime.interrupted is False


async def test_timeout_message_includes_duration():
    """The prompt fed back to the model names the configured timeout."""
    agent = _make_agent(runtime_delay=3.0, timeout=2.0)

    _, prompt = await agent._execute("slow code")

    assert "2" in prompt


async def test_timeout_prompt_guides_llm():
    """The timeout prompt tells the model to simplify."""
    agent = _make_agent(runtime_delay=3.0, timeout=1.0)

    _, prompt = await agent._execute("slow code")

    assert "simplify" in prompt.lower() or "smaller" in prompt.lower()


async def test_timeout_reports_when_stopping_it_lost_runtime_state():
    agent = CaveAgent(
        model=FakeModel(),
        runtime=StateLosingRuntime(delay=1.0),
        max_exec_timeout=0.01,
    )

    event, prompt = await agent._execute("slow code")

    assert isinstance(event, ExecutionTimeoutEvent)
    assert event.state_lost
    assert "state created by earlier code" in prompt
    assert "lost" in prompt


async def test_system_prompt_includes_timeout():
    """System prompt mentions the timeout when one is configured."""
    agent = _make_agent(runtime_delay=0, timeout=30.0)

    prompt = agent.build_system_prompt()

    assert "30" in prompt
    assert "timeout" in prompt.lower()


async def test_system_prompt_no_timeout_when_none():
    """System prompt stays silent about timeouts when none is configured."""
    agent = _make_agent(runtime_delay=0, timeout=None)

    prompt = agent.build_system_prompt()

    assert "timeout" not in prompt.lower()


class TestPreemptionContract:
    """max_exec_timeout is only offered where it can actually be honoured.

    Abandoning a worker thread left IPythonExecutor's `capture_output()` — a
    process-global sys.stdout patch — unwound out of order, so every later
    print in the host process vanished into a dead buffer. The combination is
    now unconstructible rather than silently degraded.
    """

    def test_in_process_runtime_is_not_preemptible(self):
        from cave_agent import PreemptibleRuntime

        assert not isinstance(IPythonRuntime(), PreemptibleRuntime)

    def test_in_process_runtime_rejects_exec_timeout(self):
        import pytest

        with pytest.raises(ValueError, match="IPyKernelRuntime"):
            CaveAgent(model=FakeModel([]), runtime=IPythonRuntime(), max_exec_timeout=5)

    def test_default_runtime_rejects_exec_timeout(self):
        import pytest

        with pytest.raises(ValueError):
            CaveAgent(model=FakeModel([]), max_exec_timeout=5)

    async def test_stdout_survives_a_timed_out_execution(self):
        """The original defect, inverted: stdout identity must be unchanged."""
        import sys

        original = sys.stdout
        agent = _make_agent(runtime_delay=3.0, timeout=0.05)

        await agent._execute("slow code")
        await agent._execute("slow code")

        assert sys.stdout is original

    async def test_execution_runs_on_the_callers_loop(self):
        """The structural invariant. Offloading is what let a process-global
        patch unwind out of order; nothing may reintroduce it."""
        import asyncio
        import threading

        seen = {}

        class RecordingRuntime(SlowRuntime):
            async def execute(self, code):
                seen["loop"] = asyncio.get_running_loop()
                seen["thread"] = threading.current_thread()
                return await super().execute(code)

        agent = CaveAgent(
            model=FakeModel([]),
            runtime=RecordingRuntime(delay=0),
            max_exec_timeout=5,
        )
        await agent._execute("pass")

        assert seen["loop"] is asyncio.get_running_loop()
        assert seen["thread"] is threading.current_thread()

    async def test_backend_timeout_is_not_reported_as_execution_timeout(self):
        """A TimeoutError the backend raised itself must propagate — telling the
        model its code ran for max_exec_timeout seconds would be a lie."""
        import pytest

        class BackendTimesOut(SlowRuntime):
            async def execute(self, code):
                raise TimeoutError("kernel never answered on the shell channel")

        agent = CaveAgent(
            model=FakeModel([]),
            runtime=BackendTimesOut(),
            max_exec_timeout=30,
        )
        with pytest.raises(TimeoutError, match="shell channel"):
            await agent._execute("pass")


class TestPreemptibleRuntimeCarriesTheWholeProtocol:
    """The fake must stay a real implementation, not a partial one.

    `SlowRuntime` satisfies `PreemptibleRuntime` structurally, so a protocol
    method it implements with the wrong shape is invisible until the one path
    that calls it runs. `bind_unique` became async and this fake did not;
    nothing here produced oversize output, so no test ever awaited it, and a
    timeout-configured agent hit `TypeError` on its first large result.
    """

    async def test_oversize_output_persists_on_a_preemptible_runtime(self):
        runtime = SlowRuntime(output="x" * 9000)
        agent = CaveAgent(
            model=FakeModel(["```python\nprint('big')\n```", "done"]),
            runtime=runtime,
            max_exec_timeout=5,
            max_exec_output=500,
        )

        response = await agent.run("q")

        assert response.stop_reason is StopReason.COMPLETED
        assert runtime.namespace["_output_1"] == "x" * 9000

    def test_the_fake_implements_the_protocol_it_claims(self):
        """Structural conformance is checked by shape, not just by name."""
        import inspect

        from cave_agent.runtime import PreemptibleRuntime, Runtime

        runtime = SlowRuntime()
        assert isinstance(runtime, Runtime)
        assert isinstance(runtime, PreemptibleRuntime)
        for name in (
            "execute",
            "reset",
            "retrieve",
            "get_from_namespace",
            "bind_unique",
            "interrupt",
        ):
            assert inspect.iscoroutinefunction(
                getattr(runtime, name)
            ) is inspect.iscoroutinefunction(
                getattr(Runtime, name, None) or getattr(PreemptibleRuntime, name)
            ), name
