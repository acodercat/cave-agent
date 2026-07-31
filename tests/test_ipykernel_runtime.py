"""Tests for IPyKernelRuntime — process-isolated execution."""

import asyncio
import contextlib
from dataclasses import dataclass

import pytest
import pytest_asyncio

from cave_agent.runtime import (
    ErrorFeedbackMode,
    Function,
    IPyKernelRuntime,
    Type,
    Variable,
)
from cave_agent.security import FunctionRule, ImportRule, SecurityChecker


@pytest_asyncio.fixture(scope="module")
async def shared_runtime():
    """Single kernel shared across tests that don't need isolation.
    Avoids flaky failures caused by ZMQ socket churn between kernels.
    """
    rt = IPyKernelRuntime()
    await rt.start()
    yield rt
    await rt.stop()


@pytest_asyncio.fixture
async def runtime(shared_runtime):
    """Per-test fixture that resets state in the shared kernel."""
    await shared_runtime.execute("%reset -f")
    return shared_runtime


class TestKernelExecution:
    @pytest.mark.asyncio
    async def test_simple_print(self, runtime):
        result = await runtime.execute("print(1 + 1)")
        assert result.success
        assert "2" in result.stdout

    @pytest.mark.asyncio
    async def test_expression_result_captured(self, runtime):
        """Last expression repr appears in stdout."""
        result = await runtime.execute("1 + 1")
        assert result.success
        assert "2" in result.stdout

    @pytest.mark.asyncio
    async def test_state_persists_across_calls(self, runtime):
        await runtime.execute("x = 42")
        result = await runtime.execute("print(x)")
        assert result.success
        assert "42" in result.stdout

    @pytest.mark.asyncio
    async def test_imports_persist(self, runtime):
        await runtime.execute("import math")
        result = await runtime.execute("print(math.pi)")
        assert result.success
        assert "3.14" in result.stdout

    @pytest.mark.asyncio
    async def test_multiline_code(self, runtime):
        result = await runtime.execute("for i in range(3):\n    print(i)")
        assert result.success
        assert "0" in result.stdout
        assert "2" in result.stdout

    @pytest.mark.asyncio
    async def test_function_definition_and_call(self, runtime):
        await runtime.execute("def square(n): return n * n")
        result = await runtime.execute("print(square(7))")
        assert result.success
        assert "49" in result.stdout

    @pytest.mark.asyncio
    async def test_class_definition_and_use(self, runtime):
        await runtime.execute(
            "class Counter:\n  def __init__(self): self.n = 0\n  def inc(self): self.n += 1"
        )
        await runtime.execute("c = Counter(); c.inc(); c.inc()")
        result = await runtime.execute("print(c.n)")
        assert result.success
        assert "2" in result.stdout

    @pytest.mark.asyncio
    async def test_empty_output(self, runtime):
        result = await runtime.execute("x = 1")
        assert result.success
        assert result.stdout is None


class TestKernelErrors:
    @pytest.mark.asyncio
    async def test_zero_division(self, runtime):
        result = await runtime.execute("1 / 0")
        assert not result.success
        assert "ZeroDivisionError" in result.stdout

    @pytest.mark.asyncio
    async def test_name_error(self, runtime):
        result = await runtime.execute("print(undefined_var)")
        assert not result.success
        assert "NameError" in result.stdout

    @pytest.mark.asyncio
    async def test_syntax_error(self, runtime):
        result = await runtime.execute("def")
        assert not result.success
        assert "SyntaxError" in result.stdout

    @pytest.mark.asyncio
    async def test_type_error(self, runtime):
        result = await runtime.execute("'hello' + 42")
        assert not result.success
        assert "TypeError" in result.stdout

    @pytest.mark.asyncio
    async def test_error_does_not_break_state(self, runtime):
        """State survives after an error."""
        await runtime.execute("good_var = 'alive'")
        await runtime.execute("1 / 0")  # error
        result = await runtime.execute("print(good_var)")
        assert result.success
        assert "alive" in result.stdout

    @pytest.mark.asyncio
    async def test_error_feedback_mode_minimal(self):
        async with IPyKernelRuntime(error_feedback_mode=ErrorFeedbackMode.MINIMAL) as rt:
            result = await rt.execute("1 / 0")
            assert not result.success
            assert "ZeroDivisionError" in result.stdout
            assert "Traceback" not in result.stdout


class TestKernelInjection:
    @pytest.mark.asyncio
    async def test_inject_variable(self):
        async with IPyKernelRuntime(variables=[Variable("greeting", "hello")]) as rt:
            result = await rt.execute("print(greeting)")
            assert result.success
            assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_inject_numeric_variable(self):
        async with IPyKernelRuntime(variables=[Variable("pi", 3.14159)]) as rt:
            result = await rt.execute("print(pi * 2)")
            assert result.success
            assert "6.28" in result.stdout

    @pytest.mark.asyncio
    async def test_inject_list_variable(self):
        async with IPyKernelRuntime(variables=[Variable("items", [1, 2, 3])]) as rt:
            result = await rt.execute("print(sum(items))")
            assert result.success
            assert "6" in result.stdout

    @pytest.mark.asyncio
    async def test_inject_dict_variable(self):
        async with IPyKernelRuntime(variables=[Variable("config", {"key": "value"})]) as rt:
            result = await rt.execute("print(config['key'])")
            assert result.success
            assert "value" in result.stdout

    @pytest.mark.asyncio
    async def test_inject_function(self):
        def add(a: int, b: int) -> int:
            return a + b

        async with IPyKernelRuntime(functions=[Function(add)]) as rt:
            result = await rt.execute("print(add(3, 4))")
            assert result.success
            assert "7" in result.stdout

    @pytest.mark.asyncio
    async def test_inject_function_with_default_args(self):
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        async with IPyKernelRuntime(functions=[Function(greet)]) as rt:
            result = await rt.execute("print(greet('World'))")
            assert result.success
            assert "Hello, World!" in result.stdout

    @pytest.mark.asyncio
    async def test_one_missing_annotation_does_not_hide_resolvable_ones(self):
        namespace = {}
        exec(
            "from __future__ import annotations\n"
            "class Known:\n"
            "    pass\n"
            "def consume(value: Known, missing: Missing) -> Known:\n"
            "    return value\n",
            namespace,
        )
        async with IPyKernelRuntime(
            functions=[Function(namespace["consume"])],
        ) as rt:
            result = await rt.execute("created = consume(Known(), None)")

            assert result.success
            assert (await rt.get_from_namespace("Known")).__name__ == "Known"

    @pytest.mark.asyncio
    async def test_each_runtime_owns_its_variable_descriptor(self):
        shared = Variable("value", 1, "shared descriptor")
        first = IPyKernelRuntime(variables=[shared])
        second = IPyKernelRuntime(variables=[shared])
        try:
            first.update_variable("value", 2)
            await second.execute("pass")
            await second.reset()

            assert await first.retrieve("value") == 2
            assert await second.retrieve("value") == 1
            assert shared.value == 1
        finally:
            await first.stop()
            await second.stop()

    @pytest.mark.asyncio
    async def test_inject_type(self):
        class Color:
            def __init__(self, name: str):
                self.name = name

        async with IPyKernelRuntime(types=[Type(Color)]) as rt:
            result = await rt.execute("c = Color('red'); print(c.name)")
            assert result.success
            assert "red" in result.stdout

    @pytest.mark.asyncio
    async def test_inject_multiple_functions_and_variables(self):
        def double(n: int) -> int:
            return n * 2

        async with IPyKernelRuntime(
            functions=[Function(double)],
            variables=[Variable("base", 5)],
        ) as rt:
            result = await rt.execute("print(double(base))")
            assert result.success
            assert "10" in result.stdout

    @pytest.mark.asyncio
    async def test_inject_closure(self):
        """dill can serialize closures across to the kernel."""
        multiplier = 7

        def multiply(x: int) -> int:
            return x * multiplier

        async with IPyKernelRuntime(functions=[Function(multiply)]) as rt:
            result = await rt.execute("print(multiply(6))")
            assert result.success
            assert "42" in result.stdout

    @pytest.mark.asyncio
    async def test_inject_lambda(self):
        """dill can serialize lambdas across to the kernel."""
        square = lambda x: x * x  # noqa: E731

        async with IPyKernelRuntime(variables=[Variable("square", square)]) as rt:
            result = await rt.execute("print(square(9))")
            assert result.success
            assert "81" in result.stdout


class TestKernelRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_updated_variable(self):
        async with IPyKernelRuntime(variables=[Variable("val", 99)]) as rt:
            await rt.execute("val = val + 1")
            retrieved = await rt.retrieve("val")
            assert retrieved == 100

    @pytest.mark.asyncio
    async def test_retrieve_new_variable(self):
        async with IPyKernelRuntime(variables=[Variable("seed", 10)]) as rt:
            await rt.execute("seed = seed ** 2")
            retrieved = await rt.retrieve("seed")
            assert retrieved == 100

    @pytest.mark.asyncio
    async def test_retrieve_complex_object(self):
        async with IPyKernelRuntime(variables=[Variable("data", [1, 2, 3])]) as rt:
            await rt.execute("data = {k: k*2 for k in data}")
            retrieved = await rt.retrieve("data")
            assert retrieved == {1: 2, 2: 4, 3: 6}

    @pytest.mark.asyncio
    async def test_retrieve_unmanaged_variable_raises(self):
        async with IPyKernelRuntime() as rt:
            with pytest.raises(KeyError, match="not managed"):
                await rt.retrieve("nonexistent")


class TestKernelLifecycle:
    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with IPyKernelRuntime() as rt:
            result = await rt.execute("print('alive')")
            assert result.success
            assert "alive" in result.stdout

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        async with IPyKernelRuntime() as rt:
            await rt.execute("x = 123")
            await rt.reset()
            result = await rt.execute("print(x)")
            assert not result.success
            assert "NameError" in result.stdout

    @pytest.mark.asyncio
    async def test_reset_preserves_injections(self):
        async with IPyKernelRuntime(variables=[Variable("injected", "yes")]) as rt:
            await rt.reset()
            result = await rt.execute("print(injected)")
            assert result.success
            assert "yes" in result.stdout

    @pytest.mark.asyncio
    async def test_update_during_reset_wins_over_the_reset_snapshot(self):
        import asyncio

        rt = IPyKernelRuntime(variables=[Variable("version", 1)])
        await rt.start()
        executor = rt._executor
        started = asyncio.Event()
        resume = asyncio.Event()
        start_locked = executor._start_locked

        async def paused_start():
            await start_locked()
            started.set()
            await resume.wait()

        executor._start_locked = paused_start
        try:
            resetting = asyncio.create_task(rt.reset())
            await started.wait()
            rt.update_variable("version", 2)
            resume.set()
            await resetting

            assert await rt.retrieve("version") == 2
        finally:
            resume.set()
            await rt.stop()

    @pytest.mark.asyncio
    async def test_reset_clears_imports(self):
        async with IPyKernelRuntime() as rt:
            await rt.execute("import json")
            await rt.reset()
            result = await rt.execute("json.dumps({})")
            assert not result.success
            assert "NameError" in result.stdout

    @pytest.mark.asyncio
    async def test_multiple_executions_sequential(self, runtime):
        """Many sequential executions don't break the kernel."""
        for i in range(20):
            result = await runtime.execute(f"print({i})")
            assert result.success
            assert str(i) in result.stdout

    @pytest.mark.asyncio
    async def test_execute_before_start_starts_on_demand(self):
        """Executing without an explicit start() no longer raises — the kernel
        is started on first use. See TestOnDemandStart for the full contract."""
        rt = IPyKernelRuntime()
        try:
            result = await rt.execute("print(1)")
            assert result.success
        finally:
            await rt.stop()


class TestKernelSecurity:
    @pytest.mark.asyncio
    async def test_import_rule_blocks(self):
        checker = SecurityChecker(rules=[ImportRule({"os", "subprocess"})])
        async with IPyKernelRuntime(security_checker=checker) as rt:
            result = await rt.execute("import os")
            assert not result.success

            result = await rt.execute("import subprocess")
            assert not result.success

    @pytest.mark.asyncio
    async def test_function_rule_blocks(self):
        checker = SecurityChecker(rules=[FunctionRule({"eval", "exec"})])
        async with IPyKernelRuntime(security_checker=checker) as rt:
            result = await rt.execute("eval('1+1')")
            assert not result.success

    @pytest.mark.asyncio
    async def test_allowed_code_passes_security(self):
        checker = SecurityChecker(rules=[ImportRule({"os"})])
        async with IPyKernelRuntime(security_checker=checker) as rt:
            result = await rt.execute("import math; print(math.sqrt(4))")
            assert result.success
            assert "2.0" in result.stdout


class TestKernelDescribe:
    def test_describe_functions(self):
        def my_func(a: int) -> str:
            """Does stuff."""
            return str(a)

        rt = IPyKernelRuntime(functions=[Function(my_func)])
        desc = rt.describe_functions()
        assert "my_func" in desc

    def test_describe_variables(self):
        rt = IPyKernelRuntime(variables=[Variable("x", 42, "the answer")])
        desc = rt.describe_variables()
        assert "x" in desc
        assert "the answer" in desc

    def test_describe_types(self):
        @dataclass
        class Foo:
            bar: int

        rt = IPyKernelRuntime(types=[Type(Foo, include_schema=True)])
        desc = rt.describe_types()
        assert "Foo" in desc
        assert "bar" in desc

    def test_describe_no_functions(self):
        rt = IPyKernelRuntime()
        assert rt.describe_functions() == "No functions available"

    def test_describe_no_variables(self):
        rt = IPyKernelRuntime()
        assert rt.describe_variables() == "No variables available"


class TestOnDemandStart:
    """The subprocess appears on first execution, not on construction.

    This replaces a separate LazyRuntime wrapper: the deferral was always
    native to this class — only the auto-start was missing — so a 230-line
    decorator was not buying anything the executor couldn't do in four lines.
    """

    @staticmethod
    def _count_starts(rt) -> "list[int]":
        """Count kernels *this runtime* brings up, as a one-element list.

        The predecessor shelled out to ``pgrep -fc ipykernel_launcher``, which
        counts every process whose *command line contains that string* — the
        invoking shell included — not kernels, and not kernels this test owns.
        It measured a machine-wide global that any other process could move,
        and it did: a full-suite run failed an assertion that had passed
        moments earlier standalone.

        Wrapping the executor's own startup measures the thing the tests are
        actually about, and is immune to whatever else the machine is doing.
        """
        started = [0]
        executor = rt._executor
        real = executor._start_locked

        async def counting():
            started[0] += 1
            await real()

        executor._start_locked = counting
        return started

    def test_construction_starts_nothing(self):
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        rt = IPyKernelRuntime(functions=[Function(add)])
        started = self._count_starts(rt)
        rt.inject_into_namespace("preset", 41)

        assert started == [0]
        assert rt._executor._started is False

    def test_prompt_renders_without_starting(self):
        """The system prompt is built before the model decides anything."""

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        rt = IPyKernelRuntime(
            functions=[Function(add)],
            variables=[Variable("cfg", {"a": 1}, "config")],
        )

        assert "add" in rt.describe_functions()
        assert "cfg" in rt.describe_variables()
        assert rt._executor._started is False

    @pytest.mark.asyncio
    async def test_first_execute_starts_the_kernel(self):
        rt = IPyKernelRuntime()
        try:
            result = await rt.execute("print('started on demand')")
            assert result.success
            assert "started on demand" in result.stdout
            assert rt._executor._started is True
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_registrations_survive_on_demand_start(self):
        def double(x: int) -> int:
            """Double a number."""
            return x * 2

        rt = IPyKernelRuntime(functions=[Function(double)])
        rt.inject_into_namespace("preset", 21)
        try:
            result = await rt.execute("print(double(preset))")
            assert result.success
            assert "42" in result.stdout
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_concurrent_explicit_starts_collapse(self):
        """Two direct start() calls must not race.

        The flag check alone is not atomic: both got past it and raced inside
        jupyter_client, which set its readiness future twice and raised
        InvalidStateError. The lock therefore lives in start(), not at the
        on-demand call sites.
        """
        import asyncio

        rt = IPyKernelRuntime()
        started = self._count_starts(rt)
        try:
            await asyncio.gather(*(rt.start() for _ in range(4)))
            assert started == [1]
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_explicit_start_races_on_demand_execute(self):
        import asyncio

        rt = IPyKernelRuntime()
        started = self._count_starts(rt)
        try:
            await asyncio.gather(rt.start(), rt.execute("pass"), rt.start())
            assert started == [1]
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_explicit_start_is_idempotent(self):
        """An explicit start() plus a later execution must not spawn two."""
        rt = IPyKernelRuntime()
        started = self._count_starts(rt)
        try:
            await rt.start()
            await rt.start()
            await rt.execute("pass")
            assert started == [1]
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_concurrent_first_calls_start_once(self):
        import asyncio

        rt = IPyKernelRuntime()
        started = self._count_starts(rt)
        try:
            await asyncio.gather(*(rt.execute("pass") for _ in range(6)))
            assert started == [1]
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_concurrent_calls_do_not_interleave(self):
        """Each concurrent request must get its own output back.

        A kernel serves one cell at a time over a single IOPub queue, so
        overlapping requests can otherwise consume each other's messages.

        Covers steady-state serialization only. It does NOT cover the narrower
        startup window (a task arriving after `_started` flips but before
        `start()` returns): under `gather` every task passes the lock-free fast
        path before the flag is set, so they all queue on `_lifecycle_lock` and
        never enter that window — verified by reverting the fix, which this
        test still passes. That ordering is guarded by the comment in `start()`,
        not by a test.
        """
        import asyncio

        rt = IPyKernelRuntime()
        try:
            results = await asyncio.gather(*(rt.execute(f"print('marker-{i}')") for i in range(8)))
            got = sorted((r.stdout or "").strip() for r in results)
            assert got == sorted(f"marker-{i}" for i in range(8)), got
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_reset_without_start_is_a_noop(self):
        """Resetting an unstarted kernel must not spawn one to clear it."""
        rt = IPyKernelRuntime()
        started = self._count_starts(rt)

        await rt.reset()

        assert started == [0]
        assert rt._executor._started is False


class TestAgentDefersKernel:
    @pytest.mark.asyncio
    async def test_pure_chat_run_spawns_no_kernel(self):
        import subprocess

        from cave_agent import CaveAgent

        from .fakes import FakeModel

        def count() -> int:
            out = subprocess.run(
                ["pgrep", "-fc", "ipykernel_launcher"],
                capture_output=True,
                text=True,
            )
            return int(out.stdout.strip() or 0)

        rt = IPyKernelRuntime()
        try:
            before = count()
            agent = CaveAgent(model=FakeModel(["just an answer"]), runtime=rt)
            await agent.run("hello")
            assert count() == before
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_skills_register_without_starting(self):
        """Enabling skills injects `activate_skill` at agent construction —
        long before any code runs — so it must not force a kernel."""
        from cave_agent import CaveAgent, Skill

        from .fakes import FakeModel

        rt = IPyKernelRuntime()
        try:
            agent = CaveAgent(
                model=FakeModel(["answer"]),
                runtime=rt,
                skills=[Skill(name="demo", description="A demo", body_content="steps")],
            )
            assert "activate_skill" in agent.build_system_prompt()
            assert rt._executor._started is False
        finally:
            await rt.stop()


class TestGracefulShutdown:
    """stop() releases the kernel without destroying work already in flight.

    Graceful-by-default is the Python convention for releasing a resource
    (`Executor.shutdown` waits, `asyncio.Server.wait_closed` waits, and
    `KernelManager.shutdown_kernel` defaults to `now=False`). Immediate
    shutdown composes from the existing primitive: interrupt() then stop().
    """

    @pytest.mark.asyncio
    async def test_stop_waits_for_inflight_request(self):
        import asyncio

        rt = IPyKernelRuntime()
        await rt.start()
        task = asyncio.create_task(rt.execute("import time; time.sleep(2); print('finished')"))
        await asyncio.sleep(0.5)

        await rt.stop()

        result = await task
        assert result.success
        assert "finished" in result.stdout

    @pytest.mark.asyncio
    async def test_interrupt_then_stop_is_immediate(self):
        import asyncio
        import time

        rt = IPyKernelRuntime()
        await rt.start()
        task = asyncio.create_task(rt.execute("import time; time.sleep(30); print('never')"))
        await asyncio.sleep(0.5)

        started = time.monotonic()
        await rt.interrupt()
        await rt.stop()
        elapsed = time.monotonic() - started

        assert elapsed < 5, f"interrupt+stop should not wait out the cell, took {elapsed:.1f}s"
        result = await task
        assert not result.success

    @pytest.mark.asyncio
    async def test_stop_without_start_is_a_noop(self):
        rt = IPyKernelRuntime()
        await rt.stop()
        assert rt._executor._started is False

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self):
        rt = IPyKernelRuntime()
        await rt.execute("pass")
        await rt.stop()
        await rt.stop()
        assert rt._executor._started is False

    @pytest.mark.asyncio
    async def test_channel_lock_released_after_stop(self):
        """A leaked lock would wedge every later request."""
        rt = IPyKernelRuntime()
        await rt.execute("pass")
        await rt.stop()
        assert not rt._executor._channel_lock.locked()


class TestChannelReclaim:
    """An abandoned request must stop the cell before the channel is reused.

    A timeout or cancellation stops us waiting; the kernel never hears about it.
    Releasing the channel then let the next request queue behind the still-running
    cell and time out too — one abandoned cell caused a cascade.
    """

    @pytest.mark.asyncio
    async def test_next_request_is_not_blocked_after_a_timeout(self):
        import time

        rt = IPyKernelRuntime(iopub_timeout=1.0)
        try:
            slow = await rt.execute("import time; time.sleep(20)")
            assert not slow.success

            started = time.monotonic()
            probe = await rt.execute("print('probe')")
            elapsed = time.monotonic() - started

            assert probe.success, probe.stdout
            assert "probe" in probe.stdout
            # The real assertion: had the cell merely been waited out rather
            # than stopped, this would take the remaining ~19s.
            assert elapsed < 5, f"probe took {elapsed:.1f}s — the cell kept running"
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_next_request_is_not_blocked_after_cancellation(self):
        import asyncio
        import time

        rt = IPyKernelRuntime()
        await rt.start()
        try:
            task = asyncio.create_task(rt.execute("import time; time.sleep(20)"))
            await asyncio.sleep(0.5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            started = time.monotonic()
            probe = await asyncio.wait_for(rt.execute("print('probe')"), timeout=15)
            elapsed = time.monotonic() - started

            assert probe.success and "probe" in probe.stdout
            assert elapsed < 5, f"probe took {elapsed:.1f}s — the cell kept running"
        finally:
            await rt.stop()

    async def test_repeated_cancellation_cannot_abort_channel_reclaim(self):
        rt = IPyKernelRuntime(iopub_timeout=5.0)
        code = "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(5)"
        try:
            task = asyncio.create_task(rt.execute(code))
            await asyncio.sleep(0.25)
            task.cancel()
            await asyncio.sleep(0.05)
            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task
            # A cancelled task retains the abandoned request's channel frames.
            del task

            probe = await asyncio.wait_for(
                rt.execute("print('probe')"),
                timeout=2,
            )
            assert probe.success
            assert probe.stdout == "probe\n"
        finally:
            await rt.stop()

    async def test_timeout_interrupt_cannot_escape_to_the_next_request(self):
        """The agent must not interrupt after the backend releases the channel."""
        from cave_agent import CaveAgent
        from cave_agent.events import ExecutionTimeoutEvent

        from .fakes import FakeModel

        rt = IPyKernelRuntime(iopub_timeout=5.0)
        try:
            await rt.start()
            manager = rt._executor._km
            is_alive = manager.is_alive
            calls = 0

            async def delay_first_health_check(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    await asyncio.sleep(0.6)
                return await is_alive(*args, **kwargs)

            manager.is_alive = delay_first_health_check
            agent = CaveAgent(FakeModel(), runtime=rt, max_exec_timeout=0.1)
            timed_out = asyncio.create_task(agent._execute("import time; time.sleep(10)"))
            await asyncio.sleep(0.05)
            queued = asyncio.create_task(
                rt.execute(
                    "import time\n"
                    "print('B_STARTED', flush=True)\n"
                    "try:\n"
                    "    time.sleep(1)\n"
                    "    print('B_FINISHED')\n"
                    "except KeyboardInterrupt:\n"
                    "    print('B_INTERRUPTED')"
                )
            )

            event, _ = await asyncio.wait_for(timed_out, timeout=5)
            second = await asyncio.wait_for(queued, timeout=5)

            assert isinstance(event, ExecutionTimeoutEvent)
            assert second.success
            assert second.stdout == "B_STARTED\nB_FINISHED\n"
        finally:
            await rt.stop()

    @pytest.mark.asyncio
    async def test_channel_lock_is_released_after_abandon(self):
        rt = IPyKernelRuntime(iopub_timeout=1.0)
        try:
            await rt.execute("import time; time.sleep(10)")
            assert not rt._executor._channel_lock.locked()
        finally:
            await rt.stop()

    async def test_a_cell_that_ignores_interrupt_is_terminated(self):
        import time

        from cave_agent import CaveAgent
        from cave_agent.events import ExecutionTimeoutEvent

        from .fakes import FakeModel

        rt = IPyKernelRuntime(variables=[Variable("kept", 42, "value")])
        agent = CaveAgent(
            FakeModel([]),
            runtime=rt,
            max_exec_timeout=0.15,
        )
        code = "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(5)"
        try:
            await rt.execute("ephemeral = 'created by model code'")
            started = time.monotonic()
            timed_out = asyncio.create_task(agent._execute(code))
            await asyncio.sleep(0.2)
            queued_probe = asyncio.create_task(rt.execute("print(kept)"))

            event, prompt = await timed_out
            elapsed = time.monotonic() - started
            probe = await asyncio.wait_for(queued_probe, timeout=5)
            lost = await rt.execute("print(ephemeral)")

            assert isinstance(event, ExecutionTimeoutEvent)
            assert event.state_lost
            assert "state created by earlier code" in prompt
            assert elapsed < 3, f"uncooperative cell ran for {elapsed:.1f}s"
            assert probe.stdout == "42\n"
            assert not lost.success
            assert "NameError" in lost.stdout
        finally:
            await rt.stop()

    async def test_iopub_recovery_reports_that_it_lost_runtime_state(self):
        rt = IPyKernelRuntime(iopub_timeout=0.1)
        code = (
            "import signal, time\n"
            "ephemeral = 'created by model code'\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "time.sleep(5)"
        )
        try:
            result = await rt.execute(code)
            lost = await rt.execute("print(ephemeral)")

            assert result.state_lost
            assert isinstance(result.error, TimeoutError)
            assert not lost.success
            assert "NameError" in lost.stdout
        finally:
            await rt.stop()


class TestLifecycleIsAtomic:
    """A lifecycle transition is invisible to requests queued behind it.

    `reset()` used to take the channel lock twice — once to stop, once to
    start — and a queued `execute` slipped into the gap, where it found the
    `None` client that briefly sits between the two kernels. Two concurrent
    `reset()`s raced on one manager and killed the kernel outright.
    """

    async def test_queued_execute_survives_a_reset(self):
        import asyncio

        rt = IPyKernelRuntime()
        try:
            await rt.execute("x = 1")
            slow = asyncio.create_task(rt.execute("import time; time.sleep(2)"))
            await asyncio.sleep(0.3)
            resetting = asyncio.create_task(rt.reset())
            await asyncio.sleep(0.1)
            queued = asyncio.create_task(rt.execute("print('queued')"))

            _, _, result = await asyncio.gather(slow, resetting, queued)

            assert result.stdout == "queued\n"
            assert result.success
        finally:
            await rt.stop()

    async def test_concurrent_resets_leave_one_live_kernel(self):
        import asyncio

        rt = IPyKernelRuntime()
        try:
            await rt.execute("x = 1")

            await asyncio.gather(rt.reset(), rt.reset())

            result = await rt.execute("print('alive')")
            assert result.stdout == "alive\n"
        finally:
            await rt.stop()

    async def test_a_stopped_runtime_can_start_again(self):
        """A shut-down manager cannot be restarted — its ZMQ context is closed,
        and `start_kernel` on it hangs rather than failing. Teardown leaves a
        fresh one so the next start works from any stopped state."""
        import asyncio

        rt = IPyKernelRuntime()
        try:
            await rt.start()
            await rt.stop()

            result = await asyncio.wait_for(rt.execute("print('restarted')"), timeout=30)
            assert result.stdout == "restarted\n"
        finally:
            await rt.stop()


class TestChannelBacklog:
    """Every request consumes both of its channels.

    `execute` drained IOPub but never read the matching `execute_reply`, so the
    shell queue grew without bound — after 1,200 executions the oldest unread
    reply was still execution 1, and every `_shell_reply_for` had to walk the
    whole backlog to reach its own.
    """

    async def _unread_shell_replies(self, rt) -> int:
        client = rt._executor._kernel_client
        unread = 0
        while True:
            try:
                await client.get_shell_msg(timeout=0.3)
            except Exception:
                return unread
            unread += 1

    async def test_executions_leave_no_unread_replies(self):
        rt = IPyKernelRuntime()
        try:
            for i in range(20):
                await rt.execute(f"print({i})")
            assert await self._unread_shell_replies(rt) == 0
        finally:
            await rt.stop()

    async def test_namespace_reads_and_injections_leave_none_either(self):
        rt = IPyKernelRuntime(variables=[Variable("v", 41, "a value")])
        try:
            await rt.execute("w = v + 1")
            assert await rt.get_from_namespace("w") == 42
            assert await self._unread_shell_replies(rt) == 0
        finally:
            await rt.stop()


class TestStartFailureCleanup:
    """A failed start releases everything it managed to create.

    Cleanup ran through the public `stop()`, which takes the lifecycle lock the
    caller already holds — `asyncio.Lock` is not reentrant, so it deadlocked:
    the original failure was never raised, and when the kernel had already come
    up it stayed alive with no handle left to reach it.
    """

    async def test_failure_before_the_kernel_exists_propagates(self):
        import asyncio

        rt = IPyKernelRuntime()

        async def refuse(*args, **kwargs):
            raise RuntimeError("kernel would not start")

        rt._executor._km.start_kernel = refuse

        with pytest.raises(RuntimeError, match="would not start"):
            await asyncio.wait_for(rt.start(), timeout=20)

    async def test_failure_after_the_kernel_is_live_leaves_nothing_running(self):
        """The costly case: setup fails once a real subprocess exists."""
        import asyncio

        rt = IPyKernelRuntime()
        executor = rt._executor

        async def refuse(code):
            raise RuntimeError("setup failed")

        executor._execute_silent = refuse

        with pytest.raises(RuntimeError, match="setup failed"):
            await asyncio.wait_for(rt.start(), timeout=30)

        assert not await executor._km.is_alive()
        assert executor._started is False

    async def test_silent_setup_timeout_is_a_typed_runtime_failure(self):
        """Channel loss during start is runtime failure, never model failure."""
        import asyncio

        from cave_agent.runtime.executor import RuntimeExecutionError

        rt = IPyKernelRuntime()
        executor = rt._executor

        async def timeout(*args, **kwargs):
            raise TimeoutError

        executor._shell_reply_for = timeout

        with pytest.raises(
            RuntimeExecutionError,
            match="timed out during a silent setup request",
        ):
            await asyncio.wait_for(rt.start(), timeout=30)

        assert not await executor._km.is_alive()
        assert executor._started is False

    async def test_readiness_timeout_is_a_typed_runtime_failure(self):
        """jupyter_client reports its readiness deadline as RuntimeError."""
        from cave_agent.runtime.executor import RuntimeExecutionError

        rt = IPyKernelRuntime()
        executor = rt._executor

        class UnreadyClient:
            def start_channels(self):
                pass

            def stop_channels(self):
                pass

            async def wait_for_ready(self, timeout):
                raise RuntimeError(f"Kernel didn't respond in {int(timeout)} seconds")

        executor._km.client = UnreadyClient

        with pytest.raises(
            RuntimeExecutionError,
            match="failed its readiness check",
        ):
            await asyncio.wait_for(rt.start(), timeout=30)

        assert not await executor._km.is_alive()
        assert executor._started is False

    async def test_the_runtime_is_still_usable_after_a_failed_start(self):
        """Cleanup must leave a manager the next attempt can actually start."""
        import asyncio

        rt = IPyKernelRuntime()
        executor = rt._executor
        original = executor._execute_silent
        calls = {"n": 0}

        async def fail_once(code):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("setup failed")
            return await original(code)

        executor._execute_silent = fail_once
        try:
            with pytest.raises(RuntimeError):
                await asyncio.wait_for(rt.start(), timeout=30)

            result = await asyncio.wait_for(rt.execute("print('recovered')"), timeout=30)
            assert result.stdout == "recovered\n"
        finally:
            await rt.stop()

    async def test_repeated_cancellation_cannot_abort_startup_cleanup(self):
        rt = IPyKernelRuntime()
        executor = rt._executor
        manager = executor._km
        setup_started = asyncio.Event()
        terminating = asyncio.Event()
        terminate = executor._terminate_kernel

        async def wait_during_setup(code):
            setup_started.set()
            await asyncio.Event().wait()

        async def slow_terminate():
            terminating.set()
            await asyncio.sleep(0.2)
            return await terminate()

        executor._execute_silent = wait_during_setup
        executor._terminate_kernel = slow_terminate
        starting = asyncio.create_task(rt.start())
        await setup_started.wait()
        starting.cancel()
        await terminating.wait()
        starting.cancel()

        with pytest.raises(asyncio.CancelledError):
            await starting
        assert not await manager.is_alive()
        assert executor._started is False
        assert executor._kernel_client is None


class TestTimeoutLatency:
    """A timed-out execution surfaces at its own timeout, not a later one.

    `execute` reaped the shell reply after `_collect_result` had already
    abandoned the request — which reaps too — so an IOPub timeout waited out a
    second, futile `_SHELL_REPLY_TIMEOUT`. A 0.2s timeout took 10.2s to return.
    """

    async def test_iopub_timeout_returns_promptly(self):
        import time

        from cave_agent.runtime.ipykernel_executor import _SHELL_REPLY_TIMEOUT

        rt = IPyKernelRuntime(iopub_timeout=3)
        try:
            started = time.monotonic()
            result = await rt.execute("import time; time.sleep(20)")
            elapsed = time.monotonic() - started

            assert isinstance(result.error, TimeoutError)
            # The real assertion: nowhere near a second timeout window.
            assert elapsed < _SHELL_REPLY_TIMEOUT / 2, f"took {elapsed:.2f}s"
        finally:
            await rt.stop()

    async def test_the_kernel_is_still_usable_afterwards(self):
        rt = IPyKernelRuntime(iopub_timeout=3)
        try:
            await rt.execute("import time; time.sleep(20)")
            result = await rt.execute("print('after')")
            assert result.stdout == "after\n"
        finally:
            await rt.stop()


class TestAdmissionIsAtomic:
    """ "Ensure started" and "hold the channel" are one step, not two.

    `start()` has a lock-free fast path, so a request could pass the flag
    check, queue behind a `stop()` that had already begun, and wake up holding
    the channel with the client set to None.
    """

    async def _queued_call_during_stop(self, call):
        import asyncio

        rt = IPyKernelRuntime()
        await rt.execute("x = 1")
        slow = asyncio.create_task(rt.execute("import time; time.sleep(2)"))
        await asyncio.sleep(0.3)
        stopping = asyncio.create_task(rt.stop())
        await asyncio.sleep(0.1)
        queued = asyncio.create_task(call(rt))
        try:
            return await asyncio.gather(slow, stopping, queued, return_exceptions=True)
        finally:
            await rt.stop()

    async def test_execute_queued_behind_a_stop_succeeds(self):
        results = await self._queued_call_during_stop(lambda rt: rt.execute("print('queued')"))

        errors = [r for r in results if isinstance(r, BaseException)]
        assert not errors, errors
        assert results[-1].stdout == "queued\n"

    async def test_namespace_read_queued_behind_a_stop_succeeds(self):
        async def read(rt):
            await rt.execute("marker = 'set'")
            return await rt.get_from_namespace("marker")

        results = await self._queued_call_during_stop(read)

        errors = [r for r in results if isinstance(r, BaseException)]
        assert not errors, errors


class TestInjectionDuringFlush:
    """`inject_*` is synchronous and public, so it can land mid-flush.

    The flush iterated the live dict across an await, so a registration made
    while it was in flight killed the execution that happened to be carrying
    it — with `dictionary changed size during iteration`, which reads as a
    defect in the model's generated code.
    """

    async def test_a_concurrent_injection_does_not_break_the_execution(self):
        import asyncio

        rt = IPyKernelRuntime()
        try:
            await rt.start()
            executor = rt._executor
            original = executor._execute_silent
            gate = asyncio.Event()

            async def pause_first(code):
                await gate.wait()
                return await original(code)

            executor._execute_silent = pause_first
            rt.inject_into_namespace("a", 1)
            running = asyncio.create_task(rt.execute("pass"))
            await asyncio.sleep(0.3)
            rt.inject_into_namespace("b", 2)
            gate.set()

            result = await running
            assert result.success
        finally:
            await rt.stop()

    async def test_the_later_injection_still_arrives(self):
        """Queued for the next flush, not dropped."""
        import asyncio

        rt = IPyKernelRuntime()
        try:
            await rt.start()
            executor = rt._executor
            original = executor._execute_silent
            gate = asyncio.Event()

            async def pause_first(code):
                await gate.wait()
                return await original(code)

            executor._execute_silent = pause_first
            rt.inject_into_namespace("a", 1)
            running = asyncio.create_task(rt.execute("pass"))
            await asyncio.sleep(0.3)
            rt.inject_into_namespace("b", 2)
            gate.set()
            await running

            executor._execute_silent = original
            assert await rt.get_from_namespace("b") == 2
        finally:
            await rt.stop()

    async def test_cancellation_does_not_replay_an_inflight_injection(
        self,
        tmp_path,
    ):
        """A sent injection completes once before cancellation is re-raised."""
        import os
        import shlex

        marker = tmp_path / "restores.txt"

        class SlowRestore:
            def __reduce__(self):
                path = shlex.quote(str(marker))
                return os.system, (f"printf 'restore\\n' >> {path}; sleep 1",)

        rt = IPyKernelRuntime()
        try:
            await rt.start()
            rt.inject_into_namespace("slow", SlowRestore())
            task = asyncio.create_task(rt.execute("print('never runs')"))
            for _ in range(200):
                if marker.exists():
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("the injection never started")

            task.cancel()
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            probe = await asyncio.wait_for(
                rt.execute("print('probe')"),
                timeout=3,
            )

            assert probe.stdout == "probe\n"
            assert marker.read_text().splitlines() == ["restore"]
        finally:
            await rt.stop()


class TestInternalRequestOwnership:
    async def test_cancelled_namespace_read_finishes_before_reusing_channel(
        self,
        tmp_path,
    ):
        marker = tmp_path / "read-started"
        rt = IPyKernelRuntime(iopub_timeout=3)
        try:
            setup = await rt.execute(
                "import time\n"
                "class SlowPickle:\n"
                "    def __reduce__(self):\n"
                f"        with open({str(marker)!r}, 'w') as stream:\n"
                "            stream.write('started')\n"
                "        time.sleep(1)\n"
                "        return str, ('value',)\n"
                "slow = SlowPickle()\n"
                "count = 0"
            )
            assert setup.success

            reading = asyncio.create_task(rt.get_from_namespace("slow"))
            for _ in range(200):
                if marker.exists():
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("the namespace read never reached the kernel")

            reading.cancel()
            await asyncio.sleep(0.02)
            reading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(reading, timeout=3)

            result = await rt.execute("count += 1; print(count)")
            assert result.success
            assert result.stdout == "1\n"
        finally:
            await rt.stop()

    async def test_cancelled_unique_bind_commits_before_reusing_channel(
        self,
        tmp_path,
    ):
        import os
        import shlex

        marker = tmp_path / "bind-started"

        class SlowRestore:
            def __reduce__(self):
                path = shlex.quote(str(marker))
                return os.system, (f"printf started > {path}; sleep 1",)

        rt = IPyKernelRuntime(iopub_timeout=3)
        try:
            binding = asyncio.create_task(
                rt.bind_unique("_held", SlowRestore()),
            )
            for _ in range(200):
                if marker.exists():
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("the unique binding never reached the kernel")

            binding.cancel()
            await asyncio.sleep(0.02)
            binding.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(binding, timeout=3)

            assert "_held_1" in rt._namespace
            assert await rt.get_from_namespace("_held_1") == 0
            probe = await rt.execute("print('probe')")
            assert probe.success
            assert probe.stdout == "probe\n"
        finally:
            await rt.stop()

    async def test_timed_out_injection_is_not_replayed(self, tmp_path):
        marker = tmp_path / "restore-count"

        def slow_restore(path):
            import signal
            import time

            with open(path, "a") as stream:
                stream.write("restore\n")
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            time.sleep(2)
            return "value"

        class SlowRestore:
            def __reduce__(self):
                return slow_restore, (str(marker),)

        from cave_agent.runtime.executor import RuntimeExecutionError

        rt = IPyKernelRuntime(iopub_timeout=0.1)
        try:
            rt.inject_into_namespace("slow", SlowRestore())
            with pytest.raises(RuntimeExecutionError):
                await rt.execute("pass")

            with pytest.raises(
                RuntimeExecutionError,
                match="Persistent kernel bindings failed",
            ):
                await rt.execute("pass")
            assert rt._executor._started is False
            assert marker.read_text().splitlines() == ["restore"]

            rt.inject_into_namespace("slow", "replacement")
            result = await rt.execute("print(slow)")
            assert result.success
            assert result.stdout == "replacement\n"
            assert marker.read_text().splitlines() == ["restore"]
        finally:
            await rt.stop()

    async def test_reset_repairs_failed_injection_after_kernel_stops(self, tmp_path):
        marker = tmp_path / "failed-once"

        def restore_after_one_failure(path):
            import os
            import signal
            import time

            if not os.path.exists(path):
                with open(path, "w") as stream:
                    stream.write("failed")
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                time.sleep(2)
            return "restored"

        class FailsOnce:
            def __reduce__(self):
                return restore_after_one_failure, (str(marker),)

        from cave_agent.runtime.executor import RuntimeExecutionError

        rt = IPyKernelRuntime(iopub_timeout=0.1)
        try:
            rt.inject_into_namespace("fragile", FailsOnce())
            with pytest.raises(RuntimeExecutionError):
                await rt.execute("pass")

            assert rt._executor._started is False
            assert rt._executor._failed_injections == {"fragile"}

            await rt.reset()

            assert rt._executor._started is False
            assert not rt._executor._failed_injections
            result = await rt.execute("print(fragile)")
            assert result.success
            assert result.stdout == "restored\n"
        finally:
            await rt.stop()


class TestStopReleasesTheProcess:
    """Channel and process cleanup are independent.

    One `try` around both meant a `stop_channels()` failure skipped the
    shutdown entirely — and `stop()` still returned successfully having
    replaced the manager, so the subprocess stayed alive with no handle left.
    """

    async def test_a_channel_failure_does_not_strand_the_kernel(self):
        rt = IPyKernelRuntime()
        await rt.start()
        executor = rt._executor
        manager = executor._km

        def refuse():
            raise RuntimeError("stop_channels failed")

        executor._kernel_client.stop_channels = refuse

        await rt.stop()

        assert not await manager.is_alive()
        assert executor._started is False


class TestOrphanedKernelIsRefusedNotReplaced:
    """A kernel that survives its shutdown keeps its manager, and blocks starts.

    `_started` went false on a *failed* teardown too, so the next start called
    `start_kernel()` on the manager still holding the live process: Jupyter
    spawned a replacement that died on the occupied ports, execution silently
    reconnected to the old kernel, and the original handle was lost. The log
    claimed no new kernel would start; nothing enforced it.
    """

    def _refuse_shutdown(self, rt, times):
        """Make shutdown fail *times* times, then work again."""
        manager = rt._executor._km
        real = manager.shutdown_kernel
        remaining = {"n": times}

        async def maybe_fail(*args, **kwargs):
            if remaining["n"] > 0:
                remaining["n"] -= 1
                raise RuntimeError("transport failure")
            return await real(*args, **kwargs)

        manager.shutdown_kernel = maybe_fail
        return manager

    async def test_a_single_transport_failure_is_retried(self):
        """One hiccup is common and recoverable; teardown should not give up."""
        from cave_agent.runtime.ipykernel_executor import _SHUTDOWN_ATTEMPTS

        rt = IPyKernelRuntime()
        await rt.start()
        manager = self._refuse_shutdown(rt, times=_SHUTDOWN_ATTEMPTS - 1)

        await rt.stop()

        assert not await manager.is_alive()
        assert rt._executor._orphaned is False

    async def test_stop_reports_a_kernel_it_could_not_kill(self):
        """Returning success here let a caller walk away from a live process."""
        from cave_agent.runtime.executor import RuntimeExecutionError
        from cave_agent.runtime.ipykernel_executor import _SHUTDOWN_ATTEMPTS

        rt = IPyKernelRuntime()
        try:
            await rt.start()
            manager = self._refuse_shutdown(rt, times=_SHUTDOWN_ATTEMPTS)

            with pytest.raises(RuntimeExecutionError, match="still running"):
                await rt.stop()

            assert await manager.is_alive()
            assert rt._executor._orphaned is True
        finally:
            await rt.stop()

    async def test_start_is_refused_while_the_old_kernel_lives(self):
        from cave_agent.runtime.executor import RuntimeExecutionError
        from cave_agent.runtime.ipykernel_executor import _SHUTDOWN_ATTEMPTS

        rt = IPyKernelRuntime()
        try:
            await rt.start()
            manager = self._refuse_shutdown(rt, times=_SHUTDOWN_ATTEMPTS)
            with pytest.raises(RuntimeExecutionError):
                await rt.stop()

            # Typed like every other runtime failure, so a caller catching the
            # documented type — as __aexit__ does — actually sees it.
            with pytest.raises(RuntimeExecutionError, match="survived shutdown"):
                await rt.start()
            # The handle to the live process is still the one we had.
            assert rt._executor._km is manager
        finally:
            await rt.stop()

    async def test_a_retried_stop_recovers_the_runtime(self):
        from cave_agent.runtime.executor import RuntimeExecutionError
        from cave_agent.runtime.ipykernel_executor import _SHUTDOWN_ATTEMPTS

        rt = IPyKernelRuntime()
        try:
            await rt.start()
            manager = self._refuse_shutdown(rt, times=_SHUTDOWN_ATTEMPTS)
            with pytest.raises(RuntimeExecutionError):
                await rt.stop()

            await rt.stop()

            assert not await manager.is_alive()
            assert rt._executor._orphaned is False
            result = await rt.execute("print('usable again')")
            assert result.stdout == "usable again\n"
        finally:
            await rt.stop()


class TestNoneIsAValueNotAnAbsence:
    """A variable holding `None` is not a missing variable.

    A declared-but-unset `Variable` holds exactly `None`, and that is the
    normal state of a placeholder the model is asked to fill in. Decoding the
    dill payload before checking whether one arrived collapsed the two, and
    reading such a variable raised `KeyError: not found` — which surfaced only
    against a live model, since the offline fakes never left one unset.
    """

    async def test_an_unset_variable_reads_back_as_none(self):
        rt = IPyKernelRuntime(variables=[Variable("placeholder", description="unset")])
        try:
            assert await rt.retrieve("placeholder") is None
        finally:
            await rt.stop()

    async def test_a_variable_explicitly_set_to_none_reads_back_as_none(self):
        rt = IPyKernelRuntime()
        try:
            await rt.execute("value = None")
            assert await rt.get_from_namespace("value") is None
        finally:
            await rt.stop()

    async def test_a_genuinely_missing_name_still_raises(self):
        rt = IPyKernelRuntime()
        try:
            with pytest.raises(KeyError):
                await rt.get_from_namespace("never_defined")
        finally:
            await rt.stop()

    async def test_a_real_value_is_unaffected(self):
        rt = IPyKernelRuntime()
        try:
            await rt.execute("value = {'a': 1}")
            assert await rt.get_from_namespace("value") == {"a": 1}
        finally:
            await rt.stop()


class TestGeneratedCodeCannotBreakTheTransport:
    """Data comes back on a channel the model cannot reach.

    The helper used to call `display()` — an ordinary global IPython puts in
    front of every user — so `display = "..."` in generated code disabled every
    namespace read and output persistence for the rest of the session.
    """

    SHADOWS_DISPLAY = "```python\ndisplay = 'user-owned value'\nprint('x' * 1000)\n```"

    async def test_namespace_reads_survive_a_shadowed_display(self):
        rt = IPyKernelRuntime()
        try:
            await rt.execute("display = 'user-owned value'\nkept = 42")
            assert await rt.get_from_namespace("kept") == 42
            assert await rt.get_from_namespace("display") == "user-owned value"
        finally:
            await rt.stop()

    async def test_persistence_survives_a_shadowed_display(self):
        from cave_agent import CaveAgent, StopReason

        from .fakes import FakeModel

        rt = IPyKernelRuntime()
        try:
            agent = CaveAgent(
                model=FakeModel([self.SHADOWS_DISPLAY, "done"]),
                runtime=rt,
                max_exec_output=100,
            )
            response = await agent.run("q")

            assert response.stop_reason is StopReason.COMPLETED
            assert len(await rt.get_from_namespace("_output_1")) == 1001
        finally:
            await rt.stop()

    async def test_allocation_survives_a_shadowed_display(self):
        rt = IPyKernelRuntime()
        try:
            await rt.execute("display = 'user-owned value'")
            assert await rt.bind_unique("_output", "payload") == "_output_1"
        finally:
            await rt.stop()


class TestContextExitDoesNotMaskTheBody:
    """`stop()` raising is worth knowing — but not instead of the real error."""

    def _refuse_shutdown_forever(self, rt):
        """Break shutdown, and hand back a restore so teardown can still work."""
        manager = rt._executor._km
        real = manager.shutdown_kernel

        async def always_fail(*args, **kwargs):
            raise RuntimeError("transport failure")

        manager.shutdown_kernel = always_fail
        return lambda: setattr(manager, "shutdown_kernel", real)

    async def test_a_body_exception_survives_a_failing_shutdown(self):
        rt = IPyKernelRuntime()
        restore = None
        try:
            with pytest.raises(ValueError, match="body failed"):
                async with rt:
                    restore = self._refuse_shutdown_forever(rt)
                    raise ValueError("body failed")
        finally:
            if restore:
                restore()
            await rt.stop()

    async def test_cancellation_survives_a_failing_shutdown(self):
        import asyncio

        rt = IPyKernelRuntime()
        restore = None
        try:
            with pytest.raises(asyncio.CancelledError):
                async with rt:
                    restore = self._refuse_shutdown_forever(rt)
                    raise asyncio.CancelledError()
        finally:
            if restore:
                restore()
            await rt.stop()

    async def test_a_clean_exit_still_reports_a_leaked_kernel(self):
        """Nothing is being hidden there, and a leaked kernel is the only news."""
        from cave_agent.runtime.executor import RuntimeExecutionError

        rt = IPyKernelRuntime()
        restore = None
        try:
            with pytest.raises(RuntimeExecutionError, match="still running"):
                async with rt:
                    restore = self._refuse_shutdown_forever(rt)
        finally:
            if restore:
                restore()
            await rt.stop()


class TestHelperNamesCannotBeShadowed:
    """Every name a helper resolves is one the generated code could rebind.

    Infrastructure must not resolve through any ordinary entry in the namespace
    the model writes to.
    """

    SHADOWED = "display = 1\nget_ipython = 2\nglobals = {'user': 'value'}\n__import__ = None\n"

    async def test_namespace_reads_survive(self):
        rt = IPyKernelRuntime()
        try:
            await rt.execute(self.SHADOWED + "kept = 42")
            assert await rt.get_from_namespace("kept") == 42
        finally:
            await rt.stop()

    async def test_allocation_survives(self):
        rt = IPyKernelRuntime()
        try:
            await rt.execute(self.SHADOWED)
            assert await rt.bind_unique("_output", "payload") == "_output_1"
            assert await rt.get_from_namespace("_output_1") == "payload"
        finally:
            await rt.stop()

    async def test_output_persistence_survives(self):
        from cave_agent import CaveAgent, StopReason

        from .fakes import FakeModel

        rt = IPyKernelRuntime()
        try:
            agent = CaveAgent(
                model=FakeModel([f"```python\n{self.SHADOWED}print('x' * 1000)\n```", "ok"]),
                runtime=rt,
                max_exec_output=100,
            )
            response = await agent.run("q")

            assert response.stop_reason is StopReason.COMPLETED
            assert len(await rt.get_from_namespace("_output_1")) == 1001
        finally:
            await rt.stop()

    async def test_transport_survives_replacing_builtins_mapping(self):
        rt = IPyKernelRuntime()
        try:
            result = await rt.execute("payload = 42\n__builtins__ = None")
            assert result.success
            assert await rt.get_from_namespace("payload") == 42
            assert await rt.bind_unique("_output", "kept") == "_output_1"
            assert await rt.get_from_namespace("_output_1") == "kept"
        finally:
            await rt.stop()

    async def test_output_persistence_survives_replacing_builtins_mapping(self):
        from cave_agent import CaveAgent, StopReason

        from .fakes import FakeModel

        rt = IPyKernelRuntime()
        try:
            agent = CaveAgent(
                model=FakeModel(
                    [
                        "```python\nprint('x' * 1000)\n__builtins__ = None\n```",
                        "ok",
                    ]
                ),
                runtime=rt,
                max_exec_output=100,
            )

            response = await agent.run("q")

            assert response.stop_reason is StopReason.COMPLETED
            assert len(await rt.get_from_namespace("_output_1")) == 1001
        finally:
            await rt.stop()


class TestAllocationNormalizesLikePython:
    """`bind_unique` compares names as strings, so it must compare the names
    Python will actually bind — `K` (U+212A) binds `K`."""

    KELVIN = chr(0x212A)

    async def test_the_kernel_does_not_hand_out_the_same_name_twice(self):
        rt = IPyKernelRuntime()
        try:
            first = await rt.bind_unique(self.KELVIN, "first")
            second = await rt.bind_unique(self.KELVIN, "second")

            assert (first, second) == ("K_1", "K_2")
            assert await rt.get_from_namespace("K_1") == "first"
        finally:
            await rt.stop()


class TestDisplayOutputIsNotDiscarded:
    """`display()` and rich reprs are output too.

    Taking only `stream` messages told the model "No output" for a cell that
    had produced some — the in-process backend captured it all along.
    """

    async def test_display_reaches_the_result(self):
        rt = IPyKernelRuntime()
        try:
            result = await rt.execute("display({'x': 1})")
            assert "'x': 1" in (result.stdout or "")
        finally:
            await rt.stop()

    async def test_a_bare_expression_still_reaches_the_result(self):
        rt = IPyKernelRuntime()
        try:
            result = await rt.execute("40 + 2")
            assert "42" in (result.stdout or "")
        finally:
            await rt.stop()


class TestTeardownIsCancellationSafe:
    """Once teardown starts, cancellation waits for a stable stopped state."""

    async def _cancel_midway(self, operation):
        import asyncio

        rt = IPyKernelRuntime()
        await rt.start()
        executor = rt._executor
        real = executor._terminate_kernel
        reached = asyncio.Event()

        async def slow():
            reached.set()
            await asyncio.sleep(0.2)
            return await real()

        executor._terminate_kernel = slow
        task = asyncio.create_task(getattr(rt, operation)())
        await reached.wait()
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        outcome = (await asyncio.gather(task, return_exceptions=True))[0]
        executor._terminate_kernel = real
        return rt, executor, outcome

    @pytest.mark.parametrize("operation", ["stop", "reset"])
    async def test_cancellation_finishes_process_cleanup(self, operation):
        rt, executor, outcome = await self._cancel_midway(operation)
        try:
            assert isinstance(outcome, asyncio.CancelledError)
            assert executor._started is False
            assert executor._kernel_client is None
            assert not await executor._km.is_alive()
        finally:
            with contextlib.suppress(Exception):
                await rt.stop()

    @pytest.mark.parametrize("operation", ["stop", "reset"])
    async def test_the_next_call_starts_a_clean_kernel(self, operation):
        rt, _, _ = await self._cancel_midway(operation)
        try:
            result = await rt.execute("print('probe')")
            assert result.success
            assert result.stdout == "probe\n"
        finally:
            with contextlib.suppress(Exception):
                await rt.stop()


class TestThreadOutputIsCaptured:
    """Thread output belongs to the cell whose active context started it.

    This is the difference the in-process backend documents as a limitation.
    """

    async def test_output_from_a_spawned_thread_reaches_the_result(self):
        rt = IPyKernelRuntime()
        try:
            result = await rt.execute(
                "import threading\n"
                "t = threading.Thread(target=lambda: print('CHILD-OUTPUT'))\n"
                "t.start(); t.join()"
            )
            assert "CHILD-OUTPUT" in (result.stdout or "")
        finally:
            await rt.stop()

    async def test_a_thread_that_outlives_its_cell_cannot_pollute_the_next_one(self):
        rt = IPyKernelRuntime()
        try:
            await rt.execute(
                "import threading\n"
                "release = threading.Event()\n"
                "t = threading.Thread(target=lambda: (release.wait(), print('LATE')))\n"
                "t.start()"
            )

            result = await rt.execute("release.set(); t.join(); print('CURRENT')")

            assert result.stdout == "CURRENT\n"
        finally:
            await rt.stop()

    async def test_old_thread_traffic_does_not_refresh_another_cells_timeout(self):
        rt = IPyKernelRuntime(iopub_timeout=0.15)
        try:
            await rt.execute(
                "import threading, time\n"
                "def old_writer():\n"
                "    for i in range(30):\n"
                "        print(f'OLD-{i}', flush=True)\n"
                "        time.sleep(0.03)\n"
                "threading.Thread(target=old_writer, daemon=True).start()"
            )

            result = await rt.execute("time.sleep(0.5); print('CURRENT')")

            assert isinstance(result.error, TimeoutError)
            assert "OLD-" not in (result.stdout or "")
        finally:
            await rt.stop()

    async def test_reused_pool_worker_is_reparented_for_each_submission(self):
        rt = IPyKernelRuntime()
        try:
            await rt.execute(
                "from concurrent.futures import ThreadPoolExecutor\n"
                "pool = ThreadPoolExecutor(max_workers=1)\n"
                "pool.submit(lambda: None).result()"
            )

            result = await rt.execute("pool.submit(lambda: print('POOL-CURRENT')).result()")

            assert result.stdout == "POOL-CURRENT\n"
        finally:
            await rt.stop()

    async def test_pool_routing_preserves_asyncio_to_thread_output(self):
        rt = IPyKernelRuntime()
        try:
            result = await rt.execute(
                "import asyncio\nawait asyncio.to_thread(print, 'ASYNCIO-CURRENT')"
            )

            assert result.stdout == "ASYNCIO-CURRENT\n"
        finally:
            await rt.stop()

    async def test_reused_pool_routes_done_callbacks_to_the_registering_cell(self):
        rt = IPyKernelRuntime()
        try:
            await rt.execute(
                "from concurrent.futures import ThreadPoolExecutor\n"
                "import threading\n"
                "pool = ThreadPoolExecutor(max_workers=1)\n"
                "pool.submit(lambda: None).result()"
            )

            result = await rt.execute(
                "gate = threading.Event()\n"
                "callback_done = threading.Event()\n"
                "def callback(future):\n"
                "    print('CALLBACK-CURRENT')\n"
                "    callback_done.set()\n"
                "future = pool.submit(lambda: gate.wait())\n"
                "future.add_done_callback(callback)\n"
                "gate.set()\n"
                "future.result()\n"
                "callback_done.wait()\n"
                "print('MAIN-CURRENT')"
            )

            assert result.stdout == "CALLBACK-CURRENT\nMAIN-CURRENT\n"
        finally:
            await rt.stop()

    async def test_pool_submit_preserves_task_keyword_arguments(self):
        rt = IPyKernelRuntime()
        try:
            result = await rt.execute(
                "from concurrent.futures import ThreadPoolExecutor\n"
                "def echo(*, _submit, _capture, _run_with_parent):\n"
                "    return (_submit, _capture, _run_with_parent)\n"
                "with ThreadPoolExecutor(max_workers=1) as pool:\n"
                "    print(pool.submit(\n"
                "        echo,\n"
                "        _submit='A',\n"
                "        _capture='B',\n"
                "        _run_with_parent='C',\n"
                "    ).result())"
            )

            assert result.success
            assert result.stdout == "('A', 'B', 'C')\n"
        finally:
            await rt.stop()


class TestRestartRestoresTheRegistry:
    """A stopped kernel takes the namespace; the registry outlives it.

    After `stop(); start()` the runtime still advertised its variables while
    the new kernel had never heard of them.
    """

    async def test_registered_variables_survive_a_restart(self):
        rt = IPyKernelRuntime(variables=[Variable("kept", 42, "x")])
        try:
            assert (await rt.execute("print(kept)")).stdout == "42\n"

            await rt.stop()
            await rt.start()

            assert (await rt.execute("print(kept)")).stdout == "42\n"
        finally:
            await rt.stop()

    async def test_updated_variable_survives_a_restart(self):
        rt = IPyKernelRuntime(variables=[Variable("kept", 1)])
        try:
            await rt.execute("print(kept)")
            rt.update_variable("kept", 2)

            await rt.stop()
            await rt.start()

            assert (await rt.execute("print(kept)")).stdout == "2\n"
        finally:
            await rt.stop()

    async def test_unique_binding_survives_a_restart(self):
        rt = IPyKernelRuntime()
        try:
            name = await rt.bind_unique("_output", "payload")

            await rt.stop()
            await rt.start()

            assert await rt.get_from_namespace(name) == "payload"
        finally:
            await rt.stop()

    async def test_unserializable_host_mutation_cannot_block_shutdown(self):
        """Shutdown uses the last value accepted by the runtime, not mutable
        host state that changed behind its back while releasing the process."""
        payload = []
        rt = IPyKernelRuntime(variables=[Variable("payload", payload)])
        executor = rt._executor

        try:
            with pytest.raises(ValueError, match="body failed"):
                async with rt:
                    await rt.execute("print(payload)")
                    payload.append(item for item in range(2))
                    raise ValueError("body failed")

            assert not await executor._km.is_alive()

            await rt.start()
            assert (await rt.execute("print(payload)")).stdout == "[]\n"
        finally:
            await rt.stop()

    async def test_cancelled_replacement_start_keeps_restore_batch(self):
        rt = IPyKernelRuntime(variables=[Variable("kept", 42)])
        await rt.execute("print(kept)")
        executor = rt._executor
        original_start = executor._start_locked
        entered = asyncio.Event()

        async def wait_forever():
            entered.set()
            await asyncio.Event().wait()

        executor._start_locked = wait_forever
        resetting = asyncio.create_task(rt.reset())
        await entered.wait()
        resetting.cancel()
        await asyncio.gather(resetting, return_exceptions=True)
        executor._start_locked = original_start

        try:
            await rt.start()
            assert (await rt.execute("print(kept)")).stdout == "42\n"
        finally:
            executor._start_locked = original_start
            await rt.stop()

    async def test_failed_replacement_start_keeps_restore_batch(self):
        rt = IPyKernelRuntime(variables=[Variable("kept", 42)])
        await rt.execute("print(kept)")
        executor = rt._executor
        original_start = executor._start_locked

        async def fail():
            raise RuntimeError("replacement start failed")

        executor._start_locked = fail
        with pytest.raises(RuntimeError, match="replacement start failed"):
            await rt.reset()
        executor._start_locked = original_start

        try:
            await rt.start()
            assert (await rt.execute("print(kept)")).stdout == "42\n"
        finally:
            executor._start_locked = original_start
            await rt.stop()

    async def test_an_idempotent_start_does_not_undo_the_model(self):
        """Starting an already-live runtime must not overwrite its work."""
        rt = IPyKernelRuntime(variables=[Variable("kept", 42, "x")])
        try:
            await rt.execute("kept = 'changed by the model'")
            await rt.start()

            assert (await rt.execute("print(kept)")).stdout == "changed by the model\n"
        finally:
            await rt.stop()


class TestOneLoopAtATime:
    """A kernel's channels belong to the loop that created them.

    Overlapping loops hung, and the failure surfaced later and elsewhere as
    "lock is bound to a different event loop". Only overlap is refused —
    driving one runtime from a succession of loops is ordinary and safe.
    """

    def test_a_succession_of_loops_is_fine(self):
        import asyncio

        rt = IPyKernelRuntime()
        try:
            asyncio.run(rt.start())
            assert asyncio.run(rt.execute("print('one')")).stdout == "one\n"
            assert asyncio.run(rt.execute("print('two')")).stdout == "two\n"
        finally:
            asyncio.run(rt.stop())

    def test_overlapping_loops_are_refused_not_hung(self):
        import asyncio
        import threading

        from cave_agent.runtime.executor import RuntimeExecutionError

        rt = IPyKernelRuntime()
        results, barrier = {}, threading.Barrier(2)

        def drive(tag):
            async def main():
                barrier.wait()
                results[tag] = (
                    await asyncio.wait_for(rt.execute(f"print('{tag}')"), timeout=10)
                ).stdout.strip()

            try:
                asyncio.run(main())
            except RuntimeExecutionError:
                results[tag] = "refused"
            except BaseException as error:
                results[tag] = type(error).__name__

        threads = [threading.Thread(target=drive, args=(t,)) for t in ("A", "B")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=25)

        try:
            assert "refused" in results.values(), results
            assert not any(v == "TimeoutError" for v in results.values()), results
        finally:
            asyncio.run(rt.stop())

    def test_reset_from_an_overlapping_loop_is_refused_not_hung(self):
        import asyncio
        import threading
        import time

        from cave_agent.runtime.executor import RuntimeExecutionError

        rt = IPyKernelRuntime()
        outcome = {}

        def execute_slow_cell():
            try:
                outcome["result"] = asyncio.run(
                    rt.execute(
                        "import time\ntime.sleep(1)\nsurvived_refused_reset = 42\nprint('done')"
                    )
                )
            except BaseException as error:
                outcome["error"] = error

        asyncio.run(rt.start())
        thread = threading.Thread(target=execute_slow_cell)
        thread.start()
        for _ in range(200):
            with rt._executor._active_lock:
                if rt._executor._active_depth:
                    break
            time.sleep(0.01)
        else:
            pytest.fail("the execution never entered the runtime")

        try:
            with pytest.raises(RuntimeExecutionError, match="another event loop"):
                asyncio.run(asyncio.wait_for(rt.reset(), timeout=5))

            thread.join(timeout=15)
            assert not thread.is_alive()
            assert "error" not in outcome
            assert outcome["result"].stdout == "done\n"
            assert asyncio.run(rt.get_from_namespace("survived_refused_reset")) == 42
        finally:
            thread.join(timeout=15)
            asyncio.run(rt.stop())
