"""Concurrency and failure atomicity at the Runtime/executor boundary."""

import asyncio
import threading
import time

import pytest

from cave_agent import IPyKernelRuntime, IPythonRuntime, RuntimeExecutionError, Variable


@pytest.mark.parametrize("runtime_type", [IPythonRuntime, IPyKernelRuntime])
async def test_unique_binding_is_recorded_before_reset_can_snapshot(runtime_type):
    runtime = runtime_type()
    original = runtime._executor.bind_unique
    bound = asyncio.Event()
    release = asyncio.Event()

    async def pause_after_binding(
        prefix,
        value,
        *,
        start=1,
        reserve=None,
        on_bound=None,
        on_failed=None,
    ):
        kwargs = {"start": start}
        if reserve is not None:
            kwargs["reserve"] = reserve
        if on_bound is not None:
            kwargs["on_bound"] = on_bound
        if on_failed is not None:
            kwargs["on_failed"] = on_failed
        name = await original(prefix, value, **kwargs)
        bound.set()
        await release.wait()
        return name

    runtime._executor.bind_unique = pause_after_binding
    allocating = asyncio.create_task(runtime.bind_unique("_output", "PAYLOAD"))
    try:
        await bound.wait()
        await runtime.reset()
        release.set()
        name = await allocating
        assert await runtime.get_from_namespace(name) == "PAYLOAD"
    finally:
        release.set()
        await asyncio.gather(allocating, return_exceptions=True)
        if isinstance(runtime, IPyKernelRuntime):
            await runtime.stop()


@pytest.mark.parametrize("runtime_type", [IPythonRuntime, IPyKernelRuntime])
async def test_concurrent_registrations_cannot_both_claim_one_name(runtime_type):
    entered = threading.Event()
    release = threading.Event()

    class SlowVariable(Variable):
        def __copy__(self):
            if self.value == "first":
                entered.set()
                release.wait(timeout=2)
            return Variable(self.name, self.value)

    runtime = runtime_type()
    outcomes = []

    def register(value):
        try:
            runtime.inject_variable(SlowVariable("shared", value))
            outcomes.append(("ok", value))
        except Exception as error:
            outcomes.append((type(error).__name__, value))

    first = threading.Thread(target=register, args=("first",))
    second = threading.Thread(target=register, args=("second",))
    first.start()
    assert entered.wait(timeout=2)
    second.start()
    time.sleep(0.05)
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    try:
        assert sorted(kind for kind, _ in outcomes) == ["ValueError", "ok"]
        assert await runtime.get_from_namespace("shared") == "first"
    finally:
        if isinstance(runtime, IPyKernelRuntime):
            await runtime.stop()


@pytest.mark.parametrize("runtime_type", [IPythonRuntime, IPyKernelRuntime])
async def test_unique_binding_reserves_its_name_against_registration(runtime_type):
    runtime = runtime_type()
    original = runtime._executor.bind_unique
    reserved = threading.Event()
    release = threading.Event()
    registration = {}

    async def pause_after_reservation(
        prefix,
        value,
        *,
        start=1,
        reserve=None,
        on_bound=None,
        on_failed=None,
    ):
        def pause(name):
            accepted = reserve(name)
            if accepted:
                reserved.set()
                assert release.wait(timeout=5)
            return accepted

        return await original(
            prefix,
            value,
            start=start,
            reserve=pause,
            on_bound=on_bound,
            on_failed=on_failed,
        )

    def register_same_name():
        assert reserved.wait(timeout=5)
        try:
            runtime.inject_variable(Variable("_output_1", "registration"))
        except Exception as error:
            registration["error"] = error
        finally:
            release.set()

    runtime._executor.bind_unique = pause_after_reservation
    registering = threading.Thread(target=register_same_name)
    registering.start()
    try:
        name = await runtime.bind_unique("_output", "binding")
        registering.join(timeout=5)

        assert not registering.is_alive()
        assert isinstance(registration.get("error"), ValueError)
        assert name == "_output_1"
        assert await runtime.get_from_namespace(name) == "binding"
    finally:
        release.set()
        registering.join(timeout=5)
        if isinstance(runtime, IPyKernelRuntime):
            await runtime.stop()


async def test_failed_kernel_binding_releases_its_name_reservation():
    runtime = IPyKernelRuntime()
    try:
        with pytest.raises(RuntimeExecutionError):
            await runtime.bind_unique("_output", (item for item in range(2)))

        runtime.inject_variable(Variable("_output_1", "available"))
        assert await runtime.get_from_namespace("_output_1") == "available"
    finally:
        await runtime.stop()


async def test_kernel_rejects_an_unserializable_batch_without_partial_commit():
    runtime = IPyKernelRuntime()
    values = [
        Variable("first", 1),
        Variable("second", (item for item in range(3))),
    ]

    with pytest.raises(TypeError, match="cannot pickle"):
        runtime.inject_resources(variables=values)

    assert "first" not in runtime._variables
    assert "second" not in runtime._variables
    try:
        result = await runtime.execute("print('first' in globals(), 'second' in globals())")
        assert result.success
        assert result.stdout == "False False\n"
    finally:
        await runtime.stop()
