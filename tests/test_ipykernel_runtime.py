"""Tests for IPyKernelRuntime — process-isolated execution."""

import pytest
import pytest_asyncio
from dataclasses import dataclass

from cave_agent.runtime import IPyKernelRuntime, ExecutionResult, ErrorFeedbackMode, Function, Variable, Type
from cave_agent.security import SecurityChecker, ImportRule, FunctionRule


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


# ---------------------------------------------------------------------------
# Execution basics
# ---------------------------------------------------------------------------


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
        await runtime.execute("class Counter:\n  def __init__(self): self.n = 0\n  def inc(self): self.n += 1")
        await runtime.execute("c = Counter(); c.inc(); c.inc()")
        result = await runtime.execute("print(c.n)")
        assert result.success
        assert "2" in result.stdout

    @pytest.mark.asyncio
    async def test_empty_output(self, runtime):
        result = await runtime.execute("x = 1")
        assert result.success
        assert result.stdout is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Injection (each test needs its own kernel)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


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
    async def test_execute_before_start_raises(self):
        rt = IPyKernelRuntime()
        with pytest.raises(RuntimeError, match="not started"):
            await rt.execute("print(1)")


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Describe (metadata, no kernel needed)
# ---------------------------------------------------------------------------


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
