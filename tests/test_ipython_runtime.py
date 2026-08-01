import asyncio
import time
from dataclasses import dataclass
from enum import Enum

import pytest

from cave_agent.runtime import Function, IPythonRuntime, Type, Variable


@pytest.fixture
def simple_runtime():
    return IPythonRuntime()


@pytest.fixture
def runtime_with_data():
    numbers_var = Variable(
        name="numbers", value=[3, 1, 4, 1, 5, 9, 2, 6, 5], description="List of numbers to process"
    )

    result_var = Variable(name="result", description="Store calculation results here")

    return IPythonRuntime(variables=[numbers_var, result_var])


@pytest.fixture
def runtime_with_function():
    def multiply(a, b):
        """Multiply two numbers"""
        return a * b

    func = Function(multiply, "Multiplication function")
    return IPythonRuntime(functions=[func])


@pytest.mark.asyncio
async def test_basic_execution(simple_runtime):
    """Test basic code execution works"""
    await simple_runtime.execute("x = 5 + 3")
    result = await simple_runtime.get_from_namespace("x")
    assert result == 8


@pytest.mark.asyncio
async def test_print_output(simple_runtime):
    """Test code with print output"""
    output = await simple_runtime.execute("print('Hello World')")
    assert "Hello World" in output.stdout


@pytest.mark.asyncio
async def test_writelines_output_is_captured(simple_runtime):
    output = await simple_runtime.execute("import sys; sys.stdout.writelines(['A', 'B'])")
    assert output.stdout == "AB"


@pytest.mark.asyncio
async def test_binary_stdout_is_captured(simple_runtime):
    output = await simple_runtime.execute(
        "import sys\n"
        "_written = sys.stdout.buffer.write(b'binary-output\\n')\n"
        "_written = sys.stdout.buffer.write(bytes([255]))\n"
        "sys.stdout.buffer.flush()"
    )
    assert output.stdout == "binary-output\n\ufffd"


@pytest.mark.asyncio
async def test_task_outliving_cell_falls_back_to_host(simple_runtime, capsys):
    first = await simple_runtime.execute(
        "import asyncio\n"
        "async def _later():\n"
        "    await asyncio.sleep(0.02)\n"
        "    print('LATE-OUTPUT')\n"
        "asyncio.create_task(_later())\n"
        "print('CELL-OUTPUT')"
    )
    await asyncio.sleep(0.05)

    assert "CELL-OUTPUT" in (first.stdout or "")
    assert "LATE-OUTPUT" not in (first.stdout or "")
    assert "LATE-OUTPUT" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_variable_usage(runtime_with_data):
    """Test using injected variables"""
    await runtime_with_data.execute("result = sum(numbers)")
    total = await runtime_with_data.retrieve("result")
    assert total == 36


@pytest.mark.asyncio
async def test_function_usage(runtime_with_function):
    """Test using injected functions"""
    await runtime_with_function.execute("result = multiply(6, 7)")
    result = await runtime_with_function.get_from_namespace("result")
    assert result == 42


@pytest.mark.asyncio
async def test_multiple_executions(simple_runtime):
    """Test multiple code executions share state"""
    await simple_runtime.execute("a = 10")
    await simple_runtime.execute("b = a * 2")
    await simple_runtime.execute("c = a + b")

    result = await simple_runtime.get_from_namespace("c")
    assert result == 30


def test_describe_functions(runtime_with_function):
    """Test function description works"""
    description = runtime_with_function.describe_functions()
    assert "multiply" in description
    assert "function:" in description


def test_describe_variables(runtime_with_data):
    """Test variable description works"""
    description = runtime_with_data.describe_variables()
    assert "numbers" in description
    assert "result" in description


class Light:
    """A smart light device."""

    def __init__(self, name: str = "Light"):
        self.name = name
        self.is_on = False

    def turn_on(self) -> str:
        """Turn the light on."""
        self.is_on = True
        return f"{self.name} turned on"

    def turn_off(self) -> str:
        """Turn the light off."""
        self.is_on = False
        return f"{self.name} turned off"


class Lock:
    """A smart lock device."""

    def __init__(self, name: str = "Lock"):
        self.name = name
        self.is_locked = True

    def lock(self) -> str:
        self.is_locked = True
        return f"{self.name} locked"

    def unlock(self) -> str:
        self.is_locked = False
        return f"{self.name} unlocked"


@dataclass
class DataPoint:
    """A data point with x and y coordinates."""

    x: float
    y: float


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class TestTypeCreation:
    """Test Type class creation and validation."""

    def test_type_creation_basic(self):
        """Type can be created with a class."""
        type_obj = Type(Light)
        assert type_obj.name == "Light"
        assert type_obj.value is Light
        assert type_obj.description is None
        assert type_obj.include_schema is True
        assert type_obj.include_doc is True

    def test_type_creation_with_description(self):
        """Type can be created with description."""
        type_obj = Type(Light, "Smart light class")
        assert type_obj.description == "Smart light class"

    def test_type_creation_with_options(self):
        """Type can be created with schema/doc options."""
        type_obj = Type(Light, include_schema=False, include_doc=False)
        assert type_obj.include_schema is False
        assert type_obj.include_doc is False

    def test_type_creation_requires_class(self):
        """Type raises error if value is not a class."""
        with pytest.raises(ValueError, match="must be a class"):
            Type(Light())  # Instance, not class

    def test_type_str_returns_schema(self):
        """Type __str__ returns full schema when include_schema=True."""
        type_obj = Type(Light)
        result = str(type_obj)
        assert "Light:" in result
        assert "doc: A smart light device." in result
        assert "methods:" in result
        assert "turn_on()" in result

    def test_type_str_includes_description(self):
        """Type __str__ includes description if provided."""
        type_obj = Type(Light, "Smart light controller")
        result = str(type_obj)
        assert "Light:" in result
        assert "description: Smart light controller" in result
        assert "methods:" in result

    def test_type_str_empty_when_hidden(self):
        """Type __str__ returns empty string when schema and doc are hidden."""
        type_obj = Type(Light, include_schema=False, include_doc=False)
        result = str(type_obj)
        assert result == ""


class TestTypeInjection:
    """Test Type injection into PythonRuntime."""

    def test_inject_type_via_constructor(self):
        """Types can be injected via constructor."""
        runtime = IPythonRuntime(types=[Type(Light)])
        assert "Light" in runtime._types

    def test_inject_type_via_method(self):
        """Types can be injected via inject_type method."""
        runtime = IPythonRuntime()
        runtime.inject_type(Type(Light))
        assert "Light" in runtime._types

    def test_inject_multiple_types(self):
        """Multiple types can be injected."""
        runtime = IPythonRuntime(
            types=[
                Type(Light),
                Type(Lock),
            ]
        )
        assert "Light" in runtime._types
        assert "Lock" in runtime._types

    def test_inject_duplicate_type_raises(self):
        """Injecting duplicate type raises error."""
        runtime = IPythonRuntime(types=[Type(Light)])
        with pytest.raises(ValueError, match="already exists"):
            runtime.inject_type(Type(Light))

    @pytest.mark.asyncio
    async def test_type_available_in_namespace(self):
        """Injected type is available in execution namespace."""
        runtime = IPythonRuntime(types=[Type(Light)])
        assert await runtime.get_from_namespace("Light") is Light


@pytest.mark.asyncio
class TestTypeExecution:
    """Test using injected types in code execution."""

    async def test_isinstance_check(self):
        """Can use isinstance with injected type (auto-injected from Variable)."""
        light = Light("Kitchen")
        runtime = IPythonRuntime(
            variables=[Variable("device", light, "A device")],
        )

        # Light is auto-injected when Variable is injected
        assert "Light" in runtime._types

        await runtime.execute("result = isinstance(device, Light)")
        assert await runtime.get_from_namespace("result") is True

    async def test_isinstance_check_negative(self):
        """isinstance returns False for non-matching type."""
        light = Light("Kitchen")
        runtime = IPythonRuntime(
            variables=[Variable("device", light, "A device")],
            types=[Type(Lock)],  # Only Lock needs explicit injection
        )

        await runtime.execute("result = isinstance(device, Lock)")
        assert await runtime.get_from_namespace("result") is False

    async def test_instantiate_type(self):
        """Can instantiate injected type."""
        runtime = IPythonRuntime(types=[Type(Light)])

        await runtime.execute("light = Light('Bedroom')")
        light = await runtime.get_from_namespace("light")
        assert isinstance(light, Light)
        assert light.name == "Bedroom"

    async def test_call_method_on_new_instance(self):
        """Can call methods on newly created instance."""
        runtime = IPythonRuntime(types=[Type(Light)])

        await runtime.execute("""
light = Light('Bedroom')
result = light.turn_on()
""")
        result = await runtime.get_from_namespace("result")
        assert result == "Bedroom turned on"

        light = await runtime.get_from_namespace("light")
        assert light.is_on is True

    async def test_filter_by_type(self):
        """Can filter list by type using isinstance."""
        devices = [Light("Kitchen"), Lock("Front"), Light("Bedroom")]
        runtime = IPythonRuntime(
            variables=[Variable("devices", devices, "List of devices")],
            types=[Type(Light), Type(Lock)],
        )

        await runtime.execute("lights = [d for d in devices if isinstance(d, Light)]")
        lights = await runtime.get_from_namespace("lights")
        assert len(lights) == 2
        assert all(isinstance(light, Light) for light in lights)

    async def test_type_with_dataclass(self):
        """Can use dataclass as injected type."""
        runtime = IPythonRuntime(types=[Type(DataPoint)])

        await runtime.execute("point = DataPoint(x=1.0, y=2.0)")
        point = await runtime.get_from_namespace("point")
        assert point.x == 1.0
        assert point.y == 2.0

    async def test_type_with_enum(self):
        """Can use enum as injected type."""
        runtime = IPythonRuntime(types=[Type(Priority)])

        await runtime.execute("p = Priority.HIGH")
        p = await runtime.get_from_namespace("p")
        assert p == Priority.HIGH
        assert p.value == 3


class TestTypeDescribeTypes:
    """Test Type integration with describe_types()."""

    def test_type_schema_in_describe_types(self):
        """Injected Type schema appears in describe_types()."""
        runtime = IPythonRuntime(types=[Type(Light)])
        result = runtime.describe_types()

        assert "Light:" in result
        assert "doc: A smart light device." in result
        assert "methods:" in result
        assert "turn_on()" in result
        assert "turn_off()" in result

    def test_type_schema_without_doc(self):
        """Type with include_doc=False excludes docstring."""
        runtime = IPythonRuntime(types=[Type(Light, include_schema=True, include_doc=False)])
        result = runtime.describe_types()

        assert "Light:" in result
        assert "doc:" not in result
        assert "methods:" in result
        assert "turn_on()" in result

    def test_type_doc_only(self):
        """Type with include_schema=False shows doc only."""
        runtime = IPythonRuntime(types=[Type(Light, include_schema=False, include_doc=True)])
        result = runtime.describe_types()

        assert "Light:" in result
        assert "doc: A smart light device." in result
        assert "methods:" not in result

    def test_type_no_schema_no_doc(self):
        """Type with both False shows nothing in describe_types()."""
        runtime = IPythonRuntime(types=[Type(Light, include_schema=False, include_doc=False)])
        result = runtime.describe_types()

        assert result == "No types available"

    def test_type_dataclass_schema(self):
        """Dataclass type shows fields in schema."""
        runtime = IPythonRuntime(types=[Type(DataPoint)])
        result = runtime.describe_types()

        assert "DataPoint:" in result
        assert "fields:" in result
        assert "x: float" in result
        assert "y: float" in result

    def test_type_enum_schema(self):
        """Enum type shows values in schema."""
        runtime = IPythonRuntime(types=[Type(Priority)])
        result = runtime.describe_types()

        assert "Priority (Enum):" in result
        assert "LOW = 1" in result
        assert "MEDIUM = 2" in result
        assert "HIGH = 3" in result

    def test_multiple_types_in_describe_types(self):
        """Multiple types appear in describe_types()."""
        runtime = IPythonRuntime(
            types=[
                Type(Light),
                Type(Lock),
            ]
        )
        result = runtime.describe_types()

        assert "Light:" in result
        assert "Lock:" in result
        assert "turn_on()" in result
        assert "lock()" in result


class TestTypeAutoInjection:
    """Test automatic type injection from Variables and Functions."""

    def test_variable_auto_inject_schema_hidden(self):
        """Variable auto-injects type with schema hidden."""
        light = Light("Kitchen")
        runtime = IPythonRuntime(variables=[Variable("light", light, "A light")])

        # Type is auto-injected with schema=False
        assert "Light" in runtime._types
        assert runtime._types["Light"].include_schema is False
        assert runtime._types["Light"].include_doc is False

        # Schema NOT shown
        result = runtime.describe_types()
        assert "Light:" not in result

    def test_variable_does_not_inject_builtins(self):
        """Built-in types are not auto-injected."""
        runtime = IPythonRuntime(
            variables=[
                Variable("numbers", [1, 2, 3], "A list"),
                Variable("name", "hello", "A string"),
                Variable("count", 42, "An int"),
            ]
        )

        # No types should be auto-injected
        assert len(runtime._types) == 0

    def test_function_auto_inject_schema_hidden(self):
        """Function auto-injects types with schema hidden."""

        def process(device: Light) -> str:
            return "done"

        runtime = IPythonRuntime(functions=[Function(process)])

        # Type is auto-injected with schema=False
        assert "Light" in runtime._types
        assert runtime._types["Light"].include_schema is False
        assert runtime._types["Light"].include_doc is False

        # Schema NOT shown
        result = runtime.describe_types()
        assert "Light:" not in result

    @pytest.mark.asyncio
    async def test_function_auto_injects_return_type(self):
        """Injecting a Function auto-injects return type."""

        def create_lock() -> Lock:
            return Lock()

        runtime = IPythonRuntime(functions=[Function(create_lock)])

        assert "Lock" in runtime._types
        assert await runtime.get_from_namespace("Lock") is Lock

    @pytest.mark.asyncio
    async def test_function_resolves_postponed_annotations(self):
        namespace = {}
        exec(
            "from __future__ import annotations\n"
            "class Payload:\n"
            "    pass\n"
            "def consume(value: Payload) -> Payload:\n"
            "    return value\n",
            namespace,
        )
        runtime = IPythonRuntime(functions=[Function(namespace["consume"])])

        result = await runtime.execute("created = consume(Payload())")

        assert result.success
        assert type(await runtime.get_from_namespace("created")).__name__ == "Payload"

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
        runtime = IPythonRuntime(functions=[Function(namespace["consume"])])

        result = await runtime.execute("created = consume(Known(), None)")

        assert result.success
        assert await runtime.get_from_namespace("Known") is namespace["Known"]

    def test_function_auto_injects_multiple_types(self):
        """Function with multiple custom types injects all."""

        def transfer(source: Light, target: Lock) -> DataPoint:
            pass

        runtime = IPythonRuntime(functions=[Function(transfer)])

        assert "Light" in runtime._types
        assert "Lock" in runtime._types
        assert "DataPoint" in runtime._types

    def test_function_does_not_inject_builtins(self):
        """Function with only built-in types doesn't inject anything."""

        def process(items: list, count: int) -> str:
            pass

        runtime = IPythonRuntime(functions=[Function(process)])

        assert len(runtime._types) == 0

    def test_auto_inject_skips_duplicates(self):
        """Auto-injection skips types already injected."""
        light1 = Light("Kitchen")
        light2 = Light("Bedroom")

        # Both variables have same type - should not raise
        runtime = IPythonRuntime(
            variables=[
                Variable("light1", light1, "Light 1"),
                Variable("light2", light2, "Light 2"),
            ]
        )

        # Only one Light type should exist
        assert "Light" in runtime._types
        assert len(runtime._types) == 1

    def test_explicit_type_takes_precedence(self):
        """Explicitly injected Type takes precedence over auto-injection."""
        light = Light("Kitchen")
        runtime = IPythonRuntime(
            types=[Type(Light, "Explicit light")],  # schema=True by default
            variables=[Variable("light", light, "A light")],
        )

        # Should use explicit Type settings
        assert runtime._types["Light"].description == "Explicit light"
        assert runtime._types["Light"].include_schema is True

        # Schema IS shown (because explicit Type has include_schema=True)
        result = runtime.describe_types()
        assert "Light:" in result
        assert "turn_on()" in result

    def test_explicit_type_to_show_schema(self):
        """Use explicit Type to show schema for auto-injected types."""

        def process(device: Light) -> str:
            return "done"

        # Without explicit Type, schema is hidden
        runtime1 = IPythonRuntime(functions=[Function(process)])
        assert runtime1.describe_types() == "No types available"

        # With explicit Type, schema is shown
        runtime2 = IPythonRuntime(
            types=[Type(Light)],
            functions=[Function(process)],
        )
        result = runtime2.describe_types()
        assert "Light:" in result
        assert "turn_on()" in result


class TestVariableUpdateContract:
    @pytest.mark.asyncio
    async def test_none_does_not_erase_the_declared_type(self):
        runtime = IPythonRuntime(variables=[Variable("count", 1)])
        runtime.update_variable("count", None)

        with pytest.raises(TypeError, match="Expected int"):
            runtime.update_variable("count", "not an int")

        assert await runtime.retrieve("count") is None
        assert "type: int" in runtime.describe_variables()

    @pytest.mark.asyncio
    async def test_first_value_types_an_unset_variable(self):
        runtime = IPythonRuntime(variables=[Variable("result")])

        runtime.update_variable("result", 42)

        assert await runtime.retrieve("result") == 42
        assert "type: int" in runtime.describe_variables()


class TestTypeReset:
    """Reset semantics — identical across every runtime backend.

    Reset clears the *namespace* but keeps the registry: a reset runtime is
    still the runtime the system prompt describes. Dropping the registrations
    would leave the agent advertising functions that no longer exist.
    """

    @pytest.mark.asyncio
    async def test_reset_keeps_registered_types(self):
        runtime = IPythonRuntime(types=[Type(Light)])
        assert "Light" in runtime._types

        await runtime.reset()

        assert "Light" in runtime._types

    @pytest.mark.asyncio
    async def test_reset_reinjects_type_into_namespace(self):
        runtime = IPythonRuntime(types=[Type(Light)])
        assert await runtime.get_from_namespace("Light") is Light

        await runtime.reset()

        assert await runtime.get_from_namespace("Light") is Light

    @pytest.mark.asyncio
    async def test_reset_still_clears_user_state(self):
        """Only registered resources survive — ad-hoc namespace state does not."""
        runtime = IPythonRuntime(types=[Type(Light)])
        await runtime.execute("scratch = 42")
        assert await runtime.get_from_namespace("scratch") == 42

        await runtime.reset()

        with pytest.raises(KeyError):
            await runtime.get_from_namespace("scratch")

    @pytest.mark.asyncio
    async def test_reset_preserves_functions_and_variables(self):
        def helper(x: int) -> int:
            """Double it."""
            return x * 2

        runtime = IPythonRuntime(
            functions=[Function(helper)],
            variables=[Variable("cfg", {"a": 1})],
        )

        await runtime.reset()

        result = await runtime.execute("print(helper(cfg['a']))")
        assert result.success
        assert "2" in result.stdout


class TestResourceHygiene:
    """A runtime must not leak OS resources per instance.

    IPython's HistoryManager spawns an `IPythonHistorySavingThread` holding a
    SQLite connection, one per shell, never reaped. Creating and dropping
    runtimes therefore leaked a thread and a pair of file descriptors each
    time, and a long-lived process accumulated enough of both that unrelated
    I/O — a Jupyter kernel's message channel, say — started missing deadlines.
    """

    @pytest.mark.asyncio
    async def test_no_thread_leak_across_runtimes(self):
        import threading

        before = threading.active_count()
        for _ in range(15):
            runtime = IPythonRuntime()
            await runtime.execute("x = 1")

        assert threading.active_count() == before

    @pytest.mark.asyncio
    async def test_no_history_thread_spawned(self):
        import threading

        runtime = IPythonRuntime()
        await runtime.execute("x = 1")

        names = {t.name for t in threading.enumerate()}
        assert not any("History" in name for name in names), names

    @pytest.mark.asyncio
    async def test_no_fd_leak_across_runtimes(self):
        import os

        fd_dir = "/proc/self/fd"
        if not os.path.isdir(fd_dir):
            pytest.skip("requires /proc")

        before = len(os.listdir(fd_dir))
        for _ in range(15):
            runtime = IPythonRuntime()
            await runtime.execute("x = 1")

        assert len(os.listdir(fd_dir)) == before


class TestOutputIsRoutedNotSeized:
    """A cell's output is collected per execution context, not by claiming
    ``sys.stdout`` for the process.

    Swapping the global stream for the duration of a cell takes everything the
    process writes: a host thread logging while generated code ran had its line
    captured into the model's execution result and removed from where it was
    meant to go. Routing also means separate runtimes no longer interfere, so
    they need not be serialized against each other.
    """

    async def test_a_cell_gets_its_own_output(self):
        runtime = IPythonRuntime()
        result = await runtime.execute("print('mine')")
        assert result.stdout == "mine\n"

    async def test_host_output_is_not_captured(self):
        import io
        import threading

        runtime = IPythonRuntime()
        host = io.StringIO()

        def host_writes():
            time.sleep(0.15)
            print("HOST-LOG", file=host)

        threading.Thread(target=host_writes).start()
        result = await runtime.execute(
            "import time\nprint('CELL-START')\ntime.sleep(0.35)\nprint('CELL-END')"
        )

        assert result.stdout == "CELL-START\nCELL-END\n"
        assert "HOST-LOG" in host.getvalue()

    async def test_stderr_is_kept_in_order_with_stdout(self):
        runtime = IPythonRuntime()

        result = await runtime.execute(
            "import sys\nprint('one')\nprint('warned', file=sys.stderr)\nprint('two')"
        )

        assert result.stdout == "one\nwarned\ntwo\n"

    async def test_stderr_only_output_is_not_lost(self):
        runtime = IPythonRuntime()
        result = await runtime.execute("import sys\nprint('stderr-only', file=sys.stderr)")
        assert result.stdout == "stderr-only\n"

    async def test_concurrent_runs_on_separate_runtimes_do_not_mix(self):
        import asyncio

        one, two = IPythonRuntime(), IPythonRuntime()
        first, second = await asyncio.gather(
            one.execute("import asyncio\nprint('A1')\nawait asyncio.sleep(0.2)\nprint('A2')"),
            two.execute(
                "import asyncio\nawait asyncio.sleep(0.1)\nprint('B1')\n"
                "await asyncio.sleep(0.2)\nprint('B2')"
            ),
        )

        assert first.stdout == "A1\nA2\n"
        assert second.stdout == "B1\nB2\n"

    def test_runtimes_on_separate_loop_threads_do_not_mix(self):
        import asyncio
        import threading

        results = {}

        def run(tag, code):
            async def main():
                results[tag] = (await IPythonRuntime().execute(code)).stdout

            asyncio.run(main())

        left = threading.Thread(
            target=run, args=("L", "import time\nprint('LEFT-1')\ntime.sleep(0.3)\nprint('LEFT-2')")
        )
        right = threading.Thread(
            target=run,
            args=(
                "R",
                "import time\ntime.sleep(0.1)\nprint('RIGHT-1')\ntime.sleep(0.3)\nprint('RIGHT-2')",
            ),
        )
        left.start()
        right.start()
        left.join()
        right.join()

        assert results["L"] == "LEFT-1\nLEFT-2\n"
        assert results["R"] == "RIGHT-1\nRIGHT-2\n"

    async def test_reset_waits_for_a_running_cell(self):
        import asyncio

        runtime = IPythonRuntime()
        cell = asyncio.create_task(
            runtime.execute(
                "import asyncio\nbefore = 41\nawait asyncio.sleep(0.25)\nprint(before + 1)"
            )
        )
        await asyncio.sleep(0.05)

        await runtime.reset()
        result = await cell

        assert result.error is None
        assert result.stdout == "42\n"


class TestNamesResolveTheWayPythonResolvesThem:
    """This backend looks names up in a dict; generated code goes through the
    interpreter, which NFKC-normalizes first. Without normalizing here, the two
    disagreed about which variable `K` (U+212A KELVIN SIGN) means."""

    async def test_a_compatibility_identifier_reads_back(self):
        runtime = IPythonRuntime()
        await runtime.execute("K_1 = 'kelvin'")

        assert await runtime.get_from_namespace("K_1") == "kelvin"
        assert await runtime.get_from_namespace("K_1") == "kelvin"

    async def test_injection_uses_the_name_python_would_bind(self):
        runtime = IPythonRuntime()
        runtime.inject_into_namespace("K_value", 42)

        result = await runtime.execute("print(K_value)")
        assert result.stdout == "42\n"


class TestCancellationIsNotSwallowed:
    """IPython catches cancellation and hands it back as an ordinary result.

    Returned as one, a cancelled agent carried on: it made another model call
    and reported COMPLETED after its task had been cancelled.
    """

    async def test_execute_re_raises_cancellation(self):
        import asyncio

        runtime = IPythonRuntime()
        task = asyncio.create_task(runtime.execute("import asyncio\nawait asyncio.sleep(10)"))
        await asyncio.sleep(0.2)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_a_cancelled_agent_stops(self):
        import asyncio

        from cave_agent import CaveAgent, StopReason

        from .fakes import FakeModel

        agent = CaveAgent(
            model=FakeModel(
                [
                    "```python\nimport asyncio\nawait asyncio.sleep(10)\n```",
                    "continued after cancellation",
                ],
            ),
            runtime=IPythonRuntime(),
            max_steps=3,
        )
        events = []

        async def run():
            async for event in agent.stream_events("q"):
                events.append(event)

        task = asyncio.create_task(run())
        await asyncio.sleep(0.4)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert events[-1].stop_reason is StopReason.CANCELLED


class TestTheGateIsFair:
    """An unfair gate lets a steady stream of executions starve teardown.

    Measured against the gate directly rather than through `execute`: trivial
    cells complete without ever suspending, so a loop of them starves the event
    loop itself and would prove nothing about the gate.
    """

    async def test_a_waiter_is_served_while_others_keep_arriving(self):
        import asyncio
        import time

        gate = IPythonRuntime()._executor._gate
        running = True

        async def churn():
            while running:
                async with gate.held():
                    await asyncio.sleep(0)

        workers = [asyncio.create_task(churn()) for _ in range(4)]
        await asyncio.sleep(0.05)

        started = time.monotonic()
        async with gate.held():
            elapsed = time.monotonic() - started

        running = False
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        assert elapsed < 1.0, f"a waiter took {elapsed:.1f}s to be served"

    async def test_a_cancelled_waiter_leaves_no_ticket_behind(self):
        import asyncio

        gate = IPythonRuntime()._executor._gate

        async with gate.held():
            blocked = asyncio.create_task(gate.held().__aenter__())
            await asyncio.sleep(0.02)
            blocked.cancel()
            await asyncio.gather(blocked, return_exceptions=True)

        # The gate is free again: a queue still holding the cancelled ticket
        # would never let anyone else to the front, and this would time out.
        await asyncio.wait_for(gate.held().__aenter__(), timeout=1)
        # Only this test's own ticket remains; the cancelled one is gone.
        assert len(gate._waiting) == 1, "a cancelled waiter left its ticket behind"


class TestRegisteredNamesShareOneNamespace:
    """The three registries end up in one Python namespace, so they cannot each
    police their own — and a name has to be usable before it is recorded."""

    def test_a_non_identifier_is_rejected_at_registration(self):
        with pytest.raises(ValueError, match="not a Python identifier"):
            IPythonRuntime(variables=[Variable("bad-name", 42, "x")])

    def test_names_that_normalize_together_collide(self):
        with pytest.raises(ValueError, match="NFKC"):
            IPythonRuntime(
                variables=[
                    Variable("K", "ascii", "a"),
                    Variable(chr(0x212A), "kelvin", "b"),
                ]
            )

    def test_a_collision_across_registries_is_caught(self):
        def shared(): ...

        shared.__name__ = "shared"

        with pytest.raises(ValueError, match="collides|already exists"):
            IPythonRuntime(
                variables=[Variable("shared", 1, "x")],
                functions=[Function(shared)],
            )


class TestThreadOutputBoundary:
    """Output from a thread the generated code starts is *not* captured.

    A documented limitation, pinned so it cannot change silently. Capturing it
    would require wrapping every `Thread` target in the process, imposing new
    threading semantics on the host; `IPyKernelRuntime` captures at the process
    boundary and has no such gap.
    """

    SPAWNS_A_THREAD = (
        "import threading\n"
        "t = threading.Thread(target=lambda: print('CHILD-OUTPUT'))\n"
        "t.start(); t.join()\n"
        "print('MAIN-OUTPUT')"
    )

    async def test_the_executing_context_is_still_captured(self):
        runtime = IPythonRuntime()
        result = await runtime.execute(self.SPAWNS_A_THREAD)
        assert "MAIN-OUTPUT" in result.stdout

    async def test_thread_output_is_not_captured_in_process(self):
        runtime = IPythonRuntime()
        result = await runtime.execute(self.SPAWNS_A_THREAD)
        assert "CHILD-OUTPUT" not in result.stdout


class TestEveryBindingRouteClaimsItsName:
    """Auto-injected types and raw bindings go through the same claim.

    Each used to bypass it: registering `Variable("Widget", widget)` and then a
    function taking a `Widget` replaced the instance with the class, and a raw
    injection could land on a registered variable's name without a word.
    """

    async def test_auto_injection_does_not_displace_an_explicit_variable(self):
        class Widget:
            pass

        def takes(w: "Widget") -> None:
            """Takes a widget."""

        widget = Widget()
        runtime = IPythonRuntime(
            variables=[Variable("Widget", widget, "an instance")],
            functions=[Function(takes)],
        )

        assert await runtime.retrieve("Widget") is widget

    def test_a_raw_binding_cannot_take_a_registered_name(self):
        runtime = IPythonRuntime(variables=[Variable("K", "registered", "x")])

        with pytest.raises(ValueError, match="NFKC|already exists"):
            runtime.inject_into_namespace(chr(0x212A), "raw-overwrite")

    async def test_a_raw_binding_may_be_updated(self):
        """Re-binding a name this route already owns is an update, not a clash."""
        runtime = IPythonRuntime()
        runtime.inject_into_namespace("store", {"v": 1})
        runtime.inject_into_namespace("store", {"v": 2})

        assert await runtime.get_from_namespace("store") == {"v": 2}

    async def test_runtime_owns_its_variable_descriptor(self):
        shared = Variable("value", 1, "shared descriptor")
        first = IPythonRuntime(variables=[shared])
        second = IPythonRuntime(variables=[shared])

        first.update_variable("value", 2)
        await second.reset()

        assert await first.retrieve("value") == 2
        assert await second.retrieve("value") == 1
        assert shared.value == 1


class TestNamesMustBeUsableInGeneratedCode:
    """`isidentifier()` alone is not the question — a keyword passes it."""

    @pytest.mark.parametrize("name", ["for", "True", "None", "class"])
    def test_keywords_are_rejected(self, name):
        with pytest.raises(ValueError, match="keyword"):
            IPythonRuntime(variables=[Variable(name, 7, "x")])

    async def test_bind_unique_validates_the_name_it_will_synthesize(self):
        runtime = IPythonRuntime()
        with pytest.raises(ValueError, match="not a Python identifier"):
            await runtime.bind_unique("bad-name", 1)

    async def test_bind_unique_rejects_a_nonsense_floor(self):
        runtime = IPythonRuntime()
        with pytest.raises(ValueError, match="positive integer"):
            await runtime.bind_unique("_output", 1, start=0)


class TestRegistrationDuringReset:
    """A resource registered while a reset is queued must survive it."""

    async def test_a_late_registration_is_not_wiped(self):
        import asyncio

        runtime = IPythonRuntime()
        cell = asyncio.create_task(runtime.execute("import asyncio\nawait asyncio.sleep(0.4)"))
        await asyncio.sleep(0.05)
        resetting = asyncio.create_task(runtime.reset())
        await asyncio.sleep(0.05)

        runtime.inject_variable(Variable("late_resource", 42, "added mid-reset"))
        await asyncio.gather(cell, resetting)

        assert "late_resource" in runtime.describe_variables()
        assert await runtime.retrieve("late_resource") == 42


class TestGenericOriginsAreInjected:
    """Walking a hint's arguments without considering its *origin* left the
    generic's own class uninjected, while the prompt still advertised it — so
    generated code that used the name got a NameError."""

    def test_a_custom_generic_injects_its_origin(self):
        class Cache(dict):
            pass

        def get_cache() -> Cache[str]: ...

        runtime = IPythonRuntime(functions=[Function(get_cache)])

        assert list(runtime._types) == ["Cache"]

    async def test_the_injected_origin_is_usable_in_generated_code(self):
        class Registry(dict):
            pass

        def build() -> Registry[str]: ...

        runtime = IPythonRuntime(functions=[Function(build)])
        result = await runtime.execute("made = Registry(a=1)\nprint(type(made).__name__)")

        assert result.success
        assert "Registry" in result.stdout

    def test_a_builtin_generic_origin_is_not_injected(self):
        def counts() -> dict[str, int]: ...

        assert list(IPythonRuntime(functions=[Function(counts)])._types) == []


class TestTypingConstructsAreNotUserTypes:
    """`typing.Any` is a class since 3.11 and `get_origin(int | None)` is
    `types.UnionType`, so both were injected as if the caller had defined
    them — and since names are claimed across every registry, the resulting
    `Any` entry then rejected the caller's own `Variable("Any", …)`."""

    def test_any_is_not_registered_as_a_type(self):
        from typing import Any

        def handle(x: Any) -> Any: ...

        assert list(IPythonRuntime(functions=[Function(handle)])._types) == []

    def test_a_pep604_union_does_not_register_its_origin(self):
        def maybe() -> int | None: ...

        assert list(IPythonRuntime(functions=[Function(maybe)])._types) == []

    def test_an_annotation_does_not_block_a_later_registration(self):
        from typing import Any

        def handle(x: Any) -> Any: ...

        runtime = IPythonRuntime(functions=[Function(handle)])
        runtime.inject_variable(Variable("Any", 1))

        assert runtime._variables["Any"].value == 1

    def test_a_user_class_defined_by_exec_is_still_injected(self):
        """It reports ``__module__ == "builtins"``, so excluding that module
        wholesale — rather than the specific builtin types — silently dropped
        genuine user classes."""
        namespace = {}
        exec(
            "class Payload:\n    pass\ndef consume(value: Payload) -> Payload:\n    return value\n",
            namespace,
        )

        runtime = IPythonRuntime(functions=[Function(namespace["consume"])])

        assert list(runtime._types) == ["Payload"]
