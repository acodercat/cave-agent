import pytest
from dataclasses import dataclass
from enum import Enum
from cave_agent.python_runtime import PythonRuntime, Variable, Function, Type


@pytest.fixture
def simple_runtime():
    return PythonRuntime()


@pytest.fixture
def runtime_with_data():
    numbers_var = Variable(
        name="numbers",
        value=[3, 1, 4, 1, 5, 9, 2, 6, 5],
        description="List of numbers to process"
    )
    
    result_var = Variable(
        name="result",
        description="Store calculation results here"
    )
    
    return PythonRuntime(variables=[numbers_var, result_var])


@pytest.fixture
def runtime_with_function():
    def multiply(a, b):
        """Multiply two numbers"""
        return a * b
    
    func = Function(multiply, "Multiplication function")
    return PythonRuntime(functions=[func])


@pytest.mark.asyncio
async def test_basic_execution(simple_runtime):
    """Test basic code execution works"""
    await simple_runtime.execute("x = 5 + 3")
    result = simple_runtime._executor.get_from_namespace('x')
    assert result == 8


@pytest.mark.asyncio
async def test_print_output(simple_runtime):
    """Test code with print output"""
    output = await simple_runtime.execute("print('Hello World')")
    assert "Hello World" in output.stdout


@pytest.mark.asyncio
async def test_variable_usage(runtime_with_data):
    """Test using injected variables"""
    await runtime_with_data.execute("result = sum(numbers)")
    total = runtime_with_data.get_variable('result')
    assert total == 36


@pytest.mark.asyncio
async def test_function_usage(runtime_with_function):
    """Test using injected functions"""
    await runtime_with_function.execute("result = multiply(6, 7)")
    result = runtime_with_function._executor.get_from_namespace('result')
    assert result == 42

@pytest.mark.asyncio
async def test_multiple_executions(simple_runtime):
    """Test multiple code executions share state"""
    await simple_runtime.execute("a = 10")
    await simple_runtime.execute("b = a * 2")
    await simple_runtime.execute("c = a + b")
    
    result = simple_runtime._executor.get_from_namespace('c')
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


# =============================================================================
# Type Class Tests
# =============================================================================

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
        runtime = PythonRuntime(types=[Type(Light)])
        assert "Light" in runtime._types

    def test_inject_type_via_method(self):
        """Types can be injected via inject_type method."""
        runtime = PythonRuntime()
        runtime.inject_type(Type(Light))
        assert "Light" in runtime._types

    def test_inject_multiple_types(self):
        """Multiple types can be injected."""
        runtime = PythonRuntime(types=[
            Type(Light),
            Type(Lock),
        ])
        assert "Light" in runtime._types
        assert "Lock" in runtime._types

    def test_inject_duplicate_type_raises(self):
        """Injecting duplicate type raises error."""
        runtime = PythonRuntime(types=[Type(Light)])
        with pytest.raises(ValueError, match="already exists"):
            runtime.inject_type(Type(Light))

    def test_type_available_in_namespace(self):
        """Injected type is available in execution namespace."""
        runtime = PythonRuntime(types=[Type(Light)])
        assert runtime._executor.get_from_namespace("Light") is Light


@pytest.mark.asyncio
class TestTypeExecution:
    """Test using injected types in code execution."""

    async def test_isinstance_check(self):
        """Can use isinstance with injected type (auto-injected from Variable)."""
        light = Light("Kitchen")
        runtime = PythonRuntime(
            variables=[Variable("device", light, "A device")],
        )

        # Light is auto-injected when Variable is injected
        assert "Light" in runtime._types

        result = await runtime.execute("result = isinstance(device, Light)")
        assert runtime._executor.get_from_namespace("result") is True

    async def test_isinstance_check_negative(self):
        """isinstance returns False for non-matching type."""
        light = Light("Kitchen")
        runtime = PythonRuntime(
            variables=[Variable("device", light, "A device")],
            types=[Type(Lock)],  # Only Lock needs explicit injection
        )

        result = await runtime.execute("result = isinstance(device, Lock)")
        assert runtime._executor.get_from_namespace("result") is False

    async def test_instantiate_type(self):
        """Can instantiate injected type."""
        runtime = PythonRuntime(types=[Type(Light)])

        await runtime.execute("light = Light('Bedroom')")
        light = runtime._executor.get_from_namespace("light")
        assert isinstance(light, Light)
        assert light.name == "Bedroom"

    async def test_call_method_on_new_instance(self):
        """Can call methods on newly created instance."""
        runtime = PythonRuntime(types=[Type(Light)])

        await runtime.execute("""
light = Light('Bedroom')
result = light.turn_on()
""")
        result = runtime._executor.get_from_namespace("result")
        assert result == "Bedroom turned on"

        light = runtime._executor.get_from_namespace("light")
        assert light.is_on is True

    async def test_filter_by_type(self):
        """Can filter list by type using isinstance."""
        devices = [Light("Kitchen"), Lock("Front"), Light("Bedroom")]
        runtime = PythonRuntime(
            variables=[Variable("devices", devices, "List of devices")],
            types=[Type(Light), Type(Lock)],
        )

        await runtime.execute("lights = [d for d in devices if isinstance(d, Light)]")
        lights = runtime._executor.get_from_namespace("lights")
        assert len(lights) == 2
        assert all(isinstance(l, Light) for l in lights)

    async def test_type_with_dataclass(self):
        """Can use dataclass as injected type."""
        runtime = PythonRuntime(types=[Type(DataPoint)])

        await runtime.execute("point = DataPoint(x=1.0, y=2.0)")
        point = runtime._executor.get_from_namespace("point")
        assert point.x == 1.0
        assert point.y == 2.0

    async def test_type_with_enum(self):
        """Can use enum as injected type."""
        runtime = PythonRuntime(types=[Type(Priority)])

        await runtime.execute("p = Priority.HIGH")
        p = runtime._executor.get_from_namespace("p")
        assert p == Priority.HIGH
        assert p.value == 3


class TestTypeDescribeTypes:
    """Test Type integration with describe_types()."""

    def test_type_schema_in_describe_types(self):
        """Injected Type schema appears in describe_types()."""
        runtime = PythonRuntime(types=[Type(Light)])
        result = runtime.describe_types()

        assert "Light:" in result
        assert "doc: A smart light device." in result
        assert "methods:" in result
        assert "turn_on()" in result
        assert "turn_off()" in result

    def test_type_schema_without_doc(self):
        """Type with include_doc=False excludes docstring."""
        runtime = PythonRuntime(types=[
            Type(Light, include_schema=True, include_doc=False)
        ])
        result = runtime.describe_types()

        assert "Light:" in result
        assert "doc:" not in result
        assert "methods:" in result
        assert "turn_on()" in result

    def test_type_doc_only(self):
        """Type with include_schema=False shows doc only."""
        runtime = PythonRuntime(types=[
            Type(Light, include_schema=False, include_doc=True)
        ])
        result = runtime.describe_types()

        assert "Light:" in result
        assert "doc: A smart light device." in result
        assert "methods:" not in result

    def test_type_no_schema_no_doc(self):
        """Type with both False shows nothing in describe_types()."""
        runtime = PythonRuntime(types=[
            Type(Light, include_schema=False, include_doc=False)
        ])
        result = runtime.describe_types()

        assert result == "No types available"

    def test_type_dataclass_schema(self):
        """Dataclass type shows fields in schema."""
        runtime = PythonRuntime(types=[Type(DataPoint)])
        result = runtime.describe_types()

        assert "DataPoint:" in result
        assert "fields:" in result
        assert "x: float" in result
        assert "y: float" in result

    def test_type_enum_schema(self):
        """Enum type shows values in schema."""
        runtime = PythonRuntime(types=[Type(Priority)])
        result = runtime.describe_types()

        assert "Priority (Enum):" in result
        assert "LOW = 1" in result
        assert "MEDIUM = 2" in result
        assert "HIGH = 3" in result

    def test_multiple_types_in_describe_types(self):
        """Multiple types appear in describe_types()."""
        runtime = PythonRuntime(types=[
            Type(Light),
            Type(Lock),
        ])
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
        runtime = PythonRuntime(variables=[Variable("light", light, "A light")])

        # Type is auto-injected with schema=False
        assert "Light" in runtime._types
        assert runtime._types["Light"].include_schema is False
        assert runtime._types["Light"].include_doc is False

        # Schema NOT shown
        result = runtime.describe_types()
        assert "Light:" not in result

    def test_variable_does_not_inject_builtins(self):
        """Built-in types are not auto-injected."""
        runtime = PythonRuntime(variables=[
            Variable("numbers", [1, 2, 3], "A list"),
            Variable("name", "hello", "A string"),
            Variable("count", 42, "An int"),
        ])

        # No types should be auto-injected
        assert len(runtime._types) == 0

    def test_function_auto_inject_schema_hidden(self):
        """Function auto-injects types with schema hidden."""
        def process(device: Light) -> str:
            return "done"

        runtime = PythonRuntime(functions=[Function(process)])

        # Type is auto-injected with schema=False
        assert "Light" in runtime._types
        assert runtime._types["Light"].include_schema is False
        assert runtime._types["Light"].include_doc is False

        # Schema NOT shown
        result = runtime.describe_types()
        assert "Light:" not in result

    def test_function_auto_injects_return_type(self):
        """Injecting a Function auto-injects return type."""
        def create_lock() -> Lock:
            return Lock()

        runtime = PythonRuntime(functions=[Function(create_lock)])

        assert "Lock" in runtime._types
        assert runtime._executor.get_from_namespace("Lock") is Lock

    def test_function_auto_injects_multiple_types(self):
        """Function with multiple custom types injects all."""
        def transfer(source: Light, target: Lock) -> DataPoint:
            pass

        runtime = PythonRuntime(functions=[Function(transfer)])

        assert "Light" in runtime._types
        assert "Lock" in runtime._types
        assert "DataPoint" in runtime._types

    def test_function_does_not_inject_builtins(self):
        """Function with only built-in types doesn't inject anything."""
        def process(items: list, count: int) -> str:
            pass

        runtime = PythonRuntime(functions=[Function(process)])

        assert len(runtime._types) == 0

    def test_auto_inject_skips_duplicates(self):
        """Auto-injection skips types already injected."""
        light1 = Light("Kitchen")
        light2 = Light("Bedroom")

        # Both variables have same type - should not raise
        runtime = PythonRuntime(variables=[
            Variable("light1", light1, "Light 1"),
            Variable("light2", light2, "Light 2"),
        ])

        # Only one Light type should exist
        assert "Light" in runtime._types
        assert len(runtime._types) == 1

    def test_explicit_type_takes_precedence(self):
        """Explicitly injected Type takes precedence over auto-injection."""
        light = Light("Kitchen")
        runtime = PythonRuntime(
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
        runtime1 = PythonRuntime(functions=[Function(process)])
        assert runtime1.describe_types() == "No types available"

        # With explicit Type, schema is shown
        runtime2 = PythonRuntime(
            types=[Type(Light)],
            functions=[Function(process)],
        )
        result = runtime2.describe_types()
        assert "Light:" in result
        assert "turn_on()" in result


class TestTypeReset:
    """Test Type behavior with runtime reset."""

    def test_reset_clears_types(self):
        """Reset clears injected types."""
        runtime = PythonRuntime(types=[Type(Light)])
        assert "Light" in runtime._types

        runtime.reset()
        assert len(runtime._types) == 0

    def test_reset_clears_type_from_namespace(self):
        """Reset removes type from execution namespace."""
        runtime = PythonRuntime(types=[Type(Light)])
        assert runtime._executor.get_from_namespace("Light") is Light

        runtime.reset()
        assert runtime._executor.get_from_namespace("Light") is None
