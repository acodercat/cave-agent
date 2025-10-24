from cave_agent import CaveAgent
from cave_agent.python_runtime import PythonRuntime, Function
import pytest

@pytest.fixture
def basic_agent(model):
    def add(a: int, b: int) -> int:
        """Add two numbers together"""
        return a + b

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together"""
        return a * b

    # Create Function objects
    add_func = Function(add)
    multiply_func = Function(multiply)
    
    # Create runtime with functions
    runtime = PythonRuntime(
        functions=[add_func, multiply_func]
    )
    
    return CaveAgent(
        model,
        runtime=runtime
    )

@pytest.mark.asyncio
async def test_basic_calculation(basic_agent):
    response = await basic_agent.run("Calculate 5 plus 3")
    assert any(str(8) in part.lower() for part in response.content.split())

@pytest.mark.asyncio
async def test_multiple_calculations(basic_agent):
    response1 = await basic_agent.run("Calculate 4 times 6")
    assert any(str(24) in part.lower() for part in response1.content.split())
    
    response2 = await basic_agent.run("Add 10 and 20")
    assert any(str(30) in part.lower() for part in response2.content.split()) 