import pytest
from cave_agent import CaveAgent
from cave_agent.python_runtime import PythonRuntime, Variable

@pytest.fixture
def numbers():
    return [3, 1, 4, 1, 5, 9, 2, 6, 5]

@pytest.fixture
def state_agent(model, numbers):
    # Create Variable objects
    numbers_var = Variable(
        name="numbers",
        value=numbers,
        description="List of numbers to process\nusage: print(numbers)"
    )
    
    sorted_numbers_var = Variable(
        name="sorted_numbers",
        description="Store the sorted numbers in this variable\nusage: sorted_numbers = sorted(numbers)"
    )
    
    sum_result_var = Variable(
        name="sum_result",
        description="Store the sum of numbers in this variable\nusage: sum_result = sum(numbers)"
    )
    
    # Create runtime
    runtime = PythonRuntime(
        variables=[numbers_var, sorted_numbers_var, sum_result_var]
    )
    
    return CaveAgent(
        model,
        runtime=runtime
    )

@pytest.mark.asyncio
async def test_sort_numbers(state_agent):
    await state_agent.run("Sort the numbers list")
    sorted_result = state_agent.runtime.get_variable_value('sorted_numbers')
    assert sorted_result == sorted([1, 1, 2, 3, 4, 5, 5, 6, 9])

@pytest.mark.asyncio
async def test_calculate_sum(state_agent):
    await state_agent.run("Calculate the sum of all numbers")
    total = state_agent.runtime.get_variable_value('sum_result')
    assert total == 36  # sum of [3,1,4,1,5,9,2,6,5]