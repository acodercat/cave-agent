import pytest
from dataclasses import dataclass
from cave_agent import CaveAgent
from cave_agent.python_runtime import PythonRuntime, Variable

@dataclass
class DataProcessor:
    """A utility class for processing and filtering data collections.
    
    This class provides methods for basic data processing operations such as
    sorting, removing duplicates, and filtering based on thresholds.
    
    Example:
        >>> processor = DataProcessor()
        >>> processor.process_list([3, 1, 2, 1, 3])
        [1, 2, 3]
        >>> processor.filter_numbers([1, 5, 3, 8, 2], 4)
        [5, 8]
    """
    def process_list(self, data: list) -> list:
        """Sort a list and remove duplicates"""
        return sorted(set(data))
    
    def filter_numbers(self, data: list, threshold: int) -> list:
        """Filter numbers greater than threshold"""
        return [x for x in data if x > threshold]
    
@pytest.fixture
def processor():
    return DataProcessor()

@pytest.fixture
def numbers():
    return [3, 1, 4, 1, 5, 9, 2, 6, 5]

@pytest.fixture
def object_agent(model, processor, numbers):
    # Create Variable objects
    processor_var = Variable(
        name="processor",
        value=processor,
        description="Data processing tool with various methods\nusage: processed_data = processor.process_list(numbers)"
    )
    
    numbers_var = Variable(
        name="numbers",
        value=numbers,
        description="Input list of numbers\nusage: filtered_data = processor.filter_numbers(numbers, 5)"
    )
    
    processed_data_var = Variable(
        name="processed_data",
        description="Store processed data in this variable\nusage: processed_data = processor.process_list(numbers)"
    )
    
    filtered_data_var = Variable(
        name="filtered_data",
        description="Store filtered data in this variable\nusage: filtered_data = processor.filter_numbers(numbers, 5)"
    )
    
    # Create runtime
    runtime = PythonRuntime(
        variables=[processor_var, numbers_var, processed_data_var, filtered_data_var]
    )
    
    return CaveAgent(
        model,
        runtime=runtime
    )

@pytest.mark.asyncio
async def test_process_and_deduplicate(object_agent):
    await object_agent.run("Use processor to sort and deduplicate numbers")
    processed_data = object_agent.runtime.get_variable_value('processed_data')
    expected = [1, 2, 3, 4, 5, 6, 9]
    assert sorted(set(processed_data)) == sorted(set(expected))
    

@pytest.mark.asyncio
async def test_filter_numbers(object_agent):
    await object_agent.run("Filter numbers greater than 4")
    filtered_data = object_agent.runtime.get_variable_value('filtered_data')
    expected = [5, 6, 9]
    assert sorted(set(filtered_data)) == sorted(set(expected))