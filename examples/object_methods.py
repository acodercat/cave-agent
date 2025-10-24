from cave_agent import CaveAgent
from cave_agent.models import OpenAIServerModel
from cave_agent.python_runtime import PythonRuntime, Variable
from dataclasses import dataclass
import os
import asyncio

# Initialize LLM engine
model = OpenAIServerModel(
    model_id=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

# Define a class with methods
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

async def main():
    # Prepare context
    processor = DataProcessor()
    numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5]

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
        description="Store processed data here\nusage: processed_data = processor.process_list(numbers)"
    )
    
    filtered_data_var = Variable(
        name="filtered_data",
        description="Store filtered data here\nusage: filtered_data = processor.filter_numbers(numbers, 5)"
    )

    # Create runtime
    runtime = PythonRuntime(
        variables=[processor_var, numbers_var, processed_data_var, filtered_data_var]
    )

    # Create agent
    agent = CaveAgent(model, runtime=runtime)

    # Process data
    await agent.run("Use processor to sort and deduplicate numbers")
    processed_data = agent.runtime.get_variable_value('processed_data')
    print("Processed data:", processed_data)

    # Filter data
    await agent.run("Filter numbers greater than 4")
    filtered_data = agent.runtime.get_variable_value('filtered_data')
    print("Filtered data:", filtered_data)

if __name__ == "__main__":
    asyncio.run(main())