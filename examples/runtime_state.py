from cave_agent import CaveAgent
from cave_agent.models import OpenAIServerModel
from cave_agent.python_runtime import PythonRuntime, Variable
import os
import asyncio

# Initialize LLM engine
model = OpenAIServerModel(
    model_id=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

async def main():
    # Define input data
    numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    
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
    
    # Create agent
    agent = CaveAgent(model, runtime=runtime)
    
    # Sort numbers and get result
    await agent.run("Sort the numbers list")
    sorted_result = runtime.get_variable_value('sorted_numbers')
    print("Sorted numbers:", sorted_result)
    
    # Calculate sum and get result
    await agent.run("Calculate the sum of all numbers")
    total = runtime.get_variable_value('sum_result')
    print("Sum:", total)

if __name__ == "__main__":
    asyncio.run(main())