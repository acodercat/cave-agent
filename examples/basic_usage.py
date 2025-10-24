from cave_agent import CaveAgent
from cave_agent.models import LiteLLMModel
from cave_agent.python_runtime import PythonRuntime, Function, Variable
import os
import asyncio


model = LiteLLMModel(
    model_id=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    custom_llm_provider='openai'
)

# Define some tool functions
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers together"""
    return a * b

def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers"""
    print(f"Calculating the sum of {a} and {b}")
    return a + b

# Define tools and context
class DataProcessor:
    """A data processor object that can sort lists of numbers"""
    def process(self, data: list) -> list:
        """Sort a list of numbers"""
        return sorted(data)

async def main():
    # usage 1: Simple calculations with functions
    print("=== usage 1: Simple Calculations ===")
    
    # Create function objects
    add_func = Function(add)
    multiply_func = Function(multiply)
    
    # Create runtime with functions
    runtime1 = PythonRuntime(
        functions=[add_func, multiply_func]
    )
    
    # Create agent
    agent1 = CaveAgent(model, runtime=runtime1)
    
    # Run simple calculations
    response = await agent1.run("Calculate 5 plus 3")
    print("Result:", response.content)
    
    response = await agent1.run("What is 4 times 6?")
    print("Result:", response.content)
    
    print("\n=== usage 2: Object Processing ===")
    
    # Create processor and data
    processor = DataProcessor()
    numbers = [3, 1, 4, 1, 5, 9]
    
    # Create function and variable objects
    calc_sum_func = Function(calculate_sum)
    processor_var = Variable(
        name="processor",
        value=processor,
        description="A data processor object that can sort lists of numbers \n usage: result = processor.process([3, 1, 4])"
    )
    numbers_var = Variable(
        name="numbers",
        value=numbers,
        description="Input list of numbers to be processed \n usage: print(numbers)  # Access the list directly"
    )
    result_var = Variable(
        name="result",
        description="Store the result of the processing in this variable. \n usage: print(result)  # Access the result directly"
    )
    
    # Create runtime with functions and variables
    runtime2 = PythonRuntime(
        functions=[calc_sum_func],
        variables=[processor_var, numbers_var, result_var]
    )
    
    # Create agent
    agent2 = CaveAgent(model, runtime=runtime2)
    
    # Run task using injected objects
    response = await agent2.run("Use processor to sort the numbers and store the result in the 'result' variable")
    print("Response:", response)
    
    # Retrieve results from Python environment
    sorted_result = agent2.runtime.get_variable_value('result')
    print("Sorted result:", sorted_result)  # [1, 1, 3, 4, 5, 9]
    
    print("\n=== usage 3: Streaming Events ===")
    
    # Create a simple runtime for streaming usage
    runtime3 = PythonRuntime(
        functions=[Function(add), Function(multiply)]
    )
    agent3 = CaveAgent(model, runtime=runtime3)
    
    # Stream events
    async for event in agent3.stream_events("Calculate 10 plus 20, then multiply the result by 3"):
        print(f"[{event.type.value}] {event.content}")

if __name__ == "__main__":
    asyncio.run(main())