from cave_agent import CaveAgent, LogLevel, Logger
from cave_agent.models import OpenAIServerModel
from cave_agent.python_runtime import PythonRuntime, Variable
import os
import asyncio
from rich.syntax import Syntax
from rich.console import Console
from rich.text import Text

# Initialize LLM engine
model = OpenAIServerModel(
    model_id=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

# Define tools and context
class DataProcessor:
    """A data processor object that can sort lists of numbers"""
    def process(self, data: list) -> list:
        """Sort a list of numbers"""
        return sorted(data)

async def main():
    processor = DataProcessor()
    numbers = [3, 1, 4, 1, 5, 9]
    
    # Create Variable objects
    processor_var = Variable(
        name="processor",
        value=processor,
        description="A data processor object that can sort lists of numbers\nusage: result = processor.process([3, 1, 4])"
    )
    
    numbers_var = Variable(
        name="numbers",
        value=numbers,
        description="Input list of numbers to be processed\nusage: print(numbers)  # Access the list directly"
    )
    
    result_var = Variable(
        name="result",
        description="Store the result of the processing in this variable.\nusage: result = processor.process([3, 1, 4])"
    )
    
    # Create runtime
    runtime = PythonRuntime(
        variables=[processor_var, numbers_var, result_var]
    )
    
    # Create agent
    agent = CaveAgent(
        model,
        runtime=runtime,
        log_level=LogLevel.ERROR
    )
    
    logger = Logger(LogLevel.DEBUG)
    console = Console()
    
    logger.info("User Prompt", "Use processor to sort the numbers", 'yellow')
    
    async for event in agent.stream_events("Use processor to sort the numbers"):
        if event.type.name == 'TEXT':
            text = Text(event.content)
            text.stylize("cyan")
            console.print(text, end="", highlight=True)
        
        if event.type.name == 'CODE':
            logger.debug("Executing code", event.content, "cyan")
        
        if event.type.name == 'EXECUTION_RESULT':
            logger.info("Execution result", event.content, 'yellow')
        
        if event.type.name == 'EXECUTION_ERROR':
            logger.error("Execution error", event.content, 'yellow')

        if event.type.name == 'FINAL_RESPONSE':
            logger.info("Final response", event.content, 'green')
    
    print("\n")
    logger.info("Runtime State", str({
        'processor': agent.runtime.get_variable_value('processor'),
        'numbers': agent.runtime.get_variable_value('numbers'),
        'result': agent.runtime.get_variable_value('result')
    }), 'yellow')

if __name__ == "__main__":
    asyncio.run(main())