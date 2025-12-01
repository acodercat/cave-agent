# 🤖 CaveAgent
**🚀 AI that executes, not just generates!**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/badge/pypi-0.4.4-blue.svg)](https://pypi.org/project/cave-agent)

CaveAgent is a tool-augmented agent framework that enables function-calling through LLM code generation and provides runtime state management. Unlike traditional JSON-schema approaches, it leverages LLM's inherent coding capabilities to interact with tools through a Python runtime environment, allowing direct access to execution results and runtime state.

> *"When your AI needs to run code, not just write it"*

## Why CaveAgent?

**Traditional function calling is broken.** JSON schemas are rigid, error-prone, and limit what your AI can do. CaveAgent unleashes your LLM's natural coding abilities:

- 🧠 **Native Code Generation** - LLMs excel at writing code, not parsing JSON
- ⚡ **Fewer Iterations** - Execute complex multi-step workflows in a single turn
- 🔄 **Persistent State** - Maintain variables and objects across conversations  
- 🎯 **Maximum Flexibility** - Handle dynamic workflows that JSON schemas can't express
- 🛡️ **Secure by Design** - AST validation prevents dangerous code execution
- 📡 **Real-time Streaming** - Watch your AI think and execute in real-time
- 🌐 **Universal LLM Support** - Works with OpenAI, Anthropic, Google, and 100+ providers

## Quick Start

```bash
pip install 'cave-agent[all]'
```

Choose your installation:

```bash
# OpenAI support
pip install 'cave-agent[openai]'

# 100+ LLM providers via LiteLLM 
pip install 'cave-agent[litellm]'
```

### Simple Function Calling

```python
import asyncio
from cave_agent import CaveAgent
from cave_agent.models import OpenAIServerModel
from cave_agent.python_runtime import PythonRuntime, Function, Variable

async def main():
    # Initialize LLM model
    model = OpenAIServerModel(
        model_id="your-model",
        api_key="your-api-key",
        base_url="your-base-url"
    )

    # Define tool functions
    def add_task(task_name: str) -> str:
        """Add a new task to the task list"""
        tasks.append({"name": task_name, "done": False})
        return f"Added task: {task_name}"

    def complete_task(task_name: str) -> str:
        """Mark a task as completed"""
        for task in tasks:
            if task_name.lower() in task["name"].lower():
                task["done"] = True
                return f"Completed: {task['name']}"
        return f"Task '{task_name}' not found"

    def send_reminder(message: str) -> str:
        """Send a reminder notification"""
        return f"Reminder: {message}"

    # Initialize data
    tasks = []

    # Setup Runtime
    runtime = PythonRuntime(
        variables=[
            Variable("tasks", tasks, "List of user's tasks. Example: [{'name': 'walk the dog', 'done': False}]")
        ],
        functions=[
            Function(add_task),
            Function(complete_task), 
            Function(send_reminder)
        ]
    )

    agent = CaveAgent(model, runtime=runtime)

    await agent.run("Add buy groceries and call mom to my tasks")
    print(f"Current tasks: {runtime.get_variable_value('tasks')}")

    await agent.run("Mark groceries done and remind me about mom")
    print(f"Final state: {runtime.get_variable_value('tasks')}")

    response = await agent.run("What's my progress?")
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced: Stateful Object Interactions

```python
import asyncio
from cave_agent import CaveAgent
from cave_agent.models import LiteLLMModel
from cave_agent.python_runtime import PythonRuntime, Function, Variable

async def main():
    # Initialize LLM model
    model = LiteLLMModel(
        model_id="your-model",
        api_key="your-api-key",
        base_url="your-base-url"
    )

    # Define a class with methods
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

    # Prepare context
    processor = DataProcessor()
    numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5]

    # Create runtime with variables and functions
    runtime = PythonRuntime(
        variables=[
            Variable(
                name="processor",
                value=processor,
                description="Data processing tool with various methods"
            ),
            Variable(
                name="numbers",
                value=numbers,
                description="Input list of numbers"
            ),
            Variable(
                name="processed_data",
                description="Store processed data in this variable"
            ),
            Variable(
                name="filtered_data",
                description="Store filtered data in this variable"
            )
        ]
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
```

### Real-time Streaming

Watch your AI think and execute code in real-time:

```python
async for event in agent.stream_events("Analyze this data and create a summary"):
    if event.type.value == 'CODE':
        print(f"🔧 Executing: {event.content}")
    elif event.type.value == 'EXECUTION_RESULT':
        print(f"✅ Result: {event.content}")
    elif event.type.value == 'TEXT':
        print(event.content, end="", flush=True)
```

### Security Features

CaveAgent includes rule-based security to prevent dangerous code execution:

```python
import asyncio
from cave_agent import CaveAgent
from cave_agent.models import OpenAIServerModel
from cave_agent.python_runtime import PythonRuntime
from cave_agent.security_checker import (
    SecurityChecker, ImportRule, FunctionRule, AttributeRule, RegexRule
)

async def main():
    model = OpenAIServerModel(
        model_id="gpt-4",
        api_key="your-api-key",
        base_url="https://api.openai.com/v1"
    )

    # Configure security with specific rules
    rules = [
        ImportRule({"os", "subprocess", "sys", "socket"}),  # Block dangerous imports
        FunctionRule({"eval", "exec", "compile", "open"}),  # Block dangerous functions
        AttributeRule({"__globals__", "__builtins__"}),     # Block attribute access
        RegexRule("no_print", "Block print statements", r"print\s*\(")  # Custom regex
    ]
    
    checker = SecurityChecker(rules)
    runtime = PythonRuntime(security_checker=checker)
    
    agent = CaveAgent(model, runtime=runtime)
    
    # This will be blocked by security
    try:
        await agent.run("import os and list files")
    except Exception as e:
        print(f"Blocked: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Features

- **🤖 Code-Based Function Calling**: Leverages LLM's natural coding abilities instead of rigid JSON schemas
- **🛡️ Secure Runtime Environment**: 
  - Inject Python objects, variables, and functions as tools
  - Rule-based security validation prevents dangerous code execution
  - Flexible security rules: ImportRule, FunctionRule, AttributeRule, RegexRule
  - Customizable security policies for different use cases
  - Access execution results and maintain state across interactions
- **💬 Multi-Turn Conversations**: Persistent context and runtime state across multiple interactions
- **⚡ Streaming & Async**: Real-time event streaming and full async/await support for optimal performance
- **🛡️ Execution Control**: Configurable step limits and error handling to prevent infinite loops
- **🎯 Unmatched Flexibility**: JSON schemas break with dynamic workflows. Python code adapts to any situation - conditional logic, loops, and complex data transformations.
- **🌐 Flexible LLM Support**: Works with any LLM provider via OpenAI-compatible APIs or LiteLLM

## Real-World Examples

For more examples, check out the [examples](examples) directory:

- [Basic Usage](examples/basic_usage.py): Simple function calling and object processing
- [Runtime State](examples/runtime_state.py): Managing runtime state across interactions
- [Object Methods](examples/object_methods.py): Using class methods and complex objects
- [Multi-Turn](examples/multi_turn.py): Complex analysis conversations with state persistence
- [Stream](examples/stream.py): Streaming responses and execution events

## LLM Provider Support

CaveAgent supports multiple LLM providers:

### OpenAI-Compatible Models
```python
from cave_agent.models import OpenAIServerModel

model = OpenAIServerModel(
    model_id="gpt-4",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"  # or your custom endpoint
)
```

### LiteLLM Models (Recommended)
LiteLLM provides unified access to hundreds of LLM providers:

```python
from cave_agent.models import LiteLLMModel

# OpenAI
model = LiteLLMModel(
    model_id="gpt-4",
    api_key="your-api-key"，
    custom_llm_provider='openai'
)

# Anthropic Claude
model = LiteLLMModel(
    model_id="claude-3-sonnet-20240229",
    api_key="your-api-key",
    custom_llm_provider='anthropic' 
)

# Google Gemini
model = LiteLLMModel(
    model_id="gemini/gemini-pro",
    api_key="your-api-key"
)
```


## Contributing

Contributions are welcome! Please feel free to submit a PR.
For more details, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE) for details.
