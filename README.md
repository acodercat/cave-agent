<p align="center">
  <img src="https://github.com/acodercat/cave-agent/raw/master/banner.png" alt="CaveAgent">
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://pypi.org/project/cave-agent"><img src="https://img.shields.io/badge/pypi-0.6.0-blue.svg" alt="PyPI version"></a>
</p>

<p align="center">
  <em>"When your AI needs to run code, not just write it"</em>
</p>

---

Traditional LLM agents operate under a **text-in-text-out** paradigm, with tool interactions constrained to JSON primitives. CaveAgent breaks this limitation by natively accepting, manipulating, and outputting complex Python objects—such as `DataFrames`, `ndarrays`, and custom instances—within a persistent runtime, enabling **precise computation** and **lossless data flow** across multi-turn interactions.


## Why CaveAgent?

- 🧠 **Native Code Generation** - LLMs excel at writing code, not parsing JSON
- ⚡ **Fewer Iterations** - Execute complex multi-step workflows in a single turn
- 🔄 **Persistent State** — DataFrames, graphs, and objects survive across turns
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
    print(f"Current tasks: {runtime.get_variable('tasks')}")

    await agent.run("Mark groceries done and remind me about mom")
    print(f"Final state: {runtime.get_variable('tasks')}")

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
from cave_agent.python_runtime import PythonRuntime, Variable, Type

async def main():
    # Initialize LLM model
    model = LiteLLMModel(
        model_id="your-model",
        api_key="your-api-key",
        base_url="your-base-url"
    )

    # Define smart home device classes
    class Light:
        """A smart light with brightness control."""
        def __init__(self, name: str, is_on: bool = False, brightness: int = 100):
            self.name = name
            self.is_on = is_on
            self.brightness = brightness

        def turn_on(self):
            """Turn the light on."""
            self.is_on = True

        def turn_off(self):
            """Turn the light off."""
            self.is_on = False

        def set_brightness(self, brightness: int):
            """Set brightness level (0-100)."""
            self.brightness = max(0, min(100, brightness))
            if self.brightness > 0:
                self.is_on = True

    class Thermostat:
        """A smart thermostat."""
        def __init__(self, current_temp: int = 20, target_temp: int = 20):
            self.current_temp = current_temp
            self.target_temp = target_temp

        def set_temperature(self, temp: int):
            """Set target temperature."""
            self.target_temp = temp

    # Create device instances
    living_room_light = Light("Living Room", is_on=True, brightness=80)
    bedroom_light = Light("Bedroom", is_on=False)
    thermostat = Thermostat(current_temp=20, target_temp=20)

    # Create runtime with variables and type schemas
    runtime = PythonRuntime(
        types=[
            Type(Light),
            Type(Thermostat),
        ],
        variables=[
            Variable("living_room_light", living_room_light, "Smart light in living room"),
            Variable("bedroom_light", bedroom_light, "Smart light in bedroom"),
            Variable("thermostat", thermostat, "Home thermostat"),
        ],
    )

    # Create agent
    agent = CaveAgent(model, runtime=runtime)

    # Control smart home - LLM can manipulate objects directly
    await agent.run("Dim the living room light to 20% and set thermostat to 22°C")

    # Validate the changes by getting variables from runtime
    light = runtime.get_variable("living_room_light")
    thermostat = runtime.get_variable("thermostat")

    print(f"Living room light: {light.brightness}% brightness, {'ON' if light.is_on else 'OFF'}")
    print(f"Thermostat: {thermostat.target_temp}°C")

if __name__ == "__main__":
    asyncio.run(main())
```

### Real-time Streaming

Watch your AI think and execute code in real-time:

```python
async for event in agent.stream_events("Analyze this data and create a summary"):
    if event.type.value == 'code':
        print(f"🔧 Executing: {event.content}")
    elif event.type.value == 'execution_output':
        print(f"✅ Result: {event.content}")
    elif event.type.value == 'text':
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

### Type Injection

Types from Variables and Functions are automatically injected into the runtime namespace, allowing the LLM to use `isinstance()` checks and create new instances. By default, type schemas are hidden from the prompt.

To expose type information to the LLM, use explicit `Type` injection:

```python
from cave_agent.python_runtime import PythonRuntime, Variable, Type

class Light:
    """A smart light device."""
    def turn_on(self) -> str: ...
    def turn_off(self) -> str: ...

light = Light()

# Types auto-injected but schema hidden (default)
runtime = PythonRuntime(
    variables=[Variable("light", light, "A smart light")],
)

# Explicitly show type schema in <types> section
runtime = PythonRuntime(
    types=[Type(Light)],  # Schema shown by default
    variables=[Variable("light", light, "A smart light")],
)

# Control schema and doc separately
runtime = PythonRuntime(
    types=[
        Type(Light, include_schema=True, include_doc=False),  # Methods only
        Type(Lock, include_schema=False, include_doc=True),   # Docstring only
    ],
    variables=[...],
)
```

When enabled, types appear in a dedicated `<types>` section:
```
<types>
Light:
  doc: A smart light device.
  methods:
    - turn_on() -> str
    - turn_off() -> str
</types>
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
- **📋 Type Injection**: Auto-inject types for `isinstance()` checks; explicit Type injection to expose schemas to the LLM

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
