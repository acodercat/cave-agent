<p align="center">
  <img src="https://github.com/acodercat/cave-agent/raw/master/banner.png" alt="CaveAgent">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.01569"><img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://pypi.org/project/cave-agent"><img src="https://img.shields.io/badge/pypi-0.6.2-blue.svg" alt="PyPI version"></a>
</p>

<p align="center">
  <em>"From text-in-text-out to (text&amp;object)-in-(text&amp;object)-out"</em>
</p>

---

Most LLM agents operate under a text-in-text-out paradigm, with tool interactions constrained to JSON primitives. CaveAgent breaks this with **Stateful Runtime Management**—a persistent Python runtime with direct **variable injection and retrieval**:

- **Inject** any Python object into the runtime—DataFrames, models, database connections, custom class instances—as first-class variables the LLM can manipulate
- **Persist** state across turns without serialization; objects live in the runtime, not in the context window
- **Retrieve** manipulated objects back as native Python types for downstream


**Paper**: [CaveAgent: Transforming LLMs into Stateful Runtime Operators](https://arxiv.org/abs/2601.01569) (Jan 2026)

## Table of Contents

- [Quick Start](#quick-start)
- [Examples](#examples)
- [Multi-Agent Coordination](#multi-agent-coordination)
- [Features](#features)
- [Awesome Blogs](#awesome-blogs)
- [Configuration](#configuration)
- [LLM Provider Support](#llm-provider-support)

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

## Examples

### Simple Function Calling

```python
import asyncio
from cave_agent import CaveAgent
from cave_agent.models import OpenAIServerModel
from cave_agent.runtime import PythonRuntime, Function, Variable

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
    print(f"Current tasks: {runtime.retrieve('tasks')}")

    await agent.run("Mark groceries done and remind me about mom")
    print(f"Final state: {runtime.retrieve('tasks')}")

    response = await agent.run("What's my progress?")
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### Stateful Object Interactions

```python
import asyncio
from cave_agent import CaveAgent
from cave_agent.models import LiteLLMModel
from cave_agent.runtime import PythonRuntime, Variable, Type

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
    light = runtime.retrieve("living_room_light")
    thermostat = runtime.retrieve("thermostat")

    print(f"Living room light: {light.brightness}% brightness, {'ON' if light.is_on else 'OFF'}")
    print(f"Thermostat: {thermostat.target_temp}°C")

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Agent Coordination

```python
# Sub-agents with their own runtimes
cleaner_agent = CaveAgent(model, runtime=PythonRuntime(variables=[
    Variable("data", [], "Input"), Variable("cleaned_data", [], "Output"),
]))

analyzer_agent = CaveAgent(model, runtime=PythonRuntime(variables=[
    Variable("data", [], "Input"), Variable("insights", {}, "Output"),
]))

# Orchestrator controls sub-agents as first-class objects
orchestrator = CaveAgent(model, runtime=PythonRuntime(variables=[
    Variable("raw_data", raw_data, "Raw dataset"),
    Variable("cleaner", cleaner_agent, "Cleaner agent"),
    Variable("analyzer", analyzer_agent, "Analyzer agent"),
]))

# Inject → trigger → retrieve
await orchestrator.run("Clean raw_data using cleaner, then analyze using analyzer")
insights = analyzer.runtime.retrieve("insights")
```

See [examples/multi_agent.py](examples/multi_agent.py) for a complete example.

### Real-time Streaming

Watch your AI think and execute code in real-time:

```python
async for event in agent.stream_events("Analyze this data and create a summary"):
    if event.type.value == 'code':
        print(f"Executing: {event.content}")
    elif event.type.value == 'execution_output':
        print(f"Result: {event.content}")
    elif event.type.value == 'text':
        print(event.content, end="", flush=True)
```

### Security Features

CaveAgent includes rule-based security to prevent dangerous code execution:

```python
import asyncio
from cave_agent import CaveAgent
from cave_agent.models import OpenAIServerModel
from cave_agent.runtime import PythonRuntime
from cave_agent.security import (
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
        RegexRule(r"print\s*\(", "Block print statements")  # Custom regex
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

To expose type information to the LLM, use explicit `Type` injection:

```python
from cave_agent.runtime import PythonRuntime, Variable, Type

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

### More Examples

For more examples, check out the [examples](examples) directory:

- [Basic Usage](examples/basic_usage.py): Simple function calling and object processing
- [Runtime State](examples/runtime_state.py): Managing runtime state across interactions
- [Object Methods](examples/object_methods.py): Using class methods and complex objects
- [Multi-Turn](examples/multi_turn.py): Complex analysis conversations with state persistence
- [Multi-Agent](examples/multi_agent.py): Data processing pipeline with multiple agents
- [Stream](examples/stream.py): Streaming responses and execution events

## Multi-Agent Coordination

CaveAgent introduces three foundational innovations for multi-agent coordination:

### Meta-Agent Runtime Control
Sub-agents are injected as first-class objects into an orchestrator's runtime. The orchestrator programmatically sets variables in sub-agent runtimes, triggers execution, and retrieves results—enabling adaptive pipelines, iterative loops, and conditional branching.

### State-Mediated Communication
Inter-agent data transfer bypasses message-passing entirely. Agents communicate through direct runtime variable injection—DataFrames, trained models, statistical analyses—preserving type fidelity without serialization loss.

### Shared-Runtime Synchronization
Multiple agents can operate on a unified runtime instance. When one agent modifies a shared object, all peers perceive the change immediately through direct reference. Zero coordination overhead.

## Features

- **Code-Based Function Calling**: Leverages LLM's natural coding abilities instead of rigid JSON schemas
- **Secure Runtime Environment**:
  - Inject Python objects, variables, and functions as tools
  - Rule-based security validation prevents dangerous code execution
  - Flexible security rules: ImportRule, FunctionRule, AttributeRule, RegexRule
  - Customizable security policies for different use cases
  - Access execution results and maintain state across interactions
- **Multi-Agent Coordination**: Control sub-agents programmatically through runtime injection and retrieval. Shared runtimes enable instant state synchronization.
- **Streaming & Async**: Real-time event streaming and full async/await support for optimal performance
- **Execution Control**: Configurable step limits and error handling to prevent infinite loops
- **Flexible LLM Support**: Works with any LLM provider via OpenAI-compatible APIs or LiteLLM
- **Type Injection**: Expose class schemas for type-aware LLM code generation


## Awesome Blogs

We thank these community to post our work.

- [CaveAgent让LLM学会了“跑代码”，你能把Agent变成Jupyter里的“老司机”](https://mp.weixin.qq.com/s/cJQ8ki0gXSmcbTPaBBfT5g)
- [Token消耗减半性能满分！状态化运行时管理能力让智能体性能飞升](https://mp.weixin.qq.com/s/qfVl3ATO4ueDdPb4npmTXQ)
- [Stateful environment to LLMs](https://x.com/rosinality/status/2008434433972728264)
- [TEKTA-AI](https://www.tekta.ai/ai-research-papers/caveagent-stateful-llm-runtime-2025)

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Model` | required | LLM model instance (OpenAIServerModel or LiteLLMModel) |
| `runtime` | `PythonRuntime` | `None` | Python runtime with variables, functions, and types |
| `max_steps` | `int` | `5` | Maximum execution steps per run |
| `max_history` | `int` | `10` | Maximum conversation history length |
| `max_execution_result_length` | `int` | `3000` | Max characters in execution output |
| `agent_identity` | `str` | default | Agent's role/identity description |
| `instructions` | `str` | default | Task instructions for the agent |
| `additional_context` | `str` | default | Extra context information |
| `system_prompt_template` | `str` | default | Custom system prompt template |
| `python_block_identifier` | `str` | `"python"` | Code block language identifier |
| `messages` | `List[Message]` | `None` | Initial message history |
| `log_level` | `LogLevel` | `DEBUG` | Logging verbosity level |

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
    api_key="your-api-key",
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

## Citation

If you use CaveAgent in your research, please cite:
```bibtex
@article{ran2026caveagent,
  title={CaveAgent: Transforming LLMs into Stateful Runtime Operators},
  author={Ran, Maohao and Wan, Zhenglin and Lin, Cooper and Zhang, Yanting and others},
  journal={arXiv preprint arXiv:2601.01569},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.