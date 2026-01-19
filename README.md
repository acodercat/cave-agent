<p align="center">
  <img src="https://github.com/acodercat/cave-agent/raw/master/banner.png" alt="CaveAgent">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.01569"><img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11+-blue?style=flat-square" alt="Python 3.11+"></a>
  <a href="https://pypi.org/project/cave-agent"><img src="https://img.shields.io/badge/PyPI-0.6.2-blue?style=flat-square" alt="PyPI version"></a>
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
- [Skills](#skills)
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

### Function Calling

```python
# Inject functions and variables into runtime
runtime = PythonRuntime(
    variables=[Variable("tasks", [], "User's task list")],
    functions=[Function(add_task), Function(complete_task)],
)
agent = CaveAgent(model, runtime=runtime)

await agent.run("Add 'buy groceries' to my tasks")
print(runtime.retrieve("tasks"))  # [{'name': 'buy groceries', 'done': False}]
```

See [examples/basic_usage.py](examples/basic_usage.py) for complete example.

### Stateful Object Interactions

```python
# Inject objects with methods - LLM can call them directly
runtime = PythonRuntime(
    types=[Type(Light), Type(Thermostat)],
    variables=[
        Variable("light", Light("Living Room"), "Smart light"),
        Variable("thermostat", Thermostat(), "Home thermostat"),
    ],
)
agent = CaveAgent(model, runtime=runtime)

await agent.run("Dim the light to 20% and set thermostat to 22°C")
light = runtime.retrieve("light")  # Object with updated state
```

See [examples/object_methods.py](examples/object_methods.py) for complete example.

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

See [examples/multi_agent.py](examples/multi_agent.py) for complete example.

### Real-time Streaming

```python
async for event in agent.stream_events("Analyze this data"):
    if event.type.value == 'code':
        print(f"Executing: {event.content}")
    elif event.type.value == 'execution_output':
        print(f"Result: {event.content}")
```

See [examples/stream.py](examples/stream.py) for complete example.

### Security Rules

```python
# Block dangerous operations with AST-based validation
rules = [
    ImportRule({"os", "subprocess", "sys"}),
    FunctionRule({"eval", "exec", "open"}),
    AttributeRule({"__globals__", "__builtins__"}),
]
runtime = PythonRuntime(security_checker=SecurityChecker(rules))
```

### More Examples

- [Basic Usage](examples/basic_usage.py): Function calling and object processing
- [Runtime State](examples/runtime_state.py): State management across interactions
- [Object Methods](examples/object_methods.py): Class methods and complex objects
- [Multi-Turn](examples/multi_turn.py): Conversations with state persistence
- [Multi-Agent](examples/multi_agent.py): Data pipeline with multiple agents
- [Stream](examples/stream.py): Streaming responses and events

## Skills

Skills are reusable, self-contained instruction packages that extend agent capabilities. Each skill bundles instructions, scripts, references, and injectable code—allowing agents to acquire domain expertise on-demand.

### Creating a Skill

A skill is a directory containing a `SKILL.md` file with YAML frontmatter:

```
my-skill/
├── SKILL.md           # Required: Skill definition and instructions
├── injection.py       # Optional: Functions/variables/types to inject
├── scripts/           # Optional: Executable Python scripts
├── references/        # Optional: Reference documents
└── assets/            # Optional: Static assets (JSON, images, etc.)
```

**SKILL.md** structure:

```markdown
---
name: data-processor
description: Process and analyze datasets with statistical methods
license: MIT
compatibility: Requires pandas and numpy
metadata:
  author: your-org
  version: "1.0"
---

# Data Processing Instructions

When activated, follow these steps:
1. Load the dataset using the provided scripts
2. Apply statistical analysis
3. Return structured results
```

### Using Skills

```python
from cave_agent import CaveAgent
from cave_agent.skills import SkillDiscovery

# Load skills from directory
agent = CaveAgent(model=model, skills_dir="./skills")

# Or load specific skills
skill = SkillDiscovery.from_file("./my-skill/SKILL.md")
agent = CaveAgent(model=model, skills=[skill])
```

When skills are loaded, the agent gains access to these runtime functions:

- `activate_skill(skill_name)` - Activate a skill and get its instructions
- `run_skill_script(skill_name, script_name, **kwargs)` - Run a skill's script
- `read_skill_reference(skill_name, reference_name)` - Read reference documents
- `read_skill_asset(skill_name, asset_name)` - Read asset files

### Injection Module

Skills can inject functions, variables, and types into the agent's runtime when activated. Create an `injection.py` file:

```python
from cave_agent.runtime import Function, Variable, Type
from dataclasses import dataclass

def analyze_data(data: list) -> dict:
    """Analyze data and return statistics."""
    return {"mean": sum(data) / len(data), "count": len(data)}

@dataclass
class AnalysisResult:
    mean: float
    count: int
    status: str

CONFIG = {"threshold": 0.5, "max_items": 1000}

__exports__ = [
    Function(analyze_data, description="Analyze data statistically"),
    Variable("CONFIG", value=CONFIG, description="Analysis configuration"),
    Type(AnalysisResult, description="Result structure"),
]
```

When `activate_skill()` is called, these exports are automatically injected into the runtime.

### Skill Scripts

Scripts in the `scripts/` directory can be executed via `run_skill_script()`. Each script must define a `main(runtime, **kwargs)` function:

```python
# scripts/process.py
def main(runtime, data, threshold=0.5):
    """Process data with the given threshold."""
    # Access agent's runtime state if needed
    config = runtime.retrieve("CONFIG")

    # Process and return results
    filtered = [x for x in data if x > threshold]
    return {"filtered": filtered, "count": len(filtered)}
```

Scripts support both sync and async main functions:

```python
# scripts/fetch.py
async def main(runtime, url):
    """Async script for fetching data."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

See [examples/skill_data_processor.py](examples/skill_data_processor.py) for a complete example.

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
- **Skills System**: Modular, reusable instruction packages with injectable code, scripts, and resources. Agents can activate skills on-demand to acquire domain expertise.
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
| `skills` | `List[Skill]` | `None` | List of skill objects to load |
| `skills_dir` | `str \| Path` | `None` | Directory to discover and load skills from |
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