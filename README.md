<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/acodercat/cave-agent/raw/master/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/acodercat/cave-agent/raw/master/assets/logo-light.svg">
    <img src="https://github.com/acodercat/cave-agent/raw/master/assets/logo-light.svg" alt="CaveAgent" width="50%">
  </picture>
</p>

<h3 align="center">
  <b>CaveAgent: Transforming LLMs into Stateful Runtime Operators </b>
</h3>

<p align="center">
  <a href="https://caveagent.dev"><img src="https://img.shields.io/badge/-Website-brightgreen?style=flat-square&logo=googlechrome&logoColor=white" alt="Website"></a>
  <a href="https://arxiv.org/abs/2601.01569"><img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.12+-blue?style=flat-square" alt="Python 3.12+"></a>
  <a href="https://pypi.org/project/cave-agent"><img src="https://img.shields.io/pypi/v/cave-agent?style=flat-square&color=blue&label=PyPI" alt="PyPI version"></a>
</p>

<p align="center">
  <em>"From text-in-text-out to (text&amp;object)-in-(text&amp;object)-out"</em>
</p>

---

Most LLM agents operate under a text-in-text-out paradigm, with tool interactions constrained to JSON primitives. CaveAgent breaks this with **Stateful Runtime Management**—a persistent Python runtime with direct **variable injection and retrieval**:

- **Inject** any Python object into the runtime—DataFrames, models, database connections, custom class instances—as first-class variables the LLM can manipulate
- **Persist** state across turns without serialization; objects live in the runtime, not in the context window
- **Retrieve** manipulated objects back as native Python types for downstream


https://github.com/user-attachments/assets/0e4a23b0-1afb-4408-8d87-ae1e13388aae

## Table of Contents

- [Installation](#installation)
- [Hello World](#hello-world)
- [Runtimes](#runtimes)
- [Examples](#examples)
  - [Data Visualization](#data-visualization)
  - [Function Calling](#function-calling)
  - [Stateful Object Interactions](#stateful-object-interactions)
  - [Multi-Agent Coordination](#multi-agent-coordination)
  - [Real-time Streaming](#real-time-streaming)
  - [Security Rules](#security-rules)
- [Agent Skills](#agent-skills)
  - [Creating a Skill](#creating-a-skill)
  - [How Skills Load](#how-skills-load-progressive-disclosure)
  - [Using Skills](#using-skills)
  - [Injection Module](#injection-module-caveagent-extension)
- [Context Compaction](#context-compaction)
- [API Resilience](#api-resilience)
- [Features](#features)
- [Configuration](#configuration)
- [LLM Provider Support](#llm-provider-support)

## Installation

```bash
pip install 'cave-agent[all]'
```

Choose your installation:

```bash
# OpenAI support
pip install 'cave-agent[openai]'

# 100+ LLM providers via LiteLLM
pip install 'cave-agent[litellm]'

# Process-isolated kernel runtime (IPyKernelRuntime)
pip install 'cave-agent[ipykernel]'
```

## Hello World

```python
import asyncio
from cave_agent import CaveAgent
from cave_agent.runtime import IPythonRuntime, Variable, Function
from cave_agent.models import LiteLLMModel

model = LiteLLMModel(model_id="model-id", api_key="your-api-key", custom_llm_provider="openai")

async def main():
    def reverse(s: str) -> str:
        """Reverse a string"""
        return s[::-1]

    runtime = IPythonRuntime(
        variables=[
            Variable("secret", "!dlrow ,olleH", "A reversed message"),
            Variable("greeting", "", "Store the reversed message"),
        ],
        functions=[Function(reverse)],
    )
    agent = CaveAgent(model, runtime=runtime)
    response = await agent.run("Reverse the secret")
    print(await runtime.retrieve("secret"))  # Hello, world!
    print(response.content)              # Agent's text response

asyncio.run(main())
```

## Runtimes

CaveAgent provides two runtime backends. Both share the same API for injecting functions, variables, and types — choose based on your trust and isolation requirements.

### IPythonRuntime (default)

Code runs **in the same process** via an embedded IPython shell. Injected objects (DataFrames, DB connections, custom classes) are accessed directly — no serialization overhead.

```python
from cave_agent.runtime import IPythonRuntime, Function, Variable

runtime = IPythonRuntime(
    functions=[Function(my_func)],
    variables=[Variable("data", my_dataframe, "Input data")],
)
agent = CaveAgent(model, runtime=runtime)
```

Best for: trusted environments, internal tools, when you need zero-overhead access to complex Python objects.

### IPyKernelRuntime (process-isolated)

Code runs in a **separate IPython kernel process**. If the code crashes (segfault, OOM, infinite loop), the host process stays alive — just reset the kernel and continue.

```bash
pip install 'cave-agent[ipykernel]'
```

```python
from cave_agent.runtime import IPyKernelRuntime, Function, Variable

async with IPyKernelRuntime(
    functions=[Function(my_func)],
    variables=[Variable("data", [1, 2, 3], "Input data")],
) as runtime:
    agent = CaveAgent(model, runtime=runtime)
    response = await agent.run("Analyze the data")
```

Injected objects are serialized via [dill](https://github.com/uqfoundation/dill), which supports local functions, closures, lambdas, and most Python objects.

Best for: untrusted code execution, multi-tenant environments, sandboxed agent workflows.

### Comparison

| | IPythonRuntime | IPyKernelRuntime |
|---|---|---|
| Isolation | Same process | Separate process |
| Crash impact | Host process dies | Kernel restarts, host survives |
| Object injection | Direct reference, zero-copy | Serialized via dill |
| Startup | Instant | ~1s (kernel launch) |
| Local functions / closures | Always works | Works (via dill) |
| Requires | *(included)* | `pip install 'cave-agent[ipykernel]'` |

## Examples

### Data Visualization

```python
from cave_agent import CaveAgent
from cave_agent.runtime import IPythonRuntime, Variable
from cave_agent.models import LiteLLMModel

model = LiteLLMModel(model_id="model-id", api_key="your-api-key", custom_llm_provider="openai")

# 1. Inject — real DB connection & chart config manager
runtime = IPythonRuntime(
    variables=[
        Variable("engine", database_engine),             # SQLAlchemy Engine
        Variable("echarts_config_manager", EChartsConfigManager()),  # Chart collector
    ]
)
agent = CaveAgent(model, runtime=runtime)

# 2. Query — LLM sees object types, not data
await agent.run("Show me the air quality trend for the past week")

# LLM generates & executes:
#   df = pd.read_sql("SELECT * FROM air_quality WHERE ...", engine)
#   echarts_config_manager.add_config({
#       "title": {"text": "Air Quality - Past Week"},
#       "xAxis": {"data": dates},
#       "series": [{"name": "PM2.5", "type": "line", "data": ...}]
#   })

# 3. Retrieve — get real chart configs for rendering
mgr = await runtime.retrieve("echarts_config_manager")  # Real Python object
configs = mgr.get_configs()

for config in configs:
    render_echarts(config)  # Render directly in web UI
```

### Function Calling

```python
# Inject functions and variables into runtime
runtime = IPythonRuntime(
    variables=[Variable("tasks", [], "User's task list")],
    functions=[Function(add_task), Function(complete_task)],
)
agent = CaveAgent(model, runtime=runtime)

await agent.run("Add 'buy groceries' to my tasks")
print(await runtime.retrieve("tasks"))  # [{'name': 'buy groceries', 'done': False}]
```

See [examples/basic_usage.py](examples/basic_usage.py) for a complete example.

### Stateful Object Interactions

```python
# Inject objects with methods - LLM can call them directly
runtime = IPythonRuntime(
    types=[Type(Light), Type(Thermostat)],
    variables=[
        Variable("light", Light("Living Room"), "Smart light"),
        Variable("thermostat", Thermostat(), "Home thermostat"),
    ],
)
agent = CaveAgent(model, runtime=runtime)

await agent.run("Dim the light to 20% and set thermostat to 22°C")
light = await runtime.retrieve("light")  # Object with updated state
```

See [examples/object_methods.py](examples/object_methods.py) for a complete example.

### Multi-Agent Coordination

```python
# Sub-agents with their own runtimes
cleaner_agent = CaveAgent(model, runtime=IPythonRuntime(variables=[
    Variable("data", [], "Input"), Variable("cleaned_data", [], "Output"),
]))

analyzer_agent = CaveAgent(model, runtime=IPythonRuntime(variables=[
    Variable("data", [], "Input"), Variable("insights", {}, "Output"),
]))

# Orchestrator controls sub-agents as first-class objects
orchestrator = CaveAgent(model, runtime=IPythonRuntime(variables=[
    Variable("raw_data", raw_data, "Raw dataset"),
    Variable("cleaner", cleaner_agent, "Cleaner agent"),
    Variable("analyzer", analyzer_agent, "Analyzer agent"),
]))

# Inject → trigger → retrieve
await orchestrator.run("Clean raw_data using cleaner, then analyze using analyzer")
insights = await analyzer.runtime.retrieve("insights")
```

See [examples/multi_agent.py](examples/multi_agent.py) for a complete example.

### Real-time Streaming

```python
from cave_agent.types import EventType

async for event in agent.stream_events("Analyze this data"):
    if event.type == EventType.THINKING_CHUNK:    # reasoning streams live (o1/o3, DeepSeek-R1, …)
        print(event.content, end="")
    elif event.type == EventType.TEXT:
        print(event.content, end="")
    elif event.type == EventType.CODE:
        print(f"\nExecuting: {event.content}")
    elif event.type == EventType.EXECUTION_OUTPUT:
        print(f"Result: {event.content}")
```

**Reasoning models.** When the model exposes a reasoning trace (OpenAI `o1`/`o3`, DeepSeek-R1, Qwen/Moonshot reasoning, …), CaveAgent streams it **live** as `THINKING_CHUNK` events (before the answer), then seals the segment with one `THINKING` event carrying the full trace. Reasoning is captured, kept out of the code-block parser, and never sent back on the wire.

See [examples/stream.py](examples/stream.py) for a complete example.

### Security Rules

```python
# Block dangerous operations with AST-based validation
rules = [
    ImportRule({"os", "subprocess", "sys"}),      # also blocks os.path, subprocess.run, ...
    FunctionRule({"eval", "exec", "open"}),        # also catches aliasing: f = open; f(...)
    AttributeRule({"__globals__", "__builtins__"}),
    RegexRule(r"rm\s+-rf|sudo\s+", "Block shell escapes"),
]
runtime = IPythonRuntime(security_checker=SecurityChecker(rules))
```

> ⚠️ **`SecurityChecker` is advisory hardening, not a sandbox.** Static AST
> analysis cannot catch every obfuscation (dynamic reflection, C-extension
> escapes, etc.). For genuinely untrusted code, run inside a real isolation
> boundary — a container with seccomp/gVisor and OS resource limits — and use
> the process-isolated `IPyKernelRuntime`. Treat these rules as defense-in-depth.

### More Examples

- [Basic Usage](examples/basic_usage.py): Function calling and object processing
- [Runtime State](examples/runtime_state.py): State management across interactions
- [Object Methods](examples/object_methods.py): Class methods and complex objects
- [Multi-Turn](examples/multi_turn.py): Conversations with state persistence
- [Multi-Agent](examples/multi_agent.py): Data pipeline with multiple agents
- [Stream](examples/stream.py): Streaming responses and events

## Agent Skills

CaveAgent implements the [Agent Skills](https://agentskills.io) open standard—a portable format for packaging instructions that agents can discover and use. Originally developed by Anthropic and now supported across the AI ecosystem (Claude, Gemini CLI, Cursor, VS Code, and more), Skills enable agents to acquire domain expertise on-demand.

<p align="center">
  <img src="https://github.com/acodercat/cave-agent/raw/master/assets/skills.png" alt="Agent Skills Architecture" width="85%">
</p>

### Creating a Skill

A Skill is a directory containing a `SKILL.md` file with YAML frontmatter:

```
my-skill/
├── SKILL.md           # Required: Skill definition and instructions
└── injection.py       # Optional: Functions/variables/types to inject (CaveAgent extension)
```

**SKILL.md** structure:

```markdown
---
name: data-processor
description: Process and analyze datasets with statistical methods. Use when working with data analysis tasks.
---

# Data Processing Instructions

## Quick Start
Use the injected functions to analyze datasets...

## Workflows
1. Activate the skill to load injected functions
2. Apply statistical analysis using the provided functions
3. Return structured results
```

**Required fields**: `name` (max 64 chars, lowercase with hyphens) and `description` (max 1024 chars)

**Optional fields**: `license`, `compatibility`, `metadata`

### How Skills Load (Progressive Disclosure)

Skills use progressive disclosure to minimize context usage:

| Level | When Loaded | Content |
|-------|-------------|---------|
| **Metadata** | At startup | `name` and `description` from YAML frontmatter (~100 tokens) |
| **Instructions** | When activated | SKILL.md body with guidance (loaded on-demand) |

### Using Skills

```python
from pathlib import Path
from cave_agent import CaveAgent, Skill
from cave_agent.skills import SkillDiscovery
from cave_agent.runtime import Function, Variable

# Create skills directly
skill = Skill(
    name="my-skill",
    description="A custom skill",
    body_content="# Instructions\nFollow these steps...",
    functions=[Function(my_func)],
    variables=[Variable("config", value={})],
)
agent = CaveAgent(model=model, skills=[skill])

# Or load from files
skill = SkillDiscovery.from_file(Path("./my-skill/SKILL.md"))
agent = CaveAgent(model=model, skills=[skill])

# Or load from directory
skills = SkillDiscovery.from_directory(Path("./skills"))
agent = CaveAgent(model=model, skills=skills)
```

When skills are loaded, the agent gains access to the `activate_skill(skill_name)` runtime function to activate a skill and load its instructions.

### Injection Module (CaveAgent Extension)

CaveAgent extends the Agent Skills standard with `injection.py`—allowing skills to inject functions, variables, and types directly into the runtime when activated:

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

When `activate_skill()` is called, these exports are automatically injected into the runtime namespace.

See [examples/skill_data_processor.py](examples/skill_data_processor.py) for a complete example.

<p align="center">
  <img src="https://github.com/acodercat/cave-agent/raw/master/assets/overall.png" alt="CaveAgent Architecture">
</p>

## Context Compaction

Long conversations inevitably fill up the model's context window. CaveAgent implements a multi-tier compaction strategy inspired by [Claude Code](https://docs.anthropic.com/en/docs/claude-code)'s context management system.

**How it works:**

```
Token usage exceeds threshold?
        |
        v
  Tier 1: Microcompact (no LLM, instant)
  Clear old execution results, keep recent 6.
  Tokens under threshold? → done
        |
        v
  Tier 2: Full Compact (LLM summarization)
  Summarize older messages, keep recent half.
  Uses dual-phase prompt: <analysis> (discarded) + <summary> (kept).
        |
        v
  Tier 3: Trim Fallback (last resort)
  Keep recent half of messages, drop the rest.
```

The system message (index 0) is always preserved. A **circuit breaker** stops attempting LLM summarization after 3 consecutive failures, falling back to trim to avoid wasting API calls.

Two refinements keep it accurate before an API token count is available:

- **CJK-aware estimation.** Chinese / Japanese / Korean text tokenizes ~2× denser than the `chars/4` heuristic, so estimation switches to a per-class split once CJK density crosses a threshold — compaction then triggers at the right time for non-English conversations, not too late.
- **Emergency recovery.** If a request still overflows the window despite proactive compaction, an aggressive path (keep the most recent quarter, summarize the rest) runs and the call is retried once — see [API Resilience](#api-resilience).

```python
agent = CaveAgent(
    model,
    runtime=runtime,
    context_window=128_000,  # triggers compaction at ~77% usage
)
```

## API Resilience

CaveAgent is built to survive transient failures, provider quirks, and context overflow without dropping the run.

**Typed error handling.** Provider SDK errors are classified into a provider-agnostic hierarchy so each is handled correctly instead of blindly retried:

- **Transient** (429 / 5xx / 408 / connection errors) → retried up to 5 times with exponential backoff + jitter (0.5s, 1s, 2s, 4s, 8s); `Retry-After` is honored when present.
- **Billing exhausted** (`insufficient_quota`, credit balance, HTTP 402, …) → **never retried** — a top-up, not time, is what fixes it.
- **Context too long** → routed to compaction, not retry (below).

**Reactive context-overflow recovery.** If the model still rejects a prompt as too long (the local estimate under-counted), the agent aggressively compacts — keeps the most recent quarter, summarizes the rest — and retries the call **once**, instead of failing.

**Streaming liveness.** A sliding idle-timeout watchdog (`stream_idle_timeout`, default 120s) abandons a stalled stream — catching stalls transport timeouts miss (proxy keep-alives / provider `ping`s keep the socket readable while no token arrives). A first-chunk connect failure is retried before any content is emitted; a mid-stream drop salvages what streamed instead of crashing the run.

**Output truncation recovery.** When the model's response is cut off (`finish_reason="length"`), the agent appends the partial response and asks the model to continue from the cutoff — up to 3 times.

**Resource lifecycle.** Every model implements `aclose()` (and works as an `async with` context manager) to release its HTTP connection pool cleanly.

## Features

- **Code-based function calling** — the LLM writes and runs Python against injected objects, not rigid JSON tool schemas
- **Stateful runtime** — inject/retrieve real Python objects (DataFrames, connections, class instances) with state persisting across turns; expose class schemas for type-aware code generation
- **Two runtimes** — in-process `IPythonRuntime` or crash-isolated `IPyKernelRuntime`
- **Reasoning models** — streams `o1` / DeepSeek-R1 reasoning live as `THINKING_CHUNK` events
- **Execution control** — step limits, a wall-clock run budget, and per-execution / stream-idle timeouts

Skills, compaction, resilience, security, multi-agent, and provider support each have a dedicated section above.


## Awesome Blogs

We thank these community to post our work.

- [CaveAgent让LLM学会了“跑代码”，你能把Agent变成Jupyter里的“老司机”](https://mp.weixin.qq.com/s/cJQ8ki0gXSmcbTPaBBfT5g)
- [Token消耗减半性能满分！状态化运行时管理能力让智能体性能飞升](https://mp.weixin.qq.com/s/qfVl3ATO4ueDdPb4npmTXQ)
- [Stateful environment to LLMs](https://x.com/rosinality/status/2008434433972728264)
- [TEKTA-AI](https://www.tekta.ai/ai-research-papers/caveagent-stateful-llm-runtime-2025)

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | Model | required | LLM model instance (OpenAIModel or LiteLLMModel) |
| runtime | Runtime | None | `IPythonRuntime` (default) or `IPyKernelRuntime` (process-isolated) |
| skills | List[Skill] | None | List of skill objects to load |
| max_steps | int | 10 | Maximum execution steps per run |
| max_run_time | float \| None | None | Wall-clock budget (seconds) for the whole run, checked at turn boundaries. `None` disables |
| context_window | int | 128000 | Model context window size in tokens. Controls when context compaction triggers |
| max_exec_output | int | 5000 | Max characters in execution output |
| max_exec_timeout | float \| None | None | Max seconds per code execution. LLM is guided to use timeouts in network/DB calls |
| stream_idle_timeout | float \| None | 120.0 | Max seconds between two stream chunks before the model call is abandoned (catches stalls transport timeouts miss). `None` disables |
| display | bool | True | Render events to terminal via Rich (Claude Code-style UI) |
| instructions | str | default | User instructions defining agent role and behavior |
| system_instructions | str | default | System-level execution rules and examples |
| system_prompt_template | str | default | Custom system prompt template |
| python_block_identifier | str | python | Code block language identifier |
| messages | List[Message] | None | Initial message history |

## LLM Provider Support

CaveAgent supports multiple LLM providers:

### OpenAI-Compatible Models
```python
from cave_agent.models import OpenAIModel

model = OpenAIModel(
    model_id="gpt-4",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"  # or your custom endpoint
)

# Release the HTTP connection pool when tearing down a long-lived model,
# or use it as an async context manager:
await model.aclose()
# async with OpenAIModel(model_id="gpt-4", api_key="...") as model:
#     ...
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

MIT License
