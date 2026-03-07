# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CaveAgent is a Python framework that transforms LLMs from text-in-text-out generators into stateful runtime operators. Instead of JSON-schema tool calling, it uses LLM code generation within a persistent Python runtime where complex objects (DataFrames, models, DB connections) can be injected, manipulated, and retrieved across turns.

**Paper**: arXiv:2601.01569 — "CaveAgent: Transforming LLMs into Stateful Runtime Operators"
**PyPI**: `cave-agent` (v0.6.5)

## Build & Development Commands

```bash
# Install with all dependencies (uses uv, see uv.lock)
pip install -e '.[all]'

# Install dev dependencies
pip install -e '.[all]' && pip install pytest pytest-asyncio

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_basic_usage.py

# Run a specific test
pytest tests/test_basic_usage.py::test_name -v
```

Build system: hatchling. Python >= 3.11 required.

## Architecture

The core loop in `CaveAgent.run()` (`src/cave_agent/agent.py`):
1. Build system prompt with runtime state (variables, functions, types, skills)
2. Send conversation to LLM
3. Extract Python code blocks from LLM response
4. Execute code in the persistent `PythonRuntime`
5. Feed execution output back as a new message; repeat until no code block or max_steps

Key modules under `src/cave_agent/`:

- **`agent.py`** — `CaveAgent` class, message types (`MessageRole`, `Message` subclasses), `ExecutionContext` (step tracking), `AgentResponse`, streaming via `stream_events()`
- **`runtime/`** — `PythonRuntime` (IPython-based persistent execution), `Variable`/`Function`/`Type` primitives for injection/retrieval, `CodeExecutor` for sandboxed execution
- **`models/`** — `Model` base class with `OpenAIServerModel` and `LiteLLMModel` implementations (async `call()` and `stream()`)
- **`security/`** — AST-based `SecurityChecker` with pluggable rules: `ImportRule`, `FunctionRule`, `AttributeRule`, `RegexRule`
- **`skills/`** — Agent Skills (agentskills.io) implementation: `Skill`, `SkillDiscovery` (load from files/dirs), `SkillRegistry` (activation injects functions/variables into runtime via `injection.py` `__exports__`)
- **`parsing/`** — `StreamingTextParser` for incremental code block extraction from streaming LLM output
- **`prompts.py`** — System prompt templates and execution output formatting
- **`utils.py`** — Code extraction helpers (`extract_python_code`)

## Key Design Patterns

- **Dual-stream architecture**: Semantic stream (LLM text reasoning) + Runtime stream (persistent Python state). The runtime is the primary workspace, not the context window.
- **Message role conversion**: Internal roles `CODE_EXECUTION` and `EXECUTION_RESULT` are mapped to `assistant`/`user` roles for the LLM API (`role_conversions` in agent.py).
- **Progressive skill disclosure**: Skills load only metadata at startup (~100 tokens); full instructions load on `activate_skill()` call.
- **Runtime injection**: `injection.py` in skill directories exports `__exports__` list of `Function`/`Variable`/`Type` objects injected into runtime on activation.

## Website / Paper Materials

- `Website/Cave_agent (12).pdf` — Full paper PDF
- `Website/Cave_agent_overleaf_src/` — LaTeX source (main.tex, sections/, figures/)
- `images/` — Banner, architecture diagram, skills diagram (PNG)
- `2601.01569v3.pdf` — arXiv paper PDF

## User Goal: Paper Website

The user wants to create a beautiful, interactive website for this paper. Key paper assets are in `Website/Cave_agent_overleaf_src/` (LaTeX source with figures) and the PDF. The website should present the paper's content (abstract, method, experiments, results) in an engaging, interactive format.
