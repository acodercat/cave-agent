# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CaveAgent is a Python agent framework that enables LLM function-calling through code generation (not JSON schemas) with persistent Python runtime state. It natively handles complex objects like DataFrames and ndarrays within a stateful runtime across multi-turn interactions.

## Development Commands

```bash
# Install with all optional dependencies for development
uv sync --all-groups

# Run all tests (requires LLM credentials via env vars)
pytest

# Run a single test file
pytest tests/test_basic_usage.py

# Run a specific test
pytest tests/test_ipython_runtime.py::test_simple_execution -v

# Build
python -m build
```

**Test environment variables**: Tests use a real LLM via `LLM_MODEL_ID`, `LLM_API_KEY`, and `LLM_BASE_URL` (see `tests/conftest.py`). All async tests use `@pytest.mark.asyncio`.

## Architecture

### Agent Execution Loop (`agent.py`)

`CaveAgent` drives a loop: send messages to the model, parse response for code blocks via `StreamingTextParser`, execute code in the runtime, append results as `ExecutionResultMessage`, repeat until `max_steps` or no code is generated. Before each LLM call, `_maybe_compact()` checks token usage and triggers context compaction if needed.

The agent supports two modes: `run()` (returns final `AgentResponse`) and `stream_events()` (yields `Event` objects for real-time streaming).

### Runtime System (`runtime/`)

`Runtime` (ABC) provides `inject_function/variable/type`, `execute(code)`, and `retrieve(name)`. It also generates `describe_functions/variables/types()` strings that go into the system prompt so the LLM knows what's available.

Two implementations:
- **IPythonRuntime** (default) - In-process IPython shell. Direct object access, zero serialization, but crashes affect the host process.
- **IPyKernelRuntime** - Separate Jupyter kernel subprocess. Objects serialized via `dill`. Crash-isolated, supports interrupt/reset. Requires `cave-agent[ipykernel]`.

**Primitives** (`primitives.py`): `Variable`, `Function`, `Type` wrap values for injection. `Type` uses `TypeSchemaExtractor` to auto-generate schemas from Pydantic models, dataclasses, Enums, and regular classes.

### Model Abstraction (`models/`)

`Model` ABC with `call()` and `stream()` methods. Two implementations:
- `OpenAIServerModel` - OpenAI API and compatible endpoints
- `LiteLLMModel` - 100+ providers via LiteLLM

Both are optional dependencies (`cave-agent[openai]` or `cave-agent[litellm]`). The models are imported lazily to avoid requiring both.

### Security (`security/`)

AST-based `SecurityChecker` validates code before execution using rules: `ImportRule`, `FunctionRule`, `AttributeRule`, `RegexRule`. Applied in the executor layer, violations short-circuit execution and surface as `SECURITY_ERROR` events.

### Skills System (`skills/`)

Implements the Agent Skills open standard. Each skill is a directory with `SKILL.md` (YAML frontmatter + markdown instructions) and optional `injection.py` (exports `Function`, `Variable`, `Type` objects). Skills use progressive disclosure: only metadata is loaded at startup; full instructions and runtime resources are injected on-demand when the agent calls the built-in `activate_skill(name)` function.

### Context Compaction (`compaction/`)

Three-tier strategy when token usage approaches `context_window`:
1. **Microcompact** - Clears old `ExecutionResultMessage`s (no LLM call)
2. **Full compact** - LLM summarizes older messages (with circuit breaker after 3 failures)
3. **Trim fallback** - Keeps only the latest N messages

### Display (`display.py`)

Terminal UI using Rich. Renders streaming events with Claude Code-style formatting (blue `●` for code blocks, `⎿` for output). Handles live markdown rendering, code execution spinners, and token usage summaries.

### Prompts (`prompts.py`)

System prompt template slots in `{instructions}`, `{functions}`, `{variables}`, `{types}`, `{skills}`. The prompt emphasizes persistent Jupyter-like state and guides the LLM to use only provided functions.
