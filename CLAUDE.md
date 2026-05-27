# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CaveAgent is a Python agent framework that enables LLM function-calling through code generation (not JSON schemas) with persistent Python runtime state. It natively handles complex objects like DataFrames and ndarrays within a stateful runtime across multi-turn interactions.

## Development Commands

Python >= 3.12 is required (see `pyproject.toml`).

```bash
# Install with all optional dependencies for development
uv sync --all-groups

# Run all tests (most require LLM credentials via env vars — see below)
pytest

# Run a single test file
pytest tests/test_basic_usage.py

# Run a specific test
pytest tests/test_ipython_runtime.py::test_simple_execution -v

# Build
python -m build
```

**Test environment variables**: Most tests are integration tests that hit a real LLM via `LLM_MODEL_ID`, `LLM_API_KEY`, and `LLM_BASE_URL` (see `tests/conftest.py`'s `model` fixture using `OpenAIServerModel`). Without those env vars, any test taking the `model` fixture will fail at fixture setup. Pure-unit files (e.g. `test_security_checker.py`, `test_streaming_text_parser.py`, `test_type_schema_extractor.py`) run offline. All async tests use `@pytest.mark.asyncio`.

## Architecture

### Agent Execution Loop (`agent.py`)

`CaveAgent` drives a loop: send messages to the model, parse response for code blocks via `StreamingTextParser` (in `parsing/streaming.py`), execute code in the runtime, append results as `ExecutionResultMessage`, repeat until `max_steps` or no code is generated. Before each LLM call, `_maybe_compact()` checks token usage and triggers context compaction if needed.

The agent supports two modes: `run()` (returns final `AgentResponse`) and `stream_events()` (yields `Event` objects for real-time streaming).

Message and event vocabulary lives in `types.py`: `MessageRole` (`SYSTEM`/`USER`/`ASSISTANT`/`CODE_EXECUTION`/`EXECUTION_RESULT` — the last two map to assistant/user when sent to the LLM via `_ROLE_MAP`) and `EventType` (`TEXT`, `CODE`, `EXECUTION_OUTPUT`, `EXECUTION_ERROR`, `EXECUTION_OUTPUT_EXCEEDED`, `FINAL_RESPONSE`, `MAX_STEPS_REACHED`, `SECURITY_ERROR`, `COMPACTING`, `COMPACTED`).

### Runtime System (`runtime/`)

`Runtime` (ABC) owns all LLM-facing resources — `functions`, `variables`, `types`, and `skills`. It provides `inject_function/variable/type`, `execute(code)`, and `retrieve(name)`, plus `describe_functions/variables/types/skills()` strings that get slotted into the system prompt so the LLM knows what's available. `CaveAgent` is intentionally skill-agnostic: skills are passed to the runtime constructor (`IPythonRuntime(skills=[...])`), and the agent just calls `runtime.describe_skills()` and `runtime.skill_instructions` when building the system prompt.

Two implementations:
- **IPythonRuntime** (default) - In-process IPython shell. Direct object access, zero serialization, but crashes affect the host process.
- **IPyKernelRuntime** - Separate Jupyter kernel subprocess. Objects serialized via `dill`. Crash-isolated, supports interrupt/reset. `reset()` re-injects functions/variables/types/skills after kernel restart. Requires `cave-agent[ipykernel]`.

**Primitives** (`primitives.py`): `Variable`, `Function`, `Type` wrap values for injection. `Type` uses `TypeSchemaExtractor` to auto-generate schemas from Pydantic models, dataclasses, Enums, and regular classes.

### Model Abstraction (`models/`)

`Model` ABC (in `base.py`) with `call()` and `stream()` methods. Two implementations:
- `OpenAIServerModel` - OpenAI API and compatible endpoints
- `LiteLLMModel` - 100+ providers via LiteLLM

Both are optional dependencies (`cave-agent[openai]` or `cave-agent[litellm]`). The models are imported lazily to avoid requiring both. `models/retry.py` wraps every model call with exponential-backoff retry (429/5xx/timeouts/connection errors, up to 5 attempts, respects `Retry-After`); when the model returns `finish_reason="length"`, `agent.py` separately resumes generation up to 3 times.

### Security (`security/`)

AST-based `SecurityChecker` validates code before execution using rules: `ImportRule`, `FunctionRule`, `AttributeRule`, `RegexRule`. Applied in the executor layer, violations short-circuit execution and surface as `SECURITY_ERROR` events.

### Skills System (`skills/`)

Implements the Agent Skills open standard. Each skill is a directory with `SKILL.md` (YAML frontmatter + markdown instructions) and optional `injection.py` (exports `Function`, `Variable`, `Type` objects). Skills use progressive disclosure: only metadata is loaded at startup; full instructions and runtime resources are injected on-demand when LLM-generated code calls the built-in `activate_skill(name)` function.

**Ownership:** Skills live on the `Runtime`, not the agent. `Runtime.__init__` accepts a `skills=` parameter and composes a `SkillRegistry`. On construction (and after `reset()`), the runtime injects `_skill_store` and the `activate_skill` builtin into its own namespace — exactly once, idempotently. Two `CaveAgent` instances over the same runtime naturally share its skill set. The `activate_skill` built-in itself lives in `runtime/builtins.py` and reads `_skill_store` from the execution namespace.

### Context Compaction (`compaction/`)

Three-tier strategy when token usage approaches `context_window`:
1. **Microcompact** - Clears old `ExecutionResultMessage`s (no LLM call)
2. **Full compact** - LLM summarizes older messages (with circuit breaker after 3 failures)
3. **Trim fallback** - Keeps only the latest N messages

### Display (`display.py`)

Terminal UI using Rich. Renders streaming events with Claude Code-style formatting (blue `●` for code blocks, `⎿` for output). Handles live markdown rendering, code execution spinners, and token usage summaries.

### Prompts (`prompts.py`)

System prompt template slots in `{instructions}`, `{functions}`, `{variables}`, `{types}`, `{skills}`. The prompt emphasizes persistent Jupyter-like state and guides the LLM to use only provided functions.
