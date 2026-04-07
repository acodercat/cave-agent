from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import AsyncGenerator

from .models import Model, TokenUsage
from .parsing import SegmentType, StreamingTextParser
from .prompts import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_SYSTEM_INSTRUCTIONS,
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    EXECUTION_OUTPUT_EXCEEDED_PROMPT,
    EXECUTION_OUTPUT_PROMPT,
    SECURITY_ERROR_PROMPT,
    SKILLS_INSTRUCTION,
)
from .runtime import Runtime, IPythonRuntime, Function
from .runtime.executor import ExecutionResult
from .security import SecurityError
from .runtime.builtins import activate_skill
from .skills import Skill, SkillRegistry
from .utils import extract_python_code

DEFAULT_PYTHON_BLOCK_IDENTIFIER = "python"


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    CODE_EXECUTION = "code_execution"
    EXECUTION_RESULT = "execution_result"


# Maps internal roles to LLM-facing roles
_ROLE_MAP = {
    MessageRole.CODE_EXECUTION: MessageRole.ASSISTANT,
    MessageRole.EXECUTION_RESULT: MessageRole.USER,
}


class Message:
    """Base class for all message types in the agent conversation."""

    def __init__(self, content: str, role: MessageRole):
        self.content = content
        self.role = role


class SystemMessage(Message):
    """System message that provides instructions to the LLM."""
    def __init__(self, content: str):
        super().__init__(content, MessageRole.SYSTEM)


class UserMessage(Message):
    """Message from the user to the agent."""
    def __init__(self, content: str):
        super().__init__(content, MessageRole.USER)


class AssistantMessage(Message):
    """Message from the assistant (LLM) to the user."""
    def __init__(self, content: str):
        super().__init__(content, MessageRole.ASSISTANT)


class CodeExecutionMessage(Message):
    """Message representing code to be executed by the agent."""
    def __init__(self, content: str):
        super().__init__(content, MessageRole.CODE_EXECUTION)


class ExecutionResultMessage(Message):
    """Message representing the result from code execution."""
    def __init__(self, content: str):
        super().__init__(content, MessageRole.EXECUTION_RESULT)


# ---------------------------------------------------------------------------
# Event / response types
# ---------------------------------------------------------------------------


class EventType(Enum):
    TEXT = "text"
    CODE = "code"
    EXECUTION_OUTPUT = "execution_output"
    EXECUTION_ERROR = "execution_error"
    EXECUTION_OUTPUT_EXCEEDED = "execution_output_exceeded"
    FINAL_RESPONSE = "final_response"
    MAX_STEPS_REACHED = "max_steps_reached"
    SECURITY_ERROR = "security_error"


class Event:
    def __init__(self, type: EventType, content: str):
        self.type = type
        self.content = content


class ExecutionStatus(Enum):
    """Status of agent execution."""
    SUCCESS = "success"
    MAX_STEPS_REACHED = "max_steps_reached"


class AgentResponse:
    """Response from the agent."""

    def __init__(
        self,
        content: str,
        status: ExecutionStatus,
        steps_taken: int = 0,
        max_steps: int = 0,
        code_snippets: list[str] | None = None,
        token_usage: TokenUsage | None = None,
    ):
        self.content = content
        self.status = status
        self.steps_taken = steps_taken
        self.max_steps = max_steps
        self.code_snippets = code_snippets if code_snippets else []
        self.token_usage = token_usage if token_usage else TokenUsage()

    def __str__(self) -> str:
        return (
            f"AgentResponse(status={self.status.value}, "
            f"steps={self.steps_taken}/{self.max_steps}, "
            f"tokens={self.token_usage.total_tokens}, "
            f"content={self.content})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _ExecutionContext:
    """Tracks execution state across steps."""

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.code_snippets: list[str] = []
        self.total_steps = 0
        self.token_usage = TokenUsage()
        self._completed = False

    def start(self) -> None:
        self.total_steps = 0
        self.token_usage = TokenUsage()
        self._completed = False

    def complete(self) -> None:
        self._completed = True

    def add_token_usage(self, usage: TokenUsage) -> None:
        self.token_usage = self.token_usage + usage

    @property
    def is_completed(self) -> bool:
        return self._completed


class _ExecutionOutcome:
    """Result of code execution processing."""

    def __init__(self, event_type: EventType, event_content: str, next_prompt: str):
        self.event_type = event_type
        self.event_content = event_content
        self.next_prompt = next_prompt


# ---------------------------------------------------------------------------
# CaveAgent
# ---------------------------------------------------------------------------


class CaveAgent:
    """
    A tool-augmented agent that enables function-calling through LLM code generation.

    Instead of JSON schemas, this agent generates Python code to interact with tools
    in a controlled runtime environment. It maintains state across conversations and
    supports streaming responses.

    Args:
        model: LLM model instance implementing the Model interface.
        runtime: Python runtime with functions and variables.
        instructions: User instructions defining agent role and behavior.
        skills: List of skills to load.
        max_steps: Maximum execution steps before stopping.
        max_history: Maximum message history to retain.
        max_exec_output: Maximum length of execution output.
        max_exec_timeout: Maximum seconds for a single code execution.
            None means no timeout.
        system_instructions: System-level execution rules and examples.
        system_prompt_template: Template string for system prompt.
        python_block_identifier: Code block language identifier.
        messages: Initial conversation history.
        display: Whether to render events to the terminal via Rich.

    Example:
        >>> agent = CaveAgent(
        ...     model=llm_model,
        ...     runtime=IPythonRuntime(functions=[Function(add)])
        ... )
        >>> result = await agent.run("Add 5 and 3")
    """

    def __init__(
        self,
        model: Model,
        runtime: Runtime | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        skills: list[Skill] | None = None,
        max_steps: int = 10,
        max_history: int = 20,
        max_exec_output: int = 5000,
        max_exec_timeout: float | None = None,
        system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        python_block_identifier: str = DEFAULT_PYTHON_BLOCK_IDENTIFIER,
        messages: list[Message] | None = None,
        display: bool = True,
    ):
        self.model = model
        self.system_prompt_template = system_prompt_template
        self.max_steps = max_steps
        self.runtime = runtime if runtime else IPythonRuntime()
        self.instructions = instructions
        self.system_instructions = system_instructions.format(
            python_block_identifier=python_block_identifier,
        )
        self.python_block_identifier = python_block_identifier
        self.messages: list[Message] = list(messages) if messages else []
        self.max_history = max_history
        self.max_exec_output = max_exec_output
        self.max_exec_timeout = max_exec_timeout
        self.display = display
        self._init_skills(skills)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, query: str) -> AgentResponse:
        """Execute the agent with the given user query.

        When ``display=True``, internally uses streaming to enable
        terminal display, then collects the final response.
        """
        if self.display:
            return await self._run_with_display(query)
        return await self._run(query)

    async def _run(self, query: str) -> AgentResponse:
        """Non-streaming execution."""
        context = _ExecutionContext(self.max_steps)
        context.start()
        self._initialize_conversation(query)

        response = ""
        for _ in range(self.max_steps):
            response = await self._execute_step(context)
            if context.is_completed:
                return self._build_response(context, response, ExecutionStatus.SUCCESS)

        return self._build_response(context, response, ExecutionStatus.MAX_STEPS_REACHED)

    async def _run_with_display(self, query: str) -> AgentResponse:
        """Run via streaming to enable display, collect final response."""
        from .display import render_user_prompt
        render_user_prompt(query)

        context = _ExecutionContext(self.max_steps)
        last_content = ""

        async for event in self._wrap_with_display(self._stream_events(query, context), context):
            if event.type == EventType.FINAL_RESPONSE:
                last_content = event.content

        status = ExecutionStatus.SUCCESS if context.is_completed else ExecutionStatus.MAX_STEPS_REACHED
        return self._build_response(context, last_content, status)

    async def stream_events(self, query: str) -> AsyncGenerator[Event, None]:
        """Stream events during agent execution.

        When ``display=True``, events are printed to the terminal via Rich
        as they are yielded (transparent pass-through).
        """
        context = _ExecutionContext(self.max_steps)

        if self.display:
            async for event in self._wrap_with_display(self._stream_events(query, context), context):
                yield event
        else:
            async for event in self._stream_events(query, context):
                yield event

    async def _stream_events(self, query: str, context: _ExecutionContext) -> AsyncGenerator[Event, None]:
        """Internal event stream generator."""
        context.start()
        self._initialize_conversation(query)

        for _ in range(self.max_steps):
            async for event in self._stream_step(context):
                yield event
            if context.is_completed:
                return

        yield Event(EventType.MAX_STEPS_REACHED, "Max steps reached")

    def build_system_prompt(self) -> str:
        """Build and format the system prompt with current runtime state."""
        instructions = self.system_instructions
        if self.max_exec_timeout is not None:
            instructions += (
                f"\n- Code execution timeout: {self.max_exec_timeout} seconds. "
                "For network requests and database queries, always set timeout parameters "
                "(e.g. requests.get(url, timeout=10), pd.read_sql(sql, con, params, timeout=10)) "
                "to avoid hanging."
            )

        return self.system_prompt_template.format(
            functions=self.runtime.describe_functions(),
            variables=self.runtime.describe_variables(),
            types=self.runtime.describe_types(),
            skills=self._skill_registry.describe_skills(),
            instructions=self.instructions,
            system_instructions=instructions,
        )

    def add_message(self, message: Message):
        """Add message with automatic history management."""
        self.messages.append(message)
        self._trim_history()

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    async def _execute_step(self, context: _ExecutionContext) -> str:
        """Execute a single step and return the model response."""
        context.total_steps += 1

        model_response = await self.model.call(self._prepare_messages())
        context.add_token_usage(model_response.token_usage)

        return await self._process_response(model_response.content, context)

    async def _stream_step(self, context: _ExecutionContext) -> AsyncGenerator[Event, None]:
        """Execute a single step with streaming output."""
        context.total_steps += 1

        chunks: list[str] = []
        parser = StreamingTextParser(self.python_block_identifier)
        stream_response = self.model.stream(self._prepare_messages())

        async for chunk in stream_response:
            chunks.append(chunk)

            for segment in parser.process_chunk(chunk):
                if segment.type == SegmentType.TEXT:
                    yield Event(EventType.TEXT, segment.content)
                elif segment.type == SegmentType.CODE:
                    yield Event(EventType.CODE, segment.content)
                    if parser.is_first_code_block_completed():
                        break

            if parser.is_first_code_block_completed():
                break

        if not parser.is_first_code_block_completed():
            for segment in parser.flush():
                if segment.type == SegmentType.TEXT:
                    yield Event(EventType.TEXT, segment.content)
                elif segment.type == SegmentType.CODE:
                    yield Event(EventType.CODE, segment.content)

        context.add_token_usage(stream_response.usage)

        model_response = "".join(chunks)
        async for event in self._process_response_stream(model_response, context):
            yield event

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    async def _process_response(self, model_response: str, context: _ExecutionContext) -> str:
        """Process model response and execute code if present."""
        code_snippet = extract_python_code(model_response, self.python_block_identifier)

        if not code_snippet:
            self.add_message(AssistantMessage(model_response))
            context.complete()
            return model_response

        self.add_message(CodeExecutionMessage(model_response))
        outcome = await self._execute_code(code_snippet, context)
        self.add_message(ExecutionResultMessage(outcome.next_prompt))
        return model_response

    async def _process_response_stream(
        self,
        model_response: str,
        context: _ExecutionContext,
    ) -> AsyncGenerator[Event, None]:
        """Process model response with streaming events."""
        code_snippet = extract_python_code(model_response, self.python_block_identifier)

        if not code_snippet:
            self.add_message(AssistantMessage(model_response))
            context.complete()
            yield Event(EventType.FINAL_RESPONSE, model_response)
            return

        self.add_message(CodeExecutionMessage(model_response))
        outcome = await self._execute_code(code_snippet, context)
        self.add_message(ExecutionResultMessage(outcome.next_prompt))
        yield Event(outcome.event_type, outcome.event_content)

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    async def _execute_code(
        self,
        code_snippet: str,
        context: _ExecutionContext,
    ) -> _ExecutionOutcome:
        """Execute code snippet and return the outcome."""
        context.code_snippets.append(code_snippet)

        if self.max_exec_timeout is not None:
            execution_result = await self._execute_with_timeout(code_snippet)
            if execution_result is None:
                return _ExecutionOutcome(
                    event_type=EventType.EXECUTION_ERROR,
                    event_content=f"Execution timed out after {self.max_exec_timeout}s",
                    next_prompt=f"Code execution timed out after {self.max_exec_timeout} seconds. "
                        "Simplify your code or break it into smaller steps.",
                )
        else:
            execution_result = await self.runtime.execute(code_snippet)

        # Security error
        if not execution_result.success and isinstance(execution_result.error, SecurityError):
            error_message = execution_result.error.message
            return _ExecutionOutcome(
                event_type=EventType.SECURITY_ERROR,
                event_content=error_message,
                next_prompt=SECURITY_ERROR_PROMPT.format(error=error_message),
            )

        stdout = execution_result.stdout or "No output"

        # Output too long
        if len(stdout) > self.max_exec_output:
            return _ExecutionOutcome(
                event_type=EventType.EXECUTION_OUTPUT_EXCEEDED,
                event_content=stdout,
                next_prompt=EXECUTION_OUTPUT_EXCEEDED_PROMPT.format(
                    output_length=len(stdout),
                    max_length=self.max_exec_output,
                ),
            )

        # Normal output (success or error)
        event_type = EventType.EXECUTION_OUTPUT if execution_result.success else EventType.EXECUTION_ERROR

        return _ExecutionOutcome(
            event_type=event_type,
            event_content=stdout,
            next_prompt=EXECUTION_OUTPUT_PROMPT.format(execution_output=stdout),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _execute_with_timeout(self, code_snippet: str) -> ExecutionResult | None:
        """Run code in a thread so sync-blocking code can be timed out.

        On timeout, calls ``runtime.interrupt()`` to stop the execution
        (effective for IPyKernelRuntime; best-effort for IPythonRuntime).

        Returns None on timeout, ExecutionResult otherwise.
        """
        loop = asyncio.get_event_loop()
        result_future = loop.run_in_executor(
            None,
            lambda: asyncio.run(self.runtime.execute(code_snippet)),
        )
        try:
            return await asyncio.wait_for(result_future, timeout=self.max_exec_timeout)
        except asyncio.TimeoutError:
            await self.runtime.interrupt()
            return None

    async def _wrap_with_display(
        self,
        events: AsyncGenerator[Event, None],
        context: _ExecutionContext,
    ) -> AsyncGenerator[Event, None]:
        """Wrap events with terminal display.

        Lazy import: display.py imports Event/EventType from agent.py,
        so agent.py cannot import display.py at module level.
        """
        from .display import with_display
        async for event in with_display(events, context):
            yield event

    def _init_skills(self, skills: list[Skill] | None = None) -> None:
        self._skill_registry = SkillRegistry()
        if skills:
            self._skill_registry.add_skills([s for s in skills if s is not None])
        if self._skill_registry.list_skills():
            store = self._skill_registry.build_skill_store()
            self.runtime._executor.inject_into_namespace("_skill_store", store)
            self.runtime.inject_function(Function(activate_skill))
            self.system_instructions += "\n" + SKILLS_INSTRUCTION

    def _initialize_conversation(self, user_query: str):
        self._update_system_message()
        self.add_message(UserMessage(user_query))

    def _update_system_message(self):
        system_prompt = self.build_system_prompt()
        if self.messages and isinstance(self.messages[0], SystemMessage):
            self.messages[0] = SystemMessage(system_prompt)
        else:
            self.messages.insert(0, SystemMessage(system_prompt))

    def _prepare_messages(self) -> list[dict[str, str]]:
        """Convert internal message objects to dict format for LLM API.

        Injects a system-reminder with the current date/time as the first
        user message, so the model has temporal context without polluting
        the system prompt.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:00")
        reminder = {
            "role": "user",
            "content": f"<system-reminder>\nCurrent date and time: {now}\n</system-reminder>",
        }

        messages = []
        reminder_inserted = False
        for msg in self.messages:
            role = _ROLE_MAP.get(msg.role, msg.role).value
            messages.append({"role": role, "content": msg.content})
            if not reminder_inserted and role == "system":
                messages.append(reminder)
                reminder_inserted = True

        if not reminder_inserted:
            messages.insert(0, reminder)

        return messages

    def _trim_history(self):
        if len(self.messages) > self.max_history:
            self.messages = [self.messages[0]] + self.messages[1:][-(self.max_history - 1):]

    def _build_response(
        self,
        context: _ExecutionContext,
        content: str,
        status: ExecutionStatus,
    ) -> AgentResponse:
        return AgentResponse(
            content=content,
            code_snippets=context.code_snippets,
            status=status,
            steps_taken=context.total_steps,
            max_steps=self.max_steps,
            token_usage=context.token_usage,
        )
