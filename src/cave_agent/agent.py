from __future__ import annotations

from datetime import datetime
from enum import Enum, IntEnum
from typing import AsyncGenerator

from .logger import LogLevel, Logger
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


class _ContextState(IntEnum):
    INITIALIZED = 0
    RUNNING = 1
    COMPLETED = 2
    MAX_STEPS_REACHED = 3


class _ExecutionContext:
    """Manages execution state with max steps limit."""

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.code_snippets: list[str] = []
        self.total_steps = 0
        self.state = _ContextState.INITIALIZED
        self.token_usage = TokenUsage()

    def start(self) -> None:
        self.total_steps = 0
        self.state = _ContextState.RUNNING
        self.token_usage = TokenUsage()

    def next_step(self) -> bool:
        """Record a step. Returns False if max steps reached."""
        if self.total_steps >= self.max_steps:
            self.state = _ContextState.MAX_STEPS_REACHED
            return False
        self.total_steps += 1
        return True

    def complete(self) -> None:
        self.state = _ContextState.COMPLETED

    def add_token_usage(self, usage: TokenUsage) -> None:
        self.token_usage = self.token_usage + usage

    @property
    def is_running(self) -> bool:
        return self.state == _ContextState.RUNNING


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
        system_instructions: System-level execution rules and examples.
        system_prompt_template: Template string for system prompt.
        python_block_identifier: Code block language identifier.
        messages: Initial conversation history.
        log_level: Logging verbosity level.

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
        system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        python_block_identifier: str = DEFAULT_PYTHON_BLOCK_IDENTIFIER,
        messages: list[Message] | None = None,
        log_level: LogLevel = LogLevel.DEBUG,
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
        self.logger = Logger(log_level)
        self._init_skills(skills)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, query: str) -> AgentResponse:
        """Execute the agent with the given user query."""
        context = _ExecutionContext(self.max_steps)
        context.start()
        self._initialize_conversation(query)

        while context.is_running:
            if not context.next_step():
                self.logger.info(
                    "Max steps reached",
                    f"Completed {context.total_steps}/{context.max_steps} steps",
                )
                return self._build_response(context, "", ExecutionStatus.MAX_STEPS_REACHED)

            response = await self._execute_step(context)

            if not context.is_running:
                return self._build_response(context, response, ExecutionStatus.SUCCESS)

        raise RuntimeError("Unreachable: execution loop exited without returning")

    async def stream_events(self, query: str) -> AsyncGenerator[Event, None]:
        """Stream events during agent execution."""
        context = _ExecutionContext(self.max_steps)
        context.start()
        self._initialize_conversation(query)

        while context.is_running:
            if not context.next_step():
                self.logger.info(
                    "Max steps reached",
                    f"Completed {context.total_steps}/{context.max_steps} steps",
                )
                yield Event(EventType.MAX_STEPS_REACHED, "Max steps reached")
                return

            async for event in self._stream_step(context):
                yield event
                if not context.is_running:
                    return

    def build_system_prompt(self) -> str:
        """Build and format the system prompt with current runtime state."""
        return self.system_prompt_template.format(
            functions=self.runtime.describe_functions(),
            variables=self.runtime.describe_variables(),
            types=self.runtime.describe_types(),
            skills=self._skill_registry.describe_skills(),
            instructions=self.instructions,
            system_instructions=self.system_instructions,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def add_message(self, message: Message):
        """Add message with automatic history management."""
        self.messages.append(message)
        self.logger.debug(
            "History length",
            f"Current history length: {len(self.messages)}/{self.max_history}",
            "yellow",
        )
        self._trim_history()

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    async def _execute_step(self, context: _ExecutionContext) -> str:
        """Execute a single step and return the model response."""
        self._log_step(context)

        model_response = await self.model.call(self._prepare_messages())
        context.add_token_usage(model_response.token_usage)

        return await self._process_response(model_response.content, context)

    async def _stream_step(self, context: _ExecutionContext) -> AsyncGenerator[Event, None]:
        """Execute a single step with streaming output."""
        self._log_step(context)

        chunks: list[str] = []
        parser = StreamingTextParser(self.python_block_identifier)

        async for chunk in self.model.stream(self._prepare_messages()):
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
            self.logger.debug("Final response", model_response, "green")
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
            self.logger.debug("Final response", model_response, "green")
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
        self.logger.debug("Code snippet", code_snippet, "green")

        execution_result = await self.runtime.execute(code_snippet)

        # Security error
        if not execution_result.success and isinstance(execution_result.error, SecurityError):
            error_message = execution_result.error.message
            self.logger.debug("Security error", error_message, "red")
            return _ExecutionOutcome(
                event_type=EventType.SECURITY_ERROR,
                event_content=error_message,
                next_prompt=SECURITY_ERROR_PROMPT.format(error=error_message),
            )

        stdout = execution_result.stdout or "No output"

        # Output too long
        if len(stdout) > self.max_exec_output:
            self.logger.debug(
                "Execution output too long",
                f"Output length: {len(stdout)} characters (max: {self.max_exec_output})",
                "yellow",
            )
            return _ExecutionOutcome(
                event_type=EventType.EXECUTION_OUTPUT_EXCEEDED,
                event_content=stdout,
                next_prompt=EXECUTION_OUTPUT_EXCEEDED_PROMPT.format(
                    output_length=len(stdout),
                    max_length=self.max_exec_output,
                ),
            )

        # Normal output (success or error)
        if execution_result.success:
            self.logger.debug("Execution output", stdout, "cyan")
            event_type = EventType.EXECUTION_OUTPUT
        else:
            self.logger.debug("Execution output with error", stdout, "red")
            event_type = EventType.EXECUTION_ERROR

        return _ExecutionOutcome(
            event_type=event_type,
            event_content=stdout,
            next_prompt=EXECUTION_OUTPUT_PROMPT.format(execution_output=stdout),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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
        self.logger.debug("User query received", user_query, "blue")
        self.add_message(UserMessage(user_query))

    def _update_system_message(self):
        system_prompt = self.build_system_prompt()
        self.logger.debug("System prompt loaded", system_prompt, "blue")
        if self.messages and isinstance(self.messages[0], SystemMessage):
            self.messages[0] = SystemMessage(system_prompt)
        else:
            self.messages.insert(0, SystemMessage(system_prompt))

    def _prepare_messages(self) -> list[dict[str, str]]:
        """Convert internal message objects to dict format for LLM API."""
        return [
            {
                "role": _ROLE_MAP.get(msg.role, msg.role).value,
                "content": msg.content,
            }
            for msg in self.messages
        ]

    def _trim_history(self):
        if len(self.messages) > self.max_history:
            self.messages = [self.messages[0]] + self.messages[1:][-(self.max_history - 1):]
            self.logger.debug(
                "History trimmed",
                f"Trimmed to {len(self.messages)}/{self.max_history} messages",
                "yellow",
            )

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

    def _log_step(self, context: _ExecutionContext):
        self.logger.debug(
            f"Step {context.total_steps}/{context.max_steps}",
            "Processing...",
            "yellow",
        )
