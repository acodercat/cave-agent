from .prompts import DEFAULT_SYSTEM_PROMPT, EXECUTION_OUTPUT_PROMPT, DEFAULT_INSTRUCTIONS, DEFAULT_AGENT_IDENTITY, DEFAULT_ADDITIONAL_CONTEXT, EXECUTION_OUTPUT_EXCEEDED_PROMPT, SECURITY_ERROR_PROMPT, SKILLS_INSTRUCTION
from .runtime import PythonRuntime, Function
from .security import SecurityError
from .skills import Skill, SkillRegistry, SkillDiscovery
from typing import List, Dict, Any, AsyncGenerator, Optional, Union
from pathlib import Path
from .models import Model, TokenUsage
from rich.console import Console
from rich.text import Text
from rich.style import Style
from .utils import extract_python_code
from .parsing import SegmentType, StreamingTextParser
from enum import Enum, IntEnum
from datetime import datetime

DEFAULT_PYTHON_BLOCK_IDENTIFIER = "python"

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    CODE_EXECUTION = "code_execution"
    EXECUTION_RESULT = "execution_result"


role_conversions = {
    MessageRole.CODE_EXECUTION: MessageRole.ASSISTANT,
    MessageRole.EXECUTION_RESULT: MessageRole.USER,
}

class Message():
    """Base class for all message types in the agent conversation."""
    def __init__(self, content: str, role: MessageRole):
        self.content = content
        self.role = role

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

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

class LogLevel(IntEnum):
    """Log levels for controlling output verbosity."""
    ERROR = 0  # Only errors
    INFO = 1   # Normal output
    DEBUG = 2  # Detailed output

class EventType(Enum):
    TEXT = "text"
    CODE = "code"
    EXECUTION_OUTPUT = "execution_output"
    EXECUTION_ERROR = "execution_error"
    EXECUTION_OUTPUT_EXCEEDED = "execution_output_exceeded"
    FINAL_RESPONSE = "final_response"
    MAX_STEPS_REACHED = "max_steps_reached"
    SECURITY_ERROR = "security_error"

class Logger:
    """
    A structured logger for Agent that provides leveled logging with rich formatting.

    Handles different types of log messages (debug, info, error) with customizable
    styling and visibility levels. Uses rich library for enhanced console output.

    Log Levels:
    - ERROR (0): Only critical errors
    - INFO (1): Standard operation information
    - DEBUG (2): Detailed execution traces
    """

    def __init__(self, level: LogLevel = LogLevel.INFO):
        """Initialize logger with specified verbosity level."""
        self.console = Console()
        self.level = level
        self.level_styles = {
            LogLevel.DEBUG: Style(color="yellow", bold=False),
            LogLevel.INFO: Style(color="bright_blue", bold=False),
            LogLevel.ERROR: Style(color="bright_red", bold=True)
        }

        self.level_prefix = {
            LogLevel.DEBUG: "DEBUG",
            LogLevel.INFO: "INFO",
            LogLevel.ERROR: "ERROR"
        }

    def __log(self, title: str, content: Any, style: str, level: LogLevel = LogLevel.INFO):
        if level <= self.level:
            # Create composite log message with improved formatting
            message = Text()

            # Add log level indicator
            message.append(f"[{self.level_prefix[level]}] ", self.level_styles[level])

            style = Style.parse(style)
            message.append(f"{title}: \n", style)

            message.append(content, style)

            self.console.print(message)

    def debug(self, title: str, content: Any, style: str = "yellow"):
        self.__log(title, content, style, LogLevel.DEBUG)

    def info(self, title: str, content: Any, style: str = "blue"):
        self.__log(title, content, style, LogLevel.INFO)

    def error(self, title: str, content: Any, style: str = "red"):
        self.__log(title, content, style, LogLevel.ERROR)

class ContextState(IntEnum):
    """Execution context state enumeration."""
    INITIALIZED = 0
    RUNNING = 1
    COMPLETED = 2
    MAX_STEPS_REACHED = 3

class ExecutionContext:
    """Manages execution state with max steps limit."""

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.code_snippets = []
        self.total_steps = 0
        self.state = ContextState.INITIALIZED
        self.token_usage = TokenUsage()  # Track token usage

    def start(self) -> None:
        """Initialize execution context."""
        self.total_steps = 0
        self.state = ContextState.RUNNING
        self.token_usage = TokenUsage()

    def next_step(self) -> bool:
        """Record a step execution. Returns False if max steps reached."""
        if self.total_steps >= self.max_steps:
            self.state = ContextState.MAX_STEPS_REACHED
            return False

        self.total_steps += 1
        return True

    def complete(self) -> None:
        """Mark execution as completed successfully."""
        self.state = ContextState.COMPLETED

    def add_token_usage(self, usage: TokenUsage) -> None:
        """Accumulate token usage from a model call."""
        self.token_usage = self.token_usage + usage

    @property
    def is_running(self) -> bool:
        return self.state == ContextState.RUNNING

class ExecutionStatus(Enum):
    """Status of agent execution."""
    SUCCESS = "success"
    MAX_STEPS_REACHED = "max_steps_reached"

class Event:
    def __init__(self, type: EventType, content: str):
        self.type = type
        self.content = content

class AgentResponse:
    """Response from the agent."""

    def __init__(
        self,
        content: str,
        status: ExecutionStatus,
        steps_taken: int = 0,
        max_steps: int = 0,
        code_snippets: Optional[List[str]] = None,
        token_usage: Optional[TokenUsage] = None,
    ):
        self.content = content
        self.status = status
        self.steps_taken = steps_taken
        self.max_steps = max_steps
        self.code_snippets = code_snippets if code_snippets else []
        self.token_usage = token_usage if token_usage else TokenUsage()

    def __str__(self) -> str:
        """String representation of the response."""
        return f"AgentResponse(status={self.status.value}, steps={self.steps_taken}/{self.max_steps}, tokens={self.token_usage.total_tokens}, content={self.content})"


class ExecutionOutcome:
    """Result of code execution processing."""

    event_type: EventType
    event_content: str
    next_prompt: str

    def __init__(self, event_type: EventType, event_content: str, next_prompt: str):
        self.event_type = event_type
        self.event_content = event_content
        self.next_prompt = next_prompt

class CaveAgent:
    """
    A tool-augmented agent that enables function-calling through LLM code generation.

    Instead of JSON schemas, this agent generates Python code to interact with tools
    in a controlled runtime environment. It maintains state across conversations and
    supports streaming responses.

    Args:
        model (Model): LLM model instance implementing the Model interface.
            Supports OpenAIServerModel, LiteLLMModel, or custom implementations.
        system_prompt_template (str, optional): Template string for system prompt.
            Uses placeholders for runtime context. Defaults to DEFAULT_SYSTEM_PROMPT.
        max_steps (int, optional): Maximum execution steps before stopping.
            Prevents infinite loops. Defaults to 5.
        log_level (LogLevel, optional): Logging verbosity level.
            Options: ERROR (0), INFO (1), DEBUG (2). Defaults to DEBUG.
        agent_role (str, optional): Role definition and capabilities description.
            Informs the LLM about the agent's purpose. Defaults to DEFAULT_ROLE_DEFINITION.
        instructions (str, optional): Execution instructions for the agent.
            Guides how the agent should process requests. Defaults to DEFAULT_INSTRUCTIONS.
        additional_context (str, optional): Additional context and usage examples.
            Helps the LLM understand available tools. Defaults to DEFAULT_ADDITIONAL_CONTEXT.
        runtime (PythonRuntime, optional): Python runtime with functions and variables.
            If None, creates an empty runtime. Defaults to None.
        messages (List[Message], optional): Initial conversation history.
            List of Message objects to start with. Defaults to empty list.
        max_history (int, optional): Maximum message history to retain.
            Prevents memory bloat in long conversations. Defaults to 10.
        max_execution_result_length (int, optional): Maximum length of execution result to be fed back to the LLM.
            Prevents the agent from generating too long execution results. Defaults to 3000.
        skills (List[Skill], optional): List of skills to load.
            If None, no skills are loaded. Defaults to None.
        skills_dir (Union[str, Path], optional): Directory to load skills from.
            If None, no skills are loaded from directory. Defaults to None.

    Example:
        >>> def add(a: int, b: int) -> int:
        ...     return a + b
        >>>
        >>> agent = CaveAgent(
        ...     model=llm_model,
        ...     runtime=PythonRuntime(functions=[Function(add)])
        ... )
        >>>
        >>> result = await agent.run("Add 5 and 3")
        >>> print(result)  # "The sum is: 8"
    """

    model: Model
    system_prompt_template: str
    max_steps: int
    runtime: PythonRuntime
    agent_identity: str
    instructions: str
    additional_context: str
    python_block_identifier: str
    messages: List[Message]
    max_history: int
    max_execution_result_length: int
    logger: Logger
    _skill_registry: SkillRegistry

    def __init__(
        self,
        model: Model,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 10,
        log_level: LogLevel = LogLevel.DEBUG,
        agent_identity: str = DEFAULT_AGENT_IDENTITY,
        instructions: str = DEFAULT_INSTRUCTIONS,
        additional_context: str = DEFAULT_ADDITIONAL_CONTEXT,
        runtime: Optional[PythonRuntime] = None,
        python_block_identifier: str = DEFAULT_PYTHON_BLOCK_IDENTIFIER,
        messages: Optional[List[Message]] = None,
        max_history: int = 20,
        max_execution_result_length: int = 3000,
        skills: Optional[List[Skill]] = None,
        skills_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize CaveAgent with improved parameter handling."""
        self.model = model
        self.system_prompt_template = system_prompt_template
        self.max_steps = max_steps
        self.runtime = runtime if runtime else PythonRuntime()
        self.agent_identity = agent_identity
        self.instructions = instructions.format(python_block_identifier=python_block_identifier)
        self.additional_context = additional_context.format(python_block_identifier=python_block_identifier)
        self.python_block_identifier = python_block_identifier
        self.messages = list(messages) if messages else []
        self.max_history = max_history
        self.max_execution_result_length = max_execution_result_length
        self.logger = Logger(log_level)
        self._init_skills(skills, skills_dir)

    def _init_skills(
        self,
        skills: Optional[List[Skill]] = None,
        skills_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize skills from provided list and/or directory."""
        self._skill_registry = SkillRegistry(agent_runtime=self.runtime)

        if skills_dir:
            dir_path = Path(skills_dir) if isinstance(skills_dir, str) else skills_dir
            self._skill_registry.add_skills(SkillDiscovery.from_directory(dir_path))

        if skills:
            self._skill_registry.add_skills([s for s in skills if s is not None])

        # Inject skill functions into runtime if skills are available
        if self._skill_registry.list_skills():
            self.runtime.inject_function(Function(self._skill_registry.activate_skill))
            self.instructions += "\n" + SKILLS_INSTRUCTION

    def build_system_prompt(self) -> str:
        """Build and format the system prompt with current runtime state."""
        return self.system_prompt_template.format(
            functions=self.runtime.describe_functions(),
            variables=self.runtime.describe_variables(),
            types=self.runtime.describe_types(),
            skills=self._skill_registry.describe_skills(),
            agent_identity=self.agent_identity,
            instructions=self.instructions,
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            additional_context=self.additional_context,
        )

    async def run(self, query: str) -> AgentResponse:
        """Execute the agent with the given user query."""
        context = ExecutionContext(self.max_steps)
        context.start()
        self._initialize_conversation(query)

        while context.is_running:
            # Check if we can proceed to next step
            if not context.next_step():
                # Max steps reached
                self.logger.info("Max steps reached", f"Completed {context.total_steps}/{context.max_steps} steps")
                return AgentResponse(
                    content='',
                    code_snippets=context.code_snippets,
                    status=ExecutionStatus.MAX_STEPS_REACHED,
                    steps_taken=context.total_steps,
                    max_steps=self.max_steps,
                    token_usage=context.token_usage
                )

            response = await self._execute_step(context)

            if not context.is_running:
                return AgentResponse(
                    content=response,
                    code_snippets=context.code_snippets,
                    status=ExecutionStatus.SUCCESS,
                    steps_taken=context.total_steps,
                    max_steps=self.max_steps,
                    token_usage=context.token_usage
                )

    async def stream_events(self, query: str) -> AsyncGenerator[Event, None]:
        """Stream events during agent execution."""
        context = ExecutionContext(self.max_steps)
        context.start()
        self._initialize_conversation(query)

        while context.is_running:
            # Check if we can proceed to next step
            if not context.next_step():
                # Max steps reached
                self.logger.info("Max steps reached", f"Completed {context.total_steps}/{context.max_steps} steps")
                yield Event(EventType.MAX_STEPS_REACHED, "Max steps reached")
                return

            async for event in self._stream_step_execution(context):
                yield event

                if not context.is_running:
                    return

    async def _execute_step(self, context: ExecutionContext) -> str:
        """Execute a single step and return result."""
        self._log_step(context)

        # Get LLM response (now returns ModelResponse with token_usage)
        model_response = await self.model.call(self._prepare_messages())

        # Accumulate token usage
        context.add_token_usage(model_response.token_usage)

        # Process response content
        return await self._process_model_response(model_response.content, context)

    async def _stream_step_execution(self, context: ExecutionContext) -> AsyncGenerator[Event, None]:
        """Execute a single step with streaming output."""
        self._log_step(context)

        # Stream LLM response and collect
        chunks = []
        parser = StreamingTextParser(self.python_block_identifier)

        async for chunk in self.model.stream(self._prepare_messages()):
            chunks.append(chunk)

            # Parse and yield streaming events
            parsed_segments = parser.process_chunk(chunk)
            for segment in parsed_segments:
                if segment.type == SegmentType.TEXT:
                    yield Event(EventType.TEXT, segment.content)
                elif segment.type == SegmentType.CODE:
                    yield Event(EventType.CODE, segment.content)
                    if parser.is_first_code_block_completed():
                        break

            if parser.is_first_code_block_completed():
                break

        if not parser.is_first_code_block_completed():
            final_segments = parser.flush()
            for segment in final_segments:
                if segment.type == SegmentType.TEXT:
                    yield Event(EventType.TEXT, segment.content)
                elif segment.type == SegmentType.CODE:
                    yield Event(EventType.CODE, segment.content)

        model_response = "".join(chunks)
        # Process complete response
        async for event in self._process_model_response_streaming(model_response, context):
            yield event

    async def _execute_code_snippet(
        self,
        code_snippet: str,
        context: ExecutionContext
    ) -> ExecutionOutcome:
        """
        Execute code snippet and return the outcome.

        This is the shared logic between streaming and non-streaming paths.

        Args:
            code_snippet: Python code to execute
            context: Current execution context

        Returns:
            ExecutionOutcome with event type, content, and next prompt
        """
        context.code_snippets.append(code_snippet)
        self.logger.debug("Code snippet", code_snippet, "green")

        execution_result = await self.runtime.execute(code_snippet)

        # Handle security errors
        if not execution_result.success and isinstance(execution_result.error, SecurityError):
            error_message = execution_result.error.message
            self.logger.debug("Security error", error_message, "red")
            return ExecutionOutcome(
                event_type=EventType.SECURITY_ERROR,
                event_content=error_message,
                next_prompt=SECURITY_ERROR_PROMPT.format(error=error_message)
            )

        # Handle execution output
        stdout = execution_result.stdout or "No output"
        stdout_length = len(stdout)

        # Check if output exceeds limit
        if stdout_length > self.max_execution_result_length:
            self.logger.debug(
                "Execution output too long",
                f"Output length: {stdout_length} characters (max: {self.max_execution_result_length})",
                "yellow"
            )
            return ExecutionOutcome(
                event_type=EventType.EXECUTION_OUTPUT_EXCEEDED,
                event_content=stdout,
                next_prompt=EXECUTION_OUTPUT_EXCEEDED_PROMPT.format(
                    output_length=stdout_length,
                    max_length=self.max_execution_result_length
                )
            )

        # Normal output (success or error)
        if execution_result.success:
            self.logger.debug("Execution output", stdout, "cyan")
            event_type = EventType.EXECUTION_OUTPUT
        else:
            self.logger.debug("Execution output with error", stdout, "red")
            event_type = EventType.EXECUTION_ERROR

        return ExecutionOutcome(
            event_type=event_type,
            event_content=stdout,
            next_prompt=EXECUTION_OUTPUT_PROMPT.format(execution_output=stdout)
        )

    async def _process_model_response(self, model_response: str, context: ExecutionContext) -> str:
        """Process model response and execute code if needed."""
        code_snippet = extract_python_code(model_response, self.python_block_identifier)

        if not code_snippet:
            self.add_message(AssistantMessage(model_response))
            context.complete()
            self.logger.debug("Final response", model_response, "green")
            return model_response

        self.add_message(CodeExecutionMessage(model_response))
        execution_outcome = await self._execute_code_snippet(code_snippet, context)
        self.add_message(ExecutionResultMessage(execution_outcome.next_prompt))

        return model_response

    async def _process_model_response_streaming(
        self,
        model_response: str,
        context: ExecutionContext
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
        execution_outcome = await self._execute_code_snippet(code_snippet, context)
        self.add_message(ExecutionResultMessage(execution_outcome.next_prompt))
        yield Event(execution_outcome.event_type, execution_outcome.event_content)

    def _log_step(self, context: ExecutionContext):
        """Log step execution info."""
        self.logger.debug(
            f"Step {context.total_steps}/{context.max_steps}",
            f"Processing...",
            "yellow"
        )

    def _initialize_conversation(self, user_query: str):
        """Initialize the conversation with the user prompt and system prompt."""
        self._update_system_message()
        self.logger.debug("User query received", user_query, "blue")
        self.add_message(UserMessage(user_query))


    def _update_system_message(self):
        """Update or insert system message."""

        system_prompt = self.build_system_prompt()
        self.logger.debug("System prompt loaded", system_prompt, "blue")

        if self.messages and isinstance(self.messages[0], SystemMessage):
            self.messages[0] = SystemMessage(system_prompt)
        else:
            self.messages.insert(0, SystemMessage(system_prompt))

    def _prepare_messages(self) -> List[Dict[str, str]]:
        """Convert internal message objects to dict format for LLM API."""
        return [
            {
                "role": role_conversions.get(message.role, message.role).value,
                "content": message.content
            }
            for message in self.messages
        ]

    def add_message(self, message: Message):
        """Add message with automatic history management."""
        self.messages.append(message)
        self.logger.debug("History length", f"Current history length: {len(self.messages)}/{self.max_history}", "yellow")
        self._trim_history()

    def _trim_history(self):
        """Trim message history if needed."""
        if len(self.messages) > self.max_history:
            # Always keep system message at index 0
            system_msg = self.messages[0]

            # Get all non-system messages
            non_system_messages = self.messages[1:]

            # Keep only the most recent (max_history - 1) non-system messages
            max_non_system_messages = self.max_history - 1

            if len(non_system_messages) > max_non_system_messages:
                recent_msgs = non_system_messages[-max_non_system_messages:]
            else:
                recent_msgs = non_system_messages

            # Reconstruct the message list
            self.messages = [system_msg] + recent_msgs

            self.logger.debug("History trimmed", f"Trimmed to {len(self.messages)}/{self.max_history} messages", "yellow")
