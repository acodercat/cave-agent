from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import AsyncGenerator
import logging

from .models import Model, ModelResponse, TokenUsage, PromptTooLongError, stream_with_idle_timeout
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
from .compaction import (
    CompactionState,
    compact_if_needed,
    full_compact_needed,
    recover_from_overflow,
)
from .utils import extract_python_code, sanitize_surrogates
from .types import (
    _ROLE_MAP, Message, SystemMessage, UserMessage,
    AssistantMessage, CodeExecutionMessage, ExecutionResultMessage,
    EventType, Event,
)

logger = logging.getLogger(__name__)

DEFAULT_PYTHON_BLOCK_IDENTIFIER = "python"

MAX_OUTPUT_RECOVERIES = 3
_RECOVERY_MESSAGE = (
    "Output limit hit. Resume directly, pick up mid-thought. "
    "Break remaining work into smaller pieces."
)


class ExecutionStatus(Enum):
    """Status of agent execution."""
    SUCCESS = "success"
    MAX_STEPS_REACHED = "max_steps_reached"
    TIMEOUT = "timeout"


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


class _RunState:
    """Mutable per-``run()`` state threaded through the step helpers."""

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        self.code_snippets: list[str] = []
        self.total_steps = 0
        self.token_usage = TokenUsage()
        self._completed = False
        self.output_recoveries = 0
        # Wall-clock origin for ``max_run_time``; stamped at construction,
        # which every caller does immediately before the run loop.
        self.start_time = time.monotonic()

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


@dataclass
class _ModelTurnResult:
    """Collected output from one streaming model call.

    Populated by :meth:`CaveAgent._stream_model` as it yields events, so the
    orchestrating step can act on the finished turn (execute the response,
    recover a truncated one, account usage) without re-reading the stream.
    """
    text: str = ""
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)


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
        max_run_time: Wall-clock budget in seconds for the whole run.
            Checked at each turn boundary; on exceed the run stops with
            ``ExecutionStatus.TIMEOUT``. Bounds a run across turns but does not
            interrupt a single in-flight step (use ``max_exec_timeout`` /
            ``stream_idle_timeout`` for that). ``None`` (default) disables it.
        max_exec_output: Maximum length of execution output.
        context_window: Model state window size in tokens for compaction.
        max_exec_timeout: Maximum seconds for a single code execution.
            None means no timeout. Note: with the in-process ``IPythonRuntime``,
            a timeout unblocks the agent but cannot forcibly stop CPU-bound code
            (``interrupt()`` is a no-op there), so the runaway cell keeps running
            in the background. Use ``IPyKernelRuntime`` for enforceable timeouts.
        stream_idle_timeout: Max seconds between two consecutive stream chunks
            before the streaming model call is abandoned (raises TimeoutError,
            salvaging any partial output). A sliding watchdog on chunk progress
            that catches stalls transport timeouts miss — proxy keep-alives and
            provider ping events keep the socket readable while no chunk is
            produced. Default 120.0; ``None`` disables it. Streaming path only.
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
        max_run_time: float | None = None,
        max_exec_output: int = 5000,
        context_window: int = 128_000,
        max_exec_timeout: float | None = None,
        stream_idle_timeout: float | None = 120.0,
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
        self.max_run_time = max_run_time
        self.max_exec_output = max_exec_output
        self.context_window = context_window
        self._compaction_state = CompactionState()
        self.max_exec_timeout = max_exec_timeout
        # Max seconds between two consecutive stream chunks before the model
        # call is abandoned. Catches stalls transport timeouts miss (proxy
        # keep-alives / provider pings keep the socket readable while no chunk
        # is produced). None disables the watchdog.
        self.stream_idle_timeout = stream_idle_timeout
        self.display = display
        self._init_skills(skills)
        # Last API-reported prompt-token count, used to sharpen compaction estimates.
        self._last_prompt_tokens: int = 0

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

    def _time_exceeded(self, state: _RunState) -> bool:
        """Whether the run has exceeded ``max_run_time`` (wall-clock).

        Checked at turn boundaries, so it bounds the run across turns but cannot
        interrupt a single in-flight step; a stuck tool/stream is bounded by
        ``max_exec_timeout`` / ``stream_idle_timeout`` instead.
        """
        if self.max_run_time is None:
            return False
        return (time.monotonic() - state.start_time) >= self.max_run_time

    async def _run(self, query: str) -> AgentResponse:
        """Non-streaming execution."""
        state = _RunState(self.max_steps)
        self._initialize_conversation(query)

        response = ""
        for _ in range(self.max_steps):
            if self._time_exceeded(state):
                return self._build_response(state, response, ExecutionStatus.TIMEOUT)
            response = await self._execute_step(state)
            if state.is_completed:
                return self._build_response(state, response, ExecutionStatus.SUCCESS)

        return self._build_response(state, response, ExecutionStatus.MAX_STEPS_REACHED)

    async def _run_with_display(self, query: str) -> AgentResponse:
        """Run via streaming to enable display, collect final response."""
        from .display import render_user_prompt
        render_user_prompt(query)

        state = _RunState(self.max_steps)
        last_content = ""

        timed_out = False
        async for event in self._wrap_with_display(self._stream_events(query, state), state):
            if event.type == EventType.FINAL_RESPONSE:
                last_content = event.content
            elif event.type == EventType.MAX_RUN_TIME_REACHED:
                timed_out = True

        if state.is_completed:
            status = ExecutionStatus.SUCCESS
        elif timed_out:
            status = ExecutionStatus.TIMEOUT
        else:
            status = ExecutionStatus.MAX_STEPS_REACHED
        return self._build_response(state, last_content, status)

    async def stream_events(self, query: str) -> AsyncGenerator[Event, None]:
        """Stream events during agent execution.

        When ``display=True``, events are printed to the terminal via Rich
        as they are yielded (transparent pass-through).
        """
        state = _RunState(self.max_steps)

        if self.display:
            async for event in self._wrap_with_display(self._stream_events(query, state), state):
                yield event
        else:
            async for event in self._stream_events(query, state):
                yield event

    async def _stream_events(self, query: str, state: _RunState) -> AsyncGenerator[Event, None]:
        """Internal event stream generator."""
        self._initialize_conversation(query)

        for _ in range(self.max_steps):
            if self._time_exceeded(state):
                yield Event(EventType.MAX_RUN_TIME_REACHED, "Maximum run time reached")
                return
            async for event in self._stream_step(state):
                yield event
            if state.is_completed:
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
        """Add a message to the conversation history."""
        self.messages.append(message)

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    async def _maybe_compact(self) -> tuple[str | None, int, int]:
        """Compact conversation history if over the token threshold.

        Returns (tier, before_count, after_count).
        """
        before = len(self.messages)
        self.messages, tier = await compact_if_needed(
            self.messages, self.model, self._compaction_state,
            context_window=self.context_window,
            api_token_count=self._last_prompt_tokens,
        )
        after = len(self.messages)
        if tier:
            logger.info("Context compacted (tier=%s, %d → %d messages)", tier, before, after)
        return tier, before, after

    async def _call_model_with_recovery(self) -> ModelResponse:
        """``model.call()`` with one reactive overflow recovery.

        If the API rejects the prompt as too long (:class:`PromptTooLongError`,
        raised by the translating retry layer), aggressively compact and retry
        once. A second overflow propagates — the preserved content alone
        exceeds the window, which compaction cannot fix.
        """
        try:
            return await self.model.call(self._prepare_messages())
        except PromptTooLongError:
            logger.warning("Prompt too long — attempting overflow recovery")
            self.messages = await recover_from_overflow(self.messages, self.model)
            return await self.model.call(self._prepare_messages())

    def _track_usage(self, state: _RunState, usage: TokenUsage) -> None:
        """Accumulate a turn's token usage and remember its prompt-token count
        (used to sharpen the next compaction estimate)."""
        state.add_token_usage(usage)
        if usage.prompt_tokens > 0:
            self._last_prompt_tokens = usage.prompt_tokens

    def _try_output_recovery(
        self, state: _RunState, finish_reason: str | None, text: str,
    ) -> bool:
        """Handle a length-truncated model turn.

        When the model stopped on ``finish_reason == "length"`` and the
        per-run recovery budget isn't spent, append the partial output plus a
        continuation prompt and return ``True`` — the caller ends the step so
        the next one resumes generation. Otherwise reset the counter and return
        ``False``. Shared by the streaming and non-streaming step paths.
        """
        if finish_reason == "length" and state.output_recoveries < MAX_OUTPUT_RECOVERIES:
            logger.warning(
                "Output truncated (recovery %d/%d)",
                state.output_recoveries + 1, MAX_OUTPUT_RECOVERIES,
            )
            self.add_message(AssistantMessage(text))
            self.add_message(UserMessage(_RECOVERY_MESSAGE))
            state.output_recoveries += 1
            return True
        state.output_recoveries = 0
        return False

    async def _execute_step(self, state: _RunState) -> str:
        """Run one non-streaming step and return the model's response text."""
        await self._maybe_compact()  # display handled by _stream_step only
        state.total_steps += 1

        response = await self._call_model_with_recovery()
        self._track_usage(state, response.token_usage)

        if self._try_output_recovery(state, response.finish_reason, response.content):
            return response.content
        return await self._process_response(response.content, state)

    async def _stream_step(self, state: _RunState) -> AsyncGenerator[Event, None]:
        """Run one streaming step: proactive compaction, one (self-recovering)
        model call, then execute the response — or continue a truncated one."""
        # Announce compaction only when the slow LLM-summarization tier is
        # imminent — a zero-cost microcompact shouldn't show a spinner.
        before = len(self.messages)
        if full_compact_needed(self.messages, self.context_window, self._last_prompt_tokens):
            yield Event(EventType.COMPACTING, "")
        tier, _, after = await self._maybe_compact()
        if tier == "full_compact":
            yield Event(EventType.COMPACTED, f"{before} → {after}")

        state.total_steps += 1

        result = _ModelTurnResult()
        async for event in self._stream_model_with_recovery(result):
            yield event
        self._track_usage(state, result.usage)

        if self._try_output_recovery(state, result.finish_reason, result.text):
            return
        async for event in self._process_response_stream(result.text, state):
            yield event

    async def _stream_model_with_recovery(
        self, result: _ModelTurnResult,
    ) -> AsyncGenerator[Event, None]:
        """:meth:`_stream_model` with one reactive overflow recovery.

        On :class:`PromptTooLongError` — raised at request time, before any
        event — aggressively compact and re-stream once. A second overflow
        propagates: the preserved content alone exceeds the window.
        """
        try:
            async for event in self._stream_model(result):
                yield event
        except PromptTooLongError:
            logger.warning("Prompt too long — attempting overflow recovery")
            yield Event(EventType.COMPACTING, "")
            before = len(self.messages)
            self.messages = await recover_from_overflow(self.messages, self.model)
            yield Event(EventType.COMPACTED, f"{before} → {len(self.messages)}")
            async for event in self._stream_model(result):
                yield event

    async def _stream_model(
        self, result: _ModelTurnResult,
    ) -> AsyncGenerator[Event, None]:
        """Stream one model call: yield ``THINKING_CHUNK``/``THINKING``/``TEXT``/
        ``CODE`` events and populate ``result`` with the collected text,
        finish_reason, and usage.

        Reasoning models stream their reasoning first: each reasoning delta is
        emitted live as ``THINKING_CHUNK``; when the first answer token arrives,
        the segment is sealed with a single ``THINKING`` carrying the full trace.

        Propagates :class:`PromptTooLongError` (raised at request time, before
        any chunk) for :meth:`_stream_model_with_recovery` to handle. A
        mid-stream failure (dropped connection, idle-timeout stall) is salvaged
        into whatever streamed so far rather than crashing the run.
        """
        chunks: list[str] = []
        parser = StreamingTextParser(self.python_block_identifier)
        stream_response = self.model.stream(self._prepare_messages())
        # The idle watchdog wraps iteration; usage/finish_reason are still read
        # off the original ``stream_response`` (the wrapper drives its
        # ``__anext__``, which updates those attributes).
        guarded = stream_with_idle_timeout(stream_response, self.stream_idle_timeout)
        thinking_sealed = False
        try:
            async for delta in guarded:
                if delta.thinking:
                    yield Event(EventType.THINKING_CHUNK, delta.thinking)
                if not delta.content:
                    continue
                # First answer token: seal the reasoning segment with the full
                # accumulated trace (for consumers that don't track chunks).
                if not thinking_sealed and stream_response.thinking:
                    yield Event(EventType.THINKING, stream_response.thinking)
                    thinking_sealed = True
                chunks.append(delta.content)

                for segment in parser.process_chunk(delta.content):
                    if segment.type == SegmentType.TEXT:
                        yield Event(EventType.TEXT, segment.content)
                    elif segment.type == SegmentType.CODE:
                        yield Event(EventType.CODE, segment.content)
                        if parser.is_first_code_block_completed():
                            break
                if parser.is_first_code_block_completed():
                    break
        except PromptTooLongError:
            # Request-time overflow (before any chunk): let the recovery wrapper
            # compact and re-stream. It never fires mid-stream — the model layer
            # only translates the request-initiation error to this type.
            raise
        except Exception:
            logger.warning("Streaming interrupted mid-response — continuing with partial output", exc_info=True)
            if not chunks:
                raise
        finally:
            # Close the underlying stream on every exit path — early break (code
            # block complete), completion, overflow, or error. The SDK's
            # read-to-completion auto-close doesn't fire when we stop reading early.
            await stream_response.aclose()

        if not parser.is_first_code_block_completed():
            for segment in parser.flush():
                if segment.type == SegmentType.TEXT:
                    yield Event(EventType.TEXT, segment.content)
                elif segment.type == SegmentType.CODE:
                    yield Event(EventType.CODE, segment.content)

        result.text = "".join(chunks)
        result.finish_reason = stream_response.finish_reason
        result.usage = stream_response.usage

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    async def _extract_and_execute(
        self, model_response: str, state: _RunState,
    ) -> _ExecutionOutcome | None:
        """Extract code from model response and execute it if present.

        Returns the execution outcome, or None if no code was found
        (in which case the response is added as an AssistantMessage and
        state is marked complete).
        """
        code_snippet = extract_python_code(model_response, self.python_block_identifier)
        if not code_snippet:
            self.add_message(AssistantMessage(model_response))
            state.complete()
            return None

        self.add_message(CodeExecutionMessage(model_response))
        outcome = await self._execute_code(code_snippet, state)
        self.add_message(ExecutionResultMessage(outcome.next_prompt))
        return outcome

    async def _process_response(self, model_response: str, state: _RunState) -> str:
        """Process model response and execute code if present."""
        await self._extract_and_execute(model_response, state)
        return model_response

    async def _process_response_stream(
        self,
        model_response: str,
        state: _RunState,
    ) -> AsyncGenerator[Event, None]:
        """Process model response with streaming events."""
        outcome = await self._extract_and_execute(model_response, state)
        if outcome is None:
            yield Event(EventType.FINAL_RESPONSE, model_response)
        else:
            yield Event(outcome.event_type, outcome.event_content)

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------

    async def _execute_code(
        self,
        code_snippet: str,
        state: _RunState,
    ) -> _ExecutionOutcome:
        """Execute code snippet and return the outcome."""
        state.code_snippets.append(code_snippet)

        execution_result = (
            await self._execute_with_timeout(code_snippet)
            if self.max_exec_timeout is not None
            else await self.runtime.execute(code_snippet)
        )

        if execution_result is None:
            return _ExecutionOutcome(
                event_type=EventType.EXECUTION_ERROR,
                event_content=f"Execution timed out after {self.max_exec_timeout}s",
                next_prompt=f"Code execution timed out after {self.max_exec_timeout} seconds. "
                    "Simplify your code or break it into smaller steps.",
            )

        # Security error
        if not execution_result.success and isinstance(execution_result.error, SecurityError):
            error_message = execution_result.error.message
            return _ExecutionOutcome(
                event_type=EventType.SECURITY_ERROR,
                event_content=error_message,
                next_prompt=SECURITY_ERROR_PROMPT.format(error=error_message),
            )

        # Scrub lone surrogates from execution stdout (binary bytes printed by
        # user code, half-decoded files, etc.) before it enters a message and
        # the next model call's json.dumps would choke on it.
        stdout = sanitize_surrogates(execution_result.stdout or "No output")

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

        On timeout, calls ``runtime.interrupt()`` to stop the execution.
        This is effective for IPyKernelRuntime (SIGINT to the kernel). For
        the in-process IPythonRuntime, ``interrupt()`` is a no-op — the worker
        thread keeps running the code to completion in the background; only the
        agent's *wait* is bounded. Prefer IPyKernelRuntime when you need to
        actually cancel runaway execution.

        Returns None on timeout, ExecutionResult otherwise.
        """
        loop = asyncio.get_running_loop()
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
        state: _RunState,
    ) -> AsyncGenerator[Event, None]:
        """Wrap events with terminal display.

        Lazy import: display.py imports Event/EventType from agent.py,
        so agent.py cannot import display.py at module level.
        """
        from .display import with_display
        async for event in with_display(events, state):
            yield event

    def _init_skills(self, skills: list[Skill] | None = None) -> None:
        self._skill_registry = SkillRegistry()
        if skills:
            self._skill_registry.add_skills([s for s in skills if s is not None])
        if self._skill_registry.list_skills():
            store = self._skill_registry.build_skill_store()
            self.runtime.inject_into_namespace("_skill_store", store)
            self.runtime.inject_function(Function(activate_skill))
            self.system_instructions += "\n" + SKILLS_INSTRUCTION

    def _initialize_conversation(self, user_query: str):
        self._update_system_message()
        # Scrub lone surrogates from the user's turn (rich-text paste, broken
        # emoji pipeline) before it enters history and the wire.
        self.add_message(UserMessage(sanitize_surrogates(user_query)))

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

    def _build_response(
        self,
        state: _RunState,
        content: str,
        status: ExecutionStatus,
    ) -> AgentResponse:
        return AgentResponse(
            content=content,
            code_snippets=state.code_snippets,
            status=status,
            steps_taken=state.total_steps,
            max_steps=self.max_steps,
            token_usage=state.token_usage,
        )
