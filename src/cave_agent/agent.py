from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ._placeholders import (
    EMPTY_OUTPUT_PLACEHOLDER,
    EXECUTION_CANCELLED_AFTER_STATE_LOSS_PLACEHOLDER,
    EXECUTION_CANCELLED_PLACEHOLDER,
    EXECUTION_INCOMPLETE_PLACEHOLDER,
    EXECUTION_TIMEOUT_PROMPT,
    OUTPUT_RECOVERY_PROMPT,
    PERSISTED_OUTPUT_PREFIX,
    RUNTIME_STATE_LOST_NOTICE,
    STREAM_INTERRUPTED_PROMPT,
    USER_INTERRUPTION_PLACEHOLDER,
    build_persist_marker,
    highest_persisted_index,
    normalize_identifier,
)
from .compaction import (
    DEFAULT_CONTEXT_WINDOW,
    Compactor,
    TokenAnchor,
    default_token_estimate,
)
from .events import (
    CodeEvent,
    Event,
    ExecutionResultEvent,
    ExecutionTimeoutEvent,
    FinalResponseEvent,
    SecurityErrorEvent,
    StatusEvent,
    StatusType,
    StoppedEvent,
    StopReason,
    TextEvent,
    ThinkingChunkEvent,
    ThinkingEvent,
    UserPromptEvent,
)
from .messages import (
    AssistantMessage,
    CodeExecutionMessage,
    ExecutionResultMessage,
    Message,
    SystemMessage,
    UserMessage,
    to_wire,
)
from .models import (
    Model,
    PromptTooLongError,
    ProviderError,
    StreamResponse,
    TokenUsage,
    stream_with_idle_timeout,
)
from .models.errors import ModelError, raise_model_error
from .parsing import SegmentType, StreamingTextParser
from .prompts import (
    DEFAULT_INSTRUCTIONS,
    DEFAULT_SYSTEM_INSTRUCTIONS,
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    EXECUTION_OUTPUT_PROMPT,
    SECURITY_ERROR_PROMPT,
    SKILLS_INSTRUCTION,
)
from .runtime import Function, IPythonRuntime, PreemptibleRuntime, Runtime
from .runtime.builtins import activate_skill
from .runtime.executor import (
    ExecutionResult,
    RuntimeExecutionError,
    RuntimeStateLostError,
)
from .security import SecurityError
from .skills import Skill, SkillRegistry
from .utils import sanitize_surrogates

logger = logging.getLogger(__name__)

DEFAULT_PYTHON_BLOCK_IDENTIFIER = "python"
MAX_OUTPUT_RECOVERIES = 3

# Grace period for releasing a provider stream nobody will read again. Short
# on purpose: it sits between a finished turn and executing its code. Closing
# an HTTP connection is immediate in practice, so a close still running after
# this is hung rather than slow, and waiting longer buys nothing.
_STREAM_CLOSE_TIMEOUT = 1.0


@dataclass
class AgentResponse:
    """The outcome of a :meth:`CaveAgent.run` call."""

    content: str
    """The model's final answer, or the last thing it said before stopping."""

    stop_reason: StopReason
    steps: int
    elapsed: float
    usage: TokenUsage = field(default_factory=TokenUsage)
    code_snippets: list[str] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        """Whether the run ended by answering, rather than hitting a limit."""
        return self.stop_reason is StopReason.COMPLETED

    def __str__(self) -> str:
        return (
            f"AgentResponse(stop_reason={self.stop_reason.value}, "
            f"steps={self.steps}, tokens={self.usage.total_tokens}, "
            f"content={self.content})"
        )


async def drain_to_response(events: AsyncIterator[Event]) -> AgentResponse:
    """Consume an event stream and assemble the run's :class:`AgentResponse`.

    Shared by :meth:`CaveAgent.run` and ``renderers.render_run`` so the two
    cannot drift as ``AgentResponse`` grows fields.
    """
    response: AgentResponse | None = None
    code_snippets: list[str] = []
    async for event in events:
        if isinstance(event, CodeEvent):
            code_snippets.append(event.code)
        elif isinstance(event, StoppedEvent):
            response = AgentResponse(
                content=event.content,
                stop_reason=event.stop_reason,
                steps=event.steps,
                elapsed=event.elapsed,
                usage=event.usage,
                code_snippets=code_snippets,
            )
    if response is None:
        raise RuntimeError("event stream ended without a StoppedEvent")
    return response


def _conclude_turn(
    turn: _ModelTurn,
    stream: StreamResponse,
    *,
    stopped_at_code: bool,
    defer_execution: bool,
) -> TextEvent | None:
    """Decide what this turn actually was, once the stream is done.

    A pure reading of the provider's terminal state against what was collected:
    an answer, a refusal, a fragment that may be resumed, or a failure. Returns
    the event to emit for a refusal, if any — the caller yields it, because this
    is a decision, not a stream.

    The order matters. Each branch answers "is there an answer here?" for a
    different reason, and the earlier ones describe states the later ones would
    misread as ordinary emptiness.
    """
    if turn.error is None and stream.finish_reason is None and not stopped_at_code:
        # Natural EOF is transport state, not a provider completion signal.
        # The one legitimate missing finish reason is our own early break
        # after a closed code block.
        turn.error = ProviderError("The model stream ended without a terminal finish reason.")
        turn.recoverable = not defer_execution and bool(turn.text)
        return None

    if stream.finish_reason == "content_filter" and turn.text:
        # Content *and* a filter verdict: the answer was cut off, not given.
        # Recording it presents a fragment the provider withdrew as though the
        # model had finished saying it.
        turn.error = ProviderError(
            "The provider filtered this response after partially streaming it."
        )
        turn.recoverable = False
        return None

    if stream.refusal and not turn.text:
        # A refusal is the answer when the model produced no other one, and
        # providers send it under an ordinary "stop" as readily as under
        # "content_filter". Assigned rather than streamed through the parser:
        # it is prose, and a fenced block inside it is quoted text, not code.
        turn.text = stream.refusal
        return TextEvent(stream.refusal)

    if stream.finish_reason == "content_filter":
        # Filtered with nothing to show for it: no answer here, and the run
        # must not report one.
        turn.error = ProviderError("The provider filtered this response and returned no content.")
        turn.recoverable = False
        return None

    if stream.finish_reason != "length" and not turn.text.strip() and turn.code is None:
        turn.error = ProviderError("The model completed without returning an answer.")
        turn.recoverable = False
    return None


async def _close_stream(stream) -> None:
    """Release *stream* without outranking or outlasting the run.

    Called from a ``finally``, where a raise would replace the stream outcome.
    Closing is cleanup: it cannot invalidate an answer the provider already
    declared complete or replace the read error that ended the stream, so its
    failures are logged rather than propagated. ``sys.exc_info()`` is read
    before the first await, while it still reflects whether cancellation is
    unwinding.

    The close also runs as a task the caller stops waiting on, so a hung closer
    cannot hold the loop between a finished turn and executing its code. When
    the run is already cancelled the caller does not wait at all — but the task
    still gets its grace period before being cancelled, because a close that
    never ran leaves the socket open.
    """
    unwinding = sys.exc_info()[0]
    closing = asyncio.ensure_future(stream.aclose())

    if unwinding is not None and issubclass(unwinding, asyncio.CancelledError):
        _detach(
            closing,
            "cancelled — not waiting for the provider stream to close",
            stop_after=_STREAM_CLOSE_TIMEOUT,
        )
        return

    try:
        await asyncio.wait_for(asyncio.shield(closing), _STREAM_CLOSE_TIMEOUT)
    except TimeoutError:
        _detach(closing, f"provider stream still closing after {_STREAM_CLOSE_TIMEOUT}s")
    except asyncio.CancelledError:
        _detach(
            closing, "cancelled while closing the provider stream", stop_after=_STREAM_CLOSE_TIMEOUT
        )
        raise
    except Exception:
        logger.warning("Failed to close the provider stream", exc_info=True)


def _detach(task: asyncio.Task, reason: str, *, stop_after: float = 0.0) -> None:
    """Stop waiting on *task*, and cancel it once *stop_after* has elapsed.

    ``stop_after=0`` cancels immediately, for a task that has already had its
    grace. The done-callback consumes the outcome so an abandoned task does not
    surface as "Task exception was never retrieved" — a confusing report about
    cleanup nobody was waiting for.
    """
    logger.warning("%s", reason)
    if stop_after > 0:
        asyncio.get_running_loop().call_later(stop_after, task.cancel)
    else:
        task.cancel()
    task.add_done_callback(lambda finished: finished.cancelled() or finished.exception())


@dataclass
class _ModelTurn:
    """Output collected from one streaming model call.

    ``text`` is what *this* call produced. ``prefix`` is what earlier calls in
    the same logical response produced, when the model was cut off and asked to
    resume. Parsing reads :attr:`full`; history records ``text``, because the
    prefix was already written by the turn that produced it.
    """

    text: str = ""
    prefix: str = ""
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    code: str | None = None
    """The completed code block, captured by the live parser at the moment it
    closed. Carried rather than re-derived: a second parse of the same text
    cannot tell a closed fence from an unterminated one, and used to execute
    both."""

    error: Exception | None = None
    """Set when the stream died after some content arrived — the turn is a
    fragment, and this is why."""

    recoverable: bool = True
    """Whether output recovery may persist and resume this fragment.

    Transport interruption and ``length`` are recoverable. A content-filter
    verdict — or an asynchronously filtered stream that ended without any
    verdict — is not: persisting and sending that content back would retain
    output the provider withdrew, and a later request cannot retroactively
    validate it.
    """

    @property
    def full(self) -> str:
        return self.prefix + self.text

    @property
    def complete(self) -> bool:
        """Whether this is the model's whole utterance.

        False for both ways a prose turn can be a fragment: the provider's
        output cap (``finish_reason == "length"``) and a stream that stopped
        producing. A closed code block is already the whole part this agent
        consumes; a later ``length`` only describes trailing text the normal
        early-stop path would never have read.

        An early break at the first complete code block leaves
        ``finish_reason`` unset and is complete *by design* — the agent chose to
        stop reading.
        """
        return self.error is None and (self.code is not None or self.finish_reason != "length")

    @property
    def resume_prompt(self) -> str:
        return STREAM_INTERRUPTED_PROMPT if self.error else OUTPUT_RECOVERY_PROMPT


class _RunState:
    """Mutable per-``run()`` state threaded through the step helpers."""

    def __init__(self) -> None:
        self.code_snippets: list[str] = []
        self.steps = 0
        self.usage = TokenUsage()
        self.output_recoveries = 0
        # Text carried across a truncation boundary, so the next turn is parsed
        # together with what preceded it.
        self.pending_text = ""
        self.final_content = ""
        self.completed = False
        self.start_time = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def add_usage(self, usage: TokenUsage) -> None:
        self.usage = self.usage + usage


class CaveAgent:
    """An agent that calls tools by writing Python against a persistent runtime.

    Instead of JSON schemas, the model writes code; the code runs in a live
    namespace where injected objects stay resident across turns. State does not
    round-trip through the context window, so a DataFrame stays a DataFrame.

    The loop: stream a response, stop at the first complete code block, execute
    it, feed the result back, repeat until the model answers without code or a
    limit is reached.

    Rendering lives outside the agent. To display a run, wrap it::

        from cave_agent.renderers import TerminalRenderer
        async for event in TerminalRenderer().render(agent.stream_events(q)):
            ...

    Args:
        model: LLM implementing the :class:`~cave_agent.models.Model` interface.
        runtime: Python runtime holding the injected functions and variables.
            Defaults to an in-process :class:`IPythonRuntime`.
        instructions: User-facing instructions defining the agent's role.
        skills: Skills to make available via ``activate_skill``.
        max_steps: Model calls allowed before the run stops unanswered.
        max_run_time: Wall-clock budget in seconds, checked at step boundaries.
            Bounds the run across steps but cannot interrupt one in flight —
            ``max_exec_timeout`` and ``stream_idle_timeout`` do that.
        max_exec_output: Characters of execution output inlined into the
            conversation. Beyond this the full text is kept in the runtime (see
            ``persisted_output_prefix``) and the model receives a marker with
            a preview, so nothing is lost and no re-run is needed.
        exec_output_preview_chars: Leading characters of an oversize output
            shown inline with that marker. 0 suppresses the preview.
        persisted_output_prefix: Stem for the namespace names holding oversize
            outputs. Each one gets its own (``_output_1``, ``_output_2``, …), so
            a marker the model read three turns ago still points at the text it
            described.
        max_exec_timeout: Deadline for a single execution. Requires a
            preemptible runtime (one implementing ``interrupt()``, such as
            ``IPyKernelRuntime``); after the deadline, kernel recovery may add
            a short delay before the timeout event is emitted. Constructing
            with an in-process runtime raises ``ValueError`` rather than
            offering a timeout it cannot honour.
        stream_idle_timeout: Max seconds between two stream chunks before the
            model call is abandoned. A sliding watchdog on chunk progress,
            catching stalls that transport timeouts miss — proxy keep-alives
            and provider pings keep a socket readable while producing nothing.
        max_total_tokens / max_input_tokens / max_output_tokens: Cumulative
            token budgets for the whole run. Crossing one stops the run with
            ``StopReason.BUDGET_EXHAUSTED`` before the next step begins; the
            crossing step still completes and is recorded.
        compactor: Compaction policy. Defaults to a :class:`Compactor` over the
            same model with ``context_window``.
        context_window: Convenience for the default compactor's window. Ignored
            when ``compactor`` is passed.
        system_instructions: Execution rules and examples spliced into the
            system prompt below ``instructions``.
        system_prompt_template: Template receiving ``instructions``,
            ``system_instructions``, ``functions``, ``variables``, ``types``
            and ``skills``.
        python_block_identifier: Fence language the model must use, and the one
            the parser looks for. Both sides read this, so changing it stays
            consistent.
        messages: Prior conversation to resume from.

    Example:
        >>> agent = CaveAgent(model, runtime=IPythonRuntime(functions=[Function(add)]))
        >>> response = await agent.run("Add 5 and 3")
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
        exec_output_preview_chars: int = 2000,
        persisted_output_prefix: str = PERSISTED_OUTPUT_PREFIX,
        max_exec_timeout: float | None = None,
        stream_idle_timeout: float | None = 120.0,
        max_total_tokens: int | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
        compactor: Compactor | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        python_block_identifier: str = DEFAULT_PYTHON_BLOCK_IDENTIFIER,
        messages: list[Message] | None = None,
    ):
        self.model = model
        self.runtime = runtime if runtime is not None else IPythonRuntime()
        self.instructions = instructions
        self.system_prompt_template = system_prompt_template
        self.system_instructions = system_instructions.format(
            python_block_identifier=python_block_identifier,
        )
        self.python_block_identifier = python_block_identifier
        self.messages: list[Message] = list(messages) if messages else []

        self.max_steps = max_steps
        self.max_run_time = max_run_time
        self.max_exec_output = max_exec_output
        self.exec_output_preview_chars = exec_output_preview_chars
        self.max_exec_timeout = max_exec_timeout
        self.stream_idle_timeout = stream_idle_timeout
        self.max_total_tokens = max_total_tokens
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

        # The prefix becomes a Python name the model is told to slice, so it has
        # to be one — rejected here rather than at the first oversize output,
        # which is a long way from the argument that caused it. Normalized first
        # because that is what Python does when it binds a name, so markers,
        # history scanning and allocation all name the variable that will exist.
        self.persisted_output_prefix = normalize_identifier(persisted_output_prefix)
        if not f"{self.persisted_output_prefix}_1".isidentifier():
            raise ValueError(
                f"persisted_output_prefix must be able to form a Python identifier, "
                f"and {persisted_output_prefix!r} cannot: it would name the variable "
                f"'{self.persisted_output_prefix}_1'."
            )

        if max_exec_timeout is not None and not isinstance(self.runtime, PreemptibleRuntime):
            raise ValueError(
                f"max_exec_timeout needs a runtime that can be preempted, and "
                f"{type(self.runtime).__name__} is not one: it runs generated code "
                "in this process, where a deadline can abandon the code but never "
                "stop it — and abandoning it corrupts the process's stdout. "
                "Use IPyKernelRuntime, or drop max_exec_timeout."
            )

        self.compactor = compactor or Compactor(model, context_window=context_window)

        # Floor for persisted-output names. Names already handed to the model
        # in a resumed history are off limits — reusing one would re-point a
        # marker the model still trusts — and they live in the conversation,
        # not in the runtime's namespace, so the runtime cannot know them.
        # Everything else about allocation is the runtime's job: it is what
        # sees the bindings, and what two agents would share.
        self._persisted_floor = (
            highest_persisted_index(
                (message.content for message in self.messages),
                self.persisted_output_prefix,
            )
            + 1
        )

        self._init_skills(skills)
        # Last API-reported prompt-token count, used to sharpen compaction
        # estimates once the first real number is available.
        # Last API-reported prompt size paired with the heuristic reading at
        # that moment, so compaction can add what has been appended since.
        self._token_anchor: TokenAnchor | None = None

    async def run(self, query: str) -> AgentResponse:
        """Execute the agent and return the final response.

        Drains :meth:`stream_events` — there is one execution path, so a run
        behaves identically whether or not the caller consumes events.
        """
        return await drain_to_response(self.stream_events(query))

    async def stream_events(self, query: str) -> AsyncGenerator[Event, None]:
        """Stream typed events as the agent works.

        When consumed to termination, ends with a :class:`StoppedEvent`,
        including on cancellation. If the consumer closes the generator early,
        no event can legally be yielded during ``GeneratorExit``; the history
        is still marked interrupted so a follow-up run remains coherent.
        """
        state = _RunState()
        yield UserPromptEvent(query)

        stop_reason = StopReason.COMPLETED
        try:
            # Inside the envelope: rendering the system prompt runs a caller's
            # template and the runtime's describe_*(), either of which can
            # raise. Outside, that raise escaped with no terminal event and
            # made "always ends with a StoppedEvent" false at the one moment a
            # consumer has nothing else to go on.
            self._initialize_conversation(query)
            async for event in self._loop(state):
                yield event
            stop_reason = self._terminal_reason(state)
        except GeneratorExit:
            # The consumer abandoned the stream. Yielding inside a GeneratorExit
            # handler is illegal ("async generator ignored GeneratorExit"), so
            # this arm only repairs history. It must stay ABOVE the BaseException
            # arm, which does yield — otherwise explicitly closing an abandoned
            # iterator raises RuntimeError instead of completing cleanly.
            if not state.completed:
                self._close_dangling_execution(EXECUTION_INCOMPLETE_PLACEHOLDER)
                self._mark_user_interruption()
            raise
        except asyncio.CancelledError as error:
            placeholder = (
                EXECUTION_CANCELLED_AFTER_STATE_LOSS_PLACEHOLDER
                if isinstance(error, RuntimeStateLostError)
                else EXECUTION_CANCELLED_PLACEHOLDER
            )
            self._close_dangling_execution(placeholder)
            self._mark_user_interruption()
            yield self._stopped(state, StopReason.CANCELLED)
            raise
        except ModelError:
            self._close_dangling_execution(EXECUTION_INCOMPLETE_PLACEHOLDER)
            logger.exception("Model error — ending run")
            yield self._stopped(state, StopReason.MODEL_ERROR)
            return
        except RuntimeExecutionError:
            self._close_dangling_execution(EXECUTION_INCOMPLETE_PLACEHOLDER)
            logger.exception("Runtime error — ending run")
            yield self._stopped(state, StopReason.RUNTIME_ERROR)
            return
        except BaseException:
            # A bug in cave-agent. Still emit the terminal event so a consumer
            # can close its own stream, then let the exception through — it
            # should be seen, not swallowed.
            self._close_dangling_execution(EXECUTION_INCOMPLETE_PLACEHOLDER)
            logger.exception("Unclassified failure — ending run")
            yield self._stopped(state, StopReason.INTERNAL_ERROR)
            raise

        yield self._stopped(state, stop_reason)

    def add_message(self, message: Message) -> None:
        """Append a message to the conversation history."""
        self.messages.append(message)

    def build_system_prompt(self) -> str:
        """Build the system prompt from the runtime's current registrations."""
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

    async def _loop(self, state: _RunState) -> AsyncGenerator[Event, None]:
        for _ in range(self.max_steps):
            if self._time_exceeded(state) or self._budget_exhausted(state):
                return
            async for event in self._step(state):
                yield event
            if state.completed:
                return

    def _terminal_reason(self, state: _RunState) -> StopReason:
        """Classify why the loop stopped, in precedence order."""
        if state.completed:
            return StopReason.COMPLETED
        if self._budget_exhausted(state):
            return StopReason.BUDGET_EXHAUSTED
        if self._time_exceeded(state):
            return StopReason.TIMEOUT
        return StopReason.MAX_STEPS

    def _stopped(self, state: _RunState, reason: StopReason) -> StoppedEvent:
        return StoppedEvent(
            content=state.final_content,
            stop_reason=reason,
            steps=state.steps,
            elapsed=state.elapsed,
            usage=state.usage,
        )

    def _time_exceeded(self, state: _RunState) -> bool:
        """Whether ``max_run_time`` has elapsed.

        Checked at step boundaries, so it bounds the run across steps but
        cannot interrupt one in flight.
        """
        if self.max_run_time is None:
            return False
        return state.elapsed >= self.max_run_time

    def _budget_exhausted(self, state: _RunState) -> bool:
        """Whether any cumulative token budget has been crossed.

        Compaction's own LLM calls are not counted — they are overhead of the
        loop rather than the user's visible work, so budget them with headroom
        on compaction-heavy sessions.
        """
        usage = state.usage
        checks = (
            (self.max_total_tokens, usage.total_tokens),
            (self.max_input_tokens, usage.prompt_tokens),
            (self.max_output_tokens, usage.completion_tokens),
        )
        return any(cap is not None and spent >= cap for cap, spent in checks)

    async def _step(self, state: _RunState) -> AsyncGenerator[Event, None]:
        """One step: compact if needed, stream a turn, then execute or finish."""
        async for event in self._maybe_compact():
            yield event

        state.steps += 1

        turn = _ModelTurn(prefix=state.pending_text)
        try:
            async for event in self._stream_turn(turn):
                yield event
        finally:
            # A provider can fail after emitting reasoning or refusal but before
            # content. That call was still billed, so account for it even though
            # the model error now propagates out of this step.
            state.add_usage(turn.usage)
            if turn.usage.prompt_tokens > 0:
                # Anchored to the history as it stood when the provider counted
                # it, before this turn's own output is appended below.
                self._token_anchor = self.compactor.anchor(
                    self.messages,
                    turn.usage.prompt_tokens,
                )

        if turn.error is not None and not turn.recoverable:
            # A filtered fragment is not something to preserve and ask the
            # model to continue: that writes withdrawn content into history,
            # may repeat the filtered request several times, and can terminate
            # as MAX_STEPS before the recovery budget finally surfaces the real
            # provider failure.
            raise turn.error

        if not turn.complete and state.output_recoveries < MAX_OUTPUT_RECOVERIES:
            # A fragment — capped by the provider's output limit, or cut off by
            # a dying stream. Same remedy either way: keep it in history so the
            # model can see what it already said, carry it forward so the next
            # turn is parsed *together with* it, and ask it to resume.
            state.output_recoveries += 1
            state.pending_text = turn.full
            self.add_message(AssistantMessage(turn.text))
            self.add_message(UserMessage(turn.resume_prompt))
            yield StatusEvent(
                StatusType.OUTPUT_RECOVERY,
                recovery=state.output_recoveries,
                max_recoveries=MAX_OUTPUT_RECOVERIES,
            )
            return

        state.pending_text = ""
        async for event in self._handle_turn(turn, state):
            yield event

    async def _maybe_compact(self) -> AsyncGenerator[Event, None]:
        """Run compaction if the conversation is over budget.

        The cheap tier runs silently; only the LLM tier announces itself, since
        a microcompact costs nothing and a spinner for it would be noise.
        """
        before = len(self.messages)
        messages, needs_summarize = self.compactor.microcompact(
            self.messages,
            anchor=self._token_anchor,
        )
        self.messages = messages
        if not needs_summarize:
            return

        yield StatusEvent(StatusType.COMPACTING)
        self.messages = await self.compactor.summarize(self.messages)
        yield StatusEvent(
            StatusType.COMPACTED,
            before=before,
            after=len(self.messages),
        )

    async def _recover_from_overflow(self) -> AsyncGenerator[Event, None]:
        yield StatusEvent(StatusType.COMPACTING)
        before = len(self.messages)
        self.messages = await self.compactor.recover_from_overflow(self.messages)
        yield StatusEvent(
            StatusType.COMPACTED,
            before=before,
            after=len(self.messages),
        )

    async def _stream_turn(self, turn: _ModelTurn) -> AsyncGenerator[Event, None]:
        """Stream one model call, recovering once from a prompt overflow.

        :class:`PromptTooLongError` is raised at request time, before any
        chunk, so compacting and re-streaming cannot duplicate output. A second
        overflow propagates: the preserved content alone exceeds the window,
        which compaction cannot fix.
        """
        try:
            async for event in self._stream_once(turn):
                yield event
        except PromptTooLongError:
            logger.warning("Prompt too long — attempting overflow recovery")
            async for event in self._recover_from_overflow():
                yield event
            async for event in self._stream_once(turn):
                yield event

    async def _stream_once(self, turn: _ModelTurn) -> AsyncGenerator[Event, None]:
        """Stream a single model call into *turn*, emitting text/thinking events.

        Stops reading as soon as the first code block completes: everything the
        model would have written after it was composed against imagined output,
        so it is cheaper and more accurate to re-derive it next turn from the
        real result.

        The parser is seeded with ``turn.prefix`` so a block that straddles a
        truncation boundary still closes. Seeding cannot itself complete a
        block: once a block closes it is complete even if an asynchronously
        filtered provider later reports ``length`` for ignored trailing text;
        filtered or unverified content is non-recoverable.
        """
        turn.error = None
        turn.code = None
        # Providers that filter asynchronously may reject a block *after*
        # streaming it, so with one of those the stream is read to the end
        # before anything runs. Breaking early would execute code the verdict
        # was about to withdraw.
        defer_execution = getattr(self.model, "filters_asynchronously", False)
        stopped_at_code = False
        chunks: list[str] = []
        received_content: list[str] = []
        parser = StreamingTextParser(self.python_block_identifier)
        if turn.prefix:
            # Segments from the prefix were emitted by the turn that produced it.
            parser.process_chunk(turn.prefix)
        wire = self._prepare_messages()
        try:
            # ``stream()`` is a synchronous factory, so a provider that rejects
            # the request outright — bad key, unknown model, oversize prompt —
            # raises here, outside the loop below that classifies everything
            # else. Unwrapped, those arrived as INTERNAL_ERROR: "the agent
            # broke", for the one class of failure that is squarely the
            # provider's.
            stream = self.model.stream(wire)
        except Exception as error:
            raise_model_error(error)
        # The watchdog wraps iteration; usage and finish_reason are still read
        # off the original stream object, whose __anext__ the wrapper drives.
        guarded = stream_with_idle_timeout(stream, self.stream_idle_timeout)

        thinking_started: float | None = None
        thinking_sealed = False
        fatal_stream_error: Exception | None = None

        def seal_thinking() -> ThinkingEvent | None:
            nonlocal thinking_sealed
            if thinking_sealed or not stream.thinking:
                return None
            thinking_sealed = True
            started = thinking_started if thinking_started is not None else time.monotonic()
            return ThinkingEvent(
                stream.thinking,
                int((time.monotonic() - started) * 1000),
            )

        try:
            async for delta in guarded:
                if delta.thinking:
                    if thinking_started is None:
                        thinking_started = time.monotonic()
                    yield ThinkingChunkEvent(delta.thinking)
                if not delta.content:
                    continue
                # Accounting follows the transport, not the parser. A buffered
                # chunk may contain a complete code block plus a long tail the
                # conversation deliberately discards, and asynchronously
                # filtered providers are drained after the block. Those tokens
                # were still received and billed.
                received_content.append(delta.content)
                # First answer token ends the reasoning segment. Measuring the
                # duration here — rather than letting consumers diff event
                # timestamps — keeps a later stall from being counted as
                # thinking time.
                thinking_event = seal_thinking()
                if thinking_event is not None:
                    yield thinking_event

                segments = parser.process_chunk(delta.content)
                # Only the part the parser consumed belongs to this turn. The
                # parser stops at the fence closing the first block, so a delta
                # carrying a second block contributes nothing beyond it — the
                # turn ends where the agent stopped reading, not where the
                # transport happened to split the response.
                chunks.append(delta.content[: len(delta.content) - len(parser.remainder)])
                for segment in segments:
                    if segment.type == SegmentType.TEXT:
                        yield TextEvent(segment.content)
                    else:
                        # A CODE segment here is a *closed* fence — capture it
                        # rather than re-parsing the text later, which cannot
                        # distinguish a closed block from an unterminated one.
                        # Captured even when blank: the model wrote a block and
                        # the agent stopped reading because of it, so treating
                        # an empty one as "no code" ended the run with the
                        # preamble as the answer and nothing executed.
                        turn.code = segment.content
                if parser.is_first_code_block_completed() and not defer_execution:
                    stopped_at_code = True
                    break
        except PromptTooLongError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not chunks:
                fatal_stream_error = error
            else:
                logger.warning(
                    "Streaming interrupted mid-response — resuming from partial output",
                    exc_info=True,
                )
                # Record it rather than swallow it: the turn is a fragment, and
                # letting it pass as a whole utterance is how a dropped connection
                # became a confidently truncated "answer".
                turn.error = error
                if defer_execution:
                    # There was no terminal filter verdict. A continuation request
                    # can validate only its own new content, not the block already
                    # received, so this fragment cannot safely enter recovery.
                    turn.recoverable = False
        finally:
            # Close on every exit path — early break, completion, or error.
            # The SDK's read-to-completion auto-close never fires when we stop
            # reading early, which is the common case here.
            try:
                await _close_stream(stream)
            finally:
                self._finalize_turn_usage(
                    turn,
                    stream,
                    wire,
                    "".join(received_content),
                )

        thinking_event = seal_thinking()
        if thinking_event is not None:
            yield thinking_event
        if fatal_stream_error is not None:
            raise fatal_stream_error

        if not parser.is_first_code_block_completed():
            for segment in parser.flush():
                if segment.type == SegmentType.TEXT:
                    yield TextEvent(segment.content)

        turn.text = "".join(chunks)
        turn.finish_reason = stream.finish_reason

        refusal_event = _conclude_turn(
            turn, stream, stopped_at_code=stopped_at_code, defer_execution=defer_execution
        )
        if refusal_event is not None:
            yield refusal_event

    def _finalize_turn_usage(
        self,
        turn: _ModelTurn,
        stream: StreamResponse,
        wire: list[dict[str, str]],
        output: str,
    ) -> None:
        """Record provider usage or estimate it before any stream exit.

        This runs from ``_stream_once``'s ``finally`` so reasoning-only and
        refusal-only calls remain billable even when their transport fails.
        """
        turn.usage = getattr(stream, "usage", None) or TokenUsage()
        if turn.usage.total_tokens != 0:
            return

        # Usage can be absent because this agent stopped at its first code block
        # or because the provider never reports it. Draining after a code block
        # would forfeit the generation early-stop saves, so cost every
        # unreported turn with the same estimator compaction uses instead.
        output += getattr(stream, "refusal", "") or ""
        thinking = getattr(stream, "thinking", "") or ""
        turn.usage = self._estimate_turn_usage(wire, output, thinking)

    def _estimate_turn_usage(
        self,
        wire: list[dict[str, str]],
        text: str,
        thinking: str = "",
    ) -> TokenUsage:
        """Cost a turn whose provider never reported usage.

        Uses the compactor's estimator, so a caller who plugged in a real
        tokenizer gets real numbers here too and the two subsystems cannot
        disagree about how big the same conversation is.

        *thinking* is counted toward the completion: a reasoning model bills for
        its trace even though the trace never reaches ``text``. Measured against
        a live reasoning model, omitting it under-counted completion by 5×, and
        under-counting is the unsafe direction — an output budget that never
        fires is worse than one that fires slightly early.
        """
        count = self.compactor.token_estimator or default_token_estimate
        prompt = sum(count(message["content"]) for message in wire)
        completion = count(text) + count(thinking)
        if (text or thinking) and completion == 0:
            completion = 1
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )

    async def _handle_turn(
        self,
        turn: _ModelTurn,
        state: _RunState,
    ) -> AsyncGenerator[Event, None]:
        """Execute the turn's code block, or record it as the final answer.

        ``turn.code`` was captured by the live parser the moment the fence
        closed, so an unterminated block is simply absent — it is not code, and
        re-parsing the text afterwards could not tell the difference.
        History records only ``turn.text``; the prefix is already in it.
        """
        code = turn.code
        state.final_content = turn.full
        if code is None:
            self.add_message(AssistantMessage(turn.text))
            if not turn.complete:
                # Recovery budget spent and still nothing executable. The text
                # is kept — in history and in StoppedEvent.content — but a
                # fragment is not an answer, and reporting one as COMPLETED is
                # what this whole change exists to stop.
                raise turn.error or ProviderError(
                    "Model output was truncated and could not be recovered"
                )
            state.completed = True
            yield FinalResponseEvent(turn.full)
            return

        state.code_snippets.append(code)
        yield CodeEvent(code)

        # From here the code message is in history without a result. Every
        # path below must append one — see _close_dangling_execution for the
        # cancellation case.
        self.add_message(CodeExecutionMessage(turn.text))
        event, next_prompt = await self._execute(code)
        self.add_message(ExecutionResultMessage(next_prompt))
        yield event

    async def _execute(self, code: str) -> tuple[Event, str]:
        """Run *code*, returning the event to emit and the text for history."""
        state_lost = False
        timeout = self.max_exec_timeout
        if timeout is not None:
            result, state_lost = await self._execute_with_timeout(code)
        else:
            result = await self.runtime.execute(code)
            state_lost = result.state_lost

        if result is None:
            assert timeout is not None
            prompt = EXECUTION_TIMEOUT_PROMPT.format(
                timeout=timeout,
            )
            if state_lost:
                prompt += f"\n\n{RUNTIME_STATE_LOST_NOTICE}"
            return (
                ExecutionTimeoutEvent(
                    timeout,
                    state_lost=state_lost,
                ),
                prompt,
            )

        if not result.success and isinstance(result.error, SecurityError):
            message = result.error.message
            return (
                SecurityErrorEvent(message),
                SECURITY_ERROR_PROMPT.format(error=message),
            )

        # Scrub lone surrogates from stdout (binary bytes printed by generated
        # code, half-decoded files) before it enters a message and the next
        # call's json.dumps would choke on it.
        stdout = sanitize_surrogates(result.stdout or EMPTY_OUTPUT_PLACEHOLDER)
        body, persisted_variable = await self._shape_output(stdout)

        return (
            ExecutionResultEvent(
                output=stdout,
                success=result.success,
                persisted=persisted_variable is not None,
                persisted_variable=persisted_variable,
                state_lost=state_lost,
            ),
            EXECUTION_OUTPUT_PROMPT.format(execution_output=body)
            + (f"\n\n{RUNTIME_STATE_LOST_NOTICE}" if state_lost else ""),
        )

    async def _shape_output(self, stdout: str) -> tuple[str, str | None]:
        """Bound an oversize result without discarding it.

        Returns the body to show the model, and the name holding the full text
        (``None`` when the output fitted inline) — one value, so the event and
        the marker cannot name different variables.

        Because the runtime is Python, that variable is a first-class object the
        model can slice, search or re-parse: no file round-trip, and no
        re-running code that may be slow, expensive or not reproducible.

        Every output gets a fresh name, allocated by the runtime. The marker in
        history is a *pointer*, so reusing a name does not keep the newest — it
        re-points every earlier marker at text it never described.
        """
        if len(stdout) <= self.max_exec_output:
            return stdout, None

        variable = await self.runtime.bind_unique(
            self.persisted_output_prefix,
            stdout,
            start=self._persisted_floor,
        )
        marker = build_persist_marker(stdout, self.exec_output_preview_chars, variable)
        logger.info("Execution output persisted to `%s` (%d chars)", variable, len(stdout))
        return marker, variable

    async def _execute_with_timeout(
        self,
        code: str,
    ) -> tuple[ExecutionResult | None, bool]:
        """Run *code* under ``max_exec_timeout``.

        Returns the result (``None`` on expiry) and whether stopping the
        execution replaced the runtime process.

        The runtime is preemptible by construction (``__init__`` refuses the
        combination otherwise). Execution runs in a task on this loop so the
        deadline can cancel it. The backend then interrupts and finishes
        request-scoped cleanup before re-raising cancellation; only it knows
        which request still owns the runtime. A kernel that catches the
        interrupt is terminated by its runtime before control returns. No
        worker thread, no second event loop.

        Offloading to a thread and abandoning it is not an option: the
        in-process runtime's ``capture_output()`` is a process-global
        ``sys.stdout`` patch, and unwinding it out of order sends every later
        print in the host into a dead buffer.

        ``asyncio.wait`` reports expiry without manufacturing an exception, so
        a ``TimeoutError`` raised by the backend itself (a kernel that never
        answered on its shell channel) keeps its identity.
        """
        execution = asyncio.create_task(self.runtime.execute(code))
        try:
            done, _ = await asyncio.wait(
                (execution,),
                timeout=self.max_exec_timeout,
            )
        except BaseException as error:
            execution.cancel()
            state_lost = False
            try:
                await execution
            except RuntimeStateLostError:
                state_lost = True
            except BaseException:
                # Preserve the cancellation/error already leaving the agent;
                # cleanup cannot outrank it.
                pass
            if state_lost and isinstance(error, asyncio.CancelledError):
                raise RuntimeStateLostError(
                    "Cancelling execution terminated the runtime process"
                ) from error
            raise

        if done:
            result = execution.result()
            return result, result.state_lost

        # Cancellation propagates into Runtime.execute, which owns interruption
        # and cleanup for the request it admitted. An interrupt from here could
        # arrive after execute() releases the runtime and hit the next caller.
        execution.cancel()
        state_lost = False
        try:
            await execution
        except RuntimeStateLostError:
            state_lost = True
        except asyncio.CancelledError:
            pass
        return None, state_lost

    def _close_dangling_execution(self, placeholder: str) -> None:
        """Pair a trailing code message with a synthetic result.

        A run cancelled between appending the code message and appending its
        result would otherwise leave history ending on an assistant turn that
        claims to have run code with no outcome — and the next run's user
        message would land directly after it. Idempotent: a no-op unless the
        last message is an unanswered code message.
        """
        if self.messages and isinstance(self.messages[-1], CodeExecutionMessage):
            self.add_message(ExecutionResultMessage(placeholder))

    def _mark_user_interruption(self) -> None:
        """Separate an abandoned request from the next user turn."""
        if not (
            self.messages
            and isinstance(self.messages[-1], UserMessage)
            and self.messages[-1].content == USER_INTERRUPTION_PLACEHOLDER
        ):
            self.add_message(UserMessage(USER_INTERRUPTION_PLACEHOLDER))

    def _init_skills(self, skills: list[Skill] | None = None) -> None:
        self._skill_registry = SkillRegistry()
        if skills:
            self._skill_registry.add_skills([s for s in skills if s is not None])
        if self._skill_registry.list_skills():
            store = self._skill_registry.build_skill_store()
            bindings: list[tuple[str, Any]] = []
            for skill in self._skill_registry.list_skills():
                for function in skill.functions:
                    bindings.append((function.name, function.func))
                for variable in skill.variables:
                    bindings.append((variable.name, variable.value))
                for type_obj in skill.types:
                    bindings.append((type_obj.name, type_obj.value))
            bindings.append(("_skill_store", store))
            self.runtime.inject_resources(
                functions=[Function(activate_skill)],
                bindings=bindings,
            )
            self.system_instructions += "\n" + SKILLS_INSTRUCTION

    def _initialize_conversation(self, query: str) -> None:
        self._update_system_message()
        # Scrub lone surrogates from the user's turn (rich-text paste, broken
        # emoji pipeline) before it enters history and the wire.
        self.add_message(UserMessage(sanitize_surrogates(query)))

    def _update_system_message(self) -> None:
        """Rebuild the system prompt so runtime changes between runs show up."""
        prompt = self.build_system_prompt()
        if self.messages and isinstance(self.messages[0], SystemMessage):
            self.messages[0] = SystemMessage(prompt)
        else:
            self.messages.insert(0, SystemMessage(prompt))

    def _prepare_messages(self) -> list[dict[str, str]]:
        """Render history for the wire, with a fresh time reminder.

        The reminder rides as a user message right after the system prompt
        rather than inside it, so the system prompt stays byte-stable across
        turns (prompt caches key on it) while the model still knows the date.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:00")
        reminder = {
            "role": "user",
            "content": f"<system-reminder>\nCurrent date and time: {now}\n</system-reminder>",
        }
        wire = to_wire(self.messages)
        for index, message in enumerate(wire):
            if message["role"] == "system":
                return [*wire[: index + 1], reminder, *wire[index + 1 :]]
        return [reminder, *wire]
