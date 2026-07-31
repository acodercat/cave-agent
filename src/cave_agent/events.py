"""Events emitted by :meth:`~cave_agent.agent.CaveAgent.stream_events`.

Every event is a frozen dataclass with named, typed fields — consumers use
``match`` or ``isinstance`` to react. Nothing here is stringly-typed: an
execution result carries its output *and* whether it succeeded, a stopped run
carries its token usage, step count and reason.

Rendering is not part of this module. ``cave_agent.renderers`` consumes these
events; the agent core has no knowledge of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .models import TokenUsage


class StopReason(StrEnum):
    """Why the agent loop terminated."""

    COMPLETED = "completed"
    """The model answered without emitting more code — the normal ending."""

    MAX_STEPS = "max_steps"
    """``max_steps`` model calls were made without reaching an answer."""

    TIMEOUT = "timeout"
    """``max_run_time`` elapsed. Checked at turn boundaries, so a single
    long step can overshoot it; ``max_exec_timeout`` bounds that."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    """Cumulative usage crossed ``max_total_tokens``, ``max_input_tokens``
    or ``max_output_tokens``. The crossing step completes and is recorded;
    the next one does not start."""

    CANCELLED = "cancelled"
    """The run was cancelled. History is left protocol-valid — any code
    message that never got a result is paired with a cancellation marker."""

    MODEL_ERROR = "model_error"
    """The model layer failed in a way compaction could not rescue."""

    RUNTIME_ERROR = "runtime_error"
    """The execution runtime failed — the kernel died, would not start, or its
    transport broke. Distinct from :attr:`MODEL_ERROR`: the provider did
    nothing wrong and retrying against the model would not help. Code that
    merely *raises* is not this; that is an ``ExecutionResultEvent`` the model
    reads and reacts to."""

    INTERNAL_ERROR = "internal_error"
    """A failure cave-agent could not classify — i.e. a bug in cave-agent.

    Exists so a stream consumed to termination reports a ``StoppedEvent`` even
    for an internal defect. The exception is still re-raised afterwards; the
    event is what lets a consumer close its own stream cleanly first."""


class StatusType(StrEnum):
    """Subtypes carried by :class:`StatusEvent`."""

    COMPACTING = "compacting"
    COMPACTED = "compacted"
    OUTPUT_RECOVERY = "output_recovery"


@dataclass(frozen=True)
class UserPromptEvent:
    """The user turn that opened this run — always the first event."""

    content: str


@dataclass(frozen=True)
class TextEvent:
    """One chunk of the model's user-facing prose.

    Reasoning arrives separately as :class:`ThinkingChunkEvent`, so this
    never carries chain-of-thought.
    """

    content: str


@dataclass(frozen=True)
class ThinkingChunkEvent:
    """One chunk of the model's reasoning trace, streamed live."""

    content: str


@dataclass(frozen=True)
class ThinkingEvent:
    """A completed reasoning segment.

    ``duration_ms`` is measured at the source — first reasoning token to the
    moment reasoning stops — so consumers should display it rather than
    diffing surrounding event arrival times, which would mis-attribute a
    post-reasoning stall to thinking.
    """

    content: str
    duration_ms: int


@dataclass(frozen=True)
class CodeEvent:
    """The code block the agent is about to execute.

    ``code`` is the extracted Python source, not the raw model response —
    consumers no longer need to re-parse the response to render it.
    """

    code: str


@dataclass(frozen=True)
class ExecutionResultEvent:
    """The runtime's response to a :class:`CodeEvent`.

    ``output`` is the full stdout as produced, *before* any oversize
    shaping — a renderer shows what really happened. ``persisted`` marks
    the case where the model instead received a
    ``<persisted-output>`` marker pointing at ``persisted_variable``.
    ``state_lost`` warns that stopping this execution destroyed arbitrary
    namespace state.
    """

    output: str
    success: bool
    persisted: bool = False
    persisted_variable: str | None = None
    state_lost: bool = False


@dataclass(frozen=True)
class ExecutionTimeoutEvent:
    """Code execution exceeded ``max_exec_timeout``.

    ``state_lost`` means stopping it terminated the backend process, so
    arbitrary namespace state did not survive.
    """

    timeout: float
    state_lost: bool = False


@dataclass(frozen=True)
class SecurityErrorEvent:
    """The security checker rejected the code before it ran."""

    message: str


@dataclass(frozen=True)
class StatusEvent:
    """An agent-loop status change (compaction, output recovery)."""

    status: StatusType
    before: int | None = None
    after: int | None = None
    recovery: int | None = None
    max_recoveries: int | None = None


@dataclass(frozen=True)
class FinalResponseEvent:
    """The model's answer — emitted when a turn produces no code."""

    content: str


@dataclass(frozen=True)
class StoppedEvent:
    """The run has terminated — always the last event."""

    content: str
    stop_reason: StopReason
    steps: int
    elapsed: float
    usage: TokenUsage


Event = (
    UserPromptEvent
    | TextEvent
    | ThinkingChunkEvent
    | ThinkingEvent
    | CodeEvent
    | ExecutionResultEvent
    | ExecutionTimeoutEvent
    | SecurityErrorEvent
    | StatusEvent
    | FinalResponseEvent
    | StoppedEvent
)
