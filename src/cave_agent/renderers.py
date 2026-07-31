"""Rich terminal rendering for agent event streams.

Optional component. The agent core does not depend on this module — there is
no display flag on :class:`~cave_agent.agent.CaveAgent` and no import of this
file from the agent loop. It is a pure consumer of events that re-yields them
unchanged, so downstream consumers still see the full stream::

    from cave_agent import CaveAgent
    from cave_agent.renderers import TerminalRenderer

    renderer = TerminalRenderer()
    async for event in renderer.render(agent.stream_events("hello")):
        ...  # renderer handles display; events still flow through

Keeping rendering outside the loop is what lets ``run()`` and
``stream_events()`` share a single execution path — the previous design routed
``run()`` through streaming purely to feed the UI.

Style:
- ● blue prefix for code blocks
- ⎿ prefix on the first output line, aligned continuation
- Live Markdown rendering for model prose
- A closing summary with elapsed time and token usage
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

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

if TYPE_CHECKING:
    from .agent import AgentResponse, CaveAgent

console = Console()

_RESULT_PREFIX = "  ⎿  "
_CONTINUATION_INDENT = " " * len(_RESULT_PREFIX)
_MAX_DISPLAY_LINES = 12
# Also cap by characters: ExecutionResultEvent.output is the full, unshaped
# stdout, so a single-line dump (``print(json.dumps(big))``) sails past a
# line-count limit and floods the terminal with megabytes.
_MAX_DISPLAY_CHARS = 2000

# Terminal reasons worth calling out; COMPLETED needs no banner.
_STOP_NOTICES = {
    StopReason.MAX_STEPS: "Max steps reached",
    StopReason.TIMEOUT: "Maximum run time reached",
    StopReason.BUDGET_EXHAUSTED: "Token budget exhausted",
    StopReason.CANCELLED: "Cancelled",
    StopReason.MODEL_ERROR: "Model error",
    StopReason.RUNTIME_ERROR: "Runtime error",
    StopReason.INTERNAL_ERROR: "Internal error",
}


def render_user_prompt(text: str) -> None:
    """Print a user prompt in a consistent style."""
    console.print()
    console.print(Text.assemble(("> ", "bold blue"), (text, "")))
    console.print()


def _print_prefixed_lines(lines: list[str], style: str) -> None:
    """Print lines with ⎿ on the first line and aligned indent on the rest."""
    for index, line in enumerate(lines):
        prefix = _RESULT_PREFIX if index == 0 else _CONTINUATION_INDENT
        console.print(Text(f"{prefix}{line}", style=style))


class TerminalRenderer:
    """Renders an agent event stream to the terminal, passing events through.

    One instance per run — it carries per-run display state (live blocks and
    buffers).
    """

    def __init__(self) -> None:
        self._live: Live | None = None
        self._mode: str = ""  # "thinking", "text", or "code"
        self._text_buffer: list[str] = []
        self._thinking_buffer: list[str] = []
        # A non-transient (frozen) live block leaves the cursor mid-line; when
        # set, ``_stop_live`` ends that line so the next header starts cleanly.
        self._live_needs_break: bool = False

    async def render(self, events: AsyncIterator[Event]) -> AsyncGenerator[Event, None]:
        """Consume *events*, render each, and re-yield it unchanged."""
        try:
            async for event in events:
                self.handle(event)
                yield event
        finally:
            # A cancelled or failed run must not leave the terminal inside a
            # live block with a hidden cursor.
            self._stop_live()

    def handle(self, event: Event) -> None:
        """Render a single event."""
        match event:
            case UserPromptEvent():
                render_user_prompt(event.content)
            case ThinkingChunkEvent():
                self._handle_thinking_chunk(event)
            case ThinkingEvent():
                self._handle_thinking_seal()
            case TextEvent():
                self._handle_text(event)
            case CodeEvent():
                self._handle_code(event)
            case ExecutionResultEvent():
                self._handle_result(event)
            case ExecutionTimeoutEvent():
                self._handle_timeout(event)
            case SecurityErrorEvent():
                self._handle_security_error(event)
            case StatusEvent():
                self._handle_status(event)
            case FinalResponseEvent():
                self._handle_final()
            case StoppedEvent():
                self._handle_stopped(event)

    def _handle_thinking_chunk(self, event: ThinkingChunkEvent) -> None:
        """Render a reasoning delta live in a muted style, above the answer."""
        if self._mode in ("text", "code"):
            self._stop_live()

        self._thinking_buffer.append(event.content)
        text = "".join(self._thinking_buffer)

        if self._mode != "thinking":
            console.print()
            console.print(Text.assemble(("✻ ", "magenta"), ("Thinking", "bold dim")))
            self._mode = "thinking"
            self._live = Live(
                Text(text, style="dim italic"),
                console=console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()
            self._live_needs_break = True
        else:
            live = self._live
            assert live is not None
            live.update(Text(text, style="dim italic"))

    def _handle_thinking_seal(self) -> None:
        """Freeze the reasoning block so answer text renders fresh below it.

        The completed event carries the full trace, but it was already streamed
        chunk by chunk — there is nothing more to print.
        """
        self._stop_live()
        self._thinking_buffer.clear()

    def _handle_text(self, event: TextEvent) -> None:
        if self._mode == "code":
            self._stop_live()

        self._text_buffer.append(event.content)
        text = "".join(self._text_buffer)

        if self._mode != "text":
            console.print()
            self._mode = "text"
            self._live = Live(
                Markdown(text),
                console=console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()
            self._live_needs_break = True
        else:
            live = self._live
            assert live is not None
            live.update(Markdown(text))

    def _handle_code(self, event: CodeEvent) -> None:
        self._stop_live()
        self._text_buffer.clear()
        self._mode = "code"

        console.print(Text.assemble(("● ", "blue"), ("Code", "bold")))
        code = event.code.strip()
        if code:
            console.print(Syntax(code, "python", theme="one-dark", padding=(0, 2)))

        # Spinner stand-in while the runtime works.
        self._live = Live(
            Text(f"{_CONTINUATION_INDENT}Running…", style="dim"),
            console=console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.start()

    def _handle_result(self, event: ExecutionResultEvent) -> None:
        self._stop_live()

        error = not event.success
        dot_style = "red" if error else "blue"
        console.print(Text.assemble(("● ", dot_style), ("Output", "bold")))

        style = "red dim" if error else "dim"
        body = event.output.strip()
        if len(body) > _MAX_DISPLAY_CHARS:
            body = (
                body[:_MAX_DISPLAY_CHARS]
                + f"\n… {len(body) - _MAX_DISPLAY_CHARS:,} more characters"
            )
        lines = body.splitlines()

        if lines:
            if len(lines) <= _MAX_DISPLAY_LINES:
                _print_prefixed_lines(lines, style)
            else:
                _print_prefixed_lines(lines[:4], style)
                console.print(
                    Text(
                        f"{_CONTINUATION_INDENT}… {len(lines) - 8} more lines",
                        style="dim italic",
                    )
                )
                for line in lines[-4:]:
                    console.print(Text(f"{_CONTINUATION_INDENT}{line}", style=style))

        if event.persisted:
            console.print(
                Text(
                    f"{_CONTINUATION_INDENT}full output kept in `{event.persisted_variable}`",
                    style="dim italic",
                )
            )
        if event.state_lost:
            console.print(
                Text(
                    f"{_CONTINUATION_INDENT}runtime process terminated; "
                    "non-registered state was lost",
                    style="bold red",
                )
            )

        console.print()

    def _handle_timeout(self, event: ExecutionTimeoutEvent) -> None:
        self._stop_live()
        console.print(
            Text(
                f"{_RESULT_PREFIX}Execution timed out after {event.timeout}s",
                style="bold red",
            )
        )
        if event.state_lost:
            console.print(
                Text(
                    f"{_CONTINUATION_INDENT}runtime process terminated; "
                    "non-registered state was lost",
                    style="bold red",
                )
            )
        console.print()

    def _handle_security_error(self, event: SecurityErrorEvent) -> None:
        self._stop_live()
        console.print(Text(f"{_RESULT_PREFIX}Security Error: {event.message}", style="bold red"))
        console.print()

    def _handle_status(self, event: StatusEvent) -> None:
        match event.status:
            case StatusType.COMPACTING:
                self._stop_live()
                console.print(
                    Text.assemble(
                        ("● ", "blue"),
                        ("Compacting conversation", "bold"),
                        ("...", "bold"),
                    )
                )
                self._live = Live(
                    Text(f"{_RESULT_PREFIX}Summarizing...", style="dim"),
                    console=console,
                    refresh_per_second=4,
                    transient=True,
                )
                self._live.start()
            case StatusType.COMPACTED:
                self._stop_live()
                if event.before is not None and event.after is not None:
                    console.print(
                        Text(
                            f"{_RESULT_PREFIX}Compacted {event.before} → {event.after} messages",
                            style="dim",
                        )
                    )
                console.print()
            case StatusType.OUTPUT_RECOVERY:
                self._stop_live()
                console.print(
                    Text(
                        f"{_RESULT_PREFIX}Output truncated — resuming "
                        f"({event.recovery}/{event.max_recoveries})",
                        style="dim yellow",
                    )
                )

    def _handle_final(self) -> None:
        self._stop_live()
        self._text_buffer.clear()

    def _handle_stopped(self, event: StoppedEvent) -> None:
        self._stop_live()

        notice = _STOP_NOTICES.get(event.stop_reason)
        if notice:
            console.print()
            console.print(Text(f"{_RESULT_PREFIX}{notice}", style="bold yellow"))

        console.print()
        line = Text("  ")
        line.append("done", style="bold dim")

        details = [f"{event.elapsed:.1f}s", f"{event.steps} steps"]
        if event.usage.total_tokens > 0:
            details.append(
                f"{event.usage.total_tokens:,} tokens "
                f"(prompt: {event.usage.prompt_tokens:,}, "
                f"completion: {event.usage.completion_tokens:,})"
            )
        line.append(f"  {' · '.join(details)}", style="dim")
        console.print(line)

    def _stop_live(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None
            if self._live_needs_break:
                # Terminate the frozen block's last line so the following
                # header/text doesn't run onto it.
                console.print()
        self._live_needs_break = False
        self._mode = ""


async def render_events(events: AsyncIterator[Event]) -> None:
    """Consume an event stream, rendering it and discarding the events."""
    renderer = TerminalRenderer()
    async for _ in renderer.render(events):
        pass


async def render_run(agent: CaveAgent, query: str) -> AgentResponse:
    """Run *agent* on *query* with terminal rendering, returning the response.

    The convenience form of ``run()`` plus a renderer, for scripts and demos
    that want both the display and the :class:`~cave_agent.AgentResponse`. It
    lives here rather than on the agent so the core keeps no notion of
    rendering; importing this module is what opts you in.
    """
    from .agent import drain_to_response

    return await drain_to_response(TerminalRenderer().render(agent.stream_events(query)))
