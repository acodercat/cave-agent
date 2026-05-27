"""Rich terminal display for cave-agent events.

Claude Code-style UI:
- ● blue prefix for code execution
- ⎿ prefix for execution output
- Live Markdown rendering for LLM text
- Summary with elapsed time
"""

import time
from typing import AsyncGenerator

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

from .types import Event, EventType
from .models import TokenUsage
from .utils import extract_python_code

console = Console()

_RESULT_PREFIX = "  ⎿  "
_CONTINUATION_INDENT = " " * len(_RESULT_PREFIX)
_MAX_DISPLAY_LINES = 12


async def with_display(
    events: AsyncGenerator[Event, None],
    context=None,
) -> AsyncGenerator[Event, None]:
    """Wrap an async event stream, printing each event and yielding it through.

    *context* is an optional execution context providing token usage for
    the summary line.  Passed internally by ``CaveAgent.stream_events()``.
    """
    display = _DisplayState()
    async for event in events:
        display.handle(event)
        yield event
    token_usage = getattr(context, "token_usage", None) if context else None
    display.finalize(token_usage)


async def render_events(
    events: AsyncGenerator[Event, None],
    context=None,
) -> None:
    """Consume an async event stream, printing each event to the terminal."""
    async for _ in with_display(events, context):
        pass


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


class _DisplayState:
    """Tracks rendering state across the async event stream."""

    def __init__(self):
        self._live: Live | None = None
        self._mode: str = ""  # "text" or "code"
        self._text_buffer: list[str] = []
        self._start_time: float = time.monotonic()

    def handle(self, event: Event) -> None:
        match event.type:
            case EventType.TEXT:
                self._handle_text(event)
            case EventType.CODE:
                self._handle_code(event)
            case EventType.EXECUTION_OUTPUT:
                self._handle_output(event, error=False)
            case EventType.EXECUTION_ERROR:
                self._handle_output(event, error=True)
            case EventType.EXECUTION_OUTPUT_EXCEEDED:
                self._handle_output(event, error=True)
            case EventType.SECURITY_ERROR:
                self._handle_security_error(event)
            case EventType.FINAL_RESPONSE:
                self._handle_final()
            case EventType.MAX_STEPS_REACHED:
                self._handle_max_steps(event)
            case EventType.COMPACTING:
                self._handle_compacting()
            case EventType.COMPACTED:
                self._handle_compacted(event)

    def _handle_text(self, event: Event) -> None:
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
        else:
            self._live.update(Markdown(text))

    def _handle_code(self, event: Event) -> None:
        self._stop_live()
        self._text_buffer.clear()
        self._mode = "code"

        # Extract pure code from the LLM response
        code = extract_python_code(event.content, "python") or event.content

        console.print(Text.assemble(("● ", "blue"), ("Code", "bold")))
        code = code.strip()
        if code:
            syntax = Syntax(code, "python", theme="one-dark", padding=(0, 2))
            console.print(syntax)

        # Show "Running…" while code executes
        self._live = Live(
            Text(f"{_CONTINUATION_INDENT}Running…", style="dim"),
            console=console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.start()

    def _handle_output(self, event: Event, error: bool = False) -> None:
        self._stop_live()

        dot_style = "red" if error else "blue"
        console.print(Text.assemble(("● ", dot_style), ("Output", "bold")))

        style = "red dim" if error else "dim"
        lines = event.content.strip().splitlines()

        if not lines:
            return

        if len(lines) <= _MAX_DISPLAY_LINES:
            _print_prefixed_lines(lines, style)
        else:
            _print_prefixed_lines(lines[:4], style)
            console.print(Text(f"{_CONTINUATION_INDENT}… {len(lines) - 8} more lines", style="dim italic"))
            for line in lines[-4:]:
                console.print(Text(f"{_CONTINUATION_INDENT}{line}", style=style))

        console.print()

    def _handle_security_error(self, event: Event) -> None:
        self._stop_live()
        console.print(Text(f"{_RESULT_PREFIX}Security Error: {event.content}", style="bold red"))
        console.print()

    def _handle_final(self) -> None:
        self._stop_live()
        self._text_buffer.clear()

    def _handle_compacting(self) -> None:
        self._stop_live()
        console.print(Text.assemble(
            ("● ", "blue"),
            ("Compacting conversation", "bold"),
            ("...", "bold"),
        ))
        self._live = Live(
            Text(f"{_RESULT_PREFIX}Summarizing...", style="dim"),
            console=console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.start()

    def _handle_compacted(self, event: Event) -> None:
        self._stop_live()
        if event.content:
            console.print(Text(f"{_RESULT_PREFIX}Compacted {event.content} messages", style="dim"))
        console.print()

    def _handle_max_steps(self, event: Event) -> None:
        self._stop_live()
        console.print()
        console.print(Text(f"{_RESULT_PREFIX}{event.content}", style="bold yellow"))

    def finalize(self, token_usage: TokenUsage | None = None) -> None:
        self._stop_live()
        elapsed = time.monotonic() - self._start_time
        console.print()

        line = Text("  ")
        line.append("done", style="bold dim")

        details = [f"{elapsed:.1f}s"]
        if token_usage and token_usage.total_tokens > 0:
            details.append(
                f"{token_usage.total_tokens:,} tokens "
                f"(prompt: {token_usage.prompt_tokens:,}, "
                f"completion: {token_usage.completion_tokens:,})"
            )

        line.append(f"  {' · '.join(details)}", style="dim")
        console.print(line)

    def _stop_live(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None
        self._mode = ""
