"""The renderer is an optional pass-through, invisible to the agent core."""

import asyncio

from cave_agent import CaveAgent, IPythonRuntime
from cave_agent.events import (
    CodeEvent,
    ExecutionResultEvent,
    StoppedEvent,
    StopReason,
    TextEvent,
    UserPromptEvent,
)
from cave_agent.models import TokenUsage
from cave_agent.renderers import TerminalRenderer

from .fakes import FakeModel


async def _events(*items):
    for item in items:
        yield item


class TestPassThrough:
    async def test_events_are_re_yielded_unchanged(self):
        original = [
            UserPromptEvent("q"),
            TextEvent("hello"),
            StoppedEvent("hello", StopReason.COMPLETED, 1, 0.5, TokenUsage()),
        ]

        seen = [e async for e in TerminalRenderer().render(_events(*original))]

        assert seen == original
        assert all(a is b for a, b in zip(seen, original, strict=True))

    async def test_agent_stream_passes_through(self):
        agent = CaveAgent(
            model=FakeModel(["```python\nprint('hi')\n```", "done"]),
            runtime=IPythonRuntime(),
        )

        seen = [e async for e in TerminalRenderer().render(agent.stream_events("q"))]

        assert any(isinstance(e, CodeEvent) for e in seen)
        assert any(isinstance(e, ExecutionResultEvent) for e in seen)
        assert isinstance(seen[-1], StoppedEvent)

    async def test_renders_every_event_type_without_error(self):
        from cave_agent.events import (
            ExecutionTimeoutEvent,
            FinalResponseEvent,
            SecurityErrorEvent,
            StatusEvent,
            StatusType,
            ThinkingChunkEvent,
            ThinkingEvent,
        )

        every = [
            UserPromptEvent("q"),
            ThinkingChunkEvent("reasoning..."),
            ThinkingEvent("reasoning...", 120),
            TextEvent("some prose"),
            CodeEvent("print(1)"),
            ExecutionResultEvent("1", success=True),
            ExecutionResultEvent("boom", success=False),
            ExecutionResultEvent(
                "x" * 50, success=True, persisted=True, persisted_variable="_last_output"
            ),
            ExecutionTimeoutEvent(30.0),
            ExecutionResultEvent(
                "timeout",
                success=False,
                state_lost=True,
            ),
            ExecutionTimeoutEvent(30.0, state_lost=True),
            SecurityErrorEvent("blocked"),
            StatusEvent(StatusType.COMPACTING),
            StatusEvent(StatusType.COMPACTED, before=20, after=8),
            StatusEvent(StatusType.OUTPUT_RECOVERY, recovery=1, max_recoveries=3),
            FinalResponseEvent("done"),
            StoppedEvent("done", StopReason.COMPLETED, 2, 1.0, TokenUsage(10, 5, 15)),
        ]

        seen = [e async for e in TerminalRenderer().render(_events(*every))]

        assert len(seen) == len(every)

    async def test_long_output_is_abbreviated_without_dropping_the_event(self):
        long = ExecutionResultEvent("\n".join(str(i) for i in range(200)), success=True)
        seen = [e async for e in TerminalRenderer().render(_events(long))]
        assert seen == [long]


class TestAgentIndependence:
    def test_agent_has_no_display_flag(self):
        agent = CaveAgent(model=FakeModel(), runtime=IPythonRuntime())
        assert not hasattr(agent, "display")

    def test_agent_module_does_not_import_renderers(self):
        """The core stays out of the rendering dependency, and the old circular
        import between agent and display — which forced function-body imports —
        is gone.

        Checks import *statements* (including ones nested in functions, which is
        how the cycle used to be worked around), not mentions in prose.
        """
        import ast
        import inspect

        from cave_agent import agent as agent_module

        tree = ast.parse(inspect.getsource(agent_module))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")

        assert not any("renderers" in name or "display" in name for name in imported), imported

    async def test_unrendered_run_is_identical(self):
        rendered = await CaveAgent(
            model=FakeModel(["```python\nprint(7)\n```", "seven"]),
            runtime=IPythonRuntime(),
        ).run("q")

        agent = CaveAgent(
            model=FakeModel(["```python\nprint(7)\n```", "seven"]),
            runtime=IPythonRuntime(),
        )
        collected = [e async for e in TerminalRenderer().render(agent.stream_events("q"))]
        stopped = [e for e in collected if isinstance(e, StoppedEvent)][0]

        assert rendered.content == stopped.content
        assert rendered.steps == stopped.steps


class TestCleanup:
    async def test_live_block_closed_on_cancellation(self):
        """A cancelled run must not leave the terminal inside a live block."""
        renderer = TerminalRenderer()

        async def hanging():
            yield CodeEvent("print(1)")  # opens a live "Running…" block
            await asyncio.sleep(60)

        agen = renderer.render(hanging())
        await agen.__anext__()
        await agen.aclose()

        assert renderer._live is None


class TestStopReasonCoverage:
    def test_every_stop_reason_has_a_notice(self):
        """A new StopReason must not ship without a banner.

        `_STOP_NOTICES` is read with `.get()`, so a missing member renders
        nothing at all rather than failing — this assertion is what turns that
        silence into a test failure.
        """
        from cave_agent.renderers import _STOP_NOTICES

        assert set(_STOP_NOTICES) == set(StopReason) - {StopReason.COMPLETED}
