"""Agent loop behaviour — events, stop reasons, budgets, cancellation.

Runs entirely offline against ``FakeModel``.
"""

import asyncio
from contextlib import aclosing

import pytest

from cave_agent import CaveAgent, Function, IPythonRuntime
from cave_agent._placeholders import USER_INTERRUPTION_PLACEHOLDER
from cave_agent.events import (
    CodeEvent,
    ExecutionResultEvent,
    FinalResponseEvent,
    SecurityErrorEvent,
    StatusEvent,
    StatusType,
    StoppedEvent,
    StopReason,
    TextEvent,
    UserPromptEvent,
)
from cave_agent.messages import (
    CodeExecutionMessage,
    ExecutionResultMessage,
    MessageRole,
    UserMessage,
)
from cave_agent.models import ModelError, PromptTooLongError, TokenUsage
from cave_agent.runtime.executor import ExecutionResult, RuntimeExecutionError
from cave_agent.security import ImportRule, SecurityChecker

from .fakes import FailingModel, FakeModel


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _agent(responses, **kwargs) -> CaveAgent:
    kwargs.setdefault("runtime", IPythonRuntime(functions=[Function(add)]))
    return CaveAgent(model=FakeModel(responses), **kwargs)


class TestEventStream:
    async def test_opens_with_user_prompt_and_ends_with_stopped(self):
        events = [e async for e in _agent(["hi there"]).stream_events("q")]

        assert isinstance(events[0], UserPromptEvent)
        assert events[0].content == "q"
        assert isinstance(events[-1], StoppedEvent)

    async def test_exactly_one_stopped_event(self):
        events = [e async for e in _agent(["```python\nprint(1)\n```", "done"]).stream_events("q")]
        assert sum(isinstance(e, StoppedEvent) for e in events) == 1

    async def test_code_then_result_then_answer(self):
        events = [
            e
            async for e in _agent(
                ["Computing.\n```python\nprint(add(5, 3))\n```", "The answer is 8."]
            ).stream_events("add 5 and 3")
        ]

        kinds = [type(e) for e in events]
        assert kinds.index(CodeEvent) < kinds.index(ExecutionResultEvent)
        assert kinds.index(ExecutionResultEvent) < kinds.index(FinalResponseEvent)

    async def test_code_event_carries_pure_code(self):
        """Consumers should not have to re-parse the raw response."""
        events = [
            e
            async for e in _agent(
                ["Prose here.\n```python\nprint(add(1, 2))\n```", "3"]
            ).stream_events("q")
        ]
        code = [e for e in events if isinstance(e, CodeEvent)][0]
        assert code.code == "print(add(1, 2))"

    async def test_fence_inside_triple_quoted_python_is_not_executed_early(self):
        response = (
            '```python\npayload = """embedded markdown:\n```\nstill data"""\nprint(payload)\n```'
        )

        events = [event async for event in _agent([response, "done"]).stream_events("q")]

        execution = next(event for event in events if isinstance(event, ExecutionResultEvent))
        assert execution.success
        assert "still data" in (execution.output or "")

    async def test_inline_triple_backticks_do_not_truncate_executed_code(self):
        runtime = IPythonRuntime()
        agent = CaveAgent(
            model=FakeModel(
                [
                    '```python\nmarker = "```"\n# inline ``` is code\nvalue = 42\n```',
                    "done",
                ]
            ),
            runtime=runtime,
        )

        response = await agent.run("q")

        assert response.stop_reason is StopReason.COMPLETED
        assert await runtime.get_from_namespace("marker") == "```"
        assert await runtime.get_from_namespace("value") == 42

    async def test_result_event_carries_output_and_success(self):
        events = [
            e async for e in _agent(["```python\nprint(add(5, 3))\n```", "8"]).stream_events("q")
        ]
        result = [e for e in events if isinstance(e, ExecutionResultEvent)][0]
        assert "8" in result.output
        assert result.success is True
        assert result.persisted is False

    async def test_failing_code_marks_result_unsuccessful(self):
        events = [
            e
            async for e in _agent(
                ["```python\nraise ValueError('nope')\n```", "handled"]
            ).stream_events("q")
        ]
        result = [e for e in events if isinstance(e, ExecutionResultEvent)][0]
        assert result.success is False

    async def test_text_streams_before_code(self):
        events = [
            e
            async for e in _agent(
                ["Let me think about it.\n```python\nprint(1)\n```", "ok"]
            ).stream_events("q")
        ]
        text = "".join(e.content for e in events if isinstance(e, TextEvent))
        assert "Let me think" in text


class TestStopReasons:
    async def test_completed(self):
        response = await _agent(["just an answer"]).run("q")
        assert response.stop_reason is StopReason.COMPLETED
        assert response.completed

    async def test_max_steps(self):
        response = await _agent(["```python\nprint(1)\n```"] * 10, max_steps=3).run("q")
        assert response.stop_reason is StopReason.MAX_STEPS
        assert response.steps == 3
        assert not response.completed

    async def test_limit_keeps_the_last_code_producing_turn(self):
        response = await _agent(
            ["I will calculate.\n```python\nprint(1)\n```"],
            max_steps=1,
        ).run("q")

        assert response.stop_reason is StopReason.MAX_STEPS
        assert response.content == "I will calculate.\n```python\nprint(1)\n```"

    async def test_timeout(self):
        response = await _agent(["```python\nprint(1)\n```"] * 5, max_run_time=0.0).run("q")
        assert response.stop_reason is StopReason.TIMEOUT
        assert response.steps == 0

    async def test_budget_exhausted_total(self):
        # Each step costs 15 total tokens.
        response = await _agent(
            ["```python\nprint(1)\n```"] * 10,
            max_total_tokens=40,
        ).run("q")
        assert response.stop_reason is StopReason.BUDGET_EXHAUSTED
        assert response.usage.total_tokens >= 40

    async def test_budget_exhausted_input_only(self):
        response = await _agent(
            ["```python\nprint(1)\n```"] * 10,
            max_input_tokens=25,
        ).run("q")
        assert response.stop_reason is StopReason.BUDGET_EXHAUSTED

    async def test_budget_exhausted_output_only(self):
        response = await _agent(
            ["```python\nprint(1)\n```"] * 10,
            max_output_tokens=12,
        ).run("q")
        assert response.stop_reason is StopReason.BUDGET_EXHAUSTED

    async def test_crossing_step_is_recorded(self):
        """The step that crosses the cap completes; only the next is blocked."""
        agent = _agent(["```python\nprint(1)\n```"] * 10, max_total_tokens=15)
        response = await agent.run("q")
        assert response.steps == 1
        assert any(isinstance(m, ExecutionResultMessage) for m in agent.messages)

    async def test_no_budget_by_default(self):
        response = await _agent(["```python\nprint(1)\n```", "done"]).run("q")
        assert response.stop_reason is StopReason.COMPLETED


class TestSingleExecutionPath:
    async def test_run_matches_stream(self):
        run_response = await _agent(["```python\nprint(add(2, 2))\n```", "4"]).run("q")

        events = [
            e async for e in _agent(["```python\nprint(add(2, 2))\n```", "4"]).stream_events("q")
        ]
        stopped = [e for e in events if isinstance(e, StoppedEvent)][0]

        assert run_response.content == stopped.content
        assert run_response.steps == stopped.steps
        assert run_response.stop_reason is stopped.stop_reason

    async def test_only_first_code_block_executes(self):
        """A streamed turn stops at the first block, so a non-streamed one must
        too — otherwise the same model output would run different amounts of
        code depending on how it was consumed."""
        response = await _agent(
            [
                "```python\nprint('first')\n```\nand then\n```python\nprint('second')\n```",
                "done",
            ]
        ).run("q")

        assert response.code_snippets == ["print('first')"]

    async def test_run_collects_code_snippets(self):
        response = await _agent(
            [
                "```python\nprint(1)\n```",
                "```python\nprint(2)\n```",
                "done",
            ]
        ).run("q")
        assert response.code_snippets == ["print(1)", "print(2)"]


class TestUsageAccounting:
    async def test_usage_accumulates_across_steps(self):
        one_step = await _agent(["done"]).run("q")
        two_steps = await _agent(["```python\nprint(1)\n```", "done"]).run("q")
        assert two_steps.usage.total_tokens > one_step.usage.total_tokens

    async def test_code_producing_step_is_counted(self):
        """The loop stops reading at the first complete code block, so the
        provider's terminal usage chunk never arrives — and that is the *common*
        case for this agent. Left uncounted it silently disabled the token
        budgets and kept compaction on the bare character heuristic.
        """
        agent = _agent(["```python\nprint(1)\n```"], max_steps=1)
        response = await agent.run("q")

        assert response.steps == 1
        assert response.usage.total_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    async def test_budget_stops_a_run_of_only_code_steps(self):
        """Guards the wiring, not just the counter: before the fix every step
        here reported zero and the cap never fired."""
        agent = _agent(["```python\nprint(1)\n```"] * 10, max_total_tokens=1500)
        response = await agent.run("q")

        assert response.stop_reason is StopReason.BUDGET_EXHAUSTED
        assert response.steps < 10

    async def test_stopped_event_carries_usage(self):
        agent = CaveAgent(
            FakeModel(
                ["answer"],
                usage=TokenUsage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
            ),
            runtime=IPythonRuntime(),
        )
        events = [e async for e in agent.stream_events("q")]
        stopped = [e for e in events if isinstance(e, StoppedEvent)][0]
        assert stopped.usage.total_tokens == 15
        assert stopped.elapsed >= 0


class TestContentlessCompletions:
    @staticmethod
    def _agent(*, thinking="", refusal=""):
        from types import SimpleNamespace

        from cave_agent.models import Model, ModelResponse, StreamResponse

        class Response(StreamResponse):
            async def _open_stream(self):
                async def chunks():
                    if thinking or refusal:
                        yield SimpleNamespace(
                            usage=None,
                            choices=[
                                SimpleNamespace(
                                    finish_reason=None,
                                    delta=SimpleNamespace(
                                        content=None,
                                        refusal=refusal or None,
                                        reasoning=thinking or None,
                                        reasoning_content=None,
                                    ),
                                )
                            ],
                        )
                    yield SimpleNamespace(
                        usage=None,
                        choices=[
                            SimpleNamespace(
                                finish_reason="stop",
                                delta=SimpleNamespace(
                                    content=None,
                                    refusal=None,
                                    reasoning=None,
                                    reasoning_content=None,
                                ),
                            )
                        ],
                    )

                return chunks()

        class ContentlessModel(Model):
            async def _complete(self, messages):
                return ModelResponse(content="")

            def stream(self, messages):
                return Response()

        return CaveAgent(ContentlessModel(), runtime=IPythonRuntime())

    @pytest.mark.parametrize("thinking", ["", "private reasoning"])
    async def test_no_answer_is_not_completed(self, thinking):
        response = await self._agent(thinking=thinking).run("q")

        assert response.stop_reason is StopReason.MODEL_ERROR
        assert response.content == ""
        assert response.usage.prompt_tokens > 0
        if thinking:
            assert response.usage.completion_tokens > 0

    async def test_refusal_without_provider_usage_is_still_counted(self):
        response = await self._agent(refusal="No.").run("q")

        assert response.stop_reason is StopReason.COMPLETED
        assert response.content == "No."
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    async def test_reasoning_before_transport_failure_is_still_counted(self):
        from types import SimpleNamespace

        from cave_agent.models import Model, ModelResponse, StreamResponse

        class Response(StreamResponse):
            async def _open_stream(self):
                async def chunks():
                    yield SimpleNamespace(
                        usage=None,
                        choices=[
                            SimpleNamespace(
                                finish_reason=None,
                                delta=SimpleNamespace(
                                    content=None,
                                    refusal=None,
                                    reasoning="billed reasoning",
                                    reasoning_content=None,
                                ),
                            )
                        ],
                    )
                    raise ConnectionError("wire dropped")

                return chunks()

        class ReasoningModel(Model):
            async def _complete(self, messages):
                return ModelResponse(content="")

            def stream(self, messages):
                return Response()

        response = await CaveAgent(
            ReasoningModel(),
            runtime=IPythonRuntime(),
        ).run("q")

        assert response.stop_reason is StopReason.MODEL_ERROR
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0


class TestHistory:
    async def test_code_and_result_are_paired(self):
        agent = _agent(["```python\nprint(1)\n```", "done"])
        await agent.run("q")

        roles = [m.role for m in agent.messages]
        code_i = roles.index(MessageRole.CODE_EXECUTION)
        assert roles[code_i + 1] is MessageRole.EXECUTION_RESULT

    async def test_system_prompt_rebuilt_each_run(self):
        agent = _agent(["a", "b"])
        await agent.run("first")
        agent.runtime.inject_variable(
            __import__(
                "cave_agent.runtime",
                fromlist=["Variable"],
            ).Variable("late", 1, "added between runs")
        )

        await agent.run("second")

        assert "late" in agent.messages[0].content

    async def test_multi_turn_accumulates(self):
        agent = _agent(["first answer", "second answer"])
        await agent.run("q1")
        await agent.run("q2")
        assert sum(1 for m in agent.messages if m.role is MessageRole.USER) == 2


class TestCancellationSafety:
    class HangingRuntime(IPythonRuntime):
        async def execute(self, code: str) -> ExecutionResult:
            await asyncio.sleep(60)

    class InterruptCountingRuntime(HangingRuntime):
        def __init__(self):
            super().__init__()
            self.interrupt_calls = 0

        async def interrupt(self) -> None:
            self.interrupt_calls += 1

    async def _cancel_mid_execution(self) -> CaveAgent:
        agent = CaveAgent(
            model=FakeModel(["```python\nprint(1)\n```"]),
            runtime=self.HangingRuntime(),
        )
        task = asyncio.create_task(agent.run("q"))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return agent

    async def test_dangling_code_message_gets_a_result(self):
        """Otherwise history ends on an assistant turn claiming to have run code
        with no outcome, and the next run's user turn lands right after it."""
        agent = await self._cancel_mid_execution()

        roles = [m.role for m in agent.messages]
        code_i = roles.index(MessageRole.CODE_EXECUTION)
        assert roles[code_i + 1] is MessageRole.EXECUTION_RESULT

    async def test_interruption_marker_appended(self):
        agent = await self._cancel_mid_execution()
        assert agent.messages[-1].role is MessageRole.USER
        assert "interrupted" in agent.messages[-1].content.lower()

    async def test_history_is_reusable_after_cancel(self):
        agent = await self._cancel_mid_execution()
        agent.runtime = IPythonRuntime()
        agent.model = FakeModel(["recovered"])

        response = await agent.run("continue")

        assert response.stop_reason is StopReason.COMPLETED
        assert response.content == "recovered"

    async def test_no_placeholder_when_nothing_dangling(self):
        agent = _agent(["plain answer"])
        await agent.run("q")
        assert not any(isinstance(m, CodeExecutionMessage) for m in agent.messages)

    async def test_cancellation_does_not_interrupt_after_execution_unwinds(self):
        """Cancellation reaches execute directly; a later untargeted interrupt
        can only hit whichever request acquires the shared runtime next."""
        runtime = self.InterruptCountingRuntime()
        agent = CaveAgent(
            model=FakeModel(["```python\nprint(1)\n```"]),
            runtime=runtime,
        )
        task = asyncio.create_task(agent.run("q"))
        await asyncio.sleep(0.3)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert runtime.interrupt_calls == 0


class TestOutputRecovery:
    async def test_truncated_turn_is_resumed(self):
        model = FakeModel(["partial thought"], finish_reason="length")
        agent = CaveAgent(model=model, runtime=IPythonRuntime(), max_steps=2)

        events = [e async for e in agent.stream_events("q")]

        statuses = [e for e in events if isinstance(e, StatusEvent)]
        assert any(e.status is StatusType.OUTPUT_RECOVERY for e in statuses)
        assert any("Resume directly" in m.content for m in agent.messages)


class TestSecurity:
    async def test_blocked_code_surfaces_as_event(self):
        runtime = IPythonRuntime(
            security_checker=SecurityChecker([ImportRule({"os"})]),
        )
        agent = CaveAgent(
            model=FakeModel(["```python\nimport os\n```", "sorry"]),
            runtime=runtime,
        )

        events = [e async for e in agent.stream_events("q")]

        assert any(isinstance(e, SecurityErrorEvent) for e in events)

    async def test_blocked_code_still_pairs_history(self):
        runtime = IPythonRuntime(
            security_checker=SecurityChecker([ImportRule({"os"})]),
        )
        agent = CaveAgent(
            model=FakeModel(["```python\nimport os\n```", "sorry"]),
            runtime=runtime,
        )
        await agent.run("q")

        roles = [m.role for m in agent.messages]
        code_i = roles.index(MessageRole.CODE_EXECUTION)
        assert roles[code_i + 1] is MessageRole.EXECUTION_RESULT


class TestStalledStream:
    """A stalled provider stream must end the run, not escape it.

    Reported from production on 0.7.5: `TimeoutError('LLM stream stalled')`
    unwound out of `stream_events()`, through starlette's `collapse_excgroups`,
    and killed an in-flight SSE response. A server streaming to a browser needs
    a terminal event it can forward, not an exception through its middleware.
    """

    class StallingStream:
        def __init__(self, stall: float):
            self.usage = TokenUsage()
            self.finish_reason = None
            self.thinking = ""
            self._stall = stall

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(self._stall)
            raise StopAsyncIteration

        async def aclose(self):
            return None

    class StallingModel:
        max_output_tokens = None

        async def call(self, messages):
            raise NotImplementedError

        def stream(self, messages):
            return TestStalledStream.StallingStream(5.0)

    async def _run(self):
        return CaveAgent(
            model=self.StallingModel(),
            runtime=IPythonRuntime(),
            stream_idle_timeout=0.2,
        )

    async def test_does_not_raise(self):
        agent = await self._run()
        events = [e async for e in agent.stream_events("q")]
        assert isinstance(events[-1], StoppedEvent)

    async def test_reports_model_error(self):
        agent = await self._run()
        response = await agent.run("q")
        assert response.stop_reason is StopReason.MODEL_ERROR
        assert not response.completed

    async def test_stall_is_a_model_error_and_still_a_timeout(self):
        """Typed so the loop can classify it; still a TimeoutError so existing
        `except TimeoutError` handlers keep working."""
        from cave_agent.models import ModelError, StreamStalledError

        assert issubclass(StreamStalledError, ModelError)
        assert issubclass(StreamStalledError, TimeoutError)

    async def test_stall_after_terminal_reason_keeps_the_completed_answer(self):
        """Only the optional usage event may follow a terminal provider chunk."""
        from types import SimpleNamespace

        from cave_agent.models import ModelResponse, StreamResponse

        class TerminalThenHang(StreamResponse):
            async def _open_stream(self):
                async def chunks():
                    yield SimpleNamespace(
                        usage=None,
                        choices=[
                            SimpleNamespace(
                                finish_reason="stop",
                                delta=SimpleNamespace(
                                    content="finished answer",
                                    refusal=None,
                                    reasoning=None,
                                    reasoning_content=None,
                                ),
                            )
                        ],
                    )
                    await asyncio.Event().wait()

                return chunks()

        class Model:
            max_output_tokens = None

            def __init__(self):
                self.calls = 0

            async def call(self, messages):
                return ModelResponse("")

            def stream(self, messages):
                self.calls += 1
                return TerminalThenHang()

        model = Model()
        agent = CaveAgent(
            model=model,
            runtime=IPythonRuntime(),
            max_steps=2,
            stream_idle_timeout=0.02,
        )

        response = await agent.run("q")

        assert response.stop_reason is StopReason.COMPLETED
        assert response.content == "finished answer"
        assert model.calls == 1


class TestPartialContent:
    """Incomplete output must never be reported as a completed answer.

    Two ways a turn can be a fragment — the provider's output cap, and a stream
    that stops producing — and both used to end as `COMPLETED` with a truncated
    answer. They now share one mechanism: `_ModelTurn.complete`.
    """

    async def test_code_block_split_across_truncation_is_joined(self):
        """The fragments are separate calls but one logical utterance."""
        runtime = IPythonRuntime()
        agent = CaveAgent(
            model=FakeModel(
                ["Working...\n```python\nx = 4\nprint(", "x + 1)\n```", "five"],
                finish_reasons=["length", "stop", "stop"],
            ),
            runtime=runtime,
            max_steps=4,
        )

        events = [e async for e in agent.stream_events("q")]

        codes = [e.code for e in events if isinstance(e, CodeEvent)]
        assert codes == ["x = 4\nprint(x + 1)"], codes
        assert await runtime.get_from_namespace("x") == 4
        results = [e for e in events if isinstance(e, ExecutionResultEvent)]
        assert "5" in results[0].output

    async def test_recovery_status_is_emitted(self):
        agent = CaveAgent(
            model=FakeModel(["half a thought", " and the rest"], finish_reasons=["length", "stop"]),
            runtime=IPythonRuntime(),
            max_steps=3,
        )
        events = [e async for e in agent.stream_events("q")]
        statuses = [e for e in events if isinstance(e, StatusEvent)]
        assert any(e.status is StatusType.OUTPUT_RECOVERY for e in statuses)

    async def test_joined_answer_contains_both_fragments(self):
        response = await CaveAgent(
            model=FakeModel(["The answer is", " 42."], finish_reasons=["length", "stop"]),
            runtime=IPythonRuntime(),
            max_steps=3,
        ).run("q")

        assert response.content == "The answer is 42."
        assert response.stop_reason is StopReason.COMPLETED

    async def test_unrecoverable_truncation_is_not_completed(self):
        """Budget spent and still a fragment — kept, but not called an answer."""
        agent = CaveAgent(
            model=FakeModel(["still going"] * 8, finish_reason="length"),
            runtime=IPythonRuntime(),
            max_steps=8,
        )
        response = await agent.run("q")

        assert response.stop_reason is StopReason.MODEL_ERROR
        assert not response.completed
        assert "still going" in response.content


class TestUnterminatedFence:
    async def test_unterminated_fence_is_not_executed(self):
        """The live parser only reports *closed* blocks; the second parse path
        that could not tell the difference is gone."""
        runtime = IPythonRuntime()
        agent = CaveAgent(
            model=FakeModel(["```python\nflag = 'executed-without-close'"]),
            runtime=runtime,
            max_steps=1,
        )

        events = [e async for e in agent.stream_events("q")]

        assert not [e for e in events if isinstance(e, CodeEvent)]
        with pytest.raises(KeyError):
            await runtime.get_from_namespace("flag")

    async def test_closed_fence_still_executes(self):
        runtime = IPythonRuntime()
        agent = CaveAgent(
            model=FakeModel(["```python\nflag = 'ran'\n```", "done"]),
            runtime=runtime,
            max_steps=2,
        )
        await agent.run("q")
        assert await runtime.get_from_namespace("flag") == "ran"


class TestAlwaysStops:
    """Every stream consumed to termination ends with a `StoppedEvent`.

    `drain_to_response` raises when the terminal event is missing, so `run()`
    would fail with a confusing error instead of the real cause. A renderer
    would leave its live region open forever. The invariant is load-bearing, so
    it is asserted over every terminal reason rather than one path at a time.
    Explicit early close is different: ``GeneratorExit`` forbids yielding and
    is covered by the interruption-history tests.
    """

    def _failing_runtime(self, error: Exception):
        runtime = IPythonRuntime()

        async def explode(code: str):
            raise error

        runtime.execute = explode
        return runtime

    @pytest.mark.parametrize(
        "error, expected",
        [
            (ModelError("provider said no"), StopReason.MODEL_ERROR),
            (PromptTooLongError("too long"), StopReason.MODEL_ERROR),
            (ValueError("unclassified"), StopReason.MODEL_ERROR),
        ],
    )
    async def test_model_failure_still_stops(self, error, expected):
        agent = CaveAgent(model=FailingModel(error), runtime=IPythonRuntime())

        events = [e async for e in agent.stream_events("q")]

        assert isinstance(events[-1], StoppedEvent)
        assert events[-1].stop_reason is expected

    async def test_runtime_failure_stops_and_is_not_blamed_on_the_model(self):
        """A broken kernel is not a broken answer — the reasons stay distinct."""
        agent = CaveAgent(
            model=FakeModel(["```python\nprint(1)\n```", "done"]),
            runtime=self._failing_runtime(RuntimeExecutionError("kernel died")),
        )

        events = [e async for e in agent.stream_events("q")]

        assert isinstance(events[-1], StoppedEvent)
        assert events[-1].stop_reason is StopReason.RUNTIME_ERROR

    async def test_unexpected_failure_stops_then_propagates(self):
        """An internal defect is reported *and* re-raised — neither swallowed
        into a fake answer nor left without a terminal event."""
        agent = CaveAgent(
            model=FakeModel(["```python\nprint(1)\n```", "done"]),
            runtime=self._failing_runtime(KeyError("internal")),
        )

        events = []
        with pytest.raises(KeyError):
            async for event in agent.stream_events("q"):
                events.append(event)

        assert isinstance(events[-1], StoppedEvent)
        assert events[-1].stop_reason is StopReason.INTERNAL_ERROR

    async def test_explicit_close_after_consumer_break_does_not_raise(self):
        """`GeneratorExit` must be handled before the catch-all arm.

        Breaking an ``async for`` does not close its iterator. A renderer that
        stops early must close explicitly; if the catch-all arm then yields a
        StoppedEvent in response to ``GeneratorExit``, Python raises "async
        generator ignored GeneratorExit".
        """
        agent = _agent(["```python\nprint(1)\n```", "done"])
        stream = agent.stream_events("q")

        async with aclosing(stream):
            async for event in stream:
                if isinstance(event, CodeEvent):
                    break

        assert agent.messages[-1].content == USER_INTERRUPTION_PLACEHOLDER

    async def test_closed_stream_marks_request_interrupted_before_follow_up(self):
        agent = _agent(["working", "done"])
        stream = agent.stream_events("old request")

        await stream.__anext__()
        await stream.__anext__()
        await stream.aclose()
        await agent.run("new request")

        user_contents = [
            message.content for message in agent.messages if isinstance(message, UserMessage)
        ]
        assert user_contents[:3] == [
            "old request",
            USER_INTERRUPTION_PLACEHOLDER,
            "new request",
        ]


class TestInitializationFailure:
    """Rendering the system prompt is inside the terminal-event envelope.

    It runs a caller's template and the runtime's `describe_*()`, either of
    which can raise. Outside the envelope, that raise escaped with no
    StoppedEvent — leaving `run()` to fail on the missing terminal event and a
    renderer's live region open forever, at the one moment a consumer has
    nothing else to go on.
    """

    async def test_bad_template_still_stops(self):
        agent = CaveAgent(
            model=FakeModel(["hi"]),
            runtime=IPythonRuntime(),
            system_prompt_template="{no_such_field}",
        )

        events = []
        with pytest.raises(KeyError):
            async for event in agent.stream_events("q"):
                events.append(event)

        assert isinstance(events[-1], StoppedEvent)
        assert events[-1].stop_reason is StopReason.INTERNAL_ERROR


class TestStreamCloseFailure:
    """Per-turn cleanup cannot replace the stream's semantic outcome."""

    class DyingCloseModel:
        max_output_tokens = None

        async def call(self, messages):
            raise NotImplementedError

        def stream(self, messages):
            from cave_agent.models import StreamDelta, StreamResponse, TokenUsage

            class RawStream:
                """Stands in for the provider SDK's stream object."""

                def __aiter__(self):
                    async def gen():
                        yield "the complete answer"

                    return gen()

                async def aclose(self):
                    raise ConnectionError("connection reset while closing")

            class Response(StreamResponse):
                def __init__(self):
                    super().__init__()
                    self.usage = TokenUsage(1, 1, 2)

                async def _open_stream(self):
                    return RawStream()

                def _process_stream_chunk(self, chunk):
                    self.finish_reason = "stop"
                    return StreamDelta(content=chunk)

            return Response()

    async def test_close_failure_does_not_discard_completed_answer(self, caplog):
        agent = CaveAgent(model=self.DyingCloseModel(), runtime=IPythonRuntime())

        events = [e async for e in agent.stream_events("q")]

        assert isinstance(events[-1], StoppedEvent)
        assert events[-1].stop_reason is StopReason.COMPLETED
        assert events[-1].content == "the complete answer"
        assert any(
            record.getMessage() == "Failed to close the provider stream"
            for record in caplog.records
        )

    async def test_close_failure_does_not_replace_read_error(self, caplog):
        from cave_agent.models import (
            ProviderBillingError,
            StreamResponse,
        )

        class BillingError(Exception):
            status_code = 402

        class RawStream:
            def __aiter__(self):
                async def gen():
                    raise BillingError("insufficient credits")
                    yield

                return gen()

            async def aclose(self):
                raise ConnectionError("connection reset while closing")

        class Response(StreamResponse):
            async def _open_stream(self):
                return RawStream()

        class Model:
            max_output_tokens = None

            async def call(self, messages):
                raise NotImplementedError

            def stream(self, messages):
                return Response()

        agent = CaveAgent(model=Model(), runtime=IPythonRuntime())

        events = [e async for e in agent.stream_events("q")]

        assert isinstance(events[-1], StoppedEvent)
        assert events[-1].stop_reason is StopReason.MODEL_ERROR
        model_errors = [
            record for record in caplog.records if record.getMessage() == "Model error — ending run"
        ]
        assert len(model_errors) == 1
        assert isinstance(model_errors[0].exc_info[1], ProviderBillingError)

    async def test_connect_retry_survives_a_dying_close(self):
        """The retry path closes a stream it is already discarding.

        We retry *because* the transport failed, so the close usually fails
        too. Letting that propagate would turn every recoverable connect error
        whose socket also dies on close into a terminal one.
        """
        from cave_agent.models import StreamDelta, StreamResponse

        class Flaky(StreamResponse):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            async def _open_stream(self):
                self.attempts += 1
                if self.attempts == 1:

                    class Dead:
                        def __aiter__(self):
                            async def gen():
                                raise ConnectionError("connection reset")
                                yield

                            return gen()

                        async def aclose(self):
                            raise ConnectionError("reset on close too")

                    return Dead()

                class Good:
                    def __aiter__(self):
                        async def gen():
                            yield "recovered"

                        return gen()

                    async def aclose(self):
                        return None

                return Good()

            def _process_stream_chunk(self, chunk):
                return StreamDelta(content=chunk)

        stream = Flaky()
        assert [d.content async for d in stream] == ["recovered"]
        assert stream.attempts == 2

    async def test_disconnect_after_terminal_reason_does_not_resume(self):
        """A terminal provider chunk completes the answer even if usage is lost."""
        from types import SimpleNamespace

        from cave_agent.models import Model, ModelResponse, StreamResponse

        def chunk(content=None, finish_reason=None):
            return SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason=finish_reason,
                        delta=SimpleNamespace(
                            content=content,
                            refusal=None,
                            reasoning=None,
                            reasoning_content=None,
                        ),
                    )
                ],
            )

        class Response(StreamResponse):
            async def _open_stream(self):
                async def raw():
                    yield chunk(content="COMPLETE")
                    yield chunk(finish_reason="stop")
                    raise ConnectionError("usage chunk was lost")

                return raw()

        class TerminalThenDisconnect(Model):
            def __init__(self):
                self.calls = 0

            async def _complete(self, messages):
                return ModelResponse(content="unused")

            def stream(self, messages):
                self.calls += 1
                return Response()

        model = TerminalThenDisconnect()
        response = await CaveAgent(model=model, runtime=IPythonRuntime()).run("q")

        assert response.stop_reason is StopReason.COMPLETED
        assert response.content == "COMPLETE"
        assert model.calls == 1


class TestMultipleFencesInOneDelta:
    """The agent runs the FIRST complete code block, whatever the chunking.

    A fast model or a buffering proxy can deliver a whole response as one
    delta. The parser kept going after the first fence closed, each CODE
    segment overwrote the last, and the agent executed the *second* block while
    silently dropping the first. Chunk size is a property of the transport, so
    behaviour that varies with it is a bug by construction.
    """

    TWO_BLOCKS = (
        "Here:\n```python\nwhich = 'first'\n```\nand also\n```python\nwhich = 'second'\n```\n"
    )

    def _agent_over(self, chunk_size, runtime):
        from cave_agent.models import StreamDelta, StreamResponse, TokenUsage

        two_blocks = self.TWO_BLOCKS

        class Chunked(StreamResponse):
            def __init__(self, text):
                super().__init__()
                self._text = text
                self.usage = TokenUsage(1, 1, 2)

            async def _open_stream(self):
                text, size = self._text, chunk_size

                async def gen():
                    for start in range(0, len(text), size):
                        yield text[start : start + size]
                    self.finish_reason = "stop"

                return gen()

            def _process_stream_chunk(self, chunk):
                return StreamDelta(content=chunk)

        class Model:
            max_output_tokens = None

            def __init__(self):
                self.calls = 0

            async def call(self, messages):
                raise NotImplementedError

            def stream(self, messages):
                self.calls += 1
                return Chunked(two_blocks if self.calls == 1 else "done")

        return CaveAgent(model=Model(), runtime=runtime, max_steps=3)

    @pytest.mark.parametrize("chunk_size", [1, 13, 10_000])
    async def test_first_block_executes(self, chunk_size):
        runtime = IPythonRuntime()
        agent = self._agent_over(chunk_size, runtime)

        events = [e async for e in agent.stream_events("q")]

        assert [e.code for e in events if isinstance(e, CodeEvent)] == ["which = 'first'"]
        assert await runtime.get_from_namespace("which") == "first"

    @pytest.mark.parametrize("chunk_size", [1, 13, 10_000])
    async def test_recorded_turn_ends_at_the_fence(self, chunk_size):
        """History must not claim the model wrote a block that never ran."""
        agent = self._agent_over(chunk_size, IPythonRuntime())
        await agent.run("q")

        recorded = [m.content for m in agent.messages if isinstance(m, CodeExecutionMessage)]
        assert recorded == ["Here:\n```python\nwhich = 'first'\n```"]


class TestCloseNeverOutranksTheRealFailure:
    """A raise from the `finally` close replaces whatever was propagating.

    Closing a socket that just died usually fails too, so an unguarded close
    outranked the failure it was cleaning up after: a cancelled run swallowed
    its `CancelledError` and reported MODEL_ERROR, and a mid-stream read
    failure lost the partial content that output recovery exists to salvage.
    """

    @staticmethod
    def _model(mode):
        from cave_agent.models import Model, StreamDelta, StreamResponse, TokenUsage

        class Raw:
            def __aiter__(self):
                async def gen():
                    if mode == "hang":
                        await asyncio.sleep(30)
                        yield "never"
                    else:
                        yield "partial answer, cut off"
                        raise ConnectionError("read failed mid-stream")

                return gen()

            async def aclose(self):
                raise ConnectionError("reset while closing")

        class Response(StreamResponse):
            def __init__(self):
                super().__init__()
                self.usage = TokenUsage(1, 1, 2)

            async def _open_stream(self):
                return Raw()

            def _process_stream_chunk(self, chunk):
                return StreamDelta(content=chunk)

        class DyingModel(Model):
            async def _complete(self, messages):
                raise NotImplementedError

            def stream(self, messages):
                return Response()

        return DyingModel()

    async def test_cancellation_survives_a_failing_close(self):
        agent = CaveAgent(model=self._model("hang"), runtime=IPythonRuntime())
        events = []

        async def run():
            async for event in agent.stream_events("q"):
                events.append(event)

        task = asyncio.create_task(run())
        await asyncio.sleep(0.3)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert events[-1].stop_reason is StopReason.CANCELLED

    async def test_read_failure_still_reaches_output_recovery(self):
        """The partial content is the whole point of the recovery path."""
        agent = CaveAgent(model=self._model("die"), runtime=IPythonRuntime(), max_steps=2)

        events = [e async for e in agent.stream_events("q")]

        recoveries = [
            e
            for e in events
            if isinstance(e, StatusEvent) and e.status is StatusType.OUTPUT_RECOVERY
        ]
        assert recoveries, "a mid-stream failure with content must be recoverable"
        assert isinstance(events[-1], StoppedEvent)


class TestStreamCloseIsBounded:
    """A close is bookkeeping; it never holds the run hostage.

    Unbounded, a hanging `aclose()` blocked the loop between reading a
    complete code block and executing it, and left a cancelled run alive
    needing a second cancel.
    """

    @staticmethod
    def _model(payload):
        from cave_agent.models import Model, StreamDelta, StreamResponse, TokenUsage

        class Raw:
            def __aiter__(self):
                async def gen():
                    if payload is None:
                        await asyncio.sleep(30)
                        yield "never"
                    else:
                        yield payload

                return gen()

            async def aclose(self):
                await asyncio.sleep(60)  # a closer that never returns

        class Response(StreamResponse):
            def __init__(self):
                super().__init__()
                self.usage = TokenUsage(1, 1, 2)

            async def _open_stream(self):
                return Raw()

            def _process_stream_chunk(self, chunk):
                return StreamDelta(content=chunk)

        class HangingCloseModel(Model):
            async def _complete(self, messages):
                raise NotImplementedError

            def stream(self, messages):
                return Response()

        return HangingCloseModel()

    async def test_a_hanging_close_does_not_block_execution(self):
        from cave_agent.agent import _STREAM_CLOSE_TIMEOUT

        runtime = IPythonRuntime()
        agent = CaveAgent(
            model=self._model("run this:\n```python\nx = 42\n```"),
            runtime=runtime,
            max_steps=1,
        )

        response = await asyncio.wait_for(
            agent.run("q"),
            timeout=_STREAM_CLOSE_TIMEOUT + 5.0,
        )

        # The block ran despite a closer that never returns.
        assert await runtime.get_from_namespace("x") == 42
        assert response.stop_reason is StopReason.MAX_STEPS

    async def test_cancellation_does_not_wait_for_the_close_at_all(self):
        """The user asked to stop; any wait here is a wait they asked to end."""
        agent = CaveAgent(model=self._model(None), runtime=IPythonRuntime())

        async def run():
            async for _ in agent.stream_events("q"):
                pass

        task = asyncio.create_task(run())
        await asyncio.sleep(0.3)
        task.cancel()

        # One cancel, and it lands well inside the normal close grace period.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)


class TestAllocationFailuresAreRuntimeErrors:
    """Persisting an oversize result is runtime work, and fails as runtime work.

    Only `execute` carried the envelope that types a backend failure, so an
    allocation that timed out escaped as a bare `TimeoutError` and the run
    ended as INTERNAL_ERROR — "the agent has a defect" — for a runtime that
    was merely slow.
    """

    async def test_a_failing_allocation_stops_with_runtime_error(self):
        runtime = IPythonRuntime()

        async def refuse(prefix, value, **kwargs):
            raise TimeoutError("backend timed out during allocation")

        runtime._executor.bind_unique = refuse
        agent = CaveAgent(
            model=FakeModel(["```python\nprint('x' * 9000)\n```", "done"]),
            runtime=runtime,
            max_exec_output=500,
        )

        events = [e async for e in agent.stream_events("q")]

        assert isinstance(events[-1], StoppedEvent)
        assert events[-1].stop_reason is StopReason.RUNTIME_ERROR

    async def test_an_already_typed_failure_is_not_rewrapped(self):
        runtime = IPythonRuntime()

        async def refuse(prefix, value, **kwargs):
            raise RuntimeExecutionError("kernel died")

        runtime._executor.bind_unique = refuse
        agent = CaveAgent(
            model=FakeModel(["```python\nprint('x' * 9000)\n```", "done"]),
            runtime=runtime,
            max_exec_output=500,
        )

        events = [e async for e in agent.stream_events("q")]

        assert events[-1].stop_reason is StopReason.RUNTIME_ERROR


class TestCancellationStillReleasesTheStream:
    """Cancelling must not wait for the close — but the close must still happen.

    Detaching by cancelling a just-created task runs it zero times, so the
    provider stream was never released at all: the previous version reclaimed
    the task by making sure it did nothing.
    """

    @staticmethod
    def _model(started):
        from cave_agent.models import Model, StreamDelta, StreamResponse, TokenUsage

        class Raw:
            def __aiter__(self):
                async def gen():
                    yield "reading..."
                    await asyncio.sleep(30)

                return gen()

            async def aclose(self):
                started["began"] = True
                await asyncio.sleep(30)

        class Response(StreamResponse):
            def __init__(self):
                super().__init__()
                self.usage = TokenUsage(1, 1, 2)

            async def _open_stream(self):
                return Raw()

            def _process_stream_chunk(self, chunk):
                return StreamDelta(content=chunk)

        class BlockingModel(Model):
            async def _complete(self, messages):
                raise NotImplementedError

            def stream(self, messages):
                return Response()

        return BlockingModel()

    async def test_the_close_actually_begins(self):
        started = {"began": False}
        agent = CaveAgent(model=self._model(started), runtime=IPythonRuntime())

        async def run():
            async for _ in agent.stream_events("q"):
                pass

        task = asyncio.create_task(run())
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await asyncio.sleep(0.05)
        assert started["began"], "aclose() never ran a single step"

    async def test_the_caller_is_not_made_to_wait(self):
        started = {"began": False}
        agent = CaveAgent(model=self._model(started), runtime=IPythonRuntime())

        async def run():
            async for _ in agent.stream_events("q"):
                pass

        task = asyncio.create_task(run())
        await asyncio.sleep(0.3)
        task.cancel()

        # Well inside the close grace period: cancellation does not wait it out.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)


class TestFilteredResponses:
    """A content filter is not an empty answer.

    The refusal text the provider sends is the model's answer — but it is prose,
    so it must not reach the code parser. With no text at all there is no
    answer, and the run must not report success.
    """

    @staticmethod
    def _agent(refusal):
        from openai.types.chat.chat_completion_chunk import (
            ChatCompletionChunk,
            Choice,
            ChoiceDelta,
        )

        from cave_agent.models import StreamResponse
        from cave_agent.models.openai import _OpenAIStreamResponse

        def chunk(*, refusal=None, finish_reason=None):
            return ChatCompletionChunk(
                id="refusal-probe",
                created=0,
                model="probe",
                object="chat.completion.chunk",
                choices=[
                    Choice(
                        index=0,
                        finish_reason=finish_reason,
                        delta=ChoiceDelta(refusal=refusal),
                    )
                ],
            )

        chunks = [
            chunk(refusal=refusal),
            chunk(finish_reason="content_filter"),
        ]

        class Response(_OpenAIStreamResponse):
            def __init__(self):
                StreamResponse.__init__(self)

            async def _open_stream(self):
                async def gen():
                    for chunk in chunks:
                        yield chunk

                return gen()

        class FilteredModel:
            max_output_tokens = None

            async def call(self, messages):
                raise NotImplementedError

            def stream(self, messages):
                return Response()

        return CaveAgent(model=FilteredModel(), runtime=IPythonRuntime())

    async def test_a_refusal_becomes_the_answer(self):
        response = await self._agent("I can't help with that.").run("q")

        assert response.stop_reason is StopReason.COMPLETED
        assert response.content == "I can't help with that."

    async def test_a_refusal_is_not_parsed_for_code(self):
        agent = self._agent("Here is why not:\n```python\nprint('nope')\n```")
        events = [e async for e in agent.stream_events("q")]

        assert not [e for e in events if isinstance(e, CodeEvent)]

    async def test_a_filter_with_no_text_is_a_model_error(self):
        response = await self._agent(None).run("q")

        assert response.stop_reason is StopReason.MODEL_ERROR


class TestRefusalsAreAnswers:
    """Providers send a refusal under an ordinary "stop" as readily as under
    "content_filter" — it is the answer either way, and it is prose."""

    @staticmethod
    def _agent(refusal, finish_reason):
        from openai.types.chat.chat_completion_chunk import (
            ChatCompletionChunk,
            Choice,
            ChoiceDelta,
        )

        from cave_agent.models import StreamResponse
        from cave_agent.models.openai import _OpenAIStreamResponse

        def chunk(*, refusal=None, finish_reason=None):
            return ChatCompletionChunk(
                id="refusal-probe",
                created=0,
                model="probe",
                object="chat.completion.chunk",
                choices=[
                    Choice(
                        index=0,
                        finish_reason=finish_reason,
                        delta=ChoiceDelta(refusal=refusal),
                    )
                ],
            )

        chunks = [
            chunk(refusal=refusal),
            chunk(finish_reason=finish_reason),
        ]

        class Response(_OpenAIStreamResponse):
            def __init__(self):
                StreamResponse.__init__(self)

            async def _open_stream(self):
                async def gen():
                    for chunk in chunks:
                        yield chunk

                return gen()

        class RefusingModel:
            max_output_tokens = None

            async def call(self, messages):
                raise NotImplementedError

            def stream(self, messages):
                return Response()

        return CaveAgent(model=RefusingModel(), runtime=IPythonRuntime())

    @pytest.mark.parametrize("finish_reason", ["stop", "content_filter"])
    async def test_a_refusal_is_the_answer_whatever_the_finish_reason(self, finish_reason):
        response = await self._agent("I cannot help with that.", finish_reason).run("q")

        assert response.stop_reason is StopReason.COMPLETED
        assert response.content == "I cannot help with that."

    async def test_a_non_streaming_refusal_is_not_an_empty_answer(self):
        from types import SimpleNamespace

        from cave_agent.models.openai import OpenAIModel

        message = SimpleNamespace(content=None, refusal="I cannot help with that.")
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
        )

        assert OpenAIModel._extract_response(response) == ("", "stop", "I cannot help with that.")


class TestAsynchronousContentFiltering:
    """Some providers send the filter verdict after the content it applies to.

    Stopping at the first complete code block and running it would execute code
    the verdict was about to withdraw, so `filters_asynchronously` makes the
    agent read to the end of the stream first.
    """

    CODE = "```python\nfiltered_code_ran = True\n```"

    @staticmethod
    def _model(chunks_per_call, *, filters_asynchronously):
        from cave_agent.models import StreamResponse
        from cave_agent.models.openai import _OpenAIStreamResponse

        class Response(_OpenAIStreamResponse):
            def __init__(self, chunks):
                StreamResponse.__init__(self)
                self._chunks = chunks

            async def _open_stream(self):
                async def gen():
                    for chunk in self._chunks:
                        yield chunk

                return gen()

        class FilteringModel:
            max_output_tokens = None

            def __init__(self):
                self.call_index = 0

            async def call(self, messages):
                raise NotImplementedError

            def stream(self, messages):
                chunks = chunks_per_call[min(self.call_index, len(chunks_per_call) - 1)]
                self.call_index += 1
                return Response(chunks)

        FilteringModel.filters_asynchronously = filters_asynchronously
        return FilteringModel()

    @staticmethod
    def _chunk(finish_reason=None, content=None):
        # Use the SDK's exact production type. Hand-built SimpleNamespaces used
        # to accept shapes that neither the OpenAI nor Azure stream actually
        # emits, masking parser/state-machine defects.
        from openai.types.chat.chat_completion_chunk import (
            ChatCompletionChunk,
            Choice,
            ChoiceDelta,
        )

        return ChatCompletionChunk(
            id="filter-probe",
            created=0,
            model="probe",
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    finish_reason=finish_reason,
                    delta=ChoiceDelta(content=content),
                )
            ],
        )

    async def test_filtered_code_does_not_run(self):
        runtime = IPythonRuntime()
        agent = CaveAgent(
            model=self._model(
                [[self._chunk(content=self.CODE), self._chunk(finish_reason="content_filter")]],
                filters_asynchronously=True,
            ),
            runtime=runtime,
            max_steps=1,
        )

        response = await agent.run("q")

        with pytest.raises(KeyError):
            await runtime.get_from_namespace("filtered_code_ran")
        assert response.stop_reason is StopReason.MODEL_ERROR

    async def test_a_clean_stream_still_executes(self):
        """Deferring must not disable execution, only postpone the decision."""
        runtime = IPythonRuntime()
        agent = CaveAgent(
            model=self._model(
                [
                    [
                        self._chunk(content="```python\nran = 1\n```"),
                        self._chunk(finish_reason="stop"),
                    ],
                    [self._chunk(content="done"), self._chunk(finish_reason="stop")],
                ],
                filters_asynchronously=True,
            ),
            runtime=runtime,
            max_steps=3,
        )

        response = await agent.run("q")

        assert await runtime.get_from_namespace("ran") == 1
        assert response.stop_reason is StopReason.COMPLETED

    async def test_usage_estimate_counts_content_drained_after_the_code_block(self):
        from cave_agent.compaction import default_token_estimate

        tail = "X" * 4_000
        agent = CaveAgent(
            model=self._model(
                [
                    [
                        self._chunk(content="```python\nran = 1\n```" + tail),
                        self._chunk(finish_reason="stop"),
                    ],
                    [self._chunk(content="done"), self._chunk(finish_reason="stop")],
                ],
                filters_asynchronously=True,
            ),
            runtime=IPythonRuntime(),
            max_steps=3,
        )

        response = await agent.run("q")

        assert response.usage.completion_tokens >= default_token_estimate(tail)

    async def test_partial_content_cut_by_a_filter_is_not_an_answer(self):
        agent = CaveAgent(
            model=self._model(
                [
                    [
                        self._chunk(content="partial answer"),
                        self._chunk(finish_reason="content_filter"),
                    ]
                ],
                filters_asynchronously=False,
            ),
            runtime=IPythonRuntime(),
        )

        response = await agent.run("q")

        assert response.stop_reason is StopReason.MODEL_ERROR

    async def test_clean_eof_without_filter_verdict_fails_closed(self):
        """EOF is transport state, not proof that delayed filtering approved."""
        runtime = IPythonRuntime()
        model = self._model(
            [[self._chunk(content=self.CODE)]],
            filters_asynchronously=True,
        )
        agent = CaveAgent(model=model, runtime=runtime, max_steps=3)

        response = await agent.run("q")

        assert response.stop_reason is StopReason.MODEL_ERROR
        with pytest.raises(KeyError):
            await runtime.get_from_namespace("filtered_code_ran")
        assert model.call_index == 1

    async def test_normal_clean_eof_without_finish_reason_is_recovered(self):
        """A missing terminal chunk is a fragment, even without async filtering."""
        model = self._model(
            [
                [self._chunk(content="The answer is")],
                [self._chunk(content=" 42"), self._chunk(finish_reason="stop")],
            ],
            filters_asynchronously=False,
        )
        agent = CaveAgent(model=model, runtime=IPythonRuntime(), max_steps=3)

        response = await agent.run("q")

        assert response.stop_reason is StopReason.COMPLETED
        assert response.content == "The answer is 42"
        assert model.call_index == 2

    async def test_filtered_fragment_is_not_recovered_or_persisted(self):
        """Withdrawn content must not become a continuation prompt."""
        partial = "provider withdrew this partial answer"
        model = self._model(
            [[self._chunk(content=partial), self._chunk(finish_reason="content_filter")]],
            filters_asynchronously=True,
        )
        agent = CaveAgent(model=model, runtime=IPythonRuntime(), max_steps=5)

        response = await agent.run("q")

        assert response.stop_reason is StopReason.MODEL_ERROR
        assert model.call_index == 1
        assert all(partial not in message.content for message in agent.messages)

    async def test_length_after_closed_code_only_describes_ignored_tail(self):
        """A complete block is executable even if trailing prose hit the cap."""
        runtime = IPythonRuntime()
        model = self._model(
            [
                [
                    self._chunk(content="```python\nclosed_code_ran = 1\n```"),
                    self._chunk(finish_reason="length"),
                ],
                [self._chunk(content="done"), self._chunk(finish_reason="stop")],
            ],
            filters_asynchronously=True,
        )
        agent = CaveAgent(model=model, runtime=runtime, max_steps=3)

        response = await agent.run("q")

        assert response.stop_reason is StopReason.COMPLETED
        assert response.content == "done"
        assert await runtime.get_from_namespace("closed_code_ran") == 1
        assert model.call_index == 2


class TestProviderChunkShapes:
    """A provider's message variants are the provider's business, not a defect."""

    @staticmethod
    def _chunk(finish_reason=None, content=None, delta=True):
        from types import SimpleNamespace

        body = (
            SimpleNamespace(
                content=content,
                refusal=None,
                reasoning=None,
                reasoning_content=None,
            )
            if delta
            else None
        )
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(finish_reason=finish_reason, delta=body)],
        )

    def _agent(self, chunks):
        from cave_agent.models import StreamResponse
        from cave_agent.models.openai import _OpenAIStreamResponse

        class Response(_OpenAIStreamResponse):
            def __init__(self):
                StreamResponse.__init__(self)

            async def _open_stream(self):
                async def gen():
                    for chunk in chunks:
                        yield chunk

                return gen()

        class ChunkModel:
            max_output_tokens = None

            async def call(self, messages):
                raise NotImplementedError

            def stream(self, messages):
                return Response()

        return CaveAgent(model=ChunkModel(), runtime=IPythonRuntime())

    async def test_a_chunk_with_no_delta_is_metadata(self):
        """Azure's asynchronous filter sends annotations this way."""
        agent = self._agent(
            [
                self._chunk(delta=False),
                self._chunk(content="hi"),
                self._chunk(finish_reason="stop"),
            ]
        )

        response = await agent.run("q")

        assert response.stop_reason is StopReason.COMPLETED
        assert response.content == "hi"

    async def test_usage_and_finish_reason_on_one_terminal_chunk_both_survive(self):
        """LiteLLM's Anthropic adapter combines both fields in one event."""
        from litellm.types.utils import ModelResponseStream

        from cave_agent.models import StreamResponse
        from cave_agent.models.litellm import _LiteLLMStreamResponse

        def chunk(*, content=None, finish_reason=None, usage=None):
            return ModelResponseStream(
                choices=[
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": finish_reason,
                    }
                ],
                usage=usage,
            )

        calls = [
            [
                chunk(content="PARTIAL "),
                chunk(
                    finish_reason="length",
                    usage={"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
                ),
            ],
            [
                chunk(content="FINISHED"),
                chunk(
                    finish_reason="stop",
                    usage={"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
                ),
            ],
        ]

        class Response(_LiteLLMStreamResponse):
            def __init__(self, raw_chunks):
                StreamResponse.__init__(self)
                self._raw_chunks = raw_chunks

            async def _open_stream(self):
                async def gen():
                    for raw_chunk in self._raw_chunks:
                        yield raw_chunk

                return gen()

        class CombinedTerminalModel:
            max_output_tokens = None
            filters_asynchronously = False

            def __init__(self):
                self.call_index = 0

            def stream(self, messages):
                response = Response(calls[self.call_index])
                self.call_index += 1
                return response

        model = CombinedTerminalModel()
        agent = CaveAgent(model=model, runtime=IPythonRuntime(), max_steps=3)

        response = await agent.run("q")

        assert response.stop_reason is StopReason.COMPLETED
        assert response.content == "PARTIAL FINISHED"
        assert response.usage.total_tokens == 26
        assert model.call_index == 2

    async def test_an_unparseable_chunk_is_a_provider_error(self):
        class Unparseable:
            @property
            def choices(self):
                raise TypeError("bad chunk")

        agent = self._agent([Unparseable()])

        events = [e async for e in agent.stream_events("q")]

        assert events[-1].stop_reason is StopReason.MODEL_ERROR


class TestAnEmptyFenceDoesNotEndTheTurn:
    """The agent stops at the first *complete* code block and runs it. An empty
    fence has nothing to run, so completing on it stopped the stream holding no
    code: the block that followed never ran, and the turn was recorded as an
    answer truncated at the empty fence.
    """

    async def test_the_real_block_is_the_one_executed(self):
        agent = CaveAgent(
            model=FakeModel(
                [
                    "Let me start.\n```python\n```\nOops, real code:\n"
                    "```python\nmarker = 'executed'\nprint(marker)\n```",
                ]
            ),
            runtime=IPythonRuntime(),
        )

        response = await agent.run("go")

        assert "executed" in "".join(
            m.content for m in agent.messages if m.role == MessageRole.EXECUTION_RESULT
        )
        assert response.stop_reason is StopReason.COMPLETED

    async def test_an_empty_fence_alone_is_an_answer_not_an_execution(self):
        agent = CaveAgent(
            model=FakeModel(["Nothing to run here.\n```python\n```"]),
            runtime=IPythonRuntime(),
        )

        response = await agent.run("go")

        assert response.stop_reason is StopReason.COMPLETED
        assert not [m for m in agent.messages if m.role == MessageRole.CODE_EXECUTION]


class TestAnIndentedBlockRuns:
    """A closing fence is accepted with up to three leading spaces, so an
    indented block closes — and stripping only its first line handed the model
    a SyntaxError for code it wrote correctly."""

    async def test_a_uniformly_indented_block_executes(self):
        agent = CaveAgent(
            model=FakeModel(["Here:\n```python\n   value = 6 * 7\n   print(value)\n   ```"]),
            runtime=IPythonRuntime(),
        )

        await agent.run("go")

        results = [m.content for m in agent.messages if m.role == MessageRole.EXECUTION_RESULT]
        assert "42" in "".join(results)
        assert "SyntaxError" not in "".join(results)
