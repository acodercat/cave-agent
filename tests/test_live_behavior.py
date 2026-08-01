"""Live-provider tests for behaviour a fake cannot prove.

Everything here needs ``LLM_MODEL_ID`` / ``LLM_API_KEY`` / ``LLM_BASE_URL``
(the ``model`` fixture errors without them, like every live suite). Assertions
are structural — stop reason, event shape, namespace state — never exact
wording, because the model's prose is not ours to pin.

What earns a live test is the seam between this library and a real provider:
real chunk boundaries into the streaming parser, real usage accounting into
the budget, and real multi-step responses driving persistence and recovery.
Unit suites already pin each mechanism against controlled input; these prove
the wiring under traffic we do not control.
"""

from cave_agent import CaveAgent
from cave_agent.events import ExecutionResultEvent, StoppedEvent, StopReason
from cave_agent.messages import MessageRole
from cave_agent.runtime import IPythonRuntime


def _execution_outputs(agent: CaveAgent) -> list[str]:
    """The execution results in history, oldest first — what the model saw."""
    return [m.content for m in agent.messages if m.role == MessageRole.EXECUTION_RESULT]


class TestLiveUsageAccounting:
    async def test_usage_is_reported_and_summed(self, model):
        agent = CaveAgent(model=model, runtime=IPythonRuntime(), max_steps=3)

        response = await agent.run("Reply with the single word: ready")

        assert response.stop_reason is StopReason.COMPLETED
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens >= (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

    async def test_a_spent_budget_stops_the_run_between_steps(self, model):
        """The crossing step completes and is recorded; the next never starts."""
        agent = CaveAgent(
            model=model,
            runtime=IPythonRuntime(),
            max_steps=5,
            max_total_tokens=1,
        )

        response = await agent.run(
            "Use Python to compute 7 * 6 and print the result. "
            "After seeing the output, state the answer."
        )

        assert response.stop_reason is StopReason.BUDGET_EXHAUSTED
        assert response.steps == 1
        assert response.usage.total_tokens > 1


class TestLiveOversizeOutput:
    async def test_oversize_stdout_is_persisted_and_reachable(self, model):
        runtime = IPythonRuntime()
        agent = CaveAgent(
            model=model,
            runtime=runtime,
            max_steps=4,
            max_exec_output=500,
            exec_output_preview_chars=100,
        )

        persisted = []
        async for event in agent.stream_events(
            "Run exactly this code and nothing else: print('x' * 5000). Then tell me it is done."
        ):
            if isinstance(event, ExecutionResultEvent) and event.persisted:
                persisted.append(event)

        assert persisted, "an oversize output should have been persisted"
        name = persisted[0].persisted_variable
        assert name is not None
        stored = await runtime.get_from_namespace(name)
        assert isinstance(stored, str)
        assert len(stored) >= 5000
        assert set(stored.strip()) == {"x"}

    async def test_the_model_can_keep_working_from_the_marker(self, model):
        """The marker names a real variable the model's next code can read."""
        runtime = IPythonRuntime()
        agent = CaveAgent(
            model=model,
            runtime=runtime,
            max_steps=6,
            max_exec_output=300,
            exec_output_preview_chars=50,
        )

        response = await agent.run(
            "Step 1: run print('needle-' + 'hay' * 2000). "
            "Step 2: the output was too large and was stored in a runtime "
            "variable named in the result marker — slice that variable to find "
            "the text before the first '-' and tell me what it is."
        )

        assert response.stop_reason is StopReason.COMPLETED
        assert "needle" in response.content


class TestLiveStreamingParsing:
    async def test_real_chunk_boundaries_produce_exactly_one_block_per_turn(self, model):
        """Provider chunking is arbitrary; each turn still runs one block."""
        agent = CaveAgent(model=model, runtime=IPythonRuntime(), max_steps=4)

        events = [
            event
            async for event in agent.stream_events(
                "Use Python to print the sum of the first 10 positive integers, "
                "then give me the number."
            )
        ]

        stopped = events[-1]
        assert isinstance(stopped, StoppedEvent)
        assert stopped.stop_reason is StopReason.COMPLETED
        results = [e for e in events if isinstance(e, ExecutionResultEvent)]
        assert results, "the task requires at least one execution"
        assert any("55" in (e.output or "") for e in results)

    async def test_two_blocks_in_one_answer_run_in_order_across_turns(self, model):
        """The agent executes the first block and re-prompts; a response
        carrying several fences must not have a later block jump the queue."""
        agent = CaveAgent(model=model, runtime=IPythonRuntime(), max_steps=6)

        await agent.run(
            "In one single response, write TWO separate python code blocks: "
            "the first must be exactly print('FIRST'), the second exactly "
            "print('SECOND'). Do not combine them into one block."
        )

        outputs = _execution_outputs(agent)
        first_seen = next((i for i, out in enumerate(outputs) if "FIRST" in out), None)
        second_seen = next((i for i, out in enumerate(outputs) if "SECOND" in out), None)
        assert first_seen is not None, "the first block must execute"
        if second_seen is not None:
            assert first_seen <= second_seen


class TestLiveErrorFeedback:
    async def test_a_runtime_error_reaches_the_model_and_gets_fixed(self, model):
        """The error text is model-facing output; a real model must be able to
        act on it and correct course within the same run."""
        runtime = IPythonRuntime()
        agent = CaveAgent(model=model, runtime=runtime, max_steps=6)

        response = await agent.run(
            "Run exactly this line first: result = undefined_name + 1 . "
            "It will fail. Read the error, then fix it by using result = 41 + 1 "
            "and print(result)."
        )

        assert response.stop_reason is StopReason.COMPLETED
        outputs = _execution_outputs(agent)
        assert any("NameError" in out for out in outputs)
        assert any("42" in out for out in outputs)
