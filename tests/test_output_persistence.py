"""Oversize execution output is kept in the runtime, not discarded.

The old behaviour told the model "output too long, change your code and print
less" — throwing away the result of work that may be slow, expensive, or not
reproducible at all. Because the runtime *is* Python, the full text can simply
stay bound to a variable the model can slice and search.
"""

from cave_agent import CaveAgent, IPythonRuntime
from cave_agent._placeholders import (
    build_persist_marker,
    strip_persisted_preview,
)
from cave_agent.events import ExecutionResultEvent
from cave_agent.messages import ExecutionResultMessage

from .fakes import FakeModel

BIG = "```python\nprint('x' * 9000)\n```"


def _agent(**kwargs) -> CaveAgent:
    kwargs.setdefault("max_exec_output", 1000)
    kwargs.setdefault("exec_output_preview_chars", 200)
    runtime = kwargs.pop("runtime", None) or IPythonRuntime()
    return CaveAgent(model=FakeModel([BIG, "saw it"]), runtime=runtime, **kwargs)


class TestPersistence:
    async def test_full_output_reaches_the_runtime(self):
        runtime = IPythonRuntime()
        await _agent(runtime=runtime).run("q")

        stashed = await runtime.get_from_namespace("_output_1")
        assert len(stashed) == 9001

    async def test_model_sees_a_marker_naming_the_variable(self):
        agent = _agent()
        await agent.run("q")

        results = [m.content for m in agent.messages if isinstance(m, ExecutionResultMessage)]
        assert any("<persisted-output>" in c for c in results)
        assert any("_output_1" in c for c in results)

    async def test_marker_includes_a_preview(self):
        agent = _agent()
        await agent.run("q")

        marker = [
            m.content
            for m in agent.messages
            if isinstance(m, ExecutionResultMessage) and "<persisted-output>" in m.content
        ][0]
        assert "Preview (first" in marker

    async def test_preview_can_be_suppressed(self):
        agent = _agent(exec_output_preview_chars=0)
        await agent.run("q")

        marker = [
            m.content
            for m in agent.messages
            if isinstance(m, ExecutionResultMessage) and "<persisted-output>" in m.content
        ][0]
        assert "Preview (first" not in marker
        assert "_output_1" in marker

    async def test_event_reports_persistence_and_full_output(self):
        """A renderer should show what really happened, not the shaped marker."""
        agent = _agent()
        events = [e async for e in agent.stream_events("q")]

        result = [e for e in events if isinstance(e, ExecutionResultEvent)][0]
        assert result.persisted is True
        assert result.persisted_variable == "_output_1"
        assert len(result.output) == 9001

    async def test_small_output_is_inlined_untouched(self):
        agent = CaveAgent(
            model=FakeModel(["```python\nprint('small')\n```", "ok"]),
            runtime=IPythonRuntime(),
            max_exec_output=1000,
        )
        events = [e async for e in agent.stream_events("q")]

        result = [e for e in events if isinstance(e, ExecutionResultEvent)][0]
        assert result.persisted is False
        assert result.persisted_variable is None
        results = [m.content for m in agent.messages if isinstance(m, ExecutionResultMessage)]
        assert not any("<persisted-output>" in c for c in results)

    async def test_variable_prefix_is_configurable(self):
        runtime = IPythonRuntime()
        await _agent(runtime=runtime, persisted_output_prefix="_big").run("q")

        assert len(await runtime.get_from_namespace("_big_1")) == 9001

    async def test_model_can_slice_the_stashed_output(self):
        """The point of keeping it in the runtime: it stays usable."""
        runtime = IPythonRuntime()
        agent = CaveAgent(
            model=FakeModel([BIG, "```python\nprint(len(_output_1))\n```", "9001"]),
            runtime=runtime,
            max_exec_output=1000,
        )
        await agent.run("q")

        results = [m.content for m in agent.messages if isinstance(m, ExecutionResultMessage)]
        assert any("9001" in c for c in results)


class TestEachOutputKeepsItsOwnName:
    """A marker in history is a *pointer*, so no two may claim one name.

    Every oversize output used to be bound to the same variable. That does not
    "keep the newest" — it re-points every earlier marker at text it never
    described, so a model following turn 2's pointer reads turn 5's output and
    reasons about the wrong data with no way to tell.
    """

    def _two_big_outputs(self, runtime, **kwargs):
        return CaveAgent(
            model=FakeModel(
                [
                    "```python\nprint('a' * 9000)\n```",
                    "```python\nprint('b' * 8000)\n```",
                    "done",
                ]
            ),
            runtime=runtime,
            max_exec_output=1000,
            exec_output_preview_chars=200,
            **kwargs,
        )

    async def test_both_outputs_stay_independently_readable(self):
        runtime = IPythonRuntime()
        await self._two_big_outputs(runtime).run("q")

        first = await runtime.get_from_namespace("_output_1")
        second = await runtime.get_from_namespace("_output_2")
        assert first == "a" * 9000 + "\n"
        assert second == "b" * 8000 + "\n"

    async def test_each_marker_names_the_output_it_describes(self):
        agent = self._two_big_outputs(IPythonRuntime())
        await agent.run("q")

        markers = [
            m.content
            for m in agent.messages
            if isinstance(m, ExecutionResultMessage) and "<persisted-output>" in m.content
        ]
        assert len(markers) == 2
        assert "`_output_1`" in markers[0] and "`_output_2`" not in markers[0]
        assert "`_output_2`" in markers[1]

    async def test_events_report_the_distinct_names(self):
        agent = self._two_big_outputs(IPythonRuntime())
        events = [e async for e in agent.stream_events("q")]

        persisted = [e for e in events if isinstance(e, ExecutionResultEvent) and e.persisted]
        assert [e.persisted_variable for e in persisted] == ["_output_1", "_output_2"]

    async def test_resumed_history_does_not_reclaim_a_live_name(self):
        """A pointer handed out in an earlier run is still one the model trusts."""
        runtime = IPythonRuntime()
        first = self._two_big_outputs(runtime)
        await first.run("q")

        resumed = CaveAgent(
            model=FakeModel(["```python\nprint('c' * 7000)\n```", "done"]),
            runtime=runtime,
            max_exec_output=1000,
            messages=first.messages,
        )
        await resumed.run("q2")

        assert await runtime.get_from_namespace("_output_1") == "a" * 9000 + "\n"
        assert await runtime.get_from_namespace("_output_3") == "c" * 7000 + "\n"


class TestMarkerFormat:
    def test_roundtrip_strip_is_idempotent(self):
        marker = build_persist_marker("y" * 5000, 100, "_out")
        once = strip_persisted_preview(marker)
        assert strip_persisted_preview(once) == once

    def test_strip_keeps_the_variable_name(self):
        marker = build_persist_marker("y" * 5000, 100, "_out")
        assert "_out" in strip_persisted_preview(marker)

    def test_strip_returns_none_for_plain_text(self):
        assert strip_persisted_preview("just output") is None

    def test_no_preview_marker_is_already_reduced(self):
        marker = build_persist_marker("y" * 5000, 0, "_out")
        assert strip_persisted_preview(marker) == marker

    def test_preview_prefers_newline_boundary(self):
        content = "line one\n" + "z" * 500
        marker = build_persist_marker(content, 300, "_out")
        assert "line one" in marker

    def test_marker_states_the_total_size(self):
        assert "5000 chars" in build_persist_marker("y" * 5000, 100, "_out")


class TestMarkerSurvivesCompaction:
    """The pointer must survive microcompaction, in the form the agent writes.

    Compaction sees the marker already wrapped in EXECUTION_OUTPUT_PROMPT, so
    the content starts "\\n<execution_output>\\n<persisted-output>…". Matching on
    `startswith` silently missed that and fell through to the generic
    placeholder, destroying the only pointer to the stashed data — the exact
    failure this whole mechanism exists to prevent.
    """

    async def _history_with_persisted_result(self):
        agent = _agent()
        await agent.run("q")
        wrapped = [
            m.content
            for m in agent.messages
            if isinstance(m, ExecutionResultMessage) and "<persisted-output>" in m.content
        ][0]
        return wrapped

    async def test_agent_writes_the_marker_wrapped(self):
        wrapped = await self._history_with_persisted_result()
        assert not wrapped.startswith("<persisted-output>")
        assert "<execution_output>" in wrapped

    async def test_microcompact_keeps_the_variable_name(self):
        from cave_agent.compaction.compactor import _microcompact
        from cave_agent.messages import CodeExecutionMessage

        wrapped = await self._history_with_persisted_result()
        msgs = []
        for i in range(10):
            msgs.append(CodeExecutionMessage(f"c{i}"))
            msgs.append(ExecutionResultMessage(wrapped if i == 0 else "R" * 500))

        result = _microcompact(msgs)

        assert "_output_1" in result[1].content, result[1].content[:120]
        assert "Preview (first" not in result[1].content

    async def test_microcompact_is_idempotent_on_wrapped_marker(self):
        from cave_agent.compaction.compactor import _microcompact
        from cave_agent.messages import CodeExecutionMessage

        wrapped = await self._history_with_persisted_result()

        def build(first):
            msgs = []
            for i in range(10):
                msgs.append(CodeExecutionMessage(f"c{i}"))
                msgs.append(ExecutionResultMessage(first if i == 0 else "R" * 500))
            return msgs

        once = _microcompact(build(wrapped))
        twice = _microcompact(build(once[1].content))
        assert twice[1].content == once[1].content

    async def test_preview_cannot_close_its_outer_marker(self):
        from cave_agent.compaction.compactor import _microcompact
        from cave_agent.messages import CodeExecutionMessage

        output = "before\n</persisted-output>\nSECRET-" + "x" * 120
        agent = CaveAgent(
            model=FakeModel([f"```python\nprint({output!r})\n```", "done"]),
            runtime=IPythonRuntime(),
            max_exec_output=20,
            exec_output_preview_chars=500,
        )
        await agent.run("q")
        wrapped = [
            message.content
            for message in agent.messages
            if (
                isinstance(message, ExecutionResultMessage)
                and "<persisted-output>" in message.content
            )
        ][0]

        messages = []
        for index in range(10):
            messages.append(CodeExecutionMessage(f"c{index}"))
            messages.append(ExecutionResultMessage(wrapped if index == 0 else "R" * 500))

        compacted = _microcompact(messages)
        reduced = compacted[1].content

        assert "_output_1" in reduced
        assert "SECRET-" not in reduced
        assert reduced.count("</persisted-output>") == 1
        assert strip_persisted_preview(reduced) == reduced

    def test_strip_preserves_surrounding_wrapper(self):
        wrapped = f"\n<execution_output>\n{build_persist_marker('z' * 9000, 200, '_out')}\n</execution_output>\n"
        reduced = strip_persisted_preview(wrapped)
        assert reduced.startswith("\n<execution_output>")
        assert reduced.rstrip().endswith("</execution_output>")
        assert "Preview (first" not in reduced
        assert "_out" in reduced


class TestAllocationBelongsToTheRuntime:
    """The runtime picks the name, because only it knows what is bound.

    The agent used to count its own outputs and build `_output_N` itself. That
    counter cannot see a `Variable("_output_1")` the caller registered, so an
    oversize execution result silently replaced the user's data with a string
    — and two agents sharing a runtime would have collided the same way.
    """

    async def test_a_registered_variable_is_not_overwritten(self):
        from cave_agent.runtime import Variable

        runtime = IPythonRuntime(
            variables=[Variable("_output_1", {"important": [1, 2, 3]}, "user data")]
        )
        await _agent(runtime=runtime).run("q")

        assert await runtime.retrieve("_output_1") == {"important": [1, 2, 3]}

    async def test_the_output_still_lands_somewhere_reachable(self):
        from cave_agent.runtime import Variable

        runtime = IPythonRuntime(
            variables=[Variable("_output_1", {"important": [1, 2, 3]}, "user data")]
        )
        agent = _agent(runtime=runtime)
        events = [e async for e in agent.stream_events("q")]

        result = [e for e in events if isinstance(e, ExecutionResultEvent)][0]
        assert result.persisted_variable == "_output_2"
        assert len(await runtime.get_from_namespace("_output_2")) == 9001

    async def test_two_agents_sharing_a_runtime_do_not_collide(self):
        runtime = IPythonRuntime()

        first = _agent(runtime=runtime)
        second = _agent(runtime=runtime)
        await first.run("q")
        await second.run("q")

        assert await runtime.get_from_namespace("_output_1") == "x" * 9000 + "\n"
        assert await runtime.get_from_namespace("_output_2") == "x" * 9000 + "\n"

    async def test_concurrent_allocations_never_collide(self):
        """The property behind "check and bind are one step".

        Asserted behaviourally rather than by inspecting for `await`: the
        kernel backend *must* await (the names live in another process), and
        its atomicity comes from doing both halves inside a single cell.
        """
        import asyncio

        runtime = IPythonRuntime()

        names = await asyncio.gather(
            *[runtime.bind_unique("_output", f"value {i}") for i in range(25)]
        )

        assert len(set(names)) == 25
        for index, name in enumerate(names):
            assert await runtime.get_from_namespace(name) == f"value {index}"

    async def test_a_name_the_generated_code_created_is_not_taken(self):
        """The registry cannot see this; the namespace can."""
        runtime = IPythonRuntime()
        await runtime.execute("_output_1 = {'mine': True}")

        name = await runtime.bind_unique("_output", "the output")

        assert name == "_output_2"
        assert await runtime.get_from_namespace("_output_1") == {"mine": True}


class TestPrefixMustBeAnIdentifier:
    """The prefix becomes a Python name the model is told to slice.

    Accepted blind, `persisted_output_prefix="bad-name"` promised the model a
    variable `bad-name_1` — read as a subtraction in-process, and a
    SyntaxError in the kernel reported as a runtime failure, both a long way
    from the constructor argument that caused them.
    """

    def test_a_non_identifier_prefix_is_rejected_at_construction(self):
        import pytest

        with pytest.raises(ValueError, match="identifier"):
            CaveAgent(
                model=FakeModel([]),
                runtime=IPythonRuntime(),
                persisted_output_prefix="bad-name",
            )

    def test_the_error_names_the_variable_it_would_have_produced(self):
        import pytest

        with pytest.raises(ValueError, match=r"bad-name_1"):
            CaveAgent(
                model=FakeModel([]),
                runtime=IPythonRuntime(),
                persisted_output_prefix="bad-name",
            )

    def test_an_ordinary_prefix_is_accepted(self):
        agent = CaveAgent(
            model=FakeModel([]),
            runtime=IPythonRuntime(),
            persisted_output_prefix="_out",
        )
        assert agent.persisted_output_prefix == "_out"


class TestGeneratedCodeOwnsItsNames:
    """A name the model created is as real as one that was injected.

    Allocation used to consult a registry on the agent's side, which sees only
    injected names — so an oversize result overwrote a variable the model had
    just built. Both backends now choose the name where the names live.
    """

    MODEL_MAKES_ITS_OWN = "```python\n_output_1 = {'important': [1, 2, 3]}\nprint('x' * 9000)\n```"

    async def test_in_process_does_not_clobber_it(self):
        runtime = IPythonRuntime()
        await CaveAgent(
            model=FakeModel([self.MODEL_MAKES_ITS_OWN, "done"]),
            runtime=runtime,
            max_exec_output=500,
        ).run("q")

        assert await runtime.get_from_namespace("_output_1") == {"important": [1, 2, 3]}
        assert len(await runtime.get_from_namespace("_output_2")) == 9001

    async def test_the_kernel_does_not_clobber_it(self):
        from cave_agent.runtime import IPyKernelRuntime

        runtime = IPyKernelRuntime()
        try:
            await CaveAgent(
                model=FakeModel([self.MODEL_MAKES_ITS_OWN, "done"]),
                runtime=runtime,
                max_exec_output=500,
            ).run("q")

            assert await runtime.get_from_namespace("_output_1") == {"important": [1, 2, 3]}
            assert len(await runtime.get_from_namespace("_output_2")) == 9001
        finally:
            await runtime.stop()


class TestAllocationLeavesNoTraceInTheNamespace:
    """The allocator must not disturb the namespace it is allocating in.

    An earlier kernel implementation did its bookkeeping in scratch globals and
    then `del`'d them, so a model that happened to use those names had its
    variables overwritten and then removed — the same defect the allocator
    exists to prevent, one level down.
    """

    MODEL_USES_SCRATCH_NAMES = (
        "```python\n_cave_i = 'mine'\n_cave_n = 'also mine'\nprint('x' * 9000)\n```"
    )

    async def test_in_process_leaves_them_alone(self):
        runtime = IPythonRuntime()
        await CaveAgent(
            model=FakeModel([self.MODEL_USES_SCRATCH_NAMES, "done"]),
            runtime=runtime,
            max_exec_output=500,
        ).run("q")

        assert await runtime.get_from_namespace("_cave_i") == "mine"
        assert await runtime.get_from_namespace("_cave_n") == "also mine"

    async def test_the_kernel_leaves_them_alone(self):
        from cave_agent.runtime import IPyKernelRuntime

        runtime = IPyKernelRuntime()
        try:
            await CaveAgent(
                model=FakeModel([self.MODEL_USES_SCRATCH_NAMES, "done"]),
                runtime=runtime,
                max_exec_output=500,
            ).run("q")

            assert await runtime.get_from_namespace("_cave_i") == "mine"
            assert await runtime.get_from_namespace("_cave_n") == "also mine"
        finally:
            await runtime.stop()

    async def test_no_helper_import_or_scratch_name_survives(self):
        """No helper survives under a Python identifier the model can shadow."""
        from cave_agent.runtime import IPyKernelRuntime

        runtime = IPyKernelRuntime()
        try:
            await runtime.execute("pass")
            await runtime.bind_unique("_output", "big")
            await runtime.get_from_namespace("_output_1")

            bound = await runtime._executor._names_starting_with("")
            assert not {"dill", "base64", "_dill", "_b64", "_d", "_b"} & bound
        finally:
            await runtime.stop()


class TestUnicodePrefixesResumeCorrectly:
    """A prefix the agent accepts is a prefix the history scanner must read.

    Validation allows any Python identifier, but the scanner's pattern was a
    fixed ASCII class. A prefix like `输出` produced markers that matched
    nothing, so a resumed session scanned zero, restarted at `输出_1`, and
    re-pointed the first session's marker at the second session's output.
    """

    async def _run_with_prefix(self, prefix, payload, messages=None):
        runtime = IPythonRuntime()
        agent = CaveAgent(
            model=FakeModel([f"```python\nprint('{payload}' * 9000)\n```", "done"]),
            runtime=runtime,
            max_exec_output=500,
            persisted_output_prefix=prefix,
            messages=messages,
        )
        await agent.run("q")
        return agent, runtime

    async def test_the_scanner_reads_a_unicode_prefix(self):
        from cave_agent._placeholders import highest_persisted_index

        agent, _ = await self._run_with_prefix("输出", "A")

        found = highest_persisted_index((m.content for m in agent.messages), "输出")
        assert found == 1

    async def test_a_resumed_session_does_not_reuse_the_name(self):
        first, _ = await self._run_with_prefix("输出", "A")
        second, runtime = await self._run_with_prefix("输出", "B", messages=first.messages)

        markers = [m.content for m in second.messages if "<persisted-output>" in m.content]
        assert sum("`输出_1`" in m for m in markers) == 1
        assert any("`输出_2`" in m for m in markers)
        assert (await runtime.get_from_namespace("输出_2")).startswith("B")

    def test_user_text_that_only_looks_like_a_pointer_is_not_a_marker(self):
        from cave_agent._placeholders import highest_persisted_index

        assert highest_persisted_index(["please inspect `_output_99`"], "_output") == 0

    def test_an_impossible_marker_index_cannot_crash_history_loading(self):
        from cave_agent._placeholders import highest_persisted_index

        marker = build_persist_marker("x", 0, f"_output_{'9' * 4301}")

        assert highest_persisted_index([marker], "_output") == 0
