"""Tests for context compaction — the Compactor's three tiers, the circuit
breaker, incremental re-compaction, and token estimation."""

import pytest

from cave_agent._placeholders import build_persist_marker, strip_persisted_preview
from cave_agent.compaction import CompactionState, Compactor
from cave_agent.compaction.compactor import (
    _KEEP_RECENT_EXECUTION_RESULTS,
    _drop_summary_pair,
    _microcompact,
    migrate_legacy_summaries,
)
from cave_agent.compaction.prompts import (
    COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER,
    COMPACTION_SUMMARY_MARKER,
    COMPACTION_SUMMARY_USER_TEMPLATE,
    MICROCOMPACT_PLACEHOLDER,
    extract_summary,
    format_transcript,
    parse_legacy_summary,
)
from cave_agent.compaction.state import COOLDOWN_ATTEMPTS, MAX_CONSECUTIVE_FAILURES
from cave_agent.compaction.tokens import (
    COMPACT_BUFFER_TOKENS,
    DEFAULT_TOKENS_PER_CHAR,
    TokenAnchor,
    compact_threshold,
    estimate_tokens,
)
from cave_agent.messages import (
    AssistantMessage,
    CodeExecutionMessage,
    ExecutionResultMessage,
    Message,
    MessageRole,
    SummaryAcknowledgementMessage,
    SummaryMessage,
    SystemMessage,
    UserMessage,
)
from cave_agent.models import Model, ModelResponse, PromptTooLongError, TokenUsage

_FALLBACK_RESERVE = 16_000


class SummaryModel(Model):
    """Returns a fixed summary and records the prompts it was given."""

    def __init__(self, summary: str = "Summary of the conversation.", max_output_tokens=None):
        self._summary = summary
        self.max_output_tokens = max_output_tokens
        self.prompts: list[str] = []

    async def _complete(self, messages):
        self.prompts.append(messages[-1]["content"])
        return ModelResponse(
            content=f"<analysis>thinking</analysis><summary>{self._summary}</summary>",
            finish_reason="stop",
        )

    def stream(self, messages):
        raise NotImplementedError


class BrokenModel(Model):
    """Always raises."""

    def __init__(self):
        self.call_count = 0

    async def _complete(self, messages):
        self.call_count += 1
        raise ValueError("model unavailable")

    def stream(self, messages):
        raise NotImplementedError


class CapacitySummaryModel(Model):
    """Rejects oversized requests like a provider, and carries one fact forward."""

    max_output_tokens = 0

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.successful_requests = 0

    async def _complete(self, messages):
        if sum(len(message["content"]) for message in messages) > self.capacity:
            raise PromptTooLongError("summary prompt exceeds context window")
        self.successful_requests += 1
        content = messages[-1]["content"]
        fact = "EARLY_SENTINEL" if "EARLY_SENTINEL" in content else "no sentinel yet"
        return ModelResponse(
            content=f"<summary>{fact}; batch {self.successful_requests}</summary>",
            finish_reason="stop",
        )

    def stream(self, messages):
        raise NotImplementedError


def _make_messages(n_results: int) -> list[Message]:
    """A conversation with *n_results* code/result pairs."""
    msgs: list[Message] = []
    for i in range(n_results):
        msgs.append(CodeExecutionMessage(f"```python\nprint({i})\n```"))
        msgs.append(ExecutionResultMessage(f"Output {i}: {'x' * 200}"))
    return msgs


class TestEstimateTokens:
    def test_simple(self):
        assert estimate_tokens([UserMessage("hello world")]) == int(
            len("hello world") * DEFAULT_TOKENS_PER_CHAR
        )

    def test_anchor_reports_the_measured_count(self):
        msgs = [UserMessage("hello")]
        anchor = TokenAnchor(999, estimate_tokens(msgs))
        assert estimate_tokens(msgs, anchor) == 999

    def test_anchor_zero_falls_back(self):
        assert estimate_tokens([UserMessage("hello")], TokenAnchor(0, 0)) == int(
            len("hello") * DEFAULT_TOKENS_PER_CHAR
        )

    def test_messages_added_after_the_measurement_are_counted(self):
        """The defect this replaced: `max(heuristic, api)` returned the API
        count unchanged, so every turn appended after it — including a large
        execution result — was invisible until the bare heuristic overtook it,
        which for a real conversation is never."""
        measured = [UserMessage("x" * 400)]  # heuristic 100
        anchor = TokenAnchor(16_990, estimate_tokens(measured))
        grown = measured + [UserMessage("y" * 400), UserMessage("z" * 400)]

        assert estimate_tokens(grown, anchor) == 17_190

    def test_a_shrunken_history_invalidates_its_own_anchor(self):
        """Compaction rewrites the list the count was taken against, so the
        anchor stops describing anything. Detected here rather than left for
        every rewrite site to remember."""
        measured = [UserMessage("x" * 4000)]
        anchor = TokenAnchor(50_000, estimate_tokens(measured))
        compacted = [UserMessage("x" * 40)]

        assert estimate_tokens(compacted, anchor) == 10

    def test_custom_estimator(self):
        assert estimate_tokens([UserMessage("hello")], token_estimator=lambda t: len(t)) == 5


class TestCompactThreshold:
    def test_formula(self):
        assert compact_threshold(128_000, 16_000) == 128_000 - 16_000 - COMPACT_BUFFER_TOKENS

    def test_small_window_fallback(self):
        # Reserves exceed the window → fall back to 50%.
        assert compact_threshold(20_000, 16_000) == 10_000


class TestCompactorThreshold:
    def test_uses_model_declaration(self):
        compactor = Compactor(SummaryModel(max_output_tokens=8_000), context_window=100_000)
        assert compactor.threshold() == 100_000 - 8_000 - COMPACT_BUFFER_TOKENS

    def test_explicit_reserve_overrides_model(self):
        compactor = Compactor(
            SummaryModel(max_output_tokens=8_000),
            context_window=100_000,
            output_reserve=30_000,
        )
        assert compactor.threshold() == 100_000 - 30_000 - COMPACT_BUFFER_TOKENS

    def test_falls_back_when_model_silent(self):
        compactor = Compactor(SummaryModel(), context_window=100_000)
        assert compactor.threshold() == 100_000 - _FALLBACK_RESERVE - COMPACT_BUFFER_TOKENS


class TestMicrocompact:
    def test_no_change_when_few_results(self):
        msgs = _make_messages(3)
        assert _microcompact(msgs) is msgs

    def test_clears_old_results(self):
        result = _microcompact(_make_messages(12))
        kept = [
            m
            for m in result
            if m.role == MessageRole.EXECUTION_RESULT and m.content != MICROCOMPACT_PLACEHOLDER
        ]
        assert len(kept) == _KEEP_RECENT_EXECUTION_RESULTS

    def test_preserves_message_count(self):
        msgs = _make_messages(12)
        assert len(_microcompact(msgs)) == len(msgs)

    def test_preserves_code_messages(self):
        msgs = _make_messages(12)
        result = _microcompact(msgs)
        assert sum(1 for m in result if m.role == MessageRole.CODE_EXECUTION) == 12

    def test_already_cleared_results_are_not_recounted(self):
        """A second pass must not clear the results the first pass kept."""
        once = _microcompact(_make_messages(12))
        twice = _microcompact(once)
        assert twice is once

    def test_persisted_marker_keeps_declaration(self):
        """The declaration names the variable holding the full output — clearing
        it to the generic placeholder would strand unreachable data."""
        marker = build_persist_marker("Z" * 9000, 200, "_last_output")
        msgs: list[Message] = []
        for i in range(10):
            msgs.append(CodeExecutionMessage(f"c{i}"))
            msgs.append(ExecutionResultMessage(marker if i == 0 else "R" * 500))

        result = _microcompact(msgs)

        reduced = result[1].content
        assert "_last_output" in reduced
        assert "Preview (first" not in reduced
        assert reduced == strip_persisted_preview(marker)


class TestSummarize:
    async def test_reduces_messages(self):
        msgs = _make_messages(20)
        result = await Compactor(SummaryModel()).summarize(msgs)
        assert len(result) < len(msgs)

    async def test_summary_present(self):
        result = await Compactor(SummaryModel()).summarize(_make_messages(20))
        assert any(isinstance(m, SummaryMessage) for m in result)

    async def test_summarizer_receives_complete_messages(self):
        """Compaction must not delete a message suffix before the model sees it."""
        marker = "PRESERVE_THIS_PENDING_REQUIREMENT"
        model = SummaryModel()
        messages: list[Message] = [UserMessage("x" * 2500 + marker)]
        messages.extend(_make_messages(20))

        await Compactor(model).summarize(messages)

        assert marker in model.prompts[0]

    async def test_recent_messages_preserved(self):
        msgs = _make_messages(20)
        result = await Compactor(SummaryModel()).summarize(msgs)
        assert any(m.content == msgs[-1].content for m in result)

    async def test_no_change_when_too_few(self):
        msgs = _make_messages(1)
        assert await Compactor(SummaryModel()).summarize(msgs) is msgs

    async def test_system_message_preserved_by_identity(self):
        sys_msg = SystemMessage("system prompt")
        msgs = [sys_msg] + _make_messages(20)
        result = await Compactor(SummaryModel()).summarize(msgs)
        assert result[0] is sys_msg

    async def test_no_ack_before_assistant_tail(self):
        """Two assistant turns back-to-back break strict role alternation."""
        msgs = [UserMessage(f"m{i}") for i in range(12)] + [
            CodeExecutionMessage("code"),
            ExecutionResultMessage("out"),
        ]
        result = await Compactor(SummaryModel()).summarize(msgs)
        roles = [m.wire_role for m in result]
        assert not any(
            first == second == "assistant" for first, second in zip(roles, roles[1:], strict=False)
        )


class TestAcknowledgementProvenance:
    """The ack is typed too, and only the typed one is ever removed.

    Its text is a plausible thing for a model to actually say. Matching on it
    deleted a genuine assistant turn that said it, and let a user-typed marker
    be "corroborated" by a reply the user could steer into existing.
    """

    def test_a_genuine_assistant_turn_saying_the_same_thing_survives(self):
        genuine = AssistantMessage(COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER)
        summary = SummaryMessage(
            COMPACTION_SUMMARY_USER_TEMPLATE.format(summary="OLD"),
            "OLD",
        )

        kept = _drop_summary_pair([summary, genuine, *_make_messages(4)], 0)

        assert genuine in kept

    def test_the_synthetic_ack_is_dropped(self):
        summary = SummaryMessage(
            COMPACTION_SUMMARY_USER_TEMPLATE.format(summary="OLD"),
            "OLD",
        )
        ack = SummaryAcknowledgementMessage(COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER)

        kept = _drop_summary_pair([summary, ack, *_make_messages(4)], 0)

        assert ack not in kept

    async def test_compaction_emits_the_typed_ack(self):
        # A tail starting with a user turn, so the ack is not suppressed for
        # role alternation (see _build_summary_messages).
        history = [UserMessage(f"m{i}") for i in range(24)]

        result = await Compactor(SummaryModel()).summarize(history)

        assert any(isinstance(m, SummaryAcknowledgementMessage) for m in result)


class TestSummaryIdentification:
    """A summary is identified by its type, never by its text.

    Two attempts to make the text self-identifying — a `[Previous conversation
    summary]` prefix, then a `<conversation-summary>` envelope — both fell to
    the same thing: the summary is delivered into a conversation whose other
    participant also writes text, so any syntax marking it can be reproduced in
    an ordinary request. A user message read as a summary does not just get
    mislabelled: the real turns are dropped as "already covered" and its text
    is carried forward in their place.
    """

    def test_generated_summaries_carry_their_body(self):
        message = SummaryMessage(
            COMPACTION_SUMMARY_USER_TEMPLATE.format(summary="THE FACTS"),
            "THE FACTS",
        )
        assert message.body == "THE FACTS"
        assert message.wire_role == "user"

    async def test_a_user_typed_envelope_is_not_a_summary(self):
        """The exact bytes the agent writes, typed by a user."""
        model = SummaryModel("FRESH")
        victim = UserMessage(
            COMPACTION_SUMMARY_USER_TEMPLATE.format(summary="THIS IS AN ORDINARY REQUEST")
        )
        msgs = _make_messages(4) + [victim, AssistantMessage("sure")] + _make_messages(20)

        await Compactor(model).summarize(msgs)

        assert "<existing_summary>" not in model.prompts[0]
        assert "THIS IS AN ORDINARY REQUEST" in model.prompts[0]

    async def test_a_generated_summary_is_recognized(self):
        model = SummaryModel("UPDATED")
        summary = SummaryMessage(
            COMPACTION_SUMMARY_USER_TEMPLATE.format(summary="REAL FACTS"),
            "REAL FACTS",
        )
        msgs = [summary, SummaryAcknowledgementMessage(COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER)]
        msgs += _make_messages(20)

        await Compactor(model).summarize(msgs)

        assert "<existing_summary>" in model.prompts[0]
        assert "REAL FACTS" in model.prompts[0]

    async def test_compacting_twice_reuses_the_first_summary(self):
        """The round trip that matters: what compaction writes, it must recognize."""
        model = SummaryModel("FIRST")
        compactor = Compactor(model)

        once = await compactor.summarize(_make_messages(20))
        assert any(isinstance(m, SummaryMessage) for m in once)

        model._summary = "SECOND"
        await compactor.summarize(once + _make_messages(20))

        assert "<existing_summary>" in model.prompts[-1]
        assert "FIRST" in model.prompts[-1]


class TestLegacySummaryCompatibility:
    """Histories written by <=0.8.0 have no type to carry, only text.

    Text from a live conversation cannot establish provenance — both
    participants write text — so conversion is an explicit call the caller
    makes about a history it knows the origin of, never a guess made on every
    compaction.
    """

    def test_the_legacy_shape_is_recognized(self):
        assert parse_legacy_summary(f"{COMPACTION_SUMMARY_MARKER}\nOLD FACTS") == "OLD FACTS"

    def test_a_quoted_marker_is_not_the_legacy_shape(self):
        assert (
            parse_legacy_summary(f'why did you write "{COMPACTION_SUMMARY_MARKER}" in my history?')
            is None
        )

    def test_a_plain_message_is_not_the_legacy_shape(self):
        assert parse_legacy_summary("just a question") is None

    def test_migration_needs_the_acknowledgement_too(self):
        """The text alone is what a user can type; the pair is not."""
        unpaired = [
            UserMessage(f"{COMPACTION_SUMMARY_MARKER}\nORDINARY REQUEST"),
            AssistantMessage("sure"),
        ]

        assert migrate_legacy_summaries(unpaired) == unpaired

    def _legacy_history(self):
        return [
            UserMessage(f"{COMPACTION_SUMMARY_MARKER}\nOLD FACTS FROM 0.8.0"),
            AssistantMessage(COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER),
        ] + _make_messages(20)

    async def test_migrated_legacy_history_compacts_incrementally(self):
        """End to end against exactly what <=0.8.0's builder wrote."""
        model = SummaryModel("UPDATED")

        await Compactor(model).summarize(migrate_legacy_summaries(self._legacy_history()))

        assert "<existing_summary>" in model.prompts[0]
        assert "OLD FACTS FROM 0.8.0" in model.prompts[0]

    def test_migration_types_both_halves_of_the_pair(self):
        migrated = migrate_legacy_summaries(self._legacy_history())

        assert isinstance(migrated[0], SummaryMessage)
        assert migrated[0].body == "OLD FACTS FROM 0.8.0"
        assert isinstance(migrated[1], SummaryAcknowledgementMessage)

    async def test_an_unmigrated_legacy_history_is_only_re_summarized(self):
        """The harmless direction: fidelity lost, nothing corrupted."""
        model = SummaryModel("FRESH")

        await Compactor(model).summarize(self._legacy_history())

        assert "<existing_summary>" not in model.prompts[0]
        assert "OLD FACTS FROM 0.8.0" in model.prompts[0]

    def test_migration_leaves_an_ordinary_conversation_alone(self):
        plain = _make_messages(6)
        assert migrate_legacy_summaries(plain) == plain

    async def test_legacy_text_is_never_trusted_live(self):
        """Even *with* a matching acknowledgement the model happened to produce.

        The ack text is an ordinary thing for a model to say, so a user who
        opens with the old marker and steers the reply can reproduce the whole
        shape. Live compaction does not look at either.
        """
        model = SummaryModel("FRESH")
        victim = UserMessage(f"{COMPACTION_SUMMARY_MARKER}\nTHIS IS AN ORDINARY REQUEST")
        msgs = (
            _make_messages(4)
            + [
                victim,
                AssistantMessage(COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER),
            ]
            + _make_messages(20)
        )

        await Compactor(model).summarize(msgs)

        assert "<existing_summary>" not in model.prompts[0]
        assert "THIS IS AN ORDINARY REQUEST" in model.prompts[0]


class TestIncrementalRecompaction:
    """A conversation compacted twice must not summarize its own summary."""

    def _history_with_summary(self) -> list[Message]:
        return [
            SummaryMessage(
                COMPACTION_SUMMARY_USER_TEMPLATE.format(summary="OLD FACTS ABOUT THE TASK"),
                "OLD FACTS ABOUT THE TASK",
            ),
            SummaryAcknowledgementMessage(COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER),
            UserMessage("new question"),
            CodeExecutionMessage("code"),
            ExecutionResultMessage("result"),
            UserMessage("another"),
            AssistantMessage("reply"),
            UserMessage("more"),
        ]

    async def test_uses_update_prompt(self):
        model = SummaryModel("UPDATED")
        await Compactor(model).summarize(self._history_with_summary())
        assert "<existing_summary>" in model.prompts[0]

    async def test_prior_summary_carried_verbatim(self):
        model = SummaryModel("UPDATED")
        await Compactor(model).summarize(self._history_with_summary())
        assert "OLD FACTS ABOUT THE TASK" in model.prompts[0]

    async def test_prior_summary_absent_from_transcript(self):
        """The old summary is a document to update, not a turn to re-compress."""
        model = SummaryModel("UPDATED")
        await Compactor(model).summarize(self._history_with_summary())
        transcript = model.prompts[0].split("<new_turns>", 1)[1]
        assert "OLD FACTS" not in transcript

    async def test_new_turns_are_not_truncated(self):
        marker = "PRESERVE_THIS_NEW_REQUIREMENT"
        history = self._history_with_summary()
        history[2] = UserMessage("x" * 2500 + marker)
        model = SummaryModel("UPDATED")

        await Compactor(model).summarize(history)

        assert marker in model.prompts[0]

    async def test_ack_dropped_from_transcript(self):
        model = SummaryModel("UPDATED")
        await Compactor(model).summarize(self._history_with_summary())
        transcript = model.prompts[0].split("<new_turns>", 1)[1]
        assert COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER not in transcript

    async def test_no_llm_call_when_nothing_new(self):
        """The prior summary already IS the summary of this region."""
        model = SummaryModel()
        msgs = [
            SummaryMessage(
                COMPACTION_SUMMARY_USER_TEMPLATE.format(summary="EVERYTHING SO FAR"),
                "EVERYTHING SO FAR",
            ),
            SummaryAcknowledgementMessage(COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER),
            UserMessage("a"),
            AssistantMessage("b"),
            UserMessage("c"),
            AssistantMessage("d"),
        ]
        result = await Compactor(model).summarize(msgs)
        assert model.prompts == []
        assert any("EVERYTHING SO FAR" in m.content for m in result)

    async def test_fresh_history_uses_plain_prompt(self):
        model = SummaryModel()
        await Compactor(model).summarize(_make_messages(20))
        assert "<existing_summary>" not in model.prompts[0]


class TestCircuitBreaker:
    async def _trip(self, compactor: Compactor) -> None:
        for _ in range(MAX_CONSECUTIVE_FAILURES):
            await compactor.summarize(_make_messages(20))

    async def test_failure_increments_count(self):
        compactor = Compactor(BrokenModel())
        await compactor.summarize(_make_messages(20))
        assert compactor.state.consecutive_failures == 1

    async def test_opens_after_max_failures(self):
        compactor = Compactor(BrokenModel())
        await self._trip(compactor)
        assert compactor.state.is_open

    async def test_open_breaker_skips_model(self):
        model = BrokenModel()
        compactor = Compactor(model)
        await self._trip(compactor)
        calls_before = model.call_count

        await compactor.summarize(_make_messages(20))

        assert model.call_count == calls_before

    async def test_closes_after_cooldown(self):
        """A transient provider failure must not disable summarization forever.

        Counted in attempts, and driven through the same public entry point the
        agent uses — an earlier design took the step number as an argument that
        the one production call site never passed, so the breaker latched open
        for the whole session and no unit test noticed.
        """
        model = BrokenModel()
        compactor = Compactor(model)
        await self._trip(compactor)
        calls_after_trip = model.call_count

        for _ in range(COOLDOWN_ATTEMPTS):
            await compactor.summarize(_make_messages(20))
        assert model.call_count == calls_after_trip, "cooldown should skip the model"

        await compactor.summarize(_make_messages(20))

        assert model.call_count > calls_after_trip, "breaker must reopen for a retry"

    async def test_agent_loop_recovers_the_breaker(self):
        """Guards the wiring, not just the state machine."""
        from cave_agent import CaveAgent, IPythonRuntime

        from .fakes import FakeModel

        model = BrokenModel()
        compactor = Compactor(model, context_window=200_000)
        compactor.output_reserve = 0
        compactor.context_window = COMPACT_BUFFER_TOKENS + 1  # compact every step

        agent = CaveAgent(
            FakeModel(["```python\nprint(1)\n```"] * 30),
            runtime=IPythonRuntime(),
            compactor=compactor,
            max_steps=30,
        )
        await agent.run("go")

        assert model.call_count > MAX_CONSECUTIVE_FAILURES, (
            "the breaker must let the model be retried during a long run"
        )

    async def test_success_resets_count(self):
        compactor = Compactor(SummaryModel())
        compactor.state = CompactionState(consecutive_failures=2)
        await compactor.summarize(_make_messages(20))
        assert compactor.state.consecutive_failures == 0

    async def test_failure_preserves_unsummarized_history(self):
        msgs = _make_messages(20)
        result = await Compactor(BrokenModel()).summarize(msgs)
        assert result is msgs

    async def test_empty_summary_counts_as_failure(self):
        class EmptyModel(SummaryModel):
            async def _complete(self, messages):
                return ModelResponse(content="")

        compactor = Compactor(EmptyModel())
        await compactor.summarize(_make_messages(20))
        assert compactor.state.consecutive_failures == 1


class TestSummaryInputCapacity:
    async def test_oversized_region_is_folded_without_losing_early_fact(self):
        model = CapacitySummaryModel(capacity=4_000)
        compactor = Compactor(
            model,
            context_window=4_000,
            output_reserve=0,
            token_estimator=len,
        )
        messages = [
            UserMessage("EARLY_SENTINEL " + "x" * 12_000),
            *[UserMessage(f"turn-{index}") for index in range(19)],
        ]

        result = await compactor.summarize(messages)

        summaries = [message for message in result if isinstance(message, SummaryMessage)]
        assert summaries and "EARLY_SENTINEL" in summaries[0].body
        assert model.successful_requests > 1

    async def test_agent_stored_query_takes_the_same_chunked_path(self):
        from cave_agent import CaveAgent, IPythonRuntime

        from .fakes import FakeModel

        summary_model = CapacitySummaryModel(capacity=4_000)
        compactor = Compactor(
            summary_model,
            context_window=4_000,
            output_reserve=0,
            token_estimator=len,
        )
        agent = CaveAgent(
            FakeModel(
                ["first", "second", "third"],
                usage=TokenUsage(
                    prompt_tokens=15_000,
                    completion_tokens=1,
                    total_tokens=15_001,
                ),
            ),
            runtime=IPythonRuntime(),
            compactor=compactor,
        )

        await agent.run("EARLY_SENTINEL " + "x" * 12_000)
        await agent.run("follow-up one")
        await agent.run("follow-up two")

        summaries = [message for message in agent.messages if isinstance(message, SummaryMessage)]
        assert summaries and "EARLY_SENTINEL" in summaries[0].body
        assert summary_model.successful_requests > 1

    async def test_default_fake_usage_does_not_pin_the_anchor(self):
        from cave_agent import CaveAgent, IPythonRuntime

        from .fakes import FakeModel

        summary_model = CapacitySummaryModel(capacity=4_000)
        compactor = Compactor(
            summary_model,
            context_window=4_000,
            output_reserve=0,
            token_estimator=len,
        )
        agent = CaveAgent(
            FakeModel(["first", "second", "third"]),
            runtime=IPythonRuntime(),
            compactor=compactor,
        )

        await agent.run("EARLY_SENTINEL " + "x" * 12_000)
        await agent.run("follow-up one")
        await agent.run("follow-up two")

        assert summary_model.successful_requests > 1
        assert any(
            isinstance(message, SummaryMessage) and "EARLY_SENTINEL" in message.body
            for message in agent.messages
        )


class TestCompact:
    def test_no_compact_under_threshold(self):
        msgs = [SystemMessage("sys"), UserMessage("hi")]
        result, needs = Compactor(SummaryModel(), context_window=200_000).microcompact(msgs)
        assert result is msgs
        assert needs is False

    def test_microcompact_sufficient(self):
        msgs = [SystemMessage("sys")] + _make_messages(12)
        compactor = Compactor(SummaryModel(), context_window=200_000)
        # Threshold just above the post-microcompact size.
        after = _microcompact(msgs[1:])
        compactor.output_reserve = 0
        compactor.context_window = estimate_tokens([msgs[0]] + after) + 1 + COMPACT_BUFFER_TOKENS

        result, needs = compactor.microcompact(msgs)

        assert needs is False
        assert any(m.content == MICROCOMPACT_PLACEHOLDER for m in result)

    def test_escalates_when_microcompact_insufficient(self):
        msgs = [SystemMessage("sys")] + _make_messages(30)
        compactor = Compactor(SummaryModel(), context_window=200_000)
        compactor.output_reserve = 0
        compactor.context_window = COMPACT_BUFFER_TOKENS + 1

        _, needs = compactor.microcompact(msgs)

        assert needs is True

    def test_preserves_system_message(self):
        sys_msg = SystemMessage("system prompt")
        msgs = [sys_msg] + _make_messages(20)
        compactor = Compactor(SummaryModel(), context_window=200_000)
        compactor.output_reserve = 0
        compactor.context_window = COMPACT_BUFFER_TOKENS + 1

        result, _ = compactor.microcompact(msgs)

        assert result[0] is sys_msg

    def test_microcompact_savings_anchored_to_api_count(self):
        """Re-estimating after microcompaction would swap the API-anchored count
        for the bare heuristic, which routinely reads much lower — the cheap
        tier would then look sufficient whenever the heuristic under-counts."""
        msgs = [SystemMessage("sys")] + _make_messages(12)
        compactor = Compactor(SummaryModel(), context_window=200_000)
        compactor.output_reserve = 0
        compactor.context_window = COMPACT_BUFFER_TOKENS + 5_000

        # An API count far above the heuristic: microcompaction saves only a few
        # hundred tokens, so it cannot possibly bring 50k under the threshold.
        _, needs = compactor.microcompact(msgs, anchor=compactor.anchor(msgs, 50_000))

        assert needs is True


class TestRecoverFromOverflow:
    async def test_reduces_more_than_summarize(self):
        msgs = [SystemMessage("sys")] + _make_messages(40)
        compactor = Compactor(SummaryModel())

        recovered = await compactor.recover_from_overflow(msgs)
        summarized = await Compactor(SummaryModel()).summarize(msgs)

        assert len(recovered) <= len(summarized)

    async def test_preserves_system_message(self):
        sys_msg = SystemMessage("sys")
        result = await Compactor(SummaryModel()).recover_from_overflow(
            [sys_msg] + _make_messages(40)
        )
        assert result[0] is sys_msg

    async def test_ignores_circuit_breaker(self):
        """Last resort before the request fails outright — no breaker."""
        model = SummaryModel()
        compactor = Compactor(model)
        compactor.state = CompactionState(consecutive_failures=99, cooldown_remaining=99)

        await compactor.recover_from_overflow([SystemMessage("s")] + _make_messages(40))

        assert model.prompts, "recovery must still call the model"

    async def test_failure_preserves_microcompacted_history(self):
        msgs = [SystemMessage("sys")] + _make_messages(40)
        result = await Compactor(BrokenModel()).recover_from_overflow(msgs)
        assert len(result) == len(msgs)
        assert result[0] is msgs[0]


class TestFormatTranscript:
    def test_basic_formatting(self):
        text = format_transcript([UserMessage("hello"), AssistantMessage("hi")])
        assert "[user]: hello" in text
        assert "[assistant]: hi" in text

    def test_truncation(self):
        text = format_transcript([UserMessage("x" * 5000)], max_chars_per_msg=100)
        assert "..." in text
        assert len(text) < 5000

    def test_truncation_disabled(self):
        text = format_transcript([UserMessage("x" * 5000)], max_chars_per_msg=None)
        assert "..." not in text

    def test_skips_placeholder(self):
        assert format_transcript([ExecutionResultMessage(MICROCOMPACT_PLACEHOLDER)]) == ""

    def test_execution_result_included(self):
        assert "output data" in format_transcript([ExecutionResultMessage("output data")])


class TestExtractSummary:
    def test_extracts_summary_block(self):
        raw = "<analysis>thinking</analysis><summary>The user asked for X.</summary>"
        assert extract_summary(raw) == "The user asked for X."

    def test_ignores_analysis(self):
        assert "secret" not in extract_summary(
            "<analysis>secret</analysis><summary>visible</summary>"
        )

    def test_fallback_without_tags(self):
        assert extract_summary("Plain text summary.") == "Plain text summary."

    def test_multiline(self):
        result = extract_summary("<analysis>x</analysis>\n<summary>\nLine 1\nLine 2\n</summary>")
        assert "Line 1" in result and "Line 2" in result


class TestSplitKeepsPairsTogether:
    """A code message and its result are never separated by the split.

    They used to be, and the orphaned result was then *deleted* from the kept
    tail — while sitting past the split, so it never reached the summarizer
    either. The only copy of a computation's output vanished while the code
    that produced it stayed in history describing it.
    """

    async def _history_from_real_runs(self, count: int):
        """Built the way the running system builds it, never hand-assembled.

        Each output is a string the *source* does not contain, so a result
        going missing cannot be masked by the code that produced it.
        """
        from cave_agent import CaveAgent, IPythonRuntime

        from .fakes import FakeModel

        agent = CaveAgent(model=FakeModel([]), runtime=IPythonRuntime())
        for index in range(1, count + 1):
            agent.model = FakeModel(
                [f"```python\nprint('opaque' + '-result-' + str({index}))\n```", "ok"]
            )
            await agent.run(f"step {index}")
        return agent

    @pytest.mark.parametrize("runs", [3, 4, 5, 6])
    async def test_no_execution_result_is_lost(self, runs):
        agent = await self._history_from_real_runs(runs)
        model = SummaryModel()

        compacted = await Compactor(model).summarize(agent.messages)

        kept = [m.content for m in compacted]
        prompt = model.prompts[0] if model.prompts else ""
        lost = [
            index
            for index in range(1, runs + 1)
            if f"opaque-result-{index}" not in prompt
            and not any(f"opaque-result-{index}" in c for c in kept)
        ]
        assert not lost, f"lost from both the summary and the kept history: {lost}"

    @pytest.mark.parametrize("runs", [3, 4, 5, 6])
    async def test_the_kept_tail_still_opens_cleanly(self, runs):
        agent = await self._history_from_real_runs(runs)

        compacted = await Compactor(SummaryModel()).summarize(agent.messages)

        body = [m for m in compacted if not isinstance(m, (SystemMessage, SummaryMessage))]
        body = [m for m in body if not isinstance(m, SummaryAcknowledgementMessage)]
        assert body[0].role != MessageRole.EXECUTION_RESULT


class TestUnusableSummariesAreRejected:
    """A summary replaces the history it distils, so a partial one is not a
    partial success — it is a permanent loss of everything it failed to cover.

    Judged on the provider's own signals: a refusal reads like prose and a
    truncated summary reads like a summary, so "is it non-empty" accepted both
    and the circuit breaker recorded a success.
    """

    class Responding(Model):
        max_output_tokens = None

        def __init__(self, response):
            self._response = response

        async def _complete(self, messages):
            return self._response

        def stream(self, messages):
            raise NotImplementedError

    @pytest.mark.parametrize(
        "response, why",
        [
            (
                ModelResponse(
                    content="I cannot summarize.",
                    finish_reason="stop",
                    refusal="I cannot summarize.",
                ),
                "refusal",
            ),
            (ModelResponse(content="<summary>PARTIAL", finish_reason="length"), "truncated"),
            (
                ModelResponse(content="<summary>partial</summary>", finish_reason="content_filter"),
                "filtered",
            ),
            (ModelResponse(content="<summary>partial</summary>"), "missing finish reason"),
        ],
    )
    async def test_it_is_not_committed(self, response, why):
        messages = [UserMessage(f"FACT-{i}") for i in range(24)]

        result = await Compactor(self.Responding(response)).summarize(messages)

        assert not any(isinstance(m, SummaryMessage) for m in result), why

    async def test_a_complete_summary_is_committed(self):
        response = ModelResponse(content="<summary>ALL GOOD</summary>", finish_reason="stop")

        result = await Compactor(self.Responding(response)).summarize(
            [UserMessage(f"FACT-{i}") for i in range(24)]
        )

        assert any(isinstance(m, SummaryMessage) for m in result)

    async def test_rejection_counts_against_the_circuit_breaker(self):
        """Recorded as a success, a truncating model would never trip it."""
        compactor = Compactor(self.Responding(ModelResponse(content="x", finish_reason="length")))

        for _ in range(MAX_CONSECUTIVE_FAILURES):
            await compactor.summarize([UserMessage(f"m{i}") for i in range(24)])

        assert compactor.state.should_skip()
