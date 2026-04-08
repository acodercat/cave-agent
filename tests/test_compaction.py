"""Tests for context compaction — microcompact, full compact, circuit breaker, and token estimation."""

import pytest

from cave_agent.types import (
    AssistantMessage,
    ExecutionResultMessage,
    Message,
    MessageRole,
    SystemMessage,
    UserMessage,
)
from cave_agent.compaction import CompactionState, compact_if_needed
from cave_agent.compaction.prompts import (
    MICROCOMPACT_PLACEHOLDER,
    extract_summary,
    format_transcript,
)
from cave_agent.compaction.strategies import (
    KEEP_RECENT_EXECUTION_RESULTS,
    MAX_CONSECUTIVE_FAILURES,
    MIN_KEEP_MESSAGES,
    full_compact,
    microcompact,
    trim_fallback,
)
from cave_agent.compaction.tokens import (
    CHARS_PER_TOKEN,
    COMPACT_BUFFER_TOKENS,
    OUTPUT_TOKENS_RESERVE,
    compact_threshold,
    estimate_tokens,
)
from cave_agent.models.base import Model, ModelResponse, StreamResponse, TokenUsage


# ---------------------------------------------------------------------------
# Mock model for compaction tests
# ---------------------------------------------------------------------------

class MockSummaryModel(Model):
    """Returns a fixed summary for compaction tests."""

    def __init__(self, summary: str = "Summary of the conversation."):
        self._summary = summary

    async def call(self, messages):
        return ModelResponse(content=f"<analysis>thinking</analysis><summary>{self._summary}</summary>")

    def stream(self, messages):
        raise NotImplementedError


class FailingModel(Model):
    """Always raises an error."""

    async def call(self, messages):
        raise RuntimeError("model unavailable")

    def stream(self, messages):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages(n_results: int) -> list[Message]:
    """Create a conversation with n execution results interleaved."""
    msgs: list[Message] = []
    for i in range(n_results):
        msgs.append(AssistantMessage(f"```python\nprint({i})\n```"))
        msgs.append(ExecutionResultMessage(f"Output {i}: {'x' * 200}"))
    return msgs


def _make_long_conversation(n_results: int) -> list[Message]:
    """Create a conversation with system message + many execution results."""
    msgs: list[Message] = [SystemMessage("You are a helpful assistant.")]
    for i in range(n_results):
        msgs.append(UserMessage(f"Do task {i}"))
        msgs.append(AssistantMessage(f"```python\nresult_{i} = compute({i})\n```"))
        msgs.append(ExecutionResultMessage(f"Result: {'data' * 100}"))
    return msgs


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_simple(self):
        msgs = [UserMessage("hello world")]
        tokens = estimate_tokens(msgs)
        assert tokens == len("hello world") // CHARS_PER_TOKEN

    def test_api_count_preferred(self):
        msgs = [UserMessage("hello")]
        assert estimate_tokens(msgs, api_token_count=999) == 999

    def test_api_count_zero_falls_back(self):
        msgs = [UserMessage("hello")]
        assert estimate_tokens(msgs, api_token_count=0) == len("hello") // CHARS_PER_TOKEN


class TestCompactThreshold:
    def test_formula(self):
        assert compact_threshold(128_000) == 128_000 - OUTPUT_TOKENS_RESERVE - COMPACT_BUFFER_TOKENS

    def test_small_window_fallback(self):
        # When reserves exceed window, falls back to 50% of window
        assert compact_threshold(20_000) == 10_000


# ---------------------------------------------------------------------------
# Microcompact
# ---------------------------------------------------------------------------

class TestMicrocompact:
    def test_no_change_when_few_results(self):
        msgs = _make_messages(3)
        result = microcompact(msgs)
        assert result is msgs

    def test_clears_old_results(self):
        msgs = _make_messages(12)
        result = microcompact(msgs)
        cleared = [m for m in result if isinstance(m, ExecutionResultMessage) and m.content == MICROCOMPACT_PLACEHOLDER]
        kept = [m for m in result if isinstance(m, ExecutionResultMessage) and m.content != MICROCOMPACT_PLACEHOLDER]
        assert len(cleared) > 0
        assert len(kept) == KEEP_RECENT_EXECUTION_RESULTS

    def test_preserves_message_count(self):
        msgs = _make_messages(12)
        result = microcompact(msgs)
        assert len(result) == len(msgs)

    def test_preserves_non_result_messages(self):
        msgs = _make_messages(12)
        original_assistants = [m for m in msgs if isinstance(m, AssistantMessage)]
        result = microcompact(msgs)
        result_assistants = [m for m in result if isinstance(m, AssistantMessage)]
        assert len(result_assistants) == len(original_assistants)


# ---------------------------------------------------------------------------
# Full compact
# ---------------------------------------------------------------------------

class TestFullCompact:
    @pytest.mark.asyncio
    async def test_reduces_messages(self):
        msgs = _make_messages(20)
        state = CompactionState()
        result = await full_compact(msgs, MockSummaryModel(), state)
        assert len(result) < len(msgs)

    @pytest.mark.asyncio
    async def test_summary_present(self):
        msgs = _make_messages(20)
        state = CompactionState()
        result = await full_compact(msgs, MockSummaryModel(), state)
        assert any("summary" in m.content.lower() for m in result if isinstance(m, UserMessage))

    @pytest.mark.asyncio
    async def test_recent_messages_preserved(self):
        msgs = _make_messages(20)
        state = CompactionState()
        result = await full_compact(msgs, MockSummaryModel(), state)
        keep_count = max(len(msgs) // 2, MIN_KEEP_MESSAGES)
        assert any(m.content == msgs[-1].content for m in result)

    @pytest.mark.asyncio
    async def test_no_change_when_too_few(self):
        msgs = _make_messages(1)
        state = CompactionState()
        result = await full_compact(msgs, MockSummaryModel(), state)
        assert result is msgs


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_failure_increments_count(self):
        state = CompactionState()
        msgs = _make_messages(20)
        await full_compact(msgs, FailingModel(), state)
        assert state.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_fallback_after_max_failures(self):
        state = CompactionState(consecutive_failures=MAX_CONSECUTIVE_FAILURES)
        msgs = _make_messages(20)
        result = await full_compact(msgs, FailingModel(), state)
        # Circuit breaker active — should use trim_fallback, not call model
        assert len(result) <= len(msgs)

    @pytest.mark.asyncio
    async def test_success_resets_count(self):
        state = CompactionState(consecutive_failures=2)
        msgs = _make_messages(20)
        await full_compact(msgs, MockSummaryModel(), state)
        assert state.consecutive_failures == 0


# ---------------------------------------------------------------------------
# Trim fallback
# ---------------------------------------------------------------------------

class TestTrimFallback:
    def test_keeps_recent_half(self):
        msgs = _make_messages(20)
        result = trim_fallback(msgs)
        assert len(result) <= len(msgs)
        assert len(result) >= MIN_KEEP_MESSAGES

    def test_removes_orphan_results(self):
        msgs = [ExecutionResultMessage("orphan"), UserMessage("hi"), AssistantMessage("hello")]
        result = trim_fallback(msgs)
        assert result[0].role != MessageRole.EXECUTION_RESULT


# ---------------------------------------------------------------------------
# compact_if_needed (orchestration)
# ---------------------------------------------------------------------------

class TestCompactIfNeeded:
    @pytest.mark.asyncio
    async def test_no_compact_under_threshold(self):
        msgs = [SystemMessage("sys"), UserMessage("hi")]
        state = CompactionState()
        result, tier = await compact_if_needed(msgs, MockSummaryModel(), state, context_window=200_000)
        assert result is msgs
        assert tier is None

    @pytest.mark.asyncio
    async def test_small_window_uses_fallback_threshold(self):
        msgs = [SystemMessage("sys"), UserMessage("hi")]
        state = CompactionState()
        # Small window → threshold = 10k, message tokens << 10k → no compact
        result, tier = await compact_if_needed(msgs, MockSummaryModel(), state, context_window=20_000)
        assert result is msgs
        assert tier is None

    @pytest.mark.asyncio
    async def test_microcompact_sufficient(self):
        msgs = [SystemMessage("sys")] + _make_messages(12)
        state = CompactionState()
        # Set threshold between post-microcompact and pre-microcompact tokens
        tokens_before = estimate_tokens(msgs)
        body_after_mc = microcompact(msgs[1:])
        tokens_after_mc = estimate_tokens([msgs[0]] + body_after_mc)
        # Window that makes threshold just above post-microcompact but below pre-microcompact
        context_window = tokens_after_mc + 1 + OUTPUT_TOKENS_RESERVE + COMPACT_BUFFER_TOKENS
        result, tier = await compact_if_needed(msgs, MockSummaryModel(), state, context_window=context_window)
        cleared = [m for m in result if isinstance(m, ExecutionResultMessage) and m.content == MICROCOMPACT_PLACEHOLDER]
        assert len(cleared) > 0
        assert tier == "microcompact"

    @pytest.mark.asyncio
    async def test_preserves_system_message(self):
        sys_msg = SystemMessage("system prompt")
        msgs = [sys_msg] + _make_messages(20)
        state = CompactionState()
        result, tier = await compact_if_needed(msgs, MockSummaryModel(), state, context_window=30_000)
        assert result[0] is sys_msg

    @pytest.mark.asyncio
    async def test_full_compact_when_micro_insufficient(self):
        msgs = [SystemMessage("sys")] + _make_messages(30)
        state = CompactionState()
        # Set threshold below post-microcompact tokens to force full compact
        body_after_mc = microcompact(msgs[1:])
        tokens_after_mc = estimate_tokens([msgs[0]] + body_after_mc)
        context_window = tokens_after_mc - 1 + OUTPUT_TOKENS_RESERVE + COMPACT_BUFFER_TOKENS
        result, tier = await compact_if_needed(msgs, MockSummaryModel(), state, context_window=context_window)
        assert len(result) < len(msgs)
        assert tier == "full_compact"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

class TestFormatTranscript:
    def test_basic_formatting(self):
        msgs = [UserMessage("hello"), AssistantMessage("hi")]
        text = format_transcript(msgs)
        assert "[user]: hello" in text
        assert "[assistant]: hi" in text

    def test_truncation(self):
        long_msg = UserMessage("x" * 5000)
        text = format_transcript([long_msg], max_chars_per_msg=100)
        assert "..." in text
        assert len(text) < 5000

    def test_skips_placeholder(self):
        msgs = [ExecutionResultMessage(MICROCOMPACT_PLACEHOLDER)]
        text = format_transcript(msgs)
        assert text == ""

    def test_execution_result_included(self):
        msgs = [ExecutionResultMessage("output data")]
        text = format_transcript(msgs)
        assert "output data" in text


class TestExtractSummary:
    def test_extracts_summary_block(self):
        raw = "<analysis>thinking</analysis><summary>The user asked for X.</summary>"
        assert extract_summary(raw) == "The user asked for X."

    def test_ignores_analysis(self):
        raw = "<analysis>secret</analysis><summary>visible</summary>"
        assert "secret" not in extract_summary(raw)

    def test_fallback_without_tags(self):
        raw = "Plain text summary."
        assert extract_summary(raw) == raw

    def test_multiline(self):
        raw = "<analysis>x</analysis>\n<summary>\nLine 1\nLine 2\n</summary>"
        result = extract_summary(raw)
        assert "Line 1" in result
        assert "Line 2" in result
