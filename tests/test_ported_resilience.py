"""Regression tests for resilience features ported from axon:

4. Centralized error classification (context-length / billing) + typed translation.
5. Reactive context-overflow recovery (compact + retry once).
6. Thinking/reasoning capture (surfaced, never silently dropped).
"""

import pytest

from cave_agent import CaveAgent
from cave_agent.runtime import IPythonRuntime
from cave_agent.models import (
    Model, ModelResponse, TokenUsage, StreamResponse, StreamDelta,
    PromptTooLongError, ProviderBillingError,
    is_context_length_exceeded, is_billing_exhausted,
)
from cave_agent.models.base import Model as BaseModel
from cave_agent.models.errors import classify_provider_error
from cave_agent.models.retry import is_retryable, with_retry_typed
from cave_agent.compaction import recover_from_overflow
from cave_agent.compaction.prompts import COMPACT_SYSTEM_PROMPT
from cave_agent.types import SystemMessage, UserMessage, EventType


class _Err(Exception):
    def __init__(self, msg, status=None, code=None):
        super().__init__(msg)
        self.status_code = status
        self.code = code


# ---------------------------------------------------------------------------
# #4 — error classification
# ---------------------------------------------------------------------------

class TestErrorClassification:
    def test_context_by_structured_code(self):
        assert is_context_length_exceeded(_Err("x", code="context_length_exceeded"))

    def test_context_by_phrase(self):
        assert is_context_length_exceeded(_Err("This model's maximum context length is 8192 tokens"))

    def test_context_chinese_phrase(self):
        assert is_context_length_exceeded(_Err("请求超过最大长度限制"))

    def test_not_context_on_validation_error(self):
        # A false positive here would trigger destructive compaction.
        assert not is_context_length_exceeded(_Err("max_tokens must be positive"))

    def test_billing_by_402(self):
        assert is_billing_exhausted(_Err("payment", status=402))

    def test_billing_by_phrase(self):
        assert is_billing_exhausted(_Err("You exceeded your current quota", status=429))

    def test_billing_not_retried_even_on_429(self):
        assert is_retryable(_Err("insufficient_quota", status=429)) is False

    def test_plain_429_still_retried(self):
        assert is_retryable(_Err("rate limited", status=429)) is True

    def test_classify_maps_to_typed(self):
        assert isinstance(classify_provider_error(_Err("context window exceeded")), PromptTooLongError)
        assert isinstance(classify_provider_error(_Err("credit balance too low")), ProviderBillingError)

    def test_classify_passthrough_and_idempotent(self):
        plain = _Err("something unrelated")
        assert classify_provider_error(plain) is plain
        typed = PromptTooLongError("x")
        assert classify_provider_error(typed) is typed

    @pytest.mark.asyncio
    async def test_with_retry_typed_raises_typed(self):
        async def op():
            raise _Err("maximum context length is 4096")
        with pytest.raises(PromptTooLongError):
            await with_retry_typed(op)


# ---------------------------------------------------------------------------
# #5 — reactive overflow recovery
# ---------------------------------------------------------------------------

class _OverflowStream(StreamResponse):
    def __init__(self, overflow):
        super().__init__()
        self._overflow = overflow
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._overflow:
            raise PromptTooLongError("maximum context length is 8192 tokens")
        if self._sent:
            self.finish_reason = "stop"
            raise StopAsyncIteration
        self._sent = True
        self.usage = TokenUsage(prompt_tokens=20, completion_tokens=3, total_tokens=23)
        return StreamDelta(content="All done.")


class _MockOverflowModel(Model):
    """Raises overflow on the first main call/stream, succeeds after recovery.

    Summary-generation calls (identified by the compaction system prompt) are
    served separately so recovery can produce a summary.
    """

    def __init__(self):
        self.main_calls = 0
        self.stream_calls = 0
        self.summary_calls = 0

    async def call(self, messages):
        if messages and messages[0].get("content") == COMPACT_SYSTEM_PROMPT:
            self.summary_calls += 1
            return ModelResponse(
                content="<summary>earlier turns summarized</summary>",
                token_usage=TokenUsage(prompt_tokens=10), finish_reason="stop",
            )
        self.main_calls += 1
        if self.main_calls == 1:
            raise PromptTooLongError("maximum context length is 8192 tokens")
        return ModelResponse(
            content="All done.",
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=3, total_tokens=23),
            finish_reason="stop",
        )

    def stream(self, messages):
        self.stream_calls += 1
        return _OverflowStream(overflow=(self.stream_calls == 1))


def _seed(agent):
    agent.messages = [SystemMessage("sys")] + [UserMessage(f"turn {i}") for i in range(12)]


class _SummaryModel(Model):
    async def call(self, messages):
        return ModelResponse(content="<summary>S</summary>")

    def stream(self, messages):
        raise NotImplementedError


class TestOverflowRecovery:
    @pytest.mark.asyncio
    async def test_recover_from_overflow_preserves_system_and_shrinks(self):
        msgs = [SystemMessage("sys")] + [UserMessage(f"m{i}") for i in range(12)]
        out = await recover_from_overflow(msgs, _SummaryModel())
        assert isinstance(out[0], SystemMessage)          # system preserved
        assert len(out) < len(msgs)                        # shrunk
        assert any("summary" in m.content.lower() for m in out)  # summary inserted

    @pytest.mark.asyncio
    async def test_streaming_recovers_and_retries(self):
        model = _MockOverflowModel()
        agent = CaveAgent(model=model, runtime=IPythonRuntime(), display=False)
        _seed(agent)
        events = [e async for e in agent.stream_events("hi")]
        assert model.stream_calls == 2          # overflow, then success
        assert model.summary_calls >= 1
        assert any(e.type == EventType.FINAL_RESPONSE for e in events)

    @pytest.mark.asyncio
    async def test_nonstreaming_recovers_and_retries(self):
        model = _MockOverflowModel()
        agent = CaveAgent(model=model, runtime=IPythonRuntime(), display=False)
        _seed(agent)
        resp = await agent.run("hi")
        assert model.main_calls == 2            # overflow, then success
        assert "All done." in resp.content


# ---------------------------------------------------------------------------
# #6 — thinking / reasoning capture
# ---------------------------------------------------------------------------

class _ReasoningStream(StreamResponse):
    def __init__(self):
        super().__init__()
        self._q = [("reason", "step one "), ("reason", "step two"), ("content", "All done.")]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._q:
            self.finish_reason = "stop"
            self.usage = TokenUsage(prompt_tokens=5)
            raise StopAsyncIteration
        kind, val = self._q.pop(0)
        if kind == "reason":
            self.thinking += val
            return StreamDelta(thinking=val)   # reasoning streams as a delta
        return StreamDelta(content=val)


class _ReasoningModel(Model):
    async def call(self, messages):
        raise NotImplementedError

    def stream(self, messages):
        return _ReasoningStream()


class TestThinkingCapture:
    @pytest.mark.asyncio
    async def test_reasoning_streams_as_deltas(self):
        deltas = [d async for d in _ReasoningStream()]
        assert [d.thinking for d in deltas if d.thinking] == ["step one ", "step two"]
        assert [d.content for d in deltas if d.content] == ["All done."]

    @pytest.mark.asyncio
    async def test_thinking_chunks_stream_live_then_seal_before_answer(self):
        agent = CaveAgent(model=_ReasoningModel(), runtime=IPythonRuntime(), display=False)
        events = [e async for e in agent.stream_events("hi")]
        kinds = [e.type for e in events]
        # Live reasoning chunks, in order.
        chunk_texts = [e.content for e in events if e.type == EventType.THINKING_CHUNK]
        assert chunk_texts == ["step one ", "step two"]
        # Segment sealed with the full trace, before the first answer token.
        seal_i = kinds.index(EventType.THINKING)
        text_i = kinds.index(EventType.TEXT)
        assert kinds.index(EventType.THINKING_CHUNK) < seal_i < text_i
        assert events[seal_i].content == "step one step two"

    def test_extract_thinking_non_streaming(self):
        class _Msg:
            reasoning = None
            reasoning_content = "the reasoning"
            content = "a"

        class _Choice:
            message = _Msg()
            finish_reason = "stop"

        class _Resp:
            choices = [_Choice()]

        assert BaseModel._extract_thinking(_Resp()) == "the reasoning"

    def test_modelresponse_carries_thinking(self):
        assert ModelResponse(content="x", thinking="r").thinking == "r"
