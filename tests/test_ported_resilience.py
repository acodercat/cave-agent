"""Regression tests for resilience features ported from axon:

4. Centralized error classification (context-length / billing) + typed translation.
5. Reactive context-overflow recovery (compact + retry once).
6. Thinking/reasoning capture (surfaced, never silently dropped).
"""

import asyncio
from types import SimpleNamespace

import pytest

from cave_agent import CaveAgent
from cave_agent.compaction import Compactor
from cave_agent.compaction.prompts import COMPACT_SYSTEM_PROMPT
from cave_agent.events import (
    FinalResponseEvent,
    TextEvent,
    ThinkingChunkEvent,
    ThinkingEvent,
)
from cave_agent.messages import SystemMessage, UserMessage
from cave_agent.models import (
    Model,
    ModelResponse,
    PromptTooLongError,
    ProviderBillingError,
    StreamDelta,
    StreamResponse,
    TokenUsage,
    is_billing_exhausted,
    is_context_length_exceeded,
)
from cave_agent.models.base import Model as BaseModel
from cave_agent.models.base import StreamStalledError, stream_with_idle_timeout
from cave_agent.models.errors import classify_provider_error
from cave_agent.models.retry import is_retryable, with_retry_typed
from cave_agent.runtime import IPythonRuntime


class _ProviderTestError(Exception):
    def __init__(self, msg, status=None, code=None):
        super().__init__(msg)
        self.status_code = status
        self.code = code


class TestErrorClassification:
    def test_context_by_http_413(self):
        import httpx
        import openai

        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        response = httpx.Response(413, request=request)
        error = openai.APIStatusError("Payload Too Large", response=response, body={})

        assert is_context_length_exceeded(error)
        assert isinstance(classify_provider_error(error), PromptTooLongError)

    def test_context_by_structured_code(self):
        assert is_context_length_exceeded(_ProviderTestError("x", code="context_length_exceeded"))

    def test_context_by_phrase(self):
        assert is_context_length_exceeded(
            _ProviderTestError("This model's maximum context length is 8192 tokens")
        )

    def test_context_chinese_phrase(self):
        assert is_context_length_exceeded(_ProviderTestError("请求超过最大长度限制"))

    def test_not_context_on_validation_error(self):
        # A false positive here would trigger destructive compaction.
        assert not is_context_length_exceeded(_ProviderTestError("max_tokens must be positive"))

    def test_billing_by_402(self):
        assert is_billing_exhausted(_ProviderTestError("payment", status=402))

    def test_billing_by_phrase(self):
        assert is_billing_exhausted(
            _ProviderTestError("You exceeded your current quota", status=429)
        )

    def test_billing_not_retried_even_on_429(self):
        assert is_retryable(_ProviderTestError("insufficient_quota", status=429)) is False

    def test_plain_429_still_retried(self):
        assert is_retryable(_ProviderTestError("rate limited", status=429)) is True

    def test_classify_maps_to_typed(self):
        assert isinstance(
            classify_provider_error(_ProviderTestError("context window exceeded")),
            PromptTooLongError,
        )
        assert isinstance(
            classify_provider_error(_ProviderTestError("credit balance too low")),
            ProviderBillingError,
        )

    def test_classification_is_total(self):
        """Every input yields a ModelError.

        While this returned unclassified errors unchanged, each call site
        needed its own `if typed is error` fallback — and every one of those
        was a hole a raw exception escaped the agent loop through.
        """
        from cave_agent.models import ModelError, ProviderError

        for raw in (
            _ProviderTestError("something unrelated"),
            ConnectionError("reset"),
            AttributeError("malformed body"),
            Exception("bare"),
        ):
            typed = classify_provider_error(raw)
            assert isinstance(typed, ProviderError)
            assert isinstance(typed, ModelError)

    def test_classify_is_idempotent(self):
        typed = PromptTooLongError("x")
        assert classify_provider_error(typed) is typed

    @pytest.mark.asyncio
    async def test_with_retry_typed_raises_typed(self):
        async def op():
            raise _ProviderTestError("maximum context length is 4096")

        with pytest.raises(PromptTooLongError):
            await with_retry_typed(op)


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

    async def _complete(self, messages):
        if messages and messages[0].get("content") == COMPACT_SYSTEM_PROMPT:
            self.summary_calls += 1
            return ModelResponse(
                content="<summary>earlier turns summarized</summary>",
                usage=TokenUsage(prompt_tokens=10),
                finish_reason="stop",
            )
        self.main_calls += 1
        if self.main_calls == 1:
            raise PromptTooLongError("maximum context length is 8192 tokens")
        return ModelResponse(
            content="All done.",
            usage=TokenUsage(prompt_tokens=20, completion_tokens=3, total_tokens=23),
            finish_reason="stop",
        )

    def stream(self, messages):
        self.stream_calls += 1
        return _OverflowStream(overflow=(self.stream_calls == 1))


def _seed(agent):
    agent.messages = [SystemMessage("sys")] + [UserMessage(f"turn {i}") for i in range(12)]


class _SummaryModel(Model):
    async def _complete(self, messages):
        return ModelResponse(content="<summary>S</summary>", finish_reason="stop")

    def stream(self, messages):
        raise NotImplementedError


class TestOverflowRecovery:
    @pytest.mark.asyncio
    async def test_recover_from_overflow_preserves_system_and_shrinks(self):
        msgs = [SystemMessage("sys")] + [UserMessage(f"m{i}") for i in range(12)]
        out = await Compactor(_SummaryModel()).recover_from_overflow(msgs)
        assert isinstance(out[0], SystemMessage)  # system preserved
        assert len(out) < len(msgs)  # shrunk
        assert any("summary" in m.content.lower() for m in out)  # summary inserted

    @pytest.mark.asyncio
    async def test_streaming_recovers_and_retries(self):
        model = _MockOverflowModel()
        agent = CaveAgent(model=model, runtime=IPythonRuntime())
        _seed(agent)
        events = [e async for e in agent.stream_events("hi")]
        assert model.stream_calls == 2  # overflow, then success
        assert model.summary_calls >= 1
        assert any(isinstance(e, FinalResponseEvent) for e in events)

    @pytest.mark.asyncio
    async def test_run_recovers_and_retries(self):
        """``run()`` drains ``stream_events()``, so there is one execution path
        and one set of recovery semantics — not two that can drift apart."""
        model = _MockOverflowModel()
        agent = CaveAgent(model=model, runtime=IPythonRuntime())
        _seed(agent)
        resp = await agent.run("hi")
        assert model.stream_calls == 2  # overflow, then success
        assert model.main_calls == 0  # the loop never calls non-streaming
        assert model.summary_calls >= 1
        assert "All done." in resp.content


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
            return StreamDelta(thinking=val)  # reasoning streams as a delta
        return StreamDelta(content=val)


class _ReasoningModel(Model):
    async def _complete(self, messages):
        raise NotImplementedError

    def stream(self, messages):
        return _ReasoningStream()


class _ReasoningRefusalStream(StreamResponse):
    def __init__(self):
        super().__init__()
        self._reasoning_sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._reasoning_sent:
            self._reasoning_sent = True
            self.thinking = "private"
            return StreamDelta(thinking="private")
        self.refusal = "No."
        self.finish_reason = "stop"
        raise StopAsyncIteration


class _ReasoningRefusalModel(Model):
    async def _complete(self, messages):
        raise NotImplementedError

    def stream(self, messages):
        return _ReasoningRefusalStream()


class TestThinkingCapture:
    @pytest.mark.asyncio
    async def test_reasoning_streams_as_deltas(self):
        deltas = [d async for d in _ReasoningStream()]
        assert [d.thinking for d in deltas if d.thinking] == ["step one ", "step two"]
        assert [d.content for d in deltas if d.content] == ["All done."]

    @pytest.mark.asyncio
    async def test_thinking_chunks_stream_live_then_seal_before_answer(self):
        agent = CaveAgent(model=_ReasoningModel(), runtime=IPythonRuntime())
        events = [e async for e in agent.stream_events("hi")]
        kinds = [type(e) for e in events]
        # Live reasoning chunks, in order.
        chunk_texts = [e.content for e in events if isinstance(e, ThinkingChunkEvent)]
        assert chunk_texts == ["step one ", "step two"]
        # Segment sealed with the full trace, before the first answer token.
        seal_i = kinds.index(ThinkingEvent)
        text_i = kinds.index(TextEvent)
        assert kinds.index(ThinkingChunkEvent) < seal_i < text_i
        assert events[seal_i].content == "step one step two"
        assert events[seal_i].duration_ms >= 0

    @pytest.mark.asyncio
    async def test_refusal_seals_reasoning_before_the_answer(self):
        agent = CaveAgent(
            model=_ReasoningRefusalModel(),
            runtime=IPythonRuntime(),
        )
        events = [event async for event in agent.stream_events("hi")]
        kinds = [type(event) for event in events]

        assert kinds.count(ThinkingEvent) == 1
        assert kinds.index(ThinkingChunkEvent) < kinds.index(ThinkingEvent)
        assert kinds.index(ThinkingEvent) < kinds.index(TextEvent)
        assert (
            next(event.content for event in events if isinstance(event, ThinkingEvent)) == "private"
        )

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


class TestLiteLLMStreamUsage:
    """Streamed runs must report token usage, but only ask where it is accepted.

    Without `stream_options={"include_usage": True}` an OpenAI-compatible
    stream reports zero tokens, which silently disables the cumulative budgets
    and drops compaction back to its character heuristic. But LiteLLM raises on
    an unsupported parameter (`drop_params` is off by default), and Anthropic /
    Bedrock / Gemini / Ollama do not accept it — so asking unconditionally would
    break the providers LiteLLM exists to reach.
    """

    def test_requested_for_openai_compatible(self):
        from cave_agent.models import LiteLLMModel

        model = LiteLLMModel(model_id="gpt-4o", custom_llm_provider="openai")
        assert model._supports_stream_options() is True

    def test_not_requested_for_anthropic(self):
        from cave_agent.models import LiteLLMModel

        model = LiteLLMModel(
            model_id="claude-3-5-sonnet-20241022",
            custom_llm_provider="anthropic",
        )
        assert model._supports_stream_options() is False

    def test_unknown_model_defaults_to_not_requesting(self):
        """A wrong 'yes' fails the request; a wrong 'no' only costs accounting."""
        from cave_agent.models import LiteLLMModel

        model = LiteLLMModel(model_id="some-model-that-does-not-exist")
        assert model._supports_stream_options() is False

    def test_probe_is_cached(self):
        from cave_agent.models import LiteLLMModel

        model = LiteLLMModel(model_id="gpt-4o", custom_llm_provider="openai")
        assert model._stream_options_supported is None
        model._supports_stream_options()
        assert model._stream_options_supported is True

    def test_async_filter_capability_is_configuration_not_a_wire_parameter(self):
        from cave_agent.models import LiteLLMModel

        model = LiteLLMModel(
            model_id="azure/deployment",
            filters_asynchronously=True,
            custom_llm_provider="azure",
        )

        assert model.filters_asynchronously is True
        assert "filters_asynchronously" not in model.kwargs
        assert "filters_asynchronously" not in model._prepare_params([])

    def test_openai_exposes_the_same_async_filter_configuration(self, monkeypatch):
        import openai

        from cave_agent.models import OpenAIModel

        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: object())
        model = OpenAIModel(
            model_id="azure-deployment",
            filters_asynchronously=True,
        )

        assert model.filters_asynchronously is True
        assert "filters_asynchronously" not in model.kwargs
        assert "filters_asynchronously" not in model._prepare_params([])


class TestProviderRequestMode:
    @staticmethod
    def _response():
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok",
                        refusal=None,
                        reasoning=None,
                        reasoning_content=None,
                    ),
                    finish_reason="stop",
                ),
            ],
            usage=None,
        )

    async def test_openai_call_owns_stream_parameters(self, monkeypatch):
        import openai

        from cave_agent.models import OpenAIModel

        calls = []

        async def create(**params):
            calls.append(params)
            return self._response()

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
        )
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: client)
        model = OpenAIModel(
            model_id="probe",
            stream=True,
            stream_options={"include_usage": True},
        )

        response = await model.call([{"role": "user", "content": "hello"}])

        assert response.content == "ok"
        assert calls[0]["stream"] is False
        assert "stream_options" not in calls[0]

    async def test_litellm_call_owns_stream_parameter(self, monkeypatch):
        import litellm

        from cave_agent.models import LiteLLMModel

        calls = []

        async def complete(**params):
            calls.append(params)
            return self._response()

        monkeypatch.setattr(litellm, "acompletion", complete)
        model = LiteLLMModel(model_id="probe", stream=True)

        response = await model.call([{"role": "user", "content": "hello"}])

        assert response.content == "ok"
        assert calls[0]["stream"] is False


class TestOutputTokenDeclaration:
    """Compaction reserve and the wire cap must not silently disagree."""

    def test_litellm_rejects_a_conflicting_wire_cap(self):
        from cave_agent.models import LiteLLMModel

        with pytest.raises(ValueError, match="max_output_tokens"):
            LiteLLMModel(
                model_id="gpt-4o",
                max_output_tokens=50,
                max_tokens=100,
            )

    def test_litellm_keeps_an_explicit_zero_cap(self):
        from cave_agent.models import LiteLLMModel

        model = LiteLLMModel(model_id="gpt-4o", max_output_tokens=0)

        assert model.max_output_tokens == 0
        assert model.kwargs["max_tokens"] == 0

    def test_litellm_accepts_matching_declared_and_wire_caps(self):
        from cave_agent.models import LiteLLMModel

        model = LiteLLMModel(
            model_id="gpt-4o",
            max_output_tokens=50,
            max_tokens=50,
        )

        assert model.max_output_tokens == 50
        assert model.kwargs["max_tokens"] == 50

    @pytest.mark.parametrize("wire_name", ["max_tokens", "max_completion_tokens"])
    def test_openai_rejects_a_conflicting_wire_cap(self, monkeypatch, wire_name):
        import openai

        from cave_agent.models import OpenAIModel

        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: object())
        with pytest.raises(ValueError, match="max_output_tokens"):
            OpenAIModel(
                model_id="probe",
                max_output_tokens=50,
                **{wire_name: 100},
            )

    def test_openai_rejects_two_wire_cap_spellings(self, monkeypatch):
        import openai

        from cave_agent.models import OpenAIModel

        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: object())
        with pytest.raises(ValueError, match="only one"):
            OpenAIModel(
                model_id="probe",
                max_tokens=50,
                max_completion_tokens=50,
            )

    def test_openai_keeps_an_explicit_zero_declaration(self, monkeypatch):
        import openai

        from cave_agent.models import OpenAIModel

        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: object())
        model = OpenAIModel(model_id="probe", max_output_tokens=0)

        assert model.max_output_tokens == 0

    @pytest.mark.parametrize("wire_name", ["max_tokens", "max_completion_tokens"])
    def test_openai_accepts_matching_declared_and_wire_caps(
        self,
        monkeypatch,
        wire_name,
    ):
        import openai

        from cave_agent.models import OpenAIModel

        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: object())
        model = OpenAIModel(
            model_id="probe",
            max_output_tokens=50,
            **{wire_name: 50},
        )

        assert model.max_output_tokens == 50
        assert model.kwargs[wire_name] == 50


class TestRetryReadsWhatTheProviderSent:
    """The retry policy must work against the SDKs it claims to support."""

    def test_retry_after_is_read_from_the_response(self):
        """The OpenAI SDK carries headers on the response, not the error."""
        import httpx
        import openai

        from cave_agent.models.retry import _get_retry_after

        request = httpx.Request("POST", "https://example/v1")
        response = httpx.Response(429, headers={"Retry-After": "7"}, request=request)

        error = openai.RateLimitError("rate limited", response=response, body=None)

        assert not hasattr(error, "headers")
        assert _get_retry_after(error) == 7.0

    def test_a_client_timeout_is_retryable(self):
        """`APITimeoutError` has no status, is not a builtin TimeoutError, and
        says only "Request timed out." — so every earlier check missed it."""
        import httpx
        import openai

        from cave_agent.models.retry import is_retryable

        request = httpx.Request("POST", "https://example/v1")

        assert is_retryable(openai.APITimeoutError(request=request))
        assert is_retryable(openai.APIConnectionError(request=request))

    def test_a_definite_status_still_wins(self):
        """A 400 mentioning "timeout" is an invalid parameter, not a transient."""
        import httpx
        import openai

        from cave_agent.models.retry import is_retryable

        request = httpx.Request("POST", "https://example/v1")
        response = httpx.Response(400, request=request)

        assert not is_retryable(
            openai.BadRequestError("bad timeout param", response=response, body=None)
        )


class TestTheIdleWatchdogIgnoresDeliberateBackoff:
    """`StreamResponse` retries a failed connection by sleeping *inside*
    `__anext__`, which the watchdog wraps. Counting that against the idle
    budget meant a server's own `Retry-After: 30` spent a quarter of the
    default 120s window per attempt, and the run ended as a stall with retries
    still unspent — the watchdog policing the client's own patience.
    """

    class _FlakyStream(StreamResponse):
        """Fails to connect *failures* times, then delivers one chunk."""

        def __init__(self, failures: int):
            super().__init__()
            self.remaining = failures
            self.attempts = 0

        async def _open_stream(self):
            self.attempts += 1
            if self.remaining > 0:
                self.remaining -= 1
                raise ConnectionError("connection reset by peer")

            async def chunks():
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="hi", reasoning_content=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )

            return chunks()

    @pytest.fixture
    def slow_backoff(self, monkeypatch):
        """A backoff longer than the idle window, as `Retry-After` often is."""
        monkeypatch.setattr(
            "cave_agent.models.base.get_retry_delay", lambda attempt, error=None: 0.3
        )

    async def test_a_retry_backoff_does_not_trip_the_watchdog(self, slow_backoff):
        stream = self._FlakyStream(failures=2)

        received = [item async for item in stream_with_idle_timeout(stream, idle_timeout=0.1)]

        assert stream.attempts == 3, "the retries must actually have happened"
        assert [d.content for d in received] == ["hi"]

    async def test_a_real_stall_still_trips(self):
        class Stalled(StreamResponse):
            async def _open_stream(self):
                async def chunks():
                    await asyncio.sleep(10)
                    yield None

                return chunks()

        with pytest.raises(StreamStalledError):
            async for _ in stream_with_idle_timeout(Stalled(), idle_timeout=0.05):
                pass

    async def test_the_window_still_applies_after_the_backoff(self, slow_backoff):
        """The extension covers the announced wait only — a stall once the
        connection succeeds must still be caught."""

        class RetryThenStall(StreamResponse):
            def __init__(self):
                super().__init__()
                self.opened = False

            async def _open_stream(self):
                if not self.opened:
                    self.opened = True
                    raise ConnectionError("connection reset by peer")

                async def chunks():
                    await asyncio.sleep(10)
                    yield None

                return chunks()

        stream = RetryThenStall()
        with pytest.raises(StreamStalledError):
            async for _ in stream_with_idle_timeout(stream, idle_timeout=0.1):
                pass

        assert stream.opened


class TestTheOutputReserveTracksTheGeneratingModel:
    """The reserve holds room for the *agent's* next completion, but a
    Compactor resolves it from its own model — which, in the documented
    `Compactor(model=cheap_summarizer)` setup, is the summarizer. A summarizer
    declaring 1024 against an agent generating 8192 left the threshold ~7k too
    high, so compaction returned a prompt that still overran the window."""

    class _Declaring(Model):
        def __init__(self, max_output_tokens):
            self.max_output_tokens = max_output_tokens

        async def _complete(self, messages):
            raise NotImplementedError

        def stream(self, messages):
            raise NotImplementedError

    def test_a_supplied_compactor_adopts_the_agents_model(self):
        agent = CaveAgent(
            model=self._Declaring(8192),
            compactor=Compactor(self._Declaring(1024), context_window=100_000),
        )

        assert agent.compactor.output_reserve == 8192

    def test_an_explicit_reserve_still_wins(self):
        agent = CaveAgent(
            model=self._Declaring(8192),
            compactor=Compactor(self._Declaring(1024), context_window=100_000, output_reserve=3000),
        )

        assert agent.compactor.output_reserve == 3000

    def test_a_silent_agent_model_leaves_the_fallback_in_place(self):
        agent = CaveAgent(
            model=self._Declaring(None),
            compactor=Compactor(self._Declaring(None), context_window=100_000),
        )

        assert agent.compactor.output_reserve is None


class TestLiteLLMCarriesItsDeclaredCap:
    """`max_tokens=None` passed explicitly is a *present* key, so `setdefault`
    kept the None and dropped the declaration — leaving `max_output_tokens`
    unset and the compactor sizing its threshold against a fallback."""

    def test_an_explicit_none_does_not_defeat_the_declaration(self):
        from cave_agent.models import LiteLLMModel

        model = LiteLLMModel(model_id="gpt-4o", max_output_tokens=4096, max_tokens=None)

        assert model.max_output_tokens == 4096
        assert model.kwargs["max_tokens"] == 4096

    def test_an_absent_kwarg_still_adopts_the_declaration(self):
        from cave_agent.models import LiteLLMModel

        model = LiteLLMModel(model_id="gpt-4o", max_output_tokens=4096)

        assert model.kwargs["max_tokens"] == 4096

    def test_a_conflicting_pair_is_still_rejected(self):
        from cave_agent.models import LiteLLMModel

        with pytest.raises(ValueError):
            LiteLLMModel(model_id="gpt-4o", max_output_tokens=4096, max_tokens=2048)
