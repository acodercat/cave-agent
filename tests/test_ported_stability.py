"""Regression tests for stability features ported from axon:

1. Stream close on early break (no per-turn connection leak).
3. SSE-connect retry — retry a transport failure only before any text is yielded.
5. Wall-clock cap (max_run_time) at turn boundaries.
"""

import pytest

from cave_agent import CaveAgent
from cave_agent.agent import ExecutionStatus
from cave_agent.models import Model, ModelResponse, TokenUsage, StreamResponse, StreamDelta
from cave_agent.runtime import IPythonRuntime
from cave_agent.types import EventType


# --- OpenAI-style chunk doubles -------------------------------------------

class _Delta:
    def __init__(self, content=None):
        self.content = content
        self.reasoning = None
        self.reasoning_content = None


class _Choice:
    def __init__(self, content=None, finish=None):
        self.delta = _Delta(content)
        self.finish_reason = finish


class _Chunk:
    def __init__(self, content=None, finish=None):
        self.choices = [_Choice(content, finish)]
        self.usage = None


# --- A provider-style stream that drives the StreamResponse base -----------

class _RawStream:
    """Minimal raw provider stream exposing __aiter__/__anext__/close."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunks:
            return self._chunks.pop(0)
        raise StopAsyncIteration

    async def close(self):
        self.closed = True


class _ProviderStream(StreamResponse):
    """Uses the base iteration/usage/close — only opens the raw stream."""

    def __init__(self, chunks):
        super().__init__()
        self._chunks = chunks
        self.raw = None

    async def _open_stream(self):
        self.raw = _RawStream(self._chunks)
        return self.raw


# ===========================================================================
# #1 — stream close
# ===========================================================================

class TestStreamClose:
    @pytest.mark.asyncio
    async def test_aclose_closes_underlying_and_is_idempotent(self):
        stream = _ProviderStream([_Chunk(content="hi")])
        assert (await stream.__aiter__().__anext__()).content == "hi"
        raw = stream.raw
        await stream.aclose()
        assert raw.closed is True
        await stream.aclose()  # idempotent — no error

    @pytest.mark.asyncio
    async def test_agent_closes_stream_on_early_break(self):
        # A completed code block makes the agent stop reading early — the
        # unread stream must still be closed (the leak this fixes).
        class _Model(Model):
            def __init__(self):
                self.last_stream = None

            async def call(self, messages):
                raise NotImplementedError

            def stream(self, messages):
                self.last_stream = _ProviderStream([
                    _Chunk(content="```python\nprint(1)\n```"),
                    _Chunk(content=" trailing (never read)"),
                    _Chunk(finish="stop"),
                ])
                return self.last_stream

        model = _Model()
        agent = CaveAgent(model=model, runtime=IPythonRuntime(), display=False, max_steps=1)
        [event async for event in agent.stream_events("go")]
        assert model.last_stream.raw is not None
        assert model.last_stream.raw.closed is True


# ===========================================================================
# #3 — SSE-connect retry
# ===========================================================================

class _FlakyRaw:
    def __init__(self, parent):
        self._parent = parent
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        # Drop the very first read (before any text) to simulate an SSE-connect
        # failure; succeed once the parent's fail budget is spent.
        if self._parent.fails > 0 and self._i == 0 and not self._parent._yielded_any:
            self._parent.fails -= 1
            raise ConnectionError("SSE connect dropped")
        if self._i == 0:
            self._i = 1
            return _Chunk(content="hi")
        if self._i == 1:
            self._i = 2
            return _Chunk(finish="stop")
        raise StopAsyncIteration

    async def close(self):
        pass


class _FlakyStream(StreamResponse):
    def __init__(self, fails):
        super().__init__()
        self.fails = fails
        self.opens = 0

    async def _open_stream(self):
        self.opens += 1
        return _FlakyRaw(self)


class TestConnectRetry:
    @pytest.fixture(autouse=True)
    def _fast_backoff(self, monkeypatch):
        monkeypatch.setattr("cave_agent.models.retry.BASE_DELAY", 0.001)

    @pytest.mark.asyncio
    async def test_retries_connect_before_first_text(self):
        stream = _FlakyStream(fails=2)
        out = [delta async for delta in stream]
        assert [d.content for d in out] == ["hi"]
        assert stream.opens == 3  # 2 failed opens + 1 success

    @pytest.mark.asyncio
    async def test_no_retry_after_first_text(self):
        class _Raw:
            def __init__(self):
                self._i = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._i == 0:
                    self._i = 1
                    return _Chunk(content="hi")
                raise ConnectionError("drop after first text")

            async def close(self):
                pass

        class _S(StreamResponse):
            async def _open_stream(self):
                return _Raw()

        got = []
        with pytest.raises(ConnectionError):
            async for delta in _S():
                got.append(delta.content)
        assert got == ["hi"]  # yielded once, then the error propagates un-retried

    @pytest.mark.asyncio
    async def test_non_retryable_error_propagates(self):
        class _Raw:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise ValueError("deterministic bad request")

            async def close(self):
                pass

        class _S(StreamResponse):
            async def _open_stream(self):
                return _Raw()

        with pytest.raises(ValueError):
            async for _ in _S():
                pass


# ===========================================================================
# #5 — wall-clock cap
# ===========================================================================

class _TextStream(StreamResponse):
    def __init__(self):
        super().__init__()
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            self.finish_reason = "stop"
            raise StopAsyncIteration
        self._done = True
        return StreamDelta(content="done")


class _TextModel(Model):
    async def call(self, messages):
        return ModelResponse(content="done", finish_reason="stop", token_usage=TokenUsage(prompt_tokens=1))

    def stream(self, messages):
        return _TextStream()


def _agent(**kwargs):
    return CaveAgent(model=_TextModel(), runtime=IPythonRuntime(), display=False, **kwargs)


class TestWallClockCap:
    @pytest.mark.asyncio
    async def test_non_streaming_times_out(self):
        resp = await _agent(max_run_time=0.0).run("hi")
        assert resp.status == ExecutionStatus.TIMEOUT
        assert resp.steps_taken == 0

    @pytest.mark.asyncio
    async def test_stream_emits_timeout_event(self):
        events = [e async for e in _agent(max_run_time=0.0).stream_events("hi")]
        assert any(e.type == EventType.MAX_RUN_TIME_REACHED for e in events)

    @pytest.mark.asyncio
    async def test_none_disables_and_completes(self):
        resp = await _agent(max_run_time=None).run("hi")
        assert resp.status == ExecutionStatus.SUCCESS
