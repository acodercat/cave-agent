"""Regression tests for features ported from axon:

1. Lone-surrogate sanitization at message boundaries (prevents json.dumps crash).
2. CJK-aware token estimation (prevents under-counting → late compaction).
3. Stream idle-timeout watchdog (prevents infinite hang on a stalled stream).
"""

import asyncio
import json

import pytest

from cave_agent import CaveAgent
from cave_agent.agent import _RunState
from cave_agent.compaction.tokens import default_token_estimate, estimate_tokens
from cave_agent.models import Model, stream_with_idle_timeout
from cave_agent.runtime import IPythonRuntime
from cave_agent.types import UserMessage


class _DummyModel(Model):
    """A Model that is never called — only to satisfy CaveAgent construction."""

    async def call(self, messages):
        raise NotImplementedError

    def stream(self, messages):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# #1 — surrogate sanitization
# ---------------------------------------------------------------------------

class TestSanitizeSurrogates:
    def test_strips_lone_surrogate(self):
        from cave_agent.utils import sanitize_surrogates
        out = sanitize_surrogates("a\ud800b\udfffc")
        assert "\ud800" not in out and "\udfff" not in out
        assert out == "a�b�c"

    def test_result_is_json_serializable(self):
        from cave_agent.utils import sanitize_surrogates
        # The raw string would raise UnicodeEncodeError inside json.dumps.
        json.dumps({"content": sanitize_surrogates("x\ud800y")})

    def test_preserves_clean_text_and_emoji(self):
        from cave_agent.utils import sanitize_surrogates
        assert sanitize_surrogates("hello 😀 世界") == "hello 😀 世界"

    def test_empty_passthrough(self):
        from cave_agent.utils import sanitize_surrogates
        assert sanitize_surrogates("") == ""

    @pytest.mark.asyncio
    async def test_execution_stdout_is_sanitized(self):
        # Code that prints a lone surrogate into stdout must not produce a
        # tool-result message that later crashes the provider SDK's json.dumps.
        agent = CaveAgent(model=_DummyModel(), runtime=IPythonRuntime(), display=False)
        ctx = _RunState(agent.max_steps)
        outcome = await agent._execute_code(r"print('x\ud800y')", ctx)
        assert "\ud800" not in outcome.next_prompt
        json.dumps({"content": outcome.next_prompt})  # json-safe


# ---------------------------------------------------------------------------
# #2 — CJK-aware token estimation
# ---------------------------------------------------------------------------

class TestCJKTokenEstimate:
    def test_english_uses_quarter(self):
        assert default_token_estimate("hello world") == len("hello world") // 4

    def test_pure_cjk_uses_half(self):
        text = "你好世界" * 10  # 40 CJK chars
        assert default_token_estimate(text) == 40 // 2

    def test_mixed_high_density_splits(self):
        text = "你好世界hello"  # 4 CJK + 5 latin, density 44% > 30%
        assert default_token_estimate(text) == 4 // 2 + 5 // 4

    def test_mixed_low_density_uses_english(self):
        text = "the quick 你 fox"  # 1 CJK / 15 chars < 30%
        assert default_token_estimate(text) == len(text) // 4

    def test_empty(self):
        assert default_token_estimate("") == 0

    def test_estimate_tokens_reflects_cjk(self):
        # A Chinese message estimates ~2× what the old len//4 heuristic gave.
        msgs = [UserMessage("你好世界" * 100)]  # 400 CJK chars
        assert estimate_tokens(msgs) == 400 // 2

    def test_estimate_tokens_prefers_max_of_api(self):
        assert estimate_tokens([UserMessage("hi")], api_token_count=999) == 999


# ---------------------------------------------------------------------------
# #3 — stream idle-timeout watchdog
# ---------------------------------------------------------------------------

async def _gen(items, stall_after=None, stall=0.0):
    for i, item in enumerate(items):
        if stall_after is not None and i == stall_after:
            await asyncio.sleep(stall)
        yield item


class TestStreamIdleTimeout:
    @pytest.mark.asyncio
    async def test_passthrough_when_active(self):
        out = [x async for x in stream_with_idle_timeout(_gen("abc"), idle_timeout=1.0)]
        assert out == list("abc")

    @pytest.mark.asyncio
    async def test_none_disables_watchdog(self):
        out = [x async for x in stream_with_idle_timeout(_gen("abc"), idle_timeout=None)]
        assert out == list("abc")

    @pytest.mark.asyncio
    async def test_raises_on_stall_after_partial(self):
        got = []
        with pytest.raises(TimeoutError):
            async for x in stream_with_idle_timeout(
                _gen("ab", stall_after=1, stall=1.0), idle_timeout=0.1,
            ):
                got.append(x)
        assert got == ["a"]  # first item delivered, then the stall trips

    def test_agent_defaults_and_override(self):
        assert CaveAgent(model=_DummyModel(), display=False).stream_idle_timeout == 120.0
        assert CaveAgent(
            model=_DummyModel(), stream_idle_timeout=None, display=False,
        ).stream_idle_timeout is None
