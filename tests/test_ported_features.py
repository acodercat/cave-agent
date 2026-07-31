"""Regression tests for features ported from axon:

1. Lone-surrogate sanitization at message boundaries (prevents json.dumps crash).
2. CJK-aware token estimation (prevents under-counting → late compaction).
3. Stream idle-timeout watchdog (prevents infinite hang on a stalled stream).
"""

import asyncio
import json

import pytest

from cave_agent import CaveAgent
from cave_agent.compaction.tokens import (
    CJK_TOKENS_PER_CHAR,
    default_token_estimate,
    estimate_tokens,
)
from cave_agent.messages import UserMessage
from cave_agent.models import Model, stream_with_idle_timeout
from cave_agent.runtime import IPythonRuntime


class _DummyModel(Model):
    """A Model that is never called — only to satisfy CaveAgent construction."""

    async def _complete(self, messages):
        raise NotImplementedError

    def stream(self, messages):
        raise NotImplementedError


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
        # Code that prints a lone surrogate into stdout must not produce an
        # execution-result message that later crashes the provider SDK's
        # json.dumps.
        agent = CaveAgent(model=_DummyModel(), runtime=IPythonRuntime())
        _, next_prompt = await agent._execute(r"print('x\ud800y')")
        assert "\ud800" not in next_prompt
        json.dumps({"content": next_prompt})  # json-safe


class TestCJKTokenEstimate:
    """The estimate must not under-count — that is the only dangerous direction.

    Compaction fires when the estimate crosses a threshold. Over-counting
    compacts a little early; under-counting lets the real prompt sail past the
    context window until the API rejects it. So these assert the *bound*
    against a real tokenizer, not the arithmetic of the current constants.
    """

    def test_english_charges_the_latin_rate(self):
        text = "the quick brown fox jumps over the lazy dog"
        assert default_token_estimate(text) == int(len(text) * 0.25)

    def test_cjk_charges_more_than_one_token_per_char(self):
        """The rate the old ``chars // 2`` form could not express at all."""
        text = "你好世界" * 10
        assert default_token_estimate(text) == int(40 * CJK_TOKENS_PER_CHAR)

    def test_mixed_text_charges_each_class_its_own_rate(self):
        """No density threshold: the 30% cutoff under-counted every text below it.

        A Chinese question about an English stack trace is ~20% CJK — exactly
        the shape a bilingual session produces, and exactly what the old
        threshold billed at the Latin rate end to end.
        """
        text = "你好世界" + "a" * 96  # 4% CJK, far below the old 30% cutoff
        assert default_token_estimate(text) == int(4 * CJK_TOKENS_PER_CHAR + 96 * 0.25)

    def test_empty(self):
        assert default_token_estimate("") == 0

    @pytest.mark.parametrize(
        "text",
        [
            "人工智能正在快速改变软件开发的方式，代理框架让模型可以直接生成并执行代码。" * 6,
            "エージェントはコードを生成して実行します。" * 6,
            "에이전트는 코드를 생성하고 실행합니다." * 6,
            "你好世界" * 200,
            # Supplementary-plane ideographs: no vocabulary in this class has room
            # for them, so each costs 3-4 tokens. Charged the BMP CJK rate they
            # were under-counted threefold.
            "\U00020000\U00022000\U0002cea1" * 200,
        ],
    )
    def test_never_under_counts_a_real_tokenizer(self, text):
        """CJK-dominant text, the case the old heuristic missed by 2-3×."""
        tiktoken = pytest.importorskip("tiktoken")
        actual = len(tiktoken.get_encoding("cl100k_base").encode(text))
        assert default_token_estimate(text) >= actual

    @pytest.mark.parametrize(
        "text",
        [
            "def add(a: int, b: int) -> int:\n    return a + b\n" * 20,
            '{"name": "alpha", "values": [1, 2, 3], "nested": {"k": "v"}}' * 20,
        ],
    )
    def test_latin_rate_still_under_counts_dense_structured_text(self, text):
        """Documents a *known, deferred* gap — asserted so it cannot drift unseen.

        Python and JSON measure ~0.39-0.43 tokens/char, but the Latin rate
        charges 0.25. Raising it is not free: English prose measures 0.19, so a
        rate that fits code would over-count prose by ~2×, compacting chat-heavy
        sessions long before they need it. The real answer is a role-aware rate
        (code and execution output are already distinguishable message types),
        which is a separate change. Until then the 13k-token COMPACT_BUFFER
        absorbs the gap. Flip this assertion when that lands.
        """
        tiktoken = pytest.importorskip("tiktoken")
        actual = len(tiktoken.get_encoding("cl100k_base").encode(text))
        assert default_token_estimate(text) < actual

    def test_estimate_tokens_reflects_cjk(self):
        msgs = [UserMessage("你好世界" * 100)]
        assert estimate_tokens(msgs) == int(400 * CJK_TOKENS_PER_CHAR)

    def test_estimate_tokens_uses_the_api_anchor(self):
        from cave_agent.compaction.tokens import TokenAnchor

        msgs = [UserMessage("hi")]
        assert estimate_tokens(msgs, TokenAnchor(999, estimate_tokens(msgs))) == 999


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
                _gen("ab", stall_after=1, stall=1.0),
                idle_timeout=0.1,
            ):
                got.append(x)
        assert got == ["a"]  # first item delivered, then the stall trips

    def test_agent_defaults_and_override(self):
        assert CaveAgent(model=_DummyModel()).stream_idle_timeout == 120.0
        assert (
            CaveAgent(
                model=_DummyModel(),
                stream_idle_timeout=None,
            ).stream_idle_timeout
            is None
        )


class TestUsageTotalIsDerived:
    """A provider may report the parts without the sum."""

    def test_a_missing_total_is_computed(self):
        from types import SimpleNamespace

        from cave_agent.models.base import Model

        usage = SimpleNamespace(prompt_tokens=1234, completion_tokens=77, total_tokens=0)

        extracted = Model._extract_token_usage(SimpleNamespace(usage=usage))

        assert extracted.prompt_tokens == 1234
        assert extracted.completion_tokens == 77
        assert extracted.total_tokens == 1311

    def test_a_reported_total_is_trusted(self):
        from types import SimpleNamespace

        from cave_agent.models.base import Model

        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=99)

        assert Model._extract_token_usage(SimpleNamespace(usage=usage)).total_tokens == 99
