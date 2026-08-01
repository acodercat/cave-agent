"""Shared test doubles.

``FakeModel`` replays a scripted list of responses, so agent-loop tests run
offline and deterministically. It implements the real ``Model`` interface
(including streaming) rather than a mock, so a change to that interface breaks
these tests instead of silently passing.
"""

from __future__ import annotations

from types import SimpleNamespace

from cave_agent.models import (
    Model,
    ModelResponse,
    StreamResponse,
    TokenUsage,
)


class FakeStream(StreamResponse):
    """Streams a fixed string in small chunks."""

    def __init__(
        self,
        text: str,
        usage: TokenUsage | None,
        finish_reason: str = "stop",
        thinking: str = "",
        chunk_size: int = 13,
    ):
        super().__init__()
        self._text = text
        self._thinking = thinking
        self._chunk_size = chunk_size
        # Deliberately NOT set here. Real providers deliver usage and
        # finish_reason in a terminal chunk *after* the content, so a consumer
        # that stops reading early never sees them. Setting them up front made
        # the fake more forgiving than reality and hid a live bug.
        self._final_usage = usage
        self._final_finish = finish_reason

    async def _open_stream(self):
        async def gen():
            for i in range(0, len(self._thinking), self._chunk_size):
                yield SimpleNamespace(
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(
                                content=None,
                                refusal=None,
                                reasoning=self._thinking[i : i + self._chunk_size],
                                reasoning_content=None,
                            ),
                        )
                    ],
                )
            for i in range(0, len(self._text), self._chunk_size):
                yield SimpleNamespace(
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(
                                content=self._text[i : i + self._chunk_size],
                                refusal=None,
                                reasoning=None,
                                reasoning_content=None,
                            ),
                        )
                    ],
                )
            # A deliberately adversarial but real provider shape: when usage is
            # supplied, it shares Anthropic's terminal message_delta with the
            # stop reason. The default omits it so agent tests exercise the
            # production estimation fallback.
            yield SimpleNamespace(
                usage=self._final_usage,
                choices=[
                    SimpleNamespace(
                        finish_reason=self._final_finish,
                        delta=SimpleNamespace(
                            content=None,
                            refusal=None,
                            reasoning=None,
                            reasoning_content=None,
                        ),
                    )
                ],
            )

        return gen()


class FakeModel(Model):
    """Replays scripted responses, falling back to a plain answer when spent."""

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        usage: TokenUsage | None = None,
        finish_reason: str = "stop",
        thinking: str = "",
        fallback: str = "All done.",
        max_output_tokens: int | None = None,
        finish_reasons: list[str] | None = None,
    ):
        self.responses = list(responses or [])
        # Per-response finish reasons. A single global reason cannot express
        # "truncated, then completed", which is exactly the recovery path.
        self.finish_reasons = list(finish_reasons or [])
        self.usage = usage
        self.finish_reason = finish_reason
        self.thinking = thinking
        self.fallback = fallback
        self.max_output_tokens = max_output_tokens
        self.calls: list[list[dict]] = []

    def _next(self, messages) -> str:
        self.calls.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return self.fallback

    def _next_finish_reason(self) -> str:
        if self.finish_reasons:
            return self.finish_reasons.pop(0)
        return self.finish_reason

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def _complete(self, messages):
        return ModelResponse(
            content=self._next(messages),
            usage=self.usage or TokenUsage(),
            finish_reason=self.finish_reason,
        )

    def stream(self, messages):
        text = self._next(messages)
        return FakeStream(
            text,
            self.usage,
            finish_reason=self._next_finish_reason(),
            thinking=self.thinking,
        )


class FailingModel(Model):
    """Raises *error* on every call — for circuit-breaker / error-path tests."""

    def __init__(self, error: Exception | None = None):
        self.error = error or ValueError("boom")
        self.call_count = 0

    async def _complete(self, messages):
        self.call_count += 1
        raise self.error

    def stream(self, messages):
        self.call_count += 1
        raise self.error
