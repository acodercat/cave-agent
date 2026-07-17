import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, List, Dict, TypeVar
from dataclasses import dataclass, field

from .errors import classify_provider_error
from .retry import MAX_RETRIES, get_retry_delay, is_retryable


@dataclass
class TokenUsage:
    """Token usage statistics from an LLM API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add two TokenUsage objects together."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class ModelResponse:
    """Response from an LLM model call including token usage."""
    content: str
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str | None = None
    # Provider-neutral reasoning text (OpenAI ``reasoning`` / Moonshot & Qwen &
    # DeepSeek ``reasoning_content``). ``None`` when the model isn't a reasoning
    # model. Captured so it isn't silently dropped; not sent back on the wire.
    thinking: str | None = None


@dataclass
class StreamDelta:
    """One streamed increment from a model: answer ``content`` and/or model
    ``thinking`` (reasoning) text — usually exactly one is set.

    This is what :class:`StreamResponse` yields. Terminal signals (token
    ``usage``, ``finish_reason``) are not deltas; they update the
    :class:`StreamResponse` attributes and are read after iteration.
    """
    content: str | None = None
    thinking: str | None = None


class StreamResponse(ABC):
    """Async iterator over a provider's streamed tokens.

    Subclasses implement :meth:`_open_stream` to create the raw provider stream;
    this base owns everything else — driving iteration, retrying a transport
    failure that happens *before any delta has reached the consumer* (a retry
    after that would replay content), and closing the underlying stream via
    :meth:`aclose`. Each step yields a :class:`StreamDelta` carrying answer
    ``content`` and/or reasoning ``thinking`` in separate fields, so reasoning
    never reaches the code-block parser. ``usage`` and ``finish_reason`` update
    as side-channel attributes readable after iteration, and ``thinking`` also
    accumulates the full reasoning trace.
    """

    def __init__(self):
        self.usage = TokenUsage()
        self.finish_reason: str | None = None
        self.thinking: str = ""
        self._response: Any = None
        self._iterator: AsyncIterator | None = None
        self._yielded_any: bool = False
        self._connect_attempts: int = 0

    async def _open_stream(self) -> AsyncIterator:
        """Create and return the raw provider stream (an async iterable).

        Should NOT retry — the base :meth:`__anext__` loop retries opening and
        first-chunk reads together under a single budget.
        """
        raise NotImplementedError

    def __aiter__(self) -> "StreamResponse":
        return self

    async def __anext__(self) -> StreamDelta:
        while True:
            try:
                if self._iterator is None:
                    self._response = await self._open_stream()
                    self._iterator = self._response.__aiter__()
                chunk = await self._iterator.__anext__()
            except StopAsyncIteration:
                raise
            except Exception as error:
                # Transport failure opening or reading the stream. Retry only
                # before any delta reached the consumer — a later retry would
                # replay already-streamed content. Errors are translated to the
                # typed hierarchy regardless (so overflow/billing route
                # correctly); a non-retryable one propagates at once.
                typed = classify_provider_error(error)
                if (
                    self._yielded_any
                    or self._connect_attempts >= MAX_RETRIES
                    or not is_retryable(error)
                ):
                    if typed is error:
                        raise
                    raise typed from error
                self._connect_attempts += 1
                await self.aclose()
                await asyncio.sleep(get_retry_delay(self._connect_attempts, error))
                continue
            delta = self._process_stream_chunk(chunk)
            if delta is not None:
                self._yielded_any = True
                return delta

    def _process_stream_chunk(self, chunk: Any) -> StreamDelta | None:
        """Turn one raw provider chunk into a :class:`StreamDelta` (answer
        ``content`` and/or reasoning ``thinking``), or ``None`` for a terminal
        or empty chunk.

        Terminal signals update side-channel attributes instead of yielding:
        token ``usage`` and ``finish_reason``. Reasoning is also accumulated
        onto ``self.thinking`` so the full trace is readable after the stream.
        """
        if getattr(chunk, "usage", None):
            self.usage = Model._extract_token_usage(chunk)
            return None
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            if choice.finish_reason:
                self.finish_reason = choice.finish_reason
                return None
            delta = choice.delta
            reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
            content = delta.content
            if reasoning:
                self.thinking += reasoning
            if reasoning or content:
                return StreamDelta(content=content, thinking=reasoning)
        return None

    async def aclose(self) -> None:
        """Release the underlying provider stream and its socket.

        Idempotent and safe whether or not the stream was ever opened. Called
        by the agent when it stops reading early (e.g. once a code block is
        complete) — which the SDK's read-to-completion auto-close would miss —
        and between connect-retry attempts.
        """
        response, self._response = self._response, None
        self._iterator = None
        if response is None:
            return
        closer = getattr(response, "aclose", None) or getattr(response, "close", None)
        if closer is None:
            return
        outcome = closer()
        if inspect.isawaitable(outcome):
            await outcome


class Model(ABC):
    """
    Abstract base class for language model engines.
    Defines interface for interacting with different LLM providers.
    """

    @staticmethod
    def _extract_token_usage(response: Any) -> TokenUsage:
        """Extract token usage from an LLM API response."""
        if hasattr(response, "usage") and response.usage:
            return TokenUsage(
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(response.usage, "total_tokens", 0) or 0,
            )
        return TokenUsage()

    @staticmethod
    def _extract_response(response: Any) -> tuple[str, str | None]:
        """Extract content and finish_reason from an OpenAI-style response."""
        content = ""
        finish_reason = None
        if hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
        return content, finish_reason

    @staticmethod
    def _extract_thinking(response: Any) -> str | None:
        """Extract reasoning text from an OpenAI-style non-streaming response.

        Reasoning models surface it on ``message.reasoning`` (OpenAI) or
        ``message.reasoning_content`` (Moonshot / Qwen / DeepSeek); both are
        outside the typed SDK shape, so probe with ``getattr``. ``None`` when
        absent.
        """
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            return getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        return None

    @abstractmethod
    async def call(self, messages: List[Dict[str, str]]) -> ModelResponse:
        """Generate response from message history asynchronously.

        Returns:
            ModelResponse containing content and token usage.
        """
        pass

    @abstractmethod
    def stream(self, messages: List[Dict[str, str]]) -> StreamResponse:
        """Stream response tokens from message history asynchronously.

        Returns:
            A StreamResponse that yields tokens and provides usage after exhaustion.
        """
        pass

    async def aclose(self) -> None:
        """Release resources held by the model (e.g. an HTTP connection pool).

        Default is a no-op; providers that own a client override it. Call when
        tearing down a long-lived model, or use the model as an async context
        manager. Safe to call more than once.
        """
        return None

    async def __aenter__(self) -> "Model":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


_StreamT = TypeVar("_StreamT")


async def stream_with_idle_timeout(
    stream: AsyncIterator[_StreamT],
    idle_timeout: float | None,
) -> AsyncIterator[_StreamT]:
    """Re-yield items from *stream*, enforcing a sliding idle timeout.

    If more than *idle_timeout* seconds pass between two consecutive items,
    raise ``TimeoutError``. The window resets on every item, so a stream that
    keeps producing never trips it. ``idle_timeout=None`` disables the watchdog
    (plain pass-through).

    Closes a liveness gap transport timeouts miss: proxy keep-alive comments
    and provider ``ping`` events keep the socket readable — perpetually
    resetting any read timeout — while no chunk is produced, so the consumer
    would otherwise wait forever. Keying the window off *item* progress rather
    than socket bytes is the signal that works across providers.
    ``asyncio.timeout`` re-raises a caller's ``CancelledError`` unchanged and
    converts only its own elapsed deadline.
    """
    if idle_timeout is None:
        async for item in stream:
            yield item
        return

    iterator = stream.__aiter__()
    while True:
        try:
            async with asyncio.timeout(idle_timeout):
                item = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise TimeoutError(
                f"LLM stream stalled: no chunk received for {idle_timeout:.0f}s"
            ) from exc
        yield item
