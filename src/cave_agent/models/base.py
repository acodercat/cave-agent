import asyncio
import contextlib
import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Self

from .errors import ModelError, StreamStalledError, raise_model_error
from .retry import MAX_RETRIES, get_retry_delay, is_retryable, with_retry_typed


@dataclass
class TokenUsage:
    """Token usage statistics from an LLM API call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage objects together."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ModelResponse:
    """Response from an LLM model call including token usage."""

    content: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str | None = None
    # Provider-neutral reasoning text (OpenAI ``reasoning`` / Moonshot & Qwen &
    # DeepSeek ``reasoning_content``). ``None`` when the model isn't a reasoning
    # model. Captured so it isn't silently dropped; not sent back on the wire.
    thinking: str | None = None
    # The model's refusal, kept separate from ``content``. A caller that wants
    # an answer can show it; one that wants a *product* — a summary, a code
    # block — must be able to tell that no product was made.
    refusal: str | None = None


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


class StreamResponse:
    """Async iterator over a provider's streamed tokens.

    Subclasses implement :meth:`_open_stream` to create the raw provider stream;
    this base owns everything else — driving iteration, retrying a transport
    failure that happens *before any semantic output is received* (a retry
    after that would replay content or refusal), and closing the underlying stream via
    :meth:`aclose`. Each step yields a :class:`StreamDelta` carrying answer
    ``content`` and/or reasoning ``thinking`` in separate fields, so reasoning
    never reaches the code-block parser. ``usage`` and ``finish_reason`` update
    as side-channel attributes readable after iteration; ``thinking`` and
    ``refusal`` accumulate the full reasoning trace and any refusal text, both
    of which are prose rather than something to parse for code.
    """

    def __init__(self):
        self.usage = TokenUsage()
        self.finish_reason: str | None = None
        self.thinking: str = ""
        self.refusal: str = ""
        self._response: Any = None
        self._iterator: AsyncIterator[Any] | None = None
        self._received_output: bool = False
        self._connect_attempts: int = 0

    async def _open_stream(self) -> AsyncIterator[Any]:
        """Create and return the raw provider stream (an async iterable).

        Should NOT retry — the base :meth:`__anext__` loop retries opening and
        first-chunk reads together under a single budget.
        """
        raise NotImplementedError

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> StreamDelta:
        while True:
            try:
                iterator = self._iterator
                if iterator is None:
                    response = await self._open_stream()
                    self._response = response
                    iterator = response.__aiter__()
                    self._iterator = iterator
                chunk = await iterator.__anext__()
            except StopAsyncIteration:
                raise
            except Exception as error:
                # A provider terminal reason is authoritative: no content may
                # follow it. Some APIs send an optional usage-only event after
                # that terminal chunk, so we keep reading for accounting, but
                # losing that trailing event does not make the answer partial.
                # The agent estimates usage when the event never arrives.
                if self.finish_reason is not None:
                    raise StopAsyncIteration from None
                # Transport failure opening or reading the stream. Retry only
                # before semantic output arrived — a later retry would replay
                # content or refusal. Errors are translated to the typed
                # hierarchy regardless (so overflow/billing route correctly);
                # a non-retryable one propagates at once.
                if (
                    self._received_output
                    or self._connect_attempts >= MAX_RETRIES
                    or not is_retryable(error)
                ):
                    # The retry *decision* reads the raw error (a provider's
                    # own status codes); only the raise is typed.
                    raise_model_error(error)
                self._connect_attempts += 1
                # Best-effort cleanup of a stream we are already discarding.
                # We are retrying *because* the transport failed, so closing it
                # usually fails too — and that failure carries no information
                # the retry decision above didn't already have. Letting it
                # propagate would turn every recoverable connect error whose
                # socket also dies on close into a terminal one.
                with contextlib.suppress(ModelError):
                    await self.aclose()
                await asyncio.sleep(get_retry_delay(self._connect_attempts, error))
                continue
            try:
                refusal_length = len(self.refusal)
                delta = self._process_stream_chunk(chunk)
            except Exception as error:
                # Inside the boundary, because a provider's own message shapes
                # are as much a provider concern as its transport. Left outside,
                # a chunk this parser did not expect ended the run as
                # INTERNAL_ERROR — "the agent has a defect" — and re-raised an
                # untyped exception, against this layer's stated totality.
                raise_model_error(error)
            if len(self.refusal) > refusal_length:
                # Refusal text is semantic output even though it deliberately
                # bypasses the code parser. Reopening the request after receiving
                # it would replay and duplicate the model's answer.
                self._received_output = True
            if delta is not None:
                self._received_output = True
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
            # Do not return: several providers put usage and the finish reason
            # in the same terminal event (Anthropic's ``message_delta`` is one),
            # and compatible endpoints may attach the final delta as well.
            # Treat the fields independently so accounting cannot erase the
            # signal that says whether the answer was complete.
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            if choice.finish_reason:
                self.finish_reason = choice.finish_reason
            delta = choice.delta
            if delta is None:
                # A chunk carrying no delta is metadata, not content — Azure's
                # asynchronous content filter sends annotations this way. It
                # says nothing about the answer, so there is nothing to yield.
                return None
            reasoning = getattr(delta, "reasoning", None) or getattr(
                delta, "reasoning_content", None
            )
            refusal = getattr(delta, "refusal", None)
            content = delta.content
            if reasoning:
                self.thinking += reasoning
            if refusal:
                # A refusal is the model's answer, but it is prose, not a
                # response the code parser should see — so it accumulates here
                # like reasoning rather than arriving as content.
                self.refusal += refusal
            if reasoning or content:
                return StreamDelta(content=content, thinking=reasoning)
        return None

    async def aclose(self) -> None:
        """Release the underlying provider stream and its socket.

        Idempotent and safe whether or not the stream was ever opened. Called
        by the agent when it stops reading early (e.g. once a code block is
        complete) — which the SDK's read-to-completion auto-close would miss —
        and between connect-retry attempts.

        A failure here is classified like any other provider failure for direct
        callers. The agent treats this method as per-turn cleanup and logs that
        classified failure instead: a socket reset while closing cannot
        invalidate an answer already declared complete or replace the read
        failure being cleaned up. ``CancelledError`` is not caught: it is the
        caller unwinding, not the provider failing.
        """
        response, self._response = self._response, None
        self._iterator = None
        if response is None:
            return
        closer = getattr(response, "aclose", None) or getattr(response, "close", None)
        if closer is None:
            return
        try:
            outcome = closer()
            if inspect.isawaitable(outcome):
                await outcome
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise_model_error(error)


class Model(ABC):
    """
    Abstract base class for language model engines.
    Defines interface for interacting with different LLM providers.

    ``filters_asynchronously`` declares that this provider may send its content
    filter verdict *after* the content it applies to — Azure OpenAI's
    asynchronous filter does. The agent normally stops reading at the first
    complete code block and runs it; against such a provider that would execute
    code the filter was about to reject, so it reads to the end of the stream
    first. Off by default: the wait costs a whole generation, and providers that
    filter before streaming never need it.
    """

    max_output_tokens: int | None = None
    """Completion-token ceiling this model is configured to request.

    Declaring it lets :class:`~cave_agent.compaction.Compactor` size its
    compaction threshold against the real wire constraint
    (``input + max_output <= context_window``) instead of guessing. ``None``
    means "provider default", and the compactor falls back to a conservative
    assumption. Set it on the model, not the compactor, so the two cannot
    drift apart.
    """

    filters_asynchronously: bool = False
    """Whether this provider may send its filter verdict after the content."""

    @staticmethod
    def _extract_token_usage(response: Any) -> TokenUsage:
        """Extract token usage from an LLM API response.

        The total is derived when the provider reports the parts but not the
        sum. Read verbatim, a missing total made the whole record look absent —
        the agent replaced two authoritative counts with its character
        heuristic, which then also seeded the compaction anchor.
        """
        if hasattr(response, "usage") and response.usage:
            prompt = getattr(response.usage, "prompt_tokens", 0) or 0
            completion = getattr(response.usage, "completion_tokens", 0) or 0
            return TokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=(getattr(response.usage, "total_tokens", 0) or 0)
                or (prompt + completion),
            )
        return TokenUsage()

    @classmethod
    def _build_response(cls, response: Any) -> ModelResponse:
        """Assemble a :class:`ModelResponse` from an OpenAI-shaped payload.

        Shared so the policy it encodes — a refusal stands in for absent
        content, while remaining separately readable — is stated once. Held in
        each provider, it was two copies of one decision.
        """
        content, finish_reason, refusal = cls._extract_response(response)
        return ModelResponse(
            content=content or refusal or "",
            usage=cls._extract_token_usage(response),
            finish_reason=finish_reason,
            thinking=cls._extract_thinking(response),
            refusal=refusal,
        )

    @staticmethod
    def _extract_response(response: Any) -> tuple[str, str | None, str | None]:
        """Extract ``(content, finish_reason, refusal)`` from an OpenAI response.

        The refusal is returned alongside rather than folded into ``content``:
        as an *answer* it is what the user should see, but as a *product* it is
        a failure, and only the caller knows which it needs.
        """
        content = ""
        finish_reason = refusal = None
        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            refusal = getattr(message, "refusal", None)
            content = message.content or ""
            finish_reason = response.choices[0].finish_reason
        return content, finish_reason, refusal

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
            return getattr(message, "reasoning", None) or getattr(
                message, "reasoning_content", None
            )
        return None

    async def call(self, messages: list[dict[str, str]]) -> ModelResponse:
        """Generate a response, with retry and typed errors.

        Concrete on purpose: providers implement :meth:`_complete` and this
        owns the boundary, mirroring how :class:`StreamResponse` owns
        everything but ``_open_stream``. Parsing runs *inside* the retried,
        typed operation — a malformed response body is a provider failure, and
        parsing it outside this boundary is how an ``AttributeError`` used to
        escape the model layer untyped.
        """
        return await with_retry_typed(lambda: self._complete(messages))

    @abstractmethod
    async def _complete(self, messages: list[dict[str, str]]) -> ModelResponse:
        """Issue one non-streaming request and parse the reply.

        Must NOT retry — :meth:`call` owns that. Raise whatever the SDK
        raises; it is classified at the boundary.
        """
        ...

    @abstractmethod
    def stream(self, messages: list[dict[str, str]]) -> StreamResponse:
        """Stream response tokens from message history asynchronously.

        Returns:
            A StreamResponse that yields tokens and provides usage after exhaustion.
        """
        ...

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


async def stream_with_idle_timeout[StreamT](
    stream: AsyncIterator[StreamT],
    idle_timeout: float | None,
) -> AsyncIterator[StreamT]:
    """Re-yield items from *stream*, enforcing a sliding idle timeout.

    If more than *idle_timeout* seconds pass between two consecutive items,
    raise ``TimeoutError``. Once the stream records a terminal finish reason,
    the same timeout ends iteration normally because only optional accounting
    may still arrive. The window resets on every item, so a stream that keeps
    producing never trips it. ``idle_timeout=None`` disables the watchdog
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
            # A terminal provider reason is authoritative. We keep reading
            # after it only for an optional usage event, so a timeout here
            # loses accounting data, not the completed response.
            if getattr(stream, "finish_reason", None) is not None:
                return
            raise StreamStalledError(
                f"LLM stream stalled: no chunk received for {idle_timeout:.0f}s"
            ) from exc
        yield item
