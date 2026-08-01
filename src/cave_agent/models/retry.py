"""API retry with exponential backoff and jitter.

Wraps async operations with automatic retry for transient API errors
(rate limits, server errors, connection failures). Respects Retry-After
headers when present.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

from .errors import is_billing_exhausted, raise_model_error

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BASE_DELAY = 0.5  # seconds
MAX_DELAY = 32.0  # seconds

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504, 529}


async def with_retry[T](
    operation: Callable[[], Awaitable[T]],
    max_retries: int = MAX_RETRIES,
) -> T:
    """Execute an async operation with exponential backoff retry.

    Retries on rate limits (429), server errors (5xx), timeouts (408),
    and connection errors. Respects Retry-After headers.
    """
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception as error:
            if attempt == max_retries or not is_retryable(error):
                raise
            delay = get_retry_delay(attempt, error)
            logger.warning(
                "API error (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                max_retries,
                delay,
                error,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("Unreachable")


async def with_retry_typed[T](
    operation: Callable[[], Awaitable[T]],
    max_retries: int = MAX_RETRIES,
) -> T:
    """``with_retry`` that translates a surviving raw provider error into a
    typed :class:`~cave_agent.models.errors.ModelError`.

    A context-length overflow becomes ``PromptTooLongError`` (so the agent can
    recover by compaction) and a billing exhaustion becomes
    ``ProviderBillingError``. Everything else becomes ``ProviderError`` —
    classification is total, so nothing leaves this layer untyped.
    """
    try:
        return await with_retry(operation, max_retries=max_retries)
    except Exception as error:
        raise_model_error(error)


def is_retryable(error: Exception) -> bool:
    """Check whether an API error is worth retrying."""
    # Billing exhaustion can ride a retryable status (OpenAI signals
    # ``insufficient_quota`` on a 429), but a top-up — not time — fixes it,
    # so never burn retry budget on it.
    if is_billing_exhausted(error):
        return False
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if isinstance(status, int):
        # A definite HTTP status is authoritative: retry 429/5xx, but do NOT
        # fall through to the message heuristic for e.g. a 400 whose text
        # merely mentions "timeout" (an invalid-parameter error, not transient).
        return status in RETRYABLE_STATUS_CODES
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    # Some SDKs name the condition only in the class. OpenAI's
    # ``APITimeoutError`` has no status, is not a builtin ``TimeoutError``, and
    # says "Request timed out." — so neither check above nor the message
    # heuristic below matched, and a client timeout ended the run on its first
    # occurrence while its sibling ``APIConnectionError`` was retried.
    if any(marker in type(error).__name__.lower() for marker in ("timeout", "connection")):
        return True
    # The message tier is a last resort for SDK errors the tiers above missed,
    # so it must not apply to Python's own programming errors: a TypeError for
    # a misspelled kwarg that happens to be named `request_timeout` is
    # deterministic, and retrying it spent the whole backoff budget before
    # surfacing what was an immediate, permanent failure.
    if isinstance(error, (TypeError, ValueError, AttributeError, LookupError, NameError)):
        return False
    message = str(error).lower()
    if "connection" in message or "timeout" in message:
        return True
    return False


def get_retry_delay(attempt: int, error: Exception | None = None) -> float:
    """Calculate delay with exponential backoff + jitter.

    Sequence: 0.5s, 1s, 2s, 4s, 8s, 16s, 32s (capped).
    Jitter: 0-25% of base delay to avoid thundering herd.
    """
    if error is not None:
        retry_after = _get_retry_after(error)
        if retry_after is not None:
            return min(retry_after, MAX_DELAY)

    base = min(BASE_DELAY * (2**attempt), MAX_DELAY)
    jitter = random.uniform(0, 0.25 * base)
    return base + jitter


def _get_retry_after(error: Exception) -> float | None:
    """Extract Retry-After header value in seconds, if present.

    Looked for on the error *and* on its response: the OpenAI SDK carries
    headers only at ``error.response.headers``, so reading ``error.headers``
    alone found nothing on that path and every 429 was backed off on the
    generic schedule while the server had said exactly how long to wait.
    LiteLLM's wrappers do expose ``.headers``, which is what hid it.
    """
    headers = getattr(error, "headers", None)
    if headers is None:
        headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is None:
        return None
    retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except (ValueError, TypeError):
        return None
