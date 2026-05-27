"""API retry with exponential backoff and jitter.

Wraps async operations with automatic retry for transient API errors
(rate limits, server errors, connection failures). Respects Retry-After
headers when present.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_RETRIES = 5
BASE_DELAY = 0.5   # seconds
MAX_DELAY = 32.0    # seconds

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504, 529}


async def with_retry(
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
                attempt + 1, max_retries, delay, error,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("Unreachable")


def is_retryable(error: Exception) -> bool:
    """Check whether an API error is worth retrying."""
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status and status in RETRYABLE_STATUS_CODES:
        return True
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
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

    base = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    jitter = random.uniform(0, 0.25 * base)
    return base + jitter


def _get_retry_after(error: Exception) -> float | None:
    """Extract Retry-After header value in seconds, if present."""
    headers = getattr(error, "headers", None)
    if headers is None:
        return None
    retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
    if retry_after is None:
        return None
    try:
        return float(retry_after)
    except (ValueError, TypeError):
        return None
