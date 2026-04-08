"""Token estimation and threshold calculation."""

CHARS_PER_TOKEN = 4
OUTPUT_TOKENS_RESERVE = 16_000
COMPACT_BUFFER_TOKENS = 13_000


def compact_threshold(context_window: int) -> int:
    """Calculate the token count at which compaction should trigger.

    For small context windows where reserves exceed the window size,
    falls back to 50% of the window as the threshold.
    """
    threshold = context_window - OUTPUT_TOKENS_RESERVE - COMPACT_BUFFER_TOKENS
    if threshold <= 0:
        threshold = context_window // 2
    return threshold


def estimate_tokens(
    messages: list,
    api_token_count: int | None = None,
) -> int:
    """Estimate token count, preferring API-reported usage when available."""
    if api_token_count is not None and api_token_count > 0:
        return api_token_count
    return sum(len(msg.content) for msg in messages) // CHARS_PER_TOKEN
