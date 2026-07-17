"""Context compaction — multi-tier strategy to keep conversations within token limits.

Public API:
    compact_if_needed    — proactive compaction before each LLM call
    full_compact_needed  — cheap predictor of whether the LLM tier will fire
    recover_from_overflow — reactive recovery after the API rejects an oversize prompt

Tiers (applied in order):
    1. Microcompact — clear old execution results (no LLM, fast)
    2. Full compact — LLM summarization (circuit breaker protected)
    3. Trim fallback — keep recent messages (last resort)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .state import CompactionState
from .strategies import aggressive_recover, full_compact, microcompact
from .tokens import compact_threshold, estimate_tokens

if TYPE_CHECKING:
    from ..models import Model

logger = logging.getLogger(__name__)

__all__ = [
    "CompactionState",
    "compact_if_needed",
    "full_compact_needed",
    "recover_from_overflow",
]


async def recover_from_overflow(messages: list, model: Model) -> list:
    """Reactive emergency compaction after the API reports the prompt is too long.

    Preserves ``messages[0]`` (the system message) and aggressively compacts the
    body (keep most-recent quarter, summarize the rest). Unconditional — the API
    already rejected the request, so there's nothing to re-check. Meant to be
    called once, then the model call retried; see ``Agent`` model-call paths.
    """
    system_msg = messages[0] if messages else None
    body = messages[1:] if len(messages) > 1 else []
    body = await aggressive_recover(body, model)
    result = [system_msg, *body] if system_msg else body
    logger.info("Overflow recovery: %d → %d messages", len(messages), len(result))
    return result


def full_compact_needed(
    messages: list,
    context_window: int = 128_000,
    api_token_count: int | None = None,
) -> bool:
    """Cheaply predict whether an (expensive) full compact will be required.

    Runs the free tiers (threshold check + microcompact, both pure) to decide
    whether the LLM summarization tier would fire. Lets callers show a
    "compacting" indicator only when a slow LLM call is actually imminent,
    rather than for every zero-cost microcompact.
    """
    threshold = compact_threshold(context_window)
    if estimate_tokens(messages, api_token_count) <= threshold:
        return False

    system_msg = messages[0] if messages else None
    body = microcompact(messages[1:] if len(messages) > 1 else [])
    compacted = [system_msg, *body] if system_msg else body
    return estimate_tokens(compacted) > threshold


async def compact_if_needed(
    messages: list,
    model: Model,
    state: CompactionState,
    context_window: int = 128_000,
    api_token_count: int | None = None,
) -> tuple[list, str | None]:
    """Apply microcompact first, then full compact if still over threshold.

    Preserves messages[0] (system message), compacts only the body.
    *api_token_count* is the last API-reported prompt-token count, used to
    sharpen the char/4 estimate (see :func:`estimate_tokens`).

    Returns:
        (messages, tier) where tier is None, "microcompact", or "full_compact".
    """
    threshold = compact_threshold(context_window)
    if estimate_tokens(messages, api_token_count) <= threshold:
        return messages, None

    system_msg = messages[0] if messages else None
    body = messages[1:] if len(messages) > 1 else []

    # Tier 1: microcompact
    body = microcompact(body)
    compacted = [system_msg, *body] if system_msg else body
    tokens = estimate_tokens(compacted)
    if tokens <= threshold:
        logger.info("Microcompact sufficient: ~%d tokens", tokens)
        return compacted, "microcompact"

    # Tier 2: full compact
    body = await full_compact(body, model, state)
    compacted = [system_msg, *body] if system_msg else body
    return compacted, "full_compact"
