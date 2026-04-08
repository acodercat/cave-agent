"""Context compaction — multi-tier strategy to keep conversations within token limits.

Public API:
    compact_if_needed  — proactive compaction before each LLM call

Tiers (applied in order):
    1. Microcompact — clear old execution results (no LLM, fast)
    2. Full compact — LLM summarization (circuit breaker protected)
    3. Trim fallback — keep recent messages (last resort)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .state import CompactionState
from .strategies import full_compact, microcompact
from .tokens import compact_threshold, estimate_tokens

if TYPE_CHECKING:
    from ..models import Model

logger = logging.getLogger(__name__)

__all__ = [
    "CompactionState",
    "compact_if_needed",
]


async def compact_if_needed(
    messages: list,
    model: Model,
    state: CompactionState,
    context_window: int = 128_000,
) -> tuple[list, str | None]:
    """Apply microcompact first, then full compact if still over threshold.

    Preserves messages[0] (system message), compacts only the body.

    Returns:
        (messages, tier) where tier is None, "microcompact", or "full_compact".
    """
    threshold = compact_threshold(context_window)
    if estimate_tokens(messages) <= threshold:
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
