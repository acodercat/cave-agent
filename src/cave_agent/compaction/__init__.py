"""Context compaction — multi-tier strategy to keep conversations in budget.

All compaction flows through :class:`Compactor`, which owns the summarization
model, the budget config and the per-session circuit-breaker state. Tiers, in
escalating order of cost:

1. **Microcompact** — clear old execution results (no LLM, instant)
2. **Summarize** — LLM-summarize the older half (circuit-breaker protected,
   and incremental: a prior summary is carried verbatim rather than
   re-summarized)

Plus :meth:`Compactor.recover_from_overflow`, the reactive path taken once the
API has already rejected an oversize prompt. Summary failures preserve the
unsummarized history rather than trimming it.
"""

from .compactor import (
    DEFAULT_CONTEXT_WINDOW,
    Compactor,
    migrate_legacy_summaries,
)
from .state import CompactionState
from .tokens import (
    TokenAnchor,
    compact_threshold,
    default_token_estimate,
    estimate_tokens,
    tiktoken_estimator,
)

__all__ = [
    "Compactor",
    "TokenAnchor",
    "CompactionState",
    "DEFAULT_CONTEXT_WINDOW",
    "compact_threshold",
    "default_token_estimate",
    "estimate_tokens",
    "migrate_legacy_summaries",
    "tiktoken_estimator",
]
