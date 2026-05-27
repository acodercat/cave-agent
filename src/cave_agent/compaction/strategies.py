"""Compaction strategies — microcompact, full compact, and trim fallback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..types import (
    MessageRole, ExecutionResultMessage, AssistantMessage, UserMessage,
)
from .prompts import (
    COMPACT_SYSTEM_PROMPT,
    COMPACT_USER_PROMPT,
    MICROCOMPACT_PLACEHOLDER,
    extract_summary,
    format_transcript,
)
from .state import CompactionState

if TYPE_CHECKING:
    from ..models import Model

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 3
KEEP_RECENT_EXECUTION_RESULTS = 6
MIN_KEEP_MESSAGES = 4


def microcompact(messages: list) -> list:
    """Clear old execution results, keeping the most recent ones intact.

    No LLM call — instant, zero cost.
    """
    result_indices = [
        i for i, msg in enumerate(messages)
        if msg.role == MessageRole.EXECUTION_RESULT
    ]

    if len(result_indices) <= KEEP_RECENT_EXECUTION_RESULTS:
        return messages

    to_clear = set(result_indices[:-KEEP_RECENT_EXECUTION_RESULTS])
    result = []
    for i, msg in enumerate(messages):
        if i in to_clear:
            result.append(ExecutionResultMessage(MICROCOMPACT_PLACEHOLDER))
        else:
            result.append(msg)
    return result


async def full_compact(
    messages: list,
    model: Model,
    state: CompactionState,
) -> list:
    """LLM-powered summarization with circuit breaker protection."""
    if state.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        logger.warning("Circuit breaker active — skipping full compact")
        return trim_fallback(messages)

    keep_count = max(len(messages) // 2, MIN_KEEP_MESSAGES)
    to_summarize = messages[:-keep_count]
    to_keep = messages[-keep_count:]

    if not to_summarize:
        return messages

    try:
        summary = await _generate_summary(to_summarize, model)
        state.consecutive_failures = 0
        return _build_summary_result(summary, to_keep)
    except Exception:
        state.consecutive_failures += 1
        logger.warning(
            "Full compact failed (%d/%d) — falling back to trim",
            state.consecutive_failures, MAX_CONSECUTIVE_FAILURES,
            exc_info=True,
        )
        return trim_fallback(messages)


def trim_fallback(messages: list) -> list:
    """Last resort — keep the most recent half of messages."""
    keep_count = max(len(messages) // 2, MIN_KEEP_MESSAGES)
    trimmed = messages[-keep_count:]
    return _strip_leading_execution_results(trimmed)


def _build_summary_result(summary: str, to_keep: list) -> list:
    return [
        UserMessage(f"[Previous conversation summary]\n{summary}"),
        AssistantMessage(
            "Understood. I have the context from our previous conversation "
            "and I'm ready to continue."
        ),
        *to_keep,
    ]


async def _generate_summary(messages: list, model: Model) -> str:
    transcript = format_transcript(messages)
    summary_messages = [
        {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{transcript}\n\n{COMPACT_USER_PROMPT}"},
    ]

    response = await model.call(summary_messages)
    text = extract_summary(response.content)
    if not text:
        raise ValueError("Model returned empty summary")
    return text


def _strip_leading_execution_results(messages: list) -> list:
    """Remove leading execution result messages without preceding context."""
    start = 0
    while start < len(messages) and messages[start].role == MessageRole.EXECUTION_RESULT:
        start += 1
    return messages[start:]
