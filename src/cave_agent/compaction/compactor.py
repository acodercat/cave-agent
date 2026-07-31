"""Compactor — the single object that owns all compaction for one agent.

It holds the summarization model, the budget config, and the per-session
circuit-breaker state. All work flows through three public methods, in
escalating order of cost:

1. :meth:`Compactor.microcompact` — no LLM. Clears older execution results and
   reports whether escalation is still needed. Runs silently.
2. :meth:`Compactor.summarize` — LLM summarization of the older half,
   circuit-breaker protected. This is the tier the agent announces to the user.
3. :meth:`Compactor.recover_from_overflow` — emergency path once the API has
   already rejected the prompt. Keeps the most recent quarter, no breaker.

Each method is named for the tier it performs; there is no umbrella ``compact``
that could be confused with the cheap tier it used to name.

Algorithm helpers below the class are module-private; tests target the
observable behaviour of :class:`Compactor`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import IntEnum

from .._placeholders import MICROCOMPACT_PLACEHOLDER, strip_persisted_preview
from ..messages import (
    AssistantMessage,
    ExecutionResultMessage,
    Message,
    MessageRole,
    SummaryAcknowledgementMessage,
    SummaryMessage,
)
from ..models import Model
from ..models.errors import ModelError, PromptTooLongError
from .prompts import (
    COMPACT_SYSTEM_PROMPT,
    COMPACT_UPDATE_USER_TEMPLATE,
    COMPACT_USER_PROMPT,
    COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER,
    COMPACTION_SUMMARY_USER_TEMPLATE,
    extract_summary,
    format_transcript,
    parse_legacy_summary,
)
from .state import MAX_CONSECUTIVE_FAILURES, CompactionState
from .tokens import (
    TokenAnchor,
    compact_threshold,
    default_token_estimate,
    estimate_tokens,
)

logger = logging.getLogger(__name__)


DEFAULT_CONTEXT_WINDOW = 128_000

# Output budget assumed when neither the caller nor the model declares one.
# Models that need more room should set ``max_output_tokens`` so the
# threshold tracks the real request instead of relying on this fallback.
_UNCONFIGURED_OUTPUT_TOKENS_FALLBACK = 16_000


# Execution results kept intact by microcompaction — enough recent context
# for the model to continue without re-running code.
_KEEP_RECENT_EXECUTION_RESULTS = 6

# Never shrink below this many messages, so the tail stays coherent.
_MIN_KEEP_MESSAGES = 4


class _SplitLevel(IntEnum):
    """Denominator for the keep-window: ``keep = len // level``.

    Larger value → smaller keep window → more of the conversation summarized.
    """

    NORMAL = 2  # keep ~half
    AGGRESSIVE = 4  # keep ~quarter (overflow recovery)


class Compactor:
    """Owns compaction state, model and budget for one :class:`CaveAgent`.

    Construct one per agent so circuit-breaker state stays session-scoped.
    The wrapped *model* is both the summarizer and the default source of the
    output reserve that drives :meth:`threshold`.

    A custom *token_estimator* (e.g. ``tiktoken.encoding_for_model(...).encode``
    composed with ``len``) replaces the CJK-aware character heuristic.
    """

    def __init__(
        self,
        model: Model,
        *,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        output_reserve: int | None = None,
        token_estimator: Callable[[str], int] | None = None,
    ) -> None:
        self.model = model
        self.context_window = context_window
        self.output_reserve = output_reserve
        self.token_estimator = token_estimator
        self.state = CompactionState()

    def estimate_tokens(
        self,
        messages: list[Message],
        *,
        anchor: TokenAnchor | None = None,
    ) -> int:
        """Estimate total tokens for *messages*.

        With an *anchor* — a real API count paired with the heuristic reading
        taken at the same moment — the result is that measurement plus the
        heuristic growth since. See :func:`estimate_tokens`.
        """
        return estimate_tokens(messages, anchor, self.token_estimator)

    def anchor(self, messages: list[Message], api_tokens: int) -> TokenAnchor:
        """Pair an API-reported prompt size with the heuristic for *messages*.

        Built here so the two readings always come from the same estimator and
        the same moment — the only way the difference between two later
        readings means anything.
        """
        return TokenAnchor(api_tokens, estimate_tokens(messages, None, self.token_estimator))

    def threshold(self) -> int:
        """Token count above which compaction triggers.

        Reserves room for the next completion so a request prepared just under
        the line still satisfies ``input + max_output <= context_window``.
        Precedence: explicit ``output_reserve`` → the model's own
        ``max_output_tokens`` declaration → a conservative fallback.

        Tested against ``None`` rather than falsiness so an explicit reserve of
        ``0`` ("I have my own budgeting, don't hold anything back") is honoured
        instead of silently falling through to the default.
        """
        reserve = self.output_reserve
        if reserve is None:
            reserve = getattr(self.model, "max_output_tokens", None)
        if reserve is None:
            reserve = _UNCONFIGURED_OUTPUT_TOKENS_FALLBACK
        return compact_threshold(self.context_window, reserve)

    def microcompact(
        self,
        messages: list[Message],
        *,
        anchor: TokenAnchor | None = None,
    ) -> tuple[list[Message], bool]:
        """Run the cheap tier and report whether summarization is still needed.

        Returns ``(messages, needs_summarize)``. Under threshold the input is
        returned untouched with ``False``.
        """
        threshold = self.threshold()
        tokens = self.estimate_tokens(messages, anchor=anchor)
        if tokens <= threshold:
            return messages, False

        system, body = _split_system(messages)
        compacted_body = _microcompact(body)
        compacted = ([system] if system else []) + compacted_body

        tokens = self._tokens_after_microcompact(messages, compacted, tokens)
        if tokens <= threshold:
            logger.info("Microcompact sufficient: ~%d tokens", tokens)
            return compacted, False

        return compacted, True

    def _tokens_after_microcompact(
        self,
        before: list[Message],
        after: list[Message],
        tokens_before: int,
    ) -> int:
        """Token estimate for *after*, anchored to the pre-microcompact count.

        Re-running :meth:`estimate_tokens` here would silently change units:
        *tokens_before* is anchored to a real API count, while a fresh call
        without the anchor returns the bare heuristic, which is routinely much
        lower. The cheap tier would then look successful whenever the heuristic
        under-counts, and summarization would be deferred until the API
        rejected the prompt outright.

        Instead subtract what microcompaction actually saved. The delta uses
        the raw estimator with no padding — under-crediting the savings is the
        safe direction (worst case, one unnecessary summarize).
        """
        if after is before:
            return tokens_before
        count = self.token_estimator or default_token_estimate
        saved = sum(
            count(old.content) - count(new.content)
            for old, new in zip(before, after, strict=True)
            if old is not new
        )
        return max(0, tokens_before - saved)

    async def summarize(self, messages: list[Message]) -> list[Message]:
        """Replace the older half of *messages* with an LLM-generated summary.

        Circuit-breaker protected: after repeated failures it leaves history
        untouched for the next cooldown attempts, then tries the model again.
        Unsummarized turns are never deleted merely because the summarizer is
        unavailable.
        """
        system, body = _split_system(messages)

        if self.state.should_skip():
            logger.warning("Compaction circuit breaker open — preserving history")
            return messages

        to_summarize, to_keep = _split_around_keep_window(body, _SplitLevel.NORMAL)
        if not to_summarize:
            return messages

        try:
            summary = await self._summarize_with_llm(to_summarize)
            self.state.record_success()
        except (ModelError, ValueError, OSError):
            self.state.record_failure()
            logger.warning(
                "Summarization failed (%d/%d) — preserving history",
                self.state.consecutive_failures,
                MAX_CONSECUTIVE_FAILURES,
                exc_info=True,
            )
            return messages

        result = _rejoin(system, _build_summary_messages(summary, to_keep))
        logger.info("Summarized: %d -> %d messages", len(messages), len(result))
        return result

    async def recover_from_overflow(self, messages: list[Message]) -> list[Message]:
        """Aggressive recovery after the API reported the prompt is too long.

        Microcompacts, then summarizes the older three quarters. No circuit
        breaker — this is the last resort before the request fails outright.
        Unconditional: the API has already rejected the prompt, so there is
        nothing left to re-check.
        """
        logger.warning("Emergency compact — API reported prompt too long")

        system, body = _split_system(messages)
        body = _microcompact(body)

        to_summarize, to_keep = _split_around_keep_window(body, _SplitLevel.AGGRESSIVE)
        if not to_summarize:
            return _rejoin(system, body)

        try:
            summary = await self._summarize_with_llm(to_summarize)
        except (ModelError, ValueError, OSError):
            logger.warning(
                "Emergency summarization failed — preserving unsummarized history",
                exc_info=True,
            )
            return _rejoin(system, body)

        result = _rejoin(system, _build_summary_messages(summary, to_keep))
        logger.info("Recovered from overflow: %d -> %d messages", len(messages), len(result))
        return result

    async def _summarize_with_llm(self, messages: list[Message]) -> str:
        """Summarize *messages*, incrementally when a prior summary is present.

        A conversation compacted more than once would otherwise have its own
        summary fed back through the summarizer each time — a telephone game
        that loses fidelity every pass.

        So: if the region already carries a summary, hand that summary over
        VERBATIM and transcribe every complete turn after it, asking the model
        to update rather than re-compress. When nothing followed it, the prior
        summary already *is* the summary of this region — return it and skip
        the LLM call entirely.

        Raises ``ValueError`` on an empty completion so the caller's failure
        accounting treats it like any other summarization failure.
        """
        prior = _find_prior_summary(messages)
        if prior is not None:
            index, prior_body = prior
            entries = _summary_entries(_drop_summary_pair(messages, index))
            if not entries:
                return prior_body
        else:
            prior_body = None
            entries = _summary_entries(messages)

        if not entries:
            raise ValueError("No conversation content to summarize")
        return await self._fold_summary(entries, prior_body)

    async def _fold_summary(
        self,
        entries: list[Message],
        prior_summary: str | None,
    ) -> str:
        """Summarize a fixed region, splitting it until every request fits."""
        request = self._summary_request(entries, prior_summary)
        split = (
            _split_summary_entries(entries) if self._summary_request_too_large(request) else None
        )
        if split is not None:
            left, right = split
            updated = await self._fold_summary(left, prior_summary)
            return await self._fold_summary(right, updated)

        try:
            response = await self.model.call(request)
        except PromptTooLongError:
            split = _split_summary_entries(entries)
            if split is None:
                raise
            left, right = split
            updated = await self._fold_summary(left, prior_summary)
            return await self._fold_summary(right, updated)

        _reject_unusable_summary(response)
        text = extract_summary(response.content or "")
        if not text:
            raise ValueError("Model returned empty summary")
        return text

    def _summary_request(
        self,
        entries: list[Message],
        prior_summary: str | None,
    ) -> list[dict[str, str]]:
        transcript = format_transcript(entries, max_chars_per_msg=None)
        if prior_summary is None:
            user_content = f"{transcript}\n\n{COMPACT_USER_PROMPT}"
        else:
            user_content = COMPACT_UPDATE_USER_TEMPLATE.format(
                prior_summary=prior_summary,
                transcript=transcript,
            )
        return [
            {"role": "system", "content": COMPACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _summary_request_too_large(self, request: list[dict[str, str]]) -> bool:
        reserve = self.output_reserve
        if reserve is None:
            reserve = getattr(self.model, "max_output_tokens", None)
        if reserve is None:
            reserve = _UNCONFIGURED_OUTPUT_TOKENS_FALLBACK
        limit = self.context_window - reserve
        count = self.token_estimator or default_token_estimate
        return sum(count(message["content"]) for message in request) > limit


def _summary_entries(messages: list[Message]) -> list[Message]:
    """Messages that contribute content to a summary request."""
    return [
        message
        for message in messages
        if message.content and message.content != MICROCOMPACT_PLACEHOLDER
    ]


def _split_summary_entries(
    entries: list[Message],
) -> tuple[list[Message], list[Message]] | None:
    """Bisect a summary region, down to splitting one oversized message."""
    if len(entries) > 1:
        midpoint = len(entries) // 2
        return entries[:midpoint], entries[midpoint:]

    if not entries or len(entries[0].content) < 2:
        return None

    message = entries[0]
    midpoint = len(message.content) // 2
    return (
        [Message(message.content[:midpoint], message.role)],
        [Message(message.content[midpoint:], message.role)],
    )


def _reject_unusable_summary(response) -> None:
    """Raise unless *response* is a summary the caller may safely commit.

    Checked against the provider's own signals rather than the text: a refusal
    reads like prose and a truncated summary reads like a summary, so "is it
    non-empty" accepted both — and the circuit breaker recorded a success while
    the conversation lost the turns the summary was supposed to carry.
    """
    if getattr(response, "refusal", None):
        raise ValueError(f"Model refused to summarize: {response.refusal}")
    finish_reason = getattr(response, "finish_reason", None)
    if finish_reason != "stop":
        reason = finish_reason or "missing finish reason"
        raise ValueError(f"Summary did not finish successfully ({reason})")


def _split_system(messages: list[Message]) -> tuple[Message | None, list[Message]]:
    """Peel the leading system message off, if there is one.

    Every tier preserves it: it carries the runtime's function/variable/type
    descriptions, without which the model no longer knows what it can call.
    """
    if messages and messages[0].role == MessageRole.SYSTEM:
        return messages[0], messages[1:]
    return None, list(messages)


def _rejoin(system: Message | None, body: list[Message]) -> list[Message]:
    return [system, *body] if system else body


def _align_split_to_pairs(messages: list[Message], split_idx: int) -> int:
    """Move *split_idx* back so the kept tail never opens on an orphan result.

    An ``ExecutionResultMessage`` at the split has its ``CodeExecutionMessage``
    on the other side, and a result with no preceding code reads as an answer to
    nothing. Deleting the orphan loses it from the summary *and* the kept tail,
    since it sits past the split — the only copy of a computation's output,
    while the code that produced it stays behind describing it.

    Moving the boundary backward keeps the pair together instead, and only ever
    keeps more than asked: at worst compaction reclaims a little less.
    """
    while split_idx > 0 and messages[split_idx].role == MessageRole.EXECUTION_RESULT:
        split_idx -= 1
    return split_idx


def _find_prior_summary(messages: list[Message]) -> tuple[int, str] | None:
    """Locate the newest compaction summary in *messages*.

    Returns ``(index, body)``, or ``None``. Scans newest-first: a rebuild keeps
    at most one summary alive, but a hand-assembled or rehydrated history could
    carry stragglers, and the newest subsumes them.

    A summary is one because of its *type*, and nothing else qualifies. Text
    from a live conversation cannot establish provenance — both participants
    write text — so a ``<=0.8.0`` history is converted by an explicit,
    caller-invoked :func:`migrate_legacy_summaries` rather than being guessed
    at on every compaction.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, SummaryMessage):
            return index, message.body
    return None


def migrate_legacy_summaries(messages: list[Message]) -> list[Message]:
    """Convert ``<=0.8.0`` summary pairs into typed messages. Opt-in.

    Call this once when loading a history you know came from ``<=0.8.0``;
    otherwise that history's summaries are treated as ordinary turns and get
    re-summarized (fidelity lost, nothing corrupted).

    It is not done automatically because the only evidence available in a live
    conversation is text, and text is what users write: an ordinary request
    opening with the old marker, followed by a model reply matching the old
    acknowledgement, is indistinguishable from the real thing. Invoking this
    is the caller asserting where the history came from — which is exactly the
    provenance the bytes cannot carry.
    """
    migrated: list[Message] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        body = (
            parse_legacy_summary(message.content or "")
            if message.role == MessageRole.USER and _followed_by_legacy_ack(messages, index)
            else None
        )
        if body is None:
            migrated.append(message)
            index += 1
            continue
        migrated.append(SummaryMessage(message.content, body))
        migrated.append(SummaryAcknowledgementMessage(COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER))
        index += 2
    return migrated


def _followed_by_legacy_ack(messages: list[Message], index: int) -> bool:
    """Whether a ``<=0.8.0`` acknowledgement follows *index*. Migration only."""
    follower = index + 1
    return (
        follower < len(messages)
        and isinstance(messages[follower], AssistantMessage)
        and messages[follower].content == COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER
    )


def _drop_summary_pair(messages: list[Message], index: int) -> list[Message]:
    """Return *messages* without the summary at *index* and its paired ack.

    The ack is dropped only when it is a :class:`SummaryAcknowledgementMessage`
    — this agent's own synthetic turn. Matching on its *text* deleted a genuine
    assistant message that happened to say the same thing, which is a
    completely ordinary thing for a model to say.
    """
    skip = {index}
    follower = index + 1
    if follower < len(messages) and isinstance(messages[follower], SummaryAcknowledgementMessage):
        skip.add(follower)
    return [m for i, m in enumerate(messages) if i not in skip]


def _microcompact(messages: list[Message]) -> list[Message]:
    """Clear older execution results, keeping the most recent intact.

    A ``<persisted-output>`` marker is not cleared to the generic
    placeholder — only its inline preview is stripped. The declaration line
    names the runtime variable still holding the full output; replacing it
    would strand data the model can no longer reach.
    """
    indices = [
        i
        for i, msg in enumerate(messages)
        if msg.role == MessageRole.EXECUTION_RESULT and not _is_microcompacted(msg.content)
    ]
    if len(indices) <= _KEEP_RECENT_EXECUTION_RESULTS:
        return messages

    to_clear = set(indices[:-_KEEP_RECENT_EXECUTION_RESULTS])
    result: list[Message] = []
    for i, msg in enumerate(messages):
        if i in to_clear:
            replacement = strip_persisted_preview(msg.content) or MICROCOMPACT_PLACEHOLDER
            result.append(ExecutionResultMessage(replacement))
        else:
            result.append(msg)

    logger.info("Microcompact: cleared %d old execution results", len(to_clear))
    return result


def _is_microcompacted(content: str) -> bool:
    """Whether a result is already at its microcompact floor — the generic
    placeholder, or a persist marker whose preview is already stripped
    (``strip_persisted_preview`` is identity on those)."""
    if content == MICROCOMPACT_PLACEHOLDER:
        return True
    return strip_persisted_preview(content) == content


def _split_around_keep_window(
    messages: list[Message],
    level: _SplitLevel,
) -> tuple[list[Message], list[Message]]:
    """Split into ``(to_summarize, to_keep)`` around the tail keep-window."""
    keep_count = max(len(messages) // level.value, _MIN_KEEP_MESSAGES)
    split_idx = _align_split_to_pairs(messages, max(0, len(messages) - keep_count))
    return messages[:split_idx], messages[split_idx:]


def _build_summary_messages(summary: str, to_keep: list[Message]) -> list[Message]:
    """Lay out ``[summary-as-user, ack-as-assistant, *to_keep]``.

    The synthetic pair gives the model a coherent prefix before the kept tail
    resumes. The ack is omitted when the tail already opens with an assistant
    turn, which would otherwise put two assistant messages back to back —
    valid for OpenAI, rejected by providers that enforce role alternation.
    """
    messages: list[Message] = [
        SummaryMessage(
            COMPACTION_SUMMARY_USER_TEMPLATE.format(summary=summary),
            summary,
        ),
    ]
    if not (to_keep and to_keep[0].role in (MessageRole.ASSISTANT, MessageRole.CODE_EXECUTION)):
        messages.append(SummaryAcknowledgementMessage(COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER))
    messages.extend(to_keep)
    return messages
