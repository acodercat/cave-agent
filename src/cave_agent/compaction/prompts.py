"""Prompts, markers and transcript formatting for conversation summarization."""

import re

from .._placeholders import MICROCOMPACT_PLACEHOLDER

__all__ = [
    "COMPACT_SYSTEM_PROMPT",
    "COMPACT_USER_PROMPT",
    "COMPACT_UPDATE_USER_TEMPLATE",
    "COMPACTION_SUMMARY_MARKER",
    "SUMMARY_MARKERS",
    "COMPACTION_SUMMARY_USER_TEMPLATE",
    "COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER",
    "format_transcript",
    "extract_summary",
    "parse_legacy_summary",
]

COMPACT_SYSTEM_PROMPT = """\
You are a conversation summarizer. Your task is to produce a concise yet \
technically detailed summary of the conversation provided below.

CRITICAL RULES:
- Respond with plain text ONLY. Do NOT call any tools or generate code.
- Do NOT include greetings, apologies, or meta-commentary.

OUTPUT FORMAT — you MUST produce exactly two XML blocks in this order:

<analysis>
Think step by step: review the conversation structure, identify key decisions, \
note what information is essential for continuing work. This block is your \
scratchpad — it will be discarded and does NOT count toward the summary.
</analysis>

<summary>
Write the final summary here. It MUST cover (skip sections that do not apply):
1. **User intent** — what the user explicitly asked for.
2. **Technical context** — languages, frameworks, libraries discussed.
3. **Code & execution** — code snippets executed, their outputs, and side effects.
4. **Errors & fixes** — errors encountered and how they were resolved.
5. **All user messages** — list every user message in order \
   (these reveal how the user's requests evolved).
6. **Problem solving** — key decisions and reasoning steps taken.
7. **Current state** — runtime variables, functions, and state at this point.
8. **Pending tasks** — any outstanding work the user requested.
9. **Next step** — what should happen next.
</summary>

REMINDER: Do NOT generate code. Respond with text only."""

COMPACT_USER_PROMPT = (
    "Summarise the conversation above. "
    "Focus on technical details essential for continuing the work."
)

# Used when the region being summarized already contains an earlier summary.
# The prior summary is handed over VERBATIM and the model is asked to fold the
# newer turns into it, rather than summarize a summary — see
# ``Compactor._summarize_with_llm`` for why that distinction matters.
COMPACT_UPDATE_USER_TEMPLATE = """\
Below is an existing summary of the earlier part of this conversation, \
followed by a transcript of everything that happened after it.

<existing_summary>
{prior_summary}
</existing_summary>

<new_turns>
{transcript}
</new_turns>

Produce an UPDATED summary covering the whole conversation: preserve every \
still-relevant fact from the existing summary verbatim where possible, fold \
in what the new turns added, and drop only what the new turns made obsolete. \
Do not re-compress the existing summary — it is already distilled."""

# The envelope wrapping a compaction summary. Presentation only: it tells the
# model it is reading a summary rather than an instruction. It is NOT how a
# summary is recognized — ``SummaryMessage`` is, because this is text and a
# user can type text.
_SUMMARY_OPEN = "<conversation-summary>"
_SUMMARY_CLOSE = "</conversation-summary>"

COMPACTION_SUMMARY_USER_TEMPLATE = f"{_SUMMARY_OPEN}\n{{summary}}\n{_SUMMARY_CLOSE}"

# The pre-envelope marker, kept only to recognize histories written by <=0.8.0.
# Content alone cannot separate a real legacy summary from a user message that
# happens to reproduce the old template — the bytes are identical — so the
# legacy form is recognized only together with the acknowledgement the <=0.8.0
# builder emitted with it, unconditionally, in the very next message. That pair
# is evidence a user message cannot manufacture by accident.
COMPACTION_SUMMARY_MARKER = "[Previous conversation summary]"

# Every legacy wording ever shipped. Frozen: new formats go in the envelope
# above, so this tuple only ever grows if a *past* release used another prefix.
SUMMARY_MARKERS = (COMPACTION_SUMMARY_MARKER,)

COMPACTION_SUMMARY_ASSISTANT_PLACEHOLDER = (
    "Understood. I have the context from our previous conversation and I'm ready to continue."
)

_MAX_CONTENT_DISPLAY_CHARS = 2000


def format_transcript(
    messages: list, max_chars_per_msg: int | None = _MAX_CONTENT_DISPLAY_CHARS
) -> str:
    """Render messages into a readable transcript for the summarizer.

    ``max_chars_per_msg=None`` disables per-message truncation — used when
    fidelity matters more than length.
    """
    lines: list[str] = []
    for msg in messages:
        content = msg.content
        if not content or content == MICROCOMPACT_PLACEHOLDER:
            continue
        if max_chars_per_msg is not None and len(content) > max_chars_per_msg:
            content = content[:max_chars_per_msg] + "..."
        lines.append(f"[{msg.role.value}]: {content}")
    return "\n\n".join(lines)


_SUMMARY_PATTERN = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)


def parse_legacy_summary(content: str) -> str | None:
    """The summary body of a ``<=0.8.0`` summary message, else ``None``.

    Text matching only, and text is what users write — so this alone is not
    evidence. :func:`~cave_agent.compaction.migrate_legacy_summaries` is the
    caller, and it requires the acknowledgement that builder always emitted
    alongside; the two together are what a user message cannot produce by
    accident.
    """
    for marker in SUMMARY_MARKERS:
        if content.startswith(f"{marker}\n"):
            return content[len(marker) :].strip()
    return None


def extract_summary(raw_output: str) -> str:
    """Extract the <summary> block from the dual-phase LLM output.

    Discards the <analysis> scratchpad. Falls back to the full output
    if <summary> tags are missing.
    """
    match = _SUMMARY_PATTERN.search(raw_output)
    if match:
        return match.group(1).strip()
    return raw_output.strip()
