"""Prompts and transcript formatting for conversation summarization."""

import re

MICROCOMPACT_PLACEHOLDER = "[Old execution result cleared to save context space]"

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

_MAX_CONTENT_DISPLAY_CHARS = 2000


def format_transcript(messages: list, max_chars_per_msg: int = _MAX_CONTENT_DISPLAY_CHARS) -> str:
    """Render messages into a readable transcript for the summarizer."""
    lines: list[str] = []
    for msg in messages:
        content = msg.content
        if content and content != MICROCOMPACT_PLACEHOLDER:
            truncated = content[:max_chars_per_msg]
            if len(content) > max_chars_per_msg:
                truncated += "..."
            lines.append(f"[{msg.role.value}]: {truncated}")
    return "\n\n".join(lines)


_SUMMARY_PATTERN = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)


def extract_summary(raw_output: str) -> str:
    """Extract the <summary> block from the dual-phase LLM output.

    Discards the <analysis> scratchpad. Falls back to the full output
    if <summary> tags are missing.
    """
    match = _SUMMARY_PATTERN.search(raw_output)
    if match:
        return match.group(1).strip()
    return raw_output.strip()
