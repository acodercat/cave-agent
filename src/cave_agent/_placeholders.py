"""Short LLM-facing sentinels and markers the agent splices into history.

These are the strings the agent emits *reactively* — a cleared result, a
cancelled execution, an oversize-output marker — as opposed to the authored
prompts in :mod:`cave_agent.prompts` (system prompt, execution-output wrapper)
and :mod:`cave_agent.compaction.prompts` (summarization). Keeping them together
makes the sentinels greppable for audit or translation; the per-constant
comments explain when each fires, and the wording is load-bearing.

Composed markers (the ``<persisted-output>`` block) keep their builder and
reducer here too, so a format change cannot desynchronize producer and
consumer.
"""

import keyword
import re
import sys
import unicodedata
from collections.abc import Iterable

# Execution-result placeholders ----------------------------------------------

# Replaces an ``ExecutionResultMessage`` body during microcompaction. The
# result is gone, but the message slot stays so the code/result alternation
# the conversation was built on remains readable.
MICROCOMPACT_PLACEHOLDER = "[Old execution result cleared to save context space]"

# Body used when a code block produced no stdout at all. A bare empty result
# reads as "nothing happened"; some models respond by re-running the same
# cell. An explicit marker keeps silent-success code (pure assignments,
# in-place mutations) legible on the next turn.
EMPTY_OUTPUT_PLACEHOLDER = "No output"

# Synthetic result recorded when a run is cancelled between appending the
# code message and appending its result. Without it the history ends on an
# assistant turn claiming to have run code that has no outcome, and the next
# run's user message lands directly after it.
EXECUTION_CANCELLED_PLACEHOLDER = "[Execution cancelled]"

# Same slot, but for an unexpected exception escaping the execution path
# rather than a deliberate cancellation. Kept distinct so the model can tell
# "the user stopped this" from "the agent broke".
EXECUTION_INCOMPLETE_PLACEHOLDER = "[Execution did not complete]"

# Synthetic result used when cancellation forced the isolated runtime process
# to terminate. The cancelled cell did not merely stop: imports, variables and
# functions created by earlier model code disappeared with the process.
EXECUTION_CANCELLED_AFTER_STATE_LOSS_PLACEHOLDER = (
    "[Execution cancelled; stopping it terminated the runtime process and "
    "non-registered state was lost]"
)

# Feedback after an execution deadline. Kept here with the other reactive
# conversation text rather than open-coded in the agent loop.
EXECUTION_TIMEOUT_PROMPT = (
    "Code execution timed out after {timeout} seconds. "
    "Simplify your code or break it into smaller steps."
)

# Appended whenever recovery replaced the kernel process. Registered resources
# are restored lazily; arbitrary state created by model code cannot be.
RUNTIME_STATE_LOST_NOTICE = (
    "Stopping this execution terminated the runtime process. Variables, imports, "
    "functions, and other state created by earlier code were lost; registered "
    "injected resources will be restored before the next execution. Recreate "
    "any state you still need."
)

# Conversation-flow placeholders ---------------------------------------------

# Sentinel user turn appended after a cancelled run so the model sees, on the
# next run, that its previous response was cut short by the user rather than
# inferring it from a dangling code block. Wording mirrors Claude Code's.
USER_INTERRUPTION_PLACEHOLDER = "[Request interrupted by user]"

# Injected after a ``finish_reason="length"`` truncation so the model resumes
# instead of restarting. Deliberately terse — it is prepended to a partial
# thought and should not derail it.
OUTPUT_RECOVERY_PROMPT = (
    "Output limit hit. Resume directly, pick up mid-thought. "
    "Break remaining work into smaller pieces."
)

# Injected when a stream stopped producing mid-response. Distinct from
# OUTPUT_RECOVERY_PROMPT, which names the output limit as the cause and advises
# smaller pieces — wrong and misleading advice for a dropped connection. Same
# distinction as EXECUTION_CANCELLED vs EXECUTION_INCOMPLETE.
STREAM_INTERRUPTED_PROMPT = (
    "Your previous message was cut off mid-stream. Resume directly, pick up mid-thought."
)

# Persisted-output marker -----------------------------------------------------
#
# The ``<persisted-output>`` block that replaces an oversize execution result
# once its full text has been stashed in the runtime namespace. Builder and
# reducer live side by side so the marker's shape is defined in exactly one
# module: the agent builds it (``CaveAgent._shape_output``) and compaction
# later strips just the preview (``_microcompact``) — the declaration line is
# the only pointer back to the stashed text.
#
# Unlike a file-backed agent, cave-agent stashes the output *in the runtime*
# where the code that produced it already lives. The variable is a normal
# Python string: the model can slice it, regex it, or feed it back into
# pandas, with no extra tool call and no filesystem round-trip.

_PERSISTED_OUTPUT_OPEN = "<persisted-output>"
_PERSISTED_OUTPUT_CLOSE = "</persisted-output>"
_PERSISTED_PREVIEW_HEADER = "\n\nPreview (first "

# Prefix for the namespace variables holding oversize outputs. Each output gets
# its own name (``_output_1``, ``_output_2``, …), allocated by
# ``Runtime.bind_unique``, because the marker left in history is a *pointer*:
# reusing one name does not "keep the newest", it silently re-points every
# earlier marker at data it never described.
PERSISTED_OUTPUT_PREFIX = "_output"


def normalize_identifier(name: str) -> str:
    """The name Python will actually bind for *name*.

    Python normalizes identifiers to NFKC before binding them, so ``K`` (U+212A
    KELVIN SIGN) is a valid identifier that binds ``K``. Applied once at the
    boundary, so markers, history scanning and allocation all name the variable
    that will actually exist rather than the one that was typed.
    """
    return unicodedata.normalize("NFKC", name)


def usable_identifier(name: str, kind: str = "Name") -> str:
    """*name* as Python will bind it, or raise if Python cannot bind it at all.

    Normalization and validation belong together: ``isidentifier`` answers about
    the *normalized* form, and a keyword passes it while being unusable — a
    ``Variable("for")`` is a ``SyntaxError`` waiting for the first execution
    that touches it, a long way from the registration that caused it.

    Only *hard* keywords are rejected. ``match``, ``case``, ``type`` and ``_``
    are soft keywords: they carry meaning only in the positions that introduce
    them, and are perfectly ordinary names to assign to.
    """
    normalized = normalize_identifier(name)
    if not normalized.isidentifier():
        raise ValueError(
            f"{kind} {name!r} is not a Python identifier"
            + (f" (it normalizes to {normalized!r})" if normalized != name else "")
        )
    if keyword.iskeyword(normalized):
        raise ValueError(f"{kind} {name!r} is a Python keyword and cannot name a value")
    return normalized


def highest_persisted_index(contents: Iterable[str], prefix: str) -> int:
    """Highest index any marker in *contents* already claims, else 0.

    Lets a resumed conversation keep allocating names without colliding with a
    pointer an earlier run already handed to the model.

    The pattern is built from *prefix* itself, escaped: a fixed ``[A-Za-z_]``
    class would recognize only ASCII prefixes while the agent accepts any Python
    identifier, and an unrecognized marker silently frees a name the model is
    still holding a pointer to.
    """
    marker = re.compile(
        rf"{re.escape(_PERSISTED_OUTPUT_OPEN)}\n"
        rf"Output too large \(\d+ chars\) to inline\. "
        rf"The full text is in the runtime variable "
        rf"`{re.escape(prefix)}_([0-9]+)` \(a str\)\."
    )
    largest = str(sys.maxsize)

    def parse_index(digits: str) -> int | None:
        digits = digits.lstrip("0") or "0"
        if len(digits) > len(largest):
            return None
        if len(digits) == len(largest) and digits > largest:
            return None
        return int(digits)

    return max(
        (
            index
            for content in contents
            for match in marker.finditer(content)
            if (index := parse_index(match.group(1))) is not None
        ),
        default=0,
    )


def _generate_preview(content: str, max_chars: int) -> tuple[str, bool]:
    """Slice ``content`` to fit ``max_chars``, preferring a newline cut.

    If a newline lives in the second half of the slice, cut there for a clean
    break; otherwise hard-cut. Returns ``(preview, has_more)`` so the caller
    can append an ellipsis when content was truncated.
    """
    if len(content) <= max_chars:
        return content, False
    head = content[:max_chars]
    last_newline = head.rfind("\n")
    cut = last_newline if last_newline > max_chars // 2 else max_chars
    return content[:cut], True


def build_persist_marker(output: str, preview_chars: int, variable: str) -> str:
    """Compose the ``<persisted-output>`` block for an oversize result.

    A leading declaration of total size + where the full text now lives,
    optionally followed by an inline preview of the leading characters so the
    model can react without spending a turn (suppressed when *preview_chars*
    is 0). The declaration names the runtime variable and shows how to slice
    it — the point is that the output is *still there*, not that it was lost.

    *variable* is required on purpose: a default shared by every call is exactly
    how each marker came to claim the same name, leaving all but the newest
    pointing at data that had been overwritten.
    """
    body = (
        f"Output too large ({len(output)} chars) to inline. "
        f"The full text is in the runtime variable `{variable}` (a str). "
        f"Slice it (`{variable}[:2000]`), search it "
        f"(`[l for l in {variable}.splitlines() if 'error' in l]`), "
        f"or re-parse it — do not re-run the code to see it."
    )
    if preview_chars > 0:
        preview, has_more = _generate_preview(output, preview_chars)
        ellipsis = "\n..." if has_more else ""
        body += f"{_PERSISTED_PREVIEW_HEADER}{len(preview)} chars):\n{preview}{ellipsis}"
    return f"{_PERSISTED_OUTPUT_OPEN}\n{body}\n{_PERSISTED_OUTPUT_CLOSE}"


def strip_persisted_preview(content: str) -> str | None:
    """Reduce a persist marker to its declaration line, dropping the preview.

    Returns ``None`` when *content* holds no ``<persisted-output>`` marker, and
    the (unchanged) input when the marker carries no preview — so
    ``strip_persisted_preview(c) == c`` identifies an already-reduced marker.
    The declaration line is always kept: it names the variable that still holds
    the full text, and clearing it to the generic placeholder would strand data
    the model can no longer reach.

    The marker is *located*, not required at position 0. By the time compaction
    sees it the agent has wrapped it in ``EXECUTION_OUTPUT_PROMPT``, so the
    content begins ``"\\n<execution_output>\\n<persisted-output>…"``. Anchoring
    on the start silently failed that check and fell through to the generic
    placeholder — destroying the pointer this whole mechanism exists to keep.
    Surrounding wrapper text is preserved.
    """
    open_at = content.find(_PERSISTED_OUTPUT_OPEN)
    if open_at == -1:
        return None
    # The preview is arbitrary execution output and may itself contain the
    # closing sentinel. The agent creates one outer marker per result, so its
    # boundary is the final closing sentinel in that message.
    close_at = content.rfind(_PERSISTED_OUTPUT_CLOSE)
    if close_at < open_at:
        return None

    block_end = close_at + len(_PERSISTED_OUTPUT_CLOSE)
    block = content[open_at:block_end]
    declaration, separator, _ = block.partition(_PERSISTED_PREVIEW_HEADER)
    if not separator:
        return content
    reduced = f"{declaration}\n{_PERSISTED_OUTPUT_CLOSE}"
    return content[:open_at] + reduced + content[block_end:]
