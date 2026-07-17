import re

from .parsing import StreamingTextParser, SegmentType

# Lone Unicode surrogates (U+D800–U+DFFF outside a valid pair) are valid
# ``str`` members but invalid UTF-8. They survive in memory, then crash
# ``json.dumps`` inside the provider SDK with "surrogates not allowed",
# taking the whole run down. Common sources here: a clipboard paste from a
# rich-text editor (Word / Google Docs), or binary bytes leaking from code
# execution stdout. Replacing each with U+FFFD keeps the message structure.
_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_REPLACEMENT_CHAR = "�"


def sanitize_surrogates(text: str) -> str:
    """Replace every lone Unicode surrogate in ``text`` with U+FFFD.

    Pure function applied at boundaries where caller- or execution-supplied
    text enters the conversation and would otherwise be serialized to UTF-8
    by a provider SDK. ``re.sub`` returns the input unchanged when there are
    no matches, so the common (clean) case allocates nothing.
    """
    if not text:
        return text
    return _LONE_SURROGATE_RE.sub(_REPLACEMENT_CHAR, text)


def extract_python_code(response: str, python_block_identifier: str) -> str | None:
    """Extract python code block from LLM output."""
    parser = StreamingTextParser(python_block_identifier)
    segments = parser.process_chunk(response)
    segments.extend(parser.flush())
    code_parts = [s.content for s in segments if s.type == SegmentType.CODE and s.content.strip()]
    return "\n\n".join(code_parts) if code_parts else None
