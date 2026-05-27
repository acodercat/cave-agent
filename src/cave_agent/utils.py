from .parsing import StreamingTextParser, SegmentType


def extract_python_code(response: str, python_block_identifier: str) -> str | None:
    """Extract python code block from LLM output."""
    parser = StreamingTextParser(python_block_identifier)
    segments = parser.process_chunk(response)
    segments.extend(parser.flush())
    code_parts = [s.content for s in segments if s.type == SegmentType.CODE and s.content.strip()]
    return "\n\n".join(code_parts) if code_parts else None
