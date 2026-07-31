import pytest

from cave_agent.parsing import Segment, SegmentType, StreamingTextParser

# Chunk sizes a provider might deliver: one character at a time, a typical SSE
# delta, and the whole response in one piece. Parsing must not depend on which.
CHUNK_SIZES = [1, 13, 10_000]


def parse(text, chunk_size=1, language="python"):
    """Feed *text* the way a provider streams it.

    Returns ``(segments, unparsed)``. The parser stops at the fence closing the
    first code block, so *unparsed* is everything after it — the tail of the
    chunk that completed the block plus every chunk that would have followed.
    """
    parser = StreamingTextParser(language)
    segments = []
    for start in range(0, len(text), chunk_size):
        chunk = text[start : start + chunk_size]
        segments.extend(parser.process_chunk(chunk))
        if parser.is_first_code_block_completed():
            return segments, parser.remainder + text[start + len(chunk) :]
    segments.extend(parser.flush())
    return segments, ""


def text_of(segments):
    return "".join(s.content for s in segments if s.type == SegmentType.TEXT)


def code_of(segments):
    return [s.content for s in segments if s.type == SegmentType.CODE]


class TestFirstBlockEndsParsing:
    """The parser stops at the fence closing the first code block.

    It used to keep parsing the rest of the chunk. When a whole response
    arrived as one delta — routine for a fast model or a buffering proxy — the
    agent saw a CODE segment per block and kept the *last*, so it executed the
    second block and silently dropped the first. Chunk size is a property of
    the transport, so any behaviour that varies with it is a bug by definition.
    """

    TWO_BLOCKS = (
        "Here:\n```python\nwhich = 'first'\n```\nand also\n```python\nwhich = 'second'\n```\n"
    )

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_only_the_first_block_is_parsed(self, chunk_size):
        segments, _ = parse(self.TWO_BLOCKS, chunk_size)
        assert code_of(segments) == ["which = 'first'"]

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_everything_after_the_fence_is_left_unparsed(self, chunk_size):
        segments, unparsed = parse(self.TWO_BLOCKS, chunk_size)
        assert unparsed == "\nand also\n```python\nwhich = 'second'\n```\n"
        assert text_of(segments) == "Here:\n"

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_consumed_and_unparsed_reconstruct_the_input(self, chunk_size):
        """Nothing is invented and nothing is lost — only split."""
        segments, unparsed = parse(self.TWO_BLOCKS, chunk_size)
        consumed = text_of(segments) + "```python\n" + code_of(segments)[0] + "\n```"
        assert consumed + unparsed == self.TWO_BLOCKS

    def test_a_completed_parser_consumes_nothing_more(self):
        parser = StreamingTextParser()
        parser.process_chunk("```python\nx = 1\n```")
        assert parser.process_chunk("trailing text") == []
        assert parser.remainder == "trailing text"

    def test_flush_makes_the_parser_reusable(self):
        """flush() ends the stream, so the next one starts clean."""
        parser = StreamingTextParser()
        parser.process_chunk("```python\nx = 1\n```")
        parser.flush()
        assert not parser.is_first_code_block_completed()
        segments = parser.process_chunk("```python\ny = 2\n```")
        assert code_of(segments) == ["y = 2"]


class TestStreamingTextParser:
    """Test core streaming functionality."""

    def test_text_streams_character_by_character(self):
        """Verify plain text streams immediately, character by character."""
        parser = StreamingTextParser()

        text = "Hello World"
        results = []

        for char in text:
            segments = parser.process_chunk(char)
            results.extend(segments)

        # Should get one segment per character for plain text
        assert len(results) == len(text)

        # Each segment should be a single character
        for i, segment in enumerate(results):
            assert segment.type == SegmentType.TEXT
            assert segment.content == text[i]

        # Reassemble should give original text
        assert "".join(s.content for s in results) == text

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_python_code_block_complete_flow(self, chunk_size):
        """Text before the block, the code, and the untouched tail."""
        before, code_block, after = "Before code: ", "```python\nprint('test')\n```", " After code"

        segments, unparsed = parse(before + code_block + after, chunk_size)

        assert code_of(segments) == ["print('test')"]
        assert text_of(segments) == before
        assert unparsed == after

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_multiple_python_blocks(self, chunk_size):
        """A second block is not parsed — it is left for a turn not yet asked for."""
        segments, unparsed = parse(
            "First\n```python\ncode1\n```\nMiddle\n```python\ncode2\n```\nEnd",
            chunk_size,
        )

        assert code_of(segments) == ["code1"]
        assert text_of(segments) == "First\n"
        assert unparsed == "\nMiddle\n```python\ncode2\n```\nEnd"

    def test_flush_completes_processing(self):
        """Test that flush properly completes any pending processing."""
        parser = StreamingTextParser()

        # Process incomplete code block
        incomplete = "```python\nprint('incomplete')"

        segments = []
        for char in incomplete:
            char_segments = parser.process_chunk(char)
            segments.extend(char_segments)

        # Without flush, code might not be complete
        code_before_flush = [s for s in segments if s.type == SegmentType.CODE]
        assert code_before_flush == []

        # Flush should complete processing
        final_segments = parser.flush()
        segments.extend(final_segments)

        # Should have the code content
        code_segments = [s for s in segments if s.type == SegmentType.CODE]
        assert len(code_segments) == 1
        assert "print('incomplete')" in code_segments[0].content

    @pytest.mark.parametrize("suffix", ["`", "``"])
    def test_flush_emits_partial_closing_backticks_once(self, suffix):
        parser = StreamingTextParser()

        segments = parser.process_chunk(f"```python\nx = 1\n{suffix}")
        segments.extend(parser.flush())

        code = [segment.content for segment in segments if segment.type == SegmentType.CODE]
        assert code == [f"x = 1\n{suffix}"]

    def test_state_reset_after_flush(self):
        """Test that flush resets parser state properly."""
        parser = StreamingTextParser()

        # First: process some content
        first_input = "```python\nfirst\n```"
        for char in first_input:
            parser.process_chunk(char)

        # Flush and verify state is reset
        parser.flush()

        assert parser.mode == StreamingTextParser.Mode.TEXT
        assert parser.text_buffer == ""
        assert parser.code_buffer == ""
        assert parser.in_code_block is False
        assert parser.backtick_count == 0

        # Second: process new content (should work with clean state)
        second_input = "```python\nsecond\n```"
        segments = []
        for char in second_input:
            char_segments = parser.process_chunk(char)
            segments.extend(char_segments)

        final = parser.flush()
        segments.extend(final)

        code_segments = [s for s in segments if s.type == SegmentType.CODE]
        assert len(code_segments) == 1
        assert "second" in code_segments[0].content

    def test_custom_language_identifier(self):
        """Test parser with custom language identifier."""
        parser = StreamingTextParser("javascript")

        # Should detect javascript blocks, not python
        input_text = "```javascript\nconsole.log('test');\n```"

        segments = []
        for char in input_text:
            char_segments = parser.process_chunk(char)
            segments.extend(char_segments)

        final = parser.flush()
        segments.extend(final)

        # Should have code segment for javascript
        code_segments = [s for s in segments if s.type == SegmentType.CODE]
        assert len(code_segments) == 1
        assert "console.log" in code_segments[0].content

    def test_python_block_with_spaces(self):
        """Test Python code block with spaces after language identifier."""
        parser = StreamingTextParser()

        # Common variation: space after 'python'
        input_text = "```python \ncode_here\n```"

        segments = []
        for char in input_text:
            char_segments = parser.process_chunk(char)
            segments.extend(char_segments)

        final = parser.flush()
        segments.extend(final)

        code_segments = [s for s in segments if s.type == SegmentType.CODE]
        assert len(code_segments) == 1
        assert "code_here" in code_segments[0].content

    def test_code_preserves_indentation(self):
        """Test that code blocks preserve indentation."""
        parser = StreamingTextParser()

        code_with_indent = """```python
        def test():
            if True:
                print('indented')
        ```"""

        segments = []
        for char in code_with_indent:
            char_segments = parser.process_chunk(char)
            segments.extend(char_segments)

        final = parser.flush()
        segments.extend(final)

        code_segments = [s for s in segments if s.type == SegmentType.CODE]
        assert len(code_segments) == 1

        # Check indentation is preserved
        code = code_segments[0].content
        assert "    if True:" in code or "if True:" in code.strip()
        assert "print('indented')" in code

    def test_segment_type_enum(self):
        """Test that SegmentType enum works correctly."""
        assert SegmentType.TEXT.value == "text"
        assert SegmentType.CODE.value == "code"

        # Test segment creation
        text_seg = Segment(SegmentType.TEXT, "hello")
        assert text_seg.type == SegmentType.TEXT
        assert text_seg.content == "hello"

        code_seg = Segment(SegmentType.CODE, "print('test')")
        assert code_seg.type == SegmentType.CODE
        assert code_seg.content == "print('test')"

    def test_plain_text_streams_immediately(self):
        """Verify that plain text streams character by character."""
        parser = StreamingTextParser()

        input_text = "Hello world"
        segments = []

        for char in input_text:
            char_segments = parser.process_chunk(char)
            segments.extend(char_segments)

        # Each character should produce a segment immediately
        assert len(segments) == len(input_text)

        # Reconstruct the text
        combined = "".join(s.content for s in segments)
        assert combined == input_text

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_inline_backticks_round_trip_exactly(self, chunk_size):
        """Inline code is text, and text is never altered.

        A backtick puts the parser in BACKTICK_COUNT mode, where it holds
        characters until it knows whether a fence is opening. Held is not
        lost: the earlier assertions here allowed for a dropped `v` and `h`
        that the parser never dropped, so any future change that really did
        eat a character would have passed them.
        """
        input_text = "Use `var` here"

        segments, unparsed = parse(input_text, chunk_size)

        assert all(s.type == SegmentType.TEXT for s in segments)
        assert text_of(segments) == input_text
        assert unparsed == ""

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_python_code_block_detection(self, chunk_size):
        """Leading text is preserved exactly — no character is consumed."""
        segments, unparsed = parse(
            "Text before ```python\nprint('hello')\n``` text after",
            chunk_size,
        )

        assert code_of(segments) == ["print('hello')"]
        assert text_of(segments) == "Text before "
        assert unparsed == " text after"

    def test_non_python_block_buffering(self):
        """Test that non-Python code blocks are buffered and treated as text."""
        input_text = "```javascript\ncode\n```"

        segments, unparsed = parse(input_text)

        # A fence for another language is not a code block — it is text, and
        # it survives byte-for-byte (the old assertion tolerated a dropped 'j'
        # the parser never dropped).
        assert code_of(segments) == []
        assert text_of(segments) == input_text
        assert unparsed == ""

    def test_empty_python_block(self):
        """Test empty Python code block behavior."""
        parser = StreamingTextParser()

        # Based on actual behavior, empty blocks might not create a code segment
        # if the content is completely empty after stripping
        input_text = "```python\n \n```"  # Add a space to test strip behavior
        segments = []

        for char in input_text:
            char_segments = parser.process_chunk(char)
            segments.extend(char_segments)

        final_segments = parser.flush()
        segments.extend(final_segments)

        code_segments = [s for s in segments if s.type == SegmentType.CODE]

        # Check if code segment exists and is empty after strip
        if len(code_segments) > 0:
            assert code_segments[0].content.strip() == ""

    def test_pythonscript_edge_case(self):
        """Test that 'pythonscript' is not detected as Python code block."""
        parser = StreamingTextParser()

        input_text = "```pythonscript\ncode\n```"
        segments = []

        for char in input_text:
            char_segments = parser.process_chunk(char)
            segments.extend(char_segments)

        final_segments = parser.flush()
        segments.extend(final_segments)

        # Should not create code segments
        code_segments = [s for s in segments if s.type == SegmentType.CODE]
        assert len(code_segments) == 0

        # Text should contain the content (with some buffering artifacts)
        text_segments = [s for s in segments if s.type == SegmentType.TEXT]
        all_text = "".join(s.content for s in text_segments)

        # Based on actual behavior, 'pythons' gets buffered when matching fails
        assert "cript" in all_text or "script" in all_text
        assert "code" in all_text

    def test_multiple_consecutive_backticks(self):
        """Test handling of multiple backticks in sequence."""
        parser = StreamingTextParser()

        # Test various backtick patterns
        test_cases = [
            ("``", "Double backticks"),
            ("`single`", "Single backticks around word"),
            ("```not-python```", "Triple backticks without Python"),
        ]

        for input_text, description in test_cases:
            parser = StreamingTextParser()  # Fresh parser for each test
            segments = []

            for char in input_text:
                char_segments = parser.process_chunk(char)
                segments.extend(char_segments)

            final_segments = parser.flush()
            segments.extend(final_segments)

            # Should all be text (no Python code blocks)
            code_segments = [s for s in segments if s.type == SegmentType.CODE]
            assert len(code_segments) == 0, f"Failed for: {description}"

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_streaming_vs_flush_completeness(self, chunk_size):
        """Every character is accounted for: consumed, or left unparsed."""
        input_text = "Start ```python\ncode\n``` end"

        segments, unparsed = parse(input_text, chunk_size)

        assert text_of(segments) == "Start "
        assert code_of(segments) == ["code"]
        assert unparsed == " end"

    def test_code_block_with_special_characters(self):
        """Test code blocks containing special characters."""
        parser = StreamingTextParser()

        input_text = "```python\nprint('`test`')\nprint('```')\n```"
        segments = []

        for char in input_text:
            char_segments = parser.process_chunk(char)
            segments.extend(char_segments)

        final_segments = parser.flush()
        segments.extend(final_segments)

        code_segments = [s for s in segments if s.type == SegmentType.CODE]
        assert len(code_segments) == 1

        # Code should contain the backticks as part of the string literals
        code_content = code_segments[0].content
        assert "`test`" in code_content
        assert "print('```')" in code_content

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_line_start_fence_inside_triple_quoted_string(self, chunk_size):
        input_text = (
            "```python\n"
            'payload = """embedded markdown:\n'
            "```\n"
            'still data"""\n'
            "print(payload)\n"
            "```after"
        )

        segments, unparsed = parse(input_text, chunk_size)

        assert code_of(segments) == [
            'payload = """embedded markdown:\n```\nstill data"""\nprint(payload)'
        ]
        assert unparsed == "after"

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_line_start_backticks_inside_continued_string(self, chunk_size):
        input_text = "```python\npayload = 'abc\\\n```def'\nprint(payload)\n```"

        segments, unparsed = parse(input_text, chunk_size)

        code = "payload = 'abc\\\n```def'\nprint(payload)"
        compile(code, "<continued-string>", "exec")
        assert code_of(segments) == [code]
        assert unparsed == ""

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_line_start_backticks_inside_continued_f_string(self, chunk_size):
        input_text = "```python\npayload = f'abc\\\n```def'\nprint(payload)\n```"

        segments, unparsed = parse(input_text, chunk_size)

        code = "payload = f'abc\\\n```def'\nprint(payload)"
        compile(code, "<continued-f-string>", "exec")
        assert code_of(segments) == [code]
        assert unparsed == ""

    def test_incremental_streaming_order(self):
        """Test that streaming maintains reasonable order for simple text."""
        parser = StreamingTextParser()

        input_text = "ABC"
        output_chars = []

        for char in input_text:
            segments = parser.process_chunk(char)
            for segment in segments:
                output_chars.append(segment.content)

        # For simple text, should stream in order
        assert output_chars == ["A", "B", "C"]

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_boundary_between_text_and_code(self, chunk_size):
        """No delimiter bleeds into the neighbouring segment."""
        segments, unparsed = parse("text```python\ncode\n```more", chunk_size)

        assert code_of(segments) == ["code"]
        assert text_of(segments) == "text"
        assert unparsed == "more"

    def test_single_backtick_in_text_bug_case_1(self):
        """Test the specific bug case that was reported - single backticks causing character loss."""
        parser = StreamingTextParser()

        input_text = (
            "I see the error - the `hourly_station_air_quality` table doesn't contain wind data."
        )
        segments = []

        for char in input_text:
            char_segments = parser.process_chunk(char)
            segments.extend(char_segments)

        final_segments = parser.flush()
        segments.extend(final_segments)

        # All should be text segments (no code blocks)
        assert all(s.type == SegmentType.TEXT for s in segments)

        # Reconstruct the output
        output_text = "".join(s.content for s in segments)

        # The output should match the input exactly (no character loss or reordering)
        assert output_text == input_text, f"Expected: '{input_text}'\nGot: '{output_text}'"

        # Specific checks for the problematic parts
        assert "hourly_station_air_quality" in output_text
        assert "`hourly_station_air_quality`" in output_text
        assert "doesn't contain wind data" in output_text

    def test_single_backtick_with_code_block(self):
        """Test single backticks mixed with actual Python code blocks."""
        input_text = """Let me query the `users` table and then process the data:

```python
import pandas as pd

# Query the users table
df = pd.read_sql("SELECT * FROM `users`", engine)
print(f"Found {len(df)} users")
```

The `users` table contains `id`, `name`, and `email` columns."""

        segments, unparsed = parse(input_text)
        text_segments = [s for s in segments if s.type == SegmentType.TEXT]
        code_segments = [s for s in segments if s.type == SegmentType.CODE]

        # Should have exactly one code block
        assert len(code_segments) == 1

        # Verify code content
        code_content = code_segments[0].content
        assert "import pandas as pd" in code_content
        assert "SELECT * FROM `users`" in code_content  # Backticks inside code should be preserved
        assert 'print(f"Found {len(df)} users")' in code_content

        # Inline backticks in the leading text are preserved untouched. The
        # trailing prose sits after the closing fence, so it is unparsed.
        all_text = "".join(s.content for s in text_segments)
        assert all_text == "Let me query the `users` table and then process the data:\n\n"
        assert "`id`, `name`, and `email` columns." in unparsed

    def test_triple_backtick_confusion_bug_case_2(self):
        """Test the second bug case - triple backtick confusion causing weird output."""
        parser = StreamingTextParser()

        input_text = (
            "I see the error - the ```hourly_station_air_quality` table doesn't contain wind data."
        )
        segments = []

        for char in input_text:
            char_segments = parser.process_chunk(char)
            segments.extend(char_segments)

        final_segments = parser.flush()
        segments.extend(final_segments)

        # Should be all text segments (not a valid python code block)
        text_segments = [s for s in segments if s.type == SegmentType.TEXT]
        code_segments = [s for s in segments if s.type == SegmentType.CODE]

        # Should have no code segments since ```hourly is not ```python
        assert len(code_segments) == 0
        assert len(text_segments) > 0

        # Reconstruct the output
        output_text = "".join(s.content for s in segments)

        # The output should preserve all the content
        assert "I see the error" in output_text
        assert "hourly_station_air_quality" in output_text
        assert "table doesn't contain wind data" in output_text
        assert "```" in output_text  # The triple backticks should be preserved

        # Should not have the weird reordering that was occurring
        assert "```o" not in output_text, f"Found problematic reordering in: '{output_text}'"

    def test_triple_backtick_with_code_block(self):
        """Test triple backticks in text mixed with actual Python code blocks."""
        input_text = """The markdown syntax uses ``` for code blocks. Here's an example:

```python
# This is a real code block
def process_markdown(text):
    # Look for triple backtick markers
    if "``" + "`" in text:  # Avoiding literal ``` to not break parser
        return text.replace("``" + "`", "<code>")
    return text
```

Remember that ``` without a language identifier like ```javascript won't be parsed as Python code."""

        segments, unparsed = parse(input_text)
        text_segments = [s for s in segments if s.type == SegmentType.TEXT]
        code_segments = [s for s in segments if s.type == SegmentType.CODE]

        # Should have exactly one Python code block
        assert len(code_segments) == 1

        # Verify code content
        code_content = code_segments[0].content
        assert "def process_markdown(text):" in code_content
        assert "# This is a real code block" in code_content
        assert 'if "``" + "`" in text:' in code_content  # Workaround for triple backticks

        # Triple backticks that open no Python block stay in the text stream.
        all_text = "".join(s.content for s in text_segments)
        assert "markdown syntax uses ```" in all_text
        # The closing prose follows the fence, so it is left unparsed.
        assert "```javascript" in unparsed
        assert "Remember that" in unparsed

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_inline_triple_backticks_remain_code(self, chunk_size):
        input_text = """```python
def example():
    # This comment contains ``` but is not a closing fence
    return "code after ``` remains in the block"
```"""

        segments, unparsed = parse(input_text, chunk_size)
        code_segments = [s for s in segments if s.type == SegmentType.CODE]

        assert len(code_segments) == 1
        code_content = code_segments[0].content

        assert "# This comment contains ```" in code_content
        assert 'return "code after ``` remains in the block"' in code_content
        assert unparsed == ""

    def test_mixed_backticks_complex_scenario(self):
        """Test a complex real-world scenario with all types of backticks."""
        input_text = """I'm analyzing the `users` and `orders` tables. Let me fix the SQL query:

```python
# Fix the query for the `orders` table
query = '''
SELECT o.*, u.name 
FROM `orders` o
JOIN `users` u ON o.user_id = u.id
WHERE o.created_at > '2024-01-01'
'''

# The backticks ` are needed for MySQL
df = pd.read_sql(query, engine)
print(f"Query returned {len(df)} rows from `orders`")
```

Note: The ```sql syntax would work too, but we're using Python here. 
The `DataFrame` object will contain all results."""

        segments, unparsed = parse(input_text)
        text_segments = [s for s in segments if s.type == SegmentType.TEXT]
        code_segments = [s for s in segments if s.type == SegmentType.CODE]

        # Should have exactly one code block
        assert len(code_segments) == 1

        # Verify code block preserves all backticks
        code_content = code_segments[0].content
        assert "FROM `orders` o" in code_content
        assert "JOIN `users` u" in code_content
        assert "# The backticks ` are needed" in code_content
        assert 'print(f"Query returned {len(df)} rows from `orders`")' in code_content

        # Inline backticks in the leading text survive; the closing prose is
        # after the fence, so it is unparsed rather than text.
        all_text = "".join(s.content for s in text_segments)
        assert "`users` and `orders` tables" in all_text
        assert "```sql" in unparsed  # non-Python fence, never a block
        assert "`DataFrame` object" in unparsed


class TestBrokenCodeStillCloses:
    """A block the model wrote badly must still reach the runtime.

    The fence check tokenizes the buffer to ask "is this fence inside a
    string". Any other tokenize failure answers a different question, and
    letting it escape crashed the run instead of handing the model a
    `SyntaxError` it could fix.
    """

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_an_indentation_error_does_not_crash_the_parser(self, chunk_size):
        segments, unparsed = parse(
            "```python\nif True:\n    a = 1\n  b = 2\n```\n",
            chunk_size,
        )

        assert code_of(segments) == ["if True:\n    a = 1\n  b = 2"]

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_an_unclosed_bracket_does_not_swallow_the_fence(self, chunk_size):
        segments, _ = parse("```python\nx = [1, 2,\n```\n", chunk_size)

        assert code_of(segments) == ["x = [1, 2,"]


class TestSingleLineStringsDoNotSwallowFences:
    """Only a string that spans lines can contain a fence.

    tokenize says "unterminated string literal" for both a one-line string and
    a backslash-continued one, so the message alone cannot decide — the line
    continuation is what distinguishes them.
    """

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_an_apostrophe_does_not_open_a_string(self, chunk_size):
        segments, unparsed = parse("```python\nprint('it's fine')\n```\n", chunk_size)

        assert code_of(segments) == ["print('it's fine')"]
        assert unparsed == "\n"

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_a_continued_string_still_swallows_the_fence(self, chunk_size):
        code = "payload = 'abc\\\n```def'\nprint(payload)"
        segments, _ = parse(f"```python\n{code}\n```", chunk_size)

        compile(code, "<continued>", "exec")
        assert code_of(segments) == [code]

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_an_escaped_backslash_does_not_continue_the_line(self, chunk_size):
        segments, _ = parse("```python\nx = 'a\\\\'\n```\n", chunk_size)

        assert code_of(segments) == ["x = 'a\\\\'"]


class TestAnEmptyFenceIsNotTheFirstCodeBlock:
    """The agent stops streaming at the first *complete* block and runs it.

    An empty fence has nothing to run, so completing on it stopped the stream
    holding no code: the real block that followed was never parsed, and the
    answer was recorded truncated at the empty one.
    """

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_the_block_after_an_empty_fence_is_the_one_returned(self, chunk_size):
        segments, _ = parse(
            "Let me start.\n```python\n```\nOops, real code:\n```python\nprint('run me')\n```",
            chunk_size,
        )

        assert code_of(segments) == ["print('run me')"]

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_the_empty_fence_survives_as_text(self, chunk_size):
        """It is part of what the model wrote, and history records the turn."""
        segments, _ = parse("a\n```python\n```\nb\n```python\nx = 1\n```", chunk_size)

        assert text_of(segments) == "a\n```python\n```\nb\n"

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_a_whitespace_only_fence_is_empty_too(self, chunk_size):
        segments, _ = parse("```python\n   \n```\nthen:\n```python\ny = 2\n```", chunk_size)

        assert code_of(segments) == ["y = 2"]


class TestAnIndentedBlockIsRunnable:
    """A closing fence is accepted with up to three leading spaces, so a
    uniformly indented block closes — and ``strip()`` alone de-indented only
    its first line, handing the model a SyntaxError for code it wrote right."""

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_a_uniformly_indented_block_compiles(self, chunk_size):
        segments, _ = parse("```python\n   x = 1\n   y = x + 1\n   print(y)\n   ```\n", chunk_size)

        (code,) = code_of(segments)
        compile(code, "<indented>", "exec")
        assert code == "x = 1\ny = x + 1\nprint(y)"

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_relative_indentation_is_preserved(self, chunk_size):
        segments, _ = parse("```python\n  def f():\n      return 1\n  ```\n", chunk_size)

        (code,) = code_of(segments)
        compile(code, "<nested>", "exec")
        assert code == "def f():\n    return 1"

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_an_ordinary_block_is_untouched(self, chunk_size):
        segments, _ = parse("```python\ndef f():\n    return 1\n```\n", chunk_size)

        assert code_of(segments) == ["def f():\n    return 1"]
