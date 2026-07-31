import io
import textwrap
import tokenize
from enum import Enum


class SegmentType(Enum):
    """Types of content segments."""

    TEXT = "text"
    CODE = "code"


class Segment:
    """Represents a parsed content segment."""

    def __init__(self, segment_type: SegmentType, content: str) -> None:
        self.type = segment_type
        self.content = content


class StreamingTextParser:
    """
    Parser for streaming text that identifies Python code blocks.

    This parser processes text character-by-character, immediately streaming
    regular text while buffering code blocks delimited by triple backticks.

    Attributes:
        TRIPLE_BACKTICK_COUNT: Number of backticks that delimit code blocks
    """

    TRIPLE_BACKTICK_COUNT = 3

    class Mode(Enum):
        """Parser state machine modes."""

        TEXT = "text"  # Normal text processing
        BACKTICK_COUNT = "backtick_count"  # Counting consecutive backticks
        LANGUAGE_MATCH = "language_match"  # Matching language identifier
        CODE = "code"  # Inside code block
        CODE_END_CHECK = "code_end_check"  # Checking for code block end

    def __init__(self, language_identifier: str = "python") -> None:
        """
        Initialize the parser.

        Args:
            language_identifier: Language identifier for code blocks (default: "python")
        """
        self.language_identifier = language_identifier
        self._reset_state()
        self._handlers = {
            self.Mode.TEXT: self._handle_text_mode,
            self.Mode.BACKTICK_COUNT: self._handle_backtick_count_mode,
            self.Mode.LANGUAGE_MATCH: self._handle_language_match_mode,
            self.Mode.CODE: self._handle_code_mode,
            self.Mode.CODE_END_CHECK: self._handle_code_end_check_mode,
        }

    def process_chunk(self, chunk: str) -> list[Segment]:
        """
        Process a chunk of streaming text, stopping at the first closed fence.

        Parsing halts on the character that closes the first code block. Text
        after that point is left unconsumed in :attr:`remainder` — it belongs
        to a turn the agent has not asked for yet, and consuming it is how a
        response carrying two fences in one delta came to execute the *second*
        block: the loop kept parsing, and each ``CODE`` segment overwrote the
        last. Chunk size is a property of the transport, so nothing that
        depends on where a delta happens to split can be correct.

        Args:
            chunk: Text chunk to process

        Returns:
            Segments parsed from the consumed prefix of *chunk*
        """
        if self.first_code_block_completed:
            self.remainder = chunk
            return []

        self.remainder = ""
        parsed_segments = []

        for index, char in enumerate(chunk):
            parsed_segments.extend(self._process_character(char))
            if self.first_code_block_completed:
                self.remainder = chunk[index + 1 :]
                break

        return parsed_segments

    def flush(self) -> list[Segment]:
        """
        Flush all buffers and return remaining segments.

        Should be called when the stream ends to handle any incomplete
        parsing state and return remaining buffered content.

        Returns:
            List of remaining parsed segments
        """
        segments = []

        # Handle incomplete parsing states
        if self.mode == self.Mode.BACKTICK_COUNT:
            # Incomplete backtick sequence - treat as text
            self.text_buffer += "`" * self.backtick_count

        elif self.mode == self.Mode.LANGUAGE_MATCH:
            # Incomplete language match - treat as text
            self.text_buffer += "```" + self.language_match_buffer

        elif self.mode == self.Mode.CODE_END_CHECK:
            # Incomplete code end check - add to code buffer
            self.code_buffer += "`" * self.backtick_count

        # Emit an incomplete code block only after its pending parser state has
        # been folded into the buffer. Emitting before and after that fold
        # duplicated the whole block when the stream ended on one or two
        # possible closing backticks.
        if self.in_code_block and self.code_buffer.strip():
            segments.append(Segment(SegmentType.CODE, self._finished_code()))

        # Flush any remaining text buffer
        if self.text_buffer:
            segments.append(Segment(SegmentType.TEXT, self.text_buffer))

        self._reset_state()
        return segments

    def _reset_state(self) -> None:
        """Reset parser to initial state.

        ``first_code_block_completed`` is part of that state: it now gates
        whether characters are consumed at all, so a parser left flagged after
        a flush would silently refuse every later chunk.
        """
        self.mode = self.Mode.TEXT
        self.text_buffer = ""
        self.code_buffer = ""
        self.backtick_count = 0
        self.language_match_buffer = ""
        self.in_code_block = False
        self.first_code_block_completed = False
        self.remainder = ""
        # The opening fence as written, replayed as text if the block is empty.
        self._code_block_opening = ""
        # Number of leading spaces on the current code line. ``None`` once the
        # line contains code or exceeds Markdown's three-space fence indent.
        self._code_line_indent: int | None = 0

    def _process_character(self, char: str) -> list[Segment]:
        """
        Process a single character based on current mode.

        Args:
            char: Character to process

        Returns:
            List of segments generated from this character
        """
        handler = self._handlers.get(self.mode)
        return handler(char) if handler else []

    def _handle_text_mode(self, char: str) -> list[Segment]:
        """
        Handle character in TEXT mode.

        In text mode, characters are streamed immediately unless
        a backtick is encountered, which triggers backtick counting.
        """
        if char == "`":
            segments = []
            # Flush any buffered text before switching modes
            if self.text_buffer:
                segments.append(Segment(SegmentType.TEXT, self.text_buffer))
                self.text_buffer = ""
            self.mode = self.Mode.BACKTICK_COUNT
            self.backtick_count = 1
            return segments
        else:
            # Stream text character immediately
            return [Segment(SegmentType.TEXT, char)]

    def _handle_backtick_count_mode(self, char: str) -> list[Segment]:
        """
        Handle character in BACKTICK_COUNT mode.

        Counts consecutive backticks to detect triple-backtick sequences
        that might start a code block.
        """
        if char == "`":
            self.backtick_count += 1
            if self.backtick_count == self.TRIPLE_BACKTICK_COUNT:
                # Triple backticks detected - check for language identifier
                segments = []
                if self.text_buffer:
                    segments.append(Segment(SegmentType.TEXT, self.text_buffer))
                    self.text_buffer = ""
                self.mode = self.Mode.LANGUAGE_MATCH
                self.language_match_buffer = ""
                self.backtick_count = 0
                return segments
        else:
            # Not consecutive backticks - emit as text
            segments = [Segment(SegmentType.TEXT, "`" * self.backtick_count + char)]
            self.mode = self.Mode.TEXT
            self.backtick_count = 0
            return segments

        return []

    def is_first_code_block_completed(self) -> bool:
        """Check if the first code block is completed."""
        return self.first_code_block_completed

    def _handle_language_match_mode(self, char: str) -> list[Segment]:
        """
        Handle character in LANGUAGE_MATCH mode.

        Attempts to match the language identifier after triple backticks.
        If matched, enters code block mode; otherwise, treats as text.
        """

        # Helper to handle failed match
        def failed_match() -> list[Segment]:
            segments = [Segment(SegmentType.TEXT, "```" + self.language_match_buffer + char)]
            self.mode = self.Mode.TEXT
            self.language_match_buffer = ""
            return segments

        # Still building the language identifier
        if len(self.language_match_buffer) < len(self.language_identifier):
            if char == self.language_identifier[len(self.language_match_buffer)]:
                self.language_match_buffer += char
                if self.language_match_buffer == self.language_identifier:
                    # Full match achieved, wait for delimiter
                    pass
                return []
            else:
                return failed_match()

        # Language identifier matched, check for valid delimiter
        else:
            if char in ("\n", " ", "\r"):
                # Valid code block start
                self._enter_code_block(char)
                # Don't add delimiter to code buffer
                return []
            else:
                # Invalid delimiter (e.g., ```pythonscript)
                return failed_match()

    def _handle_code_mode(self, char: str) -> list[Segment]:
        """
        Handle character in CODE mode.

        Accumulates code content until a backtick at the start of a line might
        indicate the end of the code block.
        """
        if char == "`" and self._code_line_indent is not None:
            self.mode = self.Mode.CODE_END_CHECK
            self.backtick_count = 1
        else:
            self._append_code_character(char)
        return []

    def _handle_code_end_check_mode(self, char: str) -> list[Segment]:
        """
        Handle character in CODE_END_CHECK mode.

        Checks if we have triple backticks that end the code block,
        or if it's just backticks within the code content.
        """
        if char == "`":
            self.backtick_count += 1
            if self.backtick_count == self.TRIPLE_BACKTICK_COUNT:
                if self._inside_multiline_string():
                    for _ in range(self.TRIPLE_BACKTICK_COUNT):
                        self._append_code_character("`")
                    self.mode = self.Mode.CODE
                    self.backtick_count = 0
                    return []

                if not self.code_buffer.strip():
                    # An empty fence is not the block the agent is waiting for.
                    # Completing on it stopped the stream with no code to run,
                    # so the real block that followed was never parsed and the
                    # answer was truncated at the empty one. Give the fence
                    # back as text and keep looking.
                    self.code_buffer = ""
                    self._exit_code_block()
                    return [Segment(SegmentType.TEXT, self._code_block_opening + "```")]

                # Code block ends
                segments = [Segment(SegmentType.CODE, self._finished_code())]
                self.code_buffer = ""
                self._exit_code_block()
                self.first_code_block_completed = True
                return segments
        else:
            # Not a code block end - backticks are part of code content
            for _ in range(self.backtick_count):
                self._append_code_character("`")
            self._append_code_character(char)
            self.mode = self.Mode.CODE
            self.backtick_count = 0

        return []

    # tokenize reports a string left open at EOF two ways. The triple-quoted
    # forms are unambiguous; the single-line forms are not — they are also what
    # it says for ``print('it's fine')``, where the string died at the newline
    # and the fence that follows is real.
    _MULTILINE_STRING_MARKERS = ("multi-line string", "triple-quoted")
    _OPEN_STRING_MARKERS = ("unterminated string literal", "unterminated f-string literal")

    def _inside_multiline_string(self) -> bool:
        """Whether the possible fence belongs to a string that is still open.

        Only a string spanning lines can swallow a fence. A single-quoted string
        spans one when the line ends in a backslash, so for those markers the
        continuation is the deciding evidence rather than the message: without
        that check ``print('it's fine')`` swallowed its own closing fence and
        the block never completed.

        Any other tokenize failure means the buffer is not valid Python, which
        is a different question: the fence closes, the code runs, and the model
        gets a ``SyntaxError`` it can act on. That includes ``IndentationError``
        — a ``SyntaxError``, not a ``TokenError`` — which escaped this method
        entirely and crashed the run.
        """
        try:
            list(tokenize.generate_tokens(io.StringIO(self.code_buffer).readline))
        except tokenize.TokenError as error:
            reason = str(error.args[0]).lower()
            if any(marker in reason for marker in self._MULTILINE_STRING_MARKERS):
                return True
            if any(marker in reason for marker in self._OPEN_STRING_MARKERS):
                return self._ends_with_line_continuation()
            return False
        except SyntaxError:
            return False
        return False

    def _ends_with_line_continuation(self) -> bool:
        """Whether the code so far ends mid-line, joined to the next one.

        An odd number of trailing backslashes continues the line; an even
        number is escaped backslashes and ends it.
        """
        line = self.code_buffer.rstrip("\r\n")
        return (len(line) - len(line.rstrip("\\"))) % 2 == 1

    def _finished_code(self) -> str:
        """The buffered block as runnable source.

        A closing fence is accepted with up to three leading spaces, so a
        uniformly indented block closes correctly — and ``strip()`` alone
        de-indents only its first line, handing the model a
        ``SyntaxError: unexpected indent`` for code it wrote correctly.
        ``dedent`` removes the whole block's common prefix and is a no-op on
        ordinary unindented code.
        """
        return textwrap.dedent(self.code_buffer).strip()

    def _append_code_character(self, char: str) -> None:
        """Append code while tracking whether a closing fence may start here."""
        self.code_buffer += char
        if char == "\n":
            self._code_line_indent = 0
        elif self._code_line_indent is not None and char == " " and self._code_line_indent < 3:
            self._code_line_indent += 1
        else:
            self._code_line_indent = None

    def _enter_code_block(self, delimiter: str) -> None:
        """Enter code block mode and reset temporary buffers.

        The opening is recorded verbatim so an empty fence can be handed back
        as the text the model actually wrote.
        """
        self.in_code_block = True
        self.mode = self.Mode.CODE
        self.language_match_buffer = ""
        self._code_line_indent = 0
        self._code_block_opening = f"```{self.language_identifier}{delimiter}"

    def _exit_code_block(self) -> None:
        """Exit code block mode and reset counters."""
        self.in_code_block = False
        self.mode = self.Mode.TEXT
        self.backtick_count = 0
