from enum import Enum
from typing import List


class SegmentType(Enum):
    """Types of content segments."""
    TEXT = "text"
    CODE = "code"


class Segment:
    """Represents a parsed content segment."""
    
    def __init__(self, type: SegmentType, content: str) -> None:
        self.type = type
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
        TEXT = "text"                    # Normal text processing
        BACKTICK_COUNT = "backtick_count"  # Counting consecutive backticks
        LANGUAGE_MATCH = "language_match"   # Matching language identifier
        CODE = "code"                    # Inside code block
        CODE_END_CHECK = "code_end_check"  # Checking for code block end
    
    def __init__(self, language_identifier: str = "python") -> None:
        """
        Initialize the parser.
        
        Args:
            language_identifier: Language identifier for code blocks (default: "python")
        """
        self.language_identifier = language_identifier
        self.first_code_block_completed = False
        self._reset_state()
        
    def process_chunk(self, chunk: str) -> List[Segment]:
        """
        Process a chunk of streaming text.
        
        Args:
            chunk: Text chunk to process
            
        Returns:
            List of parsed segments
        """
        parsed_segments = []
        
        for char in chunk:
            segments = self._process_character(char)
            parsed_segments.extend(segments)
        
        return parsed_segments
    
    def flush(self) -> List[Segment]:
        """
        Flush all buffers and return remaining segments.
        
        Should be called when the stream ends to handle any incomplete
        parsing state and return remaining buffered content.
        
        Returns:
            List of remaining parsed segments
        """
        segments = []
        
        # Handle incomplete code block
        if self.in_code_block and self.code_buffer:
            segments.append(Segment(SegmentType.CODE, self.code_buffer.strip()))
        
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
            if self.code_buffer:
                segments.append(Segment(SegmentType.CODE, self.code_buffer.strip()))
        
        # Flush any remaining text buffer
        if self.text_buffer:
            segments.append(Segment(SegmentType.TEXT, self.text_buffer))
        
        self._reset_state()
        return segments
    
    def _reset_state(self) -> None:
        """Reset parser to initial state."""
        self.mode = self.Mode.TEXT
        self.text_buffer = ""
        self.code_buffer = ""
        self.backtick_count = 0
        self.language_match_buffer = ""
        self.in_code_block = False
    
    def _process_character(self, char: str) -> List[Segment]:
        """
        Process a single character based on current mode.
        
        Args:
            char: Character to process
            
        Returns:
            List of segments generated from this character
        """
        # Dispatch to appropriate handler based on current mode
        handlers = {
            self.Mode.TEXT: self._handle_text_mode,
            self.Mode.BACKTICK_COUNT: self._handle_backtick_count_mode,
            self.Mode.LANGUAGE_MATCH: self._handle_language_match_mode,
            self.Mode.CODE: self._handle_code_mode,
            self.Mode.CODE_END_CHECK: self._handle_code_end_check_mode,
        }
        
        handler = handlers.get(self.mode)
        return handler(char) if handler else []
    
    def _handle_text_mode(self, char: str) -> List[Segment]:
        """
        Handle character in TEXT mode.
        
        In text mode, characters are streamed immediately unless
        a backtick is encountered, which triggers backtick counting.
        """
        if char == '`':
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
    
    def _handle_backtick_count_mode(self, char: str) -> List[Segment]:
        """
        Handle character in BACKTICK_COUNT mode.
        
        Counts consecutive backticks to detect triple-backtick sequences
        that might start a code block.
        """
        if char == '`':
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

    def _handle_language_match_mode(self, char: str) -> List[Segment]:
        """
        Handle character in LANGUAGE_MATCH mode.
        
        Attempts to match the language identifier after triple backticks.
        If matched, enters code block mode; otherwise, treats as text.
        """
        # Helper to handle failed match
        def failed_match() -> List[Segment]:
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
            if char in ('\n', ' ', '\r'):
                # Valid code block start
                self._enter_code_block()
                # Don't add delimiter to code buffer
                return []
            else:
                # Invalid delimiter (e.g., ```pythonscript)
                return failed_match()
    
    def _handle_code_mode(self, char: str) -> List[Segment]:
        """
        Handle character in CODE mode.
        
        Accumulates code content until a backtick is found,
        which might indicate the end of the code block.
        """
        if char == '`':
            self.mode = self.Mode.CODE_END_CHECK
            self.backtick_count = 1
        else:
            self.code_buffer += char
        return []
    
    def _handle_code_end_check_mode(self, char: str) -> List[Segment]:
        """
        Handle character in CODE_END_CHECK mode.
        
        Checks if we have triple backticks that end the code block,
        or if it's just backticks within the code content.
        """
        if char == '`':
            self.backtick_count += 1
            if self.backtick_count == self.TRIPLE_BACKTICK_COUNT:
                # Code block ends
                segments = []
                if self.code_buffer:
                    segments.append(Segment(SegmentType.CODE, self.code_buffer.strip()))
                    self.code_buffer = ""
                self._exit_code_block()
                self.first_code_block_completed = True
                return segments
        else:
            # Not a code block end - backticks are part of code content
            self.code_buffer += "`" * self.backtick_count + char
            self.mode = self.Mode.CODE
            self.backtick_count = 0
        
        return []
    
    def _enter_code_block(self) -> None:
        """Enter code block mode and reset temporary buffers."""
        self.in_code_block = True
        self.mode = self.Mode.CODE
        self.language_match_buffer = ""
    
    def _exit_code_block(self) -> None:
        """Exit code block mode and reset counters."""
        self.in_code_block = False
        self.mode = self.Mode.TEXT
        self.backtick_count = 0