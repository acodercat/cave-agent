"""Differential tests: the streaming parser against the shipped 0.x extractor.

Four of round-14's defects were regressions — behaviour `utils.extract_python_code`
(deleted in the rewrite) got right and the character-level parser lost, because
the old implementation's one-line fence test encoded three decisions at once
(line anchoring, case folding, whole-line matching) and none of them survived
as an explicit rule. Review cannot catch this class: the diff shows a file
replaced, not a semantic dropped. So the old implementation is vendored here
as an oracle and the two are compared over a generated corpus.

The comparison holds everywhere except where the rewrite *deliberately*
diverges. Each divergence is a decision with a reason, and the corpus
generator stays out of those zones — targeted tests elsewhere pin them:

- Unterminated fence at a *truncated* stream: old executed the fragment; the
  agent now routes it to output recovery. (At a stream the provider declared
  complete, both execute the fragment — that is the model's answer.)
- Multiple blocks: old concatenated and executed all; the agent executes the
  first and leaves the rest for a turn not yet asked for.
- Fence inside an open multi-line string: old closed there, truncating the
  string; tokenize-guarded parsing keeps it as code.
- Fence indented four or more spaces (either direction): old accepted any
  indent; Markdown says over three is not a fence.
- Closing fence with trailing text (``` and then): Markdown says not a
  closing fence; the parser closes anyway, because a model writing text after
  ``` almost always means "closed", and a genuine ```-led line inside valid
  Python can only occur inside a string, which the tokenize guard already
  keeps. Old kept it as code by whole-line matching.
- An opening fence's info string (```python copy): old treated the whole
  line as prose and lost the block; the parser discards the info string and
  takes the code from the next line, per Markdown.
- Uniform indentation: the parser dedents so the block compiles; old kept
  the indent. Compared modulo dedent below.
"""

import random
import textwrap

import pytest

from cave_agent.parsing import SegmentType, StreamingTextParser

# --- Oracle: verbatim from 0.x `utils.extract_python_code` (deleted), ---------
# reduced to its inner loop so the *first* block is observable. The shipped
# function joined all blocks with a blank line; nothing else is changed.


def extract_python_blocks(response: str) -> list[str]:
    results: list[str] = []
    lines = response.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].strip().lower() == "```python":
            code_lines: list[str] = []
            i += 1
            while i < len(lines):
                if lines[i].strip() == "```":
                    break
                code_lines.append(lines[i])
                i += 1
            code = "\n".join(code_lines).rstrip()
            if code:
                results.append(code)
        i += 1
    return results


def first_block(response: str, chunk_size: int) -> str | None:
    """The block the agent would execute, streamed at *chunk_size*."""
    parser = StreamingTextParser()
    segments = []
    for start in range(0, len(response), chunk_size):
        segments.extend(parser.process_chunk(response[start : start + chunk_size]))
        if parser.is_first_code_block_completed():
            break
    else:
        segments.extend(parser.flush())
    codes = [s.content for s in segments if s.type == SegmentType.CODE]
    return codes[0] if codes else None


# --- Corpus: compositions inside the agreed-semantics zone --------------------

PROSE = [
    "Here is the plan.",
    "First, look at `df.head()` output.",
    "Use ``inline double`` spans.",
    "Wrap code in ```python fences like this:",
    "Results:",
    "",
    "  indented prose",
    "a line with ` one backtick",
    "done ```",
]
OPENERS = ["```python", "```Python", "```PYTHON", " ```python", "   ```python"]
BODIES = [
    "x = 1",
    "print('hello')",
    "def f():\n    return 1",
    "   x = 1\n   y = 2",
    "data = {'a': 1}",
    "for i in range(3):\n    print(i)",
    "s = 'it''s'",
    "print('a `tick`')",
]
CLOSERS = ["```", " ```", "   ```"]
CHUNK_SIZES = [1, 7, 10_000]


def _corpus(seed: int, size: int) -> list[str]:
    rng = random.Random(seed)
    responses = []
    for _ in range(size):
        parts: list[str] = []
        blocks = rng.randint(0, 3)
        for index in range(max(1, blocks + rng.randint(0, 2))):
            parts.extend(rng.choice(PROSE) for _ in range(rng.randint(0, 3)))
            if index < blocks:
                parts.append(rng.choice(OPENERS))
                parts.append(rng.choice(BODIES))
                parts.append(rng.choice(CLOSERS))
        responses.append("\n".join(parts) + rng.choice(["", "\n", "\ntrailing text"]))
    return responses


class TestParserAgreesWithTheShippedExtractor:
    """Every generated response, at every chunk size, modulo dedent."""

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_first_block_matches_the_oracle(self, chunk_size):
        for response in _corpus(seed=0xCAFE, size=1500):
            olds = extract_python_blocks(response)
            expected = textwrap.dedent(olds[0]).strip() if olds else None

            assert first_block(response, chunk_size) == expected, repr(response)

    def test_output_is_chunk_size_invariant(self):
        """Chunk size is a property of the transport; nothing may vary with it."""
        for response in _corpus(seed=0xBEEF, size=500):
            results = {first_block(response, size) for size in CHUNK_SIZES}

            assert len(results) == 1, repr(response)


class TestTheInfoStringIsDiscardedNotExecuted:
    """```python copy — Markdown's info string. Treating the remainder of the
    fence line as code executed whatever trailed the tag; the old extractor
    dropped the whole block as prose. Code starts on the next line."""

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_code_starts_on_the_line_after_the_fence(self, chunk_size):
        assert first_block("```python copy\ny = 2\n```\n", chunk_size) == "y = 2"

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_trailing_spaces_alone_are_an_info_string_too(self, chunk_size):
        assert first_block("```python   \nx = 1\n```\n", chunk_size) == "x = 1"

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_a_stream_ending_on_the_fence_line_replays_it_as_text(self, chunk_size):
        parser = StreamingTextParser()
        segments = []
        text = "so:\n```python copy"
        for start in range(0, len(text), chunk_size):
            segments.extend(parser.process_chunk(text[start : start + chunk_size]))
        segments.extend(parser.flush())

        assert [s.content for s in segments if s.type == SegmentType.CODE] == []
        assert "".join(s.content for s in segments) == text
