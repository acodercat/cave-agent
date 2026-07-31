"""Token estimation and threshold calculation."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# Estimation rates, in **tokens per character** — the inverse form (chars per
# token) cannot express a rate above 1 token/char, and CJK is one.
#
# Measured with ``tiktoken``:
#
#   text                     cl100k   o200k
#   English prose             0.19     0.19
#   Python source             0.39     0.39
#   JSON                      0.43     0.43
#   Chinese prose             0.97     0.59
#   Chinese, short/repeated   1.25     0.50
#   Japanese / Korean         0.71     0.62
#   CJK supplementary planes  3.00-4.00 3.00-4.00
#
# Rates below this line are per *script* character — a sample's whole token
# count over only its non-Latin characters, so the spaces and punctuation
# around them are not credited with absorbing any of it. Measuring them over
# the mixed sample instead reads 2x low, because the ASCII in a natural
# sentence is charged the Latin rate and dilutes the average:
#
#   Cyrillic                  0.54     0.23
#   Arabic                    0.85     0.29
#   Greek / Thai              1.02     0.43
#   Hebrew                    1.23     0.33
#   Devanagari                1.24     0.35
#   Bengali                   1.47     0.29
#   Tamil                     1.65     0.42
#   Tibetan / Myanmar         2.09     1.53
#   Georgian / Armenian       2.17     0.32
#   Emoji, incl. ZWJ sequences 2.60    1.60
#   Ethiopic                  3.00     1.78
#   Cherokee                  3.27     2.57
#
# Each rate takes the dense end of its span, not the average, because the two
# directions are not symmetric: over-estimating compacts a little early,
# under-estimating lets the prompt cross the real limit and the API rejects it.
# The supplementary planes get their own rate because no vocabulary in this
# class has room for those glyphs — rate *classes* rather than one CJK flag is
# what makes that expressible.
# Three tiers rather than one "not Latin, not CJK" rate: the span runs from
# Cyrillic at 0.54 to Cherokee at 3.27, and a single rate covering the dense
# end would charge Russian six times what it costs, compacting those
# conversations long before they need it. Over-estimating is the safe
# direction, not a free one — the tiers keep every script within ~2x.
DEFAULT_TOKENS_PER_CHAR = 0.25
LIGHT_NON_LATIN_TOKENS_PER_CHAR = 1.25
DENSE_NON_LATIN_TOKENS_PER_CHAR = 1.75
RARE_SCRIPT_TOKENS_PER_CHAR = 3.5
CJK_TOKENS_PER_CHAR = 1.25
SUPPLEMENTARY_CJK_TOKENS_PER_CHAR = 4.0

# Headroom between the compaction trigger and the output reserve. Compaction
# is checked *before* a call, so the turn that trips it still has to fit; this
# absorbs that turn's growth plus estimator error.
COMPACT_BUFFER_TOKENS = 13_000

# Unicode ranges of dense-tokenization CJK characters, and the single source
# the classifier below is built from — a range added here takes effect
# everywhere.
#
# Hangul is included (Korean tokenizes similarly under the same vocabularies),
# and so is CJK punctuation: fullwidth ，？！ measure ~1 token each, and
# omitting them dragged the estimate below the true count on ordinary Chinese
# prose. One separator per clause is not a rounding error.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),  # CJK Symbols and Punctuation (、。《》)
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),  # Halfwidth and Fullwidth Forms (，？！ and halfwidth kana)
)

# Rarer ideographs, outside the Basic Multilingual Plane. Separate because they
# tokenize 3x denser than the BMP CJK above — see the rate table.
_SUPPLEMENTARY_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x20000, 0x2FFFF),  # CJK Extensions B-F and Compatibility Supplement
)

# Alphabets and abjads with enough vocabulary coverage to stay near one token
# per character.
_LIGHT_NON_LATIN_RANGES: tuple[tuple[int, int], ...] = (
    (0x0370, 0x03FF),  # Greek and Coptic
    (0x0400, 0x052F),  # Cyrillic and Cyrillic Supplement
    (0x0600, 0x06FF),  # Arabic
    (0x0700, 0x077F),  # Syriac, Arabic Supplement
    (0x0780, 0x07BF),  # Thaana
    (0x0E00, 0x0EFF),  # Thai, Lao
)

# Scripts whose combining marks and conjuncts fragment a syllable further.
_DENSE_NON_LATIN_RANGES: tuple[tuple[int, int], ...] = (
    (0x0590, 0x05FF),  # Hebrew
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x0BFF),  # Bengali, Gurmukhi, Gujarati, Oriya, Tamil
    (0x0C00, 0x0D7F),  # Telugu, Kannada, Malayalam
    (0x0D80, 0x0DFF),  # Sinhala
)

# Scripts and pictographs with little or no dedicated vocabulary space, where
# a character costs two to three tokens on its own. Emoji joiners are here
# too: a ZWJ sequence is charged per component, and the joiners are components
# the vocabulary also has to spell out.
_RARE_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0530, 0x058F),  # Armenian
    (0x0F00, 0x0FFF),  # Tibetan
    (0x1000, 0x109F),  # Myanmar
    (0x10A0, 0x10FF),  # Georgian
    (0x1200, 0x139F),  # Ethiopic
    (0x13A0, 0x13FF),  # Cherokee
    (0x200D, 0x200D),  # Zero-width joiner
    (0x2600, 0x27BF),  # Miscellaneous Symbols, Dingbats
    (0xFE00, 0xFE0F),  # Variation selectors
    (0x1F000, 0x1F2FF),  # Tiles, cards, enclosed alphanumerics
    (0x1F300, 0x1FAFF),  # Emoticons, pictographs, transport, symbols A-B
)


def _character_class(ranges: tuple[tuple[int, int], ...]) -> re.Pattern[str]:
    """One C-level regex scan in place of a per-character Python loop."""
    return re.compile("[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in ranges) + "]")


# Every character class that is not charged the default rate. Adding a class is
# a row here; nothing else changes.
# The classes must stay disjoint: each pattern's matches are counted and then
# subtracted from the default-rate remainder, so an overlap charges twice.
_RATED_CLASSES: tuple[tuple[float, re.Pattern[str]], ...] = (
    (CJK_TOKENS_PER_CHAR, _character_class(_CJK_RANGES)),
    (SUPPLEMENTARY_CJK_TOKENS_PER_CHAR, _character_class(_SUPPLEMENTARY_CJK_RANGES)),
    (LIGHT_NON_LATIN_TOKENS_PER_CHAR, _character_class(_LIGHT_NON_LATIN_RANGES)),
    (DENSE_NON_LATIN_TOKENS_PER_CHAR, _character_class(_DENSE_NON_LATIN_RANGES)),
    (RARE_SCRIPT_TOKENS_PER_CHAR, _character_class(_RARE_SCRIPT_RANGES)),
)


def compact_threshold(context_window: int, output_reserve: int) -> int:
    """Token count at which compaction should trigger.

    Reserves room for the next completion so a prompt prepared just under the
    line still satisfies the wire constraint ``input + max_output <= window``.
    For windows too small to hold both reserves, falls back to half the window.
    """
    threshold = context_window - output_reserve - COMPACT_BUFFER_TOKENS
    if threshold <= 0:
        threshold = context_window // 2
    return threshold


def default_token_estimate(text: str) -> int:
    """Estimate tokens for a string, charging each character its class rate.

    Every character is counted at its own rate, with no density threshold: a
    mixed message — a Chinese question about an English stack trace, say — is
    charged per class rather than billed entirely at whichever class dominates.
    Summing over rate classes costs one regex scan each, so a threshold would
    buy nothing.
    """
    if not text:
        return 0
    total = 0.0
    remaining = len(text)
    for rate, pattern in _RATED_CLASSES:
        matched = len(pattern.findall(text))
        total += matched * rate
        remaining -= matched
    return int(total + remaining * DEFAULT_TOKENS_PER_CHAR)


def tiktoken_estimator(encoding: str = "o200k_base") -> Callable[[str], int]:
    """A real tokenizer for ``Compactor(token_estimator=...)``, if you have one.

    Opt-in, because ``tiktoken`` is an extra dependency that downloads a vocab
    on first use, and it is only *exactly* right for OpenAI models — every other
    provider tokenizes differently, so for them it is a better estimate, not a
    measurement. Raises ``ImportError`` with the install hint when absent, at
    call time rather than at import, so the default path never pays for it.
    """
    try:
        import tiktoken
    except ImportError as error:
        raise ImportError("tiktoken_estimator needs tiktoken: pip install tiktoken") from error

    encoder = tiktoken.get_encoding(encoding)
    return lambda text: len(encoder.encode(text, disallowed_special=()))


@dataclass(frozen=True)
class TokenAnchor:
    """An API-measured prompt size, paired with the heuristic at that moment.

    The pair is the whole point. An API count alone says how big the prompt
    *was*; only the difference between two heuristic readings says how much
    has been added since. Carrying them separately is what let them drift.
    """

    api_tokens: int
    heuristic_tokens: int


def estimate_tokens(
    messages: list,
    anchor: TokenAnchor | None = None,
    token_estimator: Callable[[str], int] | None = None,
) -> int:
    """Estimate token count for *messages*, anchored to a real API count.

    Per-message counts use *token_estimator* when supplied (e.g. a real
    tokenizer), else the CJK-aware heuristic. With an *anchor*, the result is
    the measured count plus the heuristic growth since it was taken.

    This used to be ``max(heuristic, api_count)``, which sounded like it
    covered messages appended after the measurement but did not: the API count
    of a real conversation dwarfs the heuristic, so ``max`` returned it
    unchanged and every subsequent turn — including a multi-thousand-token
    execution result — was invisible until the bare heuristic alone overtook
    it. Compaction was then decided on a number that had stopped moving.

    A shrunken history invalidates its own anchor: compaction rewrites
    ``messages``, and a count taken against the old list describes nothing in
    the new one. Detecting that here, rather than asking every rewrite site to
    remember to clear it, is what keeps the invalidation from being forgotten.
    """
    count = token_estimator or default_token_estimate
    heuristic = sum(count(msg.content) for msg in messages)
    if anchor is None or anchor.api_tokens <= 0:
        return heuristic
    if heuristic < anchor.heuristic_tokens:
        return heuristic
    return anchor.api_tokens + (heuristic - anchor.heuristic_tokens)
