"""Token estimation and threshold calculation."""

# ~4 chars per token is the standard rough estimate for English text + code.
CHARS_PER_TOKEN = 4
# CJK characters tokenize at ~2 chars/token under cl100k/o200k/Claude/Gemini
# vocabularies — the 100k-class vocab can't afford one token per Han glyph, so
# most CJK chars split into 2-3 BPE pieces. Using the English ratio (4) for
# Chinese/Japanese/Korean under-estimates by 2-4×, which matters on the first
# turn before any API ``usage`` field arrives to correct course.
CHARS_PER_CJK_TOKEN = 2
# CJK ratio above which the mixed-language split kicks in. Below it, treat the
# text as effectively English and accept the small error from a few stray CJK
# chars (e.g. one Chinese name in an English doc). 30% is the common heuristic;
# not load-bearing — tune if real workloads show a different inflection.
CJK_RATIO_THRESHOLD = 0.3
OUTPUT_TOKENS_RESERVE = 16_000
COMPACT_BUFFER_TOKENS = 13_000

# Unicode ranges of dense-tokenization CJK characters. Hangul is included
# (Korean tokenizes similarly under the same vocabularies).
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xAC00, 0xD7AF),   # Hangul Syllables
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x20000, 0x2FFFF),  # CJK Extensions B-F (supplementary plane)
)

# Lowest codepoint of any CJK range — below it no character can be CJK, which
# lets ``default_token_estimate`` short-circuit all-Latin text cheaply.
_CJK_MIN = min(low for low, _ in _CJK_RANGES)


def compact_threshold(context_window: int) -> int:
    """Calculate the token count at which compaction should trigger.

    For small context windows where reserves exceed the window size,
    falls back to 50% of the window as the threshold.
    """
    threshold = context_window - OUTPUT_TOKENS_RESERVE - COMPACT_BUFFER_TOKENS
    if threshold <= 0:
        threshold = context_window // 2
    return threshold


def _is_cjk(codepoint: int) -> bool:
    """Whether ``codepoint`` falls in any of the dense-tokenization CJK ranges."""
    for low, high in _CJK_RANGES:
        if low <= codepoint <= high:
            return True
    return False


def default_token_estimate(text: str) -> int:
    """Estimate tokens for a string, with CJK-density awareness.

    Pure-English text is well-modeled by ``len // 4``; pure-CJK text by
    ``len // 2``. When CJK density crosses :data:`CJK_RATIO_THRESHOLD`, switch
    to a per-class split (CJK chars at 2 each, the rest at 4 each); below it,
    accept the small error from a few stray CJK chars. Matters most on the
    first turn, before an API ``usage`` count is available to correct course —
    the one path where under-estimating CJK by 2-4× could silently push the
    conversation past its compaction threshold and into a context overflow.
    """
    n = len(text)
    if n == 0:
        return 0
    # Fast path for the common all-Latin/ASCII case: ``max`` is a single C-level
    # scan, so text with no character up in the CJK planes skips the per-char
    # Python loop below entirely (this runs over the whole history each estimate).
    if ord(max(text)) < _CJK_MIN:
        return n // CHARS_PER_TOKEN
    cjk_count = sum(1 for ch in text if _is_cjk(ord(ch)))
    if cjk_count >= n * CJK_RATIO_THRESHOLD:
        return cjk_count // CHARS_PER_CJK_TOKEN + (n - cjk_count) // CHARS_PER_TOKEN
    return n // CHARS_PER_TOKEN


def estimate_tokens(
    messages: list,
    api_token_count: int | None = None,
) -> int:
    """Estimate token count for *messages*.

    Per-message counts use :func:`default_token_estimate` (CJK-aware). When a
    real API-reported count is supplied, return the larger of it and the
    heuristic: the API count reflects the prompt at the time of the last call,
    while messages appended since then are only captured by the heuristic, so
    the max avoids under-counting (which would delay compaction and risk
    overflowing the context window).
    """
    heuristic = sum(default_token_estimate(msg.content) for msg in messages)
    if api_token_count is not None and api_token_count > 0:
        return max(heuristic, api_token_count)
    return heuristic
