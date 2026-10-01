"""Text normalization and the matcher that compares a place answer with a hierarchy.

An answer matches a hierarchy entry if the two are equal, if one contains the other
as whole words, or if the answer contains all content words of the entry.
"""

import re
import string

_ARTICLES = {"the", "a", "an"}
_FILLER = {"of", "the", "a", "an", "and"}


def normalize(s: str) -> str:
    s = s.lower().strip()
    # Punctuation becomes a space, not nothing, so "Baden-Württemberg" stays two words.
    s = s.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    while tokens and tokens[0] in _ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


def content_tokens(s: str) -> set[str]:
    return {t for t in normalize(s).split() if t and t not in _FILLER}


def match_at_level(answer: str, hierarchy_entry: str) -> bool:
    if not answer or not hierarchy_entry:
        return False
    na, nh = normalize(answer), normalize(hierarchy_entry)
    if not na or not nh:
        return False
    if na == nh:
        return True
    # Containment only at word boundaries, so "US" does not match inside "Russia".
    if f" {nh} " in f" {na} " or f" {na} " in f" {nh} ":
        return True
    h_tok, a_tok = content_tokens(hierarchy_entry), content_tokens(answer)
    if h_tok and h_tok.issubset(a_tok):
        return True
    return False


def find_match_level(answer: str, hierarchy: list[str]) -> int | None:
    """Most-specific level index `answer` matches, or None if off-hierarchy."""
    for idx, h in enumerate(hierarchy):
        if match_at_level(answer, h):
            return idx
    return None


def find_match_level_v3(answer: str, hierarchy: list[str]) -> int | None:
    """Like find_match_level, but an exact match at any level wins over a looser
    match at a finer level.

    Without this, "Austria" against [Wiener Neustadt, Lower Austria, Austria] is
    credited at the region level, because "Austria" is contained in "Lower Austria".
    This is the matcher used for all scoring."""
    na = normalize(answer or "")
    if na:
        for idx, h in enumerate(hierarchy):
            if na == normalize(h):
                return idx
    return find_match_level(answer, hierarchy)


def match_level(answer: str, hierarchy: list[str]) -> int | None:
    """find_match_level_v3, or the older single-pass rule if CORRECTED_SCORING is off."""
    from .config import CORRECTED_SCORING
    fn = find_match_level_v3 if CORRECTED_SCORING else find_match_level
    return fn(answer, hierarchy)
