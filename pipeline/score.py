"""The scorer. It reads the evidence in base samples, checks training answers, and
grades every evaluated answer. It was checked by hand on 162 answers (results/scorer_check).

Every completion is one of three kinds:
    answered   the answer marker is present with a non-abstention answer
    abstained  an explicit abstention phrase, in the answer or anywhere if no marker
    fmt_fail   anything else, including an empty completion; scored as a wrong answer

Grading is strict: only the answer's primary claim counts (the text before the first
comma for places, the parsed date at its own precision for dates). Evidence is
lenient: a sample supports level L if it agrees with the reference at L, so a date
with the right year and a wrong day is evidence for the year.
"""
import re
from functools import lru_cache

from src.config import LOCATION_RELATIONS
from src.data import load_split
from src.dates import (_PARSER, DATE_LADDER, PRECISION, TemporalLevel, TemporalValue,
                       DATE_NOIDK_PROMPT_TEMPLATE, _snap_to_ladder,
                       credited_level, extract_date_text, level_info, make_date_prompt)
from src.entity import extract_entity_answer, make_entity_prompt
from src.levels import answer_norm_level, build_level_pool, informativeness
from src.prompt import (COUNTRY_PROMPT_TEMPLATE, NOIDK_PROMPT_TEMPLATE, PLAIN_PROMPT_TEMPLATE,
                        extract_answer, finest_element, is_idk, make_prompt)
from src.text import find_match_level_v3

PROMPTS = {
    ("locations", "deploy"): make_prompt,
    ("dates", "deploy"): make_date_prompt,
    ("entity", "deploy"): make_entity_prompt,
    ("locations", "noidk"): lambda q: NOIDK_PROMPT_TEMPLATE.format(q=q),
    ("dates", "noidk"): lambda q: DATE_NOIDK_PROMPT_TEMPLATE.format(q=q),
    ("locations", "bare"): lambda q: PLAIN_PROMPT_TEMPLATE.format(q=q),
    ("locations", "country"): lambda q: COUNTRY_PROMPT_TEMPLATE.format(q=q),
}

# Prose refusals from the bare-question baseline, which has no answer marker.
_REFUSAL = re.compile(
    r"i.?m sorry|i apologize|i (don.?t|do not) have|cannot provide|couldn.?t find|"
    r"no (specific )?information|unable to (find|provide|determine)", re.IGNORECASE)


# A year and nothing else, optionally with "c." or an era marker.
_BARE_YEAR = re.compile(r"^\s*(c\.|ca\.|circa)?\s*\d{1,4}\s*"
                        r"(AD|CE|A\.D\.|BC|BCE|B\.C\.|B\.C\.E\.)?\s*$", re.IGNORECASE)


def parse_date(text: str):
    """Parse a date answer.

    The parser ignores a bare number below 1000 without context ("912"), so a bare
    year is retried as "in <year>". Only bare years: retrying any text would read
    "21 January" as the year 21."""
    v = _PARSER.first_most_specific(text)
    if v is None and _BARE_YEAR.match(text):
        v = _PARSER.first_most_specific("in " + text)
    return v


def prompt_for(item, mode: str = "deploy") -> str:
    return PROMPTS[(item.domain, mode)](item.question)


# The dates are Wikidata values in the proleptic Gregorian calendar, but dates before the
# 1582 reform are conventionally given in the Julian calendar: Erasmus died on 12 July 1536
# and the gold says 22 July. A day-precision gold before 15 October 1582 therefore also
# accepts its Julian rendering. Later Old Style dates (Britain until 1752, Russia until
# 1918) stay unconverted, because both conventions are cited for them; BCE golds too.
_REFORM = (1582, 10, 15)
JULIAN_CREDIT = True     # on for evaluation; the training targets were built with it off


def _jdn(y: int, m: int, d: int) -> int:
    a = (14 - m) // 12
    y, m = y + 4800 - a, m + 12 * a - 3
    return d + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045


def julian_gold(gold):
    """The Julian-calendar rendering of a pre-reform day-precision gold, else None."""
    if (gold.level != TemporalLevel.DAY or gold.year is None or gold.year < 1
            or (gold.year, gold.month, gold.day) >= _REFORM):
        return None
    c = _jdn(gold.year, gold.month, gold.day) + 32082
    d = (4 * c + 3) // 1461
    e = c - 1461 * d // 4
    m = (5 * e + 2) // 153
    return TemporalValue(level=TemporalLevel.DAY, year=d - 4800 + m // 10,
                         month=m + 3 - 12 * (m // 10), day=e - (153 * m + 2) // 5 + 1)


def date_credit(v, gold, strict: bool):
    """credited_level against the gold, or against its Julian rendering, whichever is
    more specific."""
    lvl = credited_level(v, gold, strict)
    jg = julian_gold(gold) if JULIAN_CREDIT else None
    if jg is not None:
        lj = credited_level(v, jg, strict)
        if lj is not None and (lvl is None or ladder_index(lj) < ladder_index(lvl)):
            lvl = lj
    return lvl


@lru_cache(maxsize=None)
def location_pool(source: str = "all") -> dict:
    """Estimated level of each place name, used to score wrong location answers.

    Reads place names from all splits: the splits are divided by country, so a
    train-only pool would miss the test countries. It never uses whether an answer
    is right. 'train' and 'train+test' give the smaller pools."""
    if source == "all":
        return build_level_pool(sum((load_split(s, LOCATION_RELATIONS)
                                     for s in ("train", "val", "test")), []))
    if source == "train":
        return build_level_pool(load_split("train", LOCATION_RELATIONS))
    if source == "train+test":
        return build_level_pool()
    raise ValueError(source)


def extract(item, raw: str, mode: str = "deploy") -> tuple[str, str]:
    """-> (kind, answer text)."""
    raw = raw or ""
    if mode == "bare":
        text = raw.strip().split("\n")[0].strip()
        if not text:
            return "fmt_fail", ""
        return ("abstained" if is_idk(text) or _REFUSAL.search(text) else "answered"), text
    if item.domain == "locations":
        text = extract_answer(raw)
    elif item.domain == "dates":
        text = extract_date_text(raw)
    else:
        text = extract_entity_answer(raw)
    if text:
        return ("abstained" if is_idk(text) else "answered"), text
    if raw.strip() and is_idk(raw):
        return "abstained", ""
    return "fmt_fail", ""


def ladder_index(level: TemporalLevel) -> int:
    """Position on DAY, MONTH, YEAR, DECADE, CENTURY; half-levels snap coarser."""
    return DATE_LADDER.index(_snap_to_ladder(level))


def valid_levels(item) -> list[int]:
    """Level indices a target may take, most specific first."""
    if item.domain == "dates":
        return [i for i, L in enumerate(DATE_LADDER)
                if PRECISION[L] <= PRECISION[item.gold.level]]
    return list(range(len(item.hierarchy)))


def level_name(item, index) -> str:
    if index is None:
        return "abstain"
    return DATE_LADDER[index].name if item.domain == "dates" else str(index)


def evidence(item, raw: str):
    """Most specific level this sample agrees with the reference at, else None.
    Abstentions and format failures support no level."""
    kind, text = extract(item, raw)
    if kind != "answered":
        return None
    if item.domain == "dates":
        lvl = date_credit(parse_date(text), item.gold, strict=False)
        return None if lvl is None else ladder_index(lvl)
    return find_match_level_v3(text, item.hierarchy)


def grade(item, raw: str, mode: str = "deploy", pool: str = "all") -> dict:
    """Strict grade: kind, answer text, correctness, credited level index, signed score."""
    kind, text = extract(item, raw, mode)
    out = {"kind": kind, "text": text, "correct": False, "level": None, "score": 0.0}
    if kind == "abstained":
        return out
    if item.domain == "dates":
        v = parse_date(text) if kind == "answered" else None
        lvl = date_credit(v, item.gold, strict=True)
        if lvl is not None:
            out.update(correct=True, level=ladder_index(lvl), score=level_info(lvl))
        else:
            # Wrong answers lose the value of their own claimed precision;
            # unreadable ones lose the year value.
            out["score"] = -level_info(TemporalLevel.YEAR if v is None else v.level)
        return out
    sl = (find_match_level_v3(finest_element(text), item.hierarchy)
          if kind == "answered" else None)
    if sl is not None:
        out.update(correct=True, level=sl, score=informativeness(item.levels[sl]))
    elif item.domain == "locations":
        out["score"] = -informativeness(answer_norm_level(text, location_pool(pool)))
    else:
        # Entity descriptions have no level pool, so a wrong answer loses the value of
        # the coarsest reference entry.
        out["score"] = -informativeness(item.levels[-1])
    return out
