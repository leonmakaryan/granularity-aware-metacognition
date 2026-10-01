"""Date questions: loading, prompts, parsing and informativeness.

The questions ask when a person was born or died, from Lukas's persons dataset, and
answers are read with his temporal parser (ugc). Unlike place answers, every date
answer has an exact level, its parsed precision, also when it is wrong.
"""

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

from ugc.parser.temporal_parser import TemporalParser
from ugc.parser.temporal_value import TemporalValue
from ugc.parser.temporal_enums import TemporalLevel
from ugc.parser.temporal_constants import PRECISION

from evaluation.informativeness import WIDTH_IN_YEARS   # Lukas's level widths

from .prompt import is_idk

DATES_PATH = Path(__file__).resolve().parent.parent / "data" / "persons_metadata.jsonl"
DATES_TEST_FRAC = 0.15
DATES_SPLIT_SEED = 0

# The five levels a date answer is trained at, finest first. Half levels such as a
# season or "early 1900s" can still occur in answers and are scored normally.
DATE_LADDER = [TemporalLevel.DAY, TemporalLevel.MONTH, TemporalLevel.YEAR,
               TemporalLevel.DECADE, TemporalLevel.CENTURY]

DATE_PROMPT_TEMPLATE = (
    "Answer when the following happened. Reply in exactly this format:\n"
    "Date: <date>\n\n"
    "Give a single date after 'Date:'. If you are unsure of the precise date, "
    "give a broader period you are confident about instead of guessing: a month, "
    "a year, a decade, or a century. If you do not know even the century, reply "
    "exactly:\n"
    "Date: I don't know\n\n"
    "Question: {q}"
)

# The deployment prompt without the "I don't know" option.
DATE_NOIDK_PROMPT_TEMPLATE = (
    "Answer when the following happened. Reply in exactly this format:\n"
    "Date: <date>\n\n"
    "Give a single date after 'Date:'. If you are unsure of the precise date, "
    "give a broader period you are confident about instead of guessing: a month, "
    "a year, a decade, or a century.\n\n"
    "Question: {q}"
)

DATE_RE = re.compile(r"Date:\s*([^\n]+)", re.IGNORECASE)

_PARSER = TemporalParser()


@dataclass
class DateQuestion:
    qid: str                 # "<wikidata qid>_birth" / "_death"
    question: str
    gold: TemporalValue
    person: str
    kind: str                # birth | death

    @property
    def gold_precision(self) -> float:
        return self.gold.precision


def _gold_value(ts: str, precision: int, historical_bce: bool = False) -> TemporalValue | None:
    """The reference date from a Wikidata timestamp, built at its stated precision.

    Wikidata fills unknown month and day with '00', so the timestamp is not parsed as
    a full date. Wikidata counts BCE years astronomically (-43 is 44 BC), while the
    parser reads "44 BC" as -44; historical_bce=True shifts those years by one."""
    m = re.fullmatch(r"([+-]?\d+)-(\d{2})-(\d{2})T.*Z", ts or "")
    if not m or precision is None:
        return None
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if year == 0:
        return None
    if historical_bce and year < 0 and precision >= 9:
        year -= 1
    try:
        if precision == 11:
            return TemporalValue(level=TemporalLevel.DAY, year=year, month=month, day=day)
        if precision == 10:
            return TemporalValue(level=TemporalLevel.MONTH, year=year, month=month)
        if precision == 9:
            return TemporalValue(level=TemporalLevel.YEAR, year=year)
        if precision == 8:
            return TemporalValue(level=TemporalLevel.DECADE,
                                 decade=TemporalValue._year_to_decade(year))
        if precision == 7:
            return TemporalValue(level=TemporalLevel.CENTURY,
                                 century=TemporalValue._year_to_century(year))
    except (ValueError, KeyError):
        return None
    return None


def load_dates_split(split: str, path: Path = DATES_PATH,
                     historical_bce: bool = False) -> list[DateQuestion]:
    """Train or test questions, split by person so both questions of a person stay together."""
    qs = []
    persons = [json.loads(l) for l in open(path)]
    rng = random.Random(DATES_SPLIT_SEED)
    for p in persons:
        in_test = rng.random() < DATES_TEST_FRAC
        if (split == "test") != in_test:
            continue
        for kind, verb in (("birth", f"When was {p['name']} born?"),
                           ("death", f"When did {p['name']} die?")):
            gold = _gold_value(p.get(f"{kind}_date"), p.get(f"{kind}_precision"),
                               historical_bce)
            if gold is None:
                continue
            qs.append(DateQuestion(qid=f"{p['qid']}_{kind}", question=verb,
                                   gold=gold, person=p["name"], kind=kind))
    return qs


def make_date_prompt(question: str) -> str:
    return DATE_PROMPT_TEMPLATE.format(q=question)


def extract_date_text(completion: str) -> str:
    """The text after the 'Date:' marker (whole line), '' if absent."""
    if not completion:
        return ""
    m = DATE_RE.search(completion)
    return m.group(1).strip() if m else ""


def parse_answer(completion: str) -> tuple[str, bool, TemporalValue | None]:
    """Return (answer text, is_idk, parsed date). A missing marker or an "I don't know"
    counts as an abstention; text that parses to no date gives None."""
    text = extract_date_text(completion)
    if not text or is_idk(text):
        return text, True, None
    return text, False, parse_date_text(text)


def parse_date_text(text: str) -> TemporalValue | None:
    """Parse an answer, retrying as "in " + text if the first parse fails.

    The parser ignores a bare number below 1000 without context ("912"), which is
    right for free text. Our answers are cut from the "Date:" line, so the retry
    gives that context back."""
    from .config import CORRECTED_SCORING
    v = _PARSER.first_most_specific(text)
    if v is None and text and CORRECTED_SCORING:
        v = _PARSER.first_most_specific("in " + text)
    return v


def _coarser(a: TemporalLevel, b: TemporalLevel) -> TemporalLevel:
    return a if PRECISION[a] <= PRECISION[b] else b


def credited_level(v: TemporalValue | None, gold: TemporalValue,
                   strict: bool) -> TemporalLevel | None:
    """The level an answer earns credit at, or None if it is wrong or unparseable.

    strict: the answer as stated must agree with the reference. lenient: the finest
    level where the two agree. Never finer than the reference itself."""
    if v is None:
        return None
    if strict:
        try:
            ok = v.consistent_with(gold)
        except (ValueError, KeyError):
            ok = False
        return _coarser(v.level, gold.level) if ok else None
    try:
        lvl = v.max_matching_level(gold)
    except (ValueError, KeyError):
        lvl = None
    return lvl


DATE_PRIOR_WIDTH_YEARS = 1000.0   # Lukas's default prior ("some millennium")


def level_info(level: TemporalLevel) -> float:
    """Informativeness of a date at `level`, as in Lukas's evaluation/informativeness.py
    (his function takes a TemporalValue only to read its level)."""
    max_info = math.log(DATE_PRIOR_WIDTH_YEARS / WIDTH_IN_YEARS[TemporalLevel.DAY])
    return math.log(DATE_PRIOR_WIDTH_YEARS / WIDTH_IN_YEARS[level]) / max_info


def signed_date_info(v: TemporalValue | None, idk: bool, gold: TemporalValue,
                     strict: bool) -> float:
    """Score of one answer: + information if correct, 0 for an abstention, - information
    at the answer's own level if wrong (at the year level if it does not parse)."""
    if idk:
        return 0.0
    lvl = credited_level(v, gold, strict)
    if lvl is not None:
        return level_info(lvl)
    return -level_info(TemporalLevel.YEAR if v is None else v.level)


def _snap_to_ladder(level: TemporalLevel) -> TemporalLevel:
    """Move a half level onto the coarser neighbour in DATE_LADDER (season -> year)."""
    for L in DATE_LADDER:
        if PRECISION[L] <= PRECISION[level]:
            return L
    return TemporalLevel.CENTURY
