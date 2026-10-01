"""Conflicting sources probe.

Two sources disagree about a date below some level L and agree at L, for example
7 March 1901 and 21 March 1901 (L = month). Both are wrong below L and right at L,
so the answer the sources support is L itself, and taking either source is wrong.

Three conditions per question: `conflict` (L is month, year or decade, spread
evenly), `agree` (both sources give the true day, so the right answer is the day),
and `none` (no sources, the normal date prompt).
"""

import random
import re

from ugc.parser.temporal_value import TemporalValue
from ugc.parser.temporal_enums import TemporalLevel
from ugc.parser.temporal_constants import PRECISION

from .dates import (DateQuestion, level_info, credited_level, parse_answer,
                    signed_date_info)
from .prompt import is_idk

# Levels the sources agree at. Not the day (nothing below it to disagree about) and
# not the century (two dates decades apart read as unrelated events, not one event).
AGREEMENT_LEVELS = [TemporalLevel.MONTH, TemporalLevel.YEAR, TemporalLevel.DECADE]

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]

# Days 1-28 exist in every month.
_SAFE_DAYS = list(range(1, 29))

CONFLICT_PROMPT_TEMPLATE = (
    "You are given information from two sources.\n"
    "Source 1: {s1}\n"
    "Source 2: {s2}\n\n"
    "Answer when the following happened, using the sources. Reply in exactly "
    "this format:\n"
    "Date: <date>\n\n"
    "Give a single date after 'Date:'. If you are unsure of the precise date, "
    "give a broader period you are confident about instead of guessing: a month, "
    "a year, a decade, or a century. If you do not know even the century, reply "
    "exactly:\n"
    "Date: I don't know\n\n"
    "Question: {q}"
)


def fmt_day(year: int, month: int, day: int) -> str:
    return f"{day} {_MONTHS[month - 1]} {year}"


def _source_sentence(person: str, kind: str, year: int, month: int, day: int) -> str:
    verb = "was born on" if kind == "birth" else "died on"
    return f"{person} {verb} {fmt_day(year, month, day)}."


def value_at(level: TemporalLevel, year: int, month: int | None = None) -> TemporalValue:
    """The TemporalValue a correct back-off to `level` should claim."""
    if level == TemporalLevel.MONTH:
        return TemporalValue(level=TemporalLevel.MONTH, year=year, month=month)
    if level == TemporalLevel.YEAR:
        return TemporalValue(level=TemporalLevel.YEAR, year=year)
    if level == TemporalLevel.DECADE:
        return TemporalValue(level=TemporalLevel.DECADE,
                             decade=TemporalValue._year_to_decade(year))
    raise ValueError(f"unsupported agreement level {level}")


def build_case(q: DateQuestion, condition: str, level: TemporalLevel | None,
               rng: random.Random) -> dict | None:
    """One probe case, or None if the question cannot carry it.

    Needs a reference given to the day. Dates before 1 CE are skipped, since a
    source sentence would have to pick a BCE year convention."""
    g = q.gold
    if g.level != TemporalLevel.DAY:
        return None
    if g.year is None or g.year < 1:
        return None
    y, m, d = g.year, g.month, g.day

    if condition == "none":
        return {"qid": q.qid, "condition": "none", "level": None,
                "prompt_kind": "plain", "s1": "", "s2": "",
                "evidence_value": None, "gold": g, "question": q.question}

    if condition == "agree":
        # Both sources report the true day. Correct move: commit at DAY.
        s = _source_sentence(q.person, q.kind, y, m, d)
        return {"qid": q.qid, "condition": "agree", "level": TemporalLevel.DAY,
                "prompt_kind": "sources", "s1": s, "s2": s,
                "evidence_value": g, "gold": g, "question": q.question}

    if condition != "conflict":
        raise ValueError(condition)

    if level == TemporalLevel.MONTH:
        # Same year+month, two wrong days (neither is the gold day).
        pool = [x for x in _SAFE_DAYS if x != d]
        d1, d2 = rng.sample(pool, 2)
        s1 = _source_sentence(q.person, q.kind, y, m, d1)
        s2 = _source_sentence(q.person, q.kind, y, m, d2)
        ev = value_at(TemporalLevel.MONTH, y, m)

    elif level == TemporalLevel.YEAR:
        # Same year, two wrong months.
        pool = [x for x in range(1, 13) if x != m]
        m1, m2 = rng.sample(pool, 2)
        s1 = _source_sentence(q.person, q.kind, y, m1, rng.choice(_SAFE_DAYS))
        s2 = _source_sentence(q.person, q.kind, y, m2, rng.choice(_SAFE_DAYS))
        ev = value_at(TemporalLevel.YEAR, y)

    elif level == TemporalLevel.DECADE:
        # Same decade, two wrong years.
        base = (y // 10) * 10
        pool = [x for x in range(base, base + 10) if x != y]
        if len(pool) < 2:
            return None
        y1, y2 = rng.sample(pool, 2)
        s1 = _source_sentence(q.person, q.kind, y1, rng.randint(1, 12), rng.choice(_SAFE_DAYS))
        s2 = _source_sentence(q.person, q.kind, y2, rng.randint(1, 12), rng.choice(_SAFE_DAYS))
        ev = value_at(TemporalLevel.DECADE, y)
    else:
        raise ValueError(f"unsupported agreement level {level}")

    return {"qid": q.qid, "condition": "conflict", "level": level,
            "prompt_kind": "sources", "s1": s1, "s2": s2,
            "evidence_value": ev, "gold": g, "question": q.question}


def build_cases(test: list[DateQuestion], seed: int = 0) -> list[dict]:
    """All three conditions for every test question with a day-level reference.
    Agreement levels are assigned in turn over a shuffled list, so each gets a
    third of the questions."""
    rng = random.Random(seed)
    usable = [q for q in test if q.gold.level == TemporalLevel.DAY
              and q.gold.year is not None and q.gold.year >= 1]
    order = list(usable)
    rng.shuffle(order)

    cases = []
    for i, q in enumerate(order):
        lvl = AGREEMENT_LEVELS[i % len(AGREEMENT_LEVELS)]
        for cond, L in (("conflict", lvl), ("agree", None), ("none", None)):
            c = build_case(q, cond, L, rng)
            if c is not None:
                cases.append(c)
    return cases


def make_conflict_prompt(case: dict) -> str:
    from .dates import make_date_prompt
    if case["prompt_kind"] == "plain":
        return make_date_prompt(case["question"])
    return CONFLICT_PROMPT_TEMPLATE.format(s1=case["s1"], s2=case["s2"],
                                           q=case["question"])


# ---------------- scoring ----------------

def classify(case: dict, completion: str) -> dict:
    """Score one raw completion against the sources and the reference.

    `behaviour` says how the answer's level compares with the level the sources
    support. A reply without the "Date:" marker and without "I don't know" is a
    format failure and counts as an answer, not an abstention. Dates are read and
    graded by pipeline/score.py, like every other answer."""
    from pipeline.score import date_credit, parse_date
    text, idk, _ = parse_answer(completion)
    v = None if idk else parse_date(text)
    if idk and not text and not is_idk(completion):
        # No marker anywhere and no IDK phrase in the raw text: format failure.
        return {"qid": case["qid"], "condition": case["condition"],
                "agreement_level": case["level"].name if case["level"] else "",
                "s1": case["s1"], "s2": case["s2"],
                "answer": completion[:120], "parsed": "", "answer_level": "",
                "idk": False, "behaviour": "fmt_fail",
                "signed_strict": -level_info(TemporalLevel.YEAR),
                "correct_strict": False}
    target = case["evidence_value"]          # None for the `none` condition
    row = {"qid": case["qid"], "condition": case["condition"],
           "agreement_level": case["level"].name if case["level"] else "",
           "s1": case["s1"], "s2": case["s2"],
           "answer": text, "parsed": str(v) if v is not None else "",
           "answer_level": v.level.name if v is not None else "",
           "idk": idk}

    # Signed informativeness against the REAL gold, strict policy: the same
    # metric as every other table in the thesis, so this probe is comparable.
    lvl = None if (idk or v is None) else date_credit(v, case["gold"], strict=True)
    row["correct_strict"] = lvl is not None
    if idk:
        row["signed_strict"] = 0.0
    elif lvl is not None:
        row["signed_strict"] = level_info(lvl)
    else:
        row["signed_strict"] = -level_info(TemporalLevel.YEAR if v is None else v.level)

    if idk:
        row["behaviour"] = "idk"
    elif v is None:
        row["behaviour"] = "fmt_fail"          # committed but unparseable
    elif target is None:
        row["behaviour"] = "answered"          # `none` condition: no level to compare
    else:
        p_ans, p_tgt = PRECISION[v.level], PRECISION[target.level]
        if p_ans == p_tgt:
            # Right granularity. Does it also state the right value?
            try:
                ok = v.consistent_with(target)
            except (ValueError, KeyError):
                ok = False
            row["behaviour"] = "at_level" if ok else "at_level_wrong"
        elif p_ans > p_tgt:
            row["behaviour"] = "over_commit"   # finer than the evidence supports
        else:
            row["behaviour"] = "too_coarse"    # coarser than the evidence supports
    return row


def summarize(rows: list[dict]) -> dict:
    """Behaviour shares and mean score per condition and agreement level."""
    out = {}
    keys = sorted({(r["condition"], r["agreement_level"]) for r in rows})
    for cond, lvl in keys:
        sel = [r for r in rows if r["condition"] == cond and r["agreement_level"] == lvl]
        n = len(sel)
        beh = {}
        for b in ("at_level", "at_level_wrong", "over_commit", "too_coarse",
                  "idk", "fmt_fail", "answered"):
            c = sum(1 for r in sel if r["behaviour"] == b)
            if c:
                beh[b] = round(c / n, 4)
        out[f"{cond}:{lvl}" if lvl else cond] = {
            "n": n,
            "behaviour": beh,
            "signed_strict": round(sum(r["signed_strict"] for r in sel) / n, 4),
            "acc_strict": round(sum(1 for r in sel if r["correct_strict"]) / n, 4),
        }
    return out
