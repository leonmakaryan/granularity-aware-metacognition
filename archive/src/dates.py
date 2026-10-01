"""Date questions: loading, prompts, parsing and informativeness, plus the targets
and training records of the dates SFT.

The questions ask when a person was born or died, from Lukas's persons dataset, and
answers are read with his temporal parser (ugc). Unlike place answers, every date
answer has an exact level, its parsed precision, also when it is wrong.
"""

import json
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

DATES_PATH = Path(__file__).resolve().parents[2] / "data" / "persons_metadata.jsonl"
DATES_TEST_FRAC = 0.15
DATES_SPLIT_SEED = 0

# The forced-level ladder (finest -> coarsest); half-levels (season, early/mid)
# can still appear in parsed answers and score via PRECISION, they are just not
# capability targets.
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

# Direct question for the ESMA-style evaluation: the deployment prompt without
# its abstention sentence, as NOIDK_PROMPT_TEMPLATE is for locations.
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


import math

from collections import Counter

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


# ---------------- dates SFT: targets and training data ----------------

DATE_FORCE_INSTRUCTION = {
    TemporalLevel.DAY: "Give the exact date (day, month and year).",
    TemporalLevel.MONTH: "Give the month and year only.",
    TemporalLevel.YEAR: "Give the year only.",
    TemporalLevel.DECADE: "Give the decade only (like '1870s').",
    TemporalLevel.CENTURY: "Give the century only (like '19th century').",
}

DATE_DIRECT_TEMPLATE = (
    "Answer when the following happened. Reply in exactly this format:\n"
    "Date: <date>\n\n"
    "{instruction}\n\n"
    "Question: {q}"
)


def _snap_to_ladder(level: TemporalLevel) -> TemporalLevel:
    """Move a half level onto the coarser neighbour in DATE_LADDER (season -> year)."""
    for L in DATE_LADDER:
        if PRECISION[L] <= PRECISION[level]:
            return L
    return TemporalLevel.CENTURY


def derive_dates_targets(cap: dict, tau: float = 0.5, min_commit: int = 3,
                         max_wrong: int = 2) -> dict:
    """The v2 target rule (as in capability.pick_target_v2) on the dates sample cache.
        Adds 'target_level' (a level name, or None for "I don't know") to every question."""
    dist = Counter()
    for r in cap["questions"].values():
        gold_level = TemporalLevel[r["gold_level"]]
        ladder = [L for L in DATE_LADDER if PRECISION[L] <= PRECISION[gold_level]]
        committed = [s for s in r["samples"] if not s["idk"]]
        wrong = sum(1 for s in committed if s["lenient_level"] is None)
        g = r["greedy"]
        g_lvl = (None if (g["idk"] or g["lenient_level"] is None)
                 else _snap_to_ladder(TemporalLevel[g["lenient_level"]]))
        target = None
        if wrong <= max_wrong:
            if len(committed) >= min_commit:
                for L in ladder:
                    n_ok = sum(1 for s in committed if s["lenient_level"] is not None
                               and PRECISION[TemporalLevel[s["lenient_level"]]] >= PRECISION[L])
                    if n_ok / len(committed) >= tau:
                        target = L
                        break
            elif g_lvl is not None and wrong == 0:
                target = g_lvl
            if (target is not None and g_lvl is not None and wrong == 0
                    and PRECISION[g_lvl] > PRECISION[target]):
                target = g_lvl   # greedy promotion
        r["target_level"] = target.name if target is not None else None
        dist["IDK" if target is None else target.name] += 1
    n = len(cap["questions"])
    cap["target_rule"] = dict(rule="v2-dates", tau=tau, min_commit=min_commit,
                              max_wrong=max_wrong)
    print(f"dates targets (tau={tau} min_commit={min_commit} max_wrong={max_wrong}): "
          f"{dict(sorted(dist.items(), key=lambda kv: (kv[0]=='IDK', kv[0])))}  "
          f"IDK {dist['IDK']}/{n} ({100*dist['IDK']/n:.0f}%)")
    return cap


def _own_date_at(r: dict, level: TemporalLevel) -> str | None:
    """Self-distilled completion: majority vote over the model's OWN sampled
    values that are CORRECT at `level`, each projected to `level` and rendered
    (a full date with the right year projects to the right '1904'). None when no
    sample qualifies (caller falls back to the gold rendering)."""
    votes = Counter()
    cands = list(r["samples"]) + [r["greedy"]]
    for s in cands:
        if s["idk"] or s["lenient_level"] is None:
            continue
        if PRECISION[TemporalLevel[s["lenient_level"]]] < PRECISION[level]:
            continue          # content not reliable at this level
        v = _PARSER.first_most_specific(s["text"])
        if v is None or PRECISION[v.level] < PRECISION[level]:
            continue          # cannot project a coarser claim to `level`
        try:
            votes[v.to_level(level).to_text()] += 1
        except (ValueError, KeyError):
            continue
    return votes.most_common(1)[0][0] if votes else None


def build_dates_sft_records(cap: dict, golds: dict, balance_cap: float = 3.0,
                            seed: int = 0, dep_weight: int = 1) -> list[dict]:
    """SFT records for dates: the deployment prompt (the model's own answer at the
        target level, or "I don't know") and prompts asking for a given level. Gold dates
        are used only when the model never gave a usable answer. `dep_weight` repeats the
        deployment examples; at 1 they were drowned out and the model stopped abstaining."""
    deployment, direct = [], []
    n_self = n_fallback = 0
    for qid, r in cap["questions"].items():
        q = r["question"]
        t = r.get("target_level")
        if t is None:
            deployment.append(_rec(make_date_prompt(q), "Date: I don't know", "IDK"))
            continue
        t = TemporalLevel[t]
        gold = golds[qid]
        for L in [x for x in DATE_LADDER
                  if PRECISION[t] >= PRECISION[x] >= PRECISION[TemporalLevel.CENTURY]]:
            own = _own_date_at(r, L)
            if own is None:
                try:
                    own = gold.to_level(L).to_text()
                except (ValueError, KeyError):
                    continue
                n_fallback += 1
            else:
                n_self += 1
            if L == t:
                deployment.append(_rec(make_date_prompt(q), f"Date: {own}", L.name))
            direct.append(_rec(
                DATE_DIRECT_TEMPLATE.format(instruction=DATE_FORCE_INSTRUCTION[L], q=q),
                f"Date: {own}", L.name))
    print(f"  dates self-distill: own {n_self}, gold fallback {n_fallback} "
          f"({100*n_fallback/max(n_self+n_fallback,1):.0f}%)")

    # balance the direct family by level label (cap`d oversampling, like locations)
    import math as _math
    import random as _random
    groups = {}
    for rec in direct:
        groups.setdefault(rec["_label"], []).append(rec)
    maxc = max(len(g) for g in groups.values()) if groups else 0
    rng = _random.Random(seed)
    balanced = []
    for g in groups.values():
        want = min(maxc, _math.ceil(len(g) * balance_cap))
        balanced.extend(g)
        if want > len(g):
            balanced.extend(rng.choices(g, k=want - len(g)))
    records = deployment * dep_weight + balanced
    rng.shuffle(records)
    print(f"  dates SFT: deployment={len(deployment)}x{dep_weight} direct={len(balanced)} "
          f"total={len(records)}")
    print(f"  deployment labels: {dict(Counter(r['_label'] for r in deployment))}")
    return [{"prompt": r["prompt"], "completion": r["completion"]} for r in records]


def _rec(prompt_text: str, completion_text: str, label: str) -> dict:
    return {"prompt": [{"role": "user", "content": prompt_text}],
            "completion": [{"role": "assistant", "content": completion_text}],
            "_label": label}
