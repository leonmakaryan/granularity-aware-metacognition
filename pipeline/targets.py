"""Training targets (the 50%-support rule) and training records.

Target: the most specific level L supported by at least half of the K base samples
(a sample supports L if it is correct at L or finer), otherwise abstain. A correct
answer at L earns I(L) and a wrong one loses I(L), so answering at L has expected
score I(L)(2p - 1): the rule picks the most specific level where answering is not
expected to lose. Samples that abstain count as not supporting.

Training answer: the model's own most frequent sample that is correct at exactly the
target level, cut to that level (dates rewritten at that precision, places reduced to
the part before the first comma). Reference answers are never used; a question
without such a sample is skipped and counted.
"""
from collections import Counter

from src.dates import DATE_LADDER
from src.prompt import finest_element
from src.text import normalize

from . import config as C
from .score import (evidence, extract, grade, level_name, parse_date, prompt_for,
                   valid_levels)

ABSTAIN = {"locations": "Location: I don't know", "dates": "Date: I don't know"}


def support_level(item, raws: list[str]):
    ev = [evidence(item, r) for r in raws]
    for L in valid_levels(item):
        if sum(1 for e in ev if e is not None and e <= L) / len(raws) >= C.SUPPORT_THRESHOLD:
            return L
    return None


def natural_level(item, raws: list[str]):
    """The answer-or-abstain baseline, in the style of R-Tuning. If at least half the
    samples are correct as written, the target is the level those answers take most
    often, otherwise abstain. It never moves to a coarser level, so comparing it with
    support_level shows what choosing the level adds."""
    ok = [g for g in (grade(item, r) for r in raws) if g["correct"]]
    if len(ok) / len(raws) < C.SUPPORT_THRESHOLD:
        return None
    return Counter(g["level"] for g in ok).most_common(1)[0][0]


RULES = {"support": support_level, "binary": natural_level}


def _render(item, text: str, level: int):
    if item.domain == "dates":
        v = parse_date(text)
        if v is None:
            return None
        try:
            return "Date: " + v.to_level(DATE_LADDER[level]).to_text()
        except (ValueError, KeyError):
            return None
    return "Location: " + finest_element(text)


def training_completion(item, raws: list[str], target):
    if target is None:
        return ABSTAIN[item.domain]
    votes, first = Counter(), {}
    for raw in raws:
        kind, text = extract(item, raw)
        if kind != "answered":
            continue
        comp = _render(item, text, target)
        if comp is None:
            continue
        g = grade(item, comp)
        if g["correct"] and g["level"] == target:
            key = normalize(comp)
            votes[key] += 1
            first.setdefault(key, comp)
    if not votes:
        return None
    return first[votes.most_common(1)[0][0]]


def draft_line(item, raw: str) -> str:
    """First line of the draft format (an experiment not used in the thesis): the
    model's own guess, then the graded answer line."""
    kind, text = extract(item, raw)
    return C.DRAFT_PREFIX + (text if kind == "answered" else "I don't know")


def build_records(domain: str, items: list, samples: dict,
                  drafts: dict | None = None, rule: str = "support") -> tuple[list[dict], dict]:
    """Return (records, stats). Every kept completion is graded again and must hit its target.

    `drafts` (qid -> base greedy answer) adds the draft line in front of each completion."""
    records, stats = [], Counter()
    for it in items:
        raws = samples[it.qid]
        target = RULES[rule](it, raws)
        comp = training_completion(it, raws, target)
        if comp is None:
            stats["skipped_no_model_answer"] += 1
            continue
        if drafts is not None:
            comp = draft_line(it, drafts[it.qid]) + "\n" + comp
        g = grade(it, comp)
        if target is None:
            assert g["kind"] == "abstained", (it.qid, comp)
        else:
            assert g["correct"] and g["level"] == target, (it.qid, comp, target, g)
        stats[f"target_{level_name(it, target)}"] += 1
        records.append({"qid": it.qid, "domain": domain,
                        "prompt": [{"role": "user", "content": prompt_for(it)}],
                        "completion": [{"role": "assistant", "content": comp}]})
    return records, dict(sorted(stats.items()))
