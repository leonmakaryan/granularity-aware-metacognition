"""Gate 1: offline correctness checks, no GPU.

    python -m pipeline.checks

1. The format rules on constructed completions (no stored output contains a format
   failure, so this rule cannot be checked on real data).
2. Grading the raw completions stored by the earlier code (archive/phase2_results)
   reproduces its 7B numbers after its two scoring fixes, and extracts exactly the
   stored answer text.
3. The 50%-support rule on the earlier 7B sample caches reproduces the target
   distributions computed when this pipeline was set up, and every training completion
   passes the scorer. The caches serve only as a code test, never as training data.
"""
import csv
import json
import sys
from collections import Counter
from contextlib import contextmanager

from ugc.parser.temporal_enums import TemporalLevel
from ugc.parser.temporal_value import TemporalValue

import src.dates as D
from src.dates import level_info
from src.levels import informativeness

from . import config as C
from . import score
from .data import Item, load
from .score import evidence, extract, grade, julian_gold, level_name, parse_date
from .targets import build_records, support_level

FAILS = []


@contextmanager
def historical_parse():
    """The earlier date scoring (retry on any unreadable text, no Julian credit), only
    for reproducing the earlier numbers."""
    saved, saved_julian = score.parse_date, score.JULIAN_CREDIT
    score.parse_date, score.JULIAN_CREDIT = D.parse_date_text, False
    try:
        yield
    finally:
        score.parse_date, score.JULIAN_CREDIT = saved, saved_julian


def check(ok: bool, msg: str) -> None:
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def unit_tests() -> None:
    print("1. format and grading rules")
    loc = Item("locations", "u", "Where was X born?", "u",
               ["Wiener Neustadt", "Lower Austria", "Austria"], [1.0, 2.5, 4.0])
    dat = Item("dates", "u_birth", "When was X born?", "u",
               gold=TemporalValue(level=TemporalLevel.DAY, year=1904, month=6, day=16))
    ent = Item("entity", "e", "Who wrote X?", "e", ["August Derleth", "a novelist"], [1.0, 4.0])
    for item, raw, kind in [
        (loc, "Location: Austria", "answered"),
        (loc, "Location: I don't know", "abstained"),
        (loc, "I don't know where that is.", "abstained"),
        (loc, "Date: 1905", "fmt_fail"),            # the other domain's marker
        (loc, "It is in Austria.", "fmt_fail"),     # prose, no marker
        (loc, "", "fmt_fail"),                      # empty completion
        (loc, "Location:", "fmt_fail"),             # marker without an answer
        (dat, "Date: 912", "answered"),
        (dat, "Date: I don't know", "abstained"),
        (dat, "Location: Vienna", "fmt_fail"),
        (ent, "Answer: a novelist", "answered"),
        (ent, "Location: Vienna", "fmt_fail"),
    ]:
        check(extract(item, raw)[0] == kind, f"{item.domain:9} {raw!r:34} -> {kind}")
    for item, raw, correct, level, score in [
        (loc, "Location: Wiener Neustadt", True, 0, 1.0),
        (loc, "Location: Austria", True, 2, 0.25),      # exact entry beats 'Lower Austria'
        (loc, "Location: Lower Austria, Austria", True, 1, 0.625),
        (loc, "Location: ", False, None, -informativeness(2.5)),
        (dat, "Date: 1904", True, 2, level_info(TemporalLevel.YEAR)),
        (dat, "Date: June 1904", True, 1, level_info(TemporalLevel.MONTH)),
        (dat, "Date: 17 June 1904", False, None, -level_info(TemporalLevel.DAY)),
        (dat, "Date: 1908", False, None, -level_info(TemporalLevel.YEAR)),
        (dat, "no date here", False, None, -level_info(TemporalLevel.YEAR)),
    ]:
        g = grade(item, raw)
        check((g["correct"], g["level"]) == (correct, level) and abs(g["score"] - score) < 1e-12,
              f"grade {raw!r:34} -> correct={correct} level={level} score={score:+.3f}")
    check(grade(loc, "Location: Vienna, Austria")["correct"] is False,
          "strict grading ignores a correct country after a wrong city")
    check(evidence(loc, "Location: Vienna, Austria") == 2,
          "but that sample is evidence for the country level")
    check(evidence(dat, "Date: 17 June 1904") == 1,
          "a wrong day with the right month is evidence for the month")
    for ts, prec, text in [("-0043-03-15T00:00:00Z", 9, "44 BCE"),
                           ("-0029-07-30T00:00:00Z", 11, "July 30, 30 BCE"),
                           ("-0500-01-01T00:00:00Z", 7, "5th century BCE"),
                           ("-0430-01-01T00:00:00Z", 8, "430s BCE"),
                           ("+1904-06-16T00:00:00Z", 11, "June 16, 1904")]:
        g = D._gold_value(ts, prec, historical_bce=True)
        check(g.to_text() == text, f"historical BCE gold {ts} p{prec} -> {text!r}")
    check(D.credited_level(parse_date("44 BC"), D._gold_value("-0043-03-15T00:00:00Z", 9, True), True)
          == TemporalLevel.YEAR, "'44 BC' is credited against Caesar's death year")
    erasmus = Item("dates", "e_death", "When did Erasmus die?", "e",
                   gold=TemporalValue(level=TemporalLevel.DAY, year=1536, month=7, day=22))
    newton = Item("dates", "n_birth", "When was Isaac Newton born?", "n",
                  gold=TemporalValue(level=TemporalLevel.DAY, year=1643, month=1, day=4))
    j = julian_gold(D._gold_value("1223-07-21T00:00:00Z", 11, True))
    check((j.year, j.month, j.day) == (1223, 7, 14), "Philip II's death 21 July 1223 is 14 July Julian")
    for item, raw, correct in [(erasmus, "Date: July 12, 1536", True),   # Julian, conventional
                               (erasmus, "Date: July 22, 1536", True),   # proleptic Gregorian
                               (erasmus, "Date: July 17, 1536", False),
                               (newton, "Date: December 25, 1642", False)]:  # after the reform
        check(grade(item, raw)["correct"] is correct, f"calendar: {raw!r:28} -> correct={correct}")
    check(evidence(erasmus, "Date: July 12, 1536") == 0, "a Julian-dated sample is day evidence")
    check(abs(grade(loc, "Location: India")["score"] + informativeness(2.5)) > 0.1,
          "a wrong 'India' gets a pool level, not the unmatched midpoint")
    for text, level in [("912", "YEAR"), ("96 AD", "YEAR"), ("c. 335", "YEAR"),
                        ("137 CE", "YEAR"), ("21 January", None), ("353-406", None)]:
        v = parse_date(text)
        check((v.level.name if v else None) == level,
              f"bare-year retry: {text!r:14} -> {level}")


def reproduction() -> None:
    print("2. reproduce the corrected 7B numbers from stored raw completions")
    stored = json.loads((C.ROOT / "archive" / "phase2_results" / "corrected_7b.json").read_text())
    cells = json.loads((C.ROOT / "archive" / "phase2_results" / "corrections_aug12.json").read_text())
    items = {d: {it.qid: it for it in load(d, "test", historical_bce=False)}
             for d in ("locations", "dates")}
    for domain in ("locations", "dates"):
      with historical_parse():
        for cell in cells[domain]:
            path = C.ROOT / "archive" / "phase2_results" / f"transfer_7b_{cell}_on_{domain}.csv"
            rows = list(csv.DictReader(open(path)))
            gs = [grade(items[domain][r["qid"]], r["raw"], pool="train+test") for r in rows]
            n = len(rows)
            got = dict(correct=sum(g["correct"] for g in gs) / n,
                       idk=sum(g["kind"] == "abstained" for g in gs) / n,
                       wrong=sum(g["kind"] != "abstained" and not g["correct"] for g in gs) / n,
                       strict=sum(g["score"] for g in gs) / n)
            want = stored[domain][cell]
            same_text = sum(g["text"] == r["answer"] for g, r in zip(gs, rows))
            ok = all(abs(got[k] - want[k]) <= 1e-10 for k in got) and same_text == n
            check(ok, f"{domain:9} {cell:22} metrics reproduce, extraction {same_text}/{n}")


def targets() -> None:
    print("3. 50%-support targets on stored 7B caches, and every training answer passes")
    cap = json.loads((C.ROOT / "archive" / "phase2_results" / "capability_loc_7b.json").read_text())
    items, samples = [], {}
    for qid, r in cap["questions"].items():
        it = Item("locations", str(qid), r["question"], str(qid), r["hierarchy"], r["levels"])
        items.append(it)
        samples[it.qid] = [f"Location: {a}" if a else "" for a in r["sample_answers"]]
    dist = Counter(level_name(it, support_level(it, samples[it.qid])) for it in items)
    want = {"0": 154, "1": 352, "2": 809, "3": 173, "abstain": 1396}
    check(dict(dist) == want, f"locations distribution {dict(sorted(dist.items()))}")
    records, stats = build_records("locations", items, samples)
    check(len(records) + stats.get("skipped_no_model_answer", 0) == len(items),
          f"locations records {len(records)}, stats {stats}")

    cap = json.loads((C.ROOT / "archive" / "phase2_results" / "dates_7b_capability.json").read_text())
    old = {it.qid: it for it in load("dates", "train", historical_bce=False)}
    new_train = {it.qid: it for it in load("dates", "train")}
    items = [old[q] for q in sorted(cap["questions"])]
    new_items = [new_train[q] for q in sorted(cap["questions"])]
    samples = {q: [f"Date: {s['text']}" if s["text"] else "" for s in r["samples"]]
               for q, r in cap["questions"].items()}
    with historical_parse():
        dist = Counter(level_name(it, support_level(it, samples[it.qid])) for it in items)
    want = {"DAY": 383, "MONTH": 55, "YEAR": 1597, "DECADE": 110, "CENTURY": 99, "abstain": 480}
    check(dict(dist) == want, f"dates distribution, historical parser {dict(sorted(dist.items()))}")
    new = Counter(level_name(it, support_level(it, samples[it.qid])) for it in new_items)
    moved = sum(support_level(a, samples[a.qid]) != support_level(b, samples[b.qid])
                for a, b in zip(items, new_items))
    print(f"        with bare-year retry and historical BCE years {dict(sorted(new.items()))} "
          f"({moved} targets move from the BCE fix)")
    records, stats = build_records("dates", new_items, samples)
    check(len(records) + stats.get("skipped_no_model_answer", 0) == len(items),
          f"dates records {len(records)}, stats {stats}")


if __name__ == "__main__":
    unit_tests()
    reproduction()
    targets()
    print("\nGATE 1 " + ("PASSED" if not FAILS else f"FAILED ({len(FAILS)} checks)"))
    sys.exit(1 if FAILS else 0)
