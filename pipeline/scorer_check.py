"""Gate 2, the scorer check: compare the scorer with hand labels on stored answers.

build  draws a stratified sample of stored model outputs (never locations val), grades
       them, and writes a sheet without the scorer's decisions plus a separate key.
score  compares the labelled sheet with the key.

Pass criteria, fixed before labelling: outcome agreement >= 0.95; level agreement >= 0.90
on answers both mark correct; and no type of disagreement on 3 or more answers within one
domain (a repeated disagreement points to a wrong rule). Types are counted per domain,
not per stratum, because a wrong rule can show up in several strata.
"""
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.dates import DATE_LADDER, _PARSER

from . import config as C
from .data import load
from .score import grade, parse_date, valid_levels

CHECK = C.RESULTS / "scorer_check"
PER_STRATUM = {"locations": 14, "dates": 12}
STRATA = {"locations": ["correct_finest", "correct_coarser", "wrong", "abstain",
                        "format_failure", "multi_part"],
          "dates": ["correct_finest", "correct_coarser", "wrong", "abstain",
                    "format_failure", "unparseable", "retry_parsed", "short_year", "bce"]}
OUTCOMES = ("correct", "wrong", "abstain", "format_failure")
_DATE_ORDER = {n: i for i, n in enumerate(["day", "month", "year", "decade", "century"])}


def _level_direction(domain: str, scorer: str, human: str) -> str:
    """Whether the scorer credited a finer or coarser level than the human."""
    try:
        s, h = ((int(scorer), int(human)) if domain == "locations"
                else (_DATE_ORDER[scorer], _DATE_ORDER[human]))
    except (ValueError, KeyError):
        return "unreadable_level"
    return "scorer_finer" if s < h else "scorer_coarser"


def _outcome(g: dict) -> str:
    if g["kind"] == "abstained":
        return "abstain"
    if g["kind"] == "fmt_fail":
        return "format_failure"
    return "correct" if g["correct"] else "wrong"


def _level_label(item, level) -> str:
    if level is None:
        return ""
    return str(level + 1) if item.domain == "locations" else DATE_LADDER[level].name.lower()


def _reference(item) -> str:
    if item.domain == "locations":
        return "  ".join(f"{i + 1}) {h}" for i, h in enumerate(item.hierarchy))
    return f"{item.gold.to_text()} (reference precision: {item.gold.level.name.lower()})"


def _strata(item, g: dict) -> list[str]:
    o = _outcome(g)
    s = [o]
    if o == "correct":
        s = ["correct_finest" if g["level"] == valid_levels(item)[0] else "correct_coarser"]
    text = g["text"]
    if item.domain == "locations" and "," in text:
        s.append("multi_part")
    if item.domain == "dates":
        if g["kind"] == "answered" and parse_date(text) is None:
            s.append("unparseable")
        # Parses only thanks to the "in " retry, the one parsing rule we added.
        if (g["kind"] == "answered" and _PARSER.first_most_specific(text) is None
                and parse_date(text) is not None):
            s.append("retry_parsed")
        if re.fullmatch(r"\d{1,3}(\s*(AD|CE|A\.D\.))?", text.strip(), re.IGNORECASE):
            s.append("short_year")
        if re.search(r"\b(BC|BCE|B\.C\.)", text):
            s.append("bce")
    return s


def build() -> None:
    items = {d: {it.qid: it for it in load(d, "test")} for d in ("locations", "dates")}
    cands, seen = defaultdict(list), set()
    for domain in ("locations", "dates"):
        for path in sorted(glob.glob(str(C.ROOT / "archive" / "phase2_results" / f"transfer_*_on_{domain}.csv"))):
            for r in pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records"):
                key = (domain, r["qid"], r["raw"])
                if key in seen:
                    continue
                seen.add(key)
                it = items[domain][r["qid"]]
                g = grade(it, r["raw"])
                for s in _strata(it, g):
                    cands[(domain, s)].append((it, r["raw"], g, Path(path).name))

    rng, chosen, used = random.Random(C.GLOBAL_SEED), [], set()
    available = {}
    for domain in ("locations", "dates"):
        for s in STRATA[domain]:
            pool = [c for c in cands[(domain, s)] if (domain, c[0].qid, c[1]) not in used]
            available[f"{domain}/{s}"] = len(cands[(domain, s)])
            for c in rng.sample(pool, min(PER_STRATUM[domain], len(pool))):
                used.add((domain, c[0].qid, c[1]))
                chosen.append((s, *c))
    rng.shuffle(chosen)

    sheet, key = [], []
    for i, (s, it, raw, g, src) in enumerate(chosen, 1):
        sheet.append(dict(item=i, domain=it.domain, question=it.question,
                          reference=_reference(it), model_output=raw,
                          human_outcome="", human_level="", notes=""))
        key.append(dict(item=i, stratum=s, source=src, qid=it.qid,
                        scorer_outcome=_outcome(g), scorer_level=_level_label(it, g["level"])))
    CHECK.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sheet).to_csv(CHECK / "sheet.csv", index=False)
    pd.DataFrame(key).to_csv(CHECK / "key.csv", index=False)
    (CHECK / "strata_available.json").write_text(json.dumps(available, indent=2) + "\n")
    (CHECK / "README.md").write_text(
        "# Scorer check (Gate 2)\n\n"
        "Fill in `sheet.csv` without opening `key.csv`.\n\n"
        "- `human_outcome`: `correct`, `wrong`, `abstain`, or `format_failure` (the reply does "
        "not give an answer in the requested `Location:` / `Date:` form and does not decline).\n"
        "- Judge only the primary claim: for places the text before the first comma, for dates "
        "the date as stated at its own precision.\n"
        "- `human_level`, only when correct: for places the number of the reference entry the "
        "answer is correct at; for dates `day`, `month`, `year`, `decade` or `century` (a season "
        "counts as a year, an early/mid/late decade as a decade). Credit cannot be finer than "
        "the reference precision.\n\n"
        "Then run `python -m pipeline.run scorer-check-score`.\n")
    sampled = Counter(f"{sheet[i]['domain']}/{row['stratum']}" for i, row in enumerate(key))
    print(f"wrote {len(sheet)} items to {CHECK / 'sheet.csv'}")
    for k, v in available.items():
        print(f"  {k:28} available {v:5}  sampled {sampled[k]}")


def rekey() -> None:
    """Regrade the labelled answers after a scorer fix. The sample and the labels stay;
    the previous key and sheet are kept as _v<n>, and the reference column is redrawn
    with the reference the scorer now uses."""
    sheet = pd.read_csv(CHECK / "sheet.csv", dtype=str, keep_default_na=False)
    key = pd.read_csv(CHECK / "key.csv", dtype=str, keep_default_na=False)
    v = 1
    while (CHECK / f"key_v{v}.csv").exists():
        v += 1
    key.to_csv(CHECK / f"key_v{v}.csv", index=False)
    sheet.to_csv(CHECK / f"sheet_v{v}.csv", index=False)
    items = {d: {it.qid: it for it in load(d, "test")} for d in ("locations", "dates")}
    where = {r["item"]: (r["qid"], s["domain"], s["model_output"])
             for r, s in zip(key.to_dict("records"), sheet.to_dict("records"))}
    assert all(r["item"] == s["item"] for r, s in zip(key.to_dict("records"), sheet.to_dict("records")))
    changed = []
    for r in key.to_dict("records"):
        qid, domain, output = where[r["item"]]
        it = items[domain][qid]
        g = grade(it, output)
        new = (_outcome(g), _level_label(it, g["level"]))
        if new != (r["scorer_outcome"], r["scorer_level"]):
            changed.append((r["item"], r["scorer_outcome"], r["scorer_level"], *new))
        r["scorer_outcome"], r["scorer_level"] = new
        key.loc[key["item"] == r["item"], ["scorer_outcome", "scorer_level"]] = new
    sheet["reference"] = [_reference(items[where[i][1]][where[i][0]]) for i in sheet["item"]]
    key.to_csv(CHECK / "key.csv", index=False)
    sheet.to_csv(CHECK / "sheet.csv", index=False)
    print(f"rekeyed {len(key)} items with the current scorer (previous key kept as v{v}); "
          f"{len(changed)} scorer decisions changed:")
    for c in changed:
        print(f"  item {c[0]}: {c[1]}/{c[2] or '-'} -> {c[3]}/{c[4] or '-'}")


def _kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def score() -> None:
    sheet = pd.read_csv(CHECK / "sheet.csv", dtype=str, keep_default_na=False)
    key = pd.read_csv(CHECK / "key.csv", dtype=str, keep_default_na=False)
    df = sheet.merge(key, on="item")
    for col in ("human_outcome", "human_level", "scorer_outcome", "scorer_level"):
        df[col] = df[col].str.strip().str.lower()
    df = df[df["human_outcome"] != ""]
    bad = df[~df["human_outcome"].isin(OUTCOMES)]
    if len(bad):
        raise SystemExit(f"unknown human_outcome values in items {list(bad['item'])}")

    outcome_agree = float((df.human_outcome == df.scorer_outcome).mean())
    both = df[(df.human_outcome == "correct") & (df.scorer_outcome == "correct")]
    level_agree = float((both.human_level == both.scorer_level).mean()) if len(both) else None
    mism = df[df.human_outcome != df.scorer_outcome]
    patterns = Counter(zip(mism.domain, mism.scorer_outcome + "->" + mism.human_outcome))
    lvl_mism = both[both.human_level != both.scorer_level]
    patterns.update((d, "level:" + _level_direction(d, sl, hl))
                    for d, sl, hl in zip(lvl_mism.domain, lvl_mism.scorer_level, lvl_mism.human_level))
    systematic = {"/".join(k): v for k, v in patterns.items() if v >= 3}
    passed = (outcome_agree >= 0.95 and (level_agree is None or level_agree >= 0.90)
              and not systematic)
    per_stratum = {s: dict(n=len(d), outcome_agree=float((d.human_outcome == d.scorer_outcome).mean()))
                   for s, d in df.groupby("stratum")}
    result = dict(n=len(df), outcome_agreement=outcome_agree,
                  outcome_kappa=_kappa(list(df.human_outcome), list(df.scorer_outcome)),
                  level_agreement=level_agree, level_n=len(both),
                  per_stratum=per_stratum, systematic_patterns=systematic, passed=passed,
                  disagreements=mism[["item", "stratum", "question", "model_output",
                                      "scorer_outcome", "human_outcome", "notes"]]
                  .to_dict("records"))
    (CHECK / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "disagreements"}, indent=2))
    print("GATE 2 " + ("PASSED" if passed else "NOT PASSED: inspect the patterns above"))
