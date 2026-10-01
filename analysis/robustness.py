"""Robustness subsets for the main comparisons (offline, saved predictions only).

    python analysis/robustness.py > results/robustness.md

1. Dates, exact references only: the 415 test questions whose reference is an exact
   date, so every stated detail can be checked.
2. Locations without the United States: the test split shares no place names with
   training, but 47 test questions name the United States ("United States of
   America"), which training calls "United States". This drops them.
Same paired bootstrap as the reports (pipeline.report.paired_bootstrap), same scorer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import config as C
from pipeline.data import load
from pipeline.report import _fmt, load_preds, paired_bootstrap, summarize
from src.dates import TemporalLevel
from src.text import normalize

US = {"united states", "united states of america", "usa"}


def subset_qids(domain):
    items = load(domain, "test")
    if domain == "dates":
        return {it.qid for it in items if it.gold.level == TemporalLevel.DAY}
    return {it.qid for it in items if not any(normalize(h) in US for h in it.hierarchy)}


def main():
    names = {"dates": "dates test, exact references only", "locations": "locations test without United States questions"}
    for domain in ("dates", "locations"):
        keep = subset_qids(domain)
        other = "locations" if domain == "dates" else "dates"
        print(f"\n## {names[domain]} ({len(keep)} questions)\n")
        print("| model | condition | support match | delta [95% CI] | score | delta [95% CI] | correct | abstain | wrong |")
        print("|---|---|---|---|---|---|---|---|---|")
        for m in ("7b", "3b", "1p5b"):
            mdir = C.RESULTS / m
            base = load_preds(mdir, "base", domain, "test")[1]
            base = base[base.qid.isin(keep)].reset_index(drop=True)
            s = summarize([base])
            print(f"| {m} | base | {s['exact']:.3f} | | {s['score']:+.3f} | | {s['correct']:.3f} | {s['abstain']:.3f} | {s['wrong']:.3f} |")
            conds = [("in-domain", domain), ("transfer", other), ("joint", "joint")]
            if domain == "dates":
                conds.append(("answer-or-abstain", "dates_binary"))
            for label, c in conds:
                pat = "dates_s{s}_binary" if c == "dates_binary" else f"{c}_s{{s}}"
                try:
                    dfs = [load_preds(mdir, pat.format(s=s), domain, "test")[1] for s in C.SEEDS]
                except FileNotFoundError:
                    continue
                dfs = [d[d.qid.isin(keep)].reset_index(drop=True) for d in dfs]
                t = summarize(dfs)
                dm, ds = paired_bootstrap(base, dfs, "exact"), paired_bootstrap(base, dfs, "score")
                print(f"| {m} | {label} | {t['exact']:.3f} | {_fmt(dm)} | {t['score']:+.3f} | {_fmt(ds)} | "
                      f"{t['correct']:.3f} | {t['abstain']:.3f} | {t['wrong']:.3f} |")


if __name__ == "__main__":
    main()
