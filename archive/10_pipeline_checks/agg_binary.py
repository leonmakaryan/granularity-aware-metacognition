"""The answer-or-abstain baseline against the support rule on dates, with the
reports' paired bootstrap (units are people).

    python 10_pipeline_checks/agg_binary.py 7b
"""
import sys

from pipeline import config as C
from pipeline.report import _fmt, load_preds, paired_bootstrap, summarize

model = sys.argv[1] if len(sys.argv) > 1 else "7b"
pattern = sys.argv[2] if len(sys.argv) > 2 else "dates_s{s}_binary"
mdir = C.RESULTS / model
get = lambda a: load_preds(mdir, a, "dates", "test")[1]
base = get("base")
ours = [get(f"dates_s{s}") for s in range(3)]
binary = [get(pattern.format(s=s)) for s in range(3)]


def seed_mean(dfs):
    """One frame whose score and support-match columns are the per-question seed means."""
    out = dfs[0].copy()
    for col in ("score", "exact"):
        out[col] = sum(d.set_index("qid").loc[out.qid, col].to_numpy() for d in dfs) / len(dfs)
    return out


print(f"# {model}: answer-or-abstain baseline vs ours, dates test ({len(base)} questions)"
      + ("" if pattern == "dates_s{s}_binary" else f"  [DRY RUN: '{pattern}' stands in]") + "\n")
print("| condition | signed score | support match | correct | abstain | wrong |")
print("|---|---|---|---|---|---|")
for name, dfs in (("base", [base]), ("ours (support rule)", ours), ("answer-or-abstain", binary)):
    s = summarize(dfs)
    print(f"| {name} | {s['score']:+.3f} | {s['exact']:.3f} | {s['correct']:.3f} | "
          f"{s['abstain']:.3f} | {s['wrong']:.3f} |")
print()
for label, ref, seeds in (("answer-or-abstain - base", base, binary),
                          ("ours - base", base, ours),
                          ("ours - answer-or-abstain", seed_mean(binary), ours)):
    print(f"{label:26s} score {_fmt(paired_bootstrap(ref, seeds, 'score'))}   "
          f"support match {_fmt(paired_bootstrap(ref, seeds, 'exact'))}")
