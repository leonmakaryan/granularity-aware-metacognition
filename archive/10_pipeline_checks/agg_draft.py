"""The draft format against the same recipe without the draft line, on the questions
both trained without.

    python 10_pipeline_checks/agg_draft.py locations test
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import config as C
from pipeline.report import load_preds, paired_bootstrap
from pipeline.support import CATEGORIES

domain, split = (sys.argv[1:3] + ["locations", "test"])[:2]
mdir = C.RESULTS / "7b"
_, base = load_preds(mdir, "base", domain, split)
rows = {"base": ([base], None)}
for label, suffix in (("plain (fit)", "_fit"), ("draft (fit)", "_draft_fit")):
    dfs = []
    for s in C.SEEDS:
        p = mdir / "preds" / f"{domain}_s{s}{suffix}__{domain}_{split}__deploy.csv"
        if p.exists():
            dfs.append(load_preds(mdir, f"{domain}_s{s}{suffix}", domain, split)[1])
    if dfs:
        rows[label] = (dfs, [f"seed {s}" for s in range(len(dfs))])

print(f"{domain} {split}: {len(base)} questions, base support match {base.exact.mean():.3f}\n")
print(f"| condition | seeds | support match | delta [95% CI] | signed score | delta [95% CI] "
      f"| correct | abstain | wrong | fmt_fail |")
print("|---|---|---|---|---|---|---|---|---|")
for label, (dfs, _) in rows.items():
    m = float(np.mean([d.exact.mean() for d in dfs]))
    sc = float(np.mean([d.score.mean() for d in dfs]))
    cor = float(np.mean([d.correct.mean() for d in dfs]))
    ab = float(np.mean([d.abstain.mean() for d in dfs]))
    wr = float(np.mean([d.wrong.mean() for d in dfs]))
    ff = int(sum((d.kind == "fmt_fail").sum() for d in dfs))
    if label == "base":
        print(f"| base | 1 | {m:.3f} | | {sc:+.3f} | | {cor:.3f} | {ab:.3f} | {wr:.3f} | {ff} |")
        continue
    dm = paired_bootstrap(base, dfs, "exact")
    ds = paired_bootstrap(base, dfs, "score")
    star = lambda d: "" if d["lo"] <= 0 <= d["hi"] else " *"
    print(f"| {label} | {len(dfs)} | {m:.3f} | {dm['delta']:+.3f} [{dm['lo']:+.3f}, {dm['hi']:+.3f}]{star(dm)} "
          f"| {sc:+.3f} | {ds['delta']:+.3f} [{ds['lo']:+.3f}, {ds['hi']:+.3f}]{star(ds)} "
          f"| {cor:.3f} | {ab:.3f} | {wr:.3f} | {ff} |")

print("\nEstimated-support breakdown (share of questions):\n")
print("| condition | " + " | ".join(c.replace("_", " ") for c in CATEGORIES) + " |")
print("|---|" + "---|" * len(CATEGORIES))
for label, (dfs, _) in rows.items():
    vals = [float(np.mean([(d.category == c).mean() for d in dfs])) for c in CATEGORIES]
    print(f"| {label} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

for label, (dfs, _) in rows.items():
    if label != "base":
        print(f"\n{label} per seed: match " +
              ", ".join(f"{d.exact.mean():.3f}" for d in dfs) +
              " | score " + ", ".join(f"{d.score.mean():+.3f}" for d in dfs))
