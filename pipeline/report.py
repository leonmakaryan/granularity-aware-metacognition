"""Tables for one model: development sets (locations test, dates test) or the
confirmation set (locations val). Reads only this pipeline's prediction files.

Intervals come from a paired bootstrap: each of BOOTSTRAP_B replicates resamples units
(questions; people for dates) with the same indices for the base model and all trained
seeds, and averages the seeds' differences to the base. The 95% interval is the
percentile interval of that average. The spread across seeds is reported separately.
"""
import hashlib
import json
import subprocess
from collections import defaultdict
from importlib import metadata

import numpy as np
import pandas as pd

from . import config as C
from .support import CATEGORIES

SETS = {"dev": [("locations", "test"), ("dates", "test")], "val": [("locations", "val")]}
OTHER = {"locations": "dates", "dates": "locations"}


def load_preds(mdir, adapter, domain, split, mode="deploy"):
    path = mdir / "preds" / f"{adapter}__{domain}_{split}__{mode}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"qid": str, "unit": str})
    df["abstain"] = (df["kind"] == "abstained").astype(float)
    df["wrong"] = ((df["kind"] != "abstained") & ~df["correct"].astype(bool)).astype(float)
    df["correct"] = df["correct"].astype(float)
    return path, df


def paired_bootstrap(base: pd.DataFrame, seeds: list[pd.DataFrame], col: str) -> dict:
    qids = list(base["qid"])
    b = base.set_index("qid").loc[qids, col].astype(float).to_numpy()
    units = defaultdict(list)
    for i, u in enumerate(base["unit"]):
        units[u].append(i)
    groups = [np.array(v) for v in units.values()]
    sizes = np.array([len(g) for g in groups])
    idx = np.random.default_rng(C.GLOBAL_SEED).integers(
        0, len(groups), size=(C.BOOTSTRAP_B, len(groups)))
    denom = sizes[idx].sum(1)
    reps, full = np.zeros(C.BOOTSTRAP_B), []
    for s in seeds:
        d = s.set_index("qid").loc[qids, col].astype(float).to_numpy() - b
        full.append(float(d.mean()))
        unit_sums = np.array([d[g].sum() for g in groups])
        reps += unit_sums[idx].sum(1) / denom
    reps /= len(seeds)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return dict(delta=float(np.mean(full)), lo=float(lo), hi=float(hi),
                seed_sd=float(np.std(full)) if len(full) > 1 else None, per_seed=full)


def summarize(dfs: list[pd.DataFrame]) -> dict:
    out = {k: float(np.mean([d[k].mean() for d in dfs]))
           for k in ("exact", "score", "correct", "abstain", "wrong")}
    out["fmt_fail"] = int(sum((d["kind"] == "fmt_fail").sum() for d in dfs))
    out["categories"] = {c: float(np.mean([(d["category"] == c).mean() for d in dfs]))
                         for c in CATEGORIES}
    return out


def _fmt(ci: dict) -> str:
    flag = "" if ci["lo"] > 0 or ci["hi"] < 0 else " (includes 0)"
    return f"{ci['delta']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]{flag}"


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _marker(name: str):
    path = C.ROOT / name
    return path.read_text().strip() if path.exists() else None


def manifest(files: list) -> dict:
    # On the GPU server the code was a copy, not a git checkout, so the commit came from
    # a CODE_COMMIT file. FROZEN_COMMIT marks the commit at which the method was frozen.
    code = _marker("CODE_COMMIT")
    if code:
        commit = f"{code} (CODE_COMMIT)"
    else:
        commit = subprocess.run(["git", "-C", str(C.ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or "unknown"
    versions = {}
    for pkg in ("torch", "transformers", "trl", "peft", "datasets", "numpy", "pandas"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return dict(commit=commit, frozen_commit=_marker("FROZEN_COMMIT"), versions=versions,
                config={k: (str(v) if not isinstance(v, (int, float, str, list, tuple, dict))
                            else v) for k, v in vars(C).items() if k.isupper()},
                files={str(p.relative_to(C.ROOT)): _sha(p) for p in files})


def main(model: str, which: str) -> None:
    mdir = C.RESULTS / model
    report, lines, used = {}, [f"# {model}: {which} report\n"], []
    for domain, split in SETS[which]:
        bpath, base = load_preds(mdir, "base", domain, split)
        used.append(bpath)
        rows = {"base": {**summarize([base]), "seeds": 1}}
        for label, cond in (("in-domain", domain), ("transfer", OTHER[domain]),
                            ("joint", "joint")):
            seeds = []
            for s in C.SEEDS:
                p, df = load_preds(mdir, f"{cond}_s{s}", domain, split)
                used.append(p)
                assert list(df["qid"]) == list(base["qid"]), f"question order differs: {p}"
                seeds.append(df)
            rows[label] = {**summarize(seeds), "seeds": len(seeds),
                           "delta_exact": paired_bootstrap(base, seeds, "exact"),
                           "delta_score": paired_bootstrap(base, seeds, "score")}
        report[f"{domain}_{split}"] = rows

        lines += [f"\n## {domain} {split} ({len(base)} questions)\n",
                  "| Condition | Support match | Delta [95% CI] | Signed score | Delta [95% CI] "
                  "| Correct | Abstain | Wrong | Seed SD (match / score) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for label, r in rows.items():
            if label == "base":
                lines.append(f"| base | {r['exact']:.3f} | | {r['score']:+.3f} | | "
                             f"{r['correct']:.3f} | {r['abstain']:.3f} | {r['wrong']:.3f} | |")
            else:
                de, ds = r["delta_exact"], r["delta_score"]
                lines.append(f"| {label} | {r['exact']:.3f} | {_fmt(de)} | {r['score']:+.3f} | "
                             f"{_fmt(ds)} | {r['correct']:.3f} | {r['abstain']:.3f} | "
                             f"{r['wrong']:.3f} | {de['seed_sd']:.3f} / {ds['seed_sd']:.3f} |")
        lines += ["\nEstimated-support breakdown (share of questions):\n",
                  "| Condition | " + " | ".join(CATEGORIES) + " |",
                  "|---|" + "---|" * len(CATEGORIES)]
        for label, r in rows.items():
            lines.append(f"| {label} | " + " | ".join(f"{r['categories'][c]:.3f}"
                                                    for c in CATEGORIES) + " |")

    report["manifest"] = manifest(used)
    # The model's own checks, plus the ones that run once for the whole pipeline (stored
    # under 7b, added with a "7b/" prefix).
    gates = {p.name: json.loads(p.read_text()) for p in sorted((mdir / "gates").glob("*.json"))}
    if model != "7b":
        gates |= {f"7b/{p.name}": json.loads(p.read_text())
                  for p in sorted((C.RESULTS / "7b" / "gates").glob("*.json"))
                  if p.name not in gates}
    report["gates"] = gates
    (mdir / f"report_{which}.json").write_text(json.dumps(report, indent=2) + "\n")
    (mdir / f"report_{which}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {mdir / f'report_{which}.md'}")
