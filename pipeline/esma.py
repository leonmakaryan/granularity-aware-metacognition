"""ESMA-style metacognition measures (Park et al. 2026) for the base and trained models.

    python -m pipeline.esma --model 7b

Direct answer: `pipeline.run eval --mode noidk`, the deployment prompt without its
abstention sentence. C = 1 if that answer is strictly correct at any level; an
abstention or format failure there counts as C = 0. Two Yes/No signals, scored against C:
  behavioural  the deployment answer: any commitment is Yes, "I don't know" is No
               (ESMA's IDK format)
  explicit     ESMA's meta question (pipeline/meta.py): Yes if "yes" is in the reply
d'_type2 = Phi^-1(P(Yes | C=1)) - Phi^-1(P(Yes | C=0)), both rates clipped to
[1e-4, 1 - 1e-4] as in ESMA's metric.py. The explicit signal also gets a type-2 AUROC of
log P(Yes) - log P(No). Uncertainty: the reports' paired bootstrap (units are questions,
persons for dates), seed-averaged, with the same resampled units for every condition.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from . import config as C

FAMILIES = {"base": ["base"],
            **{c: [f"{c}_s{s}" for s in range(3)] for c in ("locations", "dates", "joint")},
            "dates_binary": [f"dates_s{s}_binary" for s in range(3)]}
CLIP = 1e-4


def dprime(h1, n1, h0, n0):
    hr = np.clip(np.divide(h1, n1), CLIP, 1 - CLIP)
    fa = np.clip(np.divide(h0, n0), CLIP, 1 - CLIP)
    return norm.ppf(hr) - norm.ppf(fa)


def auroc(score, c):
    pos, neg = c.sum(), (~c).sum()
    if not pos or not neg:
        return float("nan")
    r = rankdata(score)
    return float((r[c].sum() - pos * (pos + 1) / 2) / (pos * neg))


def load_signals(mdir, adapter, domain, qids):
    """-> dict of aligned boolean arrays, or None when the direct answers are missing."""
    def csv(mode):
        p = mdir / "preds" / f"{adapter}__{domain}_test__{mode}.csv"
        return pd.read_csv(p, dtype={"qid": str}, keep_default_na=False).set_index("qid") if p.exists() else None
    direct, deploy = csv("noidk"), csv("deploy")
    if direct is None or deploy is None:
        return None
    out = {"c": direct.loc[qids, "correct"].astype(str).eq("True").to_numpy(),
           "direct_abstain": direct.loc[qids, "kind"].eq("abstained").to_numpy(),
           "behavioural": deploy.loc[qids, "kind"].ne("abstained").to_numpy()}
    mp = mdir / "meta" / f"{adapter}__{domain}_test__meta.jsonl"
    if mp.exists():
        m = {str(r["qid"]): r for r in map(json.loads, open(mp))}
        out["explicit"] = np.array([bool(m[q]["yes"]) for q in qids])
        out["yes_no"] = np.array([m[q]["yes_no"] for q in qids])
        out["mass"] = np.array([m[q]["mass"] for q in qids])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7b", choices=list(C.MODELS))
    args = ap.parse_args()
    mdir = C.RESULTS / args.model
    report, lines = {}, [f"# {args.model}: ESMA-style d'_type2 (direct answer = no-IDK prompt)\n"]
    for domain in ("locations", "dates"):
        base = pd.read_csv(mdir / "preds" / f"base__{domain}_test__deploy.csv",
                           dtype={"qid": str, "unit": str}, keep_default_na=False)
        qids, units = list(base.qid), list(base.unit)
        groups = {}
        for i, u in enumerate(units):
            groups.setdefault(u, []).append(i)
        member = np.zeros((len(groups), len(qids)))
        for g, idx in enumerate(groups.values()):
            member[g, idx] = 1
        boot = np.random.default_rng(C.GLOBAL_SEED).integers(
            0, len(groups), size=(C.BOOTSTRAP_B, len(groups)))

        def stats(sig, key):
            """full-sample d' and bootstrap replicates for one adapter and signal"""
            y, c = sig[key], sig["c"]
            cnt = member @ np.stack([y & c, c, y & ~c, ~c], 1).astype(float)   # units x 4
            full = dprime(*cnt.sum(0))
            rep = cnt[boot].sum(1)                                              # B x 4
            return full, dprime(rep[:, 0], rep[:, 1], rep[:, 2], rep[:, 3])

        report[domain] = {}
        lines.append(f"\n## {domain} test ({len(qids)} questions)\n")
        for key in ("behavioural", "explicit"):
            lines += [f"\n### {key}\n",
                      "| condition | d' | delta vs base [95% CI] | hit rate | false alarm | yes rate | direct acc | extra |",
                      "|---|---|---|---|---|---|---|---|"]
            base_rep = None
            for fam, adapters in FAMILIES.items():
                sigs = [load_signals(mdir, a, domain, qids) for a in adapters]
                if any(s is None or key not in s for s in sigs):
                    continue
                fulls, reps = zip(*(stats(s, key) for s in sigs))
                full, rep = float(np.mean(fulls)), np.mean(reps, 0)
                row = {"dprime": full, "per_seed": [float(f) for f in fulls],
                       "hit_rate": float(np.mean([s[key][s["c"]].mean() for s in sigs])),
                       "false_alarm": float(np.mean([s[key][~s["c"]].mean() for s in sigs])),
                       "yes_rate": float(np.mean([s[key].mean() for s in sigs])),
                       "direct_acc": float(np.mean([s["c"].mean() for s in sigs])),
                       "direct_abstain": float(np.mean([s["direct_abstain"].mean() for s in sigs]))}
                # A hit or false-alarm count of 0 or all puts d' on ESMA's clip, not the data.
                row["clipped"] = any(int((s[key] & m).sum()) in (0, int(m.sum()))
                                     for s in sigs for m in (s["c"], ~s["c"]))
                extra = f"direct IDK {row['direct_abstain']:.3f}"
                if row["clipped"]:
                    extra += ", d' AT THE CLIP (a rate is 0 or 1): use AUROC"
                if key == "explicit":
                    row["auroc"] = float(np.mean([auroc(s["yes_no"], s["c"]) for s in sigs]))
                    row["yes_no_mass"] = float(np.mean([s["mass"].mean() for s in sigs]))
                    extra += f", AUROC {row['auroc']:.3f}, Yes/No mass {row['yes_no_mass']:.2f}"
                if fam == "base":
                    base_rep, delta = rep, ""
                elif base_rep is None:
                    delta = "no base"
                else:
                    d = rep - base_rep
                    lo, hi = np.nanpercentile(d, [2.5, 97.5])
                    row["delta"] = dict(delta=full - report[domain][key]["base"]["dprime"],
                                        lo=float(lo), hi=float(hi))
                    delta = f"{row['delta']['delta']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                report[domain].setdefault(key, {})[fam] = row
                lines.append(f"| {fam} | {full:.3f} | {delta} | {row['hit_rate']:.3f} | "
                             f"{row['false_alarm']:.3f} | {row['yes_rate']:.3f} | "
                             f"{row['direct_acc']:.3f} | {extra} |")
    (mdir / "report_esma.json").write_text(json.dumps(report, indent=1))
    (mdir / "report_esma.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
