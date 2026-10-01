"""1.5B against 7B under the corrected scoring, with paired bootstrap intervals.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("phase2_results")
B, SEED = 10000, 0
PAIRS = [("in-domain", "base"), ("transfer", "base"), ("joint", "base"),
         ("transfer", "in-domain")]

SPEC = {
    "7B": {
        "locations": {
            "base": ["transfer_7b_base_on_locations.csv"],
            "in-domain": [f"transfer_7b_sft_loc_7b_t2_s{s}_on_locations.csv" for s in range(3)],
            "transfer": [f"transfer_7b_dates_sft_7b_s{s}w2_on_locations.csv" for s in range(4)],
            "joint": [f"transfer_7b_joint_7b_t2_s{s}_on_locations.csv" for s in range(3)],
        },
        "dates": {
            "base": ["transfer_7b_base_on_dates.csv"],
            "in-domain": ["transfer_7b_dates_sft_7b_s0w2_on_dates.csv"],
            "transfer": [f"transfer_7b_sft_loc_7b_t2_s{s}_on_dates.csv" for s in range(3)],
            "joint": [f"transfer_7b_joint_7b_t2_s{s}_on_dates.csv" for s in range(3)],
        },
    },
    "1.5B": {
        "locations": {
            "base": ["transfer_1p5b_base_on_locations.csv"],
            "in-domain": ["transfer_1p5b_sft_v2_loc_on_locations.csv"],
            "transfer": (["transfer_1p5b_dates_sft_s0_on_locations.csv"]
                         + [f"transfer_1p5b_dates_sft_s{s}w2_on_locations.csv"
                            for s in (1, 2, 3)]),
            "joint": ["transfer_1p5b_joint_s0w2_on_locations.csv"],
        },
        "dates": {
            "base": ["transfer_1p5b_base_on_dates.csv"],
            "in-domain": ["transfer_1p5b_dates_sft_s0_on_dates.csv"],
            "transfer": ["transfer_1p5b_sft_v2_loc_on_dates.csv"],
            "joint": ["transfer_1p5b_joint_s0w2_on_dates.csv"],
        },
    },
}


def rows(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def main():
    from src.config import LOCATION_RELATIONS
    from src.data import load_split
    from src.text import match_level
    from src.prompt import finest_element
    from src.eval import build_level_pool, signed_informativeness
    import src.dates as D

    test = {q.qid: q for q in load_split("test", LOCATION_RELATIONS)}
    pool = build_level_pool()
    golds = {q.qid: q.gold for q in D.load_dates_split("test")}

    def loc_scores(name):
        out = {}
        for r in rows(name):
            q, a = test[int(r["qid"])], r["answer"]
            abstain = r["is_idk"] == "True"
            sl = None if abstain else match_level(finest_element(a), q.hierarchy)
            out[int(r["qid"])] = signed_informativeness(q, sl, abstain, a, pool)
        return out

    def dat_scores(name):
        out = {}
        for r in rows(name):
            a, abstain, g = r["answer"], r["is_idk"] == "True", golds[r["qid"]]
            v = None if abstain else D.parse_date_text(a)
            out[r["qid"]] = D.signed_date_info(v, abstain, g, True)
        return out

    SCORER = {"locations": loc_scores, "dates": dat_scores}
    report = {}

    for domain in ("locations", "dates"):
        scorer = SCORER[domain]
        qids = sorted(scorer(SPEC["7B"][domain]["base"][0]))
        if domain == "dates":
            g = defaultdict(list)
            for i, q in enumerate(qids):
                g[q.rsplit("_", 1)[0]].append(i)
            units = [np.array(v) for v in g.values()]
        else:
            units = [np.array([i]) for i in range(len(qids))]
        sizes = np.array([len(u) for u in units])
        idx = np.random.default_rng(SEED).integers(0, len(units),
                                                   size=(B, len(units)))

        print(f"\n{'='*74}\n{domain}  ({len(qids)} questions, "
              f"{len(units)} resampling units)\n{'='*74}")
        print(f"{'condition':14}{'1.5B':>12}{'7B':>12}   {'7B - 1.5B':>12}")
        cond = {}
        for scale in ("1.5B", "7B"):
            cond[scale] = {}
            for label, files in SPEC[scale][domain].items():
                per = [scorer(f) for f in files]
                cond[scale][label] = np.array(
                    [np.mean([p[q] for p in per]) for q in qids])
        for label in ("base", "in-domain", "transfer", "joint"):
            a, b = cond["1.5B"][label].mean(), cond["7B"][label].mean()
            print(f"{label:14}{a:>+12.4f}{b:>+12.4f}   {b-a:>+12.4f}")

        dom = {}
        for scale in ("1.5B", "7B"):
            print(f"\n  {scale} paired bootstrap, 95% CI:")
            for x, y in PAIRS:
                d = cond[scale][x] - cond[scale][y]
                us = np.array([d[u].sum() for u in units])
                boot = us[idx].sum(1) / sizes[idx].sum(1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                sig = lo > 0 or hi < 0
                print(f"    {x+' - '+y:24}{d.mean():>+9.4f}"
                      f"{f'[{lo:+.4f}, {hi:+.4f}]':>21}"
                      f"{'' if sig else '   includes 0'}")
                dom[f"{scale}/{x} - {y}"] = dict(diff=float(d.mean()),
                                                 lo=float(lo), hi=float(hi),
                                                 significant=bool(sig))
        report[domain] = dict(
            means={s: {k: float(v.mean()) for k, v in cond[s].items()}
                   for s in cond}, comparisons=dom)

    # ---- gate: the 7B half must match the committed intervals exactly ----
    prev = json.loads((OUT / "bootstrap_ci_sep09.json").read_text())
    bad = []
    for domain in ("locations", "dates"):
        for x, y in PAIRS:
            got = report[domain]["comparisons"][f"7B/{x} - {y}"]
            want = prev[domain][f"{x} - {y}"]
            for k in ("diff", "lo", "hi"):
                if abs(got[k] - want[k]) > 1e-9:
                    bad.append(f"{domain} {x}-{y} {k}: {got[k]} vs {want[k]}")
    print(f"\nGATE: 7B half reproduces bootstrap_ci_sep09.json exactly: "
          f"{'YES' if not bad else 'NO'}")
    for b in bad:
        print("  ", b)

    path = OUT / "scale_comparison_sep09.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
