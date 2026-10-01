"""Paired bootstrap intervals for the main score differences of the earlier models.
Resamples test questions for locations and people for dates (a person's two
questions are not independent); 10,000 resamples, 95% percentile intervals.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("phase2_results")
B = 10000
SEED = 0

LOC = {
    "base": ["transfer_7b_base_on_locations.csv"],
    "in-domain": [f"transfer_7b_sft_loc_7b_t2_s{s}_on_locations.csv" for s in range(3)],
    "transfer": [f"transfer_7b_dates_sft_7b_s{s}w2_on_locations.csv" for s in range(4)],
    "joint": [f"transfer_7b_joint_7b_t2_s{s}_on_locations.csv" for s in range(3)],
}
DAT = {
    "base": ["transfer_7b_base_on_dates.csv"],
    "in-domain": ["transfer_7b_dates_sft_7b_s0w2_on_dates.csv"],
    "transfer": [f"transfer_7b_sft_loc_7b_t2_s{s}_on_dates.csv" for s in range(3)],
    "joint": [f"transfer_7b_joint_7b_t2_s{s}_on_dates.csv" for s in range(3)],
}
PAIRS = [("in-domain", "base"), ("transfer", "base"), ("joint", "base"),
         ("transfer", "in-domain"), ("joint", "in-domain"), ("joint", "transfer")]


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

    report = {}
    for domain, spec, scorer in (("locations", LOC, loc_scores),
                                 ("dates", DAT, dat_scores)):
        # Mean per-question score for each condition, averaged over its seeds.
        cond = {}
        for label, files in spec.items():
            per = [scorer(f) for f in files]
            qids = sorted(per[0])
            cond[label] = np.array([np.mean([p[q] for p in per]) for q in qids])
        qids = sorted(scorer(spec["base"][0]))

        # Resampling units: questions for locations, people for dates.
        if domain == "dates":
            groups = defaultdict(list)
            for i, q in enumerate(qids):
                groups[q.rsplit("_", 1)[0]].append(i)
            units = [np.array(v) for v in groups.values()]
            unit_name = f"{len(units)} people"
        else:
            units = [np.array([i]) for i in range(len(qids))]
            unit_name = f"{len(units)} questions"

        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(units), size=(B, len(units)))
        # Precompute per-unit sums and sizes so each resample is a gather.
        sizes = np.array([len(u) for u in units])
        print(f"\n=== {domain} ({unit_name}, {len(qids)} questions, "
              f"{B} resamples) ===")
        print(f"{'comparison':28}{'diff':>9}{'95% CI':>20}{'P(>0)':>9}")
        dom = {}
        for a, b in PAIRS:
            d = cond[a] - cond[b]
            unit_sums = np.array([d[u].sum() for u in units])
            boot = unit_sums[idx].sum(1) / sizes[idx].sum(1)
            obs = d.mean()
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p = float((boot > 0).mean())
            flag = "" if lo > 0 or hi < 0 else "   includes 0"
            print(f"{a+' - '+b:28}{obs:>+9.4f}"
                  f"{f'[{lo:+.4f}, {hi:+.4f}]':>20}{p:>9.3f}{flag}")
            dom[f"{a} - {b}"] = dict(diff=float(obs), lo=float(lo), hi=float(hi),
                                     p_gt_0=p, significant=bool(lo > 0 or hi < 0))
        report[domain] = dict(unit=unit_name, n_questions=len(qids), **dom)

    path = OUT / "bootstrap_ci_sep09.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
