"""Location weight ablation: rescore the 7B location
evaluations with five weight curves (linear, log of an explicit width, steep, flat,
and 1 at every level) to see if the ranking of the models depends on the curve.
The linear form is already logarithmic, because the dataset's normalized levels
are a log scale.
"""

import csv
import json
import math
import statistics
from pathlib import Path

OUT = Path("phase2_results")

CELLS = {
    "base": "transfer_7b_base_on_locations.csv",
    "in-domain (loc) s0": "transfer_7b_sft_loc_7b_t2_s0_on_locations.csv",
    "in-domain (loc) s1": "transfer_7b_sft_loc_7b_t2_s1_on_locations.csv",
    "in-domain (loc) s2": "transfer_7b_sft_loc_7b_t2_s2_on_locations.csv",
    "transfer (dates) s0": "transfer_7b_dates_sft_7b_s0w2_on_locations.csv",
    "transfer (dates) s1": "transfer_7b_dates_sft_7b_s1w2_on_locations.csv",
    "transfer (dates) s2": "transfer_7b_dates_sft_7b_s2w2_on_locations.csv",
    "transfer (dates) s3": "transfer_7b_dates_sft_7b_s3w2_on_locations.csv",
    "joint s0": "transfer_7b_joint_7b_t2_s0_on_locations.csv",
    "joint s1": "transfer_7b_joint_7b_t2_s1_on_locations.csv",
    "joint s2": "transfer_7b_joint_7b_t2_s2_on_locations.csv",
}
GROUPS = {
    "base": ["base"],
    "in-domain (locations)": [k for k in CELLS if k.startswith("in-domain")],
    "transfer (dates-trained)": [k for k in CELLS if k.startswith("transfer")],
    "joint": [k for k in CELLS if k.startswith("joint")],
}

PRIOR_NL, FINEST_NL = 5.0, 1.0


def linear(l):
    return max(0.0, (PRIOR_NL - l) / (PRIOR_NL - FINEST_NL))


def log_width(l):
    """Explicit width ladder: treat level l as spanning 10^(l-1) units, with the
    prior one decade beyond the coarsest. Normalised so the finest level is 1."""
    w, w0, prior = 10.0 ** (l - 1), 10.0 ** (FINEST_NL - 1), 10.0 ** PRIOR_NL
    return max(0.0, math.log(prior / w) / math.log(prior / w0))


WEIGHTS = {
    "linear": linear,
    "log": log_width,
    "steep": lambda l: linear(l) ** 2,
    "flat": lambda l: math.sqrt(linear(l)),
    "binary": lambda l: 1.0 if l <= PRIOR_NL else 0.0,
}


def main():
    from src.data import load_split
    from src.config import LOCATION_RELATIONS
    from src.text import match_level
    from src.prompt import finest_element
    from src.eval import build_level_pool, answer_norm_level

    test = {q.qid: q for q in load_split("test", LOCATION_RELATIONS)}
    pool = build_level_pool()

    def score(name, w):
        with (OUT / name).open() as f:
            rs = list(csv.DictReader(f))
        assert len(rs) == 303
        total = 0.0
        for r in rs:
            q = test[int(r["qid"])]
            abstain = r["is_idk"] == "True"
            a = r["answer"]
            if abstain:
                continue                      # abstention scores 0 in every column
            sl = match_level(finest_element(a), q.hierarchy)
            if sl is not None and sl < len(q.levels):
                total += w(q.levels[sl])
            else:
                total -= w(answer_norm_level(a or "", pool))
        return total / len(rs)

    table = {label: {cell: score(f, WEIGHTS[label]) for cell, f in CELLS.items()}
             for label in WEIGHTS}

    print(f"{'weighting':10}" + "".join(f"{g:>26}" for g in GROUPS))
    print(f"{'':10}" + "".join(f"{'mean +/- sd':>26}" for _ in GROUPS))
    report, orders = {}, {}
    for label in WEIGHTS:
        cells = []
        line = f"{label:10}"
        for gname, keys in GROUPS.items():
            vals = [table[label][k] for k in keys]
            m = statistics.mean(vals)
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            cells.append((gname, m))
            line += f"{m:>+18.4f} +/-{sd:.3f}"
        print(line)
        order = [g for g, _ in sorted(cells, key=lambda kv: -kv[1])]
        orders[label] = order
        report[label] = {g: m for g, m in cells}

    print("\nranking under each weighting (best first):")
    for label, order in orders.items():
        print(f"  {label:8} {' > '.join(order)}")

    ref = orders["linear"]
    stable = all(o == ref for o in orders.values())
    print(f"\nranking identical under all {len(WEIGHTS)} weightings: {stable}")
    if not stable:
        for label, o in orders.items():
            if o != ref:
                print(f"  DIFFERS under {label}: {' > '.join(o)}")
    # Does every trained group still beat base everywhere?
    beats = {label: all(report[label][g] > report[label]["base"]
                        for g in GROUPS if g != "base") for label in WEIGHTS}
    print(f"every trained group beats base under all weightings: {all(beats.values())}")

    path = OUT / "weight_ablation_sep09.json"
    path.write_text(json.dumps(dict(per_cell=table, per_group=report,
                                    rankings=orders, ranking_stable=stable,
                                    trained_beats_base=beats), indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
