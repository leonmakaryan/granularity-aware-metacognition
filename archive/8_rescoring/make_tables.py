"""Result tables of the earlier models under the corrected scoring, recomputed from
the saved answers.
"""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("phase2_results")
GEN = Path("writing/generated")
B, SEED = 10000, 0

CELLS = {
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
                         + [f"transfer_1p5b_dates_sft_s{s}w2_on_locations.csv" for s in (1, 2, 3)]),
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
ORDER = ["base", "in-domain", "transfer", "joint"]
LONG = {"base": "Base (IDK prompt)", "in-domain": "In-domain SFT",
        "transfer": "Cross-domain transfer", "joint": "Joint training"}


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

    def loc(name):
        out = {}
        for r in rows(name):
            q, a = test[int(r["qid"])], r["answer"]
            ab = r["is_idk"] == "True"
            sl = None if ab else match_level(finest_element(a), q.hierarchy)
            out[int(r["qid"])] = (signed_informativeness(q, sl, ab, a, pool),
                                  sl is not None, ab, not ab and sl is None)
        return out

    def dat(name):
        out = {}
        for r in rows(name):
            a, ab, g = r["answer"], r["is_idk"] == "True", golds[r["qid"]]
            v = None if ab else D.parse_date_text(a)
            ok = not ab and D.credited_level(v, g, True) is not None
            out[r["qid"]] = (D.signed_date_info(v, ab, g, True), ok, ab,
                             not ab and not ok)
        return out

    SC = {"locations": loc, "dates": dat}
    data = {}
    for scale in CELLS:
        for dom in ("locations", "dates"):
            for cond, files in CELLS[scale][dom].items():
                per = [SC[dom](f) for f in files]
                qids = sorted(per[0])
                data[(scale, dom, cond)] = dict(
                    score=np.array([np.mean([p[q][0] for p in per]) for q in qids]),
                    correct=np.mean([np.mean([p[q][1] for p in per]) for q in qids]),
                    idk=np.mean([np.mean([p[q][2] for p in per]) for q in qids]),
                    wrong=np.mean([np.mean([p[q][3] for p in per]) for q in qids]),
                    seeds=len(files),
                    seed_sd=(statistics.pstdev(
                        [np.mean([p[q][0] for q in qids]) for p in per])
                        if len(files) > 1 else None))

    # bootstrap machinery per domain
    boot = {}
    for dom in ("locations", "dates"):
        qids = sorted(SC[dom](CELLS["7B"][dom]["base"][0]))
        if dom == "dates":
            g = defaultdict(list)
            for i, q in enumerate(qids):
                g[q.rsplit("_", 1)[0]].append(i)
            units = [np.array(v) for v in g.values()]
        else:
            units = [np.array([i]) for i in range(len(qids))]
        boot[dom] = (units, np.array([len(u) for u in units]),
                     np.random.default_rng(SEED).integers(0, len(units),
                                                          size=(B, len(units))),
                     len(qids))

    def ci(dom, scale, cond):
        units, sizes, idx, _ = boot[dom]
        d = data[(scale, dom, cond)]["score"] - data[(scale, dom, "base")]["score"]
        us = np.array([d[u].sum() for u in units])
        bs = us[idx].sum(1) / sizes[idx].sum(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return d.mean(), lo, hi

    md, tex = [], []
    md.append("# Main results, corrected scoring (9 September 2026)\n")
    md.append("Strict signed informativeness. Higher is better; always abstaining "
              "scores 0. Intervals are 95% paired bootstrap over test units "
              "(locations: 303 questions; dates: 258 people, since birth and death "
              "questions share a person). `+/-` is the standard deviation over "
              "training seeds, which is a different and much smaller quantity.\n")

    for dom in ("locations", "dates"):
        n = boot[dom][3]
        md.append(f"\n## {dom.capitalize()} ({n} test questions), 7B\n")
        md.append("| Condition | Score | Seed sd | Correct | Abstain | Wrong | "
                  "Gain over base | 95% CI |")
        md.append("|---|---|---|---|---|---|---|---|")
        tex.append(r"\begin{table}[t]\centering")
        tex.append(r"\caption{%s, 7B, %d test questions. Strict signed "
                   r"informativeness with 95\%% paired bootstrap intervals on the "
                   r"gain over the base model.}" % (dom.capitalize(), n))
        tex.append(r"\begin{tabular}{lrrrrrr}\toprule")
        tex.append(r"Condition & Score & Correct & Abstain & Wrong & Gain & 95\% CI \\\midrule")
        for cond in ORDER:
            d = data[("7B", dom, cond)]
            sd = f"{d['seed_sd']:.3f}" if d["seed_sd"] is not None else "-"
            if cond == "base":
                md.append(f"| {LONG[cond]} | {d['score'].mean():+.4f} | {sd} | "
                          f"{d['correct']:.3f} | {d['idk']:.3f} | {d['wrong']:.3f} | - | - |")
                tex.append(r"%s & $%+.4f$ & %.3f & %.3f & %.3f & -- & -- \\"
                           % (LONG[cond], d["score"].mean(), d["correct"], d["idk"], d["wrong"]))
            else:
                g, lo, hi = ci(dom, "7B", cond)
                star = "" if (lo > 0 or hi < 0) else " (includes 0)"
                md.append(f"| {LONG[cond]} | {d['score'].mean():+.4f} | {sd} | "
                          f"{d['correct']:.3f} | {d['idk']:.3f} | {d['wrong']:.3f} | "
                          f"{g:+.4f} | [{lo:+.4f}, {hi:+.4f}]{star} |")
                tex.append(r"%s & $%+.4f$ & %.3f & %.3f & %.3f & $%+.4f$ & $[%+.4f, %+.4f]$%s \\"
                           % (LONG[cond], d["score"].mean(), d["correct"], d["idk"],
                              d["wrong"], g, lo, hi,
                              "" if (lo > 0 or hi < 0) else r"$^{\dagger}$"))
        tex.append(r"\bottomrule\end{tabular}")
        tex.append(r"\begin{flushleft}\footnotesize $\dagger$ interval includes zero."
                   r"\end{flushleft}\end{table}")
        md.append("")

    # two-scale gains
    md.append("\n## Gain from training, by scale\n")
    md.append("The absolute score rises sharply with scale, but the gain from "
              "training does not. Five of six cells shrink.\n")
    md.append("| Domain | Condition | 1.5B score | 7B score | 1.5B gain | 7B gain |")
    md.append("|---|---|---|---|---|---|")
    tex.append(r"\begin{table}[t]\centering")
    tex.append(r"\caption{Gain from training by model scale. The absolute score "
               r"rises with scale while the gain from training does not.}")
    tex.append(r"\begin{tabular}{llrrrr}\toprule")
    tex.append(r"Domain & Condition & \multicolumn{2}{c}{Score} & "
               r"\multicolumn{2}{c}{Gain over base} \\")
    tex.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6} & & 1.5B & 7B & 1.5B & 7B \\\midrule")
    for dom in ("locations", "dates"):
        for cond in ORDER:
            a = data[("1.5B", dom, cond)]["score"].mean()
            b = data[("7B", dom, cond)]["score"].mean()
            if cond == "base":
                md.append(f"| {dom} | {LONG[cond]} | {a:+.4f} | {b:+.4f} | - | - |")
                tex.append(r"%s & %s & $%+.4f$ & $%+.4f$ & -- & -- \\"
                           % (dom, LONG[cond], a, b))
            else:
                ga = ci(dom, "1.5B", cond)[0]
                gb = ci(dom, "7B", cond)[0]
                md.append(f"| {dom} | {LONG[cond]} | {a:+.4f} | {b:+.4f} | "
                          f"{ga:+.4f} | {gb:+.4f} |")
                tex.append(r"%s & %s & $%+.4f$ & $%+.4f$ & $%+.4f$ & $%+.4f$ \\"
                           % (dom, LONG[cond], a, b, ga, gb))
        if dom == "locations":
            tex.append(r"\midrule")
    tex.append(r"\bottomrule\end{tabular}\end{table}")

    # weight ablation
    wa = json.loads((OUT / "weight_ablation_sep09.json").read_text())
    md.append("\n## Location weight ablation\n")
    md.append("Every trained condition beats base under all five weight curves. "
              "The transfer-versus-in-domain ordering flips, so that gap is inside "
              "the metric's shape sensitivity.\n")
    md.append("| Weighting | base | in-domain | transfer | joint |")
    md.append("|---|---|---|---|---|")
    keymap = {"base": "base", "in-domain": "in-domain (locations)",
              "transfer": "transfer (dates-trained)", "joint": "joint"}
    for w, per in wa["per_group"].items():
        md.append(f"| {w} | " + " | ".join(f"{per[keymap[c]]:+.4f}" for c in ORDER) + " |")

    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "main_results.md").write_text("\n".join(md) + "\n")
    (GEN / "main_results.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))
    print(f"\nwrote {GEN/'main_results.md'} and {GEN/'main_results.tex'}")


if __name__ == "__main__":
    main()
