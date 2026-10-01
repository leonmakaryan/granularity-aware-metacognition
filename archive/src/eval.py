"""Evaluation of the earlier models on the location test set: accuracy, abstention,
d', informativeness, and GranuScore (median and spread).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import OUT_DIR, INFO_FINEST_NL, INFO_PRIOR_NL, LOCATION_RELATIONS
from .data import Question, load_split
from .prompt import is_idk, finest_element
from .scoring import gscore
from .text import match_level, normalize
from .inference import load_trained_model, greedy_answers

# Reference rows from earlier documented runs, for the comparison table.
# (Formulation C = level reward; D = SFT + specificity reward, two seeds.)
PRIOR_RESULTS = {
    "trained-C (level)": dict(acc_on_hier=0.705, selective_acc=0.768, specific_acc=0.049,
                              avg_norm_level=None, idk_rate=0.082, d_prime=2.50, yfr=0.23, nfr=0.00),
    "trained-D run1": dict(acc_on_hier=0.590, selective_acc=0.735, specific_acc=0.115,
                           avg_norm_level=None, idk_rate=0.197, d_prime=3.04, yfr=0.27, nfr=0.00),
    "trained-D run2": dict(acc_on_hier=0.689, selective_acc=0.737, specific_acc=0.082,
                           avg_norm_level=None, idk_rate=0.066, d_prime=2.29, yfr=0.26, nfr=0.00),
}

# Method -> phase1 per-answer CSV, so baselines get a real informativeness number too.
BASELINE_CSV = {"base_direct": "base_direct.csv", "meta_question": "meta_question.csv",
                "drag": "granola.csv"}


@dataclass
class EvalResult:
    method: str
    acc_on_hier: float
    selective_acc: float
    specific_acc: float
    avg_norm_level: float | None
    idk_rate: float
    d_prime: float | None
    yfr: float | None
    nfr: float | None
    info_wacc: float | None = None           # informativeness (Lukas's measure)
    info_signed: float | None = None         # same, wrong answers count negative
    level_mean: float | None = None          # mean and spread of the level the model
    level_std: float | None = None           # chooses, wrong answers included
    info_strict: float | None = None         # only the most specific part of the answer counts
    info_signed_strict: float | None = None


def informativeness(norm_level: float | None) -> float:
    """Information of a correct answer at this normalized level: 1 at the finest
        level, less for coarser ones, 0 for None (see config.INFO_*)."""
    if norm_level is None:
        return 0.0
    info = (INFO_PRIOR_NL - norm_level) / (INFO_PRIOR_NL - INFO_FINEST_NL)
    return max(0.0, info)


def info_weighted_acc(norm_levels: list[float | None]) -> float:
    """Mean informativeness over all questions (0 for wrong answers and abstentions).
        The matcher credits the coarsest level an answer matches, so a wrong city with
        the right country counts at the country level."""
    if not norm_levels:
        return 0.0
    return sum(informativeness(nl) for nl in norm_levels) / len(norm_levels)


def build_level_pool(qs: list[Question] | None = None) -> dict[str, float]:
    """Map each place name in the dataset to its mean normalized level.

        A wrong answer matches nothing in its own hierarchy, so its level is looked up
        here. Levels are relative to each hierarchy, so the mean over all hierarchies a
        place appears in gives its usual granularity."""
    if qs is None:
        qs = load_split("train", LOCATION_RELATIONS) + load_split("test", LOCATION_RELATIONS)
    sums: dict[str, list[float]] = {}
    for q in qs:
        for ent, lv in zip(q.hierarchy, q.levels):
            key = normalize(ent)
            if key:
                sums.setdefault(key, []).append(lv)
    return {k: sum(v) / len(v) for k, v in sums.items()}


# Level of a wrong answer that appears nowhere in the dataset: the middle of the scale.
WRONG_FALLBACK_NL = 2.5


def answer_norm_level(answer: str, pool: dict[str, float],
                      fallback: float = WRONG_FALLBACK_NL) -> float:
    """Estimated normalized level of an answer, right or wrong.

        Looks up the most specific part of the answer, then the whole answer; for a
        sentence, takes the finest place named in it; otherwise returns `fallback`."""
    for cand in (finest_element(answer), answer):
        lv = pool.get(normalize(cand))
        if lv is not None:
            return lv
    na = f" {normalize(answer)} "
    if na.strip():
        hits = [lv for key, lv in pool.items() if f" {key} " in na]
        if hits:
            return min(hits)
    return fallback


def signed_informativeness(q: Question, match_level: int | None, idk: bool,
                           answer: str, pool: dict[str, float]) -> float:
    """Signed score: + information if correct, 0 for an abstention, - information at
        the answer's estimated level if wrong."""
    if match_level is not None and match_level < len(q.levels):
        return informativeness(q.levels[match_level])
    if idk:
        return 0.0
    return -informativeness(answer_norm_level(answer or "", pool))


def strict_match_level(answer: str, hierarchy: list[str]) -> int | None:
    """Match only the most specific part of the answer, so "Bucharest, Romania" is
        wrong if Bucharest is wrong."""
    if not answer:
        return None
    return match_level(finest_element(answer), hierarchy)


def output_level(q: Question, match_level: int | None, idk: bool,
                 answer: str, pool: dict[str, float]) -> float | None:
    """The level the model chose, right or wrong; None for an abstention."""
    if idk:
        return None
    if match_level is not None and match_level < len(q.levels):
        return q.levels[match_level]
    return answer_norm_level(answer or "", pool)


def level_mean_std(levels: list[float | None]) -> tuple[float | None, float | None]:
    """Mean and std of the chosen output levels (committed answers only)."""
    vals = [v for v in levels if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def _baseline_info_wacc(csv_name: str) -> float | None:
    """Compute info-weighted accuracy for a phase1 baseline from its saved per-answer
    CSV (columns level_value = normalized level, NaN when IDK / off-hierarchy)."""
    if not csv_name:
        return None
    path = Path("phase1_results") / csv_name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "level_value" not in df.columns:
        return None
    norm_levels = [float(v) if pd.notna(v) else None for v in df["level_value"]]
    return info_weighted_acc(norm_levels)


def _phi_inv(p: float) -> float:
    eps = 1e-3
    return NormalDist().inv_cdf(max(eps, min(1 - eps, p)))


def metacognition(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs: (committed, correct) per question."""
    if not pairs:
        return {"d_prime": None, "yfr": None, "nfr": None}
    yes = [p for p in pairs if p[0]]
    no = [p for p in pairs if not p[0]]
    correct = [p for p in pairs if p[1]]
    incorrect = [p for p in pairs if not p[1]]
    if correct and incorrect:
        hit = sum(1 for p in correct if p[0]) / len(correct)
        fa = sum(1 for p in incorrect if p[0]) / len(incorrect)
        d = _phi_inv(hit) - _phi_inv(fa)
    else:
        d = None
    yfr = sum(1 for p in yes if not p[1]) / len(yes) if yes else None
    nfr = sum(1 for p in no if p[1]) / len(no) if no else None
    return {"d_prime": d, "yfr": yfr, "nfr": nfr}


def step_eval(adapter: str = "final", relations: list[str] | None = None,
              out_csv: str = "trained_eval.csv", cmp_out: str = "comparison.json"):
    """Evaluate an adapter on the location test set. Pass your own `out_csv` and
        `cmp_out`, or the default result files are overwritten."""
    print(f"=== evaluating trained model ({adapter}) on test set ===")
    model, tok = load_trained_model(adapter)
    test_qs = load_split("test", relations)
    print(f"test: {len(test_qs)} questions")

    answers = greedy_answers(model, tok, test_qs)

    rows = []
    for q, ans in zip(test_qs, answers):
        idk = is_idk(ans)
        ml = match_level(ans, q.hierarchy) if (ans and not idk) else None
        rows.append({"qid": q.qid, "question": q.question, "hierarchy": q.hierarchy,
                     "answer": ans, "is_idk": idk, "match_level": ml})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / out_csv, index=False)

    n = len(df)
    on_hier = int(df["match_level"].notna().sum())
    idk_count = int(df["is_idk"].sum())
    committed = n - idk_count
    specific_correct = sum(1 for r in rows if r["match_level"] == 0)

    levels_committed = [q.levels[r["match_level"]]
                        for q, r in zip(test_qs, rows)
                        if r["match_level"] is not None and r["match_level"] < len(q.levels)]
    avg_level = float(np.mean(levels_committed)) if levels_committed else None

    # Info-weighted accuracy: one entry per question, the matched normalized level or None.
    info_levels = [q.levels[r["match_level"]]
                   if (r["match_level"] is not None and r["match_level"] < len(q.levels))
                   else None
                   for q, r in zip(test_qs, rows)]

    # Signed variant: wrong answers count negative at their estimated level.
    pool = build_level_pool()
    signed_vals = [signed_informativeness(q, r["match_level"], r["is_idk"], r["answer"], pool)
                   for q, r in zip(test_qs, rows)]
    out_levels = [output_level(q, r["match_level"], r["is_idk"], r["answer"], pool)
                  for q, r in zip(test_qs, rows)]
    lvl_mean, lvl_std = level_mean_std(out_levels)

    # Strict: only the most specific part of the answer counts.
    strict_mls = [strict_match_level(r["answer"], q.hierarchy)
                  if (r["answer"] and not r["is_idk"]) else None
                  for q, r in zip(test_qs, rows)]
    strict_info_levels = [q.levels[m] if (m is not None and m < len(q.levels)) else None
                          for q, m in zip(test_qs, strict_mls)]
    strict_signed = [signed_informativeness(q, m, r["is_idk"], r["answer"], pool)
                     for q, m, r in zip(test_qs, strict_mls, rows)]

    pairs = [(not r["is_idk"], r["match_level"] is not None) for r in rows]
    meta = metacognition(pairs)

    trained = EvalResult(
        method=f"B'-{adapter}",
        acc_on_hier=on_hier / n,
        selective_acc=on_hier / committed if committed else 0.0,
        specific_acc=specific_correct / n,
        avg_norm_level=avg_level,
        idk_rate=idk_count / n,
        d_prime=meta["d_prime"], yfr=meta["yfr"], nfr=meta["nfr"],
        info_wacc=info_weighted_acc(info_levels),
        info_signed=float(np.mean(signed_vals)) if signed_vals else None,
        level_mean=lvl_mean, level_std=lvl_std,
        info_strict=info_weighted_acc(strict_info_levels),
        info_signed_strict=float(np.mean(strict_signed)) if strict_signed else None)

    # Per-relation breakdown when the test set spans more than one relation.
    rels_present = sorted({q.relation for q in test_qs if q.relation})
    if len(rels_present) > 1:
        print("\nper-relation (trained):")
        for rel in rels_present:
            idx = [i for i, q in enumerate(test_qs) if q.relation == rel]
            oh = sum(1 for i in idx if rows[i]["match_level"] is not None)
            idk_r = sum(1 for i in idx if rows[i]["is_idk"])
            print(f"  {rel}: n={len(idx):3d}  acc_hier={oh/len(idx):.3f}  "
                  f"info={info_weighted_acc([info_levels[i] for i in idx]):.3f}  "
                  f"info_signed={float(np.mean([signed_vals[i] for i in idx])):.3f}  "
                  f"idk={idk_r/len(idx):.3f}")

    matched = [r["answer"] for r in rows if r["match_level"] is not None]
    if matched:
        gs = gscore(matched)
        print(f"\nGranuScore of on-hierarchy answers: "
              f"median={np.median(gs):.1f}  mean={np.mean(gs):.1f}  std={np.std(gs):.1f}")

    table = []
    phase1 = Path("phase1_results/summary.json")
    if phase1.exists():
        with open(phase1) as f:
            p1 = json.load(f)
        for key in ["base_direct", "meta_question", "drag"]:
            s = p1[key]
            mc = s.get("metacognition", {})
            table.append(EvalResult(
                method=key, acc_on_hier=s.get("acc_on_hierarchy"),
                selective_acc=s.get("selective_acc"),
                specific_acc=s.get("specific_accuracy"),
                avg_norm_level=s.get("avg_normalized_level"),
                idk_rate=s.get("idk", 0) / s.get("n", 1),
                d_prime=mc.get("d_prime"), yfr=mc.get("yfr"), nfr=mc.get("nfr"),
                info_wacc=_baseline_info_wacc(BASELINE_CSV.get(key))))
    else:
        print("(phase1 summary not found — trained only)")

    for name, d in PRIOR_RESULTS.items():
        table.append(EvalResult(method=name, **d))
    table.append(trained)

    def fmt(v, p=3):
        if v is None:
            return "—"
        return f"{v:.{p}f}" if isinstance(v, float) else str(v)

    def fmt_lvl(r):
        if r.level_mean is None:
            return "—"
        return f"{r.level_mean:.2f}±{r.level_std:.2f}"

    headers = ["method", "acc_hier", "sel_acc", "spec_acc", "info", "info_signed",
               "signed_strict", "lvl μ±σ", "idk%", "d′", "YFR", "NFR"]
    print("\n" + " | ".join(f"{h:>15s}" for h in headers))
    print("-" * (16 * len(headers)))
    for r in table:
        print(" | ".join(f"{x:>15s}" for x in [
            r.method[:15], fmt(r.acc_on_hier), fmt(r.selective_acc), fmt(r.specific_acc),
            fmt(r.info_wacc), fmt(r.info_signed), fmt(r.info_signed_strict),
            fmt_lvl(r), fmt(r.idk_rate), fmt(r.d_prime), fmt(r.yfr), fmt(r.nfr)]))

    with open(OUT_DIR / cmp_out, "w") as f:
        json.dump({r.method: r.__dict__ for r in table}, f, indent=2)
    print(f"\nresults: {OUT_DIR}/{out_csv}  {OUT_DIR}/{cmp_out}")
    return trained
