"""Level readout: keep the base model's greedy answer and choose from the model's own
signals whether to keep it, make it coarser, or abstain. A logistic regression on
log-probabilities, the model's yes/no verdict on its answer, sample agreement and
one hidden-state vector. It worked on locations but did not go into the thesis.
(Belonged to pipeline/.)
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dates import DATE_LADDER, credited_level
from src.prompt import finest_element
from src.text import normalize

from . import config as C
from .data import holdout, load
from .report import paired_bootstrap
from .score import extract, grade, ladder_index, parse_date
from .support import category

ABST = {"kind": "abstained", "text": "", "correct": False, "level": None, "score": 0.0}
LEVELS = [L.name for L in DATE_LADDER]
CONF_FRAC, CONF_SEED = 0.20, 11      # confirmation slice of train, fixed before any result
BIASES = np.round(np.arange(-4.0, 4.01, 0.25), 2)


def entity_key(domain, item):
    if domain == "dates":
        return item.unit
    s = item.question.strip().rstrip("?")
    s = re.sub(r"^Where (is|was|are|did) (the (headquarters?|capital) of )?", "", s, flags=re.I)
    s = re.sub(r"\s+(located|born|die|died|founded|from|situated)$", "", s, flags=re.I)
    return s.strip().lower()


def read_jsonl(path, key="qid"):
    return {json.loads(l)[key]: json.loads(l) for l in open(path)} if path.exists() else {}


def agreement(domain, item, raws, g):
    """How often the K deployment samples agree with the greedy answer."""
    if domain == "locations":
        gk = normalize(finest_element(g["text"])) if g["kind"] == "answered" else None
        keys = [normalize(finest_element(extract(item, r)[1])) if extract(item, r)[0] == "answered"
                else None for r in raws]
        same = [float(k is not None and k == gk) for k in keys]
        top = max(Counter(k for k in keys if k).values(), default=0) / len(raws)
        return [float(np.mean(same)), float(np.mean([k is None for k in keys])), top,
                len({k for k in keys if k}) / len(raws)]
    vg = parse_date(g["text"]) if g["kind"] == "answered" else None
    lv = []
    for r in raws:
        k, t = extract(item, r)
        v = parse_date(t) if k == "answered" else None
        c = credited_level(v, vg, strict=False) if (v is not None and vg is not None) else None
        lv.append(None if c is None else ladder_index(c))
    per = [float(np.mean([e is not None and e <= L for e in lv])) for L in range(len(DATE_LADDER))]
    return per + [float(np.mean([e is None for e in lv]))]


def build(domain, split, part="all"):
    """-> (rows DataFrame, hidden-state arrays, layer list)."""
    chosen = load(domain, split)
    if split == "train" and part != "all":
        fit_items, conf_items = holdout(chosen, CONF_FRAC, CONF_SEED)
        if part == "confirm" and os.environ.get("CONFIRM_SLICE") != "1":
            raise RuntimeError("the confirmation slice is read only after the freeze")
        chosen = fit_items if part == "fit" else conf_items
    items = {it.qid: it for it in chosen}
    fdir = C.RESULTS / "7b" / "features"
    z = np.load(fdir / f"base__{domain}_{split}.npz")
    meta = [json.loads(l) for l in open(fdir / f"base__{domain}_{split}.jsonl")]
    order = [str(q) for q in z["qid"]]
    assert order == [m["qid"] for m in meta]
    keep = [i for i, q in enumerate(order) if q in items]
    assert len(keep) == len(items), "features missing for some questions"
    ver = read_jsonl(fdir / f"base__{domain}_{split}__verify.jsonl")
    smp = read_jsonl(C.RESULTS / "7b" / "samples" / f"{domain}_{split}.jsonl")
    preds = pd.read_csv(C.RESULTS / "7b" / "preds" / f"base__{domain}_{split}__deploy.csv",
                        dtype={"qid": str, "unit": str}, keep_default_na=False).set_index("qid")
    rows = []
    for i in keep:
        qid, m = order[i], meta[i]
        it, p = items[qid], preds.loc[qid]
        assert p.raw == m["raw"]
        sup = None if p.support == "" else int(float(p.support))
        g = grade(it, m["raw"])
        opts = {}
        if g["kind"] == "abstained":
            opts["abstain"] = g
        else:
            if domain == "locations":
                opts["answer"] = g
            else:
                names = [c["name"] for c in m["candidates"] if c["name"] in LEVELS]
                if names:
                    opts[names[0]] = g
                    for c in m["candidates"]:
                        if c["name"] in names[1:]:
                            opts[c["name"]] = grade(it, c["text"])
                else:
                    opts["answer"] = g
            opts["abstain"] = ABST
        cats = {k: category(sup, v) for k, v in opts.items()}
        exact = [k for k, c in cats.items() if c == "exact_support_match"]
        lp = {c["name"]: c["lp"] for c in m["candidates"]}
        glp = np.array(m["greedy_lp"][:-1] or [0.0])
        blocks = {"lp": [glp.mean(), glp.min(), glp.sum(), m["greedy_lp"][-1],
                         lp["abstain"] - lp["greedy"]]}
        if domain == "dates":
            blocks["lp"] += [lp.get(L, lp["greedy"] - 30.0) - lp["greedy"] for L in LEVELS]
        v = ver.get(qid, {}).get("verify", {})
        gy = v.get("greedy", {}).get("yes_no", 0.0)
        blocks["vf"] = [gy, float(bool(v))]
        if domain == "dates":
            blocks["vf"] += [v[L]["yes_no"] - gy if L in v and "greedy" in v else 0.0
                             for L in LEVELS]
        blocks["ag"] = agreement(domain, it, smp[qid]["raws"], g) if qid in smp else []
        rows.append(dict(qid=qid, unit=it.unit, group=entity_key(domain, it),
                         relation=p.relation, support=sup, opts=opts, cats=cats,
                         base_cat=category(sup, g),
                         base_exact=float(category(sup, g) == "exact_support_match"),
                         base_score=g["score"], decision=len(opts) > 1,
                         label=exact[0] if len(exact) == 1 else None, **blocks))
    df = pd.DataFrame(rows)
    return df, {k: z[k][keep] for k in ("h_prompt", "h_last", "h_mean")}, list(z["layers"])


def matrix(df, H, layers, spec):
    parts = []
    for s in spec.split("+"):
        if s.startswith("h"):
            _, pos, layer = s.split(":")
            parts.append(H["h_" + pos][:, layers.index(int(layer))].astype(np.float32))
        else:
            parts.append(np.array(df[s].tolist(), dtype=np.float32))
    return np.hstack(parts)


def classes_for(domain):
    return ["answer", "abstain"] if domain == "locations" else LEVELS + ["answer", "abstain"]


def fit(X, y, C_):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C_, max_iter=5000)).fit(X, y)


def proba_all(model, X, classes):
    p = np.zeros((len(X), len(classes)))
    p[:, [classes.index(c) for c in model.classes_]] = model.predict_proba(X)
    return p


def fit_mask(df):
    return (df.decision & df.label.notna()).to_numpy()


def option_matrices(df, classes):
    """Per question and option: allowed, whether it is an exact support match, its score."""
    n, k = len(df), len(classes)
    allowed, ex, sc = np.zeros((n, k), bool), np.zeros((n, k)), np.zeros((n, k))
    for i, (_, r) in enumerate(df.iterrows()):
        for key, g in r.opts.items():
            j = classes.index(key)
            allowed[i, j] = True
            ex[i, j] = float(r.cats[key] == "exact_support_match")
            sc[i, j] = g["score"]
    return allowed, ex, sc


def bias_vector(bias, classes):
    if np.isscalar(bias):
        v = np.zeros(len(classes))
        v[classes.index("abstain")] = bias
        return v
    return np.asarray(bias, dtype=float)


def choose(proba, allowed, bias, classes):
    """Index of the chosen option per question. A question with one option keeps it."""
    s = np.log(proba + 1e-9) + bias_vector(bias, classes)
    return np.where(allowed, s, -np.inf).argmax(1)


def decide(df, proba, classes, bias, allowed=None):
    allowed = option_matrices(df, classes)[0] if allowed is None else allowed
    return [classes[j] for j in choose(proba, allowed, bias, classes)]


def choice_shares(idx, allowed, classes):
    multi = allowed.sum(1) > 1                      # questions with a real choice
    counts = np.bincount(idx[multi], minlength=len(classes))
    return counts / max(multi.sum(), 1)


def match_rates(proba, allowed, classes, target, rounds=300, damp=0.3):
    """Per-class offsets that make the chosen levels match the target rate. Uses only
        the unlabelled evaluation questions, because a fixed probability cutoff did not
        carry over from train to test."""
    b, best, best_d = np.zeros(len(classes)), np.zeros(len(classes)), np.inf
    for _ in range(rounds):
        cur = choice_shares(choose(proba, allowed, b, classes), allowed, classes)
        d = float(np.abs(cur - target).sum())
        if d < best_d:
            best, best_d = b.copy(), d
        if d < 1e-3:
            break
        b = b + damp * np.log((target + 1e-3) / (cur + 1e-3))
    return best, best_d


def outcome(idx, ex, sc):
    """Support-match indicator and signed score of the chosen option per question."""
    rows = np.arange(len(idx))
    return ex[rows, idx], sc[rows, idx]


def oof_proba(df, X, classes, C_, scheme):
    """Out-of-fold probabilities: leave-one-relation-out, or 5 folds grouped by entity."""
    proba = np.zeros((len(df), len(classes)))
    fm = fit_mask(df)
    y = df.label.to_numpy()
    if scheme == "loro":
        splits = [(np.where(df.relation.to_numpy() != r)[0], np.where(df.relation.to_numpy() == r)[0])
                  for r in sorted(set(df.relation))]
    else:
        splits = list(GroupKFold(n_splits=5).split(X, groups=df.group))
    for a, b in splits:
        al = a[fm[a]]
        proba[b] = proba_all(fit(X[al], y[al], C_), X[b], classes)
    return proba


def select(df, H, layers, specs, Cs, scheme):
    classes, best = classes_for(df.attrs["domain"]), None
    allowed, exm, scm = option_matrices(df, classes)
    for spec in specs:
        X = matrix(df, H, layers, spec)
        for C_ in Cs:
            proba = oof_proba(df, X, classes, C_, scheme)
            bias = max(BIASES, key=lambda b: (outcome(choose(proba, allowed, b, classes), exm, scm)[0].mean(),
                                              -abs(b)))
            idx = choose(proba, allowed, bias, classes)
            ex, sc = outcome(idx, exm, scm)
            row = dict(spec=spec, C=C_, bias=float(bias), oof_match=float(ex.mean()),
                       oof_score=float(sc.mean()),
                       target_shares=choice_shares(idx, allowed, classes).tolist())
            print(f"  {spec:24s} C={C_:<8} bias={bias:+.2f}  {scheme} match {ex.mean():.3f} "
                  f"(base {df.base_exact.mean():.3f})  score {sc.mean():+.3f} "
                  f"(base {df.base_score.mean():+.3f})", flush=True)
            if best is None or row["oof_match"] > best["oof_match"]:
                best = row
    return best


def evaluate(name, tr, Htr, layers, ev, Hev, choice_cfg, domain, point="rate"):
    classes = classes_for(domain)
    fm = fit_mask(tr)
    model = fit(matrix(tr, Htr, layers, choice_cfg["spec"])[fm], tr.label.to_numpy()[fm], choice_cfg["C"])
    proba = proba_all(model, matrix(ev, Hev, layers, choice_cfg["spec"]), classes)
    allowed, exm, scm = option_matrices(ev, classes)
    if point == "rate":
        bias, gap = match_rates(proba, allowed, classes, np.array(choice_cfg["target_shares"]))
        print(f"   rate-matched biases {np.round(bias, 2).tolist()} (share gap {gap:.3f})")
    else:
        bias = choice_cfg["bias"]
    idx = choose(proba, allowed, bias, classes)
    choice = [classes[j] for j in idx]
    ex, sc = outcome(idx, exm, scm)
    out, base = ev[["qid", "unit"]].copy(), ev[["qid", "unit"]].copy()
    out["exact"], out["score"] = ex, sc
    out["choice"], out["support"] = choice, ev.support.to_numpy()
    out["category"] = [r.cats[c] for (_, r), c in zip(ev.iterrows(), choice)]
    out["base_exact"], out["base_score"] = ev.base_exact.to_numpy(), ev.base_score.to_numpy()
    out["base_category"] = ev.base_cat.to_numpy()
    csv = C.RESULTS / "7b" / "readout" / f"{name.replace(' ', '_').replace('/', '-').replace('[', '').replace(']', '')}.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv, index=False)
    base["exact"], base["score"] = ev.base_exact.to_numpy(), ev.base_score.to_numpy()
    dm, ds = paired_bootstrap(base, [out], "exact"), paired_bootstrap(base, [out], "score")
    print(f"{name}: n={len(ev)} match {ev.base_exact.mean():.3f} -> {ex.mean():.3f} "
          f"delta {dm['delta']:+.3f} [{dm['lo']:+.3f}, {dm['hi']:+.3f}] | "
          f"score {ev.base_score.mean():+.3f} -> {sc.mean():+.3f} "
          f"delta {ds['delta']:+.3f} [{ds['lo']:+.3f}, {ds['hi']:+.3f}]")
    rng = np.random.default_rng(0)
    ctl = []
    for _ in range(20):                     # same rates, random choice of which questions
        perm = proba[rng.permutation(len(proba))]
        b, _g = match_rates(perm, allowed, classes, np.array(choice_cfg["target_shares"]))
        ctl.append(outcome(choose(perm, allowed, b, classes), exm, scm)[0].mean())
    print(f"   control (same rates, questions chosen at random): match {np.mean(ctl):.3f} "
          f"+- {np.std(ctl):.3f}")
    print("   choices:", pd.Series(choice).value_counts().to_dict())
    print("   categories:", pd.Series([r.cats[c] for (_, r), c in zip(ev.iterrows(), choice)])
          .value_counts(normalize=True).round(3).to_dict())
    return dict(name=name, n=len(ev), control_match=float(np.mean(ctl)),
                base_match=float(ev.base_exact.mean()), match=float(ex.mean()),
                delta_match=dm, base_score=float(ev.base_score.mean()), score=float(sc.mean()),
                delta_score=ds, choices=dict(pd.Series(choice).value_counts()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["locations", "dates"])
    ap.add_argument("--eval", default="test")
    ap.add_argument("--eval-part", default="all", choices=["all", "fit", "confirm"])
    ap.add_argument("--specs", default="lp,lp+vf,lp+vf+ag,h:mean:20,h:mean:20+lp+vf,h:last:16+lp+vf")
    ap.add_argument("--Cs", default="0.0001,0.0003,0.001,0.003")
    ap.add_argument("--scheme", default=None)
    ap.add_argument("--point", default="rate", choices=["rate", "bias"])
    ap.add_argument("--config", default=None,
                    help="frozen selection to apply instead of selecting again")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    scheme = args.scheme or ("loro" if args.domain == "locations" else "group")
    tr, Htr, layers = build(args.domain, "train", "fit")
    tr.attrs["domain"] = args.domain
    print(f"##### {args.domain}: train-fit {len(tr)} ({fit_mask(tr).sum()} fitted), "
          f"selection by {scheme}")
    if args.config:
        best = json.loads(Path(args.config).read_text())
        best = best.get("selected", best)
        print("frozen configuration:", best)
    else:
        best = select(tr, Htr, layers, args.specs.split(","),
                      [float(c) for c in args.Cs.split(",")], scheme)
        print("selected:", best)
    ev, Hev, _ = build(args.domain, args.eval, args.eval_part)
    res = {}
    for point in (["rate", "bias"] if args.point == "rate" else ["bias"]):
        res[point] = evaluate(f"{args.domain} {args.eval}/{args.eval_part} [{point}]", tr, Htr,
                              layers, ev, Hev, best, args.domain, point)
    if args.out:
        Path(args.out).write_text(json.dumps(dict(selected=best, result=res), indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
