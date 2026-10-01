"""Held-out test of the explicit self-report: does the trained model's answer to a
meta question ("which level can you answer reliably?") match the level the base
model is actually reliable at, on test questions it never saw? Also reports
controllability (asking for a city, region or country).
"""

import json

import numpy as np
import torch
from scipy.stats import spearmanr

from .config import CKPT_DIR, OUT_DIR, CAPABILITY_TAU, SEED
from .data import load_split
from .prompt import (make_meta_prompt, make_direct_prompt, extract_meta_label,
                     is_idk, level_label)
from .text import match_level
from .inference import load_tokenizer, load_base_model, greedy_answers
from .capability import _sample_open, reliability_curve, pick_target

LABELS = ["city", "region", "country", "idk"]
_RANK = {"city": 0, "region": 1, "country": 2}   # idk handled separately


def _base_true_targets(tok, test_qs, k, tau) -> dict:
    """Base-model capability curve on the test set -> true target label per qid."""
    model = load_base_model()
    torch.manual_seed(SEED)
    true = {}
    with torch.no_grad():
        for q in test_qs:
            answers = _sample_open(model, tok, q.question, k)
            mls = [None if is_idk(a) else match_level(a, q.hierarchy) for a in answers]
            L = pick_target(reliability_curve(mls, len(q.hierarchy)), tau)
            true[q.qid] = level_label(q.hierarchy, L)   # "city/region/country/I don't know"
    del model
    torch.cuda.empty_cache()
    return true


def _direct_match_levels(model, tok, test_qs, label) -> list[int | None]:
    ans = greedy_answers(model, tok, test_qs,
                         prompt_fn=lambda q: make_direct_prompt(q, label))
    return [None if is_idk(a) else match_level(a, q.hierarchy)
            for q, a in zip(test_qs, ans)]


def _norm(label: str) -> str:
    return "idk" if label.lower().startswith("i") or label == "idk" else label


def step_generalization(adapter: str = "sft", k: int = 8, tau: float = CAPABILITY_TAU):
    from peft import PeftModel
    print(f"=== held-out generalization test ({adapter}) ===")
    test_qs = load_split("test")
    tok = load_tokenizer()

    # ground truth: base capability on the held-out test set
    print("rolling out base model for true test targets...")
    true_labels = {qid: _norm(v) for qid, v in
                   _base_true_targets(tok, test_qs, k, tau).items()}

    # prediction: trained student meta label  (+ direct answers for controllability)
    base = load_base_model()
    model = PeftModel.from_pretrained(base, str(CKPT_DIR / adapter))
    model.eval()

    meta_raw = greedy_answers(model, tok, test_qs, prompt_fn=make_meta_prompt, raw=True)
    pred_labels = [extract_meta_label(r) or "idk" for r in meta_raw]

    direct = {lab: _direct_match_levels(model, tok, test_qs, lab)
              for lab in ("city", "region", "country")}
    del model, base
    torch.cuda.empty_cache()

    # ---- meta (pred) vs base-true confusion ----
    rows = []
    conf = {a: {b: 0 for b in LABELS} for a in LABELS}
    for q, pred in zip(test_qs, pred_labels):
        true = true_labels[q.qid]
        conf[true][pred] += 1
        rows.append({"qid": q.qid, "true": true, "pred": pred})

    n = len(test_qs)
    exact = sum(1 for r in rows if r["true"] == r["pred"]) / n

    # ordinal agreement on the committable ones (both not idk)
    comm = [(r["true"], r["pred"]) for r in rows if r["true"] != "idk" and r["pred"] != "idk"]
    off_by_one = (sum(1 for t, p in comm if abs(_RANK[t] - _RANK[p]) <= 1) / len(comm)
                  if comm else None)
    rho = (spearmanr([_RANK[t] for t, _ in comm], [_RANK[p] for _, p in comm])[0]
           if len(comm) >= 3 and len(set(p for _, p in comm)) > 1 else None)

    # idk detection (does meta say idk exactly when the base truly cannot commit)
    true_idk = [r["true"] == "idk" for r in rows]
    pred_idk = [r["pred"] == "idk" for r in rows]
    tp = sum(1 for t, p in zip(true_idk, pred_idk) if t and p)
    idk_prec = tp / sum(pred_idk) if any(pred_idk) else None
    idk_rec = tp / sum(true_idk) if any(true_idk) else None

    # ---- controllability (trained): match level should rise city->country ----
    def mean_ml(mls):
        v = [m for m in mls if m is not None]
        return float(np.mean(v)) if v else None
    ctrl = {lab: mean_ml(direct[lab]) for lab in ("city", "region", "country")}

    # ---- report ----
    print(f"\nmeta vs base-true target  (n={n})")
    print(f"{'true|pred':>10s} | " + " | ".join(f"{l:>7s}" for l in LABELS))
    for a in LABELS:
        print(f"{a:>10s} | " + " | ".join(f"{conf[a][b]:7d}" for b in LABELS))
    print(f"\nexact agreement      : {exact:.3f}")
    print(f"off-by-one (committable): {off_by_one if off_by_one is None else round(off_by_one,3)}  (n_comm={len(comm)})")
    print(f"ordinal Spearman     : {rho if rho is None else round(rho,3)}")
    print(f"IDK detection        : precision={idk_prec if idk_prec is None else round(idk_prec,3)}  "
          f"recall={idk_rec if idk_rec is None else round(idk_rec,3)}")
    print(f"\ncontrollability (trained mean match level, want city<region<country):")
    print(f"  city={ctrl['city']}  region={ctrl['region']}  country={ctrl['country']}")

    out = {"adapter": adapter, "tau": tau, "n": n, "exact": exact,
           "off_by_one": off_by_one, "ordinal_spearman": rho,
           "idk_precision": idk_prec, "idk_recall": idk_rec,
           "controllability_mean_ml": ctrl,
           "confusion": conf, "rows": rows}
    with open(OUT_DIR / f"generalization_{adapter}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT_DIR}/generalization_{adapter}.json")
    return out
