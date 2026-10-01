"""Sanity inspection and the granularity diagnostic (Spearman)."""

import numpy as np

from .data import load_split
from .prompt import is_idk
from .scoring import gscore
from .text import find_match_level
from .inference import load_trained_model, greedy_answers


def step_inspect(adapter: str = "final", n: int = 20):
    model, tok = load_trained_model(adapter)
    test = load_split("test")[:n]
    answers = greedy_answers(model, tok, test)
    scores = gscore(answers)
    for q, a, s in zip(test, answers, scores):
        idk = is_idk(a)
        ml = find_match_level(a, q.hierarchy) if (a and not idk) else None
        tag = "IDK" if idk else f"match@L{ml}"
        print(f"{q.question[:46]:46s} | '{a}' gs={s:5.1f} | {tag} | {q.hierarchy}")


def step_diagnose(adapter: str = "final"):
    """Does measured granularity track where the answer lands in the hierarchy?"""
    from scipy.stats import spearmanr
    model, tok = load_trained_model(adapter)
    test = load_split("test")
    answers = greedy_answers(model, tok, test)
    scores = gscore(answers)
    gss, mls = [], []
    for q, a, s in zip(test, answers, scores):
        idk = is_idk(a)
        ml = find_match_level(a, q.hierarchy) if (a and not idk) else None
        if ml is not None:
            gss.append(s)
            mls.append(ml)
    print(f"\n=== granularity diagnostic ({adapter}) ===")
    print(f"matched answers: {len(gss)}/{len(test)}")
    print(f"measured gscore: mean={np.mean(scores):.1f} median={np.median(scores):.1f} "
          f"std={np.std(scores):.1f}")
    if len(set(mls)) > 1 and len(gss) >= 3:
        rho, p = spearmanr(gss, mls)
        print(f"Spearman(gscore, match_level) = {rho:.3f} (p={p:.3f})")
        print("positive rho -> coarser answers land at coarser levels (working)")
    else:
        print("not enough level variance to correlate")
