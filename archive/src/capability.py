"""The capability curve: the training target used by the SFT and DPO experiments.

For each question the base model answers K times. reliability(L) is the share of the
K samples that are correct at level L or finer, and the target is the finest level
with reliability >= tau, or "I don't know" if there is none. Computed once on the
base model and cached, never from the model being trained, which could otherwise
change its own signal.

The final pipeline's 50%-support rule (pipeline/targets.py) grew out of this.
"""

import json
from collections import Counter

import numpy as np
import torch

from .config import (CAPABILITY_PATH, CAPABILITY_K, CAPABILITY_TAU,
                     TEMPERATURE, MAX_COMPLETION_LEN, MIN_HITS, SEED)
from .data import load_split, max_hierarchy_depth
from .prompt import make_prompt, extract_answer, is_idk
from .text import match_level
from .inference import load_tokenizer, load_base_model


def _sample_open(model, tok, question: str, k: int) -> list[str]:
    """k independent base-model samples for one question (open prompt)."""
    chat = tok.apply_chat_template(
        [{"role": "user", "content": make_prompt(question)}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=MAX_COMPLETION_LEN,
        do_sample=True, temperature=TEMPERATURE, top_p=0.95,
        num_return_sequences=k, pad_token_id=tok.pad_token_id)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return [extract_answer(s) for s in tok.batch_decode(gen, skip_special_tokens=True)]


def _greedy_open(model, tok, question: str) -> str:
    """One greedy answer with the deployment prompt (used by the v2 target rule)."""
    chat = tok.apply_chat_template(
        [{"role": "user", "content": make_prompt(question)}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=MAX_COMPLETION_LEN,
        do_sample=False, pad_token_id=tok.pad_token_id)
    return extract_answer(tok.decode(out[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True))


def reliability_curve(match_levels: list[int | None], depth: int) -> list[float]:
    """reliability[L] for each level, from the match level of all k samples (None for a
    wrong answer or an abstention). Divides by all k samples, not only the answered ones,
    so a question answered correctly 3 times out of 8 stays below tau = 0.5."""
    k = len(match_levels)
    if k == 0:
        return [0.0] * depth
    return [sum(1 for m in match_levels if m is not None and m <= L) / k
            for L in range(depth)]


def pick_target(curve: list[float], tau: float) -> int | None:
    """Finest level whose reliability >= tau, else None (=> IDK target)."""
    for L, r in enumerate(curve):
        if r >= tau:
            return L
    return None


def pick_target_v2(sample_answers: list[str], sample_match_levels: list[int | None],
                   greedy_is_idk: bool, greedy_match_level: int | None, depth: int,
                   tau: float, min_commit: int = 3, max_wrong: int = 2) -> int | None:
    """Second version of the target rule: wrong samples count against a level, the
    model's own abstentions do not. Too many wrong samples give "I don't know"; with
    few answered samples the greedy answer decides."""
    from .prompt import is_idk as _is_idk
    committed_mls = [m for a, m in zip(sample_answers, sample_match_levels)
                     if not _is_idk(a)]
    wrong = sum(1 for m in committed_mls if m is None)
    if wrong > max_wrong:
        return None
    if len(committed_mls) >= min_commit:
        target = None
        for L in range(depth):
            rel = sum(1 for m in committed_mls if m is not None and m <= L) / len(committed_mls)
            if rel >= tau:
                target = L
                break
    else:
        target = None
        if (not greedy_is_idk) and greedy_match_level is not None and wrong == 0:
            target = greedy_match_level
    if (target is not None and (not greedy_is_idk) and greedy_match_level is not None
            and greedy_match_level < target and wrong == 0):
        target = greedy_match_level
    return target


def retarget(cap: dict, tau: float = CAPABILITY_TAU, min_commit: int = 3,
             max_wrong: int = 2) -> dict:
    """Recompute every target in a cached curve with the v2 rule, offline."""
    changed = promoted = 0
    for r in cap["questions"].values():
        old = r["target_level"]
        new = pick_target_v2(r["sample_answers"], r["sample_match_levels"],
                             r["greedy_is_idk"], r["greedy_match_level"],
                             len(r["hierarchy"]), tau, min_commit, max_wrong)
        if new != old:
            changed += 1
            if old is None and new is not None:
                promoted += 1
        r["target_level"] = new
        r["target_is_idk"] = new is None
        r["target_norm_level"] = (r["levels"][new] if new is not None
                                  and new < len(r["levels"]) else None)
    n = len(cap["questions"])
    n_idk = sum(1 for r in cap["questions"].values() if r["target_is_idk"])
    print(f"retarget v2 (tau={tau} min_commit={min_commit} max_wrong={max_wrong}): "
          f"changed {changed}/{n}, IDK->level promotions {promoted}, "
          f"IDK share {n_idk}/{n} ({100*n_idk/n:.0f}%)")
    cap["target_rule"] = {"rule": "v2", "tau": tau, "min_commit": min_commit,
                          "max_wrong": max_wrong}
    return cap


def _target_dist(recs: list[dict], tau: float) -> dict:
    """Target-level distribution at a given tau (IDK shown as key 'IDK')."""
    c = Counter()
    for r in recs:
        L = pick_target(r["curve"], tau)
        c["IDK" if L is None else L] += 1
    return dict(sorted(c.items(), key=lambda kv: (kv[0] == "IDK", kv[0])))


def step_capability(limit: int | None = None, tau: float = CAPABILITY_TAU,
                    out_path=CAPABILITY_PATH, relations: list[str] | None = None):
    print("=== capability curve on train set (reliability-vs-level target) ===")
    train = load_split("train", relations)
    if limit:
        train = train[:limit]
        print(f"*** SMOKE: first {len(train)} questions ***")
    depth = max_hierarchy_depth(load_split("train", relations))
    print(f"questions: {len(train)}   max hierarchy depth: {depth}   k={CAPABILITY_K}   tau={tau}")

    tok = load_tokenizer()
    model = load_base_model()
    torch.manual_seed(SEED)   # reproducible cached target (rollouts are sampled)

    recs = []
    n_idk_samples = n_wrong_samples = n_samples = 0
    with torch.no_grad():
        for i, q in enumerate(train):
            answers = _sample_open(model, tok, q.question, CAPABILITY_K)
            greedy_a = _greedy_open(model, tok, q.question)
            qdepth = len(q.hierarchy)
            mls = []
            for a in answers:
                if is_idk(a):
                    mls.append(None)          # abstention, not a wrong place
                    n_idk_samples += 1
                else:
                    m = match_level(a, q.hierarchy)
                    mls.append(m)
                    if m is None:
                        n_wrong_samples += 1   # confident, off-hierarchy
            n_samples += len(answers)

            # Wrong answers and abstentions both count as None in the curve.
            curve = reliability_curve(mls, qdepth)
            target_L = pick_target(curve, tau)
            n_hits = sum(1 for m in mls if m is not None)

            recs.append({
                "qid": q.qid,
                "question": q.question,
                "hierarchy": q.hierarchy,
                "levels": q.levels,
                "curve": curve,                       # reliability at each level
                "target_level": target_L,             # int index or None (v1 rule)
                "target_is_idk": target_L is None,
                "target_norm_level": (q.levels[target_L] if target_L is not None
                                      and target_L < len(q.levels) else None),
                "n_hits": n_hits,
                "confident_wrong": sum(1 for a, m in zip(answers, mls)
                                       if not is_idk(a) and m is None),
                "idk_samples": sum(1 for a in answers if is_idk(a)),
                # Raw samples, so targets can be recomputed offline and SFT can
                # train on the model's own answers.
                "sample_answers": answers,
                "sample_match_levels": mls,
                "greedy_answer": greedy_a,
                "greedy_is_idk": is_idk(greedy_a),
                "greedy_match_level": (match_level(greedy_a, q.hierarchy)
                                       if (greedy_a and not is_idk(greedy_a)) else None),
            })
            if i % 25 == 0:
                print(f"  cap {i+1}/{len(train)}  curve={[round(c,2) for c in curve]} "
                      f"-> target={'IDK' if target_L is None else target_L}")

    # ---- summary ----
    idk_rate = n_idk_samples / n_samples if n_samples else 0.0
    wrong_rate = n_wrong_samples / n_samples if n_samples else 0.0
    kept = sum(1 for r in recs if r["n_hits"] >= MIN_HITS)
    n_idk_targets = sum(1 for r in recs if r["target_is_idk"])

    print(f"\nbase sample mix: confident-wrong={wrong_rate:.3f}  idk={idk_rate:.3f}  "
          f"({n_samples} samples)")
    print(f"target dist (tau={tau}): {_target_dist(recs, tau)}")
    print(f"IDK targets: {n_idk_targets}/{len(recs)} "
          f"({100*n_idk_targets/len(recs):.0f}%)  <- model can't reliably name even the country")
    print(f"kept (n_hits>={MIN_HITS}): {kept}/{len(recs)} ({100*kept/len(recs):.0f}%)")
    print("\ntau sweep (target distribution):")
    for t in (0.25, 0.375, 0.5, 0.625, 0.75):
        print(f"  tau={t}: {_target_dist(recs, t)}")

    out = {
        "tau": tau, "k": CAPABILITY_K, "depth": depth,
        "n_total": len(recs), "n_kept": kept, "min_hits": MIN_HITS,
        "base_confident_wrong_rate": wrong_rate, "base_idk_rate": idk_rate,
        "n_idk_targets": n_idk_targets,
        "questions": {str(r["qid"]): r for r in recs},
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}")
