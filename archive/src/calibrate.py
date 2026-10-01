"""The first training target, for the GRPO runs: an entropy bin per question.

The base model answers each training question several times; the entropy of its
answers picks a bin, and the bin picks a level of the gold hierarchy. Computed once
on the base model and cached. Replaced by the capability curve (capability.py),
because one entropy number cannot tell a confident right answer from a confident
wrong one.
"""

import json
from collections import Counter

import numpy as np
import torch

from .config import (MODEL_NAME, CALIB_PATH, TARGETS_PATH, NUM_ROLLOUTS,
                     TEMPERATURE, MAX_COMPLETION_LEN, MIN_HITS)
from .data import load_split, max_hierarchy_depth
from .prompt import make_prompt, extract_answer, is_idk, PROMPT_TEMPLATE
from .reward import semantic_entropy, entropy_to_bin
from .scoring import gscore
from .text import find_match_level
from .inference import load_tokenizer, load_base_model


def step_calibrate(template: str = PROMPT_TEMPLATE):
    print("=== calibrating on full train set (entropy targets) ===")
    train = load_split("train")
    n_levels = max_hierarchy_depth(train)
    print(f"train questions: {len(train)}   hierarchy depth: {n_levels}")

    tok = load_tokenizer()
    model = load_base_model()

    recs = []
    n_idk_total = 0
    n_completions = 0
    with torch.no_grad():
        for i, q in enumerate(train):
            chat = tok.apply_chat_template(
                [{"role": "user", "content": make_prompt(q.question, template)}],
                tokenize=False, add_generation_prompt=True)
            inputs = tok(chat, return_tensors="pt").to(model.device)
            out = model.generate(
                **inputs, max_new_tokens=MAX_COMPLETION_LEN,
                do_sample=True, temperature=TEMPERATURE, top_p=0.95,
                num_return_sequences=NUM_ROLLOUTS, pad_token_id=tok.pad_token_id)
            gen = out[:, inputs["input_ids"].shape[1]:]
            answers = [extract_answer(s)
                       for s in tok.batch_decode(gen, skip_special_tokens=True)]
            H = semantic_entropy(answers)
            n_hits = sum(1 for a in answers
                         if a and not is_idk(a) and find_match_level(a, q.hierarchy) is not None)
            n_idk_total += sum(1 for a in answers if is_idk(a))
            n_completions += len(answers)
            recs.append({"qid": q.qid, "entropy": H, "n_hits": n_hits})
            if i % 25 == 0:
                print(f"  calib {i+1}/{len(train)}  H={H:.3f}  hits={n_hits}/{NUM_ROLLOUTS}")

    entropies = [r["entropy"] for r in recs]
    pcts = [100 * k / n_levels for k in range(1, n_levels)]
    bins = np.percentile(entropies, pcts).tolist()

    qid_to_q = {q.qid: q for q in train}
    for r in recs:
        q = qid_to_q[r["qid"]]
        b = entropy_to_bin(r["entropy"], bins, n_levels)
        r["target_bin"] = b
        r["target_level"] = min(b, len(q.hierarchy) - 1)

    # GranuScore of each gold answer at its target level (one batched call).
    flat = [qid_to_q[r["qid"]].hierarchy[r["target_level"]] for r in recs]
    flat_gs = gscore(flat)
    for r, gs in zip(recs, flat_gs):
        r["target_gscore"] = gs

    kept = sum(1 for r in recs if r["n_hits"] >= MIN_HITS)
    idk_rate = n_idk_total / n_completions if n_completions else 0.0

    calib = {
        "n_levels": n_levels, "bins": bins, "percentiles": pcts,
        "n_train_total": len(recs), "n_train_kept": kept, "min_hits": MIN_HITS,
        "entropy_min": float(min(entropies)), "entropy_max": float(max(entropies)),
        "entropy_mean": float(np.mean(entropies)),
        "base_idk_rollout_rate": idk_rate,
    }
    with open(CALIB_PATH, "w") as f:
        json.dump(calib, f, indent=2)
    targets = {str(r["qid"]): {"entropy": r["entropy"], "n_hits": r["n_hits"],
                               "target_bin": r["target_bin"],
                               "target_level": r["target_level"],
                               "target_gscore": r["target_gscore"]}
               for r in recs}
    with open(TARGETS_PATH, "w") as f:
        json.dump(targets, f, indent=2)

    print(f"\nn_levels={n_levels}  bins={[round(b, 3) for b in bins]}")
    print(f"target_level dist: {dict(sorted(Counter(r['target_level'] for r in recs).items()))}")
    print(f"target_bin   dist: {dict(sorted(Counter(r['target_bin'] for r in recs).items()))}")
    print(f"hit dist: {dict(sorted(Counter(r['n_hits'] for r in recs).items()))}")
    print(f"kept (n_hits>={MIN_HITS}): {kept}/{len(recs)} ({100*kept/len(recs):.0f}%)")
    print(f"\n*** base IDK-rollout-rate: {idk_rate:.3f} "
          f"({n_idk_total}/{n_completions}) ***")
    if idk_rate < 0.02:
        print("    ~0 IDK in base rollouts — GRPO has little/no abstention gradient.")
        print("    Switch to FEWSHOT_PROMPT_TEMPLATE (prompt.py) and recalibrate before training.")
    else:
        print("    base emits IDK — GRPO has an abstention gradient, no seeding needed.")
    print(f"saved {CALIB_PATH} and {TARGETS_PATH}")
