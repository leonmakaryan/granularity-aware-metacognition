"""Prompt baselines on the 303 location test questions, scored like the trained models.

    plain  the bare question, first line as the answer
    noidk  the deployment prompt without the "I don't know" option
    idk    the deployment prompt
    drag   DRAG from the GRANOLA paper: 10 samples, then an aggregation prompt

    TAG=7b MODEL_NAME=Qwen/Qwen2.5-7B-Instruct python 3_sft/run_baselines.py
"""
import json
import os
from pathlib import Path

import pandas as pd
import torch

from src.config import LOCATION_RELATIONS, OUT_DIR, SEED, MAX_COMPLETION_LEN
from src.data import load_split
from src.inference import load_tokenizer, load_base_model, greedy_answers
from src.prompt import (PLAIN_PROMPT_TEMPLATE, NOIDK_PROMPT_TEMPLATE, is_idk,
                        make_prompt)
from src.text import match_level
from src.eval import (info_weighted_acc, metacognition, build_level_pool,
                      signed_informativeness, output_level, level_mean_std,
                      strict_match_level)

# DRAG constants, phase1.py verbatim.
DRAG_NUM_SAMPLES = 10
DRAG_TEMPERATURE = 0.7
DRAG_AGGREGATION_PROMPT = """You will be given a list of responses; replace them with the most specific answer that is still consistent with all the original responses. If the responses have nothing meaningful in common with respect to the question, output IDK.

Here are some examples:

Question: Where was [X] born?
Responses:
- Hamburg
- Hamburg
- Bonn
- Berlin
Correct aggregated answer: Germany
Incorrect aggregated answer: Hamburg
Explanation: These are all different cities in Germany. Hamburg is not a correct aggregation, since it is not consistent with other responses, such as Berlin or Bonn.

Question: When was [X] born?
Responses:
- February 1, 1937
- November 20, 1937
- January 1937
Correct aggregated answer: 1937
Incorrect aggregated answer: November 1937
Explanation: These are all dates in 1937.

Question: {q}
Responses:
{candidates}
Correct aggregated answer:"""


def first_line(s: str) -> str:
    return s.split("\n")[0].strip()


# The bare question gets prose refusals ("I'm sorry, but I don't have any specific
# information...") that is_idk misses. A refusal is an abstention, not a wrong answer.
import re
_REFUSAL = re.compile(
    r"i.?m sorry|i apologize|i (don.?t|do not) have|cannot provide|couldn.?t find|"
    r"no (specific )?information|unable to (find|provide|determine)",
    re.IGNORECASE)


def plain_is_idk(answer: str) -> bool:
    return is_idk(answer) or bool(_REFUSAL.search(answer))


def sample_n(model, tok, user_msg: str, n: int) -> list[str]:
    chat = tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_COMPLETION_LEN,
                             do_sample=True, temperature=DRAG_TEMPERATURE, top_p=0.95,
                             num_return_sequences=n, pad_token_id=tok.pad_token_id)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return [d.strip() for d in tok.batch_decode(gen, skip_special_tokens=True)]


def greedy_one(model, tok, user_msg: str) -> str:
    chat = tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                   tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_COMPLETION_LEN,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def to_recs(qs, answers, idk_fn=is_idk):
    recs = []
    for q, a in zip(qs, answers):
        idk = idk_fn(a)
        ml = match_level(a, q.hierarchy) if (a and not idk) else None
        recs.append((q, a, idk, ml))
    return recs


def save_csv(recs, name):
    if LIMIT:
        return
    pd.DataFrame([{"qid": q.qid, "question": q.question, "hierarchy": q.hierarchy,
                   "answer": a, "is_idk": idk, "match_level": ml}
                  for q, a, idk, ml in recs]).to_csv(OUT_DIR / _tagged(name), index=False)


def metrics(recs, pool):
    n = len(recs)
    nls = [q.levels[ml] if (ml is not None and ml < len(q.levels)) else None
           for q, _, _, ml in recs]
    signed = [signed_informativeness(q, ml, idk, a, pool) for q, a, idk, ml in recs]
    lm, ls = level_mean_std([output_level(q, ml, idk, a, pool) for q, a, idk, ml in recs])
    # B4 strict policy (primary claim only) next to the lenient columns.
    smls = [strict_match_level(a, q.hierarchy) if (a and not idk) else None
            for q, a, idk, _ in recs]
    snls = [q.levels[m] if (m is not None and m < len(q.levels)) else None
            for (q, _, _, _), m in zip(recs, smls)]
    ssigned = [signed_informativeness(q, m, idk, a, pool)
               for (q, a, idk, _), m in zip(recs, smls)]
    committed = sum(1 for _, _, idk, _ in recs if not idk)
    on_hier = sum(1 for _, _, _, ml in recs if ml is not None)
    mc = metacognition([(not idk, ml is not None) for _, _, idk, ml in recs])
    return dict(n=n, acc=on_hier / n, info=info_weighted_acc(nls),
                info_signed=sum(signed) / n,
                info_strict=info_weighted_acc(snls),
                info_signed_strict=sum(ssigned) / n,
                level_mean=lm, level_std=ls,
                idk=sum(1 for _, _, idk, _ in recs if idk) / n,
                wrong=sum(1 for _, _, idk, ml in recs if (not idk) and ml is None) / n,
                sel=on_hier / committed if committed else 0.0,
                d=mc["d_prime"], yfr=mc["yfr"], nfr=mc["nfr"])


def show(tag, m):
    d = f"{m['d']:.3f}" if m["d"] is not None else "  —  "
    print(f"{tag:>6s}: acc={m['acc']:.3f} info={m['info']:.3f} "
          f"info_signed={m['info_signed']:+.3f} signed_strict={m['info_signed_strict']:+.3f} "
          f"lvl={m['level_mean']:.2f}±{m['level_std']:.2f} "
          f"idk={m['idk']:.3f} wrong={m['wrong']:.3f} d'={d}", flush=True)


LIMIT = int(os.environ.get("LIMIT", "0") or 0)   # smoke mode: first N questions
TAG = os.environ.get("TAG", "")                  # output-file suffix, e.g. "7b"


def _tagged(name: str) -> str:
    if not TAG:
        return name
    stem, dot, ext = name.rpartition(".")
    return f"{stem}_{TAG}{dot}{ext}" if dot else f"{name}_{TAG}"


def main():
    torch.manual_seed(SEED)   # DRAG samples reproducible
    test = load_split("test", LOCATION_RELATIONS)
    if LIMIT:
        test = test[:LIMIT]
        print(f"*** SMOKE: {LIMIT} questions, files NOT saved ***", flush=True)
    tok = load_tokenizer()
    model = load_base_model()
    pool = build_level_pool()
    results = {}

    # ---- plain (bare prompt, first-line extraction) ----
    print("=== plain ===", flush=True)
    raw = greedy_answers(model, tok, test,
                         prompt_fn=lambda q: PLAIN_PROMPT_TEMPLATE.format(q=q), raw=True)
    recs = to_recs(test, [first_line(a) for a in raw], idk_fn=plain_is_idk)
    save_csv(recs, "eval_base_plain.csv")
    results["plain"] = metrics(recs, pool)
    show("plain", results["plain"])

    # ---- noidk (marker + back-off, no IDK escape) ----
    print("=== noidk ===", flush=True)
    ans = greedy_answers(model, tok, test,
                         prompt_fn=lambda q: NOIDK_PROMPT_TEMPLATE.format(q=q))
    recs = to_recs(test, ans)
    save_csv(recs, "eval_base_noidk.csv")
    results["noidk"] = metrics(recs, pool)
    show("noidk", results["noidk"])

    # ---- idk (full prompt) — reuse the saved run if present ----
    print("=== idk ===", flush=True)
    # The cached run is the 1.5B one; never reuse it for another model (TAG set).
    idk_csv = OUT_DIR / "base_combined_eval.csv"
    if idk_csv.exists() and not LIMIT and not TAG:
        df = pd.read_csv(idk_csv)
        byqid = {q.qid: q for q in test}
        recs = [(byqid[r["qid"]],
                 "" if pd.isna(r["answer"]) else str(r["answer"]),
                 bool(r["is_idk"]),
                 None if pd.isna(r["match_level"]) else int(r["match_level"]))
                for _, r in df.iterrows()]
    else:
        recs = to_recs(test, greedy_answers(model, tok, test))
        save_csv(recs, "base_combined_eval.csv")   # save_csv applies the TAG suffix
    results["idk"] = metrics(recs, pool)
    show("idk", results["idk"])

    # ---- drag ----
    print("=== drag (10 samples + aggregation per question, slow) ===", flush=True)
    recs = []
    for i, q in enumerate(test):
        samples = sample_n(model, tok, PLAIN_PROMPT_TEMPLATE.format(q=q.question),
                           DRAG_NUM_SAMPLES)
        samples = [first_line(s) for s in samples]
        agg = greedy_one(model, tok, DRAG_AGGREGATION_PROMPT.format(
            q=q.question, candidates="\n".join(f"- {s}" for s in samples)))
        agg = first_line(agg)
        idk = agg.strip().upper().startswith("IDK") or is_idk(agg)
        ml = match_level(agg, q.hierarchy) if (agg and not idk) else None
        recs.append((q, agg, idk, ml))
        if i % 25 == 0:
            print(f"  drag {i + 1}/{len(test)}", flush=True)
    save_csv(recs, "eval_drag.csv")
    results["drag"] = metrics(recs, pool)
    show("drag", results["drag"])

    if not LIMIT:
        with open(OUT_DIR / _tagged("baselines_combined.json"), "w") as f:
            json.dump(results, f, indent=2)
    print(f"\nsummary ({OUT_DIR}/{_tagged('baselines_combined.json')}):")
    for tag in ("plain", "noidk", "idk", "drag"):
        show(tag, results[tag])


if __name__ == "__main__":
    main()
